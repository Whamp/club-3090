# DeepSeek V4 low-bit artifact tools

CPU-side planning and conversion tools for the MTP-free, mixed-WNA16 DeepSeek V4 Flash artifact described in [the research plan](../../.research/deepseek-v4-lowbit/PLAN.md).

## Exact artifact planner

The planner reads captured safetensors headers without loading model weights. It:

- preserves non-MTP, non-routed-expert tensors byte-for-byte;
- omits every `mtp.*` tensor;
- replaces routed-expert `w1`, `w3`, and `w2` weights and source scales
  with symmetric group-size-128 Humming WNA16 weights, FP16 scales, and INT64
  logical-shape tensors;
- supports one `w13` bit width and one `w2` bit width per layer; and
- rejects dimensions that the pinned vLLM Humming loader cannot pack.

Create a recipe such as:

```json
{
  "default": {"w13_bits": 2, "w2_bits": 2},
  "layers": {
    "26": {"w13_bits": 4, "w2_bits": 4},
    "42": {"w13_bits": 2, "w2_bits": 4}
  }
}
```

Run:

```bash
uv run deepseek-v4-plan \
  /path/to/safetensors-headers.json \
  /path/to/recipe.json
```

The JSON result separates preserved bytes, packed weights, FP16 scales, INT64
logical-shape tensors, replaced source bytes, and omitted MTP bytes.
`total_bytes` is the expected raw tensor payload; filesystem and
safetensors-header overhead are not included.

## Routed-expert imatrix

`ImatrixFile` indexes Antirez's legacy llama.cpp `.dat` file with `mmap`, so opening the approximately 450 MB artifact does not copy all importance values into Python objects. `expert_vector()` maps an official checkpoint tensor such as `layers.26.ffn.experts.17.w2.weight` to `blk.26.ffn_down_exps.weight`, slices expert 17 from the packed 256-expert entry, and applies the file's call-count normalization.

The parser rejects corrupt lengths, duplicate names, incompatible expert geometry, non-finite selected values, and trailing data. Keep the file open while requesting vectors:

```python
from pathlib import Path

from deepseek_v4_lowbit.imatrix import ImatrixFile

with ImatrixFile.open(Path("routed-moe-imatrix.dat")) as imatrix:
    importance = imatrix.expert_vector(
        "layers.26.ffn.experts.17.w2.weight",
        expert_count=256,
        input_columns=2048,
    )
```

The parser contract follows `antirez/ds4@84cc882352757baf628a1776badf7cc54d584e28`. The published imatrix itself still needs a direct checksum and full-geometry check when staged for the pilot.

## Quantizer comparison

`quantize_symmetric()` wraps the pinned AutoRound RTN primitives rather than copying their scale-search implementation:

- plain RTN uses `quant_tensor_rtn_sym`;
- imatrix-weighted scale search uses `quant_tensor_opt_rtn_sym`;
- both return signed codes, stored FP16 group scales, the reconstruction produced from those persisted values, and comparable weighted and unweighted MSE.

Quantization is an optional heavy path. Run it in the AutoRound checkout pinned by the plan at `f17d9cd4b36982006bad21ff87127aac739072e3`; metadata planning and imatrix indexing do not import Torch or AutoRound. The CPU regression fixture shows weighted W2 search reducing its weighted MSE from about `0.05703` to `0.05031`. This verifies the comparison mechanism, not expected DeepSeek quality; representative real layers remain the rental-pilot decision point.

```bash
PYTHONPATH=src /path/to/auto-round/.venv/bin/python \
  -m unittest tests/test_quantizer.py -v
```

## Checkpoint packing

`pack_quantized_tensor()` delegates to the `pack_to_int32` implementation in `compressed-tensors==0.17.0`, the version pinned by the selected vLLM fork. It preserves the package's signed-code offset and low-bit-first word order; the regression fixture packs repeated W2 codes `[-2, -1, 0, 1]` as the literal word `0xE4E4E4E4`.

`packed_checkpoint_tensors()` expands one official per-expert weight name into the three keys consumed by vLLM's `RoutedExperts` loader:

```text
layers.N.ffn.experts.E.w1.weight_packed
layers.N.ffn.experts.E.w1.weight_scale
layers.N.ffn.experts.E.w1.weight_shape
```

The same mapping applies to `w3` and `w2`. The checkpoint stays per-expert; vLLM fuses `w1` and `w3` into `w13` while loading.

## Resumable shard output

`ResumableSafetensorsWriter` writes each output shard through a temporary file and records a receipt under `.conversion-state/receipts/`. A receipt binds:

- the source-shard SHA-256;
- a canonical recipe SHA-256;
- the output-shard SHA-256 and byte length; and
- each tensor's name, dtype, shape, and raw byte count.

A completed shard is skipped only after its identity and checksum verify.
Recipe changes, source changes, orphaned final files, malformed receipts,
checksum failures, missing expected shards, mixed recipe fingerprints,
duplicate tensor names, and any missing, unexpected, or mis-sharded final
tensor fail closed. The writer can recover the narrow crash window where the
final shard rename completed but the prepared receipt rename did not.
`finalize_index()` creates `model.safetensors.index.json` only when verified
receipts exactly match the output inventory derived from the source index.

Call `completed_shard()` before loading a source shard so a resume avoids dequantization and quantization work. Keep the output and `.conversion-state` directories together on durable storage until artifact validation finishes.

## Bounded quantizer pilot

`deepseek-v4-pilot` compares plain and imatrix-weighted RTN only on named `LAYER:EXPERT` samples. It validates the published imatrix's complete 43-layer/129-entry geometry, loads only the source shards containing those samples, runs both candidates at the requested bit widths, verifies that each result packs, and records elapsed time plus weighted and unweighted error.

The initial rental pilot is intentionally limited to layers 0, 26, 37, and 42; experts 0 and 127; all three projections; and W2. That is 24 matrices and 48 candidate fits from source shards 2, 28, 39, and 44:

```bash
deepseek-v4-pilot \
  /durable/source/DeepSeek-V4-Flash-0731 \
  /durable/routed-moe-imatrix.dat \
  /durable/pilot/w2-quantizer-comparison.json \
  --sample 0:0 --sample 0:127 \
  --sample 26:0 --sample 26:127 \
  --sample 37:0 --sample 37:127 \
  --sample 42:0 --sample 42:127 \
  --bits 2 --device cuda
```

Use weighted RTN for the full artifact only if its measured weighted-error improvement is meaningful relative to its runtime. This pilot is a method screen, not an end-to-end quality claim.

`deepseek-v4-summarize-pilot PILOT_REPORT SUMMARY_REPORT` pairs candidates by tensor and bit width. It reports improvement, tie, and worsening counts; median weighted-error change; per-projection timing; and a projection-aware extrapolation across all 43 layers × 256 experts. The extrapolation covers quantize-and-pack time only, explicitly excluding download, source dequantization, writing, finalization, and upload. Its `decision` remains null: the evidence informs the explicit full-run quantizer argument rather than silently selecting one. The summary records the pilot report's SHA-256.

Before a resumed full conversion, `deepseek-v4-validate-pilot` checks the report schema, source-index checksum, imatrix checksum, exact sample and candidate sets, source-shard assignments and checksums, bit width, group size, device, finite metrics, report-to-summary SHA-256, and a freshly recomputed summary. The rental full-run script refuses stale, incomplete, changed, or mismatched pilot evidence.

`rental/run-verda-quantizer-pilot.sh` is the idempotent, secret-free staging entry point for the selected Verda A100 pilot. It pins the conversion and AutoRound commits, official checkpoint and imatrix revisions, CUDA 13.0 Torch environment, representative downloads, imatrix checksum, and command above. It writes a durable JSON report and timestamped log under the supplied rental root.

`rental/run-verda-vllm-w2-oracle.sh` is a separate, attributable A100
stage. It reconstructs the exact haosdent vLLM patch tree from the vendored
series and creates an isolated environment through vLLM's documented
precompiled-extension development path. It requires
`humming-kernels==0.1.10`, runs the W2/group-128/BF16 indexed-MoE numerical
oracle under NVRTC, and records SHA-256 plus `cuobjdump` output for every
`sm_80` cubin. Passing on A100 establishes generic integration correctness
only. It cannot establish SM86 compilation, dispatch, or performance.

## Streamed conversion

`deepseek-v4-convert` processes the official indexed checkpoint one source
shard at a time. For each routed expert it delegates DeepSeek MXFP4/E8M0
normalization and dequantization to pinned AutoRound, optionally loads the
matching expert imatrix vector, fits WNA16, emits compressed-tensors keys,
releases transient tensors, and hands the completed shard to the resumable
writer. Preserved tensors retain their values and dtypes; source routed scales
are replaced; every `mtp.*` tensor is omitted. The official checkpoint has 48
source shards, but shards 46–48 contain only MTP tensors, so the MTP-free
artifact writes and finalizes exactly 45 output shards.

The official 72,317-tensor header capture verifies the streaming boundary: all 35,328 routed weights, including the omitted MTP weights, have their source scale in the same one of 48 shards. There are no missing or cross-shard pairs.

Run a resumable plain-RTN conversion:

```bash
deepseek-v4-convert \
  /durable/source/DeepSeek-V4-Flash-0731 \
  /durable/output/deepseek-v4-flash-wna16 \
  /durable/recipe.json \
  --device cuda \
  --quantizer plain-rtn
```

For an imatrix-weighted run, use `--quantizer imatrix-weighted-rtn --imatrix /durable/routed-moe-imatrix.dat`. Repeat `--shard NAME` to process only an intentional pilot subset. Subset runs write verified shards and receipts but deliberately do not create a model index or config. A full run accounts for all 48 source shards, writes 45 output shards, and additionally:

- creates `model.safetensors.index.json` from verified receipts;
- replaces the source FP8 metadata with exact per-layer/per-projection compressed-tensors groups;
- sets `num_nextn_predict_layers` to zero and records MTP as omitted;
- copies tokenizer, model-code, and other non-weight assets atomically; and
- writes `conversion-metrics.json` from metrics retained in shard receipts.

The transform recipe checksum includes layer bits, group size, quantizer, imatrix checksum, compute device, and pinned AutoRound/compressed-tensors revisions. Resume fails closed if any of these change.

After a Hugging Face upload, `deepseek-v4-verify-upload LOCAL_DIRECTORY REPOSITORY REPORT` compares the exact local and remote inventories. It verifies LFS/Xet objects with content SHA-256, ordinary files with their Git blob SHA-1, and byte sizes for every file; only Hub-managed `.gitattributes` may be extra. A successful upload command alone is not sufficient evidence for deleting the rental.

`rental/run-verda-full-conversion.sh QUANTIZER [RENTAL_ROOT] [HF_REPOSITORY]` is the corresponding resumable full-run entry point. `QUANTIZER` must be the explicit pilot decision (`plain-rtn` or `imatrix-weighted-rtn`); the script does not choose by threshold. It requires the completed 48-candidate pilot report, CUDA, at least 260 GiB free, and `HF_TOKEN` in the environment before downloading the remaining checkpoint. It creates one all-W2, MTP-free artifact, uploads it privately, and runs `deepseek-v4-verify-upload`. Copy rental scripts outside the source checkout before execution because they deliberately replace that checkout with pinned revisions.

## W3 constraint

The pinned vLLM implementation allocates each packed dimension using a pack factor of `32 // bits`. W3 therefore uses ten values per 32-bit word. DeepSeek V4's 4096- and 2048-wide expert matrices are not divisible by ten, so the current planner rejects W3. Supporting it requires a tested padding contract in the writer and loader; silently truncating dimensions would corrupt the artifact.
