# DeepSeek V4 low-bit artifact tools

CPU-side planning and conversion tools for the MTP-free, mixed-WNA16 DeepSeek V4 Flash artifact described in [the research plan](../../.research/deepseek-v4-lowbit/PLAN.md).

## Exact artifact planner

The planner reads captured safetensors headers without loading model weights. It:

- preserves non-MTP, non-routed-expert tensors byte-for-byte;
- omits every `mtp.*` tensor;
- replaces routed-expert `w1`, `w3`, and `w2` weights and source scales
  with symmetric Humming WNA16 weights, FP16 scales, and INT64 logical-shape
  tensors;
- supports separate `w13` and `w2` bit widths and group sizes per layer, with
  group sizes 128, 256, and 512; and
- rejects dimensions that the pinned vLLM Humming loader cannot pack.

Create a recipe such as:

```json
{
  "default": {
    "w13_bits": 2,
    "w2_bits": 2,
    "w13_group_size": 512,
    "w2_group_size": 256
  },
  "layers": {
    "26": {
      "w13_bits": 2,
      "w2_bits": 4,
      "w13_group_size": 128,
      "w2_group_size": 128
    }
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

The parser contract follows
`antirez/ds4@84cc882352757baf628a1776badf7cc54d584e28`. Frontier runs pin
`antirez/deepseek-v4-gguf@e7f04037032990db0346398d249baf9fb9df1ccc`
and require imatrix SHA-256
`02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e`
plus the complete 43-layer geometry.

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

The transform recipe checksum includes per-layer projection bits and group
sizes, quantizer, imatrix checksum, compute device, and pinned
AutoRound/compressed-tensors revisions. Resume fails closed if any of these
change.

After a Hugging Face upload, `deepseek-v4-verify-upload LOCAL_DIRECTORY REPOSITORY REPORT` compares the exact local and remote inventories. It verifies LFS/Xet objects with content SHA-256, ordinary files with their Git blob SHA-1, and byte sizes for every file; only Hub-managed `.gitattributes` may be extra. A successful upload command alone is not sufficient evidence for deleting the rental.

`rental/run-verda-full-conversion.sh QUANTIZER [RENTAL_ROOT] [HF_REPOSITORY]` is the corresponding resumable full-run entry point. `QUANTIZER` must be the explicit pilot decision (`plain-rtn` or `imatrix-weighted-rtn`); the script does not choose by threshold. It requires the completed 48-candidate pilot report, A100 compute capability 8.0, at least 260 GiB free on the `RENTAL_ROOT` filesystem, and `HF_TOKEN` in the environment before downloading the remaining checkpoint. The token must grant `repo.write` to the target namespace, and the existing target repository must be private. An ambient read-only token can shadow a stored upload credential, so verify the exact transferred environment token. The script creates one all-W2, MTP-free artifact, uploads it privately, and runs `deepseek-v4-verify-upload`. Copy rental scripts outside the source checkout before execution because they deliberately replace that checkout with pinned revisions.

## Projection-sensitive frontier

The uniform all-W2/group-128 and projection-sensitive artifacts both exposed a
shared vLLM DSML tool-turn stop bug during coding-agent tests. Whamp/vLLM commit
`9a2ffbb4534400064e645cb4fef8ab2f2a987f11` fixed that runtime defect; those
runs did not establish artifact causality. The projection-sensitive frontier
keeps W2 and WNA16 but treats gate/up and down projections separately.
`deepseek-v4-run-frontier-screen` gathers independent
`w13` and `w2` reconstruction evidence. `deepseek-v4-build-frontier-recipes`
then applies two quality priors before measured byte allocation:

- down group size may never be coarser than gate/up;
- layers 26 and 42 anchor W4 down in `balanced`, while layers 37–42 anchor W4
  down in `quality`.

For the exact 72,317-tensor source header capture, the four bounded recipes are:

| Candidate | Modeled payload | Protected pattern |
| --- | ---: | --- |
| `cliff` | 74.238934 GiB | W2, coarse gate/up, finer down |
| `capacity` | 74.863934 GiB | W2/group-128 down throughout |
| `balanced` | 76.238934 GiB | W4 down at layers 26 and 42 |
| `quality` | 78.738934 GiB | W4 down at layer 26 and layers 37–42 |

The generated `config.json`, candidate manifest, and model card retain the
minimum mixed-group contract at
`Whamp/vllm@dd2d1fd6779addccc73094f77fa4ada7d9106a41`, tree
`f73b30cc5a2ed9de200ca2e4de3cdef1a06f6538`. The promoted runtime is
`Whamp/vllm@7b39c93043ffa88729d2cd3dd1f8f482df6ea98c`, tree
`670643653f99448f90192b79dd0842bcfa073ab8`; it retains the SwiGLU and DSML
fixes and uses FP8 Marlin for the grouped output projection. The quality
candidate passed all seven rollback-wrapped SM86 numerical/cubin cases, served
230,144 tokens, recalled a needle at 211,551 tokens, and sustained two
simultaneous 90K-token recalls. See
`.research/deepseek-v4-lowbit/CAPACITY-MARLIN-20260814.md` for the performance
and thin-VRAM-margin evidence.

`rental/run-verda-frontier-host.sh` is the bounded host orchestrator. Larger
A100 shapes disappeared before provisioning, so its guarded default is now one
on-demand `1A100.22V` in FIN-03 with one A100 80 GB, 120 GB host RAM, a 350 GiB
boot volume, and Ubuntu 24.04 CUDA 13.0. The 2026-08-13 live estimate was
$1.8859/hour including storage, or $30.1744 for the 16-hour cap, against a
$31.64546 `main` balance. This exact host class previously completed the same
converter at 112.9 GiB peak RAM with no swap. The remote runner uses one spawned
GPU worker; the same coordinator can use more fixed one-GPU workers when a
larger shape is intentionally pinned. Whole layers remain atomic during
screening, output shards remain disjoint during conversion, and the parent
retains canonical ordering, checksum-bound receipts, finalization, and
publication. The orchestrator rechecks live price, balance, capacity, image,
and SSH key before every create call; records exact VM and volume IDs; arms a
persistent deletion watchdog before provisioning; and proves zero VMs, zero
volumes, and zero running cost after completion.
`VERDA_FRONTIER_DRY_RUN=1` performs those checks without creating resources.
The host exports `VERDA_PROFILE` from `VERDA_FRONTIER_PROFILE`, which defaults
to `main`; it never mutates the active profile with `auth use`. Using another
profile is an explicit budget decision, not an automatic fallback. The host
accepts completion only after the remote process exits and an atomic receipt
binds the candidate revision to the final publication report hash.
The remote runner pins `HF_HOME`, `HF_HUB_CACHE`, and `HF_XET_CACHE` under the
rental root, disables the Xet chunk cache, and aborts if residual Hugging Face
cache data exceeds 2 GiB after source download.

### Recovered checkpoint resume

The 2026-08-13 screen completed before conversion failed on a JSON mapping-order
check. Its nine-file report set is preserved at immutable Hugging Face commit
`2686304a68557827d847e1954050cde6b5e7fd08`. The checked-in
`rental/frontier-recovery-manifest-20260813.json` binds those files to restored
Verda volume `9a7105b5-3c04-4bd7-b9fb-84c7be98c961`. Do not rerun the screen.

Resume uses two explicit host modes:

1. `validate-resume` boots that exact disk on `CPU.4V.16G`, rebuilds the recipe
   bundle from the recovered reports, requires byte identity with the recovered
   bundle, verifies source, baseline, imatrix, reusable shards, and any partial
   conversion receipts, then writes `frontier-resume-validation.json`.
2. `resume-conversion` boots the same disk on `1A100.22V`, requires the CPU
   receipt, skips source download, header capture, screening, recipe selection,
   and baseline-shard download, and converts only `quality`.

Run the non-billable live-contract checks before either stage:

```bash
VERDA_FRONTIER_RUN_MODE=validate-resume \
VERDA_FRONTIER_RESUME_VOLUME_ID=9a7105b5-3c04-4bd7-b9fb-84c7be98c961 \
VERDA_FRONTIER_DRY_RUN=1 \
bash tools/deepseek-v4-lowbit/rental/run-verda-frontier-host.sh

VERDA_FRONTIER_RUN_MODE=resume-conversion \
VERDA_FRONTIER_RESUME_VOLUME_ID=9a7105b5-3c04-4bd7-b9fb-84c7be98c961 \
VERDA_FRONTIER_DRY_RUN=1 \
bash tools/deepseek-v4-lowbit/rental/run-verda-frontier-host.sh
```

On ordinary failure, interruption, or watchdog expiry, the host deletes compute
without `--with-volumes`, verifies that the exact OS volume is detached, and
records a resumable `preserved` state. Only a completed immutable Hub
publication with locally verified completion evidence may use
`--delete-volume`. The CPU validation stage also preserves the volume on
success; the A100 conversion stage deletes it only after publication
verification.

Each rental campaign names exactly one candidate through
`VERDA_FRONTIER_CANDIDATE`; the default and first candidate is `quality`. The
remote runner captures source provenance, screens all routed experts, builds
the complete recipe bundle, converts only the selected candidate, publishes it
as a one-commit branch directly over immutable parent revision
`75d9286c37f3037f3ab390cfbc10747466eac714`, independently verifies its remote
inventory and branch history, and only then deletes local candidate bytes. A
failed verification preserves local output. Do not generate the remaining
ladder before the selected artifact passes the approved one-worker DeepSWE
`superjson-error-stack-serialization` gate. That gate requires
`Whamp/deep-swe-bench@4645f56d14137ed0e1aa409aee0d60e59215150e` with the
confirmed-plan `coding-agent-early-gate-v1` watchdog profile; the exact plan
identity still requires approval before execution.

The copied rental runner refuses to execute while
`CLUB_3090_REVISION` is unpinned. Publish and checksum-pin the exact source
revision before provisioning. An ambient `HF_TOKEN` can shadow the intended
stored credential; the runner verifies namespace write access without printing
the token.

## W3 constraint

The pinned vLLM implementation allocates each packed dimension using a pack factor of `32 // bits`. W3 therefore uses ten values per 32-bit word. DeepSeek V4's 4096- and 2048-wide expert matrices are not divisible by ten, so the current planner rejects W3. Supporting it requires a tested padding contract in the writer and loader; silently truncating dimensions would corrupt the artifact.
