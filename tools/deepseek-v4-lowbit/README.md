# DeepSeek V4 low-bit artifact tools

CPU-side planning and conversion tools for the MTP-free, mixed-WNA16 DeepSeek V4 Flash artifact described in [the research plan](../../.research/deepseek-v4-lowbit/PLAN.md).

## Exact artifact planner

The planner reads captured safetensors headers without loading model weights. It:

- preserves non-MTP, non-routed-expert tensors byte-for-byte;
- omits every `mtp.*` tensor;
- replaces routed-expert `w1`, `w3`, and `w2` weights and source scales with symmetric group-size-128 Humming WNA16 weights and FP16 scales;
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

The JSON result separates preserved bytes, new packed weights and scales, replaced source bytes, and omitted MTP bytes. `total_bytes` is the expected raw tensor payload; filesystem and safetensors-header overhead are not included.

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

## W3 constraint

The pinned vLLM implementation allocates each packed dimension using a pack factor of `32 // bits`. W3 therefore uses ten values per 32-bit word. DeepSeek V4's 4096- and 2048-wide expert matrices are not divisible by ten, so the current planner rejects W3. Supporting it requires a tested padding contract in the writer and loader; silently truncating dimensions would corrupt the artifact.
