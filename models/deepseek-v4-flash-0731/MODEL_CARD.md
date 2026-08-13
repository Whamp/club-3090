---
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: hampsonw
tags:
  - deepseek-v4
  - compressed-tensors
  - wna16
  - int2
  - experimental
---

# DeepSeek V4 Flash 0731 routed-expert W2A16

This is an experimental, MTP-free quantization of
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
Its routed experts use W2A16, while other tensors retain their source values and
dtypes. It was built to test whether a roughly 77 GiB safetensors checkpoint
could run through vLLM and Humming on four 24 GB RTX 3090 GPUs.

The checkpoint loads and generates with a patched SM86 runtime. It does not
work with stock vLLM. On four RTX 3090s, the selected 215,000-token profile
measures 60.79 single-stream decode tokens/s. Short-request concurrency tests
passed with two and four simultaneous requests.

## What changed

- Every routed-expert `w1`, `w3`, and `w2` matrix uses symmetric W2A16
  quantization with group size 128.
- Packed weights use the compressed-tensors 0.17.0 WNA16 format with FP16 group
  scales and INT32 packed values.
- Non-routed tensors retain their source values and dtypes, including the native
  DeepSeek FP8 attention and shared-expert weights.
- The MTP/DSpark draft layers are omitted. `num_nextn_predict_layers` is zero.
- The resulting checkpoint contains 45 safetensors shards. The three omitted
  source shards contained only MTP tensors.

## How it was created

The conversion used the official checkpoint at revision
[`7872f01b1d1fe23eabc4c98b48bffcef5a386062`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062).
It processed one source shard at a time and wrote checksum-bound, resumable
safetensors shards.

For each routed-expert matrix, the converter:

1. used Intel AutoRound's DeepSeek V4 path to decode the source MXFP4/E8M0
   representation;
2. fitted symmetric 2-bit weights with group size 128;
3. weighted the scale search with Antirez's routed-expert activation imatrix;
4. packed the signed codes with compressed-tensors 0.17.0; and
5. recorded weighted and unweighted reconstruction error in
   `conversion-metrics.json`.

The conversion selected imatrix-weighted RTN after a bounded comparison on 24
matrices from layers 0, 26, 37, and 42. It improved weighted reconstruction
error in all 24 comparisons. The median improvement was 31.03% relative to
plain RTN. This result measures reconstruction error, not end-to-end model
quality.

The Antirez imatrix predates the 0731 checkpoint. Antirez later reused it for
0731 GGUFs, and the 24-matrix comparison showed that it improved this WNA16
conversion, but it is not a 0731-native calibration run.

### Pinned inputs and tools

| Component | Revision or version |
| --- | --- |
| Base checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Routed-expert imatrix | `antirez/deepseek-v4-gguf@e7f04037032990db0346398d249baf9fb9df1ccc` |
| Imatrix content SHA-256 | `02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e` |
| AutoRound | `intel/auto-round@f17d9cd4b36982006bad21ff87127aac739072e3` |
| compressed-tensors | `0.17.0` |
| Humming | `humming-kernels==0.1.10`, source `inclusionAI/humming@4351af3a8fcdce1a8dee50104ba49566af2427fb` |
| Conversion and runtime record | [`Whamp/club-3090@d6776fba`](https://github.com/Whamp/club-3090/tree/d6776fbac8a4d062102e57d7cfe0e3eb4d0be1b6) |

The full conversion ran on one A100 80 GB VM and used 112.9 GiB peak host
memory with no swap.

The immutable weight snapshot is
[`75d9286c37f3037f3ab390cfbc10747466eac714`](https://huggingface.co/hampsonw/DeepSeek-V4-Flash-0731-WNA16/tree/75d9286c37f3037f3ab390cfbc10747466eac714).
It contains 54 files totaling 82,464,249,582 bytes, including 45 model shards.

## Runtime compatibility

This checkpoint requires the experimental integration in
[`Whamp/vllm#1`](https://github.com/Whamp/vllm/pull/1), based on
[`haosdent/vllm@12810046c799cbe874967e19b1c0fa134ab7b209`](https://github.com/haosdent/vllm/tree/12810046c799cbe874967e19b1c0fa134ab7b209).
The canonical source branch is
[`incubate/deepseek-v4-wna16-sm86`](https://github.com/Whamp/vllm/tree/incubate/deepseek-v4-wna16-sm86).
Club-3090 retains a
[checksum-pinned deployment mirror](https://github.com/Whamp/club-3090/tree/26ae767aa98c14761ac4a69d4f492f418fd29578/models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86).
Both paths produce the final tested vLLM tree
`aeb62948e33074514a742d19c2f9a1a3c2ee3e1f`.

The patches provide:

- compressed-tensors W2 Humming MoE loading;
- separate routed-expert and native DeepSeek FP8 handling;
- DeepSeek V4 execution on SM86; and
- RTX 3090 sparse-attention fallbacks for kernels that exceed SM86 shared
  memory; and
- runtime-bounded DeepSeek V4 RoPE cache materialization that preserves the
  model's original YaRN frequency span.

Stock vLLM does not implement this complete path. Do not expect
`vllm serve hampsonw/DeepSeek-V4-Flash-0731-WNA16` to work in an unpatched
environment.

## Measured results

The promoted server60 profile uses four RTX 3090 GPUs with tensor parallelism 4,
breakable CUDA graphs, `max_num_seqs=4`, a 256-token batch budget, no CPU weight
offload, FP8 DeepSeek MLA cache, and the required SM86 sparse-attention
fallbacks. It serves 215,000 tokens.

A thermally warm matched code benchmark used 3 warmups and 5 measured
512-token generations:

| Runtime | Mean decode | CV | Mean TTFT |
| --- | ---: | ---: | ---: |
| llama.cpp Unsloth UD-IQ1_M baseline | 32.67 tok/s | 0.1% | 109 ms |
| vLLM WNA16, selected 215K image | 60.79 tok/s | 0.0% | 218 ms |

The vLLM result is about 86% faster on this single-stream decode workload. A
matched eager ablation measured 4.96 tok/s, while graph-enabled runs before and
after it measured 60.27 and 60.07 tok/s. This isolates breakable CUDA graph
replay as the cause of the gain.

Additional validation on the selected profile:

- three cache-busted 8,984-token prefill runs averaged 968.97 prompt tok/s;
- exact needle retrieval passed at 204,900 prompt tokens;
- concurrency 2 sustained 65.47 aggregate tok/s and 41.55 tok/s per stream;
- concurrency 4 sustained 89.94 aggregate tok/s and 29.86 tok/s per stream;
- both concurrency tests reported zero post-warm VRAM growth;
- deterministic short generation, forced tool calling, and separated reasoning
  output passed;
- `verify-full` passed; and
- `verify-stress` passed every functional class and all ceiling rungs through
  197,580 tokens, with zero serving-process swap.

The runtime-bounded RoPE patch removed about 407 MiB of registered storage per
rank at 215K. A 230K profile with `max_num_seqs=4` failed admission and estimated
a 215,552-token maximum. Reducing the request bound to two raised the estimate
to 223,488 but did not justify losing concurrency 4, so the selected profile
stops at 215,000 tokens.

## Evaluation and limitations

Available quality evidence consists of:

- the 24-matrix reconstruction-error comparison;
- deterministic short-generation and parser canaries;
- exact needle retrieval at 204,900 prompt tokens; and
- a syntactically correct quicksort response that still added Markdown fencing
  despite an “only code” instruction.

The historical benchlocal quick result is smoke evidence only. The capability
quality gate will use DeepSWE through `~/evals/deep-swe-bench/`; no broader
benchlocal run is planned for this path.

No broad comparison against the official checkpoint or Antirez and Unsloth
GGUF quants has been completed. The context tests prove selected retrieval
cases, not general long-context quality. The repository stress suite passed all
functional probes but returned nonzero on its generic 1 GiB free-VRAM policy:
the minimum observed reserve was 127 MiB per card. That policy was written for
llama.cpp and is not the vLLM acceptance rule here. The 215K profile instead
passed repeated startup, decode, prefill, near-ceiling retrieval, and short
concurrency tests without CUDA allocation failure, serving-process swap, or
post-warm VRAM growth. The reserve remains narrow and is reported, not hidden.

Other limits:

- MTP/DSpark is absent.
- The artifact uses one uniform W2 recipe for every routed projection.
- The calibration imatrix came from the pre-0731 model.
- Runtime support depends on research patches and JIT-compiled Humming kernels.
- Only the four-RTX-3090 SM86 path and a generic A100 numerical oracle were
  tested.
- Concurrency 4 is validated for short requests, not four simultaneous
  215,000-token requests.

## Reproduction

The converter, pilot, resumable writer, upload verifier, runtime patches, and
selected Compose contract are pinned in
[`Whamp/club-3090@26ae767a`](https://github.com/Whamp/club-3090/commit/26ae767aa98c14761ac4a69d4f492f418fd29578).
Start with:

- [the canonical vLLM experiment](https://github.com/Whamp/vllm/pull/1)
- [`tools/deepseek-v4-lowbit/README.md`](https://github.com/Whamp/club-3090/blob/26ae767aa98c14761ac4a69d4f492f418fd29578/tools/deepseek-v4-lowbit/README.md)
- [the final experiment record](https://github.com/Whamp/club-3090/blob/d10ccf25da5551cbddbac42e228d3260856e8db4/.research/deepseek-v4-lowbit/PLAN.md)
- [the vLLM patch series](https://github.com/Whamp/club-3090/tree/26ae767aa98c14761ac4a69d4f492f418fd29578/models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86)
- [the direct four-GPU Compose](https://github.com/Whamp/club-3090/blob/26ae767aa98c14761ac4a69d4f492f418fd29578/models/deepseek-v4-flash-0731/vllm/compose/multi4/wna16/base.yml)

## Credits

- DeepSeek created and released DeepSeek-V4-Flash-0731.
- Antirez created the routed-expert imatrix used to guide this quantization.
- Intel AutoRound provided the source dequantization and RTN scale-search
  primitives.
- Neural Magic and the vLLM project provide the compressed-tensors format and
  loader infrastructure.
- InclusionAI provides the Humming WNA16 kernels.
- Haosdent and Lasimeri developed the Ampere DeepSeek V4 work that made the
  runtime experiment possible.

## License

The base checkpoint and this redistributed quantization use the
[MIT License](https://huggingface.co/hampsonw/DeepSeek-V4-Flash-0731-WNA16/blob/main/LICENSE).
DeepSeek's copyright and permission notice are included in this repository.
