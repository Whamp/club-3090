---
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
  - deepseek-v4
  - compressed-tensors
  - wna16
  - int2
  - experimental
---

# DeepSeek V4 Flash 0731 W2A16

This is an experimental, MTP-free W2A16 quantization of
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
It was built to test whether a roughly 77 GiB safetensors checkpoint could run
through vLLM and Humming on four 24 GB RTX 3090 GPUs.

The checkpoint loads and generates with a patched research runtime. It does not
work with stock vLLM, and its first measured runtime was much slower than
llama.cpp. Treat it as a reproducible research artifact, not a production
release.

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

The full conversion ran on one A100 80 GB VM. It used 112.9 GiB peak host
memory, no swap, and 2 hours 31 minutes of service CPU time. The upload verifier
then matched every local and remote file by byte size and Git, LFS, or Xet
object hash before the VM was deleted.

The immutable weight snapshot is
[`75d9286c37f3037f3ab390cfbc10747466eac714`](https://huggingface.co/hampsonw/DeepSeek-V4-Flash-0731-WNA16/tree/75d9286c37f3037f3ab390cfbc10747466eac714).
It contains 54 files totaling 82,464,249,582 bytes, including 45 model shards.

## Runtime compatibility

This checkpoint requires a research integration based on
[`haosdent/vllm@12810046c799cbe874967e19b1c0fa134ab7b209`](https://github.com/haosdent/vllm/tree/12810046c799cbe874967e19b1c0fa134ab7b209)
plus the checksum-pinned patches in
[club-3090](https://github.com/Whamp/club-3090/tree/d6776fbac8a4d062102e57d7cfe0e3eb4d0be1b6/models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86).
The final tested vLLM tree was
`12b87bcd52bb2973685fa8f38b5fc8bbbfe7519c`.

The patches provide:

- compressed-tensors W2 Humming MoE loading;
- separate routed-expert and native DeepSeek FP8 handling;
- DeepSeek V4 execution on SM86; and
- RTX 3090 sparse-attention fallbacks for kernels that exceed SM86 shared
  memory.

Stock vLLM does not implement this complete path. Do not expect
`vllm serve hampsonw/DeepSeek-V4-Flash-0731-WNA16` to work in an unpatched
environment.

## Measured results

The tested host used four RTX 3090 GPUs with tensor parallelism 4. The patched
runtime:

- loaded all 45 shards;
- generated a correct deterministic short response;
- exposed a 200,000-token context with 210,826 reported cache tokens;
- measured 809 prompt tokens/s on a 9,009-token prompt; and
- measured about 5.55 single-stream decode tokens/s on a short code prompt.

The comparison llama.cpp service measured about 1,379 prompt tokens/s on a
similar 9,212-token prompt and 34–38 decode tokens/s. The first WNA16 runtime
was therefore rejected as a replacement on performance grounds. The vLLM test
used conservative eager execution and correctness fallbacks, so these numbers
describe that runtime configuration rather than the fastest possible WNA16
implementation.

## Evaluation and limitations

Quality evaluation stopped after the candidate failed the performance gate.
Available evidence consists of:

- the 24-matrix reconstruction-error comparison;
- one exact deterministic short response; and
- one syntactically correct quicksort response that added prose despite an
  “only code” instruction.

No broad quality suite, 200K needle test, concurrency stress test, or comparison
against the official checkpoint was completed. This repository makes no claim
that W2A16 preserves the official model's benchmark scores or matches Antirez
or Unsloth GGUF quality.

Other limits:

- MTP/DSpark is absent.
- The artifact uses one uniform W2 recipe for every routed projection.
- The calibration imatrix came from the pre-0731 model.
- Runtime support depends on research patches and JIT-compiled Humming kernels.
- Only the four-RTX-3090 SM86 path and a generic A100 numerical oracle were
  tested.

## Reproduction

The converter, pilot, resumable writer, upload verifier, runtime patches, and
research record live in the
[`feat/deepseek-v4-lowbit-vllm`](https://github.com/Whamp/club-3090/tree/feat/deepseek-v4-lowbit-vllm)
branch of club-3090. Start with:

- [`tools/deepseek-v4-lowbit/README.md`](https://github.com/Whamp/club-3090/blob/feat/deepseek-v4-lowbit-vllm/tools/deepseek-v4-lowbit/README.md)
- [the experiment record](https://github.com/Whamp/club-3090/blob/feat/deepseek-v4-lowbit-vllm/.research/deepseek-v4-lowbit/PLAN.md)
- [the vLLM patch series](https://github.com/Whamp/club-3090/tree/feat/deepseek-v4-lowbit-vllm/models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86)

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
