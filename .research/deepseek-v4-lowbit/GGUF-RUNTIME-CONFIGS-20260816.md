# DeepSeek V4 GGUF runtime configurations

## Conclusion

Only the **Antirez IQ2_XXS** artifact ran the validated DeepSeek-specific high-performance prefill path.

The Unsloth UD-IQ1_M router had a preset named `fast`, but that name meant the shortest-context stock preset. It ran upstream llama.cpp `b10200`; it did not contain the external fork's capture-always prefill scheduler, Q8 repair, or `DSV4_*` runtime contract. More importantly, UD-IQ1_M expert matrices failed the fork's MMQ graph-safety check, so the Unsloth artifact was not a drop-in replacement for the validated Antirez profile.

## Configuration matrix

| Configuration | Weights | Engine | Context and batch shape | Evidence-backed role |
| --- | --- | --- | --- | --- |
| Antirez high-performance prefill | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` | `alesha-pro/llama.cpp` `ds4-longctx@b001c8cd7` plus Whamp Q8 repair `0379cf4bf`; published image `ghcr.io/whamp/llama-cpp-ds4-longctx@sha256:a96bd947d63eb81d8baf9f6f5ecb26669476383976717237450fbb5727b03745` | 430,080 context, batch 8,192, ubatch 384, q8_0 K/V, layer split `1,1,0.95,1.05`, one slot | The only validated capture-always fast-prefill profile; DeepSWE max baseline used its port 8033/model identity and solved 6/12 tasks |
| Antirez stock | Same Antirez IQ2_XXS GGUF | Upstream llama.cpp `b10200`, image digest `sha256:48b8053c05319cde97e64463d117b5747d3fb27475b176f85edf27bd503fa7f9` | 200,000 context, batch/ubatch 2,048, q8_0 K/V, equal layer split, one slot | Correct stock comparison and earlier port-8200 baseline; not the high-performance path |
| Unsloth UD-IQ1_M `fast` | Three-shard `DeepSeek-V4-Flash-0731-UD-IQ1_M` | Upstream llama.cpp `b10200`, image digest `sha256:48b8053c05319cde97e64463d117b5747d3fb27475b176f85edf27bd503fa7f9` | 200,000 context, ubatch 2,048, q8_0 K/V, equal layer split, one slot | Router preset optimized for shorter context; measured 32.67 decode tok/s baseline; not the external fast-prefill fork |
| Unsloth UD-IQ1_M `balanced` | Same | Same upstream image | 393,216 context, ubatch 768, `output_norm.weight=CUDA3` | More context at lower prefill throughput |
| Unsloth UD-IQ1_M `long` | Same | Same upstream image | 655,360 context, ubatch 512, `output_norm.weight=CUDA3` | Maximum advertised router context; no high-performance-prefill validation |

## What made the Antirez path different

The specialized engine combined several linked mechanisms:

- capture-always CUDA graphs for prefill micro-batches;
- stable-topology graph construction;
- resident MoE scheduling and fused MMQ;
- sparse attention and allocator controls;
- repaired packed-Q8 concat, CUDA-native Q8 padding, and type-preserving long-context gathers;
- `LLAMA_ATTN_ROT_DISABLE=1` and matching q8_0 K/V;
- a mandatory synthetic warm-up to `context_size - 512` before opening the public port.

A graph-only port to stock `b10200` launched graphs but did not produce the target speedup. The complete system was required.

## Measured performance

On four RTX 3090 GPUs at the rig's unchanged 230 W/card setting:

- 200K Antirez fork, repaired Q8: 1,538 tok/s at 98K; 1,475 at 130K; 1,374 at 180K; 1,341 at 195K.
- 430K Antirez fork: 1,056 tok/s at 263K and 913 tok/s at 395,282, both with exact needle recall.
- Coding-agent decode held about 33–38 tok/s.
- Full 430K startup warm-up took about 26 minutes; a 200K warm-up took about eight minutes.
- The stock `b10200` Q8 path took 21.8 seconds to first token at 33K in the measured agent loop, versus about 7.8 seconds on the repaired fork.

The 430K profile reached only 794 MiB minimum free VRAM, below the repository's normal 1,024 MiB production margin. The 200K profile is the higher-margin fallback.

## Quant compatibility boundary

The validated Antirez artifact uses IQ2_XXS for routed gate/up projections and Q2_K for routed down projections. The local Unsloth UD-IQ1_M and role-splice artifacts contain IQ1_M expert matrices. Those matrices failed the MMQ graph-safety check, so changing only the model path could not carry Unsloth onto the high-performance profile.

This is a kernel-path compatibility result, not a judgment that Unsloth UD-IQ1_M has lower model quality. Both artifacts have shown strong qualitative behavior. The Antirez artifact is simply the one for which the fast-prefill engine was proven.

## Benchmark provenance

The 12-task GGUF comparator in the reconstructed WNA16 report was Antirez, not Unsloth:

- DeepSWE config: `baseline-llamacpp-deepseek-v4-flash-0731-iq2xxs@1.0.0`.
- Endpoint: `http://100.92.238.117:8033/v1`.
- Served model: `deepseek-v4-flash-0731-q8-fast-prefill`.
- Context: 430,080; one slot; q8_0 KV.
- Max-reasoning result: 6/12 strict solves, 96.57% mean partial reward.

The Unsloth UD-IQ1_M figure used during the vLLM campaign was the matched llama.cpp throughput baseline: 32.67 decode tok/s. It was not the 12-task GGUF quality comparator.

## Current server60 artifacts

Read-only inspection on 2026-08-16 found:

- the active service remained the vLLM WNA16 container;
- stopped Antirez high-performance container `llama-cpp-deepseek-v4-q8-fast-prefill` retained the exact Antirez model mount and all `DSV4_*` settings;
- stopped Unsloth router `llama-cpp-deepseek-v4-flash-router` retained upstream `b10200` image digest `sha256:48b8053c05319cde97e64463d117b5747d3fb27475b176f85edf27bd503fa7f9` and the three `fast`/`balanced`/`long` presets;
- stopped stock Antirez container `llama-cpp-deepseek-v4-flash` also used upstream `b10200`.

No service was changed during this research.

## Sources

- `Whamp/club-3090@19f65b82`, `models/deepseek-v4-flash-0731/INTERNALS.md`.
- `Whamp/club-3090@19f65b82`, `models/deepseek-v4-flash-0731/llama-cpp/compose/multi4/antirez-iq2-xxs/fast-prefill.yml`.
- `Whamp/club-3090@19f65b82`, `scripts/lib/profiles/engines/llama-cpp-ds4-longctx.yml`.
- `Whamp/club-3090@19f65b82`, `docs/UPSTREAM.md`, external DeepSeek fast-prefill dependency row.
- DeepSWE config `configs/baseline-llamacpp-deepseek-v4-flash-0731-iq2xxs@1.0.0/README.md`.
- DeepSWE report `reports/wna16-final-12v2/comparison.json`.
- server60 Unsloth preset `/home/will/inference/serving/club-3090/.worktrees/deepseek-v4-role-splice-profiles/models/deepseek-v4-flash-0731/llama-cpp/presets/deepseek-v4-router-unsloth.ini`, SHA-256 `71497fa648a9dd708b134aa5b6a19e8e5ac2611c626203ae841f78102c855de3`.
- server60 Unsloth Compose file beside that preset, SHA-256 `858529f4da52d634dc59ccdcdeee607fe2ffea6ff1246a489241e276273e118b`.
- Read-only `docker inspect` of the three stopped server60 llama.cpp containers on 2026-08-16.
