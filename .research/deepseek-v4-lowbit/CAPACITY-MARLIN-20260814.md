# DeepSeek V4 quality-candidate capacity campaign

Date: 2026-08-14
Host: server60, four RTX 3090 GPUs, tensor parallel size 4
Artifact: `hampsonw/DeepSeek-V4-Flash-0731-WNA16@12035985bf555d0ddc603c6305586a8fa915589c`
Canonical runtime: `Whamp/vllm@7b39c93043ffa88729d2cd3dd1f8f482df6ea98c`
Runtime tree: `670643653f99448f90192b79dd0842bcfa073ab8`
Final image: `sha256:f56910530683326051cfdf4e7c8e4d6afc5bace8804cb78b2af9ea799bbba4e6`

## Result

The selected profile serves 230,144 tokens with `max_num_seqs=2`, `max_num_batched_tokens=256`, `gpu_memory_utilization=0.98`, and no CPU weight offload. It uses FP8 Marlin for DeepSeek V4's grouped `wo_a` output projection instead of retaining one BF16-dequantized copy per attention layer.

Matched performance against the fresh 131,072-token quality-candidate baseline:

| Metric | Baseline | Selected profile | Change | Required floor |
| --- | ---: | ---: | ---: | ---: |
| Decode | 61.56 tok/s | 61.91 tok/s | +0.57% | 46.17 tok/s |
| Cache-busted prefill | 920.91 tok/s | 875.93 tok/s | -4.88% | 552.546 tok/s |
| Aggregate KV capacity | 148,290 tokens | 275,238 tokens | +85.6% | — |
| Advertised context | 131,072 | 230,144 | +75.6% | — |

The 260,000-token probe failed cleanly: 1.22 GiB of KV cache was required and 1.21 GiB was available. vLLM estimated a 255,232-token allocation ceiling. The profile stays at 230,144 rather than operating at that exact allocator boundary.

## Causal mechanism

The residency diagnostic found a persistent 16 MiB BF16 `wo_a` tensor in each of 43 attention layers: 688 MiB per rank. The original SM86 path dequantized the block-FP8 weight once and retained the BF16 copy for `torch.einsum`.

Disabling that cache increased available KV memory but forced full dequantization on every call. It reached 230,144 context but reduced decode to 34.01 tok/s, below the 46.17 tok/s floor.

The accepted path keeps the original FP8 weights, packs them for the existing FP8 Marlin kernel, flattens the two local groups into one projection, and selects the block-diagonal group outputs. This removes the retained BF16 copy without paying per-token dequantization. It also reduced measured peak activation from about 0.32 GiB to 0.29 GiB and CUDA graph memory from about 0.09 GiB to 0.07 GiB in the selected profile.

The path is opt-in through `VLLM_DSV4_WO_A_MARLIN_DIAGONAL=1`. Default behavior is unchanged.

## Experiment decisions

| Change | Capacity result | Performance result | Decision |
| --- | --- | --- | --- |
| `max_num_seqs`: 4 → 2 | 138,240 context at batch budget 256 | 61.45 decode, 968.49 prefill | Useful but small capacity gain |
| `max_num_batched_tokens`: 256 → 128 | 156,000 context | 59.19 decode, 465.90 prefill | Rejected: prefill below floor |
| BF16 `wo_a` cache disabled | 230,144 context | 34.01 decode, 899.36 prefill | Rejected: decode below floor |
| FP8 Marlin diagonal `wo_a` | 230,144 context, 275,238 aggregate KV tokens | 61.91 decode, 875.93 prefill | Selected |
| 260,000 context | Planner rejected; estimated ceiling 255,232 | Not run | Rejected: allocation boundary |

CPU/UVA weight offload was not retested. Earlier matched evidence showed 1–2 GiB per-rank offload reduced decode to 12.68–19.54 tok/s, far below the current floor.

## Correctness and long-context evidence

The selected path passed:

- deterministic basic generation;
- automatic tool selection;
- a tool call followed by a tool result and a normal 18-token final response;
- IDE-agent, multi-turn-agent, coding, and 7,862-token reasoning probes;
- exact NIAH recall at 9,213, 27,513, 141,000, and 211,551 prompt tokens on the acceptance image, plus 211,031 tokens on the clean final image;
- two simultaneous requests with 90,029 prompt tokens each and distinct exact recalls;
- five matched 512-token decode runs and three cache-busted approximately 9K-token prefill runs;
- zero serving-process swap and only 8 MiB post-benchmark VRAM growth on the clean final image.

## Failed production gate

`verify-stress.sh` reported one failure even though every functional probe passed. After the 211,551-token request, physical VRAM headroom was 93 MiB per GPU, below the script's 1 GiB sustained-agent advisory threshold. Rank 0 also held about 437 MiB of reclaimable Torch allocator cache.

Treat 230,144 as a measured single-request capacity profile, not a generous-headroom profile. Two-request validation covered two 90K requests within the 275K aggregate pool; it does not support two simultaneous 230K requests.
