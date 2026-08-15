# Server60 SM86 speed campaign — 2026-08-15

## Decision

Promote the combined FlashMLA decode and hierarchical all-reduce profile:

- `VLLM_DSV4_FLASH_MLA_DECODE=1`
- `VLLM_HIER_ALL_REDUCE=0,1;2,3`
- runtime image `sha256:eb2884fc60ee332d7adb9d5e424e35acf8817dad0f93c8bb7ea7095cb8f58a0e`
- vLLM commit `b7766cfe4d15d9b68acea43097ceff221e8a739f`, tree `6354125afd1306c9286f734d1c47c23c767d77a9`

The profile keeps the projection-sensitive WNA16 artifact, `fp8_ds_mla`, 230,144-token served context, TP=4, `max_num_seqs=2`, `max_num_batched_tokens=256`, zero CPU weight offload, and the 16 GiB host KV eviction tier.

## Matched results

| Comparison | Decode tok/s | Decode change | Prefill tok/s | Prefill change |
| --- | ---: | ---: | ---: | ---: |
| Final default baseline → `BLOCK_M=2` | 62.06 → 62.18 | +0.19% | 895.78 → 897.72 | +0.22% |
| Fresh plain baseline → hierarchy | 61.17 → 65.00 | +6.26% | 917.19 → 929.15 | +1.30% |
| Fresh FlashMLA parent → combined | 71.04 → 74.98 | +5.55% | 922.15 → 887.52 | −3.76% |
| Fresh plain baseline → combined | 61.17 → 74.98 | +22.58% | 917.19 → 887.52 | −3.23% |

All accepted measurements reported zero serving-process swap. The combined prefill result is a measured tradeoff, not a claimed win. Its three samples were noisy (3.8% CV), remained above the 552.546 tok/s floor, and long-depth prefill matched FlashMLA-only behavior closely: 764.9 tok/s versus 768.2 tok/s near 211K.

`BLOCK_M=2` is rejected as noise. `indexer96` was not run because the trace did not establish its mediator. `batched320` was not run because the winner retained only 58 MiB at the 211K depth gate, below its 256 MiB prerequisite.

## Correctness and depth

The promoted combination passed:

- 17 pinned AppMana SM86 numerical cases and packaged `sm_86` cubin checks;
- hierarchical BF16 oracle checks at 4,096 through 262,144 elements;
- basic generation, automatic tool call, post-tool continuation, tool-response prefill, IDE-agent, multi-turn agent, and coding probes;
- a 7,726-token reasoning completion;
- exact needle recall at approximately 9K, 27K, 55K, 85K, 94K, 124K, 154K, 184K, and 211K tokens;
- a fill to 211,551 tokens with 58 MiB physical VRAM free at the final rung;
- zero serving-process swap.

The profile remains a measured capacity ceiling, not a generous-headroom configuration.

## Nsight gate

The raw report is retained on server60 at:

`/home/will/inference/runtime/deepseek-v4-wna16-sm86/speed-experiments-20260815/nsight-baseline-6/profile/deepseek-v4-decode-baseline.nsys-rep`

Its SHA-256 is recorded in `nsight-baseline-6/analysis/profile.sha256`. The raw 75 MB report and derived 258 MB SQLite database are intentionally excluded from Git.

The summed-kernel screen reported 19.65% NCCL time. Timeline interval review found no overlap between NCCL and non-NCCL kernels on any GPU. The conservative GPU3 lower bound was 791.739 ms of NCCL over a 4,615.287 ms decode span, or 17.15%. GPUs 0–2 measured 19.89%, 20.95%, and 20.40%. This cleared the 10% gate for the hierarchical experiment.

## Provenance

Each run directory contains its approved `plan.json`, `plan.sha256`, dispatch or oracle evidence, benchmark or stress output, and swap record. `SHA256SUMS` covers every committed evidence file in this directory.
