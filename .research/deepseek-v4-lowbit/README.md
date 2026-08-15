# DeepSeek V4 low-bit research index

Start with [`REPORT.md`](REPORT.md). It is the cold-reader account of the artifact, runtime integration, failures, causal experiments, final measurements, limitations, and safe public claims.

## Documents

| Document | Purpose |
| --- | --- |
| [`REPORT.md`](REPORT.md) | Shareable end-to-end result and evidence map |
| [`PLAN.md`](PLAN.md) | Artifact design, source research, rental, conversion, and implementation ledger |
| [`VLLM-PERFORMANCE-RESEARCH.md`](VLLM-PERFORMANCE-RESEARCH.md) | vLLM scheduler, CUDA graph, KV-cache, attention, memory, and communication research |
| [`VLLM-PERFORMANCE-PLAN.md`](VLLM-PERFORMANCE-PLAN.md) | Performance campaign plan and execution ledger |
| [`CAPACITY-MARLIN-20260814.md`](CAPACITY-MARLIN-20260814.md) | Causal 131K-to-230K capacity campaign |
| [`evidence/capacity-marlin-20260814/`](evidence/capacity-marlin-20260814/) | Compact final-image evidence and SHA-256 manifest |

## Immutable external evidence

- Full quantization screen and recovered recipe bundle: `hampsonw/DeepSeek-V4-Flash-0731-WNA16`, branch `evidence-quant-frontier-screen-20260813`, commit `2686304a68557827d847e1954050cde6b5e7fd08`.
- Projection-sensitive artifact: revision `12035985bf555d0ddc603c6305586a8fa915589c`.
- Validated runtime source: `Whamp/vllm@7b39c93043ffa88729d2cd3dd1f8f482df6ea98c`, tree `670643653f99448f90192b79dd0842bcfa073ab8`; merged through Whamp/vLLM PR #2 as `28db4816298293b74fca358cf735ac51c5144acb`.
- Deployment source: `Whamp/club-3090@e625a892b3af6c32ec22394e2341eed4bb8bdc17` plus later research-index commits.

## Local forensic archives

The working tree retains several large untracked Verda recovery directories. They are not canonical publication artifacts and must not be added to Git blindly. Their irreplaceable screen and recipe contents are already preserved in the immutable Hugging Face evidence commit above. The local copies remain useful for orchestration forensics and byte-level recovery checks.
