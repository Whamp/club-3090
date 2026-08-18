# M5–M7 — full runtime acceptance

Decision: **M5, M6 functional gates, and M7 performance floors pass.** M8 paired DeepSWE remains required before promotion. The profile is a measured capacity ceiling with unsafe release headroom.

## Bring-up

M5 attempt 1 allocated and loaded the exact 21.19 GiB/rank raw plan, then failed during LM-head Q8 preparation because the original repacker expanded codes to a 1,010 MiB INT64 temporary with ~902 MiB free. This was a transient implementation defect, not steady-state capacity.

Whamp/vLLM `3ec20cebe` replaced whole-tensor INT64 expansion with 2,048-row INT32 chunks. Attempt 2 then passed:

- image `sha256:f91e8283e7ad116b8664b4a936dba88ebafcb8910a968dce2a3c34420f010adf`;
- GGUF SHA-256 `ca22ae2f…` and 86,720,111,488 bytes;
- full load in 271.90 s;
- model loading 21.53 GiB/rank;
- consumed weights + non-Torch 22.01 GiB/rank;
- peak activation 0.27 GiB/rank;
- actual CUDA graph pool 0.06 GiB/rank;
- KV cache 0.81 GiB/rank / 154,519 tokens;
- 1.10× concurrency at 140,000 context;
- zero serving-process swap after RAM-gated normalization.

The service reached API readiness with Ampere FlashMLA decode, Triton sparse-prefill/indexer fallback, HIERARCHICAL then PYNCCL collective dispatch, breakable CUDA graphs, and the custom GGUF quant/load method.

## Functional and long-context correctness

The live service returned:

- exact deterministic `GGUF TP READY`;
- a valid automatic `get_weather({"city":"Paris"})` call;
- coherent post-tool continuation;
- exact `NEEDLE-GGUF-842731` retrieval from a 119,730-token prompt, normal stop, 230.02 s wall time;
- zero residual requests/KV and zero swap afterward.

Quick quality scored **27/30 pass@1 and 27/30 pass@3**:

- ToolCall-15: 12/15;
- InstructFollow-15: 15/15.

The prior WNA quick gate was 25/30 pass@1 and 26/30 pass@3. These packs are smoke evidence, not a substitute for M8.

## Matched performance

### Decode

Three warmups plus five measured 512-token length-capped generations:

- **76.6973 wall tok/s mean**;
- 0.0334% CV;
- every measured response generated 512 tokens with `finish_reason=length`.

This exceeds the 58 floor and 70 target. The same inherited WNA speed stack measured 74.98 tok/s, so GGUF-TP is approximately 2.3% faster in this matched single-stream screen.

### Prefill

Three cache-busted ~9K prompts with unique prefixes before filler:

- 548.23, 553.36, 554.07 tok/s;
- **551.89 tok/s mean**.

This clears the 550 floor by only 0.34% and misses the 700 target. It is 37.8% below the WNA speed stack's 887.52 tok/s. Prefill is the leading promotion risk and has essentially no regression margin.

### Concurrency 2

Two simultaneous 512-token requests completed normally:

- 61.16 and 61.05 tok/s per stream;
- 121.86 aggregate tok/s;
- zero swap, zero residual requests/KV.

This covers short requests only, not two 140K contexts.

## Capacity warning

Idle physical headroom was 101–102 MiB after readiness and 71–73 MiB after long-context JIT. This is below the normal 1 GiB release guard. Will accepted the 140–142K context floor for development, but this measured profile remains a ceiling and must carry the warning into any promotion decision.

## Next gate

The exact one-seed SuperJSON DeepSWE pilot plan is compiled and awaits explicit plan-hash approval. M8 then requires at least three seeds per engine on all 12 tasks (≥72 cells), task-clustered analysis, mean-partial-reward non-inferiority, and strict solves no lower than Antirez llama.cpp baseline minus one.

Evidence: `evidence/m5-m7-runtime/`.
