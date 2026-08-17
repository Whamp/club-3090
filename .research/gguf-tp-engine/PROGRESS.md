# GGUF-TP engine — progress log

Branch `feat/gguf-tp-engine` (club-3090, plans/evidence) ·
`incubate/gguf-tp-sm86` (Whamp/vllm, implementation).

## 2026-08-17 — goal started; M0 (local part) + M1 (contract)

- Goal `1aeea276-cf88-4117-8161-aeee24bbdfbf` created (plan v5 @ `2485108f`).
- Skills loaded: perform-like-jeff-and-sanjay, nvidia-cuda-performance, testing.
- **M0 done (local):** vLLM worktree `/home/will/projects/vllm/.worktrees/gguf-tp-sm86`
  created on `incubate/gguf-tp-sm86` from `b7766cfe4d15d9b68acea43097ceff221e8a739f`
  (tree `6354125afd1306c9286f734d1c47c23c767d77a9` — verified equals plan pin).
- **M0 deferred (server60):** fresh nsys trace of the 74.98 WNA16 stack.
  Requires standing the WNA16 service back up (server60 currently runs the
  canonical Antirez llama.cpp service on 8033) → authorized-window item with
  the validated rollback contract. Consumer of the trace is the M2 screening
  projection only; does not gate M1. Existing baseline-6 trace
  (SHA `c0e0ec99…`, pre-FlashMLA mix) is the interim anchor.
- **M1 started:** `FORMAT-CONTRACT.md` v1 written — exact byte layouts and
  decode operation order for q8_0 / q2_K / iq2_xxs with pinned-source line
  citations, GGUF tensor-axis contract (down-projection K/N swap,
  fused_wqa_wkv slot order), L0 oracle spec, aligned-SoA repack gate.
- Next (M1, all local): L0 oracle (pinned C reference vs independent
  NumPy-fp32 decoder, random+adversarial, bitwise pass);
  per-tensor inventory via read-only server60 GGUF headers;
  §4.7 TP mapping table; per-kernel dtype contracts; tokenizer pin tests;
  wo_a design; capacity table.

## 2026-08-17 — M1 L0 oracle PASS (class-A gate)

- `oracle/ref_a.c`: verbatim extraction of dequantize_row_q8_0 / q2_K / iq2_xxs
  + fp16→fp32 + tables from Whamp/llama.cpp@0379cf4bf; compiled standalone.
- `oracle/l0_oracle.py`: independent NumPy-float32 decoders written from
  FORMAT-CONTRACT.md; 10,000 random blocks/format (seed 20260817, finite-scale
  masking), adversarial corpora (LUT boundaries, sub-scale extremes, chunk
  boundaries, scale-nibble extremes, ±max/subnormal d), NaN/Inf probe with
  NaN-aware compare.
- Result: **bitwise pass 100%** for q8_0, q2_K, iq2_xxs (random + adversarial
  + nonfinite). Evidence: `evidence/l0-report.json` (struct sizes 34/84/66,
  qs offsets 2/16/2, table SHA-256s).
- Red→green discrimination: first run failed q2_K from weight 32 on — the
  independent decoder wrote chunk-1 outputs at weights 32..159 instead of
  128..255 (`32*chunk` vs `128*chunk`). Fixed; contract text unchanged.
