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
