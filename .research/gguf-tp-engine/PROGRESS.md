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

## 2026-08-17 — M1 per-tensor inventory complete (read-only server60)

- `oracle/gguf_inventory.py` (bounded 16 MiB header read, fail-closed on
  unknown types) run against the pinned blob on server60; SHA-256 re-verified
  `ca22ae2f…b1c0`. Full directory: `evidence/gguf-inventory.json`; family
  summary: `evidence/gguf-family-summary.txt`.
- Consistency proofs: 1,328 tensors; offsets monotonic, zero overlaps; last
  tensor ends exactly at file_size − data_start (data_start 5,333,824);
  Σ(nbytes) + Σalignment-gaps (86×20B + 16B + 28B) = capacity exactly.
- Family bytes (total 80.7594 GiB, matches 2026-08-13 audit): routed-experts
  72.5625 (IQ2_XXS gate/up [4096,2048,256]=528 MiB each ×86, Q2_K down
  [2048,4096,256]=672 MiB each ×43 — down K/N swap confirmed), attention
  4.5509 (5 Q8_0 tensors/layer), shared-experts 1.0708, token_embd 0.9863
  F16 [4096,129280], indexer-compressor 0.9075, output 0.5240 Q8_0
  [4096,129280], router 0.0927 (incl. 3× tid2eid I32 [6,129280]),
  hyperconnection 0.0631, norms 0.0016.
- Config parity anchors captured (yarn 16×/orig 65,536/freq 10,000,
  compressor rope 160,000, SWA 128, indexer 64×128 top-512, q/out lora 1024,
  output groups 8, expert scale 1.5, clamp 10.0, HC count 4, hash layers 3,
  nextn=1 metadata only — no MTP tensors in this file, consistent with the
  separate-MTP-file contract).
- Next (M1 remainder): §4.7 tensor-level TP mapping from pinned vLLM model
  source; per-kernel dtype contracts; tokenizer bootstrap pin + golden tests;
  wo_a Q8 design + VRAM delta; aligned-SoA repack spec; capacity table.

## 2026-08-17 — M1 §4.7 tensor-level TP mapping complete

- `TP-MAPPING.md`: every GGUF family → vLLM destination with constructor
  file:line citations from pinned tree 6354125a. Key rules: fused_wqa_wkv
  and both compressors are `disable_tp=True` replicated; indexer wq_b and
  weights_proj are ReplicatedLinear; wq_b/wo_a column-shard, wo_b row-shard;
  routed experts expert-shard whole-matrix (64/rank); token_embd/output
  vocab-shard; router/HC/norms/tid2eid replicated.
- Per-rank weights: **21.1893 GiB** (replicated 1.3326 + sharded 19.8567);
  +0.50 GiB/rank over WNA16-quality anchor → ≈141.9K KV token projection,
  consistent with PLAN §10 (139.1K with graph-pool delta).
- Fused-slot boundaries and per-(layer,expert,tensor) byte ranges recorded
  as class-A2 oracle requirements; all Q8_0 ne0 divisible by 32 (no partial
  blocks) verified from inventory.

## 2026-08-17 — M1 remaining gates complete; M1 PASS (narrow capacity)

- `DTYPE-CONTRACTS.md`: every family storage→runtime cast pinned. Hard facts:
  compressor state/scratch fp32; merged indexer/compressor fast path requires
  bf16 weights; HC F16→fp32 lossless; router/indexer/embedding F16→bf16 casts
  are lossy and class-B-gated (broad fp32 fallbacks exceed capacity).
- `TOKENIZER-PIN.md` + `evidence/tokenizer-parity.json`: PASS — GGUF and HF
  tokenizer alphabets identical by id (129,280 tokens, zero mismatches),
  127,741 merges identical in order, control ids 0/1/1. Explicit
  `tokenizer_mode=deepseek_v4`; runtime text/API golden tests specified.
- `WOA-DESIGN.md`: mandatory int8-g32 Marlin-diagonal path, no BF16 cache;
  naive 688 MiB/rank cache costs ~100–120K context (fatal). M2 kill gate set.
- `REPACK-SPEC.md`: byte-neutral aligned-SoA streams, content hash and class-A
  decode-identity gate, DwarfStar attribution.
- `CAPACITY.md`: exact weights + measured fixed-state anchors → 140–142K
  point estimate; M1 capacity gate **passes narrowly** but expected physical
  headroom (~0.52 GiB at 140K) is below the normal 1 GiB release guard. M5
  falsifier: fixed/runtime >22.78 GiB/rank before KV → stop or return with a
  named reclaim lever; no CPU weight-offload concealment.
- **M1 PASS mapping:** L0 class-A 100%; A2 oracle design recorded; inventory
  exact; TP table exact; dtype/tokenizer/wo_a/repack contracts recorded;
  capacity ≥140K narrowly supported. M0 fresh speed-stack trace remains the
  only deferred pre-M2 evidence item (requires a server60 GPU window).
