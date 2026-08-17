# PLAN: Native GGUF tensor-parallel inference for DeepSeek-V4-Flash on 4× RTX 3090

Status: draft v1 — awaiting Will's review before M0.

## 1. Objective and decision frame

Run the **exact proven-good Antirez GGUF bytes** (IQ2_XXS routed gate/up, Q2_K
routed down, Q8_0 elsewhere) inside a **vLLM-style tensor-parallel runtime**
at vLLM-class speed, on server60's four RTX 3090s. No requantization, no
lossy conversion, no format translation of the routed experts. This is an
inference-engine ASIC: one model, one artifact family, one hardware target,
zero backward compatibility.

**Why this is worth doing:** the model is the most intelligent thing that fits
our VRAM; the GGUF encoding is the only quantization of it that has *not*
measurably damaged intelligence (DeepSWE: 6 strict solves / 96.6% partial vs
WNA16 requant 0 solves / 80.6%); llama.cpp cannot tensor-parallel it (audited:
CUDA-TP doesn't fit 24 GiB cards, row-split breaks grouped-attention graphs);
vLLM reached 74.98 decode tok/s on inferior weights while llama.cpp is
structurally capped at ~38–39 (serial pipeline, measured). Future RL-refreshed
weights of the same architecture load day one. And "GGUF in tensor
parallelism" is a standing community ask nobody has shipped.

**Decision thresholds:**

| Outcome | Minimum success | Target | Stretch |
|---|---:|---:|---:|
| Single-stream decode (engine) | ≥ 58 tok/s (1.5× llama.cpp 38.4) | 70 tok/s | 75 tok/s (WNA16 parity) |
| Cache-busted prefill | ≥ 550 tok/s | 700 tok/s | 900 tok/s |
| Context on-GPU / with offload tier | ≥ 150K / 230K | 180K / 260K | 230K / 430K |
| Quality | DeepSWE parity with GGUF llama.cpp (same weights → gate catches integration bugs only) | quick-pack within noise of GGUF baseline | full 8-pack parity |
| Correctness | Bit-exact dequant vs llama.cpp reference; per-layer output tolerance gates | deterministic canaries | NIAH exact recall ≥ 200K |

Guardrails (inherited, non-negotiable): GPU safety policy 230 W / 210–1650 MHz
untouched; one causal variable per experiment; zero-swap final states;
verified rollback to the canonical llama.cpp service; every claim measured.

## 2. Evidence inventory — what already exists and is proven

Nothing below is speculative; each row cites where it was proven.

| Component needed | Proven source | Evidence |
|---|---|---|
| IQ2_XXS/Q2_K exact dequant semantics | pinned Whamp/llama.cpp `0379cf4bf` source, read line-by-line during the MMVQ campaign | decode campaign M1–M4; archived kernels in club-3090 |
| These formats feeding GEMM-class throughput on our GPUs | the fork's own prefill: MMQ dequant-inline GEMM at 1,056 tok/s through 263K context | decode campaign attribution |
| TP-sharded indexed grouped-MoE skeleton with in-mainloop weight unpack | Humming indexed experts (humming-kernels 0.1.10 @ `4351af3a`), which we extended 3× (W2/W3 bridge, mixed w13/w2, mixed group sizes) | commits `e5a8452c7`, `dd2d1fd6`, 7-oracle SM86 runs |
| Whole non-MoE stack at speed | DSML fix, SwiGLU fix, FlashMLA decode, hierarchical all-reduce, KV-offload | 74.98 tok/s composite arm, Whamp/vllm PRs #1–2 |
| Correctness-oracle methodology | WNA16 oracle ladder: CPU bit-exact → CUDA deterministic oracle → NCU dispatch proof → canaries → DeepSWE | every campaign since `f4d05732a` |
| Capacity levers | KV-offload tier measured performance-inert below ~275K; Marlin-wo_a recovered 688 MiB/rank; runtime-bounded RoPE | capacity campaign (`b4a570ac`) |
| vLLM GGUF container parsing precedent | vLLM's existing GGUF loader expands to BF16 — wrong for us, but its metadata/tensor-walk code shape is reusable | upstream `vllm/model_executor/layers/quantization/gguf.py` |

**Scope reduction that makes this tractable:** only *two* codebook formats
(IQ2_XXS, Q2_K) need new kernel fragments. Q8_0 is symmetric block-32 int8 —
numerically identically expressible as compressed-tensors int8 group-32
(pure repack, zero quality change, existing W8A16 kernels). F16/F32 control
tensors load as-is. The GGUF's non-routed storage (8.2 GiB) is within 1% of
the WNA16 artifact's preserved non-routed storage (8.24 GiB), so all
non-expert memory behavior carries over.

## 3. The artifact contract (pinned; M1 verifies byte-level)

GGUF: `antirez/deepseek-v4-gguf`, `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-
SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, blob SHA-256 `ca22ae2f…b1c0`,
86,720,111,488 bytes, verified on server60.

Block layouts, from pinned source `ggml/src/ggml-common.h` + `vecdotq.cuh`:

| Format | Block | Bytes/block | bpw | Structure |
|---|---|---:|---:|---|
| IQ2_XXS | 256 weights | 66 | 2.0625 | fp16 `d` + 32× uint16; per uint16: low byte = index into 256-entry grid LUT (each entry = 8 int8 weights, LUT = 2 KiB constant), high byte = 8 sign bits |
| Q2_K | 256 weights | 84 | 2.625 | super-block: fp16 `d` + `dmin`, 16 sub-blocks × (4-bit scale + 4-bit min), 2-bit quants (64 B) — **dual-scale structure** |
| Q8_0 | 32 weights | 34 | 8.5 | fp16 `d` + 32× int8 |

Reference vec_dot implementations: `vecdotq.cuh:985` (IQ2_XXS),
`:814` (Q2_K, with `_impl_mmq` integer variant at `:393` worth studying for
the GEMM fragment). CPU dequant in `ggml-quants.c` is the golden oracle.
M1 must also pin: per-tensor format inventory for all tensors, MTP presence
or absence (if MTP tensors exist and are droppable, ~2 GiB recovered), and
the exact TP-shard geometry for each tensor class.

## 4. Architecture: keep / adapt / delete

**Keep (proven, untouched):** vLLM scheduler + continuous batching; TP=4
launcher; FlashMLA sparse decode (AppMana, ours); hierarchical all-reduce;
DSML parser fix; SwiGLU semantics; fp8_ds_mla KV cache + KV-offload tier;
CUDA-graph capture path; tokenizer/chat path; Marlin-wo_a.

**Adapt (the project):**
1. **Weight source**: new GGUF-native loader — mmap, verify blob SHA-256,
   map GGUF tensor names → vLLM DeepSeek-V4 modules (we hold the official
   mapping from the frontier work), TP-shard per existing rules (experts on
   intermediate dim, vocab-parallel embed/head), emit per-rank packed views.
   Packed bytes are **never** re-encoded for IQ2_XXS/Q2_K.
2. **Q8_0 tensors**: one-time repack to compressed-tensors int8 group-32
   symmetric (fp16 scales) at load; existing W8A16/Marlin kernels execute.
3. **Routed experts (the core)**: extend the Humming indexed-expert mainloop
   with two new weight-format fragments — IQ2_XXS and Q2_K dequant fused at
   the fragment-load stage, feeding the same mma pipeline. New schema entries
   + Sm86 heuristics + JIT dispatch. No other format, model, or fallback.
4. **Quant-method config**: a `gguf_dsv4` quant method expressing "experts =
   GGUF-native IQ2_XXS/Q2_K, dense = CT int8-g32, control = BF16" — replaces
   the compressed-tensors metadata path for this model only.

**Delete / never build:** generic GGUF support; other GGUF quants; multi-model
abstraction; AWQ/GPTQ/CT schemas for experts; CPU fallbacks; config-probing
heuristics beyond exact-match fail-closed; any backward-compat shim. If the
tensor inventory doesn't match the pinned contract exactly, load fails.

## 5. Causal budget and performance estimate

Weights per decode token (TP=4, per rank): routed experts ≈ 72.56 GiB ×
6/256 activated ≈ 1.70 GiB, + shared/attention Q8_0 ≈ 0.5 GiB, ÷ 4 ranks ≈
**0.55 GiB/token/rank**. At WNA16's measured kernel efficiency this demand
produced 13.3 ms/token (74.98 tok/s). GGUF routed bytes are +6% vs WNA16's,
in the same bpw class; IQ2_XXS/Q2_K fragments are instruction-heavier than
int-unpack (grid LUT + sign chain) but prefill MMQ proves these encodings
sustain GEMM-class throughput at larger tiles.

Estimate band: **55–75 tok/s decode**, labeled estimate until M7 measures it.
Prefill inherits the vLLM chunked-prefill stack with our fragments in the
batch-shape regime MMQ already wins: estimate 700–950 tok/s.
Falsifier: M2/M3 microbench — if the fused fragment at serving shapes
(delivery: R≤8 per expert at decode, batched M at prefill) cannot reach ≥60%
of the W2-WNA16 kernel's throughput after two tuning iterations, the 58-tok/s
floor is at risk and we re-decide (hybrid dispatch or stop).

Lose-conditions to watch: small-batch decode (M=1–8) is where codebook
instruction overhead bites hardest (MMVQ campaign measured i-quant matvec at
43–50% of Q8_0 bandwidth); prefill large-M is where we expect to *beat*
WNA16 (MMQ already wins there in llama.cpp).

## 6. Correctness doctrine — the Bun test-suite analog

The Bun rewrite survived on: language-independent million-assertion suite,
adversarial review, and "fix the process, not the code." Our analog:

1. **Golden oracle = llama.cpp itself.** CPU `dequantize_row_*` gives exact
   reference values for every block (deterministic, integer-exact scales,
   fp-only in the final accumulate). Our fragment must reproduce dequant
   values bit-exactly and dot products to documented fp-reassociation
   tolerance. This is *stronger* than the WNA16 oracle position: there is no
   quantization step to distrust — any divergence is our bug.
2. **Oracle ladder (all pre-built, re-run per milestone):**
   L0 CPU bit-exact dequant vs llama.cpp (random + adversarial blocks: sign
   patterns, extreme scales, LUT boundary indices);
   L1 standalone CUDA fragment test: fragment-load → mma output vs CPU
   reference GEMM, per format × {decode shapes, prefill shapes};
   L2 full-kernel deterministic oracle on real tensors (the
   `HummingIndexedExperts.apply` pattern from `f4d05732a`, extended with
   GGUF schemas — the 7-case mixed-group suite is the template);
   L3 NCU dispatch proof: sm_86 cubin, expected symbol, real serving shapes;
   L4 per-layer output comparison vs llama.cpp full forward on a fixed prompt
   set (fp tolerance, KL bound on next-token logits);
   L5 serving canaries: deterministic generation, tool round-trip, post-tool
   continuation (the DSML lesson), NIAH exact recall at 100K/200K;
   L6 DeepSWE single-worker SuperJSON gate vs the GGUF-llama.cpp baseline —
   **parity required**; divergence means an integration bug, never "quant
   noise," because the weights are identical.
3. **Adversarial review loops** (Bun pattern): implementer agent + 2 reviewer
   agents in separate contexts, reviewers see only the diff + format contract
   and are told to assume it's wrong. Mandatory for every kernel fragment and
   every loader mapping. Review attribution recorded in commit subjects.
4. **Compiler/oracle errors as the work queue:** L0/L1 failures get machine-
   grouped and batched to fixer loops exactly like Bun's `errors.txt` crates.
5. **Checksums everywhere:** every tensor view carries source-offset +
   SHA-256; load fails closed on any mismatch. The A4 campaign showed why —
   performance-looking numbers from broken kernels are seductive.

## 7. Execution methodology

- **Prep before code (Bun's PORTING.md analog):** M1 produces
  `FORMAT-CONTRACT.md` — exact block layouts, lane mappings, TP shard rules,
  tensor-name mapping, tolerances, and the reference-source line citations.
  Every implementer and reviewer agent works from this file, not from memory.
- **Trial-then-scale:** first fragment = IQ2_XXS only, reviewed to death,
  oracle-passed, microbenched. Only then Q2_K, then the loader.
- **Everything-at-once over incremental** for the loader swap (Bun's finding:
  incremental rewrites accumulate temporary seams): one cutover branch, the
  old CT-expert path deleted in the same series, no dual-path period beyond
  the milestone gates.
- **Process fixes over hand fixes:** when an oracle fails, the fix loop also
  patches the contract doc or generator that allowed the bug.
- **Worktree/commit discipline:** new branch `incubate/gguf-tp-sm86` in
  Whamp/vllm from merged PR #2 base (`28db4816`…tree `67064365`); club-3090
  `feat/gguf-tp-engine` (this worktree) owns deployment, evidence, plans.
  Agents commit per-file; no stash/reset; no shared-worktree races.
- **One causal variable per perf experiment; matched A/B; Nsight attribution
  before kernel tuning; unprofiled end-to-end numbers for claims.**

## 8. Milestones, gates, kill criteria

| # | Deliverable | Gate to pass | Kill / pivot criterion | Est. |
|---|---|---|---|---|
| M0 | Worktrees, pinned GGUF copy, humming-kernels source extracted & pinned, baselines re-anchored (llama.cpp decode/prefill; WNA16 74.98 reference) | plan approved by Will; environment pins recorded | — | 0.5 d |
| M1 | `FORMAT-CONTRACT.md` + L0 CPU oracle (bit-exact dequant, adversarial corpus) + tensor inventory + MTP audit + TP shard plan | L0 100% pass incl. adversarial blocks; inventory matches expected counts exactly | inventory mismatch vs pinned GGUF → re-scope before any CUDA | 2–4 d |
| M2 | IQ2_XXS fragment in Humming mainloop; L1 pass; A100 microbench at decode+prefill shapes | L1 bit-exact/tolerance; ≥60% of W2-kernel throughput at serving shapes | <60% after 2 tuning iterations → hybrid-dispatch pivot decision (w13-only fused, w2 GEMV) or stop | 1–2 wk |
| M3 | Q2_K fragment same gates (watch dual-scale register pressure) | same | same | 1 wk |
| M4 | GGUF loader + Q8_0 repack + `gguf_dsv4` quant config + TP shard; CPU tests; checksum fail-closed | full-tensor mapping test; no requant of IQ2_XXS/Q2_K bytes (assert byte-identity of packed views) | — | 1 wk |
| M5 | server60 TP=4 bring-up (authorized window; canonical llama.cpp down, validated rollback) | L2 full-kernel oracle on-GPU; L3 NCU dispatch; model loads all tensors; readiness | repeated OOM/instability → re-plan capacity | 0.5–1 wk |
| M6 | L4 per-layer vs llama.cpp; L5 canaries + NIAH | tolerance gates pass; canaries + recall pass | unexplained divergence → bisect by layer (we have per-layer harness) | 3–5 d |
| M7 | Matched perf campaign: decode/prefill/context ladders, one variable at a time | ≥58 decode, ≥550 prefill, ≥150K on-GPU ctx, zero swap | miss → keep llama.cpp canonical; publish findings | 3–5 d |
| M8 | Quality: quick pack + DeepSWE SuperJSON single-worker vs GGUF baseline | parity within baseline noise; tool/post-tool behavior normal | quality divergence = integration bug hunt (weights identical) | 2–4 d |
| M9 | Promotion package: compose, image, INTERNALS/report, evidence bundle, rollback drill; open-source decision (Whamp/vllm PR vs standalone repo) | Will's promotion approval; healthy final service | — | 2–3 d |

Effort envelope: **6–9 focused weeks** at campaign intensity; M2/M3 are the
only research-grade risk; M4–M9 are machinery we have run three times.

Rental plan: one on-demand A100 for M2/M3 microbench + L1/L2 (~$10–25 total,
watchdog-guarded, established pattern); server60 for M5+ (free; user-
authorized maintenance windows; llama.cpp restorable via validated rollback).

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Codebook fragment too slow at decode M≤8 (instruction-bound; MMVQ measured 43–50% of Q8_0 BW) | medium | misses 58 floor | fragment tuned for M≤8 tiles; hybrid dispatch fallback (w2 via dequant-GEMV); prefill win may justify alone |
| Q2_K dual-scale register pressure spills | medium | Q2_K slow | separate fp accumulators; study `_impl_mmq` integer variant; hybrid dispatch |
| **VRAM: GGUF is +2 GiB vs the 78.74 GiB artifact that hit 230K with 93 MiB spare** | certain | on-GPU ctx ~150–160K without levers | KV-offload tier (measured inert <275K) as default; MTP omission if tensors exist; Marlin-wo_a + bounded-RoPE already in base |
| Grid LUT (2 KiB) smem pressure in mainloop | low-medium | pipeline stages shrink | LUT in constant memory + per-tile smem cache; measure stage count at M2 |
| Subtle numeric divergence → silent quality loss | low (oracle ladder) | catastrophic if missed | L0 bit-exact + L4 per-layer + L6 DeepSWE parity; checksums; adversarial review |
| Integration drift vs Whamp/vllm base | low | rebase pain | pin base tree `67064365`; this project is the tip, no merges mid-project |
| Effort overrun | medium | opportunity cost | kill gates at M2/M3 cap kernel investment; everything after is proven machinery |

## 10. Capacity plan (the honest arithmetic)

Measured anchors: the 78.74 GiB quality artifact reached 230,144 ctx with a
1.26 GiB KV pool and 93 MiB headroom per GPU. The GGUF is +2.0 GiB total
(+0.5 GiB/rank) → on-GPU KV pool ≈ 0.85 GiB → **~150–160K context** with all
existing levers. Reaching 230K+: engage the 16 GiB KV-offload tier (measured
performance-inert below ~275K tokens, already the serving default pattern).
MTP audit (M1) may recover ~2 GiB if those tensors exist and are excluded —
we never use MTP. Stretch 430K depends on offload behavior at depth; measure,
don't promise.

## 11. What "done" means

server60 serves `deepseek-v4-flash-0731-gguf-tp` from the pinned GGUF blob at
≥58/70 decode, ≥550/700 prefill, ≥150K on-GPU (230K+ with offload), zero
swap, safety policy intact, DeepSWE-parity quality, validated rollback to
canonical llama.cpp, everything committed and pushed, evidence bundled, and a
decision recorded on upstreaming. The llama.cpp service remains canonical
until M8 passes; promotion is a separate explicit step.

## 12. Immediate next actions (on approval)

1. M0 half-day setup (worktrees, pins, baseline re-anchor).
2. M1 format contract + CPU oracle — starts immediately after.
3. Standing question for Will at M1 exit: on-GPU context floor — accept
   ~155K default-with-offload, or require additional levers first?
