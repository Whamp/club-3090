# PLAN: Native GGUF tensor-parallel inference for DeepSeek-V4-Flash on 4× RTX 3090

Status: draft v3 — revised after two independent adversarial reviews (Grok 4.6
xhigh and Claude Fable 5 medium, 2026-08-17; both archived under `reviews/`,
with the second run under an explicit one-time user override of the standing
Claude-routing policy). Awaiting Will's approval before M0.

v2→v3 changes: correct IQ2_XXS layout (Fable F1 — v2's description omitted the
5-bit sub-scale and misstated sign packing); re-order the expert-kernel route
to measured-wrapped-ggml-first (Fable F2 — the pinned fork already ships
device-side indexed MoE MMVQ with DSV4 decode tuning); rebuild the §5
tolerance math on a fresh-trace requirement with the Q8_0 dense path measured,
not exempted (F3); relabel capacity as a point estimate with a stated ~17K
context per 100 MiB/rank sensitivity and un-failable-gate removal (F4/F7);
replace "F16/F32 loads as-is" with a per-kernel dtype-contract inventory (F5);
harden the DeepSWE gate to a paired, multi-seed protocol (F6); prefill
falsifier added (F7); schedule contingency wording (F8); tokenizer/config/
load-time operational risks named (F9); class-A oracle pinned to dequantized
values (F10).

## 1. Objective and decision frame

Run the **exact proven-good Antirez GGUF bytes** (IQ2_XXS routed gate/up, Q2_K
routed down, Q8_0 attention/shared/output, F16/F32/I32 control tensors)
inside a **vLLM-style tensor-parallel runtime** at vLLM-class speed, on
server60's four RTX 3090s. No requantization, no lossy conversion of the
routed experts. This is an inference-engine ASIC: one model, one artifact
family, one hardware target, zero backward compatibility.

**Why this is worth doing:** the model is the most intelligent thing that fits
our VRAM; the GGUF encoding is the only quantization of it that has held up
under our hardest gate — the post-DSML-fix final 12-task DeepSWE comparison
measured GGUF-via-llama.cpp at 6 strict solves / 96.57% partial reward versus
WNA16 requant at 0 solves / 80.62% (the earlier "bad quant" *degeneration*
attribution was withdrawn — that collapse was the DSML parser bug — but the
post-fix solve-rate gap stands); llama.cpp cannot tensor-parallel it (audited:
CUDA-TP doesn't fit 24 GiB cards, row-split breaks grouped-attention graphs);
vLLM reached 74.98 decode tok/s on inferior weights while llama.cpp is
structurally capped at ~38–39 engine (measured). Same-architecture weight
updates load through the same pinned contract (a new inventory pass, not a
rewrite). And "GGUF in tensor parallelism" is a standing community ask nobody
has shipped.

**Decision thresholds** (decode = engine tokens/s; client-wall ~12% lower,
reported alongside):

| Outcome | Minimum success | Target | Stretch |
|---|---:|---:|---:|
| Single-stream decode (engine) | ≥ 58 tok/s (~1.5× llama.cpp engine 38.4) | 70 tok/s | 75 tok/s (WNA16 stack parity) |
| Cache-busted prefill | ≥ 550 tok/s | 700 tok/s | 887 tok/s (inherited-stack parity) |
| **On-GPU unique-request context** | ≥ 140K *point estimate subject to §10 sensitivity* | 155K | 170K+ via levers |
| Prefix-reuse offload tier | present, measured hit-rate ≥ 60% on repeated-prefix workload | — | — |
| Quality | **paired** DeepSWE protocol §6: candidate ≥ baseline on the pre-registered paired statistic | quick-pack within noise of GGUF baseline | full 8-pack parity |
| Correctness | class-A dequant oracle; known-delta paths within pre-registered windows | deterministic canaries | NIAH exact recall at achieved on-GPU context |

**Context, stated honestly:** the 16 GiB host tier is an eviction/prefix-
restore tier — the promoted compose records it "is not part of the measured
230K performance path." It does **not** buy active context on top of the
on-GPU pool. 430K-class active context remains llama.cpp's exclusive
advantage (Q8_0 KV + layer split). Both services normally cannot own the four
GPUs simultaneously; during evaluation they alternate via the validated
rollback contract.

Guardrails (inherited, non-negotiable): GPU safety policy 230 W / 210–1650 MHz
untouched; one causal variable per experiment; zero-swap final states;
verified rollback to the canonical llama.cpp service; every claim measured.

## 2. Evidence inventory — what exists, correctly labeled

Three evidence grades: **proven-here** (measured on our stack),
**proven-adjacent** (real measurement, different kernel/stack — supports
feasibility, not performance), **unmeasured** (assumption to retire).

| Component | Grade | Evidence |
|---|---|---|
| IQ2_XXS/Q2_K dequant semantics **in the pinned source** | proven-here (source); **the plan's transcription of them is verified only by the L0 oracle** | pinned Whamp/llama.cpp `0379cf4bf`; v2's own layout prose was wrong (reviews/fable F1) — FORMAT-CONTRACT is generated from source and gated by L0, never trusted from prose |
| These formats sustaining GEMM-class throughput | proven-adjacent | llama.cpp fused MMQ prefill 1,056 tok/s — different kernel than either planned route |
| **Device-side indexed MoE MMVQ machinery in the pinned fork** | proven-here (existence; perf at our shapes = M2a) | `mmid.cu` id compaction, `mul_mat_vec_q_moe_launch`, gate/bias fusion args, `DSV4_MMVQ_SMALLK` tuned for routed IQ2_XXS at n_embd 4096 |
| TP-sharded indexed grouped-MoE skeleton with in-mainloop unpack | proven-here for int-unpack W2/W4, unmeasured for codebook fragments | Humming extensions `e5a8452c7`, `dd2d1fd6`; 7-case SM86 oracle |
| Whole non-MoE stack at speed on FP8 weights | proven-here | DSML fix, SwiGLU fix, FlashMLA, hierarchical all-reduce: 74.98 tok/s |
| Same stack on Q8_0 weights (dense + `wo_a`) | **unmeasured — largest unmeasured kernel share (Marlin ~23% of the pre-FlashMLA trace)** | measured at M2 before any go/no-go |
| Correctness-oracle methodology | proven-here | WNA16 oracle ladder |
| KV-offload tier inertness | proven-here as an unused reservation | eviction-pressure behavior unmeasured — hence the separate hit-rate line |
| GGUF container parsing precedent | weak | GGUF support is out-of-tree/experimental; our loader is new code |

**Scope reduction:** only IQ2_XXS and Q2_K need codebook paths (native or
wrapped). Q8_0 becomes int8 group-32 via a last-bits-lossy repack (§4).
F16/F32/I32 control tensors pass through **subject to per-kernel dtype
contracts** (§4.5) — the consuming kernels assert/cast (e.g., compressor
fp32, indexer fused-quant, merged-GEMM fp32 hand-off), so "loads as-is" is
decided per tensor at M1, not assumed.

## 3. The artifact contract (pinned; M1 verifies byte-level)

GGUF: `antirez/deepseek-v4-gguf`, `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-
SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, blob SHA-256 `ca22ae2f…b1c0`,
86,720,111,488 bytes, verified on server60. MTP is a separate 3.6 GiB file,
not in this blob — no MTP lever.

Tensor classes (M1 produces the authoritative per-tensor inventory):

| Class | Format | Notes |
|---|---|---|
| Routed gate/up (`w1`/`w3`) | IQ2_XXS | 66 B/256 weights = 2.0625 bpw |
| Routed down (`w2`) | Q2_K | 84 B/256 = 2.625 bpw, dual-scale affine |
| Attention projections, shared experts, `output` | Q8_0 | includes `wo_a` |
| `token_embd`, router, indexer, compressor, HC | F16 | replicated families; dtype contracts §4.5 |
| Norms, sinks, biases | F32 | no downcast |
| `ffn_gate_tid2eid` (early layers) | I32 | pass-through |

**IQ2_XXS layout, corrected (source: `vecdotq.cuh:985-1014`):** block = fp16
`d` + 32 uint16. Per 32 weights (8 bytes): two uint16 = **4 grid-index
bytes** (each indexes the 256-entry × 8-weight LUT, 2 KiB); the following
uint32 = **four 7-bit sign fields** (`unpack_ksigns(aux32 >> 7*k)`) **plus a
5-bit integer sub-scale** `ls = aux32 >> 27 | 1`, applied with integer
truncation: `sumi = sumi * ls / 8`, then `d * bq8_1.ds * sumi`. Q2_K =
fp16 `d`+`dmin`, 16 sub-blocks of 4-bit scale/min + 2-bit quants; Q8_0 =
fp16 `d` + 32×int8. Reference vec_dot: `vecdotq.cuh:985` (IQ2_XXS),
`:814`/`:393` (Q2_K). CPU `dequantize_row_*` is the golden oracle — and the
integer-truncation semantics above are exactly why class A covers
**dequantized values in fp32**, while fused outputs live in class B (§6).
M1 also pins TP-shard geometry per family (experts/attention/vocab shard;
indexer, compressor, HC, router replicate — residency audit `965aebbfb502`),
and the tokenizer/config contract (§4.6).

## 4. Architecture: keep / adapt / new

**Base pin:** branch `incubate/gguf-tp-sm86` in Whamp/vllm from the
speed-stack tip `b7766cfe` (tree `6354125a`) — DSML fix + SwiGLU +
Marlin-wo_a + FlashMLA + hierarchical all-reduce + KV-offload repair,
measured 74.98/887. club-3090 `feat/gguf-tp-engine` (this worktree) owns
deployment, evidence, plans.

**Keep (proven on the pinned base):** vLLM scheduler + continuous batching;
TP=4 launcher; FlashMLA sparse decode; hierarchical all-reduce; DSML parser
fix; SwiGLU semantics; fp8_ds_mla KV + offload tier; CUDA-graph capture path;
tokenizer/chat path **subject to the §4.6 parity audit**.

**Adapt / new:**
1. **GGUF-native loader**: mmap, verify blob SHA-256 (see §4.6 on load-time
   hashing cost), GGUF tensor-name → vLLM module mapping, per-family TP shard
   rules, per-rank packed views, checksum fail-closed against the M1
   inventory. IQ2_XXS/Q2_K/F16/F32/I32 bytes are never re-encoded.
2. **Q8_0 → int8 group-32 repack (documented lossy-in-last-bits):** exact
   int8 codes; fp16 block scale → CT scale (kept fp16 where the consuming
   kernel allows, else bf16-rounded), Marlin tile-packing, uint8b128 offset.
   Tolerance oracle vs Q8_0 dequant+GEMM; excluded from the bit-exact ladder.
   **Its dense GEMV/GEMM perf is measured at M2** — it is the largest
   unmeasured kernel share.
3. **`wo_a` Q8_0 output projection:** FP8 Marlin-diagonal doesn't apply to
   Q8_0; no-cache fallback measured 34.01 tok/s (fatal); BF16 cache 688
   MiB/rank (more than the whole GGUF tax). M1 scopes a Q8 grouped-Marlin (or
   equivalent) diagonal path with VRAM delta and measured decode number.
   **Kill:** if the only working options are BF16 cache or ~34 tok/s dequant,
   stop at M1/M4.
4. **Routed experts — two routes, measured order (Fable F2):**
   - **Route A (measured first, M2a): wrap the pinned fork's existing MoE
     MMVQ/MMQ machinery as vLLM expert ops.** The fork already ships
     device-side expert-id compaction (`mmid.cu`), a dedicated
     `mul_mat_vec_q_moe_launch` path with gate/bias fusion, and
     `DSV4_MMVQ_SMALLK` decode tuning for the exact routed IQ2_XXS
     n_embd-4096 shape. Remaining work: id-format/TP-offset plumbing and vLLM
     op integration. Microbenching this is nearly free and gives Humming a
     measured bar to beat.
   - **Route B (escalation, M2b): Humming indexed-expert mainloop fragments**
     for IQ2_XXS/Q2_K fused at fragment-load. Required only if Route A misses
     the mapped budget; carries the codebook-into-MMA risk.
   Both routes must be graph-capturable (eager-only falls toward the measured
   5.5 tok/s regime).
5. **Per-kernel dtype contracts for replicated families (Fable F5):** M1
   inventories the dtype/layout contract of every kernel consuming a
   replicated family (compressor fp32 assert; indexer fused-quant transforms;
   merged-GEMM fp32 hand-off; activation dtype bf16). Any transform or cast
   becomes a documented conversion with a class-B window and a capacity-table
   line. In particular the **router** cast policy is explicit: F16 → bf16 is
   lossy in a top-6 tie-break-sensitive place; if the router kernel accepts
   fp32/fp16 natively, prefer it.
6. **Tokenizer/config source of truth (Fable F9):** the baseline ran through
   the GGUF-embedded tokenizer, llama.cpp chat template, and sampler chain;
   the WNA16 stack uses HF tokenizer.json + vLLM template/samplers. M1 diffs
   these (byte-merge edges, added tokens, template whitespace, sampling
   order) and pins the authoritative config source (GGUF KV metadata vs HF
   config: RoPE theta/YaRN, compress ratios, SWA window). Class-B/D prompts
   are pinned to **token IDs**, not text.

**Quant-method config**: `gguf_dsv4` expressing experts = GGUF-native, dense
Q8_0 = CT int8-g32, control = per-kernel-contract passthrough.

**Retained:** the compressed-tensors expert path stays for the same-tree WNA16
A/B that attributes any performance miss to the new route rather than to
FlashMLA/graphs/AR regressions.

## 5. Performance reasoning

The v2 "3× headroom" claim is withdrawn. Corrected frame:

- The 15% expert share came from the **pre-FlashMLA** trace; after hier-AR +
  FlashMLA the surviving kernels renormalize upward (~18% ≈ 2.4 ms of the
  13.3 ms/token), and the **Q8_0 dense replacement (Marlin ~23% share,
  larger than experts) was silently assumed cost-free** despite being graded
  unmeasured. If the dense path runs 30% slower, ~0.9 ms of budget is gone
  before experts spend anything. Realistic expert-kernel tolerance to hold
  58 tok/s: **~2.2–2.6× slower than W2 Humming**, contingent on the dense
  path measuring near FP8-Marlin parity.
- **M2 therefore requires, all measured on a 3090:** (a) a **fresh nsys
  decode trace of the running 74.98 stack** (it exists; no gate arithmetic on
  stale traces); (b) expert-kernel time via Route A microbench; (c) dense
  Q8_0-g32 Marlin GEMV at serving shapes; (d) `wo_a` path time; (e) Route B
  fragment time if Route A misses. The 58-floor projection is computed from
  (a)+(b)+(c)+(d) — measured components only.
- **Prefill falsifier (added):** dense Q8_0 GEMM at prefill M≈256 is
  microbenched at M2/M4; if its mapped contribution projects < 550 tok/s,
  that is a named failure before bring-up, not a surprise at M7.
- Overall decode estimate band: **55–75 tok/s**, estimate until M7.

## 6. Correctness doctrine

- **A. Bit-exact (no tolerance):** IQ2_XXS/Q2_K **dequantized weight values
  in fp32** vs llama.cpp CPU `dequantize_row_*`, random + adversarial corpora
  (sign patterns, extreme scales, LUT boundary indices, sub-scale extremes).
  Fused fragment/kernel *outputs* are class B by construction (integer
  truncation semantics like `sumi*ls/8` are not bit-equal to CPU
  float-dequant-then-FMA) — explicitly assigned, never re-classified until
  green.
- **B. Known-delta (pre-registered windows):** Q8_0-repack GEMM vs dequant+
  GEMM (atol/rtol per shape); fused expert kernel outputs vs reference GEMM;
  full-model forward vs llama.cpp on fixed **token-ID-pinned** prompt sets
  (KL bound stated before first run); fp8_ds_mla KV / FlashMLA / hier-AR /
  replicated-family casts (§4.5) each with their own window.
- **C. Determinism:** CUDA-graph replay vs eager equality; AR rank-order
  consistency.
- **D. End-to-end:** deterministic canaries; tool round-trip and post-tool
  continuation (DSML lesson); NIAH exact recall at achieved on-GPU context;
  **DeepSWE as a paired, multi-seed protocol**: same task set run on both
  engines, ≥3 seeds or seed-fixed where the harness allows, decision
  pre-registered on the paired statistic (discordant tasks + mean-partial
  delta), acknowledging that an unpaired n=12 binary threshold has ~19%
  false-fail / ~42% false-pass error (reviews/fable F6). A single SuperJSON
  run is a smoke signal, not the gate. Divergence = integration bug or a
  class-B window that is wrong → component bisect, never hand-wave.

Ladder L0→L6, adversarial review loops (1 implementer + 2 reviewers on diff +
format contract), checksums on every tensor view, oracle failures batched as
work queue — unchanged.

## 7. Execution methodology

Unchanged: FORMAT-CONTRACT.md generated from source before code (now with the
L0 oracle as its acceptance test — prose proven wrong once already);
trial-fragment-first; loader cutover on its own branch; process fixes over
hand fixes; worktree/commit discipline; one causal variable; Nsight
attribution before kernel tuning; unprofiled end-to-end numbers for claims.

**Hardware:** all kernel gates on a server60 RTX 3090 (co-resident
microbenching proven under the safety cap). **No rental compute planned.**

## 8. Milestones, gates, kill criteria

| # | Deliverable | Gate | Kill / pivot | Est. |
|---|---|---|---|---|
| M0 | Worktrees, base pin `b7766cfe`, fresh nsys trace of the 74.98 stack, baseline re-anchor | pins + trace recorded | — | 1 d |
| M1 | `FORMAT-CONTRACT.md` from source; per-tensor inventory; per-kernel dtype contracts (§4.5); tokenizer/config diff (§4.6); `wo_a` path scoped with VRAM delta; **capacity table with every delta sized in MiB → context tokens** (§10) | L0 class-A oracle 100% incl. adversarial; inventory matches blob; capacity table shows ≥140K or levers whose summed size closes the gap in tokens | inventory mismatch → re-scope; `wo_a` no viable path → **stop**; capacity gap unclosable → re-decide scope with Will | 2–4 d |
| M2a | **Route A first: wrapped-ggml MoE MMVQ/MMQ as vLLM ops**, microbenched at serving shapes (decode M=1–8, prefill M≈256 incl. dense Q8_0 GEMM) | projection from fresh-trace mix + all measured kernel times ≥ 58 decode and ≥ 550 prefill | both Route A projection < floor → escalate to M2b with a measured bar | 3–5 d |
| M2b | Humming IQ2_XXS fragment (escalation only) | beats Route A's measured bar; graph-capturable | misses Route A's bar after 2 tuning iterations → take Route A if ≥ floor, else stop | 1–2 wk |
| M3 | Q2_K on the winning route (affine dual-scale; register pressure watch) | same | same | 1 wk |
| M4 | GGUF loader + Q8_0 repack + `gguf_dsv4` config + `wo_a` path; CPU tests; checksum fail-closed; **calendar kill: 10 working days** | full-tensor mapping test; byte-identity assert on packed IQ2_XXS/Q2_K views | calendar breach → descope review | 1–2 wk |
| M5 | server60 TP=4 bring-up (authorized window; validated rollback) | class-A/B full-kernel oracle on-GPU; NCU dispatch; readiness | repeated OOM/instability → capacity re-plan | 0.5–1 wk |
| M6 | Per-layer vs llama.cpp (class B, token-ID-pinned); canaries + NIAH at achieved context | pre-registered windows pass | unexplained divergence → bisect | 3–5 d |
| M7 | Matched perf campaign; same-tree WNA16 A/B attribution | ≥58 engine decode, ≥550 prefill, ≥140K on-GPU, zero swap | miss → keep llama.cpp canonical; publish | 3–5 d |
| M8 | Quality: quick pack + **paired multi-seed DeepSWE** vs GGUF baseline | §6 paired statistic passes its pre-registered rule | divergence → component bisect | 2–4 d |
| M9 | Promotion package; open-source decision | Will's approval; healthy final service | — | 2–3 d |

Effort envelope: **6–9 weeks if gates pass first-try**; M2b/M3 iterations,
M6/M8 bisects, and a `wo_a` redesign loop are the anticipated contingency
sources — kills bound the downside, not the calendar.

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `wo_a` Q8_0 path slow or fat | medium-high | fatal if unsolved | M1 scope + kill; Q8 grouped-Marlin hypothesis |
| Dense Q8_0 path slower than FP8 Marlin (largest unmeasured share) | medium | eats the 58 budget before experts | measured at M2a; feeds the go/no-go directly |
| Expert route too slow at decode M≤8 | medium | misses 58 | Route A measured first (cheap); Route B escalation; ~2.2–2.6× tolerance (not 3×) |
| Capacity: 140K is a **point estimate** — every unmodeled 100 MiB/rank costs ~17K context; replicated-family delta unquantified at plan time | certain (the sensitivity, not the value) | context floor | M1 sizes every delta in MiB → tokens; no "named lever" escape |
| Replicated-family dtype conversions (router!) degrade quality silently | medium | class-B escape | §4.5 per-kernel inventory; router prefers fp32/fp16 native |
| Tokenizer/chat-template/config divergence contaminates all class-B/D comparisons | medium | false attribution | §4.6 M1 audit; token-ID-pinned prompts |
| DeepSWE gate statistical noise | medium | false pass/fail | paired multi-seed protocol §6 |
| Q2_K dual-scale register pressure (Route B only) | medium | Route B slow | separate accumulators; `_impl_mmq` study |
| Grid LUT smem pressure (Route B only) | low-medium | pipeline stages shrink | constant memory + per-tile cache; measure |
| Effort overrun | medium | opportunity cost | M1/M2/M4 kills; calendar caps |
| Load-time hash of 80.76 GiB adds minutes per restart; host-RAM churn (page cache + 16 GiB pinned tier + repack scratch) | low-medium | operational | size at M1; consider trust-once-then-stat caching (design at M4) |

## 10. Capacity plan

Method: M1's table — per-rank registered weights by family (sharded vs
replicated, post-transform), graph pool (~0.19 GiB measured), Humming/NVRTC
workspace, loader/repack scratch, Marlin tile padding (`marlin_padded_nk`),
KV pool, headroom. Anchors: 78.74 GiB WNA16 artifact reached 230,144 ctx with
1.28 GiB available KV; KV density ≈ 5.8 KiB/token/rank → **sensitivity:
~17–18K context per 100 MiB/rank**. The sharded GGUF tax alone (~0.5 GiB/rank)
gives ~(1.28−0.5) GiB ≈ **137–140K — a point estimate with the
replicated-family delta at zero**; precedent (indexer 191→767 MiB/rank after
transforms on WNA16) says that delta is not zero and must be measured, not
assumed small. The M1 gate requires the completed table to show ≥140K or
levers whose summed MiB close the gap explicitly. The 16 GiB host tier ships
as prefix-reuse only, gated by its hit-rate line. 430K active stays
llama.cpp's exclusive advantage.

## 11. What "done" means

server60 serves `deepseek-v4-flash-0731-gguf-tp` from the pinned GGUF blob at
≥58/70 engine decode, ≥550/700 prefill, ≥140K on-GPU unique context (per the
M1 capacity table), zero swap, safety policy intact, paired-protocol DeepSWE
quality within its pre-registered window, validated rollback, everything
committed and pushed, evidence bundled, upstreaming decision recorded. The
llama.cpp service remains canonical until M8 passes.

## 12. Immediate next actions (on approval)

1. M0: worktrees, base pin `b7766cfe`, **fresh nsys trace of the 74.98
   stack**, baseline re-anchor.
2. M1: FORMAT-CONTRACT from source + L0 oracle + inventory + dtype contracts
   + tokenizer/config diff + `wo_a` scope + capacity table.
3. M2a: wrapped-ggml Route A microbench — cheapest first hard number.
4. Standing question for Will at M1 exit: accept the measured on-GPU context
   floor as this service's contract (llama.cpp retained for 430K-class
   needs), or require named-and-sized levers first?
