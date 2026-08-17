# PLAN: Native GGUF tensor-parallel inference for DeepSeek-V4-Flash on 4× RTX 3090

Status: draft v4 — revised after three independent adversarial reviews (Grok
4.6 xhigh; Claude Fable 5 medium under a one-time user override; GPT-5.6 Sol
xhigh via pi one-shot — all 2026-08-17, archived under `reviews/`). The third
review returned **STOP**, and its findings are incorporated here; M0 stays
blocked until Will approves this revision. All three reviewers were
instructed not to read prior reviews.

v3→v4 changes: Route A rewritten as an honest unfused decode operation
sequence with its mandatory bf16→F32→Q8_1 activation pipeline (Sol F1/F2 —
v3 claimed a fused MoE MMVQ path that does not exist for clamped SwiGLU
decode); capture-safe ABI made an M2a gate (F3 — static launcher,
ctx.stream() binding, pool-miss allocation all break naive wrapping); linear
projection demoted to screening bound with a TP=4 graph-captured
decoder-layer slice as the real go/no-go (F4); a minimal `wo_a` serving
prototype and Q8 repacker moved into M2 with a causal-matrix kill logic (F5);
family-level TP rules replaced by a tensor-level mapping table (F6 —
`fused_wqa_wkv` is `disable_tp=True` replicated, token_embd vocab-shards);
class-A2 coordinate-aware mapping oracle added because byte checksums cannot
catch transposed/mis-fused loads (F7 — routed down is `{N,K,E}` vs gate/up
`{K,N,E}`); tokenizer bootstrap pinned to `tokenizer_mode="deepseek_v4"` with
text-level golden tests (F8); DeepSWE paired protocol made executable and
costed (F9). Sol independently confirmed the v3 capacity arithmetic (5.832
KiB/token/rank; 17,559 tokens per 100 MiB/rank; ~139.1K point estimate).

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
| Quality | **paired** DeepSWE protocol §6: pre-registered executable statistic | quick-pack within noise of GGUF baseline | full 8-pack parity |
| Correctness | class-A/A2 dequant+mapping oracles; known-delta paths within pre-registered windows | deterministic canaries | NIAH exact recall at achieved on-GPU context |

**Context, stated honestly:** the 16 GiB host tier is an eviction/prefix-
restore tier — the promoted compose records it "is not part of the measured
230K performance path." It does **not** buy active context on top of the
on-GPU pool. 430K-class active context remains llama.cpp's exclusive
advantage (Q8_0 KV + layer split). During evaluation the two services
alternate via the validated rollback contract.

Guardrails (inherited, non-negotiable): GPU safety policy 230 W / 210–1650 MHz
untouched; one causal variable per experiment; zero-swap final states;
verified rollback to the canonical llama.cpp service; every claim measured.

## 2. Evidence inventory — what exists, correctly labeled

Three evidence grades: **proven-here** (measured on our stack),
**proven-adjacent** (real measurement, different kernel/stack — supports
feasibility, not performance), **unmeasured** (assumption to retire).

| Component | Grade | Evidence |
|---|---|---|
| IQ2_XXS/Q2_K dequant semantics **in the pinned source** | proven-here (source); plan transcription gated by the L0 oracle | pinned Whamp/llama.cpp `0379cf4bf`; v2's prose was wrong (Fable F1), v3's Route A framing was wrong (Sol F1) — the contract is generated from source and gated by oracle, never trusted from prose |
| These formats sustaining GEMM-class throughput | proven-adjacent | llama.cpp fused MMQ prefill 1,056 tok/s |
| **MoE-indexed MMVQ/MMQ machinery in the pinned fork** | proven-here for existence and llama.cpp integration; **vLLM integrability unmeasured** | device-side id compaction (`mmid.cu`, MMQ path), `mul_mat_vec_q_moe_launch` (unfused; multi-token branch), `DSV4_MMVQ_SMALLK`; decode uses unfused ops for clamped SwiGLU (`ggml-cuda.cu:4331+`) |
| F32↔Q8_1 activation pipeline cost | **unmeasured** | kernels assert F32 in/out and internally quantize Q8_1 via pool scratch (`mmvq.cu:1070-1135`, `mmq.cu:108-248`) — measured at M2 |
| TP-sharded indexed grouped-MoE with in-mainloop unpack | proven-here for int-unpack W2/W4 | Humming extensions `e5a8452c7`, `dd2d1fd6`; 7-case SM86 oracle |
| Whole non-MoE stack at speed on FP8 weights | proven-here | 74.98 tok/s composite |
| Same stack on Q8_0 weights (dense + `wo_a`) | **unmeasured — largest unmeasured kernel share** | measured at M2 |
| Correctness-oracle methodology | proven-here | WNA16 oracle ladder |
| KV-offload tier inertness | proven-here as an unused reservation | eviction-pressure behavior unmeasured |
| GGUF container parsing precedent | weak | out-of-tree/experimental; our loader is new code |

**Scope reduction:** only IQ2_XXS and Q2_K need codebook paths. Q8_0 becomes
int8 group-32 via a last-bits-lossy repack (§4). F16/F32/I32 control tensors
pass through **subject to per-kernel dtype contracts** (§4.5) and the
tensor-level TP mapping (§4.7).

## 3. The artifact contract (pinned; M1 verifies byte-level)

GGUF: `antirez/deepseek-v4-gguf`, `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-
SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, blob SHA-256 `ca22ae2f…b1c0`,
86,720,111,488 bytes, verified on server60. MTP is a separate 3.6 GiB file,
not in this blob — no MTP lever.

Tensor classes (M1 produces the authoritative per-tensor inventory):

| Class | Format | Notes |
|---|---|---|
| Routed gate/up (`ffn_gate_exps`/`ffn_up_exps`) | IQ2_XXS | GGUF axis order `{K, N, E}` |
| Routed down (`ffn_down_exps`) | Q2_K | **GGUF axis order `{N, K, E}`** — differs from gate/up |
| Attention projections, shared experts, `output` | Q8_0 | `wq_a` and `attn_kv` stored **separately** in GGUF |
| `token_embd` | F16 | **vocab-sharded at runtime** (VocabParallelEmbedding) |
| Router, indexer, compressor, HC | F16 | per-tensor TP rules from §4.7 table |
| Norms, sinks, biases | F32 | no downcast |
| `ffn_gate_tid2eid` (early layers) | I32 | `{n_expert_used, n_vocab}` hash table |

**IQ2_XXS layout (source: `vecdotq.cuh:985-1014`):** block = fp16 `d` +
32 uint16. Per 32 weights (8 bytes): two uint16 = **4 grid-index bytes**
(256-entry × 8-weight LUT, 2 KiB); the following uint32 = **four 7-bit sign
fields** plus a **5-bit integer sub-scale** `ls = aux32 >> 27 | 1`, applied
with integer truncation: `sumi = sumi * ls / 8`, then `d * bq8_1.ds * sumi`.
Q2_K = fp16 `d`+`dmin`, 16 sub-blocks of 4-bit scale/min + 2-bit quants.
Q8_0 = fp16 `d` + 32×int8. CPU `dequantize_row_*` is the golden oracle;
integer-truncation semantics are why class A covers **dequantized values in
fp32** while fused outputs live in class B. M1 also pins the tokenizer/config
contract (§4.6).

## 4. Architecture: keep / adapt / new

**Base pin:** branch `incubate/gguf-tp-sm86` in Whamp/vllm from the
speed-stack tip `b7766cfe` (tree `6354125a`) — DSML fix + SwiGLU + Marlin-wo_a
+ FlashMLA + hierarchical all-reduce + KV-offload repair, measured 74.98/887.
club-3090 `feat/gguf-tp-engine` (this worktree) owns deployment, evidence,
plans.

**Keep (proven on the pinned base):** vLLM scheduler + continuous batching;
TP=4 launcher; FlashMLA sparse decode; hierarchical all-reduce; DSML parser
fix; SwiGLU semantics; fp8_ds_mla KV + offload tier; CUDA-graph capture path;
tokenizer/chat path **subject to §4.6**.

**Adapt / new:**
1. **GGUF-native loader**: mmap, verify blob SHA-256, GGUF tensor-name →
   vLLM module mapping via the §4.7 tensor-level table, per-rank packed
   views, checksum fail-closed against the M1 inventory. IQ2_XXS/Q2_K/
   F16/F32/I32 bytes are never re-encoded.
2. **Q8_0 → int8 group-32 repack (documented lossy-in-last-bits):** exact
   int8 codes; fp16 block scale → CT scale (fp16 where the kernel allows,
   else bf16-rounded), Marlin tile-packing, uint8b128 offset. Tolerance
   oracle vs Q8_0 dequant+GEMM; excluded from the bit-exact ladder. Dense
   GEMV/GEMM perf measured at M2. **A minimal repacker + `wo_a`
   serving-shape prototype moves into M2** (Sol F5) so the go/no-go uses
   measured numbers, not sketches.
3. **`wo_a` Q8_0 output projection:** FP8 Marlin-diagonal doesn't apply to
   Q8_0; no-cache fallback measured 34.01 tok/s (fatal); BF16 cache 688
   MiB/rank. M1 scopes a Q8 grouped-Marlin (or equivalent) diagonal path
   with VRAM delta; **M2 measures a real serving-shape prototype.** Kill: if
   the only working options are BF16 cache or ~34 tok/s dequant, stop.
4. **Routed experts — two routes, measured order:**
   - **Route A (M2a): wrap the pinned fork's MoE-indexed MMVQ/MMQ machinery
     as a vLLM `FusedMoEExperts`-style operation.** Honest composition
     (Sol F1/F2): decode = routing-id prep → **bf16→F32 view → shared
     Q8_1 activation quantization for up/gate** → two IQ2_XXS MoE-indexed
     MMVQ launches (unfused — clamped SwiGLU has no vec-kernel epilogue) →
     clamp+SwiGLU → per-expert Q8_1 quantization of routed activations →
     Q2_K down → top-k weighted combine → F32→bf16. Prefill (batch > 8)
     may use the fused MMQ up/gate path. All F32/Q8_1 scratch is
     caller-owned.
   - **Capture-safe ABI is part of M2a, not an afterthought (Sol F3):**
     export a real `torch.ops` wrapper that accepts the current CUDA
     stream (not `ctx.stream()`), takes caller-owned output/Q8_1/compaction
     workspaces, pre-grows every allocation before capture, then
     capture-and-replay M=1–4 on TP=4 and test multiple graph sizes for
     pointer aliasing. The pinned launcher is `static` and its pool can
     `cudaMalloc` on miss during capture — naive wrapping is invalid.
   - **Route B (escalation, M2b): Humming indexed-expert fragments** for
     IQ2_XXS/Q2_K fused at fragment-load, required only if Route A misses
     the measured bar; carries codebook-into-MMA and smem-LUT risk.
5. **Per-kernel dtype contracts for replicated families:** M1 inventories
   the dtype/layout contract of every kernel consuming each family
   (compressor fp32 assert; indexer fused-quant; merged-GEMM fp32 hand-off;
   activation dtype bf16). Any transform or cast becomes a documented
   conversion with a class-B window and a capacity-table line. The
   **router** cast policy is explicit (F16 → bf16 is lossy where top-6
   tie-breaks live; prefer fp32/fp16-native if the kernel accepts it).
6. **Tokenizer/config source of truth (Sol F8):** the pinned stack
   auto-selects `tokenizer_mode="deepseek_v4"` from the architecture and
   that tokenizer overrides `apply_chat_template` with custom
   `encode_messages` (DSML, tools, thinking) — **the GGUF bootstrap must
   pin this mode explicitly before module/tokenizer construction**, and M1
   adds text/API→token-ID golden tests for ordinary chat, high/max
   reasoning, tool calls, and post-tool continuation. Token-ID-pinned
   probes remain for kernel isolation only, never as tokenizer evidence.
   M1 also diffs GGUF KV metadata vs HF config (RoPE theta/YaRN, compress
   ratios, SWA window) and names the authoritative source.
7. **Tensor-level TP mapping table (Sol F6) — replaces family rules:** M1
   produces, per GGUF tensor: logical shape and axis order, destination
   parameter (incl. fused slots such as `fused_wqa_wkv` = GGUF `wq_a` +
   `attn_kv` stacked in fixed order), TP axis (e.g. token_embd
   vocab-sharded; `fused_wqa_wkv` `disable_tp=True` replicated; wq_b/wo_a/
   wo_b distinct column/row rules), rank slice, quant-block row axis,
   runtime dtype, post-load storage. Capacity is derived from this table.

**Quant-method config**: `gguf_dsv4` expressing experts = GGUF-native, dense
Q8_0 = CT int8-g32, control = per-kernel-contract passthrough.

**Retained:** the compressed-tensors expert path stays for the same-tree
WNA16 A/B that attributes misses to the new route rather than to
FlashMLA/graphs/AR regressions.

## 5. Performance reasoning

- **Linear time-mix projection is a screening bound only (Sol F4).** The
  replacement swaps expert *and* dense kernels simultaneously inside
  stream-parallel, graph-captured, NCCL-overlapped decode; the base stack's
  own benchmark history shows M-regime transfer errors up to 5.6×. The real
  go/no-go is a **TP=4 graph-captured decoder-layer slice** containing the
  Q8 dense kernels, the full Route A expert operation, and the real
  all-reduce. For prefill, benchmark the **observed M distribution** from
  the fresh trace, not one nominal M≈256 point.
- Screening bound arithmetic (to be re-anchored on a fresh nsys trace of the
  running 74.98 stack at M0): experts ~15–18% of kernel time (~2.0–2.4 ms of
  13.3 ms/token), dense Q8_0 replacement ~23% share unmeasured — if dense
  runs 30% slower, ~0.9 ms of budget is gone before experts spend anything.
  Realistic expert tolerance to hold 58 tok/s: **~2.2–2.6× slower than W2
  Humming**, contingent on dense-path near-parity. The activation
  bf16→F32→Q8_1 pipeline (Route A) is included in the expert-op budget.
- **M2 measured components (all on a 3090):** fresh-trace mix; Route A full
  operation-sequence time at decode shapes; dense Q8_0-g32 Marlin GEMV;
  `wo_a` prototype time; Route B fragment time only if Route A misses.
- **Prefill falsifier:** dense Q8_0 GEMM across the observed M distribution;
  projection < 550 tok/s is a named failure before bring-up.
- Overall decode estimate band: **55–75 tok/s**, estimate until M7.

## 6. Correctness doctrine

- **A. Bit-exact (no tolerance):** IQ2_XXS/Q2_K **dequantized weight values
  in fp32** vs llama.cpp CPU `dequantize_row_*`, random + adversarial
  corpora (sign patterns, extreme scales, LUT boundary indices, sub-scale
  extremes). Fused outputs are class B by construction.
- **A2. Coordinate-aware mapping oracle (Sol F7):** byte checksums cannot
  catch a transposed or mis-fused load (routed down `{N,K,E}` vs gate/up
  `{K,N,E}`; `fused_wqa_wkv` slot order). For every tensor family and TP
  boundary, sample `(expert, output row, input column)` coordinates, derive
  the GGUF byte offset independently, decode, and compare the destination's
  logical value — covering first/last block, first/last rank, fused-slot
  boundaries, and hash-table (`tid2eid`) rows.
- **B. Known-delta (pre-registered windows):** Q8_0-repack GEMM vs dequant+
  GEMM; **Route A end-to-end expert op** (incl. activation quantization and
  F32→bf16 conversion) vs reference GEMM; full-model forward vs llama.cpp
  on fixed prompt sets (KL bound stated before first run); fp8_ds_mla KV /
  FlashMLA / hier-AR / replicated-family casts each with their own window.
- **C. Determinism:** CUDA-graph replay vs eager equality; AR rank-order
  consistency; **graph-size sweep for pointer aliasing** (Route A
  workspaces).
- **D. End-to-end:** deterministic canaries; tool round-trip and post-tool
  continuation; NIAH exact recall at achieved on-GPU context; **DeepSWE
  paired protocol, executable spec (Sol F9):** unit of analysis = task
  (seeds clustered within task); task-cluster bootstrap or hierarchical
  permutation (task, then seed) over ≥3 seeds per engine on the 12-task
  set (≥72 cells minimum); pre-registered combination rule — e.g.
  non-inferiority margin on mean partial reward AND strict-solve count not
  lower than baseline − 1 by the permutation null; explicit tie/missing-cell
  policy; **one paired-seed pilot first** to measure wall time and failure
  rate before committing to the full grid. A single SuperJSON run is a
  smoke signal only.

Ladder L0→L6, adversarial review loops (1 implementer + 2 reviewers on diff +
format contract), checksums on every tensor view, oracle failures batched as
work queue — unchanged.

## 7. Execution methodology

Unchanged: FORMAT-CONTRACT.md generated from source before code and gated by
the L0+A2 oracles; trial-first; loader cutover on its own branch; process
fixes over hand fixes; worktree/commit discipline; one causal variable;
Nsight attribution before kernel tuning; unprofiled end-to-end numbers for
claims. All kernel gates on a server60 RTX 3090. **No rental compute
planned.**

## 8. Milestones, gates, kill criteria

| # | Deliverable | Gate | Kill / pivot | Est. |
|---|---|---|---|---|
| M0 | Worktrees, base pin `b7766cfe`, fresh nsys trace of the 74.98 stack, baseline re-anchor | pins + trace recorded | — | 1 d |
| M1 | `FORMAT-CONTRACT.md` from source; per-tensor inventory; **§4.7 tensor-level TP table**; per-kernel dtype contracts; tokenizer bootstrap pin + text-level golden tests; `wo_a` design + VRAM delta; capacity table with every delta sized in MiB → tokens | L0 class-A oracle 100%; A2 mapping oracle design; inventory matches blob; capacity table shows ≥140K or levers whose summed size closes the gap | inventory mismatch → re-scope; `wo_a` no viable design → **stop**; capacity gap unclosable → re-decide scope with Will | 2–4 d |
| M2a | Route A wrapper with **capture-safe ABI** (torch.ops, current-stream, caller-owned workspaces, pre-grown allocations); capture/replay M=1–4 TP=4 + aliasing sweep; full-op microbench at decode shapes; **minimal Q8 repacker + dense GEMV + `wo_a` prototype measured**; prefill M-distribution bench; **TP=4 graph-captured decoder-layer slice** as the real projection | screening projection ≥ 58 decode and ≥ 550 prefill from the layer slice, dense and `wo_a` components measured | causal matrix (Sol F5): expert-op miss → escalate M2b; dense/`wo_a` miss → dense redesign or **stop**; one-metric-only miss → explicit decision with Will | 1–2 wk |
| M2b | Humming IQ2_XXS fragment (escalation only) | beats Route A's measured bar; graph-capturable | misses after 2 tuning iterations → take Route A if ≥ floor, else stop | 1–2 wk |
| M3 | Q2_K on the winning route | same | same | 1 wk |
| M4 | Production GGUF loader + Q8_0 repack + `gguf_dsv4` config + `wo_a` path; CPU tests; checksum fail-closed; **calendar kill: 10 working days** | full-tensor mapping test incl. A2 boundaries; byte-identity assert on packed IQ2_XXS/Q2_K views | calendar breach → descope review | 1–2 wk |
| M5 | server60 TP=4 bring-up (authorized window; validated rollback) | class-A/B full-path oracle on-GPU; NCU dispatch; readiness | repeated OOM/instability → capacity re-plan | 0.5–1 wk |
| M6 | Per-layer vs llama.cpp (class B); canaries + NIAH at achieved context | pre-registered windows pass | unexplained divergence → bisect | 3–5 d |
| M7 | Matched perf campaign; same-tree WNA16 A/B attribution | ≥58 engine decode, ≥550 prefill, ≥140K on-GPU, zero swap | miss → keep llama.cpp canonical; publish | 3–5 d |
| M8 | Quality: quick pack + **paired multi-seed DeepSWE** (pilot-priced full grid) | §6 executable paired statistic passes its pre-registered rule | divergence → component bisect | 1–3 wk (pilot-priced) |
| M9 | Promotion package; open-source decision | Will's approval; healthy final service | — | 2–3 d |

Effort envelope: **7–10 weeks if gates pass first-try**; M2b/M3 iterations,
M6/M8 bisects, `wo_a` redesign, and the DeepSWE grid cost are the
contingency sources — kills bound the downside, not the calendar.

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `wo_a` Q8_0 path slow or fat | medium-high | fatal if unsolved | M1 design + M2 prototype; kill |
| Dense Q8_0 path slower than FP8 Marlin | medium | eats the 58 budget | measured at M2a; feeds go/no-go |
| Route A activation pipeline (bf16→F32→Q8_1, F32→bf16) too costly at decode | medium | misses 58 | measured inside the M2a full-op budget; shared up/gate Q8_1 buffer |
| Capture-safe ABI breaks on graph sizes / aliasing | medium | eager fallback ≈ 5.5 tok/s | ABI is an M2a gate with replay + sweep |
| Expert route too slow at decode M≤8 | medium | misses 58 | Route A measured first; Route B escalation; ~2.2–2.6× tolerance |
| Linear projection misleads route selection | medium | wrong route chosen | screening only; layer-slice + M-distribution are the gates |
| Capacity: 140K point estimate; ~17.6K context per 100 MiB/rank unmodeled | certain (sensitivity) | context floor | M1 sizes every delta from the §4.7 table |
| Byte-valid but logically wrong load (transpose/fused-slot/rank offset) | medium | silent quality damage | class-A2 coordinate oracle |
| Tokenizer bootstrap falls back to generic HF mode | medium | contaminated comparisons | §4.6 explicit pin + text-level golden tests |
| Router cast degrades top-6 tie-breaks | medium | silent quality damage | §4.5 explicit policy; class-B window |
| DeepSWE grid cost/noise | medium | false pass/fail or overrun | executable paired statistic; pilot prices the grid |
| Effort overrun | medium | opportunity cost | M1/M2a/M4 kills; calendar caps |

## 10. Capacity plan

Method: M1's table from §4.7 — per-rank registered weights by tensor (sharded
vs replicated incl. `fused_wqa_wkv` replication and vocab-sharded
token_embd), post-transform sizes, graph pool (~0.19 GiB measured), Humming/
NVRTC workspace, loader/repack scratch, Marlin tile padding, KV pool,
headroom. Anchors: 78.74 GiB WNA16 artifact reached 230,144 ctx with 1.28 GiB
available KV; KV density ≈ 5.832 KiB/token/rank (independently recomputed by
the third review) → **~17.6K context per 100 MiB/rank**. The sharded GGUF tax
(~0.5 GiB/rank) gives ~139.1K — a point estimate with the replicated-family
delta at zero; precedent (indexer 191→767 MiB/rank after transforms) says it
must be measured, not assumed small. The M1 gate requires the completed table
to show ≥140K or levers whose summed MiB close the gap explicitly. The 16 GiB
host tier ships as prefix-reuse only, gated by its hit-rate line. 430K active
stays llama.cpp's exclusive advantage.

## 11. What "done" means

server60 serves `deepseek-v4-flash-0731-gguf-tp` from the pinned GGUF blob at
≥58/70 engine decode, ≥550/700 prefill, ≥140K on-GPU unique context (per the
M1 capacity table), zero swap, safety policy intact, paired-protocol DeepSWE
quality within its pre-registered window, validated rollback, everything
committed and pushed, evidence bundled, upstreaming decision recorded. The
llama.cpp service remains canonical until M8 passes.

## 12. Immediate next actions (on approval)

1. M0: worktrees, base pin `b7766cfe`, fresh nsys trace of the 74.98 stack,
   baseline re-anchor.
2. M1: FORMAT-CONTRACT + L0/A2 oracle design + inventory + §4.7 TP table +
   dtype contracts + tokenizer pin + `wo_a` design + capacity table.
3. M2a: Route A wrapper with capture-safe ABI + full-op microbench + dense/
   `wo_a` prototypes + decoder-layer slice — the first hard end-to-end
   number.
4. Standing question for Will at M1 exit: accept the measured on-GPU context
   floor as this service's contract (llama.cpp retained for 430K-class
   needs), or require named-and-sized levers first?
