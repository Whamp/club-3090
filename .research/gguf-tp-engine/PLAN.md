# PLAN: Native GGUF tensor-parallel inference for DeepSeek-V4-Flash on 4× RTX 3090

Status: draft v2 — revised after adversarial review (Grok 4.6 xhigh, 2026-08-17;
archived at `reviews/2026-08-17-grok-4.6-xhigh.md`). Awaiting Will's approval
before M0. v1→v2 changes: honest capacity contract, `wo_a` Q8_0 made a scoped
critical-path item, correct base pin, per-family perf reasoning replacing the
bad 0.55 GiB budget, kernel gates moved to a 3090 with a mapped e2e gate,
Q8_0 repack reclassified as last-bits-lossy, full tensor-type inventory,
oracle ladder split into bit-exact vs known-delta, CT expert path retained.

## 1. Objective and decision frame

Run the **exact proven-good Antirez GGUF bytes** (IQ2_XXS routed gate/up, Q2_K
routed down, Q8_0 attention/shared/output, F16/F32 control tensors) inside a
**vLLM-style tensor-parallel runtime** at vLLM-class speed, on server60's four
RTX 3090s. No requantization, no lossy conversion of the routed experts or
control tensors. This is an inference-engine ASIC: one model, one artifact
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
updates would load through the same pinned contract (a new inventory pass, not
a rewrite — the fail-closed contract is versioned to this artifact revision).
And "GGUF in tensor parallelism" is a standing community ask nobody has
shipped.

**Decision thresholds** (decode = engine tokens/s, matching the 74.98
reference and the 37.5–39.2 llama.cpp engine baseline; client-wall numbers are
~12% lower and reported alongside):

| Outcome | Minimum success | Target | Stretch |
|---|---:|---:|---:|
| Single-stream decode (engine) | ≥ 58 tok/s (~1.5× llama.cpp engine 38.4) | 70 tok/s | 75 tok/s (WNA16 stack parity) |
| Cache-busted prefill | ≥ 550 tok/s | 700 tok/s | 887 tok/s (inherited-stack parity) |
| **On-GPU unique-request context** | ≥ 140K | 155K | 170K+ via levers |
| Prefix-reuse offload tier | present, measured hit-rate ≥ 60% on repeated-prefix workload | — | — |
| Quality | DeepSWE window vs GGUF baseline: strict solves ≥ baseline−1 **and** mean partial ≥ baseline−2.0 pp | quick-pack within noise of GGUF baseline | full 8-pack parity |
| Correctness | IQ2_XXS/Q2_K bit-exact dequant vs llama.cpp; known-delta paths within pre-registered windows | deterministic canaries | NIAH exact recall at achieved on-GPU context |

**Context, stated honestly (review F1/F10):** the 16 GiB host tier is an
eviction/prefix-restore tier — the promoted compose itself records it "is not
part of the measured 230K performance path," and native vLLM CPU offload
restores reusable prefixes; it does not let active decode attend to
CPU-resident history. It therefore does **not** buy 230K active context on top
of a ~140–155K on-GPU pool, and this plan does not claim it. 430K-class
active context remains llama.cpp's exclusive advantage (Q8_0 KV + layer
split). Reaching ≥200K active context in this engine would require a measured
paging path beating an ITL floor on GPU0's ~3 GiB/s Gen3-x4 link — out of
scope unless capacity levers surprise us at M1.

Guardrails (inherited, non-negotiable): GPU safety policy 230 W / 210–1650 MHz
untouched; one causal variable per experiment; zero-swap final states;
verified rollback to the canonical llama.cpp service; every claim measured.

## 2. Evidence inventory — what exists, correctly labeled

Three evidence grades (review F11): **proven-here** (measured on our stack),
**proven-adjacent** (real measurement, different kernel/stack — supports
feasibility, not performance), **unmeasured** (assumption to retire).

| Component | Grade | Evidence |
|---|---|---|
| IQ2_XXS/Q2_K exact dequant semantics | proven-here | pinned Whamp/llama.cpp `0379cf4bf` source read line-by-line; MMVQ campaign |
| These formats sustaining GEMM-class throughput | proven-adjacent | llama.cpp's own fused MMQ prefill at 1,056 tok/s — a different kernel than our planned fragment, but proves the encodings feed Tensor-Core-class pipelines at large tiles |
| TP-sharded indexed grouped-MoE skeleton with in-mainloop unpack | proven-here for int-unpack W2/W4, **unmeasured for codebook fragments** | Humming extensions `e5a8452c7`, `dd2d1fd6`; 7-case SM86 oracle with verified sm_86 cubins |
| Whole non-MoE stack at speed on FP8 weights | proven-here | DSML fix, SwiGLU fix, FlashMLA decode, hierarchical all-reduce: 74.98 tok/s composite |
| Same stack on Q8_0 weights | **unmeasured** | Q8_0 dense path (incl. `wo_a`) is new work — see §4.3 |
| Correctness-oracle methodology | proven-here | WNA16 oracle ladder, every campaign since `f4d05732a` |
| KV-offload tier inertness | proven-here **as an unused reservation** | measured inert as an idle tier below ~275K; its behavior under actual eviction pressure is unmeasured — hence the separate hit-rate success line |
| GGUF container parsing precedent | weak | GGUF support moved out-of-tree (experimental plugin, dequant-oriented) in this lineage; our loader is new code regardless — cite nothing as reusable beyond the format spec itself |

**Scope reduction that makes this tractable:** only *two* codebook formats
(IQ2_XXS, Q2_K) need new kernel fragments. Q8_0 becomes int8 group-32 via a
last-bits-lossy repack (§4.2). F16/F32/I32 control tensors load as-is — no
BF16 downcast, no dtype changes at all outside the two documented conversions.

## 3. The artifact contract (pinned; M1 verifies byte-level)

GGUF: `antirez/deepseek-v4-gguf`, `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-
SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, blob SHA-256 `ca22ae2f…b1c0`,
86,720,111,488 bytes, verified on server60. MTP is a **separate 3.6 GiB file,
not in this blob** — there is no MTP recovery lever (review F8/F10).

Tensor classes (M1 produces the authoritative per-tensor inventory; counts
are not assumed):

| Class | Format | Notes |
|---|---|---|
| Routed gate/up (`w1`/`w3`) | IQ2_XXS | 66 B/256 weights = 2.0625 bpw |
| Routed down (`w2`) | Q2_K | 84 B/256 = 2.625 bpw, dual-scale affine |
| Attention projections, shared experts, `output` | Q8_0 | 34 B/32 = 8.5 bpw, includes `wo_a` |
| `token_embd`, router, indexer, compressor, HC | F16 | replicated families — see §5 |
| Norms, sinks, biases | F32 | no downcast |
| `ffn_gate_tid2eid` (early layers) | I32 | pass-through |

Block layouts from pinned source (`ggml-common.h`, `vecdotq.cuh`): IQ2_XXS =
fp16 `d` + 32×uint16, each uint16 = grid-LUT byte (256-entry LUT, 2 KiB) +
8 sign bits; Q2_K = fp16 `d`+`dmin` super-block, 16 sub-blocks of 4-bit
scale/min + 2-bit quants; Q8_0 = fp16 `d` + 32×int8. Reference vec_dot:
`vecdotq.cuh:985` (IQ2_XXS), `:814`/`:393` (Q2_K mmvq/mmq variants). CPU
`dequantize_row_*` in `ggml-quants.c` is the golden oracle. M1 also pins the
TP-shard geometry per family (experts/attention shard; indexer, compressor,
HC, router replicate — residency audit `965aebbfb502`).

## 4. Architecture: keep / adapt / new

**Base pin (review F3):** branch `incubate/gguf-tp-sm86` in Whamp/vllm from
the **speed-stack tip** `b7766cfe` (tree `6354125a`) — this is the tree that
actually contains DSML fix + SwiGLU + Marlin-wo_a + FlashMLA + hierarchical
all-reduce + KV-offload repair and measured 74.98/887. The earlier capacity
tree (`67064365`, 61.91 tok/s) lacks FlashMLA/hier-AR and is **not** the
base. club-3090 `feat/gguf-tp-engine` (this worktree) owns deployment,
evidence, plans.

**Keep (proven on the pinned base):** vLLM scheduler + continuous batching;
TP=4 launcher; FlashMLA sparse decode; hierarchical all-reduce; DSML parser
fix; SwiGLU semantics; fp8_ds_mla KV + offload tier; CUDA-graph capture path;
tokenizer/chat path.

**Adapt / new:**
1. **GGUF-native loader**: mmap, verify blob SHA-256, GGUF tensor-name →
   vLLM DeepSeek-V4 module mapping (held from frontier work), per-family TP
   shard rules, per-rank packed views, checksum fail-closed on any mismatch
   against the M1 inventory. IQ2_XXS/Q2_K/F16/F32/I32 bytes are never
   re-encoded.
2. **Q8_0 → int8 group-32 repack (documented lossy-in-last-bits, review F7):**
   symmetric int8 codes are exact; the fp16 block scale becomes the CT scale
   — we keep it fp16 where the consuming kernel allows and otherwise accept a
   bf16-rounded scale (mantissa-bit loss), plus Marlin tile-packing and
   uint8b128 offset. This is a conversion with a stated tolerance oracle
   (atol/rtol vs Q8_0 dequant+GEMM), **not** "zero quality change," and it is
   excluded from the bit-exact ladder.
3. **`wo_a` Q8_0 output projection (new critical-path item, review F2):** the
   FP8 Marlin-diagonal path keeps block-FP8 weights and does not apply to Q8_0
   inputs; its no-cache fallback measured 34.01 tok/s (fatal), and the BF16
   cache costs 688 MiB/rank (more than the entire GGUF weight tax). M1 scopes
   a Q8 grouped-Marlin (or equivalent) diagonal path with a VRAM delta and a
   measured decode number. **Kill:** if the only working options are BF16
   cache at unacceptable capacity or per-token dequant at ~34 tok/s, the
   project stops at M1/M4 — that path already lost both gates once.
4. **Routed experts (the core):** extend the Humming indexed-expert mainloop
   with IQ2_XXS and Q2_K dequant fragments fused at fragment-load, feeding
   the same mma pipeline; new schema entries + Sm86 heuristics + JIT dispatch;
   graph-capturability required (an eager-only fragment falls toward the
   measured 5.5 tok/s eager regime).
5. **Quant-method config**: `gguf_dsv4` expressing experts = GGUF-native
   IQ2_XXS/Q2_K, dense Q8_0 = CT int8-g32, control = F16/F32 passthrough.

**Retained, not deleted (review F12):** the compressed-tensors expert path
stays — it is a different quant method serving the existing safetensors
artifacts, and keeping it preserves the only same-tree A/B that can attribute
a performance miss to the new fragments rather than to FlashMLA/graphs/AR
regressions. No dual-path serving; both methods simply coexist in the tree.

**Recorded alternative (review F13):** wrap the measured ggml-cuda MMVQ/MMQ
kernels as vLLM expert ops instead of writing Humming codebook fragments.
Counter-analysis: those are single-GEMM kernels; vLLM needs a graph-capturable
fused indexed grouped GEMM with device-side top-6 routing over 256 experts —
exactly the machinery Humming provides and llama.cpp solves with launch-bound
CPU-side per-expert dispatch. The hidden cost is reimplementing indexed
dispatch, not the vec_dot. Decision: Humming-fragment is primary (reuses all
outer machinery); wrapped-ggml is the fallback arm if fragments miss the M2
gate. Nsight says WNA16 expert kernels are only ~15% of decode kernel time,
which is what makes either route viable at all.

## 5. Performance reasoning (replaces v1's incorrect 0.55 GiB budget)

v1 divided a wrong byte total by four and called it causal. Correct frame:

- **Per-family decode traffic** (residency audit `965aebbfb502`, TP=4):
  experts shard (~0.43 GiB activated/rank/token at top-6/256), attention and
  shared experts shard, vocab-parallel embed/head (~0.53 GiB/rank), but
  indexer/compressor/HC/router **replicate** — total order ~3.5 GiB/rank/token,
  not 0.55. Weight movement alone is not the decoder's causal budget.
- **Measured time mix** (Nsight decode trace, pre-FlashMLA baseline):
  NCCL all-reduce ~19%, sparse decode ~14%, Marlin ~23%, all Humming W2/W4
  kernels ~15%, remainder sync/launch/other. The 74.98 stack already attacks
  the top two.
- **Expert-kernel slowdown tolerance (the real M2 gate arithmetic):** at
  74.98 tok/s (13.3 ms/token) with experts ≈15% ≈ 2.0 ms, holding everything
  else constant: 58 tok/s tolerates experts ~3× slower than W2 Humming;
  70 tok/s ~1.5×; 75 tok/s requires parity. These are estimates from kernel-
  time shares, pinned at M2 by measuring the W2 fragment's µs at the exact
  serving shapes (M=1–8, K=4096/2048, ≤8 rows/expert) on a 3090.
- **Prefill:** inherits chunked prefill at `max_num_batched_tokens=256`
  (raising it failed headroom gates on the *smaller* WNA16 artifact; GGUF is
  tighter, so M stays ~256). llama.cpp's 1,056 tok/s is ubatch-384 fused MMQ —
  a different regime. Honest band: **550–887 tok/s**, floor 550, stretch =
  inherited-stack parity 887. v1's 950 stretch is withdrawn (review F6).
- Overall decode estimate band: **55–75 tok/s**, labeled estimate until M7.

**Falsifier:** M2 measures the IQ2_XXS fragment at serving shapes and maps it
through the measured time mix. If projected engine decode < 58 after two
tuning iterations **and** the wrapped-ggml fallback arm also projects < 58,
the speed objective fails → stop or re-scope (prefill-only win does not
justify the project alone).

## 6. Correctness doctrine

Split into four oracle classes (review F9) — v1's "any divergence is our bug"
overstated, because the serving stack legitimately differs from llama.cpp
(fp8_ds_mla KV vs Q8_0 KV, FlashMLA vs fork sparse attention, Marlin vs MMVQ,
hier-AR vs none):

- **A. Bit-exact (must hold exactly):** IQ2_XXS/Q2_K fragment dequant vs
  llama.cpp CPU `dequantize_row_*` on random + adversarial corpora (sign
  patterns, extreme scales, LUT boundary indices). No tolerance.
- **B. Known-delta (pre-registered windows):** Q8_0-repack GEMM vs Q8_0
  dequant+GEMM (atol/rtol stated per shape); full-model forward vs llama.cpp
  on a fixed prompt set (KL bound on next-token logits, stated before the
  first run); fp8_ds_mla KV / FlashMLA / hier-AR deltas inherited from the
  WNA16 stack — re-anchored, not re-litigated.
- **C. Determinism:** CUDA-graph replay vs eager equality; AR rank-order
  reduction consistency across runs.
- **D. End-to-end:** deterministic generation canaries; tool round-trip and
  post-tool continuation (the DSML lesson); NIAH exact recall at the achieved
  on-GPU context; DeepSWE single-worker SuperJSON vs the GGUF-llama.cpp
  baseline within the pre-registered window (≥ baseline−1 strict solves,
  ≥ baseline−2.0 pp partial). Divergence means an integration bug **or** a
  stack delta escaping its window — bisect by layer and by component, never
  hand-wave.

Ladder L0→L6 from v1 otherwise stands: CPU bit-exact → standalone CUDA
fragment → full-kernel oracle on real tensors (`HummingIndexedExperts.apply`
pattern, extended with GGUF schemas) → NCU sm_86 dispatch proof → per-layer
forward → canaries/NIAH → DeepSWE. Adversarial review loops (1 implementer +
2 reviewers seeing only diff + format contract), checksums on every tensor
view, oracle failures batched as work queue — all unchanged from v1 §6.

## 7. Execution methodology

Unchanged from v1: FORMAT-CONTRACT.md before code; trial-fragment-first;
everything-at-once loader cutover on its own branch (the CT path stays in-tree
per §4); process fixes over hand fixes; worktree/commit discipline; one causal
variable per perf experiment; Nsight attribution before kernel tuning;
unprofiled end-to-end numbers for claims.

**Hardware for gates (review F5):** all kernel gates run on a server60 RTX
3090 — the MMVQ campaign proved co-resident single-GPU microbenching under
the safety cap, and the full-kernel oracle ran on a 3090 (78.63 s). A100
rental is **dropped**: sm_80 cubins validate nothing about sm_86 smem/SM
limits, and nothing in M0–M4 needs rental compute. Current rental plan:
**none required**; revisit only if a need actually appears.

## 8. Milestones, gates, kill criteria

| # | Deliverable | Gate | Kill / pivot | Est. |
|---|---|---|---|---|
| M0 | Worktrees, pins (base `b7766cfe`), baselines re-anchored (llama.cpp engine 37.5–39.2 decode / 1,056–1,175 prefill; WNA16 74.98/887 reference) | plan approved; pins recorded | — | 0.5 d |
| M1 | `FORMAT-CONTRACT.md`: per-tensor inventory (all six type classes), TP shard plan, **`wo_a` Q8 path scoped with VRAM delta**, residency-based capacity table (registered/replicated/graph-pool 0.19 GiB/KV/headroom) | inventory matches blob exactly; L0 class-A oracle 100%; capacity table yields ≥140K projection or a named lever | inventory mismatch → re-scope; `wo_a` has no viable path → **stop** | 2–4 d |
| M2 | IQ2_XXS fragment in Humming mainloop; class-A/B fragment oracles; 3090 microbench at serving shapes; mapped e2e projection | fragment graph-capturable; projected decode ≥ 58 via §5 arithmetic; fallback arm costed | both fragment and wrapped-ggml project < 58 → stop or re-scope | 1–2 wk |
| M3 | Q2_K fragment (affine dual-scale; watch register pressure) | same | same | 1 wk |
| M4 | GGUF loader + Q8_0 repack + `gguf_dsv4` config + `wo_a` path; CPU tests; checksum fail-closed; **calendar kill: 10 working days** | full-tensor mapping test; byte-identity assert on packed IQ2_XXS/Q2_K views | calendar breach → descope review | 1 wk |
| M5 | server60 TP=4 bring-up (authorized window; canonical llama.cpp down, validated rollback) | class-A/B full-kernel oracle on-GPU; NCU dispatch; readiness | repeated OOM/instability → capacity re-plan | 0.5–1 wk |
| M6 | Per-layer vs llama.cpp (class B); canaries + NIAH at achieved context | pre-registered windows pass | unexplained divergence → bisect | 3–5 d |
| M7 | Matched perf campaign, one variable at a time; same-tree WNA16 A/B to attribute | ≥58 engine decode, ≥550 prefill, ≥140K on-GPU, zero swap | miss → keep llama.cpp canonical; publish | 3–5 d |
| M8 | Quality: quick pack + DeepSWE window vs GGUF baseline | §1 window passes | divergence → component bisect, not promotion | 2–4 d |
| M9 | Promotion package: compose, image, INTERNALS/report, evidence bundle, rollback drill; open-source decision | Will's promotion approval; healthy final service | — | 2–3 d |

Effort envelope: **6–9 focused weeks**. v1 called M4–M9 "machinery we have run
three times" — partially false (new quant method, mmap loader, Q8 repack, two
fragments are all firsts); the calendar kill on M4 and the retained CT A/B are
the hedges.

## 9. Risk register (deltas from v1 in bold)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **`wo_a` Q8_0 path slow or fat** (FP8 Marlin inapplicable; dequant = 34 tok/s; BF16 cache = 688 MiB/rank) | medium-high | fatal if unsolved | scoped at M1 with kill; Q8 grouped-Marlin diagonal is the design hypothesis |
| Codebook fragment too slow at decode M≤8 | medium | misses 58 | §5 tolerance math (3× headroom to floor); wrapped-ggml fallback arm; hybrid dispatch |
| Q2_K dual-scale register pressure | medium | Q2_K slow | separate accumulators; `_impl_mmq` integer variant study |
| **Capacity: ~140K on-GPU honest floor** (1.28 GiB KV avail − ~0.5+ GiB tax; replicated F16 indexer/HC not in the file-size delta; graph pool 0.19 GiB) | certain | context floor | residency table at M1; levers: `wo_a` path choice, fragment smem; offload tier = prefix-reuse only |
| Grid LUT smem pressure | low-medium | pipeline stages shrink | constant memory + per-tile cache; measure at M2 |
| Stack-delta numeric drift read as loader bug (or waved through) | medium | wasted weeks / silent damage | class-B pre-registered windows; component bisect |
| Effort overrun | medium | opportunity cost | M1/M2/M4 kills; calendar caps |
| Integration drift vs base | low | rebase pain | pin `b7766cfe`; no mid-project merges |

## 10. Capacity plan (residency-based, per review F10)

Method: M1 builds the table — per-rank registered weights by family (sharded
vs **replicated**), graph pool (~0.19 GiB measured on this stack), Humming
workspace, load scratch, KV pool, headroom — not file-size ÷ 4. Anchors: the
78.74 GiB WNA16 artifact reached 230,144 ctx with 1.28 GiB available KV /
93 MiB headroom; the GGUF adds ~0.5 GiB/rank sharded weight tax **plus**
replicated-family deltas (F16 indexer/compressor/HC vs the artifact's mixed
dtypes), so the honest on-GPU projection is **~140K**, floor 140K, stretch
155–170K with levers. The 16 GiB host tier ships as prefix-reuse only and is
gated by its own hit-rate line. 430K active stays llama.cpp's exclusive
advantage; users who need it keep the canonical service (both can coexist on
different ports only transiently during evaluation — normally one or the
other owns the GPUs).

## 11. What "done" means

server60 serves `deepseek-v4-flash-0731-gguf-tp` from the pinned GGUF blob at
≥58/70 engine decode, ≥550/700 prefill, ≥140K on-GPU unique context, zero
swap, safety policy intact, DeepSWE within the pre-registered window of the
GGUF baseline, validated rollback to canonical llama.cpp, everything
committed and pushed, evidence bundled, and an upstreaming decision recorded.
The llama.cpp service remains canonical until M8 passes; promotion is a
separate explicit step.

## 12. Immediate next actions (on approval)

1. M0 half-day setup (worktrees, base pin `b7766cfe`, baseline re-anchor).
2. M1 format contract + inventory + `wo_a` scope + capacity table — the
   go/no-go density of M1 is now much higher than in v1.
3. Standing question for Will at M1 exit: accept the measured on-GPU context
   floor as the GGUF-TP service's contract (with llama.cpp retained for
   430K-class needs), or require named levers first?
