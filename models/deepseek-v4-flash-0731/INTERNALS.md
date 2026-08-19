# DeepSeek V4 Flash internals

This page records the engineering behind the canonical DeepSeek V4 Flash
llama.cpp profile:
`llamacpp/deepseek-flash-multi4-antirez-iq2-fast-prefill`.

The former Unsloth CPU-offload profiles were retired from the catalog and moved
to `compose/_archive/`. They were slower, less completely validated, and added
ambiguity after the Antirez profile proved superior for the four-RTX-3090
target. The archive preserves historical links; it is not a supported launch
surface.

## Validated configuration

| Component | Validated value |
|---|---|
| GPUs | 4× RTX 3090, PCIe only, sm_86 |
| Engine base | [`alesha-pro/llama.cpp` `ds4-longctx` at `b001c8cd7`](https://github.com/alesha-pro/llama.cpp/commit/b001c8cd73855c9c4d5d89226f08179b6f3417d6) |
| Q8 repair | [`Whamp/llama.cpp` `0379cf4bf`](https://github.com/Whamp/llama.cpp/commit/0379cf4bf889f3d28038a005210c4bc193fc8ba1) |
| Published image | `ghcr.io/whamp/llama-cpp-ds4-longctx@sha256:a96bd947d63eb81d8baf9f6f5ecb26669476383976717237450fbb5727b03745` |
| Weights | [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf) IQ2_XXS, revision `e7f04037032990db0346398d249baf9fb9df1ccc` |
| KV cache | matching `q8_0` K and V |
| Context | 430,080 tokens reserved; 395,282-token exact-recall validation |
| Split | layer, `1,1,0.95,1.05` |
| Power | observed at the rig's unchanged 230 W/card setting |
| Reasoning controls | off, or thinking at `low` (default), `high`, or `max` effort |

## Canonical server60 deployment

The profile became server60's sole DeepSeek service on 2026-08-16:

- deployment source: `Whamp/club-3090@8e63f07c0dec044ebbc7818a557e5fcce2db1c12`;
- image: `sha256:a96bd947d63eb81d8baf9f6f5ecb26669476383976717237450fbb5727b03745`;
- artifact: 86,720,111,488-byte Antirez GGUF with SHA-256
  `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`;
- service identity: `deepseek-v4-flash-0731-q8-fast-prefill` on port 8033;
- warm-up: 429,568 tokens at 276.43 tokens/s;
- functional gate: all applicable `verify-full.sh` checks passed, including
  ordinary and streaming tool calls, thinking separation, and output-quality
  checks;
- final state: healthy, zero restarts, `unless-stopped`, and zero process swap.

The 430K profile still carries the measured low-headroom caveat. After final
verification, server60 removed the stopped WNA16, stock llama.cpp, Unsloth
router, and duplicate pre-publication fast-prefill containers. Model artifacts,
research evidence, and non-launchable archived composes remain available.

The engine image is separately named and digest-pinned in
`scripts/lib/profiles/engines/llama-cpp-ds4-longctx.yml`. It does not replace
`llama-cpp-local`. See [`docs/UPSTREAM.md`](../../docs/UPSTREAM.md) for the pin,
source dependency, and retirement trigger.

## Graded reasoning is a profile capability

Most club-3090 profiles expose reasoning as a binary on/off choice. This
profile also preserves DeepSeek V4 Flash 0731's three official thinking levels:
`low`, `high`, and `max`. The registry exports those levels and identifies
`low` as the local default when thinking is enabled.

Pass both controls through `chat_template_kwargs`:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true,
    "reasoning_effort": "max"
  }
}
```

The current llama-server ignores a top-level `reasoning_effort` field. Clients
must use the nested form above. A live prompt-rendering probe on the published
image measured 11 prompt tokens for off and low, 90 for high, and 103 for max.
The distinct high/max lengths confirm that the endpoint injects both official
prefixes instead of silently collapsing the levels.

The effort setting and output budget are separate. `low` adds no prefix;
`high` and `max` prepend different instructions before the system message.
Full 8-pack runs exercised all three levels:

| Effort | Output cap | Pass@1 | Pass@3 | Cap audit |
|---|---:|---:|---:|---|
| `low` | 16,384 | 121/150 | 131/150 | no cap hits |
| `high` | 65,536 | 121/150 | 132/150 | no cap hits |
| `max` | 65,536 | 123/150\* | 130/150 | 4 of 226 API responses hit the cap |

\* The four capped `max` responses ended with `finish_reason=length`; all were
in CLI-40. The score is retained with this limitation rather than rerunning at
an even larger budget.

These scores are regression evidence for the quantized serving path, not a
measure of DeepSeek V4 Flash's intelligence ceiling. The same Antirez weights
and Q8 cache scored 122/150 at stock `b10200`'s default `low` effort, one point
above the fork. The thinking-off comparison was similarly close: 111/150 on
stock versus 109/150 on the fork. Several failed cases also overlap open
benchlocal-cli benchmark-definition problems in DataExtract, StructOutput, and
ReasonMath. DeepSeek's [official model
card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) documents the
three effort levels and reports its code-agent evaluations at `max` with
temperature 1.0 and top-p 0.95.

## Why this fork is faster

The fork's main gain is capture-always CUDA graphs for prefill micro-batches.
The stock path submits roughly thousands of CUDA kernels per micro-batch from
one host thread. The fork captures that work and replays it with one graph
launch. Its stable-topology graph builder, resident MoE scheduler, fused MMQ,
sparse attention, and allocator controls keep the captured graph reusable.

This is a system, not a one-commit optimization. A minimal graph-only port onto
the stock `b10200` image captured and launched graphs but produced no speedup.
Adding only stable-topology padding improved stock prefill by about 22% at 32K,
but the remaining graph churn still prevented the capture-always gain. The
incubating profile therefore uses the separately named external engine rather
than presenting a partial patch as mainline llama.cpp.

The validated compose enables all of the fork's linked `DSV4_*` paths. Do not
remove one as an isolated cleanup: several are prerequisites for reusable graph
capture.

## Q8 repair

The old fork forced DeepSeek V4 to F16 KV because its Q8 path produced corrupt
output. Q8 was not fundamentally incompatible with the architecture. Four
storage-path defects caused the failure.

1. **CUDA concat rejected packed Q8 rows.** DeepSeek V4 concatenates raw and
   compressed cache rows on every attention layer. The fork accepted only F32
   and F16 on CUDA, so Q8 fell back to CPU.
2. **The CPU concat fallback copied logical elements as packed blocks.** It
   advanced by `ggml_type_size` for each logical Q8 element instead of iterating
   block storage. This corrupted the expected-value path as well as runtime
   fallback.
3. **Padding converted F16 to Q8 on CPU.** The fork's CUDA copy kernel supports
   F32-to-Q8, not F16-to-Q8. Filling padding in F32 keeps the conversion on CUDA
   and reduces prefill graph splits from 87 to 5, matching F16.
4. **Long-context gathers forced Q8 rows to F16.** Two compressed-attention paths
   cast gathered rows to F16 and then concatenated them with raw Q8 cache rows.
   Multi-chunk prompts eventually hit a type assertion. Preserving the source
   cache type fixes the boundary.

Commit `0379cf4bf` repairs both concat backends, adds the required strided-view
predicate, keeps padding conversion on CUDA, preserves the cache type in both
gathers, and adds Q8 concat regression cases. The CUDA backend test passes
45/45 concat cases, including packed Q8 and strided layer views.

The validated path still sets `LLAMA_ATTN_ROT_DISABLE=1`. Rotation-enabled Q8
was not established during this work. The compose also requires matching Q8 K
and V; mixed cache types are outside the validated contract.

## Warm-up is part of readiness

The speedup requires one synthetic prefill to `context_size - 512` after every
server start. This does not cache a user's future conversation. It prepares the
allocator and CUDA graphs at the deepest configured shape. Without it, the first
request reaching a new depth pays allocation and graph setup and can run at less
than half the steady-state rate.

At the 430,080-token default, warm-up processed 429,568 tokens at 276.4 prefill
tokens/s and took about 26 minutes. A 200K configuration warmed in about eight
minutes.

The profile-local `serve-fast-prefill.sh` supervisor starts llama-server on a
private internal port, performs the warm-up, and only then starts the TCP
forwarder on the public port. The normal launcher therefore cannot report the
profile ready before graph preparation finishes. The registry exports a
2,400-second readiness timeout for this slug.

This profile suits one long-lived coding-agent session. It is a poor fit for
frequent model switching.

## Performance

All measurements below came from the same 4× RTX 3090 rig at its unchanged
230 W/card setting.

### Decode profile (2026-08-16)

Baselines exist for two authorized safety ceilings. Under the prior 210–1550
MHz policy the canonical profile measured 31.88 client-wall tokens/s (CV 0.08%)
and about 37.55 engine decode tokens/s. After Will changed the persistent
policy to 210–1650 MHz (230 W unchanged), a fresh matched run measured:

| Metric (1650 MHz ceiling) | Result |
|---|---:|
| Client wall, 5×512-token fixed-output runs | 33.05 tok/s (CV 0.18%) |
| Engine decode | 39.15 tok/s |
| Engine prefill, 3× ~28K-token fresh prompts | 1,166.6 tok/s |
| Max SM clock observed under load (6,136 samples) | 1650 MHz |

A later 1995 MHz clock-lock experiment measured higher decode throughput, but
it violated server60's hardware-safety boundary and is rejected. Never raise
this host's power limits or clocks for performance work, and do not deploy a
dynamic clock controller. The rejected controller was removed. The pre-existing
safety owner, `gpu-power-limit.service`, was restored and Will then explicitly
changed its RTX 3090 graphics-clock range from 210–1550 to 210–1650 MHz while
retaining the 230 W/card limit. A generation-time audit sampled 180 readings and
observed a maximum SM clock of 1650 MHz. The persistent script and unit SHA-256
values are `da373fbd32fb34c1c63ae8b1c37489ce7af4fb589e071c509dca9f932a6724f9`
and `a734642d9d14cca2139fcf7c99747a0465050546e29b3e59079ad60887552029`.

Nsight Systems 2026.4.1 (attribution only; node tracing more than doubled
request wall time) showed quantized Q8_0/IQ2_XXS/Q2_K matrix-vector kernels as
the dominant device work, rising from about 28.5 shallow to 29.5 aggregate
GPU-ms/token near 100K context, with layer placement balanced across the four
cards as one serial per-token chain. Sparse attention, Lightning Indexer, and
radix selection grew with context but lack the causal budget to close the gap
to vLLM's ~70-token/s path.

An isolated MMVQ microbenchmark
(`llama-cpp/bench-mmvq/`, fork's real dispatch, serving launch env) measured
saturated achieved bandwidth on one RTX 3090 under the 1650 MHz ceiling:

| Kernel (K=4096) | Achieved | NCU attribution |
|---|---:|---|
| Q8_0 matvec, R4096 | 713 GB/s | latency-bound: 47% SM, 20.1 cycles/issue, 56% occupancy |
| Q2_K matvec, R12288 | ~310 GB/s | instruction-bound: 78% SM, 70% occupancy, fp32 accumulation |
| IQ2_XXS matvec, R12288 | 346 GB/s | instruction-bound: 64% SM, 36% occupancy (grid-lookup/sign chain) |

The routed-expert i-quant kernels achieve only ~43–49% of the Q8_0 matvec
bandwidth on identical silicon. Gated decode hypotheses, in priority order:

1. **Q2_K MMVQ integer accumulation** — port the fork's own
   `vec_dot_q2_K_q8_1_impl_mmq` integer-accumulate pattern into the MMVQ path
   behind an env guard. Gate: 78% SM throughput on fp32 FMAs. Falsifier: SM
   throughput does not drop below ~60% or kernel time fails to improve ≥15%.
2. **IQ2_XXS MMVQ instruction reduction** — cut per-byte instructions
   (shared-memory grid table or cheaper sign unpack), possibly raising
   occupancy past the 41.67% theoretical cap. Gate: 64% SM at 36% occupancy
   with only 7.19 cycles/issue. Falsifier: SM% unchanged or bandwidth flat.
3. **Q8_0 rows-per-block overlap** — more output rows per block to overlap
   DRAM latency (56% achieved occupancy, 20 cycles/issue). Bounded by the
   existing `calc_rows_per_block` table; small expected gain.

Estimated combined decode ceiling from measured pool shares: about 6–8%.
Each move is one-variable, correctness-gated (reference-identical dot
products on random blocks, then deterministic generation canaries), and
must preserve context capacity and the Q8 KV contract.

Two arms were implemented as compile-time variants and measured in the
microbench (three A/B repetitions each, co-resident with the idle service):

- **A1 (Q2_K integer-domain micro-optimization) — rejected.** Hoisting the
  scale product into the integer domain and building the m broadcast with one
  `__byte_perm` measured ~288 vs ~285 GB/s on Q2_K K4096 R12288 (within
  noise). Falsifier met: no ≥15% kernel improvement.
- **A3 (Q8_0 rows-per-block 2) — rejected.** Overlapping two output rows per
  block measured ~713 vs 713 GB/s on Q8_0 K4096 R4096 (within noise). The
  latency-bound pool does not benefit from row overlap at these shapes.

A duty-cycle audit during live decode (50 ms sampling, prefill samples
excluded) measured 23.2/23.9/24.8/24.7% mean per-GPU utilization with 24–26%
maxima: each GPU is busy ~6.4 ms per 25.5 ms token — exactly its quarter of
the tight serial layer-split chain. There is no idle inter-GPU time to
reclaim; decode wall time equals the sum of per-GPU kernel time.

### Arm A4: VDR amortization — measured and closed

**Falsifier trap (rejected numbers).** Raising only the
`VDR_*_Q8_1_MMVQ` macros appeared to give +31…+145% kernel bandwidth. The
added output checksum exposed it as computing 25–50% of the work: the MMVQ
kernel strides blocks by `vdr` while the vec_dot implementations are
hard-coded to their structural per-call unit count. Those numbers are
invalid; the macros are documentation of the implementation, not knobs.

**Real rewrite.** `vec_dot_iq2_xxs_q8_1` and `vec_dot_q2_K_q8_1` were
rewritten to genuinely process `VDR` units per call (identical per-unit
arithmetic, fp accumulation across units; VDR at the structural default
reproduces the originals bit-for-bit). Validated by checksum across all ten
bench shapes: untouched types match the stock binary exactly; changed types
match to fp reassociation tolerance (max-abs identical).

Legal VDRs must divide the per-block unit count (IQ2_XXS qi=4: only VDR 4;
Q2_K qi=8: VDR 2 or 4; VDR 8 for IQ2_XXS divides by zero in the launch
mapping and is invalid). Order-swapped A/B on the dominant 6-expert decode
shape (K=4096, R=12288):

| Variant | K4096/R12288 | K4096/R2048 | K2048/R2048 | K1024/R4096 |
|---|---:|---:|---:|---:|
| IQ2_XXS VDR4 | **+16%** (335 GB/s) | −6% | +13% | −32% |
| IQ2_XXS VDR8 | invalid | invalid | invalid | invalid |
| Q2_K VDR2 | ~flat | −7% | flat | −7% |
| Q2_K VDR4 | **+18%** (367 GB/s) | −12% | −6% | −32% |

Q8_0 VDR4 re-measured with correct work: 713 GB/s, exactly flat — the DRAM
floor conclusion stands.

**End-to-end bound.** Serving shape mapping closes this arm. Routed gate/up
(IQ2_XXS) matmuls at K=4096 with 6 active experts (R=12288) — the one shape
VDR4 improves (+16%). But routed down (Q2_K) matmuls at K=2048 (per-expert
`[4096, 2048]`), where VDR4 measured −6% (K2048/R2048) to −32% (K1024);
its +18% K4096 result applies to no serving shape. Net: Q2_K VDR4 is a
serving wash-or-regression, and the IQ2_XXS pool is ~9% of kernel time, so
its +16% bounds the campaign's serving gain at **≈+1.5%** before subtracting
the single-expert R2048 (−6%) slices. That is far below the 10% materiality
threshold, and the remaining mechanisms are bounded below it: Q8_0 at its
DRAM floor, no idle time in the serial chain, no viable speculative path
(DSpark measured net negative), P2P already flat. A serving A/B for the
residual ≈1.5% was judged not worth the fork-delta maintenance and
regression risk; this section is the infeasibility record for the 10%
threshold. Campaign closed with the service unchanged.

The profile also rejects several software shortcuts for this build: CUDA weight
repacking is not available for the dominant GPU tensors; the previous 1/2/4/8
MoE rows-per-block sweep was flat; four-way expert parallelism improved prefill
but reduced decode; row/tensor split cannot reserve the DeepSeek V4 grouped
attention graph; and the existing DSpark speculative path was net-negative.

Raw attribution reports remain on server60 rather than in Git:

- `decode-baseline.nsys-rep`, SHA-256
  `aab5d74d42aaa0614ecbdaa737dd3e0927e99d478fd82aae4c04a5d913f194a1`;
- `decode-deep-100k.nsys-rep`, SHA-256
  `e3d54ba9d4b5e0baf3194b982641bd594c1c07157ea18f649f273ed55d79dcb1`.

### 430K profile

| Probe | Result |
|---|---:|
| Warm-up | 429,568 tokens at 276.4 prefill tokens/s |
| Fresh prompt at 263K | 1,056 prefill tokens/s, exact needle recall |
| Fresh prompt at 395,282 | 913 prefill tokens/s, exact needle recall |
| Minimum observed free VRAM | 794 MiB |

The 794 MiB floor is 44 MiB above the experiment's accepted 750 MiB floor but
below the repo's normal 1,024 MiB production guard. That narrow margin remains
an operational caveat even though this is now the canonical profile. Set
`CTX_SIZE=200000` for the higher-margin fallback.

### Q8 versus F16 at 200K

| Fresh prompt depth | Repaired Q8 prefill | F16 prefill | Q8 difference |
|---:|---:|---:|---:|
| 98K | 1,538 tokens/s | 1,414 tokens/s | +8.8% |
| 130K | 1,475 tokens/s | 1,360 tokens/s | +8.5% |
| 180K | 1,374 tokens/s | 1,334 tokens/s | +3.0% |
| 195K | 1,341 tokens/s | 1,306 tokens/s | +2.7% |

Q8 recovered the full graph path and was 3–9% faster than F16 at depth. The
stock `b10200` Q8 router was much slower on repeated deep prefill; in the
measured agent loop it took 21.8 seconds to first token at 33K, versus about
7.8 seconds on the repaired fork.

### Coding-agent loop

A 15-turn fixture grew from 1,140 to 54,146 prompt tokens. Time to first token
grew from the shallow start to 9.5 seconds at 54K while decode held 33–38
tokens/s.
Across the measured curve, context grew about 40× while TTFT grew 4.2×. The
stock Q8 baseline's measured TTFT grew nearly linearly with context.

## Validation

The clean image built from `0379cf4bf` passed:

- the final catalog launch through `switch.sh --force` pulled the published
  digest, kept its public port closed during warm-up, processed 429,568 tokens
  at 276.28 prefill tokens/s, became ready after 1,628 seconds, and passed `verify-full.sh`
  9/9 on port 8033;
- `verify-full.sh`: 9/9 at 200K and 430K on the pre-publication validation builds;
- `verify-stress.sh`: 8/8 at 200K, including exact recall at 9K, 27K, 55K,
  85K, and the 184K ceiling;
- 430K fast stress gate with a 750 MiB floor: exact recall at 263K and 395,282,
  with 796 MiB free throughout the ceiling ladder;
- `quality-test.sh --medium`: 67/75 pass@1, equal to the stock `b10200` Q8
  baseline;
- full 8-pack thinking-off: 109/150 after replacing an invalid network-blocked
  HermesAgent 0/20 leg with its reachable 14/20 rerun; the same weights and Q8
  cache scored 111/150 on stock `b10200`;
- full 8-pack thinking-on: `low` 121/150 at a 16,384-token output cap, `high`
  121/150 at 65,536, and `max` 123/150 at 65,536; four of 226 `max` API
  responses hit the cap, while `low` and `high` had no cap hits;
- stock `b10200` scored 122/150 at default `low` effort with the same weights
  and Q8 cache; together with the 109/150 fork versus 111/150 stock
  thinking-off comparison, this is the serving/quantization regression check;
- live reasoning-effort rendering: off 11 prompt tokens, low 11, high 90, max
  103;
- `soak-test.sh`: 25/25 responses, zero errors, zero silent-empty responses;
- `bench-agentic.sh`: all 15 turns through 54K;
- Q8 CUDA concat regression: 45/45.

The 430K run used the fast stress mode: the near-ceiling ladder and functional
probes ran, while the redundant 60K/90K intermediate needle section was skipped.
The complete stress gate ran at 200K.

## Quant constraint

The fast MoE capture path is quant-sensitive. The validated Antirez artifact
uses IQ2_XXS for routed-expert gate/up weights and Q2_K for routed-expert down
weights. The locally available Unsloth IQ1_M and role-splice variants contain
IQ1_M expert matrices that failed the MMQ graph-safety check. They are not drop-in
replacements for this profile.

## Rejected paths

- **Minimal graph forward-port to stock `b10200`:** graphs launched but did not
  produce the target gain; incomplete stable topology and scheduler behavior
  kept rebuilding the work.
- **F16-only fork profile:** correct and fast, but Q8 is now equally correct,
  slightly faster, and holds more context.
- **Q8 with concat repair only:** coherent on short prompts but produced 87 graph
  splits and ran at about one-third of F16 speed at depth.
- **Q8 after concat and padding repair only:** fast, but long prompts crossing
  micro-batch boundaries hit a cache-type assertion until the gather fix landed.
- **IQ1_M weights:** incompatible with the validated MMQ capture path.
- **DwarfStar CUDA tensor parallelism:** the audited implementation cannot load
  this GGUF on four 24 GiB cards because three selective weight slabs exceed
  24 GiB after its required 2 GiB runtime reserve. See
  [`DWARFSTAR-CUDA-TP-DEAD-END.md`](DWARFSTAR-CUDA-TP-DEAD-END.md) for the pinned
  source audit, exact tensor accounting, and CPU planner results.

Keep these failures in mind when changing the engine base, cache operations,
quant, or warm-up. Short smoke prompts are insufficient: the bugs appeared only
at depth or across micro-batch boundaries.

# GGUF-TP engine (vLLM-native, production since 2026-08-18)

Since 2026-08-18 the production DeepSeek V4 serving path on server60 is the
**GGUF-TP engine** — a native vLLM tensor-parallel engine that loads the exact
Antirez GGUF bytes (IQ2_XXS/Q2_K/Q8_0, sha256 `ca22ae2f…`) with from-scratch
Ampere kernels. It replaces the canonical llama.cpp profile (which becomes the
validated rollback). Full detail: `vllm/gguf-tp/README.md` (build contract,
MANIFEST, operating contract). This section records the milestone trail
(M0–M9) and why this engine wins.

## Why GGUF-TP

- **Quality:** loads the community-standard Antirez GGUF bit-exact — the same
  weights llama.cpp serves. The earlier WNA16 path served a *requantized*
  artifact and lost quality (DeepSWE SuperJSON gate: ~0.92 partial vs 0.995).
- **Serving surface:** vLLM OpenAI API, native tool calling, reasoning parser,
  prefix cache, 8-way concurrency at full 140K context.
- **Speed:** decode 78.3 tok/s single / 254.0 aggregate at 8 concurrent;
  cache-busted prefill 513.6 tok/s; DeepSWE pilot 2.65× faster wall-clock than
  llama.cpp at equal-or-better reward.

## Milestone trail (evidence: `feat/gguf-tp-engine` worktree)

- **M0** optimized WNA16 FlashMLA + hierarchical all-reduce trace.
- **M1** capacity floor 140–142K accepted; Q8_0/Q2_K/IQ2_XXS L0 oracle bitwise.
- **M2** raw/aligned+DP4A/DwarfStar IQ2_XXS kernels; fused indexed gate/up
  (247 GB/s); grouped DwarfStar gate/up + grouped Q2_K down (1.6811× realistic
  uniform M=256); TP=4 graph slice decode 0.1934 ms/layer (74.13 tok/s
  projection). 34/34 GPU tests + memcheck/racecheck clean.
- **M3** Q2_K down 300.23 GB/s (raw layout kept).
- **M4** full-model TP=4 meta verifier: 1,328 GGUF tensors → 1,180 runtime
  targets, exact per-rank name/element sets; raw Q8_0 load-to-Marlin lifecycle
  11/11; identity caching.
- **M5** TP=4 API at 140K (load 271.9 s; 21.53 GiB/rank; KV 0.81 GiB @
  154,519 tokens); functional gates passed (deterministic gen, auto tool call,
  post-tool continuation, NIAH exact recall @ 119,730); quick quality 27/30;
  decode 76.70, prefill 551.89, agg-2 121.86; zero swap. Measured capacity
  ceiling, not release-safe (71–73 MiB free/GPU).
- **M6** functional gates passed; per-layer oracle preregistered gate FAILED
  (28/43 layers; median cos 0.993, NRMSE 0.119) but final logits PASSED
  (cos 0.9973). Bisection rejected FP16 router storage, forced indexed
  experts, FP32 router compute as single mechanisms → drift is the documented
  class-B accumulation (Q8_0→Marlin scale rounding, DP4A order, FlashMLA-vs-
  llama attention). Open finding + weight-rounding follow-up: TODO-175a7261.
- **M7** (service/watchdog hardening during pilot ops).
- **M8** DeepSWE SuperJSON pilot (one cell): reward **0.9949** (F2P 79/80,
  P2P 116/116), 2,520 s wall vs llama.cpp control 0.9898 / 6,678 s — passed
  by Will's judgment. Harness fix upstreamed (deep-swe-bench `d856c630`).
- **M9** promotion 2026-08-18: seq8@140K verified (see below), Compose profile
  `vllm/compose/multi4/gguf-tp/base.yml` made the production default.

## M9 promotion verification (2026-08-18)

Concurrency sweep on server60 (canonical 3 warm + 5 measured):

| max_num_seqs | max_model_len | KV pool | single decode | aggregate decode |
|---|---|---|---|---|
| 2 | 140,000 | 154,519 | 78.6 | 128.1 |
| 4 | 140,000 | 149,321 | 78.6 | ~141.7 |
| 6 | 125,000 | 147,994 | 78.2 | 167.9 |
| 8 | 125,000 | 141,770 | 78.4 | 253.2 |
| **8** | **140,000** | **151,330** | **78.3** | **254.0** |

Two hard engine gates found while pushing seq8 to 140K:

1. **KV-pool gate:** at `max_num_seqs=8`, `max-num-batched-tokens=256` yields
   a 141,770-token pool (0.75 GiB) — below the 0.77 GiB needed for
   max_model_len 140,000; the engine refuses startup (estimated max length
   137,216). Fix: `max-num-batched-tokens 256 → 192`, freeing 9,560 pool
   tokens (151,330) at ~5% cache-busted prefill cost (540.7 → 513.6 tok/s).
2. **Pre-flight gate:** `gpu-memory-utilization 0.985` fails ("free memory on
   startup less than desired utilization"); 0.98 is the ceiling.

Full-140K single-sequence recall passed (139,565 prompt tokens, unique
prompt); 8×~40K concurrent probes completed with zero preemption; zero swap.
VRAM idle headroom 35–41 MiB/card — capacity-ceiling class; the promote
decision records that explicitly (reopen condition = OOM at/below operating
context).

## Operational amendment (2026-08-18, post-M9)

By operator direction the same day, production switched **max_num_seqs 8→2**
and **max_model_len 140,000→148,000**, returning max-num-batched-tokens to
**256**. KV pool at the new profile: 156,738 tokens (1.06× at 148K); idle
VRAM ~99 MiB/card; zero serving swap. Batched 256 restores full cache-busted
prefill (540.7 tok/s vs 513.6 at 192). The 148K operating point is
fit-gate-confirmed only — long-context recall at the new ceiling was not
re-run by direction. The seq8 arm stays available (254.0 aggregate decode)
with the 192 requirement documented in the compose header.
