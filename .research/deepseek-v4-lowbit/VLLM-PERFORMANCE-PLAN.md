# DeepSeek V4 WNA16 server60 performance plan

## Context

The campaign started from a TP=4 correctness configuration at 200K context: eager execution, `max_num_seqs=1`, `max_num_batched_tokens=256`, and SM86 attention fallbacks. It measured about 809 prompt tokens/s and 5.55 decode tokens/s. The matched thermally warm llama.cpp baseline later established the actual promotion threshold at 32.67 decode tokens/s.

The goal was to make matched vLLM single-stream decode exceed llama.cpp on server60, while preserving usable prefill and supporting at most four concurrent requests. This plan applies the gated workflows from `$perform-like-jeff-and-sanjay` and `$nvidia-cuda-performance`; no flag or kernel change becomes a recommendation without its precondition and predicted mediator being measured.

## Outcome

The performance goal passed without kernel changes or Verda spend. The causal discriminator was breakable CUDA graph replay:

- matched eager arm: 4.96 decode tokens/s;
- first graph arm: 60.27 decode tokens/s;
- graph-after-eager repeat: 60.07 decode tokens/s; and
- checksum-pinned final image: 60.82 decode tokens/s over 3 warmups and 5 measured 512-token runs, with 0.1% CV.

The first promoted result was 86% faster than the matched 32.67-token/s llama.cpp baseline at 131,072 tokens. A follow-up residency phase retained that execution path and added one scoped change: materialize DeepSeek V4's two shared FP32 RoPE tables to runtime `max_model_len` while preserving the original YaRN frequency span.

The selected profile now uses TP=4, 215,000-token context, `max_num_seqs=4`, `max_num_batched_tokens=256`, no CPU weight offload, and the existing SM86 attention guards. It measured 60.79 decode tokens/s, 968.97 prompt tokens/s on the cache-busted 8,984-token guardrail, retrieved the exact needle at 204,900 prompt tokens, and passed concurrency 2 and 4 at 65.47 and 89.94 aggregate tokens/s with zero post-warm VRAM growth. The RoPE change removed about 407 MiB of registered storage per rank at 215K.

The measured context frontier is 215K with concurrency 4. A 230K/c4 arm failed with an estimated 215,552-token maximum. Reducing `max_num_seqs` to 2 raised the estimate to 223,488; a 220K/c2 arm fit, but Will selected 215K/c4 because 5K more context did not justify halving supported concurrency.

## Approach

1. Optimize single-stream decode first. Record prefill as a regression guardrail; defer prefill tuning until decode wins.
2. Establish a matched graph-enabled baseline with measured memory accounting. Hard-cap `max_num_seqs` at 4 in every experiment; use 1 to isolate latency, then validate only 2 and 4.
3. Isolate breakable CUDA-graph replay with a one-variable graph-versus-eager A/B before changing scheduler, kernel, or communication code.
4. If graphs do not close the decode gap, capture a system timeline before selecting Humming MoE, sparse MLA/indexer, PCIe collective, or host/launch work.
5. Implement only the largest causally supported move whose recoverable time can materially close the measured gap, behind an SM86/workload guard with a complete fallback.
6. Validate correctness, dispatch, direct mechanism, decode, prefill regression, and end-to-end attribution before keeping it.
7. Use up to the authorized $30 Verda budget only when it shortens a build, oracle, or isolated experiment. Server60 remains authoritative for SM86, four-GPU PCIe, and final performance.

The first hypothesis is conditional but specific: the eager run paid repeated launch/submission cost; enabling and proving the fork's intended breakable CUDA-graph replay should reduce per-token decode step time. Its gate is actual graph capture/replay on the request path; its falsifier is no material reduction in unprofiled ITL despite graph hits.

## Files to modify

Expected boundary; exact source files will be narrowed only after tracing identifies the critical segment:

- `/home/will/projects/club-3090/.worktrees/deepseek-v4-lowbit-vllm/.research/deepseek-v4-lowbit/VLLM-PERFORMANCE-PLAN.md` — durable experiment decisions and evidence.
- `/home/will/projects/club-3090/.worktrees/deepseek-v4-lowbit-vllm/models/deepseek-v4-flash-0731/vllm/compose/multi4/wna16/base.yml` — incubating profile, only after a measured winner exists.
- Existing DeepSeek V4 launch/benchmark helpers under `models/deepseek-v4-flash-0731/vllm/` and `scripts/` — minimally extend instead of creating a parallel harness.
- Vendored patch series under `models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86/` — only if a traced code/kernel hypothesis passes its gate.
- The pinned vLLM module owning the selected critical segment — deferred until tracing.

## Reuse

- Canonical `scripts/bench.sh` protocol: 3 warmups plus 5 measured essay/quicksort runs, engine and wall throughput, and per-card peak VRAM.
- `scripts/verify-full.sh`, `scripts/verify-stress.sh`, and `scripts/soak-test.sh` only at the validation stage they answer; do not repeatedly run the whole stack during diagnosis.
- `scripts/quality-test.sh --scenario/--scenarios-file` for targeted canaries, not blanket `--medium`, `--full`, or `--reasoning` runs during performance iteration.
- Existing guarded server60 runtime/rollback mechanisms. There is no per-round downtime limit, but preserve deterministic recovery and restore llama.cpp at campaign end if vLLM is not promoted.
- Pinned vLLM A100 campaign methodology in `benchmarks/kernels/dsv4_sm80_refutations.md` and `benchmarks/kernels/benchmark_dsv4_sm80.py`, adapted to TP=4/SM86 without transferring its measurements or constants.
- Existing Humming W2 numerical oracle and SM86 cubin/dispatch checks.
- Existing server60 hardware, topology, runtime, artifact, and llama.cpp measurement dossiers.
- Existing fail-closed Verda provisioning/watchdog patterns if the remaining $30 is used; A100 evidence may validate generic integration but cannot establish server60 performance.

## Steps

- [x] Inventory the exact reusable launch, rollback, decode benchmark, and evidence-recording seams. Reuse the repository benchmark and Compose seams; package the winner as a thin checksum-pinned image rather than rebuilding the roughly 40 GB base.
- [x] Write the experiment contract: exact tree/image/artifact, matched llama.cpp and vLLM prompts/sampling/output length, warmup, repetitions, prefix-cache state, clocks/thermals, per-card VRAM, logs, correctness checks, and recovery procedure.
- [x] Re-measure the running llama.cpp single-stream decode baseline. It produced 32.67 decode tokens/s with 0.1% CV.
- [x] Establish a graph-enabled vLLM baseline. The normal 2048-token budget did not fit; the 256-token budget reached readiness at 8K and isolated the graph path without spending the first campaign on 200K capacity.
- [x] Run a matched breakable-graphs-versus-eager decode A/B. Graph replay produced about a 12.2× gain, and graph-after-eager reproduced the result.
- [x] Stop before Nsight Systems: graph-enabled decode already exceeded the promotion threshold, so a timeline could not change the accept/reject decision.
- [x] Stop before an Amdahl kernel budget: no remaining gap to llama.cpp required recovery.
- [x] Implement the first delivery change: bake patches 0005–0007's three production source files into a checksum-pinned thin image without a speculative kernel change. The later patch-0008 image expands the overlay to seven production files.
- [x] Validate attribution with graph → eager → graph. The two graph arms measured 60.27 and 60.07 tokens/s; eager measured 4.96.
- [x] Stop the optimization loop after the attributed graph change exceeded llama.cpp.
- [x] Preserve prefill usability. The first accepted 131K profile measured 964.09 prompt tokens/s on three cache-busted 8,984-token runs.
- [x] Validate concurrency 2 and 4. They sustained 65.19 and 90.23 aggregate tokens/s with zero post-warm VRAM growth on the first profile.
- [x] Run final matched benchmarks and focused checks. The first packaged image measured 60.82 decode tokens/s, returned exact deterministic generation, preserved forced-tool and reasoning parsing, and remained healthy without source overlays.
- [x] Instrument storage-deduplicated model residency and reconcile the major per-rank allocation families.
- [x] Compare the WNA16 runtime with a controlled Antirez IQ2_XXS llama.cpp run at 8K, 131K, and 200K.
- [x] Implement and isolate runtime-bounded RoPE materialization. At 215K it removes about 407 MiB per rank without changing YaRN frequencies or position semantics.
- [x] Re-establish context capacity with one-variable 200K, 215K, 220K/c2, and failed 230K admission arms. Select 215K/c4 by user decision.
- [x] Validate the selected 215K profile with matched decode, prefill, 204,900-token retrieval, concurrency 2/4, `verify-full`, `verify-stress`, restart, zero serving-process swap, and VRAM-stability gates.
- [x] Use no Verda credit; server60 answered every performance and capacity question needed for promotion.

## Signal-focused quality policy

Performance changes should preserve the same model behavior, so broad capability packs are usually the wrong iterative signal.

- **Config-only graph, scheduler, memory, and tracing changes:** compare deterministic greedy token output against the pinned eager baseline on a small fixed corpus. Include the known-sensitive “only code” quicksort prompt, one exact short response, and one tool-call request. Skip broad quality packs when outputs match.
- **MoE, attention, indexer, cache, collective, or fused-kernel changes:** first run the affected operator's numerical oracle at actual SM86/TP shapes, then repeat the deterministic API differential. A kernel timing win without oracle agreement is rejected.
- **Attention/cache changes:** add one short-context and one longest-supported NIAH check because that mechanism can corrupt context behavior without changing easy prompts.
- **Concurrency changes:** compare each request at concurrency 2 and 4 with its isolated deterministic result to detect cross-request state or ordering defects.
- **Escalation:** run only a small pinned set of known-fragile code/agent scenarios if deterministic outputs diverge or the change intentionally alters numerical order. Do not run `--medium`, `--full`, or `--reasoning` by default.
- **Promotion compliance:** the repository-required `quality-test.sh --quick` ran once and remains historical smoke evidence. Do not expand benchlocal for this path. Evaluate model capability with DeepSWE through `~/evals/deep-swe-bench/` before claiming quality parity or broader promotion readiness.

## Verification

- **Success outcome:** under the same server60 prompts, sampling, generated-token window, warm state, and 3-warmup/5-measured protocol, vLLM single-stream decode tokens/s must exceed llama.cpp rather than merely approach it.
- **Correctness:** the signal-focused policy above, Humming/operator numerical oracles when touched, and relevant unit/fallback tests.
- **Build and dispatch:** exact image/tree/artifact hashes, SM86 identity, selected DeepSeek backend, generated cubins where applicable, graph capture/replay, and intended kernel dispatch.
- **Mechanism:** graph hit/fallback rate, launch/API gaps, NCCL exposure, Humming/attention time, rank imbalance, or the selected kernel metric.
- **Kernel:** stable replay-based timing on real server shapes; use Nsight Compute only after the system trace identifies a consequential kernel.
- **Phase:** unprofiled ITL/decode tokens/s as the primary metric; TTFT and prompt tokens/s as guardrails, with warmup, samples, variance, and prefix-cache state recorded.
- **Scale:** concurrency 1, 2, and 4 only; relevant prompt/output lengths and fitting contexts; make no p95/p99 claims from five runs.
- **Capacity/regressions:** per-card peak/idle VRAM, graph pool, KV tokens, admitted context/concurrency, thermals/clocks, prefill usability, errors, and final service state.
- **Promotion:** run only the applicable repository verify/stress/bench/soak gates plus the focused quality policy before changing the incubating profile’s status.

## Decisions

- Decode is the first and decisive target; the approximately 809 prompt tokens/s result is usable and remains a regression guardrail.
- Replacement requires matched vLLM single-stream decode to exceed matched llama.cpp single-stream decode.
- Four concurrent requests is ideal, two is acceptable, and more than four is out of scope.
- Server60 may remain on the experimental runtime for as long as the campaign needs; restore llama.cpp only when recovery is required or the campaign ends without promotion.
- Up to $30 of new Verda spend is authorized for deliberately bounded acceleration, but cloud results cannot replace server60 acceptance.
- Diagnostic runs have no fixed 200K context floor; context may be reduced to expose the normal graph path, then capacity is re-established after decode is competitive.
- The selected capacity/performance point is 215,000 tokens with `max_num_seqs=4`. A fitting 220K/seqs=2 arm is rejected because the extra 5K does not justify halving supported concurrency.

## Execution ledger

- 2026-08-12: matched thermally warm llama.cpp fast-profile baseline on server60: 3 warmups + 5 measured, fixed 512-token code completion, chat endpoint, temperature 0.6, top_p 0.95. Decode mean 32.67 tokens/s, CV 0.1%; wall mean 32.44 tokens/s; TTFT mean 109 ms; 5/5 usable; capture status OK.
- 2026-08-12: graph-enabled TP=4 discriminator reached 60.27 decode tokens/s. The one-variable eager ablation reached 4.96; graph-after-eager repeated at 60.07. This establishes breakable CUDA graph replay as the causal change.
- 2026-08-12: the zero-offload 131,072-token profile measured 60.71 decode tokens/s, 964.09 cache-busted prefill tokens/s at 8,984 tokens, exact NIAH retrieval at 119,895 tokens, and clean concurrency 2/4 at 65.19/90.23 aggregate tokens/s.
- 2026-08-12: tested graph-enabled 200K profiles on the pre-RoPE-fix runtime required CPU weight offload and measured only 12.68–19.54 decode tokens/s. The first accepted profile therefore stopped at 131K and opened a separate memory-residency investigation.
- 2026-08-12: promoted local image `sha256:ed5227673011058a04675b913c8f67b6bb83baba3d85f3b83675e765c51379c7`, source tree `12b87bcd52bb2973685fa8f38b5fc8bbbfe7519c`, over base image `sha256:0e8cc6dc48081e907d553febc8002b1f6d61298454340840f27f18b3a2e66c6c`. At that promotion point, the overlay-free image reproduced 60.82 decode tokens/s with 0.1% CV, no serving-process swap, and zero VRAM growth as Compose project `dsv4-wna16-prod` on port 8034.
- 2026-08-12: repository `verify-full` passed all applicable checks. `verify-stress` passed all eight functional probe classes, including exact recall at 120,476 tokens, but exited 1 because only 118 MiB/card remained at the ceiling rung versus its 1 GiB safety threshold. This triggered the separate GPU-residency audit.
- 2026-08-12: storage-deduplicated instrumentation measured 22,145,468,956 registered bytes per rank before the RoPE change. Routed WNA16 experts and vocabulary tensors were correctly TP-sharded; two model-maximum RoPE tables consumed 512 MiB per rank and were the largest isolated avoidable allocation.
- 2026-08-12: patch 0008 bounded only materialized RoPE rows to runtime context while retaining the model's original YaRN frequency span. The exact image `sha256:0beb1f0cba2e41837f4ba5af01cc5c4686afde4f40ab1df5147a6ad945b0af1f` uses vLLM tree `aeb62948e33074514a742d19c2f9a1a3c2ee3e1f`.
- 2026-08-12: the selected zero-offload 215K/c4 profile reported 233,817 cache tokens and 1.09x one-request concurrency. It measured 60.79 decode and 968.97 prefill tokens/s, recalled the exact needle at 204,900 prompt tokens, and passed short-request concurrency 2/4 at 65.47/89.94 aggregate tokens/s with zero VRAM growth.
- 2026-08-12: `verify-full` passed. `verify-stress` passed every functional class and all ceiling rungs through 197,580 tokens, returning nonzero only on its generic 1 GiB margin policy. The observed minimum reserve was 127 MiB/card; controlled restart and all serving processes remained swap-free.
- 2026-08-12: club-3090 commit `26ae767aa98c14761ac4a69d4f492f418fd29578` published patch 0008 and the canonical 215K/c4 Compose. Server60 cut over to that exact detached checkout as `dsv4-wna16-prod` with restart policy `unless-stopped`. The first attempt intentionally rolled back after an evidence file made the checkout fail its clean-tree checker despite API readiness; the validated service recovered. The corrected cutover passed `verify-full` and a deterministic post-clear request. At the recorded final-state capture, system swap and every serving process were at zero; stable GPU use was 23,986 MiB/card with 141–142 MiB free, unchanged GPU controls, and no rollback timer. A later live check still found every serving process at zero swap.
- Full server evidence is under `/home/will/inference/runtime/deepseek-v4-wna16-sm86/performance-20260812/`, `/home/will/inference/runtime/deepseek-v4-wna16-sm86/final-promotion-20260812/`, `/home/will/inference/runtime/deepseek-v4-wna16-sm86/memory-audit-20260812/`, `/home/will/inference/runtime/deepseek-v4-wna16-sm86/llamacpp-antirez-memory-audit-20260813/`, `/home/will/inference/runtime/deepseek-v4-wna16-sm86/rope-capacity-20260812/`, and `/home/will/inference/runtime/deepseek-v4-wna16-sm86/canonical-promotion-20260812/`.
