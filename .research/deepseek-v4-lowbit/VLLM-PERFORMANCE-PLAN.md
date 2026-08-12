# DeepSeek V4 WNA16 server60 performance plan

## Context

The campaign started from a TP=4 correctness configuration at 200K context: eager execution, `max_num_seqs=1`, `max_num_batched_tokens=256`, and SM86 attention fallbacks. It measured about 809 prompt tokens/s and 5.55 decode tokens/s. The matched thermally warm llama.cpp baseline later established the actual promotion threshold at 32.67 decode tokens/s.

The goal was to make matched vLLM single-stream decode exceed llama.cpp on server60, while preserving usable prefill and supporting at most four concurrent requests. This plan applies the gated workflows from `$perform-like-jeff-and-sanjay` and `$nvidia-cuda-performance`; no flag or kernel change becomes a recommendation without its precondition and predicted mediator being measured.

## Outcome

The goal passed without kernel changes or Verda spend. The causal discriminator was breakable CUDA graph replay:

- matched eager arm: 4.96 decode tokens/s;
- first graph arm: 60.27 decode tokens/s;
- graph-after-eager repeat: 60.07 decode tokens/s; and
- checksum-pinned final image: 60.82 decode tokens/s over 3 warmups and 5 measured 512-token runs, with 0.1% CV.

The final result is 86% faster than the matched 32.67-token/s llama.cpp baseline. The accepted profile uses TP=4, 131,072-token context, `max_num_seqs=4`, `max_num_batched_tokens=256`, no CPU weight offload, and the existing SM86 attention guards. It measured 964.09 prompt tokens/s on the cache-busted 8,984-token guardrail, retrieved the exact needle at 119,895 tokens, and passed concurrency 2 and 4 at 65.19 and 90.23 aggregate tokens/s with zero post-warm VRAM growth.

The 200K graph profiles fit only with CPU weight offload in the tested runtime and fell to 12.68–19.54 decode tokens/s. Will accepted 131,072 tokens for this goal. The unexplained gap between the 76.8 GiB artifact and about 21.91 GiB of consumed memory per rank is tracked separately as `TODO-3be9650e`; it is not explained by the roughly 0.82 GiB KV cache or 0.12 GiB CUDA graph pool.

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
- [x] Implement the smallest delivery change. Keep the measured runtime behavior and bake the three production source files into a checksum-pinned thin image; do not add a speculative kernel change.
- [x] Validate attribution with graph → eager → graph. The two graph arms measured 60.27 and 60.07 tokens/s; eager measured 4.96.
- [x] Stop the optimization loop after the attributed graph change exceeded llama.cpp.
- [x] Preserve prefill usability. The accepted 131K profile measured 964.09 prompt tokens/s on three cache-busted 8,984-token runs.
- [x] Validate concurrency 2 and 4. They sustained 65.19 and 90.23 aggregate tokens/s with zero post-warm VRAM growth.
- [x] Run final matched benchmarks and focused checks. The packaged image measured 60.82 decode tokens/s, returned exact deterministic generation, preserved forced-tool and reasoning parsing, and remained healthy without source overlays.
- [x] Use no Verda credit; server60 answered every performance question needed for promotion.

## Signal-focused quality policy

Performance changes should preserve the same model behavior, so broad capability packs are usually the wrong iterative signal.

- **Config-only graph, scheduler, memory, and tracing changes:** compare deterministic greedy token output against the pinned eager baseline on a small fixed corpus. Include the known-sensitive “only code” quicksort prompt, one exact short response, and one tool-call request. Skip broad quality packs when outputs match.
- **MoE, attention, indexer, cache, collective, or fused-kernel changes:** first run the affected operator's numerical oracle at actual SM86/TP shapes, then repeat the deterministic API differential. A kernel timing win without oracle agreement is rejected.
- **Attention/cache changes:** add one short-context and one longest-supported NIAH check because that mechanism can corrupt context behavior without changing easy prompts.
- **Concurrency changes:** compare each request at concurrency 2 and 4 with its isolated deterministic result to detect cross-request state or ordering defects.
- **Escalation:** run only a small pinned set of known-fragile code/agent scenarios if deterministic outputs diverge or the change intentionally alters numerical order. Do not run `--medium`, `--full`, or `--reasoning` by default.
- **Promotion compliance:** if the profile is promoted from incubating, run the repository-required `quality-test.sh --quick` once. Treat it as a release gate, not as the mechanism test; rerun only failed scenarios when investigating. A broader pack needs a specific observed risk or capability claim.

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

## Execution ledger

- 2026-08-12: matched thermally warm llama.cpp fast-profile baseline on server60: 3 warmups + 5 measured, fixed 512-token code completion, chat endpoint, temperature 0.6, top_p 0.95. Decode mean 32.67 tokens/s, CV 0.1%; wall mean 32.44 tokens/s; TTFT mean 109 ms; 5/5 usable; capture status OK.
- 2026-08-12: graph-enabled TP=4 discriminator reached 60.27 decode tokens/s. The one-variable eager ablation reached 4.96; graph-after-eager repeated at 60.07. This establishes breakable CUDA graph replay as the causal change.
- 2026-08-12: the zero-offload 131,072-token profile measured 60.71 decode tokens/s, 964.09 cache-busted prefill tokens/s at 8,984 tokens, exact NIAH retrieval at 119,895 tokens, and clean concurrency 2/4 at 65.19/90.23 aggregate tokens/s.
- 2026-08-12: tested graph-enabled 200K profiles required CPU weight offload and measured only 12.68–19.54 decode tokens/s. The accepted 131K profile therefore favors decode performance; 200K capacity is a separate memory-residency investigation.
- 2026-08-12: promoted local image `sha256:ed5227673011058a04675b913c8f67b6bb83baba3d85f3b83675e765c51379c7`, source tree `12b87bcd52bb2973685fa8f38b5fc8bbbfe7519c`, over base image `sha256:0e8cc6dc48081e907d553febc8002b1f6d61298454340840f27f18b3a2e66c6c`. The overlay-free image reproduced 60.82 decode tokens/s with 0.1% CV, no serving-process swap, and zero VRAM growth. It remains healthy as Compose project `dsv4-wna16-prod` on port 8034 with restart policy `unless-stopped`.
- 2026-08-12: repository `verify-full` passed all applicable checks. `verify-stress` passed all eight functional probe classes, including exact recall at 120,476 tokens, but exited 1 because only 118 MiB/card remained at the ceiling rung versus its 1 GiB safety threshold. The profile remains experimental; this headroom failure reinforces the separate GPU-residency audit.
- Full server evidence is under `/home/will/inference/runtime/deepseek-v4-wna16-sm86/performance-20260812/` and `/home/will/inference/runtime/deepseek-v4-wna16-sm86/final-promotion-20260812/`.
