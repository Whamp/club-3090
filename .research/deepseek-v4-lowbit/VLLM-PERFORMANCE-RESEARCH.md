# DeepSeek V4 WNA16 on server60: vLLM performance research

Research date: 2026-08-12
Runtime tree: `/home/will/projects/vllm/.worktrees/deepseek-v4-wna16-sm86`
Promoted vLLM tree: `aeb62948e33074514a742d19c2f9a1a3c2ee3e1f`
Canonical experiment: [`Whamp/vllm#1`](https://github.com/Whamp/vllm/pull/1)
Ampere base: `haosdent/vllm@12810046c799cbe874967e19b1c0fa134ab7b209`

## Executive conclusion

The first server60 benchmark was a valid **correctness and fit result**, but it was not a valid steady-state vLLM performance baseline. The subsequent campaign confirmed that conclusion: breakable CUDA graphs raised matched decode from 4.96 tokens/s in eager mode to about 60 tokens/s, and the first packaged 131K image repeated 60.82 tokens/s versus the matched llama.cpp baseline of 32.67.

A follow-up residency audit found one avoidable allocation: two shared FP32 RoPE tables were materialized for the model's 1,048,576-token maximum even when the runtime served a shorter context. Bounding only those tables to runtime `max_model_len`, while preserving the original YaRN frequency span, removed about 407 MiB per rank at the selected 215K context. The resulting zero-offload TP=4 profile serves 215,000 tokens with `max_num_seqs=4`, measures 60.79 decode and 968.97 cache-busted prefill tokens/s, and retrieves an exact needle at 204,900 prompt tokens.

Two settings changed the execution regime materially:

1. `--enforce-eager` disabled both CUDA graphs and compilation.
2. `--max-num-batched-tokens 256` split a 9K prefill into at least 36 scheduler chunks before any additional indexer chunking.

The pinned fork has a DeepSeek-V4-specific execution mode that the first run never exercised: it disables Torch/Inductor compilation but keeps CUDA graphs through the fork's **breakable CUDA graph** implementation. In that mode, unsupported attention/indexer operations run eagerly between captured graph segments. This distinction matters: “compile mode NONE” does not mean “no CUDA graphs” for this model.

The research also resolves the earlier KV-capacity discrepancy. In this fork, async scheduling reserves two in-flight batches. At `max_num_batched_tokens=256`, the hybrid cache admission model reserves 512 in-flight tokens; at 1024 it reserves 2048. Reproducing the allocator with the real 43-layer cache schedule gives approximately 0.995 GiB versus 1.606 GiB for one 200K request. The earlier 1 GiB success and 1.61 GiB failure are therefore consistent.

The likely performance problems divide into two separate classes:

- **Prefill:** the 256-token scheduler ceiling, indexer chunking, and the SM86-forced per-query sparse-attention path.
- **Decode:** loss of CUDA graph replay, PCIe tensor-parallel collectives, batch-one Humming W2 MoE, and the SM86 single-pass sparse-decode fallback.

Source reading cannot rank those decode costs. A gated measurement must first compare eager and graph execution, then trace the remaining step wall. The A100 campaign does not answer that question because it used 8× A100/NVSwitch, DSpark speculative decoding, and Marlin MXFP4 experts rather than 4× RTX 3090/PCIe and Humming W2 experts.

## What “default vLLM” means in this fork

### Scheduler defaults are not safe defaults for this experiment

For an OpenAI API server on an RTX 3090, this tree defaults to:

- `max_num_batched_tokens=2048`
- `max_num_seqs=256`

The hardware-dependent defaults are set in `vllm/engine/arg_utils.py:2508-2556`; non-70-GiB hardware takes the 2048/256 API-server branch. Chunked prefill is enabled when supported, and requests larger than the token budget are split automatically (`vllm/config/scheduler.py:70-80`).

Therefore:

- Removing the 256-token override returns the prefill budget to 2048, not an unbounded prefill.
- Removing the `max_num_seqs` override would permit 256 resident sequences and is inappropriate on this 4×24-GiB machine.
- `max_num_seqs=1` is not merely a concurrency policy. It also sizes persistent request buffers and bounds CUDA graph capture.

Official documentation describes the same scheduler tradeoff: smaller token budgets favor inter-token latency, while larger budgets favor TTFT; V1 chunks prefills automatically when they exceed the budget. The docs' generic `>8192` throughput suggestion is not directly applicable to this 76.8-GiB model on four 24-GiB cards. [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)

### `max_num_batched_tokens` is also a memory and cache-admission control

The model runner stores both scheduler limits directly as persistent maxima:

- `max_num_tokens = max_num_batched_tokens`
- `max_num_reqs = max_num_seqs`

See `vllm/v1/worker/gpu_model_runner.py:499-506`.

The scheduler config also includes `max_num_batched_tokens` in its graph/config hash because it affects static buffers and shape-dependent compilation (`vllm/config/scheduler.py:207-216`).

More importantly for DeepSeek V4's heterogeneous cache, async scheduling permits two concurrent batches at PP=1 (`vllm/config/vllm.py:539-550`). The cache admission model therefore uses:

```text
max_in_flight_tokens = 2 × max_num_batched_tokens
```

See `vllm/config/vllm.py:552-561`. Sliding-window and chunked-local cache specs add that in-flight reserve to their retained window before calculating required blocks (`vllm/v1/kv_cache_interface.py:518-546,587-618`).

This is why increasing the prefill budget can increase the minimum KV pool even when the maximum request length is unchanged.

## DeepSeek V4's actual compilation and CUDA graph path

### `--enforce-eager` disables both systems

The config path is explicit:

```text
enforce_eager -> CompilationMode.NONE + CUDAGraphMode.NONE
```

See `vllm/config/vllm.py:1193-1199`. Official vLLM documentation likewise describes `--enforce-eager` as a development/debugging mode that sacrifices steady-state decode performance by skipping compilation and CUDA graph capture. [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)

### Without `--enforce-eager`, DeepSeek V4 uses breakable CUDA graphs

DeepSeek V4 does not follow ordinary O2 Torch/Inductor execution in this fork. If the user has not explicitly set `VLLM_USE_BREAKABLE_CUDAGRAPH`, the config auto-enables it for `DeepseekV4ForCausalLM` (`vllm/config/vllm.py:1208-1234`). It then disables the normal Torch/Inductor compilation pipeline while leaving CUDA graph mode intact (`vllm/config/vllm.py:1236-1241`).

The breakable implementation performs runtime stream capture around eager breaks rather than using Torch FX graph splitting. Attention and other decorated custom operations end the current graph segment, run eagerly, then resume capture (`vllm/compilation/breakable_cudagraph.py:3-20,59-115`).

Consequences:

- A startup log showing `CompilationMode.NONE` is expected and does not prove eager-only execution.
- The relevant proof is the selected `cudagraph_mode`, capture descriptors, actual graph-pool memory, and runtime dispatch.
- For this model, enabling Torch/Inductor by forcing `VLLM_USE_BREAKABLE_CUDAGRAPH=0` is not the default optimization path.

The fork's A100 campaign measured Torch/Inductor as a decode regression and retained breakable graphs. That result is useful directionally because it concerns the same model graph, but it is not a server60 acceptance result. See `benchmarks/kernels/dsv4_sm80_refutations.md:20-23` in the pinned runtime tree.

### `max_num_seqs=1` tightly bounds capture sizes

If capture sizes are not supplied manually, this fork sets:

```text
max_cudagraph_capture_size = min(2 × max_num_seqs, 512)
```

See `vllm/config/compilation.py:692-706`. Therefore:

- `max_num_seqs=1` yields capture sizes up to 2.
- `max_num_seqs=4` yields capture sizes up to 8.
- The API default of 256 allows sizes up to 512.

The generated size list is not itself the graph count. The dispatcher creates descriptors for piecewise and, where supported, full decode modes (`vllm/v1/cudagraph_dispatcher.py:166-231`), while breakable capture may create multiple graph segments inside one descriptor.

The startup memory profiler explicitly counts descriptors, profiles the first two in each runtime mode, estimates shared and per-graph memory, and overlays FULL and PIECEWISE pools because they do not replay concurrently (`vllm/v1/worker/gpu_model_runner.py:6637-6803`). The actual graph pool is then logged after capture (`vllm/v1/worker/gpu_worker.py:715-733`).

This means graph memory should be measured rather than inferred from the capture-size list. With `max_num_seqs=1`, the feared hundreds-of-sizes configuration is excluded by construction.

Official vLLM documentation describes `FULL_AND_PIECEWISE` as the normal high-performance mode and explains that full decode graphs can coexist with piecewise handling of graph-incompatible operations. [CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) and [torch.compile integration](https://docs.vllm.ai/en/latest/design/torch_compile/)

## Memory accounting

### `gpu_memory_utilization` is a total executor budget

The automatic path profiles a max-token forward, measures non-KV memory, optionally estimates CUDA graph memory, and computes:

```text
available KV = requested executor memory
             - profiled non-KV memory
             - applied CUDA graph estimate
```

See `vllm/v1/worker/gpu_worker.py:460-548`. The runtime later reports consumed memory, peak activation memory, actual CUDA graph memory, and suggested explicit KV sizes (`vllm/v1/worker/gpu_worker.py:735-789`).

Official documentation defines `gpu_memory_utilization` as the fraction reserved for the model executor, not as a KV-only percentage. The current public-doc default is 0.92; this pinned fork's effective behavior must be read from its own config and startup log. [Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### Explicit KV memory changes the profiling path

When `kv_cache_memory_bytes` is provided, the worker still runs a max-token profile forward to compile/JIT the model, but it skips automatic memory profiling and CUDA graph-memory estimation. It also ignores `gpu_memory_utilization` for KV sizing (`vllm/v1/worker/gpu_worker.py:474-496`).

Therefore automatic and explicit-KV boots answer different questions:

- **Automatic KV:** What does this execution configuration naturally leave for cache after its measured startup profile?
- **Explicit KV:** Can this known cache size coexist with the runtime, without asking the profiler to predict it?

The first discovery run should use automatic KV if the goal is to understand the natural memory split. A later explicit-KV run can reproduce a known allocation, but should not be treated as equivalent profiling evidence.

### The earlier 1.0-versus-1.61-GiB result is resolved

The apparent contradiction came from comparing different scheduler budgets. Reproduction with the pinned cache classes and the real 43-layer DeepSeek V4 schedule gives:

| `max_num_batched_tokens` | Async in-flight reserve | Minimum pool for one 200K request |
| ---: | ---: | ---: |
| 256 | 512 tokens | ~0.9952 GiB |
| 1024 | 2048 tokens | ~1.6065 GiB |
| 2048 | 4096 tokens | ~2.4214 GiB |

This is an admission-model result, not raw history storage. The additional memory is driven by recycling safety for sliding/chunked cache families while scheduler batches remain in flight.

The earlier manual 1-GiB run used the 256-token budget and reported 210,826 tokens. The failed automatic run that asked for about 1.61 GiB used a 1024-token budget. They agree with the source model.

## Ampere/SM86 attention behavior

### Backend selection is forced

On any SM8x GPU, DeepSeek V4 selects `TRITON_MLA_SPARSE_DSV4`; another explicitly requested backend is rejected (`vllm/models/deepseek_v4/nvidia/model.py:790-817`). The Ampere class inherits the ROCm-named implementation because the kernels and metadata builders are portable Triton/Torch, with software FP8 encode/decode below SM89 (`vllm/models/deepseek_v4/ampere/ampere_sparse.py:3-30`).

Thus FlashInfer or FlashMLA tuning advice for ordinary MLA does not apply to this tested SM86 path.

### The A100 prefill optimization is unlaunchable on RTX 3090

For ratio-128 layers, the fork defaults `VLLM_SPARSE_DENSE_QUERY_BLOCK=-1` to the measured `BLOCK_M=8` path. The source records that its query tile needs 204,800 bytes of shared memory (`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:2713-2727`). RTX 3090 exposes a 101,376-byte per-block opt-in limit, so the server60 run must disable this path unless a smaller valid kernel is implemented.

`VLLM_SPARSE_DENSE_QUERY_BLOCK=0` is therefore a correctness requirement for the current SM86 tree, not a tuning preference. It forces the older per-query prefill implementation and removes an A100-measured 2.81× kernel-level win for the 20 ratio-128 layers. That makes sparse prefill a credible contributor to server60's gap even after increasing the scheduler chunk size.

### SM86 also loses split-K sparse decode

The fork's split-K sparse-decode path requires 157,696 bytes of per-block shared memory. The local SM86 guard routes devices below that limit to single-pass decode (`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:2803-2827`).

The source explains why split-K exists: at batch one, the single-pass grid occupies only a few SMs. The fallback is correct but likely underfills a 3090. Its actual contribution must be measured; the A100 campaign's split-K timings do not apply after dispatch changes.

The separate blocked decode path is not an alternative: it is default-off because it measured 1.2–1.9× slower on A100 (`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:2766-2800`).

### Indexer logits budget is a chunking control

`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` defaults to 512 MiB in this fork (`vllm/envs.py:60`). The indexer uses it to split prefill work into chunks (`vllm/v1/attention/backends/mla/indexer.py:1071-1082`).

The earlier 64-MiB override reduced workspace pressure but may have added indexer chunks and launches. It should be treated as a one-variable memory/performance experiment, not carried forward silently and not removed blindly before confirming startup headroom.

## Tensor-parallel communication on server60

`--disable-custom-all-reduce` selects NCCL/PyNCCL instead of vLLM's custom all-reduce (`vllm/config/parallel.py:205-206`). This is required by the server60 PCIe-only topology policy. PyNCCL exists specifically to make NCCL callable from CUDA graph capture (`vllm/distributed/device_communicators/pynccl_wrapper.py:3-22`), so disabling custom all-reduce does not inherently disable graph replay.

Communication remains a serious decode hypothesis because TP=4 imposes repeated collectives across PCIe. However, source reading does not establish that it dominates the 5.55 tok/s result.

The A100 campaign measured 86 decode all-reduces per step and found launch/barrier cost dominant there, but that machine used 8× A100/NVSwitch. The same campaign explicitly skipped an A6000 fork's island-aware all-reduce because its own topology was NVSwitch (`benchmarks/kernels/dsv4_sm80_refutations.md:768-786`). Those findings cannot be transferred numerically to server60.

The correct discriminator is a timeline showing collective duration, inter-rank skew, and gaps around the collectives. Nsight Compute should follow only after a timeline identifies a kernel rather than a host/communication gap.

## What transfers from the A100 campaign—and what does not

The pinned base includes an unusually detailed measured refutation record: `benchmarks/kernels/dsv4_sm80_refutations.md`. Its final A100 configuration reached approximately 450 ms cold TTFT at 8K and a 12.1-ms decode step on 8× A100 with DSpark.

Useful transferable findings:

- Breakable CUDA graphs are the intended DeepSeek V4 path.
- Torch/Inductor was not beneficial for that custom-op-heavy graph.
- Prefill and decode need separate analysis.
- Kernel microbenchmarks must use CUDA graph replay for small decode kernels.
- Kernel microseconds cannot be summed without accounting for streams and overlap.
- Capture-size holes and eager fallbacks can cause order-of-magnitude step changes.
- Profiler traces can heavily tax launch-dense eager paths; untraced A/Bs determine magnitude.

Non-transferable performance conclusions:

- A100 shared-memory-qualified kernels versus RTX 3090 fallbacks.
- NVSwitch collective costs versus four-card PCIe.
- TP=8 shapes versus TP=4 shapes.
- DSpark speculative step/acceptance behavior versus one-token non-speculative decode.
- Marlin MXFP4 routed experts versus Humming W2 routed experts.
- A100's 108 SMs, bandwidth, cache, and tensor-core rates versus RTX 3090 SM86.

The campaign's benchmark harness is still valuable as a methodology template. It times small kernels through CUDA graph replay and keeps operating-point geometry explicit (`benchmarks/kernels/benchmark_dsv4_sm80.py`). Its hard-coded TP=8/A100 constants must not be used as server60 measurements.

## Evidence-backed diagnosis plan

This is a measurement order, not a final production configuration.

### Gate 1: establish the natural startup state

Keep `max_num_seqs=1` and the two SM86 correctness constraints:

- custom all-reduce disabled on PCIe-only server60;
- `VLLM_SPARSE_DENSE_QUERY_BLOCK=0` until a valid SM86 tile exists.

Use automatic KV sizing and the normal DeepSeek breakable-graph path. Record, rather than infer:

- resolved compilation mode and CUDA graph mode;
- capture descriptors and number of graph segments;
- estimated and actual CUDA graph pool memory;
- model/consumed, peak activation, and available KV memory;
- cache token capacity and 200K maximum concurrency;
- selected attention backend;
- per-card startup peak and steady-state VRAM.

A shorter context can be used as a boot discriminator if automatic admission rejects 200K, but context length should not be blamed before the log identifies the memory term.

### Gate 2: isolate graph value on decode

Run matched single-request decode measurements with only graph execution changed:

- breakable CUDA graphs;
- `--enforce-eager`.

Use the same cache size, scheduler budget, prompt, output length, clocks, and warmed state. Measure decode step wall or token-attributed ITL over enough output tokens; do not infer decode from a short end-to-end request.

If graphs recover most of the gap, profile only the residual. If they do not, graph tuning is not the primary lever.

### Gate 3: isolate scheduler and indexer chunking on prefill

Use a fixed prompt and change one limit at a time:

1. scheduler token budget;
2. sparse-indexer logits budget.

Record scheduler chunk count, indexer sub-chunk count, TTFT, prefill tok/s, peak activation memory, and steady-state VRAM. This distinguishes launch multiplication from kernel throughput.

Do not compare a 256-token arm and a 2048-token arm without accounting for the cache admission increase described above.

### Gate 4: obtain a timeline before kernel tuning

Add Nsight Systems to the reproducible image. Capture prefill and decode separately, with CUDA API, kernels, NCCL, and NVTX where available. The trace should answer:

- Is decode launch-bound outside CUDA graphs?
- What fraction of the step is NCCL/PyNCCL?
- Is one rank consistently late, or is waiting absorbed at barriers?
- What fraction is Humming MoE?
- What fraction is sparse MLA/indexer?
- Are there unexpected eager fallbacks or graph-key misses?

Use `ncu` only on the kernel family that owns a material wall-time pool. The current image's `ncu` can inspect kernels but cannot identify CPU launch gaps or end-to-end collective scheduling by itself.

### Gate 5: compare engines only after the vLLM baseline is fair

Compare against llama.cpp with matched:

- model/prompt and sampling where possible;
- prompt token count and generated token count;
- one active request for latency, then explicit concurrency rungs for throughput;
- warm/cold prefix-cache state;
- context target and per-card peak VRAM.

The original 809 prefill tok/s and 5.55 decode tok/s remain valid measurements of the conservative eager configuration. They are not the replacement verdict.

## Measured verdict

The campaign proved the hard functional points and the performance gate:

- SM86 Humming correctness and generated `sm_86` cubins;
- exact W2 dispatch, TP=4 load, native DeepSeek FP8 linears, and quantized MLA cache;
- matched eager decode at 4.96 tokens/s;
- graph-enabled decode at 60.27 tokens/s and a graph-after-eager repeat at 60.07;
- final checksum-pinned image decode at 60.82 tokens/s versus llama.cpp at 32.67;
- 964.09 cache-busted prefill tokens/s on an 8,984-token prompt;
- exact NIAH retrieval at 119,895 tokens; and
- clean concurrency 2/4 at 65.19/90.23 aggregate tokens/s.

The source prediction about `--enforce-eager` was correct. Graph replay closed the decode gap, so Nsight and kernel changes were not needed for the promotion decision. The original profile stopped at 131,072 tokens because its full-length RoPE allocation hid the zero-offload 200K path.

## Residency and capacity result

### The artifact/runtime gap

The immutable safetensors tensor payload is 82,431,357,148 bytes, or 76.770184 GiB. Its 54 repository files total 82,464,249,582 bytes, or 76.800817 GiB. File headers and non-weight assets therefore explain only 0.030633 GiB; they do not explain runtime residency.

A storage-deduplicated rank-0 inventory measured the pre-patch runtime after Humming postprocessing. Replacing only its two 256 MiB full-length RoPE storages with the deterministic 215,000-row allocation yields 21,718,678,044 registered bytes per rank, or 20.227095 GiB. Across four ranks that is 80.908381 GiB, 4.138197 GiB above the raw artifact payload.

The per-rank difference from one quarter of the artifact is:

| Family | Artifact quarter | Runtime rank | Difference |
| --- | ---: | ---: | ---: |
| Routed WNA16 experts | 17,544.13 MiB | 17,544.00 MiB | -0.13 MiB |
| Embedding + LM head | 505.00 MiB | 505.00 MiB | 0.00 MiB |
| Ordinary attention | 1,096.60 MiB | 1,304.93 MiB | +208.33 MiB |
| Compressor + sparse indexer, aliases counted once | 190.99 MiB | 766.58 MiB | +575.59 MiB |
| Shared experts | 258.02 MiB | 262.03 MiB | +4.02 MiB |
| Hyperconnection | 32.31 MiB | 129.26 MiB | +96.94 MiB |
| Router | 25.95 MiB | 94.92 MiB | +68.97 MiB |
| Norms | 0.17 MiB | 0.68 MiB | +0.51 MiB |
| Runtime-only RoPE tables at 215K | 0.00 MiB | 104.98 MiB | +104.98 MiB |
| Other runtime FFN state | 0.00 MiB | 0.17 MiB | +0.17 MiB |

The routed experts and vocabulary matrices are correctly TP-sharded. The remaining registered-storage gap is deliberate replication, transformed layouts, and runtime tables concentrated in attention, compressor/indexer, hyperconnection, router, and RoPE state. Humming postprocessing does not retain a large packed-weight duplicate: it reduced registered storage by about 365.6 MiB in the instrumented run.

At 215K, startup reported 21.64 GiB per rank for weights plus non-Torch state, 0.35 GiB peak activation, 0.12 GiB CUDA graphs, and 1.09 GiB KV cache. These named buckets total 23.20 GiB. Warm NVML residency was 23,986 MiB, or 23.424 GiB, leaving about 0.224 GiB for allocator and runtime slack. Within the 21.64 GiB first bucket, registered model tensors account for 20.227 GiB and the remaining 1.413 GiB covers unregistered model/runtime allocations and non-Torch CUDA/NCCL/JIT state. The earlier instrumented 131K warm state independently decomposed the same classes into 20.625 GiB registered tensors, 1.815 GiB unregistered Torch allocations, 0.429 GiB Torch allocator cache, and 0.552 GiB non-Torch state. The exact values move with RoPE and KV sizing, but all major residency families are identified and measured rather than inferred from artifact size.

### Matched llama.cpp comparison

The controlled comparison used the exact 86,720,111,488-byte Antirez IQ2_XXS GGUF, pinned llama.cpp image, fixed layer split `1,1,0.95,1.05`, Q8_0 K/V cache, batch 8192, ubatch 384, parallelism 1, and context as the only changed input.

At 200K:

- llama.cpp used 85.158 GiB aggregate GPU memory after full-context prefill and the matched decode benchmark. Its logs account for 79.773 GiB of GPU model buffers, 0.986 GiB CPU-mapped model data, 0.681 GiB compressed KV, 0.011 GiB ordinary KV, 0.020 GiB recurrent state, and 1.475 GiB compute buffers. The remaining 3.197 GiB of GPU residency is CUDA-pool and other unlogged engine state.
- vLLM used 93.742 GiB aggregate GPU memory after its 200K stress run, 8.584 GiB more than llama.cpp despite a raw artifact that is 3.994 GiB smaller. Its reconstructed registered tensors total 80.880 GiB at 200K; the rest is the larger vLLM KV pool, graph memory, unregistered Torch buffers, allocator retention, and CUDA/NCCL/JIT/workspace state.

The engines make different tradeoffs. llama.cpp leaves 1,010 MiB of model data CPU-mapped and uses an uneven layer split. vLLM TP=4 shards routed experts and vocabulary tensors evenly but replicates several non-expert families, maintains a larger heterogeneous cache pool, and carries graph/distributed-runtime state. File size alone was therefore the wrong comparison boundary.

### Cache-format and context decision

The active cache is `fp8_ds_mla`: one inseparable UE8M0 block-scaled 584-byte MLA row per stored token, comprising 448 NoPE bytes, 128 RoPE bytes, and 8 scale bytes. This is the current Q8-equivalent storage path. The pinned SM86 backend requires it.

Q4 and asymmetric Q8-K/Q4-V are not launch flags for this layout. DeepSeek V4 stores a latent MLA row rather than separable K and V arrays, and the pinned backend has no compatible Q4 writer, reader, sparse-indexer contract, or decode kernel. Decode context parallelism could reduce replicated history, but compressed-cache writes and compressor-state storage still reject DCP. Those options failed the support gate and were not implemented as speculative memory formats.

The retained one-variable change is runtime-bounded RoPE materialization. It removed about 407 MiB per rank at 215K and enabled zero-offload long context without changing attention math, cache encoding, or YaRN frequencies.

The measured frontier is:

- 215K, `max_num_seqs=4`: ready with 233,817 cache tokens and 1.09x one-request concurrency; 60.79 decode tok/s; 968.97 prefill tok/s; exact retrieval at 204,900 prompt tokens; concurrency 2/4 at 65.47/89.94 aggregate tok/s; zero post-warm VRAM growth and zero serving-process swap after controlled restart.
- 230K, `max_num_seqs=4`: rejected; 1.11 GiB KV required versus 1.05 GiB available, estimated maximum 215,552.
- 230K, `max_num_seqs=2`: rejected; estimated maximum 223,488.
- 220K, `max_num_seqs=2`: fits, but Will selected 215K/c4 because 5K more context does not justify halving supported concurrency.

`verify-full` passed. `verify-stress` passed every functional class and all ceiling rungs through 197,580 tokens; it returned nonzero only because its generic 1 GiB free-VRAM policy saw 127 MiB per card. The vLLM acceptance contract instead required repeatable startup, zero serving-process swap, no post-warm growth or allocation failure under decode, prefill, near-ceiling, and concurrency workloads, correctness, and the 45.6/723 performance floors. The selected profile passed those gates. The 127 MiB minimum reserve remains a reported operational risk, not an automatic rejection.

Club-3090 commit `26ae767aa98c14761ac4a69d4f492f418fd29578` publishes the exact patch and Compose contract. Server60 runs it as `dsv4-wna16-prod` with restart policy `unless-stopped`. A fail-closed rollback was exercised during canonical cutover: the first container reached API readiness, but the clean-tree checker rejected an evidence file written inside the detached checkout and restored the validated service. The corrected cutover then passed startup, `verify-full`, and a deterministic request after clearing cold host swap. At the recorded final-state capture, system swap and every serving process were at zero; the request caused no VRAM growth or logged runtime errors, GPU controls were unchanged, the KV pool held 233,817 tokens, and 141–142 MiB remained free per card. A later live check still found every serving process at zero swap. The final-state record is `/home/will/inference/runtime/deepseek-v4-wna16-sm86/canonical-promotion-20260812/FINAL-STATE.txt` (SHA-256 `6c7344498727f867116d1161da0aae36f86f822551d488f35d73afd4dd376bfb`).

## Sources

### Pinned source

- Canonical experiment: [`Whamp/vllm#1`](https://github.com/Whamp/vllm/pull/1)
- Canonical branch: [`incubate/deepseek-v4-wna16-sm86`](https://github.com/Whamp/vllm/tree/incubate/deepseek-v4-wna16-sm86)
- Runtime tree: `/home/will/projects/vllm/.worktrees/deepseek-v4-wna16-sm86`
- Published deployment mirror: [`Whamp/club-3090@26ae767a`](https://github.com/Whamp/club-3090/commit/26ae767aa98c14761ac4a69d4f492f418fd29578)
- Final vLLM tree: `aeb62948e33074514a742d19c2f9a1a3c2ee3e1f`
- A100 campaign base: [`haosdent/vllm@12810046c`](https://github.com/haosdent/vllm/commit/12810046c799cbe874967e19b1c0fa134ab7b209)
- A100 campaign record: `benchmarks/kernels/dsv4_sm80_refutations.md`
- A100 benchmark harness: `benchmarks/kernels/benchmark_dsv4_sm80.py`

### Official vLLM documentation

- [Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [CUDA Graphs design](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [torch.compile integration](https://docs.vllm.ai/en/latest/design/torch_compile/)

The official docs describe upstream concepts and current public defaults. Fork-specific behavior and exact defaults in this note are grounded in the pinned runtime source above.
