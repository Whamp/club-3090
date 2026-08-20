# GGUF-TP decode fusion trace

Status: trace gate complete. The preregistered launch/dependency-latency premise is falsified. Do not implement the original F1+F2 package as proposed.

## Decision

The TP=4 decoder-layer graph is kernel and collective dominated, not launch-gap dominated.

| Stable replay metric | Median | Range |
| --- | ---: | ---: |
| First graph node to final graph node | 195.746 µs | 194.147–197.794 µs |
| GPU busy-time union | 194.561 µs | 192.867–196.545 µs |
| Internal idle time | 1.184 µs | 1.056–1.378 µs |
| Gap before the next graph replay | 4.576 µs | 4.512–4.928 µs |
| First-node to next first-node period | 200.322 µs | 198.723–202.338 µs |

The preregistration expected 60–100 µs/layer of launch or dependency latency. The trace finds about 1.2 µs inside the graph and 4.6 µs between graph replays. CUDA Graph node launch gaps inside a replay are generally 0.06–0.10 µs.

The first replay on each rank is excluded from these statistics because the profiling-start barrier perturbs its first collective. The remaining sample contains 49 stable replays on each of four GPUs, 196 layer replays total.

## Where the layer time goes

Median node-duration sums across stable replays:

| Group | Time | Share of 195.746 µs graph span |
| --- | ---: | ---: |
| Six dense Q8 Marlin projections | 84.416 µs | 43.1% |
| Two hierarchical all-reduces | 45.729 µs | 23.4% |
| Indexed IQ2 gate/up and Q2 down | 38.304 µs | 19.6% |
| Original F1+F2 removable standalone nodes | 10.432 µs | 5.3% |
| Shared-expert SwiGLU pointwise chain | 10.880 µs | 5.6% |
| Final shared-convert, routed-add, and BF16-cast chain | 5.760 µs | 2.9% |

The original F1+F2 upper bound is the complete duration of the five standalone nodes it proposed to absorb: routed input quantization, weighted SwiGLU requantization, top-k reduction, routed/shared add, and final BF16 cast. Their median sum is 10.432 µs. Real savings must be lower because the fused producer kernels still have to perform that arithmetic.

At the whole-model level, even deleting all 10.432 µs from all 43 layers would save only 0.449 ms/token before accounting for epilogue cost. Against the measured 13.34 ms/token baseline used in M2, that is a 3.4% optimistic ceiling. The original package's complexity is not justified by the trace.

## Re-derived fusion target

Two pointwise chains are better first targets because they occupy more measured time and do not require rewriting the quantized matvecs:

1. **Shared SwiGLU:** fuse BF16→FP32 conversion, gate clamp, up clamp, SiLU, multiply, and BF16 cast. Six graph nodes currently total 10.880 µs.
2. **Final add/cast:** fuse shared BF16→FP32 conversion, routed/shared FP32 add, and BF16 cast. Three graph nodes currently total 5.760 µs.

Together these chains consume 16.640 µs/layer before replacement. Two small fused kernels should retain the current operation order and dtypes while avoiding seven launches and intermediate tensor traffic. This is a measured target, not a speedup claim. Numerical parity and a fresh layer-slice trace remain mandatory.

Dense Marlin and hierarchical collectives are larger targets, but neither exposes a comparably bounded fusion seam. The indexed IQ2/Q2 matvecs remain worthwhile kernel-optimization targets, but the trace does not support rewriting both merely to eliminate launch latency.

## Method

The trace used the exact M2 TP=4 graph-captured layer-slice harness at Whamp/vLLM `0ef05fe53` and capture image `sha256:5fab88440740a6033bcacda473ffaeed7a4f4e386d494b516432487f0df09729` on server60. The only harness change adds `--profile-decode`, which brackets indexed-decode graph replays with `cudaProfilerStart/Stop` and an NVTX range.

Configuration:

- four RTX 3090 GPUs;
- `VLLM_HIER_ALL_REDUCE=0,1;2,3`;
- custom all-reduce disabled by the existing PCIe-only dispatch;
- 10 warmup graph replays;
- 50 captured decode replays per rank;
- CUDA Graph trace granularity `node`;
- Nsight Systems 2025.3.1;
- prefill runs after capture and is not part of the trace.

The profiled run reported 0.2028–0.2042 ms/layer across ranks. Profiler overhead explains the difference from M2's unprofiled five-run mean of 0.193402 ms/layer.

## Evidence

`evidence/fusion-trace-20260820/` contains:

- `tp4-decode-layer-slice.nsys-rep`: compact four-process trace;
- `tp4-decode-layer-slice.sqlite`: queryable event export;
- `analysis.json`: asserted 23-node replay reconstruction and timing summary;
- `analyze_trace.py`: deterministic SQLite analyzer;
- `nsys-stats.csv`: standard CUDA kernel/API/NVTX summaries;
- rank-local benchmark results and complete smoke/profile logs;
- the exact benchmark scripts used for capture.

The 2.5 GB target-side `.qdstrm` was deleted after successful import to the compact report and successful SQLite export. It contained no additional durable information.

## Next step

Implement the two re-derived pointwise fusions test-first. Compare each fused kernel against the current PyTorch sequence, then replace only the indexed-decode layer-slice call sites. Re-run the TP=4 trace before considering any quantized-matvec epilogue fusion.
