# DeepSeek V4 MMVQ decode microbenchmark

`bench-mmvq.c` drives the pinned Whamp/llama.cpp fork's real CUDA MMVQ
dispatch (`mul_mat_vec_q`) at the DeepSeek V4 Flash decode shapes, isolated
from the serving engine. It links statically against the fork's ggml built
with `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, CUDA 12.8.1.

Build (CPU-only; run on the host under the fixed 230 W / 210-1650 MHz
policy, co-resident with the idle serving process):

```bash
docker build -t mmvq-build:base - <<'D'
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends cmake build-essential file && rm -rf /var/lib/apt/lists/*
D
# configure + build ggml from the pinned fork source, then:
g++ bench/bench-mmvq.c -O2 -I ggml/include -I ggml/src -I ggml/src/ggml-cuda -o build/bench-mmvq \
  $(find build -name "libggml-cuda.a") $(find build -name "libggml-cpu.a") $(find build -name "libggml-base.a") \
  -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcuda -lcudart -lcublas -lcublasLt -lnccl
```

Usage: `./bench-mmvq <device> [shape-index]`. Set `DSV4_MMVQ_SMALLK=1` to
match the serving launch mapping for the K=4096 gate/up shape.

Weight blocks are filled from a deterministic LCG with fp16 block scales
pinned to 1.0: these vec_dot kernels have no data-dependent control flow,
and the CPU i-quant/K-quant quantizers abort on degenerate synthetic data.
Each iteration is a graph of 16 independent `mul_mat` nodes on one stream so
per-launch overhead is amortized; 20 warmup + 100 measured graphs.

Nsight Compute filter: `-k 'regex:mul_mat_vec' --launch-skip 320
--launch-count 3` with SpeedOfLight, SchedulerStats, WarpStateStats,
Occupancy sections.
