# DeepSeek V4 WNA16 SM86 vLLM patch series

This private research patch series supports the MTP-free DeepSeek V4 Flash
W2A16 artifact. It applies only to
`haosdent/vllm@12810046c799cbe874967e19b1c0fa134ab7b209` on branch
`dsv4-flash-a100`.

That fork owns the selected SM8x DeepSeek V4 attention, cache, indexer, MHC,
and model-dispatch integration.

## Contents

Apply these patches in filename order:

1. `0001-feat-support-W2A16-MoE-with-Humming.patch`
   - Adapts compressed-tensors WNA16 MoE for Humming W2 and W3.
   - Preserves ordinary checkpoint layout.
   - Widens GPTQ INT2 loader and oracle gates.
   - Adapts concepts from open vLLM PR #48918 to the selected fork. See the
     [upstream tracker](../../../../../docs/UPSTREAM.md). It does not claim
     that the PR merged.
2. `0002-feat-support-mixed-WNA16-MoE-projection-bits.patch`
   - Permits one gate/up (`w13`) schema and a different down (`w2`) width.
   - Requires matching group size and layout.
   - Rejects mixed widths outside Humming.
   - Preserves shared-schema compatibility.
3. `0003-test-add-Humming-W2A16-MoE-oracle.patch`
   - Adds a deterministic CUDA numerical oracle for symmetric W2,
     group 128, and BF16 indexed MoE.
   - Exercises vLLM's real Humming conversion and experts seam.
   - Does not alter production code.
4. `0004-fix-load-hybrid-DeepSeek-FP8-linears.patch`
   - Detects the explicit compressed-tensors FP8 linear fallback.
   - Maps preserved native DeepSeek FP8 scales to compressed-tensors' pre-load
     parameter name without changing ordinary DeepSeek FP8 checkpoints.

Patch SHA-256 values:

```text
f88b96897566663411d9d09a41e3f3eec54bd9b958fd34165412a2d288310d2b  0001-feat-support-W2A16-MoE-with-Humming.patch
73c33d6f1aec0d87738d4e1cb51e0bb0bab776c6a2b0c5b670639729d0f8896a  0002-feat-support-mixed-WNA16-MoE-projection-bits.patch
1f8c8c1734f4415b1d490bb7d3dbc290f49c9fb1dfeb0e268cdab728072030aa  0003-test-add-Humming-W2A16-MoE-oracle.patch
3be16754f61170ff2da57a1c64edcd7c524ed6ad9b10c5189d3661e6f55ffc8f  0004-fix-load-hybrid-DeepSeek-FP8-linears.patch
```

The expected final Git tree is
`7f4c19003f808a28ec5adcb5675468c5d34af97b`.

## Apply

The installer requires a clean target at the exact base revision. It accepts
an already-applied worktree only when its full Git tree matches the expected
result.

```bash
./install.sh /path/to/vllm
```

The installer uses exact `git am`, without fuzzy or three-way application.
Base drift requires review rather than silent acceptance.

## Verified without GPU

- A fresh reconstruction from the vendored patches passes 43 focused
  compressed-tensors and Humming WNA16 CPU tests. Two optional-dependency
  cases skip.
- The CUDA oracle collects and skips on a CPU-only host.
- All applicable vLLM pre-commit hooks pass for the oracle patch.
- CodeGraph reports one changed test file, three affected helper callers, and
  no new file cycles.

## Runtime image

`build-runtime-image.sh` builds the rental-proven Python environment over the
exact patched vLLM tree. It rejects dirty or drifted source, pins the Linux
amd64 CUDA 13.0.2 development image by manifest digest, installs Torch
2.13.0+cu130, reuses vLLM's precompiled native extensions from pinned upstream
base `62195e9784ebec1ece42b88a861734e0702cc2d5`, and verifies
`humming-kernels==0.1.10`.

```bash
./build-runtime-image.sh \
  /path/to/patched-vllm \
  club-3090/deepseek-v4-wna16-sm86:7f4c1900-cu130
```

Building the image uses CPU, disk, and network only. It does not establish
SM86 kernel compilation, numerical correctness, model loading, dispatch, or
performance. Those remain runtime gates.

`run-sm86-oracle.sh` implements the next bounded gate. It requires the literal
`I_AUTHORIZE_SERVER60_GPU_ORACLE` argument, the exact image-tree label, no
existing GPU compute processes, an empty report directory, and compute
capability 8.6. It exposes only GPU 0, runs the deterministic numerical oracle,
and rejects success unless at least one generated Humming cubin reports
`sm_86` under `cuobjdump`. It does not load or serve the DeepSeek artifact.

`run-server60-oracle-with-rollback.sh` wraps that gate for the current server60
service. It requires a separate literal authorization, verifies the healthy
llama.cpp baseline and exact image digest, stops only that Compose service,
bounds the oracle to 20 minutes, and restores the exact Compose project from an
EXIT trap on success, failure, interruption, or timeout. Restoration is not
complete until the original service is healthy on the same image digest. This
wrapper is server60-specific and must not be used on another host unchanged.

## Deferred GPU proof

The patch series is not a runtime-support or performance claim until these
stages pass:

1. On rental A100/SM80, build and import the selected vLLM fork. Run the
   W2/group-128 MoE oracle and inspect the generated cubin. This proves generic
   Humming W2 MoE correctness on the selected software stack, not SM86 support.
2. On server60/SM86, and only after explicit GPU-use authorization, rerun the
   oracle. Verify that NVRTC or NVCC targeted `sm_86`, inspect the generated
   cubin, prove vLLM selected Humming for both expert projections, and exercise
   the real artifact.
3. Measure representative prefill, decode, throughput, concurrency, VRAM, and
   basic quality before comparing with the live llama.cpp service.

`humming-kernels==0.1.10` is a pure-Python JIT package. The project research
plan documents its exact SM86 source path. No packaged cubin exists before
first execution.
