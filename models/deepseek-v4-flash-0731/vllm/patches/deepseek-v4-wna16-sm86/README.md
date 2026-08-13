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
5. `0005-fix-forward-layer-to-Humming-MoE-kernel.patch`
   - Passes the production routed-experts layer into the Humming WNA16 factory.
   - Locks down the compressed-tensors setup path that the standalone oracle
     did not previously exercise.
6. `0006-fix-compose-DeepSeek-FP8-with-WNA16-experts.patch`
   - Keeps preserved DeepSeek E4M3/UE8M0 linears on the native grouped-FP8 path.
   - Delegates only routed experts to compressed-tensors WNA16/Humming.
7. `0007-fix-gate-sparse-split-K-decode-by-shared-memory.patch`
   - Keeps split-K sparse decode on CUDA devices that can launch its tile.
   - Routes lower-shared-memory devices such as SM86 to the existing single-pass decode.
8. `0008-fix-bound-DeepSeek-V4-RoPE-cache-to-runtime-context.patch`
   - Keeps DeepSeek V4 YaRN frequencies and scaling based on the original model
     configuration.
   - Bounds the two shared FP32 RoPE cache tables to the served context instead
     of always materializing the full 1,048,576-token model maximum.
   - Requires direct GPU residency, capacity, output, and performance validation
     before promotion.

Patch SHA-256 values:

```text
f88b96897566663411d9d09a41e3f3eec54bd9b958fd34165412a2d288310d2b  0001-feat-support-W2A16-MoE-with-Humming.patch
73c33d6f1aec0d87738d4e1cb51e0bb0bab776c6a2b0c5b670639729d0f8896a  0002-feat-support-mixed-WNA16-MoE-projection-bits.patch
1f8c8c1734f4415b1d490bb7d3dbc290f49c9fb1dfeb0e268cdab728072030aa  0003-test-add-Humming-W2A16-MoE-oracle.patch
3be16754f61170ff2da57a1c64edcd7c524ed6ad9b10c5189d3661e6f55ffc8f  0004-fix-load-hybrid-DeepSeek-FP8-linears.patch
f446a73a37b7715023f05aeec526b714fdadbefa80772268e242218c69efc34e  0005-fix-forward-layer-to-Humming-MoE-kernel.patch
9af88957c5900e741794002907183a324510bcc7ebb7dd60fef22d66cd5ac005  0006-fix-compose-DeepSeek-FP8-with-WNA16-experts.patch
f4dec6b898ec327a06b8bd85841ad9e662eb9be7ab59a6cd3a75f60e4c0bc672  0007-fix-gate-sparse-split-K-decode-by-shared-memory.patch
173dc71a669f1ab7cbffd19256b4eb2dd30329597bf9de54b7f95cec8dc76c52  0008-fix-bound-DeepSeek-V4-RoPE-cache-to-runtime-context.patch
```

The expected final Git tree is
`aeb62948e33074514a742d19c2f9a1a3c2ee3e1f`. Patch 0008 passed its
server60 GPU acceptance gates and is part of the promoted image.

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

## Runtime images

`build-runtime-image.sh` builds the CUDA 13.0 environment over the exact patched
vLLM tree. It rejects dirty or drifted source, pins the Linux/amd64 CUDA 13.0.2
development image by manifest digest, installs Torch 2.13.0+cu130, reuses
vLLM's precompiled native extensions from upstream base
`62195e9784ebec1ece42b88a861734e0702cc2d5`, and verifies
`humming-kernels==0.1.10`.

The promoted server60 image is a thin final layer over a verified runtime
base. `build-final-overlay-image.sh` accepts either the measured server60 base
image
`sha256:0e8cc6dc48081e907d553febc8002b1f6d61298454340840f27f18b3a2e66c6c`
or a freshly built base carrying both the exact vLLM tree
`aeb62948e33074514a742d19c2f9a1a3c2ee3e1f` and runtime-Dockerfile contract
SHA-256 `7d4ab7f124d1ca5fc68facaafec8c55b98683e249cf669a2c102ac8ba6013838`.
It verifies all seven production source files changed by patches 0005–0008 plus
both startup scripts by SHA-256. It then:

- bakes patches 0005–0008 into `/workspace/vllm` instead of mounting source;
- adds a fail-closed model-view materializer;
- checks the immutable artifact config and tensor-index hashes;
- verifies that all 45 indexed shards exist;
- injects `base_quant_method=deepseek_v4_fp8` into a tmpfs runtime view; and
- starts `vllm serve` only after that transformation reproduces config SHA-256
  `891883c0c40b28cbec2c9bca6f5e4a8278824fb42ff32695ba4640ebdee7dc91`.

On server60, where the measured base exists locally:

```bash
./build-final-overlay-image.sh \
  /path/to/patched-vllm \
  club-3090/deepseek-v4-wna16-sm86:aeb62948-rope-cu130
```

On a fresh host, build the full pinned base first and pass it explicitly:

```bash
./build-runtime-image.sh \
  /path/to/patched-vllm \
  club-3090/deepseek-v4-wna16-sm86:runtime-aeb62948-cu130

./build-final-overlay-image.sh \
  /path/to/patched-vllm \
  club-3090/deepseek-v4-wna16-sm86:local-aeb62948-cu130 \
  club-3090/deepseek-v4-wna16-sm86:runtime-aeb62948-cu130
```

The promoted server60 image is
`club-3090/deepseek-v4-wna16-sm86:aeb62948-rope-cu130` at
`sha256:0beb1f0cba2e41837f4ba5af01cc5c4686afde4f40ab1df5147a6ad945b0af1f`.
It is not published to a registry. The direct Compose profile pins that exact
local image ID. A fresh build may have a different manifest ID even when its
source and runtime contracts match; repeat the GPU acceptance gates before
changing the Compose digest. This is not a portable public-image release.

## Launch

The Compose profile is intentionally outside the public c3 registry because the
custom image is local and has no catalog installer:

```bash
MODEL_SNAPSHOT=/path/to/immutable/75d9286c-snapshot \
MODEL_BLOBS=/path/to/models--hampsonw--DeepSeek-V4-Flash-0731-WNA16/blobs \
RUNTIME_CACHE_ROOT=/path/to/runtime-cache \
docker compose \
  --profile authorized-gpu-test \
  -f ../../compose/multi4/wna16/base.yml \
  up --detach
```

Plain `docker compose up` starts nothing. The authorized profile occupies all
four GPUs. The measured defaults are TP=4, 215,000-token context,
`max_num_seqs=4`, `max_num_batched_tokens=256`, breakable CUDA graphs, no CPU
weight offload, a 64 MiB sparse-indexer logits cap, and the SM86 per-query
sparse-prefill fallback.

## GPU evidence

The deferred GPU gates are complete:

1. The A100 oracle passed and generated 13 inspected `sm_80` cubins.
2. The RTX 3090 oracle passed in 78.63 seconds and generated 13 inspected
   `sm_86` cubins.
3. The real 45-shard artifact loaded through native DeepSeek FP8 linears and
   Humming W2/group-128/BF16 routed experts on TP=4.
4. The pre-0008 131K profile returned exact deterministic generation, a
   structured forced tool call, separated reasoning, and an exact needle at
   119,895 tokens.
5. The pre-0008 packaged image measured 60.82 decode tokens/s over 3 warmups
   and 5 measured 512-token runs, versus 32.67 tokens/s for the matched
   llama.cpp baseline.
6. Patch 0008 reduced the two materialized RoPE caches from 512 MiB to about
   105 MiB per rank at 215K while preserving the original YaRN frequency span.
   The selected 215K profile measured 60.79 decode tokens/s and 968.97
   cache-busted prefill tokens/s, retrieved an exact needle at 204,900 tokens,
   and sustained concurrency 2 and 4 at 65.47 and 89.94 aggregate tokens/s
   with zero post-warm VRAM growth or serving-process swap.
7. `verify-full` passed. `verify-stress` passed every functional class and all
   five ceiling rungs through 197,580 tokens; its only nonzero result was the
   repository's llama.cpp-oriented 1 GiB free-VRAM warning. The observed
   minimum reserve was 127 MiB per card.
8. A 230K/`max_num_seqs=4` arm failed admission with an estimated 215,552-token
   maximum. Reducing `max_num_seqs` to 2 moved the estimate to 223,488 but did
   not justify losing concurrency 4; the selected profile therefore stops at
   215,000 tokens.
9. `quality-test.sh --quick` scored tool calling 11/15 and instruction following
   14/15 at pass@1 (25/30 total; 26/30 at pass@3). This is limited evidence, not
   a broad quality claim.

`humming-kernels==0.1.10` remains a pure-Python JIT package. The generated
cubins and runtime dispatch evidence are specific to the pinned software,
artifact, and server60 RTX 3090 environment.
