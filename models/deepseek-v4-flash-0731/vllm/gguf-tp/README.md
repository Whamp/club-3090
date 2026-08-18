# DeepSeek-V4-Flash-0731 GGUF-TP engine (vLLM-native, SM86)

Production DeepSeek V4 serving engine on server60's 4× RTX 3090 rig — a
native GGUF tensor-parallel vLLM engine that loads the exact Antirez GGUF
bytes (IQ2_XXS/Q2_K/Q8_0) with from-scratch Ampere kernels. No llama.cpp /
ggml wrapping: weights are read directly from the GGUF file (`--load-format
gguf_dsv4`), quantized operators run as vLLM-native CUDA kernels.

- **Source:** `Whamp/vLLM` branch `incubate/gguf-tp-sm86`, promoted commit
  `3ec20cebe` (tree `82a1def1…`). All engine sources are in that branch;
  nothing in this directory is a fork-vendored copy.
- **Production profile:** `../compose/multi4/gguf-tp/base.yml` (port 8034).
- **Status:** ✅ production default for DeepSeek V4 since 2026-08-18; the
  canonical llama.cpp profile (`models/deepseek-v4-flash-0731/llama-cpp/
  compose/multi4/antirez-iq2-xxs/fast-prefill.yml`) is the validated rollback.

## Why a native GGUF engine

The earlier WNA16 path served a *requantized* safetensors artifact (FP8→WNA16
2-bit), which lost quality (DeepSWE partial reward ~0.92 on the SuperJSON
gate) and bloated prefill. GGUF-TP loads the community-standard Antirez GGUF
bit-exact and matches llama.cpp quality while adding vLLM's serving surface:
OpenAI-compatible API, native tool-calling, reasoning parser, 8-way
concurrency at full 140K context.

Milestone evidence (M0–M9, worktree `feat/gguf-tp-engine` in this repo):
kernel-by-kernel build (IQ2_XXS grouped gate/up 247 GB/s, Q2_K down 300
GB/s), TP=4 graph slice decode 0.1934 ms/layer, 1,328 GGUF tensors → 1,180
runtime targets verified per rank, 140K functional gates (NIAH recall at
119,730 tokens), M6 layer-oracle drift documented, and the M8 SuperJSON
pilot: reward 0.9949 vs llama.cpp 0.9898, 2.65× faster wall-clock.

## Operating contract (promoted 2026-08-18)

| knob | value | why |
|---|---|---|
| `--max-model-len` | 140000 | full-context capability; hard per-seq cap (beyond → 400) |
| `--max-num-seqs` | 8 | aggregate decode 254.0 tok/s (vs 128 at 2, 168 at 6) |
| `--max-num-batched-tokens` | 192 | **required** for 140K at 8 seqs: at 256 the KV pool (141,770) < 140K need and the engine refuses (estimated max 137,216). 192 frees 9,560 pool tokens (151,330). Costs ~5% cache-busted prefill (540.7 → 513.6 tok/s). Do not revert without re-running the capacity gate. |
| `--gpu-memory-utilization` | 0.98 | 0.985+ fails the startup pre-flight (free-memory gate) |
| `--kv-cache-dtype` | fp8_ds_mla | DeepSeek UE8M0-packed MLA cache |
| env `VLLM_HIER_ALL_REDUCE` | `0,1;2,3` | PCIe islands; no NVLink; custom all-reduce disabled (`--disable-custom-all-reduce`) |

Measured (2026-08-18, 3 warm + 5 measured): decode **78.3** single / **254.0**
aggregate @ 8 concurrent; cache-busted prefill **513.6** tok/s; full-140K
recall correct; 0 preemptions/evictions/OOM; zero swap. VRAM idle headroom
35–41 MiB/card at 140K — capacity-ceiling class; reopen condition = OOM at
or below operating context.

## Image build contract

`MANIFEST.json` pins every content hash. `build-image.sh` is the fail-closed
builder (base digest check → 21-file hash verification → extension hash →
layer builds). Image lineage:

```
club-3090/gguf-tp-base:b7766cfe                     (sha256:eb2884fc…)
  + 21 engine files @ 3ec20cebe (byte-identical, verified)
  + _C_stable_libtorch.abi3.so (sha256:9f1315be…)
  → club-3090/deepseek-v4-gguf-tp:741b3abf
  + q8_0_marlin.py @ 3ec20cebe (chunked-repack memory bound)
  → club-3090/deepseek-v4-gguf-tp:3ec20ceb
    digest sha256:f91e8283e7ad116b8664b4a936dba88ebafcb8910a968dce2a3c34420f010adf
```

### Rebuilding the native extension

The 73 MiB `_C_stable_libtorch.abi3.so` is the vLLM stable-ABI extension
built by `setup.py`'s CMakeExtension `vllm._C_stable_libtorch` with
`nvcc -gencode arch=compute_86,code=sm_86`, compiling the full extension
target (~110 objects incl. the `csrc/libtorch_stable/quantization/gguf_dsv4/`
kernels) from the source tree at commit `741b3abf`. Build it the same way
the rig's M2 build did: inside the CUDA 13 devel image,
`python setup.py build_ext` (or the vLLM prebuilt-wheel flow with
`VLLM_USE_PRECOMPILED=1`) and keep the artifact hash-pinned
(`GGUF_TP_EXTENSION=… ./build-image.sh`). The current artifact is stored on
server60 at `/home/will/inference/runtime/gguf-tp-m5-image/`.

## Model view

`materialize-model-view.py` regenerates the derived model view fail-closed:
symlinks the tokenizer/generation files from the WNA16 artifact blobs
(content-addressed git-blob names) and derives `config.json` via three
deterministic edits from the uniform WNA16 artifact config (rev 75d9286c),
hash-pinned to `e973e27b…`. The GGUF itself is pinned separately
(sha256 `ca22ae2f…`, 86.7 GiB, 1,328 tensors).

## Known limits (honest)

- **M6 layer drift:** per-layer oracle vs FP16 reference fails 28/43 layers
  (median cos 0.993, NRMSE 0.119) — a smooth accumulation of class-B
  arithmetic differences (Q8_0→Marlin scale rounding, DP4A reduction order,
  FlashMLA-vs-llama attention), not a bug. Final logits cos 0.9973, and no
  task-level damage was observed (pilot 0.9949). Follow-up + Will's
  weight-rounding idea: TODO-175a7261.
- **8×140K concurrent** is graceful-but-unmeasured: sparse-SWA never
  preempted at 4×~140K against a 149K pool; the 64 MiB sparse-indexer cap
  bounds the active logits map. Single 140K contexts are fully verified.
- **Single-machine repo:** compose mounts are server60 absolute paths.
