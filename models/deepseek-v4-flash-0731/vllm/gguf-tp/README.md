# DeepSeek-V4-Flash-0731 GGUF-TP engine (vLLM-native, SM86)

Production DeepSeek V4 serving engine on server60's 4× RTX 3090 rig — a
native GGUF tensor-parallel vLLM engine that loads the exact Antirez GGUF
bytes (IQ2_XXS/Q2_K/Q8_0) with from-scratch Ampere kernels. No llama.cpp /
ggml wrapping: weights are read directly from the GGUF file (`--load-format
gguf_dsv4`), quantized operators run as vLLM-native CUDA kernels.

- **Source:** `Whamp/vLLM` branch `incubate/gguf-tp-sm86`, promoted commit
  `3ec20cebe` (tree `82a1def1…`). All engine sources are in that branch;
  nothing in this directory is a fork-vendored copy.
- **Production profile:** `../compose/multi4/gguf-tp/base.yml` (port 8034,
  `fp8_ds_mla`). The validated opt-in `fp4.yml` uses `fp4_ds_mla`, and the
  experimental `fp4-indexer.yml` also compresses the sparse-indexer cache.
  Neither changes the production default.
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
| `--max-model-len` | 148000 | full-context capability; hard per-seq cap (beyond → 400). 140,000 → 148,000 on 2026-08-18 (operator direction); fit-gate-confirmed only at the new ceiling |
| `--max-num-seqs` | 2 | operator-chosen 2026-08-18; aggregate 128.1 tok/s at 2. Raising to 8 gives 254.0 tok/s but forces batched 192 (see below) |
| `--max-num-batched-tokens` | 256 | default again after seq8→2: restores full cache-busted prefill (540.7 tok/s). **Only drop to 192 if max_num_seqs is raised to 8** (at 256 the pool 141,770 < 140K need; engine refuses, estimated max 137,216) |
| `--gpu-memory-utilization` | 0.98 | 0.985+ fails the startup pre-flight (free-memory gate) |
| `--kv-cache-dtype` | fp8_ds_mla | DeepSeek UE8M0-packed MLA cache; `fp4.yml` is the validated opt-in |
| env `VLLM_HIER_ALL_REDUCE` | `0,1;2,3` | PCIe islands; no NVLink; custom all-reduce disabled (`--disable-custom-all-reduce`) |

Measured (2026-08-18, 3 warm + 5 measured, at 140K): decode **78.6** single /
**128.1** aggregate @ 2 concurrent; cache-busted prefill **540.7** tok/s;
full-140K recall correct; 0 preemptions/evictions/OOM; zero swap. seq8@140K
arm: 254.0 aggregate decode (batched 192, prefill 513.6). Pool at the
current profile: 156,738 tokens (1.06× at 148K). VRAM idle headroom ~99
MiB/card; under load at 140K it was 35–41 MiB/card — capacity-ceiling
class; reopen condition = OOM at or below operating context.

## Optional FP4 DS-MLA cache

`compose/multi4/gguf-tp/fp4.yml` adds a native `fp4_ds_mla` cache without
changing the GGUF weights, attention math after cache dequantization, FP8
indexer cache, Q8 KV fallback, or production `base.yml` profile.

Each 512-value MLA row stores 448 NoPE values as packed E2M1 with fourteen
UE8M0 group-32 scales, followed by the original 64 BF16 RoPE values. The
physical row is 368 bytes: 224 packed NoPE bytes, 128 RoPE bytes, and a
16-byte scale tail. The FP8 row remains byte-for-byte 584 bytes.

Matched server60 results (2026-08-20, 4× RTX 3090, zero serving-process swap):

| result | `fp8_ds_mla` | `fp4_ds_mla` | FP4 delta |
|---|---:|---:|---:|
| cache tokens in the 0.8 GiB pool | 156,373 | 180,039 | +15.1% |
| narrative decode | 79.84 tok/s | 80.36 tok/s | +0.7% |
| code decode | 79.82 tok/s | 80.37 tok/s | +0.7% |
| concurrency-2 aggregate | 126.02 tok/s | 127.27 tok/s | +1.0% |
| cache-busted prefill, 10K | 541.79 tok/s | 524.87 tok/s | -3.1% |
| cache-busted prefill, 93K | 518.82 tok/s | 495.79 tok/s | -4.4% |
| quick quality | 27/30 | 27/30 | identical failures |

The FP4 path passed independent E2M1/UE8M0 writer/reader oracles, native
FlashMLA decode and prefill parity, deterministic CUDA Graph replay, SM86
cubin inspection, Compute Sanitizer memcheck/racecheck, API/tool/reasoning
checks, and exact NIAH retrieval at 136K. It remains opt-in because only
31 MiB/card remained during the 136K stress ladder, below the normal 1 GiB
release guard. The evidence and decision are under
`.research/gguf-tp-q4-kv/`.

`FP4-MANIFEST.json` and `build-fp4-kv-image.sh` pin the thin-image inputs:
Whamp/vLLM `633815f68`, Whamp/forks-flash-mla-int `81a06aa6`, the SM86 stable
extension, the FlashMLA wheel, all 14 runtime overlay files, and the final
image digest.

## Experimental MXFP4 sparse-indexer cache

`compose/multi4/gguf-tp/fp4-indexer.yml` keeps the FP4 main MLA cache and
compresses the 21 ratio-4 sparse-indexer caches from 132-byte FP8 rows to
68-byte E2M1/UE8M0 rows. It is an explicit capacity experiment, not a default.

At 200K configured context it reports 199,409 KV tokens and passed exact NIAH
retrieval through 195,812 prompt tokens. The trade is material: versus the
FP8-indexer FP4 profile, decode and concurrency-2 throughput fall about 4%,
10K prefill falls about 5%, and 90K prefill falls about 30%. BenchLocal quick
pass@3 remains 27/30. Only 25-26 MiB VRAM per card remained under near-ceiling
work, so this is a functional ceiling below the normal 1 GiB release margin.

`FP4-INDEXER-MANIFEST.json` and `build-fp4-indexer-image.sh` pin the reviewed
Whamp/vLLM commit `ccd463e6d`, tree `de72d166…`, all 11 production overlay
files, the FP4 base-image digest, and final image digest. Detailed allocation,
kernel, sanitizer, quality, long-context, and performance evidence is in the
Whamp/vLLM branch's
`benchmarks/kernels/deepseek_v4/fp4_indexer_sm86/RESULTS.md`.

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
