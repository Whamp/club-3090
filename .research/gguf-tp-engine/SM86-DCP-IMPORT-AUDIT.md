# SM86 DCP import audit — tomylin890/vllm-sm86-dsv4

Audited 2026-08-21, GPU-free. Verdict: **PROCEED — layered adoption**.
This is the highest-value external find for the DeepSeek V4 stack since
AppMana's FlashMLA fork: an independent SM86 implementation of decode
context parallelism over the same haosdent DSV4 base we run, with
measured 262K context on 24 GB cards and context-flat ~50 tok/s decode.

Study clones (pinned):

- `/home/will/projects/tomylin890-study/vllm-sm86-dsv4` @ `996979edb`
  (branch `dcp-sm86`)
- `/home/will/projects/tomylin890-study/flash-mla-sm86-dsv4` @ `59b1386f`
  (branch `dcp-sm86-patches`)

Both Apache-2.0 (vLLM lineage). Attribution rows owed in docs if we port.

## Lineage and port difficulty

| Tree | Base | Delta |
| --- | --- | --- |
| Their vLLM | upstream `62195e978` + haosdent `f8ea5bb16` (Aug 1) | 103 files, +16,936/−584, Python-only except `csrc/fs_io.cpp` (kv-offload robustness) |
| Our GGUF-TP engine (`incubate/gguf-tp-sm86` @ `6f4f658ab`) | upstream `62195e978` + haosdent `12810046c` (Aug 11) + our 168-file delta | — |

Key facts:

- `f8ea5bb16` and our `12810046c` are **siblings**, not ancestors: haosdent
  evolved his branch between Aug 1 and Aug 11 (94 files, ±20K lines
  drift). Their delta therefore **cannot be replayed** onto our tree;
  file-level semantic port required.
- 26 files are changed by both them and us (attention, compressor,
  indexer, scheduler, model runner, kv_cache_manager, ampere_sparse,
  fused_compress_quant_cache, …). Real conflict surface, but most of our
  changes there are FP4/GGUF additions that are semantically orthogonal.
- Dedup check: **none** of their env flags, the block-zeroing coverage
  fix, or the madvise population exist in our tree. Tier-1 below is all
  genuinely new to us.
- Their flash-mla fork = `AppMana/forks-flash-mla-int` @ `7f41a5b` —
  **our exact integration anchor** — plus 4 commits (+819/−127, 7 files:
  partial decode op, BLOCK_M=16 fused prefill, DCP test). Our FP4 branch
  (`feat/fp4-ds-mla-sm86` @ `81a06aa6`) is also off `7f41a5b`; 6/7 files
  overlap but theirs extend FP8 paths (partial-output + tile size) while
  ours added the FP4 format. Cherry-pick with conflict resolution is
  feasible.

## Tier 1 — DCP-independent wins (port first)

Each is gated and default-off in their tree; all verified absent from
ours.

1. **`fwd_sparse_decode_mla_partial`** (flash-mla). Partial decode op
   emitting `(out, lse)` pre-sink. We currently use the non-partial
   AppMana op; the partial op is also the prerequisite for context-flat
   decode and any future DCP merge. Batch-invariant split logic included
   (their `59b1386f`).
2. **Fused SM86 sparse prefill kernel** + `BLOCK_M=16` instantiation
   (`flash_sparse_mla_prefill_fused_sm80.cu`). Replaces the three-piece
   bf16-dequant-workspace + combine + Triton path with one op per
   chunk/layer. Note: AppMana's README claims gathered-bf16 Triton was
   faster on RTX A5000; tomylin measured otherwise after P6. Requires a
   local matched A/B (our prefill floor is 552 tok/s; current ~876–906).
3. **`VLLM_DSV4_COMPRESSOR_WINDOWED`** (P8 ring buffer). Compressor-state
   reservation becomes constant (`cdiv(W, block)`) instead of linear in
   chunk size — their 858→262 blocks. DCP-independent; mutually
   exclusive with prefix caching; requires the hybrid KV manager.
   Direct KV-capacity lever for our engine at dcp=1.
4. **Block-zeroing coverage fix** (`vllm/v1/worker/utils.py`, +212).
   Per-group segment tables; every attention-family group zeroes its own
   payload at acquisition, fail-closed when coverage is lost. They hit
   Inf/NaN from uninitialized `SlidingWindowMLASpec` state blocks read
   as fp32 — a plausible **latent bug in our tree** (we run the hybrid
   manager with sliding-window groups). Audit exposure before porting.
5. **kv_offload tiering/shared-region robustness** (+809/−187 across 16
   files): `MADV_POPULATE_WRITE` population with read-modify-write
   fallback, shm free-space checks, transient-vs-corrupt block-load
   handling (`fs_io.cpp`). Hardens the CPU KV-offload tier our
   `--kv-offloading-size 16` production default uses — the same
   subsystem where we previously hit `cudaHostRegister`/shm failures.
6. **`VLLM_DSV4_SM86_INDEXER_TILES`** — indexer Triton autotune retuned
   for SM86 instead of A100 defaults.
7. **`VLLM_DSV4_WARMUP`** (default on) — runs the resolved prefill/decode
   kernel family once at engine init; kills first-request JIT stalls.
8. **`VLLM_LONG_PREFILL_THRESHOLD_ADAPTIVE`** (P10) — applies the
   long-prefill cap only when ≥2 prefills are queued; single stream
   keeps full chunk size.
9. **`VLLM_MARLIN_MOE_BLOCK_SIZE_M`** — forced Marlin MoE block size
   (8/16/32/48/64). Relevant to the WNA16 path's `fused_marlin_moe`;
   GGUF-TP dense Marlin does not go through `fused_marlin_moe`, so
   lower priority for us.
10. **SWA store-rereachability horizon fix** (their HEAD `996979e`) —
    projects SWA store reachability to end of prompt.

## Tier 2 — DCP itself (the big capacity lever)

What they proved: compressed-KV groups (C4A, C128A, indexer KV) shard
round-robin in compressed-entry space; SWA KV + fp32 compressor state
are `dcp_exempt` (replicated); global top-k via two stable argsorts
(ties → lower global index); cross-rank merge via LSE with the sink
applied once against the global max; capture-safe by redundant fp32
compute on every rank with ownership-filtered writes (no collectives in
the write path). Decode runs a `FULL_DECODE_ONLY` graph (~70 tok/s under
DCP vs 4–6 eager).

Their four solved blockers map exactly onto what we catalogued when we
found DCP blocked in the haosdent fork: the hybrid-coordinator type
assert, unshardable compressor state, missing LSE, and the indexer's
global top-k (`NotImplementedError` we hit).

Capacity math for our engine: the sharded groups dominate per-token KV
bytes, so dcp=4 approaches ÷4 on the dominant component while weights
and replicated groups are unchanged. Stacks multiplicatively with our
FP4 368-byte rows (vs 584 FP8). Their measured result: 262K context on
24 GB cards at FP8 rows.

Port cost is the largest item: 26 overlapping files, base drift, and
their own development order (P1→P11, each phase gated) is the right
template. Treat as its own milestone after Tier-1 lands, following their
`docs-dsv4/architecture.md` precision rules as the checklist (12 rules,
section 10 of their ARCHITECTURE.md).

## Tier 3 — behavioral / advisory (do not default-adopt)

- **Reasoning-effort prompt rewrite** (`deepseek_v4_encoding.py`): they
  replaced the single max prompt with a low/high/max dict and made "max"
  more aggressive than before. This is a behavior change to prompts, not
  a perf fix — our DSML/reasoning behavior is validated against current
  prompts. Only adopt behind an explicit quality gate if ever.
- `PYTHONOPTIMIZE` refuse-to-start guard: good hygiene, cheap port.
- Their `deploy/verify/` scripts (needle grids, determinism probes,
  concurrency probes, eviction tests) are reusable harness ideas even
  where the code does not port.

## Risks

- Single-developer personal project; every number is from a dual-PCIe-
  switch 8×3090 box. Nothing transfers unmeasured; their own docs
  repeatedly separate static derivation from machine measurement.
- Their base is the *older* haosdent snapshot: during port, dedup each
  fix against `12810046c` (some may exist in evolved form).
- Their prefill kernel claim contradicts AppMana's A5000 measurement —
  resolve locally before adopting.
- The ring buffer's mutual exclusion with prefix caching is structural
  (position-modulo slots are not prefix-addressable). Our production
  profile must choose.

## Recommended sequence

1. **GPU-free now**: port-prep their 4 flash-mla commits onto a branch
   off our FP4 lineage (resolve the 6-file conflicts); audit our
   block-zeroing exposure in `utils.py`; dedup-pass their kv_offload
   changes against our tree.
2. **First GPU window**: one-variable A/Bs on the GGUF-TP engine —
   partial-decode op, fused prefill, indexer tiles — matched benches,
   correctness canaries each.
3. **Capacity arm**: `VLLM_DSV4_COMPRESSOR_WINDOWED` A/B (KV tokens
   gained vs prefill cost; we give up nothing we use — no prefix caching
   in the production profile).
4. **DCP milestone**: schedule after Tier-1; follow their P1→P11 order
   with our gates (numerical oracles, CUDA-Graph, sanitizer, NIAH,
   matched perf).

## Integration record (2026-08-21, GPU-free port-prep)

Their four flash-mla P9 commits are now merged onto our FP4 lineage:

- Branch `feat/dcp-partial-fp4` in
  `/home/will/projects/flash-mla-ampere-dsv4/.worktrees/dcp-partial-import`,
  pushed to `Whamp/forks-flash-mla-int`. Tip `a5337e2`; history:
  our FP4 commit `81a06aa` + their `828a35a..59b1386` replayed.
- Merge design: unified prefill kernel template
  `<int kBlockM, Sparse_mla_cache_format>` — their PfGeom BLOCK_M-generic
  geometry (reproduces the M=32 mapping instruction-for-instruction) with
  our FP8/INT8/FP4 staging and in-kernel dequant grafted as format
  branches. `SmemFP4<kBlockM>` added alongside their templated smem
  structs. Dispatcher: FP8 follows their narrow-tile chooser (TP=4 shards
  H to 16 → exactly their BLOCK_M=16 case); INT8 keeps both widths as
  they shipped; **FP4 pins to BLOCK_M=32 until GPU-validated at 16**.
- Decode: their partial epilogue (`kPartial` template, LSE store,
  -1e30 sentinel) auto-merged; combined paths unchanged via
  `lse_ptr == nullptr` dispatch. The mma-prefill fast path now excludes
  partial mode. Their new `mha_fwd_sparse_decode_mla_partial` host
  function sets `cache_format = FP8_DS_MLA` explicitly for our unified
  params struct.
- Validation state: Python interface compiles; no stale `int8_cache`
  refs; conflict-free tree. **No CUDA compile or GPU execution yet** —
  no local nvcc, and server60 disk/GPUs are constrained. The kernel
  merge is structural, not validated: first GPU window must run their
  267-line DCP partial test plus our 47-test native-format suite before
  any claim.

Next GPU-free items from Tier 1: audit our block-zeroing exposure in
`vllm/v1/worker/utils.py`, dedup-pass their kv_offload changes against
our tree, and scope the vLLM-side DCP plumbing port.

## Block-zeroing exposure CONFIRMED in our tree (2026-08-21)

`vllm/v1/core/single_type_kv_cache_manager.py:86` gates zeroing on
`type(kv_cache_spec) in (FullAttentionSpec, TQFullAttentionSpec,
MLAAttentionSpec, HiddenStateCacheSpec)` — an exact-type list that
**excludes `SlidingWindowMLASpec`**, i.e. both our SWA KV group and the
fp32 compressor-state groups. New blocks in those groups are never
zeroed; the worker-side `zero_block_ids` never sees them.

Why production has not visibly broken: sequential first-fill writes
cover the compressor's lookback reads within one request, so stale bytes
surface only on reuse patterns — preemption/retraction re-admission,
prefix-cache-hit resumes, or async-scheduling admission windows — some
of which we do exercise (`--kv-offloading-size 16`, seq2 agent
workloads). This matches tomylin's Inf/NaN incident under caching +
trimming. Porting their per-group segment-table coverage fix
(`vllm/v1/worker/utils.py` +212, plus manager/coordinator threading) is
Tier-1 and should land before any DCP or ring-buffer work.

## Step 1 COMPLETE: block-zeroing coverage port (2026-08-21)

Whamp/vllm `incubate/gguf-tp-sm86` commit `d193d6aa0` (rebased onto the
FP4 merge `81593507f`), ported from tomylin `996979edb`'s zeroing work:

- Scheduler gate `SingleTypeKVCacheManager._record_new_block_ids`:
  exact-type tuple -> `isinstance(kv_cache_spec, AttentionSpec)`.
  SlidingWindowSpec/SlidingWindowMLASpec groups (SWA KV + fp32
  compressor state) now record newly allocated block ids; Mamba stays
  excluded.
- Zeroer `KVBlockZeroer.__init__`: same widened gate; global data_ptr
  dedup replaced with per-`(group, address)` dedup (the packed DSv4 slab
  can alias group base addresses); payload extent switched from
  product-of-trailing-dims + dense-interior assert to a stride span
  asserted `0 < payload <= block step` (fail-closed at init); boot
  census log line reports per-group segment counts.
- Async-load skip path (`_skip_zero_block_ids`) already per-group and
  now covers sliding-window groups correctly.

Evidence: 3 new CPU zeroer coverage tests + 1 new CUDA-gated packed-slab
zeroing test (tests/v1/worker/test_kv_block_zeroer.py), 2 new manager
recording tests (tests/v1/core/test_single_type_kv_cache_manager.py);
locally 88 passed / 9 CUDA-skipped across zeroer + manager +
kv_cache_utils + scheduler suites; Ruff check/format clean;
`git diff --check` clean. GPU functional validation of the widened
zeroing happens with the next server60 window (step 3).
