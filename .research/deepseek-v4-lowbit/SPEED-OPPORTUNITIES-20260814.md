# DeepSeek V4 SM86 speed opportunity audit

Date: 2026-08-14

Scope: prefill and decode improvements that retain the projection-sensitive
quality artifact, `fp8_ds_mla` cache, Marlin-diagonal `wo_a`, and nearly all of
the promoted 230,144-token KV capacity. KV-cache offloading is outside scope.

## Current measured baseline

The promoted server60 service uses four RTX 3090 GPUs, context 230,144,
`max_num_seqs=2`, `max_num_batched_tokens=256`, and zero CPU offload. Its final
matched measurements were:

- decode: 61.91 tokens/s
- cache-busted 8,984-token prefill: 875.93 tokens/s
- exact recall: 211,031 prompt tokens
- free physical VRAM after long-context validation: about 91–94 MiB per GPU

The service therefore has little room for persistent allocations. Candidate
speedups must preserve the cache format and avoid material new buffers.

## Recommended sequence

### 1. Sparse-prefill query block A/B

The pinned Ampere sparse prefill uses a disabled query-block fallback. A
`BLOCK_M=2` arm changes launch shape without changing persistent state. Measure
it first because it is cheap and affects prefill directly.

Evidence status: source-supported hypothesis; no server60 measurement.

### 2. AppMana native SM86 sparse decode

Pinned source:
`AppMana/forks-flash-mla-ampere-dsv4@7f41a5baa5cf57bfbce06458794b4b05737a162a`.
The repository contains SM80/SM86 sparse MLA decode and prefill kernels,
numerical tests, and shape benchmarks. Its native decode consumes the existing
584-byte `fp8_ds_mla` row. It therefore does not require a persistent-cache
format change.

The source reports RTX A5000/SM86 top-k-512 decode at 22.5 microseconds versus
213.7 microseconds for its Triton comparator, and top-k-1024 at 28.2 versus
409.2 microseconds. These are external microbenchmarks, not server60 end-to-end
results.

The prepared integration is decode-only. Triton remains the default and handles
prefill. The experiment must pass the pinned numerical suite, prove `sm_86`
device code, prove runtime dispatch, and beat a matched serving baseline.

Evidence status: source and external SM86 microbenchmark; server60 dispatch and
end-to-end gain unproven.

### 3. Trace before changing collectives

External 8×A800 evidence found all-reduce on most of the decode critical path,
but that system differs from server60's 2×2 PHB topology. The selected runtime
already contains an opt-in hierarchical all-reduce implementation controlled by
`VLLM_HIER_ALL_REDUCE`.

Capture a warmed decode with Nsight Systems before enabling it. Continue only
if an operator-reviewed timeline assigns at least 10% of the decode critical
path to all-reduce. Then run the BF16 numerical and message-size latency gate
with server60 islands `0,1;2,3` before the serving A/B.

Evidence status: existing runtime code and external A800 result; local critical
path unmeasured.

### 4. Small buffer and chunk A/Bs

Two lower-priority launch-only arms may improve prefill:

- sparse-indexer logits workspace: 64 to 96 MiB
- `max_num_batched_tokens`: 256 to 320

Both can reduce splitting or increase prefill work per chunk, but each may
reduce KV capacity. Measure realized capacity and retain only negligible loss.

Evidence status: causal configuration hypotheses; no server60 measurement.

## Exclusions

- Do not adapt upstream narrow-eager-region merge
  `79c865b838e34f7a98a936771284773819d79c8f` to the current V1 PIECEWISE path.
  Later upstream evidence reports a GSM8K collapse on that configuration.
- Do not enable query-blocked sparse decode by default. Pinned source reports it
  1.2–1.9× slower than the per-query kernel at 200K.
- Do not transfer the A800 hierarchical crossover directly to server60. Measure
  the 2×2 PHB topology.
- Do not use Nsight trace throughput as benchmark data.
- Do not duplicate the separate KV-cache offloading investigation.

## Prepared artifacts

- opt-in Whamp/vLLM FlashMLA commit: `17ca4bad13e08e326a7a84af89676816e80bc1e7`
- logging-only follow-up: `1d6b37c8eb904bb2d1db7ddd05b002157d5e9f26`
- initial KV host-registration fallback fix: `91a39786d48f48efb45fbe3a160d448c783b0131`
- creator-only shared-mmap registration: `b7766cfe4d15d9b68acea43097ceff221e8a739f`
- resulting vLLM tree: `6354125afd1306c9286f734d1c47c23c767d77a9`
- checksum-pinned patches 0011 through 0014
- experiment-only FlashMLA and Nsight image builders
- FlashMLA and hierarchical all-reduce numerical gates
- rollback-safe, identity-approved serving runner
- matched canary, benchmark, swap, and long-context measurement scripts
- trace summary and reviewed-evidence gate

## Server60 outcome — 2026-08-15

The authorized campaign completed with zero-swap matched measurements:

- `BLOCK_M=2`: rejected as noise (+0.19% decode, +0.22% prefill).
- FlashMLA decode: accepted after 17 SM86 numerical cases; 71.04 decode
  tokens/s in the final parent run.
- Hierarchical all-reduce: accepted after the reviewed trace assigned a
  conservative 17.15% of decode wall time to non-overlapped NCCL and the local
  oracle found hierarchy 14.5–21.5% faster across tested sizes.
- Combined winner: 74.98 decode tokens/s, +22.58% over the fresh plain
  baseline. Cache-busted ~9K prefill measured 887.52 tokens/s, −3.23%; this is
  a disclosed tradeoff, not a prefill win.
- `indexer96`: skipped because the trace did not establish its mediator.
- `batched320`: skipped because 58 MiB final-rung headroom failed its 256 MiB
  prerequisite.

The combined profile passed tools, multi-turn agent prompts, coding, long
reasoning, and exact recall through 211,551 tokens. Evidence is in
[`evidence/sm86-speed-20260815/`](evidence/sm86-speed-20260815/).
