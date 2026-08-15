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
- resulting vLLM tree: `1260b4aba8fb5bf92e6632882326eb2b800ff3df`
- checksum-pinned patches 0011 and 0012
- experiment-only FlashMLA and Nsight image builders
- FlashMLA and hierarchical all-reduce numerical gates
- rollback-safe, identity-approved serving runner
- matched canary, benchmark, swap, and long-context measurement scripts
- trace summary and reviewed-evidence gate

No server60 experiment was run while this preparation was produced.
