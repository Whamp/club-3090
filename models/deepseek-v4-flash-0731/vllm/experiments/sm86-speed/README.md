# DeepSeek V4 SM86 speed experiments

Status: prepared off-server; no server60 experiment has run.

These experiments target prefill or decode speed without changing the
`fp8_ds_mla` cache layout or the promoted 230,144-token capacity profile. Each
serving arm changes one variable. The runner leaves the production container
unchanged, starts a separate non-restarting container, and restores production
on every exit path.

## Fixed baseline

All arms keep these measured-profile settings:

- model: `deepseek-v4-flash-0731-wna16-quality-12035985`
- context: 230,144 tokens
- tensor parallelism: 4
- `max_num_seqs=2`
- `max_num_batched_tokens=256`, except the `batched320` arm
- FP8 Marlin-diagonal `wo_a`
- zero CPU offload
- breakable CUDA graphs enabled

Measure each benchmark arm with three warmups and five 512-token decode runs,
then three cache-busted 8,984-token prefill runs. The measurement also requires
deterministic, automatic-tool, and post-tool canaries, zero serving-process
swap, and a startup proof for arm-specific dispatch.

Treat these as regression floors, not targets:

- decode: 46.17 tokens/s
- cache-busted prefill: 552.546 tokens/s

A candidate must also beat a fresh matched control from the same experiment
image. Do not promote a small gain within run-to-run noise.

## Arms

| Arm | One changed variable | Predicted mediator | Keep only if |
| --- | --- | --- | --- |
| `baseline` | none | matched control for the experiment image | canaries and baseline protocol pass |
| `prefill-block2` | `VLLM_SPARSE_DENSE_QUERY_BLOCK=2` | fewer sparse-prefill query blocks than the disabled fallback | prefill rises without decode or capacity regression |
| `flashmla-decode` | `VLLM_DSV4_FLASH_MLA_DECODE=1` | AppMana SM86 native sparse MLA replaces Triton only during decode | numerical gate passes and end-to-end decode rises |
| `hier-allreduce` | `VLLM_HIER_ALL_REDUCE=0,1;2,3` | island-local reduce plus cross-island transfer replaces PyNCCL | a reviewed Nsight trace shows all-reduce on the critical path, the numerical gate passes, and decode rises |
| `flashmla-hier` | compose the independently proven FlashMLA and hierarchical changes | lower sparse-MLA and collective decode time | both numerical gates pass, both dispatches are present, and gains compose without release-gate regression |
| `indexer96` | sparse-indexer logits workspace 64 MiB to 96 MiB | fewer query-dimension splits | prefill or decode rises with negligible KV loss |
| `batched320` | `max_num_batched_tokens=256` to `320` | larger prefill chunks | prefill rises while decode and KV capacity remain acceptable |
| `trace-baseline` | observational trace with `KV_OFFLOADING_SIZE=0.001` | attributes warmed plain-stack decode time without an unused 16 GiB host tier competing with Nsight | evidence only; never use trace throughput as benchmark data |
| `trace-flashmla-hier` | observational trace of the proven combined arm, also with the minimal KV-offload tier | re-anchors sparse-MLA, collective, MoE, and host-gap shares after FlashMLA + hierarchical all-reduce | both numerical gates pass; evidence only, never benchmarked |

The upstream narrow-eager-region change is excluded. Its current V1 PIECEWISE
configuration has a documented correctness failure. Query-blocked sparse decode
is also excluded because the pinned implementation reports it slower at 200K.

## Prerequisites

1. Preserve the current production container and image identity.
2. Build the experiment image with
   `patches/deepseek-v4-sm86-speed-experiments/build-flash-mla-decode-image.sh`.
3. Build the Nsight image only for `trace-baseline` or
   `trace-flashmla-hier`.
4. Set the model snapshot, Hub blobs, runtime-cache, speed-image, and production
   image identity explicitly.
5. Run `--dry-run` first. Record its `plan_sha256`.
6. Approve that exact identity through `SERVER60_SPEED_PLAN_SHA256` before the
   actual run.

Do not execute these steps while another server60 campaign owns the GPUs.

## FlashMLA acceptance gate

Run this before the `flashmla-decode` serving arm:

```bash
models/deepseek-v4-flash-0731/vllm/experiments/sm86-speed/run-flash-mla-sm86-gate.sh \
  IMAGE OUTPUT_DIRECTORY
```

The gate checks the pinned AppMana source archive, requires an RTX 3090/SM86,
verifies `sm_86` device code with `cuobjdump`, and runs the upstream sparse MLA
numerical suite. The `flash_mla` 2.0.0 wheel includes the AppMana MIT license.

## Matched serving arm

Render a plan without touching Docker:

```bash
models/deepseek-v4-flash-0731/vllm/experiments/sm86-speed/run-speed-arm-with-rollback.sh \
  --dry-run ARM EVIDENCE_DIRECTORY -- \
  models/deepseek-v4-flash-0731/vllm/experiments/sm86-speed/normalize-swap-then-measure.sh \
  models/deepseek-v4-flash-0731/vllm/experiments/sm86-speed/measure-speed-arm.sh \
  EVIDENCE_DIRECTORY/measurement
```

After reviewing the manifest, rerun the same command without `--dry-run` and
set `SERVER60_SPEED_PLAN_SHA256` to the printed hash. When rollback targets a
production service whose warmup exceeds the default 15 minutes, set
`PRODUCTION_HEALTH_WAIT_ATTEMPTS` before **both** dry-run and execution (checks
are 5 seconds each; 480 allows 40 minutes). The value is bound into the plan
hash.

The normalization wrapper checks that available RAM exceeds used swap by at
least 8 GiB, resets host swap with passwordless `sudo`, verifies zero swap for
every serving process, and only then starts measurement. The plan hash binds
the exact wrapper command and a digest of the complete harness directory.

Run `baseline` immediately before each candidate arm. Compare results from the
same thermal window and experiment image:

```bash
cd tools/deepseek-v4-lowbit
uv run deepseek-v4-compare-speed-results \
  BASELINE_MEASUREMENT_DIRECTORY CANDIDATE_MEASUREMENT_DIRECTORY \
  --output COMPARISON.json
```

The comparison enforces performance floors and zero worker swap, but it does
not declare a speed winner. Review variance, dispatch proof, realized KV
capacity, and the predicted mediator before keeping an arm.

## Nsight attribution and hierarchical all-reduce

1. Run `trace-baseline` with the Nsight image and
   `capture-nsys-decode.sh` as the measurement command. This observational arm
   records `KV_OFFLOADING_SIZE=0.001`: the minimum practical tier keeps the
   connector valid, its 256-token request cannot use host KV, and it avoids an
   unused 16 GiB tier making Nsight pressure host swap.
2. Summarize the `.nsys-rep` with `analyze-nsys-decode.sh`.
3. Review the timeline. Record the critical-path all-reduce fraction and note.
4. Validate the review with `deepseek-v4-speed-trace-gate`.
5. Proceed only if all-reduce consumes at least 10% of the reviewed critical
   path.
6. Run `run-hier-all-reduce-sm86-gate.sh` before the serving arm.
7. After the combined arm wins, use `trace-flashmla-hier` to re-anchor the
   post-optimization critical-path mix. It runs both numerical gates, selects
   the Nsight image, requires the reviewed hierarchical trace gate, and enables
   both runtime dispatches while retaining the trace-only 0.001 GiB host tier.

The hierarchical gate uses islands `[[0,1],[2,3]]`, compares BF16 output with
NCCL, and records latency for 4,096 through 262,144 elements. The A800-derived
24K crossover is not assumed to transfer to server60.

## Long-context gate

Run `run-long-context-gate.sh` only for a candidate that wins its matched speed
comparison. The script runs the functional stress and NIAH ladder, records
physical headroom separately, and fails on serving-process swap. A result below
1 GiB free per GPU remains a capacity ceiling, not a safe release point.

## Rollback contract

The runner never recreates production. It stops the exact verified production
container, removes only unreferenced `vllm_offload_*.mmap` files, runs a uniquely
named experiment with restart disabled, removes that experiment, and repeats
the unreferenced-file cleanup. It then restarts the unchanged production
container and requires its health check to pass. Open offload files are never
removed. If rollback fails, the runner exits nonzero and prints the production
container state and recent logs.
