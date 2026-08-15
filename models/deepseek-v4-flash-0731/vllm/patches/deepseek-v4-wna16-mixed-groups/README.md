# DeepSeek V4 WNA16 mixed projection groups

This research extension lets one routed layer use separate WNA16 group sizes
for fused gate/up (`w13`) and down (`w2`). It exists to support
projection-sensitive DeepSeek V4 quants without changing the accepted
uniform-W2 service.

The canonical source is branch `incubate/deepseek-v4-wna16-sm86` at
[`Whamp/vllm@7b39c9304`](https://github.com/Whamp/vllm/commit/7b39c93043ffa88729d2cd3dd1f8f482df6ea98c).
Mixed projection groups were introduced at
[`dd2d1fd67`](https://github.com/Whamp/vllm/commit/dd2d1fd6779addccc73094f77fa4ada7d9106a41);
the later commit retains the SwiGLU and DSML tool-turn fixes and adds the
promoted FP8 Marlin output-projection path.

## Source contract

- Parent eight-patch tree: `aeb62948e33074514a742d19c2f9a1a3c2ee3e1f`.
- Final candidate tree: `f73b30cc5a2ed9de200ca2e4de3cdef1a06f6538`.
- Patch SHA-256:
  `48095e6d336f6f852d699ddce74e1c192c4b0c9f9bee566e56a15478a95b1def  0009-feat-support-mixed-WNA16-MoE-projection-groups.patch`.
- Changed production file SHA-256:
  `fd512829989af7d86f39a618990d52916aab6ae4b4d70259523c340b2574a830`.
- Promoted Marlin overlay patch SHA-256:
  `bb9fdf4e2452647bccd29934cb2c073a0efa21474a41f0ae659c7be18da4b2fd  0010-perf-run-DeepSeek-V4-output-projection-with-Marlin.patch`.

`install.sh` accepts only a clean checkout at the exact parent tree or the exact
mixed-group tree and applies patch 0009 with non-fuzzy `git am`. Patch 0010 is a
separate deployment mirror of canonical commit `7b39c9304`; the quality-capacity
builder verifies that exact canonical tree and the copied production files.

## Runtime images

`build-runtime-overlay-image.sh` copies only the changed WNA16 MoE loader over
the verified local production image. It does not modify the current Compose or
registry entry. The default candidate tag is:

```text
club-3090/deepseek-v4-wna16-sm86:f73b30cc-mixed-groups-cu130
```

`build-mixed-group-oracle-image.sh` adds only the checksum-pinned tests required
for GPU acceptance. Its image is labeled `mixed-group-oracle-only` and must not
serve requests.

`build-quality-candidate-image.sh` overlays a fail-closed model-view materializer
for Hugging Face revision `12035985bf555d0ddc603c6305586a8fa915589c`. It pins
the candidate config and index hashes and accepts only the four published mixed
W2/W4 groups. Its base is the local runtime for Whamp/vLLM commit
`a7758f7436a713f042e245b3e0aaab64b3a2f2c6`, which adds DeepSeek V4 SwiGLU
alpha, beta, and clamp forwarding after mixed-group support.

`build-quality-capacity-image.sh` is the promoted delivery path. It starts from
the exact DSML-fixed quality image, requires a clean checkout at canonical
commit `7b39c93043ffa88729d2cd3dd1f8f482df6ea98c` and tree
`670643653f99448f90192b79dd0842bcfa073ab8`, checksum-verifies the three
production files, and copies only the FP8 Marlin output-projection change. It
excludes the acceptance-only memory instrumentation and rejected per-token BF16
recomputation experiment. The resulting Compose profile serves 230,144 tokens
with `max_num_seqs=2`.

## Acceptance order

1. Reconstruct the final tree from the accepted parent and verify the tree hash.
2. Run the focused CPU factory, allocation, schema, and resolver tests.
3. Build the oracle-only image over the exact promoted production image.
4. With the promoted service stopped by the rollback wrapper, run all seven
   SM86 numerical cases: W2 or W4 down with gate/up and down group sizes drawn
   from 128, 256, and 512. Require generated `sm_86` cubins.
5. Build a full artifact only after the numerical gate passes.
6. For each full artifact, compile a fresh one-cell DeepSWE launch plan and get
   approval for its exact identity. Run the single-worker
   `superjson-error-stack-serialization` gate before broader quality,
   concurrency, long-context, or performance tests.

The earlier DeepSWE collapse reproduced across both WNA16 artifacts but was
traced to the shared vLLM DSML tool-turn stop bug and fixed at
`9a2ffbb4534400064e645cb4fef8ab2f2a987f11`; it did not establish artifact
causality. Kernel agreement still does not establish model quality, so coding
agent gates remain required for future quantization changes.
