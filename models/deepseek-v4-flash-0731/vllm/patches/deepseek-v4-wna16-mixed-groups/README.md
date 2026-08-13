# DeepSeek V4 WNA16 mixed projection groups

This research extension lets one routed layer use separate WNA16 group sizes
for fused gate/up (`w13`) and down (`w2`). It exists to support
projection-sensitive DeepSeek V4 quants without changing the accepted
uniform-W2 service.

The canonical source is
[`Whamp/vllm@dd2d1fd67`](https://github.com/Whamp/vllm/commit/dd2d1fd6779addccc73094f77fa4ada7d9106a41)
on branch `incubate/deepseek-v4-wna16-sm86`.

## Source contract

- Parent eight-patch tree: `aeb62948e33074514a742d19c2f9a1a3c2ee3e1f`.
- Final candidate tree: `f73b30cc5a2ed9de200ca2e4de3cdef1a06f6538`.
- Patch SHA-256:
  `48095e6d336f6f852d699ddce74e1c192c4b0c9f9bee566e56a15478a95b1def  0009-feat-support-mixed-WNA16-MoE-projection-groups.patch`.
- Changed production file SHA-256:
  `fd512829989af7d86f39a618990d52916aab6ae4b4d70259523c340b2574a830`.

`install.sh` accepts only a clean checkout at the exact parent tree or the exact
final tree. It applies the mail patch with non-fuzzy `git am`.

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

The old all-W2/group-128 artifact failed that DeepSWE behavior gate. Kernel
agreement alone does not establish model quality.
