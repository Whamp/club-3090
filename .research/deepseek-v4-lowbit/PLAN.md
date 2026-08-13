# DeepSeek V4 Flash low-bit safetensors plan and outcome

## Outcome

Completed on 2026-08-12. The project generated one MTP-free, imatrix-weighted
all-W2 artifact with 76.770184 GiB of tensor payload and published immutable Hub
revision `75d9286c37f3037f3ab390cfbc10747466eac714`. The checksum-pinned SM86
runtime serves it on four RTX 3090s at 215,000 tokens with
`max_num_seqs=4`, 233,817 KV-cache tokens, 60.79 decode tokens/s, and 968.97
cache-busted prefill tokens/s. Club-3090 commit
`26ae767aa98c14761ac4a69d4f492f418fd29578` owns the delivery. Server60 runs
that exact Compose. At the final-state capture, system swap and every serving
process were at zero; a later live check still found every serving process at
zero. The remaining operational risk is only 141–142 MiB of free VRAM per card.

## Original decision

The following section preserves the pre-execution decision. Later implementation
ledger entries record the selected recipe, completed rental, runtime validation,
and final deployment.

Rent compute when it shortens the path, but define the job before creating the instance: hardware, memory and storage needs, expected duration and cost, durable outputs, and a stop condition. A rented GPU is likely useful for the calibrated pilot and probably necessary for efficient full artifact generation. It can be used while server60 remains occupied.

The earlier all-W2 model-free RTN conversion established a useful storage baseline, but it did not choose the final quality recipe. The original target was a calibrated, mixed-precision safetensors artifact that:

- occupies about 75–85 GiB, subject to the actual four-RTX-3090 runtime envelope;
- serves the selected 215,000-token server60 profile with `max_num_seqs=4`; this supersedes the first 131,072-token promotion after the residency audit removed avoidable RoPE storage;
- loads through compressed-tensors into Humming WNA16 MoE kernels;
- preserves or improves on the quality of the specific Antirez and Unsloth artifacts Will trusts, as far as practical to check with fast metrics before deeper evaluation;
- improves enough on the current llama.cpp service in prefill speed, decode speed, throughput, or useful concurrency to justify switching.

At planning time, no single quantization recipe was selected. The bounded pilot later chose imatrix-weighted RTN, and the byte budget selected all-W2 routed experts. Runtime fit, basic quality, loader correctness, and SM86 dispatch were established before deeper evaluation.

## What the evidence changes

### The quality signal is useful but not directly portable

Antirez published a 450,892,648-byte DeepSeek V4 Flash routed-expert imatrix at Hugging Face revision `e7f04037032990db0346398d249baf9fb9df1ccc`. Pinned Hub metadata identifies its LFS content SHA-256 as `02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e` and its distinct Xet hash as `cf8a1815d71086e3ec47cefda1fdf381effcb160716ea0c44190a49e2b7614e8`; the rental script verifies the content SHA-256 after download.

This imatrix was collected for the original `deepseek-ai/DeepSeek-V4-Flash`, not the 0731 checkpoint. Antirez uploaded it on May 12, 2026; the repository added 0731 artifacts on July 31 and August 1. Antirez then reused the same imatrix to build 0731 imatrix GGUFs. That makes it a relevant prior for 0731, not a 0731-native activation measurement. The pilot must check whether it improves 0731 WNA16 results; collecting a new 0731 imatrix remains an option if it does not.

The imatrix contains one activation-importance vector per routed expert and input column. Gate/up statistics use squared normalized FFN inputs. Down statistics use squared routed SwiGLU rows after route weighting. The collector used the real DeepSeek V4 inference graph and about 1.5 million calibration tokens drawn from a broader 4,690-prompt, 2.9-million-token corpus.

This is stronger evidence than blind RTN. It can weight reconstruction error when choosing W2/W3/W4 group-128 scales and codes. It does not prove that a WNA16 artifact will match any GGUF: the grids differ, the imatrix comes from another execution path, and GGUF as a format says nothing about quality. The relevant baselines are the particular Antirez and Unsloth artifacts that have performed well for Will.

Antirez measured the imatrix on Q4 routed experts against 100 continuations from the pre-0731 DeepSeek V4 Flash. Average target-token NLL improved from `0.177357819` to `0.173895148`, with 54 of 100 cases improving. The repository later refreshed its continuation fixtures for 0731, but those published Q4 comparison numbers predate that refresh. We can reuse teacher-forced target-token NLL, first-token agreement, and greedy longest-common-prefix when they are cheap to obtain; the result supports testing the imatrix, but it is neither a W2 nor a 0731 result.

Will supplied a temporary OpenRouter budget for this task so a hosted DeepSeek teacher is available if local reference inference is impractical. Keep the credential out of files and logs. Before spending materially, use a few requests to verify the selected model/provider, exact sampling controls, and target-token logprob support, then estimate the full run. Baseten or Cloudflare provider routing may be useful if available when exact controls matter. Stay within the supplied seven-day, $20 ceiling. If the API cannot provide comparable token logprobs, use it only to create fixed reference continuations or skip this metric rather than building a complicated evaluator around it.

### Specific GGUF artifacts suggest a precision hierarchy

The selected GGUF recipes agree on two points:

1. Routed experts consume nearly all of the low-bit budget.
2. Precision should vary by projection and layer.

Pinned Antirez data keeps attention, shared experts, output, routing, compressor, indexer, hyperconnection, embedding, norms, and sinks at higher precision. Its mixed artifact upgrades every routed projection in layers 37–42.

Pinned Unsloth data uses more bits for routed down projections than gate/up in its small artifacts, gives layer 26 special treatment, and upgrades layer 42 down. Its published B200 comparison also shows a steep quality frontier between roughly 91–104 GB GGUFs and the 155 GB MXFP4-preserving artifact. Those B200 results do not transfer numerically to WNA16 or RTX 3090. Treat layer 26 and layer 42 as optional pilot hypotheses, not requirements: use the simpler allocation unless special treatment produces enough benefit to justify extra configuration, conversion, and loader complexity.

Will's own DeepSWE benchmarking finds both the Antirez IQ2_XXS artifact and Unsloth IQ1_M artifact very high quality. That practical result lowers the concern that an aggressively quantized expert recipe is inherently unusable. WNA16 uses different grids, so retain a quick quality screen, but do not make exhaustive quality validation block the experiment.

### WNA16 leaves little upgrade room

Exact sizing over the official checkpoint at revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, excluding 10.117 GiB of MTP tensors, gives:

| Candidate structure | Estimated artifact size |
| --- | ---: |
| All routed experts W2, preserved base tensors | 76.770 GiB |
| Layer 26 all W4; layer 42 down W4 | 78.770 GiB |
| Layer 26 all W4; layers 37–42 down W4 | 81.270 GiB |
| Layers 37–42 all W4 | 85.770 GiB |

Humming stores W3 as ten values per 32-bit word, or about 3.2 packed bits before scales. A broad W3-down recipe exceeds 90 GiB. W3 remains a selective option, not a default middle tier.

The four sizes show that the likely final artifact is in the right neighborhood; they are planning anchors, not a commitment to create four checkpoints. Expect to create one final artifact, with a second candidate only if a real quality/context decision remains unresolved. The initial 200K minimum was a planning requirement, not the first promotion boundary: after the graph-enabled candidate exceeded llama.cpp decode at 131,072 tokens and the pre-RoPE-fix 200K runtime required throughput-killing CPU weight offload, Will accepted 131K for that phase. The 2026-08-12 residency audit later superseded that runtime boundary with the selected zero-offload 215K/c4 profile. Choose the recipe that gives the best practical balance between model quality and context capacity rather than automatically choosing the largest artifact that boots.

## Original quantization method proposal

### 1. Keep non-expert precision conservative

Preserve the established high-precision base:

- attention and shared experts at their source INT8/FP8-style precision where the loader supports it safely;
- router, compressor, sparse indexer, hyperconnection, embedding, output-sensitive tensors, norms, sinks, and biases at source or lossless precision;
- omit MTP from every artifact in this project. It has not helped on Will's hardware. DS4 packages MTP separately, so it can be reconsidered later, but adding it to vLLM would be separate compatibility and performance work.

Only routed expert projection tensors named `w1`, `w3`, and `w2` enter the WNA16 bit planner. These lowercase names are model projections, not bit widths.

Humming itself supports symmetric INT1 weights with BF16 activations on SM80+, and its repository contains W1 kernel tests and benchmarks. That does not give us a usable W1A16 checkpoint path. On August 4, 2026, Benjamin Marie reported reliable mixed-precision vLLM support for 2-, 3-, 4-, and 8-bit layers using AutoRound, LLM Compressor, a custom repacker, and vLLM 0.26 Humming. He explicitly wrote: "Support for 1-bit and ternary layers is still missing." The post is `https://x.com/bnjmn_marie/status/2084816271007449187`. W1 is therefore a raw Humming capability but not part of the demonstrated end-to-end pipeline. It remains outside this artifact unless that integration changes.

### 2. Fit each WNA16 candidate with activation importance

For every routed-expert tensor:

1. Stream one official safetensors shard.
2. Dequantize the source MXFP4/E8M0 representation.
3. Map the corresponding Antirez imatrix entry and slice the correct expert vector.
4. Fit symmetric group-size-128 W2, W3, and W4 candidates using activation-weighted reconstruction error.
5. Record scale, packed weight, weighted error, unweighted error, byte cost, layer, projection, and expert.
6. Discard transient BF16/FP32 buffers before loading the next tensor.

The first implementation should reuse proven AutoRound packing and scale-search primitives where they match the required format. It should not invoke the stock model-free CLI as the recipe owner.

### 3. Compare quantizers before the full run

Run a rented pilot on representative routed layers: an early layer, layer 26, a middle layer, and layers 37 and 42. Compare:

- plain symmetric RTN;
- imatrix-weighted RTN/scale search;
- calibrated AutoRound rounding, if it can operate on a streamed block and export the same Humming schema.

Measure held-out block-output error and downstream target-token NLL. Use the simplest method that produces a repeatable material gain. Do not pay for full-checkpoint calibrated optimization unless the pilot beats imatrix-weighted RTN.

### 4. Allocate bits under a hard byte budget

Aggregate candidate gains at the smallest schema granularity the runtime can express safely:

- separate `w13` and `w2` choices per layer;
- one choice shared by a layer's experts unless runtime tests prove mixed expert dtypes;
- optional W3 only where its measured error reduction per byte beats W4 upgrades.

Solve a constrained allocation problem using marginal held-out loss reduction per added byte. Include the Antirez late-layer and Unsloth layer-26/layer-42 patterns as priors and comparison recipes, not hard-coded truth.

## Required runtime work

Stock vLLM does not currently provide the complete DeepSeek V4 CUDA path for RTX 3090. Two independent community forks are relevant:

- `haosdent/vllm@12810046c799cbe874967e19b1c0fa134ab7b209` is the current leading candidate. Its August 5 squash is a broad, tested SM80/A100 campaign over 111 files. It adds sparse MLA, FP8 cache handling, indexer and MHC paths, communication work, correctness tests, a measurement/refutation record, and backported A6000 fixes. It became the leading candidate because it is newer and more complete and auditable—not because it is already proven on RTX 3090.
- `Lasimeri/vllm-dsv4-ampere@634be6de4382a9da731393805cab90f81a071f85` is a separate, earlier SM86/Ampere implementation built around portable PyTorch/Triton replacements. Its direct A6000/3090 relevance makes it a useful comparison and source of ideas. It also has a reported silent FP8 indexer-cache writer/reader layout mismatch, so that path needs a fix and direct correctness testing before use.

The selected integration base is `haosdent/vllm@12810046c799cbe874967e19b1c0fa134ab7b209`. CodeGraph indexed it as a full 4,720-file vLLM fork; the pinned Lasimeri repository contains a 74-file overlay with duplicated legacy and main-port trees. Source inspection also showed that haosdent keeps current vLLM's NVIDIA model path and selects its SM8x sparse-attention implementation at runtime, while Lasimeri carries a separate full `ampere/model.py`. Haosdent is therefore the lower-friction base for the existing WNA16 patch and broader current-vLLM tests. This is an integration choice, not an RTX 3090 performance verdict. Retain Lasimeri's direct SM86 kernels, parity harnesses, and prefill work as comparison material.

Implementation ledger (chronological; later entries supersede earlier open gates):

1. The isolated vLLM worktree is `/home/will/projects/vllm/.worktrees/deepseek-v4-wna16-sm86` on branch `research/deepseek-v4-wna16-sm86` from the pinned haosdent revision.
2. Private W2/W3 Humming bridge `b5a0637b4` applied cleanly as `095557aba`.
3. Commit `e5a8452c7` adds projection-sensitive compressed-tensors WNA16 MoE support: fused gate/up must share a schema, down may use a different bit width, mixed schemas are Humming-only, and packed shapes and Humming descriptors remain distinct. It also makes the haosdent SM80 FP8 helper import safely when Triton is disabled, so CPU tests can collect without a GPU.
4. The focused CPU run passed 44 tests covering 2–8-bit Humming support, INT2 loading, mixed W2/W4 selection and packing, per-sublayer schema conversion, invalid mixed layouts, and existing WNA16 behavior. The changed-file ruff, format, mypy, typo, SPDX, import, and repository policy hooks passed.
5. GPU kernel-oracle comparison, packaged SM86 verification, and runtime dispatch proof remain open. Static `SM75+` support is insufficient.
6. `tools/deepseek-v4-lowbit` now provides a dependency-free exact artifact planner and JSON recipe command. Against all 72,317 captured tensor headers, the corrected raw-payload anchors are 76.770184, 78.770184, 81.270184, and 85.770184 GiB. These values include the 528,384 bytes consumed by one two-element INT64 `weight_shape` tensor for each of 33,024 routed projection weights; the earlier anchors omitted that fixed cost. The planner preserves 8.238442 GiB and omits 10.116807 GiB of MTP tensors. Filesystem and safetensors-header overhead remain outside these raw-payload figures.
7. W3 is not currently safe for DeepSeek V4 through the pinned vLLM loader. The runtime allocates the packed dimension with `32 // 3 = 10`, but the 4096- and 2048-wide expert matrices are not divisible by ten. The planner rejects this case rather than silently truncating. W3 remains a candidate only after the writer and loader gain a tested padding contract.
8. The CPU tool now has a memory-mapped legacy llama.cpp imatrix parser and exact DS4-to-official tensor mapper. It indexes packed entries without materializing the approximately 450 MB float payload, slices one expert vector at a time, normalizes by call count, and rejects corrupt lengths, duplicate names, wrong expert geometry, non-finite selected values, and trailing data. Direct checksum and 43-layer/129-entry geometry validation against the published artifact remains part of staging the rental pilot.
9. The tool now wraps pinned AutoRound's plain symmetric RTN and imatrix-weighted scale-search primitives behind one comparison interface. It returns signed codes, stored FP16 group scales, reconstruction from those persisted values, and weighted and unweighted MSE. The synthetic W2 fixture improves weighted MSE from about 0.05703 to 0.05031. This verifies the mechanism only. Representative DeepSeek layers still decide whether weighted search is useful.
10. Exact compressed-tensors packing and resumable shard output are implemented. Packing delegates to the `compressed-tensors==0.17.0` primitive pinned by the selected vLLM fork, verifies the literal low-bit word order, emits per-expert `weight_packed`, `weight_scale`, and `weight_shape` keys, and preserves the W3 fail-closed geometry rule. Each atomic safetensors shard has a receipt binding source and canonical recipe fingerprints to its output checksum, actual safetensors-header inventory, and transform metrics; resume, crash-window recovery, corruption rejection, and final index assembly are covered.
11. The full source-shard transform and `deepseek-v4-convert` CLI are implemented. They reuse pinned AutoRound's official DeepSeek MXFP4/E8M0 path one routed weight at a time, support plain or imatrix-weighted RTN, preserve non-routed tensors, omit MTP, generate exact projection-specific compressed-tensors metadata, disable MTP in the output config, copy non-weight assets, and finalize only after all 48 source shards are accounted for. Source shards 46–48 contain only the 4,705 omitted MTP tensors, so the final artifact has 45 output shards and exactly 100,636 tensors. Finalization now compares every produced tensor name and shard assignment with the output map derived independently from the source index before publishing `model.safetensors.index.json`. The captured official headers confirm zero missing or cross-shard weight/scale pairs across all 35,328 routed weights. A combined-toolchain test exercises real FP4 dequantization, an MTP-only source shard, and finalized MTP-free artifact output; the selected vLLM fork resolves the generated mixed W2/W4 gate/up/down targets as intended.
12. `deepseek-v4-pilot` now provides the bounded rental method screen. The first run uses source shards 2, 28, 39, and 44 for layers 0, 26, 37, and 42; experts 0 and 127; all w1/w2/w3 projections; and W2 only. It validates the imatrix checksum and complete 129-entry geometry, then compares 24 matrices and 48 plain-versus-weighted candidates with packing checks, elapsed time, and both error metrics. Weighted RTN advances only if measured error improvement justifies its runtime; this is not an end-to-end quality gate.
13. The bounded Verda run completed on 2026-08-12 using one on-demand, non-interruptible `1A100.22V` in FIN-01 with A100 SXM4 80 GB, 120 GB RAM, and 350 GB boot NVMe. The W2/group-128/BF16 Humming indexed-MoE numerical oracle passed through the real compressed-tensors conversion and experts path; Humming generated 13 `sm_80` cubins, all of which passed ELF inspection. The 24-matrix pilot found imatrix-weighted RTN improved weighted reconstruction error in 24/24 comparisons, with 31.03% median improvement and projected quantize-and-pack time of 5,987.65 seconds versus 5,432.44 seconds for plain RTN. The full conversion therefore selected `imatrix-weighted-rtn`, used 112.9 GiB peak host memory with no swap, finalized 45 output shards, omitted all three MTP-only source shards, and uploaded directly from Verda to the repository while it was private. The exact remote inventory verified at `hampsonw/DeepSeek-V4-Flash-0731-WNA16@75d9286c37f3037f3ab390cfbc10747466eac714`: 54 repository files, 45 SHA-256 LFS/Xet objects, 9 Git blobs including Hub-managed `.gitattributes`, and 82,464,249,582 artifact bytes. The VM and detached OS volume were deleted after verification; final Verda state was zero VMs, zero volumes, and zero running cost.
14. The raw pilot now has a paired summarization contract: count error improvements/ties/regressions, report median weighted-error change by projection, and extrapolate measured quantize-and-pack time over all 43 × 256 routed experts. It deliberately leaves the quantizer decision unset and labels excluded full-run costs rather than turning a small method screen into an automatic gate.
15. Server60 staged the immutable, then-private Hub revision directly into its standard `/mnt/models/huggingface/hub` cache on 2026-08-12, with Xet transients directed to the root filesystem. The download completed in 11 minutes 51 seconds without a second local artifact copy. A post-download audit matched all 54 snapshot files and all 82,464,249,582 bytes to the immutable Hub object digests. `/mnt/models` retained 22 GiB free. The existing `llama-cpp-deepseek-v4-fast-prefill` container stayed healthy throughout, GPU utilization remained at its pre-existing roughly 20–26%, and the only GPU process remained `/app/llama-server` PID 3402959. The artifact is staged but has not been loaded or executed.
16. The exact `humming-kernels==0.1.10` PyPI artifact is a 184,889-byte pure-Python wheel with SHA-256 `4ded0998ff085afeddde70baf93f97c2929969ec3d4a63a52cfec5072bc972b4`, built from `inclusionAI/humming@4351af3a8fcdce1a8dee50104ba49566af2427fb`. It contains JIT CUDA sources but no packaged `.so`, PTX, cubin, or fatbin. Runtime reads the current device capability, maps exact capability 8.6 to dedicated `Sm86Heuristics` with a 99 KiB shared-memory limit and BF16 support, then passes `sm_86` to NVRTC or `compute_86,code=sm_86` to NVCC. Its compressed-tensors schema maps symmetric pack-quantized W2/group-128 to uint2 weights and BF16 scales. These facts prove a specific source/build path, not successful compilation or dispatch. Pinned upstream GPU tests cover dense uint2 and uint4 MoE separately but not uint2 + group-128 + BF16 MoE. vLLM commit `f4d05732a` adds that exact deterministic indexed-MoE numerical oracle at the real conversion/experts seam; it collects and skips on the CPU host, passes all applicable pre-commit hooks and slop checks, and remains deferred first to rental A100 for generic kernel correctness and then—only after explicit GPU-use authorization—to server60 for generated sm_86 cubin, numerical, and dispatch evidence.
17. The private vLLM work is portable without a third-party push. `models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86/` vendors eight SHA-256-listed mail patches plus a fail-closed installer. The installer accepts only clean `haosdent/vllm@12810046c799cbe874967e19b1c0fa134ab7b209` or exact final tree `aeb62948e33074514a742d19c2f9a1a3c2ee3e1f`, uses non-fuzzy `git am`, is idempotent on the final tree, and rejects base drift without modification. A checksum-pinned thin image bakes the seven production files changed by patches 0005–0008 over the verified runtime base, creates the corrected model view at startup, and is wired to the direct-Compose 215K profile. The patch registry records this as verified `runtime_image` delivery. vLLM PR #48918 remains tracked in `docs/UPSTREAM.md`.
18. `run-verda-vllm-w2-oracle.sh` stages the first GPU contract separately from quantization so time and failures remain attributable. It reconstructs the exact vLLM tree, uses vLLM's documented precompiled-extension editable install in an isolated Torch 2.13/cu130 environment, requires Humming 0.1.10 and A100 capability 8.0, runs the deterministic W2/group-128/BF16 indexed-MoE oracle under NVRTC, and records each generated cubin's SHA-256 and `cuobjdump` ELF report with an `sm_80` assertion. This A100 stage can validate the generic software/kernel path but cannot satisfy the later server60 SM86 compile, dispatch, or performance gates.
19. Rental continuation now fails closed on the exact pilot handoff rather than accepting any 48-row JSON file. The pilot report binds its source index, selected source shards, and imatrix by SHA-256; the summary binds the report by SHA-256. Before full conversion, `deepseek-v4-validate-pilot` reconstructs the expected samples, tensors, source-shard assignments, W2 plain/weighted candidate set, device, and group size from the pinned input files, checks finite metrics, recomputes the summary, and rejects any stale, changed, incomplete, duplicate, or mismatched evidence.
20. The artifact repository `hampsonw/DeepSeek-V4-Flash-0731-WNA16` was created private before the rental. The active shell's `HF_TOKEN` was read-only, while the stored `upload` credential had fine-grained `repo.write` permission. Rental setup transferred the upload credential without printing it. The full-run preflight validated the exact environment token's namespace write grant and required a private target repository before downloading the full checkpoint.
21. The verified artifact was published after the experiment. Hub commit `18383644489821a6d2b7356b13f53b4bd6bc2ac4` replaced the copied upstream README with an artifact-specific model card covering the conversion, pinned inputs, runtime dependency, initial eager result, limited quality evidence, credits, and license. The repository model card now records the later graph-enabled 215K profile; republishing that update to the Hub is separate from this runtime goal.
22. The selected 350 GB boot NVMe supplies about 325.96 GiB. The pinned source payload is 155.42 GiB and the all-W2 output payload is 76.77 GiB, leaving about 93.77 GiB before the OS, environments, metadata, headers, and transient state. The full-run 260 GiB free-space gate runs against `RENTAL_ROOT`; at that boundary, source plus output still leave about 27.81 GiB. Hugging Face local-directory downloads write payloads directly under the source directory and keep only bookkeeping under its `.cache`, avoiding a second full Hub-cache copy.
23. The performance campaign superseded the eager rejection without changing the artifact. Matched graph → eager → graph arms measured 60.27 → 4.96 → 60.07 decode tokens/s. The first zero-offload 131,072-token profile measured 60.71 decode tokens/s, 964.09 prompt tokens/s at 8,984 tokens, exact NIAH retrieval at 119,895 tokens, and clean concurrency 2/4 at 65.19/90.23 aggregate tokens/s. The first overlay-free image repeated 60.82 decode tokens/s versus the matched llama.cpp baseline of 32.67.
24. The residency audit explained the major runtime buckets and found two shared FP32 RoPE tables materialized to 1,048,576 positions regardless of served context. Patch 0008 preserves the original YaRN frequency span while bounding only materialized rows to runtime `max_model_len`. At the selected 215K profile this removes about 407 MiB per rank. Exact image `sha256:0beb1f0cba2e41837f4ba5af01cc5c4686afde4f40ab1df5147a6ad945b0af1f` measured 60.79 decode and 968.97 prefill tokens/s, retrieved the exact needle at 204,900 prompt tokens, and passed concurrency 2/4 at 65.47/89.94 aggregate tokens/s with zero VRAM growth. A 230K/c4 arm estimated a 215,552-token maximum; 230K/c2 estimated 223,488. Although 220K/c2 fit, Will selected 215K/c4 because 5K more context did not justify halving supported concurrency.
25. Club-3090 commit `26ae767aa98c14761ac4a69d4f492f418fd29578` publishes the eight-patch delivery and canonical 215K/c4 Compose. Server60 now runs that exact detached checkout as Compose project `dsv4-wna16-prod`, container `vllm-deepseek-v4-wna16-sm86`, with restart policy `unless-stopped`. The first canonical cutover reached API readiness but an evidence file inside the checkout tripped its clean-tree checker; the armed rollback restored the already validated 215K service. After moving evidence outside the checkout, the same canonical Compose passed startup, `verify-full`, and a deterministic post-swap-clear request. At the recorded final-state capture, system swap and every serving process were at zero; the service had 233,817 KV-cache tokens, no rollback timer, unchanged 230 W power limits and clock caps, and 141–142 MiB free VRAM per card. The fail-closed record is `/home/will/inference/runtime/deepseek-v4-wna16-sm86/canonical-promotion-20260812/FINAL-STATE.txt` with SHA-256 `6c7344498727f867116d1161da0aae36f86f822551d488f35d73afd4dd376bfb`. The residency audit is complete.

## Original target-runtime fit check

This check is complete; implementation-ledger entries 23–25 and
`VLLM-PERFORMANCE-RESEARCH.md` record the measured result. The following text
preserves the original framing.

The four modeled layouts span an acceptable size range; they do not all need to become artifacts. The open question was how much context each size would leave on a 24 GiB card. An 85.770 GiB model averages about 21.4 GiB across four cards before uneven or replicated weights. vLLM also needs memory for Humming workspaces, CUDA/NCCL state, graph capture, sparse indexer state, and the attention cache. This check should map each modeled size to usable context capacity, with 200K as the floor rather than the target.

When the active server60 evaluation is finished and Will authorizes GPU use:

1. Build the selected Ampere fork and Humming package for SM86.
2. Use dummy/exact-shape quantized model construction to try the 76.770, 78.770, 81.270, and 85.770 GiB layouts without converting the checkpoint first.
3. Start each viable layout at 200K using low-memory eager settings first and the intended graph mode afterward.
4. Record per-card weight memory, other runtime memory, peak VRAM, reported KV-token capacity, and the highest practical context for each layout.
5. Choose among the resulting quality/context tradeoffs; do not automatically promote the largest model.

Pinned vLLM's specialized DeepSeek V4 cache stores 584 bytes per compressed history row. The 43-layer compression schedule contributes about 0.588 GiB at 200K tokens before its smaller additional structures and allocation overhead. That cost grows with context, so weight savings above the 200K floor may be valuable even when a larger artifact fits.

## Research checks

This is a research project, so use progressive checks rather than arbitrary release thresholds. Correctness checks here mean cheap safeguards against producing a corrupt checkpoint or unknowingly running the wrong kernel—not rigid tolerances or repeated full-run validation. Run them on small samples and at resumable shard boundaries so a failure costs minutes or one shard, not the entire conversion. Keep quality and performance decisions comparative and cheap until the candidate proves worth deeper evaluation.

### Minimum correctness

- Check tensor names, counts, shapes, dtypes, indexes, and checksums as each output shard is completed.
- Sample a few packed W2/W3/W4 tensors and compare their dequantized values with the quantizer output.
- Compare a few representative Humming MoE outputs with a dequantized PyTorch reference using a tolerance chosen after observing normal numeric error.
- Smoke-test tensor-parallel/expert-parallel loading and confirm that the intended Humming path is selected.
- Checkpoint progress so a failed check resumes from the last completed shard rather than restarting conversion.

### Fast quality screen

Start with metrics that can reject a bad candidate quickly:

- teacher-forced target-token NLL on a small held-out subset, then the full 100-continuation set only for promising candidates;
- first-token agreement and greedy longest-common-prefix length;
- a small representative slice of club-3090 thinking, non-thinking, tool-call, and code prompts;
- one short and one 200K needle test.

Compare trends against the official model where practical and against the specific Antirez and Unsloth artifacts Will trusts. Do not impose a percentage cutoff without seeing the metric variance and cross-engine comparability. Investigate obvious collapse or broad regression; otherwise use judgment and continue if the performance result is promising.

Defer the full quality packs, broad code/agent evaluation, repeated stability runs, and the complete long-context ladder until a candidate loads, dispatches the intended kernels, and has competitive speed. A promising artifact may justify deeper quality work after the initial research cycle rather than before it.

### Performance and deployment

The main comparison with llama.cpp is:

- prefill speed at representative short and long prompts;
- single-stream decode speed;
- aggregate throughput and useful concurrency under multiple requests;
- per-card peak VRAM and maximum usable context.

First confirm Humming WNA16 MoE dispatch from runtime evidence. Use brief repeatable runs to establish direction. Add fuller latency distributions, stress, soak, and Nsight attribution only if the result could plausibly replace the current service.

A safetensors artifact that loads but silently falls back to an unsuitable path has not answered the research question.

## Original rental plan

Rental is an expected accelerator, not a last resort. Do not wait for every server60 or local implementation question to close if a bounded cloud experiment can answer it faster. Before each rental, record the exact command or workload, required GPU and host memory, disk size, expected runtime, maximum intended spend, outputs to preserve, and the condition for stopping or expanding the run.

Use two stages:

1. A short, capped pilot that validates the environment and processes representative layers to compare quantizers.
2. The full streamed conversion after the pilot shows enough signal to justify the larger spend.

A previously observed single-A100 option had 80 GiB GPU memory and 120 GiB host memory. It is a plausible pilot machine because one dequantized routed layer is much smaller than 80 GiB and AutoRound publishes model-free A100 results. It is not preselected. Recheck live inventory, total price including storage, and durable-output options immediately before booking; a different GPU or CPU-heavy instance may offer lower total cost for the defined job.

Use an on-demand, non-interruptible instance by default. Do not use spot or other interruptible capacity for calibration or conversion unless the exact stage has already proven that it can resume without lost work or duplicated cost, and Will explicitly agrees to that tradeoff.

Plan at least 450–500 GB of fast local storage if source, work files, and more than one candidate coexist. Prefer sequential upload and deletion of completed candidates. Keep worker count at one until measured peak committed memory establishes a safe higher value. The prior local failure showed about 26.5 GiB committed memory for a worst-case model-free worker after shard adjustment; RSS alone is not a safe sizing metric.

## Ordered next steps

1. **Done:** compare haosdent and Lasimeri and select the pinned haosdent base.
2. **Done for CPU contracts:** port the W2/W3 bridge and add separate `w13`/`w2` schema support in the isolated vLLM worktree.
3. **Done:** implement and validate the complete CPU conversion, resumability, exact-inventory, metrics, upload, and private-Hub verification contracts.
4. **Done:** run the bounded on-demand A100 rental with a deletion watchdog and prove the generic Humming W2/group-128/BF16 indexed-MoE path through a numerical oracle and 13 inspected `sm_80` cubins.
5. **Done:** run the 24-matrix plain-versus-imatrix pilot and select imatrix-weighted RTN from 24/24 weighted-error improvements with modest projected time cost.
6. **Done:** generate one all-W2, MTP-free artifact; upload it directly from Verda; verify all Hub hashes and inventory; and delete all rental compute and storage.
7. **Done:** stage the immutable Hub revision on server60 and verify all 54 cache objects without loading the model, touching the existing GPU process, or disrupting llama.cpp.
8. **Done:** reconstruct the pinned SM86 runtime and pass the exact
   W2/group-128/BF16 numerical oracle on one RTX 3090 in 78.24 seconds.
   Humming generated 13 inspected `sm_86` cubins. The final eight-patch
   integration tree is `aeb62948e33074514a742d19c2f9a1a3c2ee3e1f`; it
   composes native DeepSeek FP8 linears with routed WNA16 experts, routes SM86
   sparse decode away from a split-K tile that exceeds the GPU's 101,376-byte
   shared-memory limit, and bounds RoPE materialization to runtime context.
9. **Done; graph-enabled 215K/c4 profile selected:**
   the first accepted eager configuration loaded all 45 shards at 200K with a
   256-token chunk ceiling, 64 MiB sparse-indexer logits budget, per-query
   sparse-prefill fallback, and explicit 1 GiB packed KV allocation per GPU.
   It exposed 210,826 cache tokens, returned an exact short response, measured
   809 prompt tok/s at 9,009 tokens, and decoded at 5.55 tok/s. That was a
   correctness result, not a fair vLLM performance baseline.

   A later matched campaign measured thermally warm llama.cpp at 32.67 decode
   tok/s. Breakable CUDA graphs raised vLLM from 4.96 tok/s in the matched eager
   ablation to 60.27 tok/s; graph-after-eager repeated at 60.07. The first
   graph-enabled zero-offload profile served 131,072 tokens with
   `max_num_seqs=4` and `max_num_batched_tokens=256`. It measured 60.71 decode
   tok/s, 964.09
   cache-busted prompt tok/s at 8,984 tokens, exact NIAH retrieval at 119,895
   tokens, and clean concurrency 2/4 at 65.19/90.23 aggregate tok/s. The
   checksum-pinned final image repeated 60.82 decode tok/s with no source
   overlays, no serving-process swap, and zero VRAM growth.

   The later residency audit found and removed about 407 MiB per rank of
   avoidable RoPE storage at 215K. The selected zero-offload profile serves
   215,000 tokens with `max_num_seqs=4`, measures 60.79 decode tok/s and 968.97
   cache-busted prefill tok/s, retrieves exactly at 204,900 prompt tokens, and
   passes short concurrency 2/4 at 65.47/89.94 aggregate tok/s with zero
   post-warm VRAM growth. `verify-full` passed; every functional
   `verify-stress` check passed through 197,580 tokens. The profile also retains
   the earlier deterministic generation, forced-tool, reasoning, and
   `quality-test.sh --quick` evidence. Published commit `26ae767a` is live as
   canonical Compose project `dsv4-wna16-prod` with restart policy
   `unless-stopped`; its final functional smoke and deterministic post-clear
   request passed after a zero-system-swap clear, and every serving process
   remained at zero swap in the later live check. The separate residency audit
   is closed.

## Pinned evidence

- Official checkpoint: `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- Antirez GGUF and imatrix repository: `antirez/deepseek-v4-gguf@e7f04037032990db0346398d249baf9fb9df1ccc`
- Antirez DS4 source and imatrix documentation: `antirez/ds4@84cc882352757baf628a1776badf7cc54d584e28`
- Unsloth GGUF metadata: `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93fb787c21338159b0af3318bb3f4d9768`
- AutoRound source used in the prior experiment: `intel/auto-round@f17d9cd4`
- Selected Ampere vLLM integration base: `haosdent/vllm@12810046c799cbe874967e19b1c0fa134ab7b209`
- SM86 comparison overlay: `Lasimeri/vllm-dsv4-ampere@634be6de4382a9da731393805cab90f81a071f85`
- Humming package: `humming-kernels==0.1.10`, source `inclusionAI/humming@4351af3a8fcdce1a8dee50104ba49566af2427fb`, PyPI wheel SHA-256 `4ded0998ff085afeddde70baf93f97c2929969ec3d4a63a52cfec5072bc972b4`
- Benjamin Marie mixed-precision vLLM post, including the W1/ternary limitation: `https://x.com/bnjmn_marie/status/2084816271007449187`
- Private current-main vLLM compatibility commit: `b5a0637b4`
