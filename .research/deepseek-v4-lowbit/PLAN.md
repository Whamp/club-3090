# DeepSeek V4 Flash low-bit safetensors plan

## Decision

Rent compute when it shortens the path, but define the job before creating the instance: hardware, memory and storage needs, expected duration and cost, durable outputs, and a stop condition. A rented GPU is likely useful for the calibrated pilot and probably necessary for efficient full artifact generation. It can be used while server60 remains occupied.

The earlier all-W2 model-free RTN conversion established a useful storage baseline, but it did not choose the final quality recipe. The target is a calibrated, mixed-precision safetensors artifact that:

- occupies about 75–85 GiB, subject to the actual four-RTX-3090 runtime envelope;
- serves at least 200,000 tokens on server60;
- loads through compressed-tensors into Humming WNA16 MoE kernels;
- preserves or improves on the quality of the specific Antirez and Unsloth artifacts Will trusts, as far as practical to check with fast metrics before deeper evaluation;
- improves enough on the current llama.cpp service in prefill speed, decode speed, throughput, or useful concurrency to justify switching.

No single quantization recipe is selected yet. Runtime fit, basic quality, loader correctness, and SM86 dispatch must be established before expensive evaluation. Broader quality work can follow once a candidate runs and performs well.

## What the evidence changes

### The quality signal is useful but not directly portable

Antirez published a 450,892,648-byte DeepSeek V4 Flash routed-expert imatrix at Hugging Face revision `e7f04037032990db0346398d249baf9fb9df1ccc`, SHA-256/Xet etag `02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e`.

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

The four sizes show that the likely final artifact is in the right neighborhood; they are planning anchors, not a commitment to create four checkpoints. Expect to create one final artifact, with a second candidate only if a real quality/context decision remains unresolved. A 200K context is the minimum; more is better. Choose the recipe that gives the best practical balance between model quality and the context capacity left after weights and runtime allocations, rather than automatically choosing the largest artifact that reaches 200K.

## Proposed quantization method

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

Implementation status:

1. The isolated vLLM worktree is `/home/will/projects/vllm/.worktrees/deepseek-v4-wna16-sm86` on branch `research/deepseek-v4-wna16-sm86` from the pinned haosdent revision.
2. Private W2/W3 Humming bridge `b5a0637b4` applied cleanly as `095557aba`.
3. Commit `e5a8452c7` adds projection-sensitive compressed-tensors WNA16 MoE support: fused gate/up must share a schema, down may use a different bit width, mixed schemas are Humming-only, and packed shapes and Humming descriptors remain distinct. It also makes the haosdent SM80 FP8 helper import safely when Triton is disabled, so CPU tests can collect without a GPU.
4. The focused CPU run passed 44 tests covering 2–8-bit Humming support, INT2 loading, mixed W2/W4 selection and packing, per-sublayer schema conversion, invalid mixed layouts, and existing WNA16 behavior. The changed-file ruff, format, mypy, typo, SPDX, import, and repository policy hooks passed.
5. GPU kernel-oracle comparison, packaged SM86 verification, and runtime dispatch proof remain open. Static `SM75+` support is insufficient.
6. `tools/deepseek-v4-lowbit` now provides a dependency-free exact artifact planner and JSON recipe command. Against all 72,317 captured tensor headers it reproduces the four planning anchors at 76.769692, 78.769692, 81.269692, and 85.769692 GiB; classifies 33,024 routed projection weights; preserves 8.238442 GiB; and omits 10.116807 GiB of MTP tensors. Its unit tests, Ruff, formatting, and `ty` checks pass.
7. W3 is not currently safe for DeepSeek V4 through the pinned vLLM loader. The runtime allocates the packed dimension with `32 // 3 = 10`, but the 4096- and 2048-wide expert matrices are not divisible by ten. The planner rejects this case rather than silently truncating. W3 remains a candidate only after the writer and loader gain a tested padding contract.
8. The CPU tool now has a memory-mapped legacy llama.cpp imatrix parser and exact DS4-to-official tensor mapper. It indexes packed entries without materializing the approximately 450 MB float payload, slices one expert vector at a time, normalizes by call count, and rejects corrupt lengths, duplicate names, wrong expert geometry, non-finite selected values, and trailing data. Twelve package tests pass. Direct checksum and 43-layer/129-entry geometry validation against the published artifact remains part of staging the rental pilot.

## Target-runtime fit check

The four modeled layouts span an acceptable size range; they do not all need to become artifacts. The open question is how much context each size would leave on a 24 GiB card. An 85.770 GiB model averages about 21.4 GiB across four cards before uneven or replicated weights. vLLM also needs memory for Humming workspaces, CUDA/NCCL state, graph capture, sparse indexer state, and the attention cache. This check should map each modeled size to usable context capacity, with 200K as the floor rather than the target.

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

## Rental plan

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
3. **In progress:** the tensor-name classifier, exact size planner, imatrix parser, and expert-vector mapping are implemented and verified against pinned format fixtures. Prepare the quantizer comparison next, followed by the resumable shard writer and durable output location; directly validate the parser against the published imatrix while staging the pilot.
4. Define the first bounded rental experiment from that executable tooling, then recheck live on-demand instance availability and total cost.
5. Rent the selected non-interruptible machine and run the representative-layer pilot; expand into full conversion only if the pilot is useful.
6. Do not disturb the active server60 service. When it becomes available and Will authorizes GPU use, test exact-shape modeled layouts and verify packaged Humming SM86 kernels against a small PyTorch oracle.
7. Select and generate the best practical artifact within the measured model-size/context envelope. Generate a second candidate only if planning and pilot data leave a meaningful quality-versus-context decision unresolved.
8. Upload completed artifacts durably and verify checksums and tensor indexes.
9. Load the selected candidate on server60, prove the intended runtime dispatch, and compare prefill, decode, throughput, concurrency, context capacity, VRAM, and basic quality with llama.cpp before deciding how much deeper evaluation is worthwhile.

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
