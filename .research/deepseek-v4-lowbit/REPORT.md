# DeepSeek V4 Flash low-bit inference on four RTX 3090s

Status: completed and deployed on 2026-08-14

## Executive result

A projection- and layer-sensitive DeepSeek-V4-Flash-0731 safetensors artifact now runs in vLLM on four 24 GB RTX 3090 GPUs with:

- 230,144 advertised context tokens;
- 275,238 aggregate KV-cache tokens;
- tensor parallel size 4 and `max_num_seqs=2`;
- 61.91 single-stream decode tokens/s;
- 875.93 cache-busted prefill tokens/s on an approximately 9K-token prompt;
- no CPU weight offload;
- zero serving-process swap during final validation;
- exact retrieval from a 211,031-token prompt; and
- exact retrieval from two simultaneous 90,029-token prompts.

The model is `hampsonw/DeepSeek-V4-Flash-0731-WNA16` revision `12035985bf555d0ddc603c6305586a8fa915589c`. The validated vLLM source is `Whamp/vllm@7b39c93043ffa88729d2cd3dd1f8f482df6ea98c`, tree `670643653f99448f90192b79dd0842bcfa073ab8`. The server60 image is `sha256:f56910530683326051cfdf4e7c8e4d6afc5bace8804cb78b2af9ea799bbba4e6`.

This result depends on custom DeepSeek V4 Ampere support, compressed-tensors WNA16 loading, Humming low-bit MoE kernels, runtime-bounded RoPE caches, SM86 sparse-attention fallbacks, corrected SwiGLU semantics, a DSML tool-turn stop fix, and an FP8 Marlin output-projection path.

## Scope

This work answers one narrow question: can a DeepSeek V4 Flash low-bit safetensors checkpoint serve useful long-context agent workloads through vLLM on a PCIe-only four-RTX-3090 host?

It does not establish that the same settings transfer to another host, GPU generation, vLLM revision, model revision, or quantization format. It also does not establish broad benchmark parity with full-precision DeepSeek V4.

## Target host

The measured host is server60:

- four NVIDIA GeForce RTX 3090 GPUs, 24,576 MiB each, compute capability 8.6;
- no active NVLink;
- CUDA peer reads and writes available for every GPU pair;
- PCIe Gen3 links at x4, x16, x8, and x16 respectively;
- one AMD Threadripper 2950X NUMA node; and
- 60 GiB system RAM.

The runtime uses tensor parallel size 4. vLLM's custom all-reduce is disabled because this is a PCIe-only four-GPU topology; NCCL still uses peer transfers.

## Artifact

### Source checkpoint

The official DeepSeek-V4-Flash-0731 checkpoint has 284,334,567,511 base parameters across 43 layers. Routed experts contain 277,025,390,592 weights, or 97.43% of the base parameter count. The model has 256 routed experts, top-6 routing, and one shared expert per layer.

MTP tensors are omitted. They did not accelerate the target host and would consume storage and VRAM without helping the selected runtime profile.

### Quantization recipe

The selected quality artifact uses separate precision policies for fused gate/up (`w13`) and down (`w2`) routed-expert projections:

- every `w2` down projection uses W2, group size 128;
- `w13` uses group size 128 on layers 26 and 37–42;
- `w13` uses group size 256 on layers 0–2, 23, 25, 27–29, 31, and 32;
- `w13` uses group size 512 on the remaining 26 layers; and
- `w2` uses W4 on layers 26 and 37–42.

The checkpoint preserves non-routed tensors in their source-compatible higher-precision formats. Its 45 model shards contain 84,556,396,276 payload bytes. It is larger than the first uniform W2/group-128 artifact because it deliberately spends more bits on down projections and sensitive late layers.

### Why this recipe

Antirez and Unsloth low-bit DeepSeek recipes independently spend precision by projection and layer rather than quantizing all routed matrices uniformly. Their maps repeatedly protect routed down projections, layer 26, and late layers, especially layers 37–42.

The project combined those demonstrated priors with a full-expert activation-weighted screen. It did not treat reconstruction error as an end-to-end quality result. The screen chose among runtime-supported W2/W4 and group-128/256/512 schemas; coding-agent behavior remained a separate acceptance gate.

### Generation and publication

The quality artifact was generated on a rented A100 after a checksum-bound screen and recipe recovery. The completed screen was recovered after an orchestration defect deleted the first VM too aggressively. Its nine-file evidence bundle is immutable in the model repository at commit `2686304a68557827d847e1954050cde6b5e7fd08` on branch `evidence-quant-frontier-screen-20260813`.

The candidate is immutable at revision `12035985bf555d0ddc603c6305586a8fa915589c`. Independent verification checked exactly 62 expected files, all 45 shard sizes and SHA-256 identities, and every artifact-owned small file. Hub-managed `.gitattributes` is excluded from byte identity because Hugging Face rewrites it.

No Verda VM or volume remains. The final Verda balance after cleanup was $18.97272, with $0 hourly cost.

## Runtime integration

### Base

The runtime began from `haosdent/vllm@12810046c799cbe874967e19b1c0fa134ab7b209`, an Ampere DeepSeek V4 fork. The maintained implementation now lives in `Whamp/vllm`.

The custom path covers:

- native DeepSeek block-FP8 handling for ordinary linears;
- compressed-tensors WNA16 routed experts;
- Humming indexed MoE kernels;
- separate `w13` and `w2` bit widths and group sizes;
- SM86-compatible sparse prefill and decode fallbacks;
- runtime-bounded RoPE materialization;
- DeepSeek V4 SwiGLU alpha, beta, clamp, and fused-MoE group semantics;
- DSML tool-call termination; and
- FP8 Marlin for the grouped `wo_a` output projection.

### Loader failures found during bring-up

The first full artifact loads exposed several independent seams:

1. Compressed-tensors metadata initially replaced the model-wide DeepSeek FP8 declaration, so preserved attention scales did not load.
2. Generic compressed-tensors FP8 transformed grouped `wo_a` weights into a layout incompatible with the DeepSeek attention path.
3. The WNA16 MoE factory failed to forward its layer object into Humming setup.
4. A100-oriented sparse prefill and split-K decode kernels exceeded RTX 3090 shared-memory limits.
5. The first artifact fit only with conservative eager settings and was slow until breakable CUDA graphs were enabled.
6. The DeepSeek DSML parser failed to terminate after the outer tool-call block, causing post-tool responses with hundreds of repeated tool calls.

Each failure received a focused regression or numerical gate before the next full-model run.

### SM86 kernel acceptance

Seven Humming numerical oracle cases ran on RTX 3090:

- W2 `w13`/`w2` group pairs 128/128, 256/128, 512/256, and 512/512;
- W2 `w13` with W4 `w2` at 512/128, 512/256, and 512/512.

All seven passed in 328.36 seconds and produced 56 `cuobjdump`-confirmed `sm_86` cubins. The cubin manifest SHA-256 is `831fff2fa023c92056804b24247d48e8940ff8cc57297cf37674d7c1a3e65ad3`.

This establishes numerical agreement and SM86 device-code generation for the tested synthetic cases. It does not establish full-model quality by itself.

## The coding-agent failure was a runtime bug

Both the uniform and projection-sensitive WNA16 artifacts initially failed a DeepSWE coding-agent task after one or more normal tool turns. The model then emitted a single enormous response containing hundreds of repeated read calls and produced no patch.

The recurrence across two quantization recipes made artifact causality plausible but did not prove it. Captured requests showed that the collapse began after a tool result entered conversation history. The custom DeepSeek V4 tokenizer rendered structurally valid DSML, but vLLM did not add the outer `</｜tool▁calls▁end｜>` marker as a stop sequence.

`Whamp/vllm@9a2ffbb4534400064e645cb4fef8ab2f2a987f11` fixed that shared runtime bug while preserving caller-provided stop sequences. After the fix, the projection-sensitive model completed a tool call, accepted its result, and returned a normal 18-token response without another tool call.

The earlier “bad quant” conclusion is therefore withdrawn. The incident remains a useful warning: a model-specific parser defect can look like catastrophic quantization damage, and small one-turn quality suites cannot expose a post-tool state-machine failure.

## Performance progression

### Fair baseline

The first correctness bring-up ran eager and decoded at about 5.55 tokens/s. That was not a fair vLLM baseline. Enabling the fork's breakable CUDA graphs raised decode to about 60 tokens/s. A matched llama.cpp Unsloth UD-IQ1_M baseline measured 32.67 tokens/s.

The projection-sensitive artifact's fresh 131,072-token baseline measured:

| Metric | Result |
| --- | ---: |
| Decode, 3 warmups + 5 measured 512-token runs | 61.56 tok/s |
| Cache-busted prefill, 3 approximately 9K-token runs | 920.91 tok/s |
| Aggregate KV capacity | 148,290 tokens |
| Peak serving-process swap | 0 |

The campaign required at least 46.17 decode tokens/s and 552.546 prefill tokens/s.

### Capacity experiments

| Change | Capacity | Decode | Prefill | Decision |
| --- | ---: | ---: | ---: | --- |
| `max_num_seqs` 4 → 2 | 138,240 context | 61.45 | 968.49 | Kept; small gain |
| `max_num_batched_tokens` 256 → 128 | 156,000 context | 59.19 | 465.90 | Rejected; prefill below floor |
| Disable retained BF16 `wo_a` | 230,144 context | 34.01 | 899.36 | Rejected; decode below floor |
| FP8 Marlin diagonal `wo_a` | 230,144 context | 61.91 | 875.93 | Selected |
| 260,000 context | Planner estimated 255,232 maximum | — | — | Rejected; allocator boundary |

Earlier CPU/UVA offload tests reached 200K but reduced decode to 12.68–19.54 tokens/s. They were not repeated because they could not meet the new floor.

## Capacity mechanism

A storage-deduplicated residency trace found a persistent 16 MiB BF16 `wo_a` tensor in each of 43 attention layers. The SM86 path dequantized each block-FP8 output-projection weight once and retained the result for `torch.einsum`, consuming 688 MiB per tensor-parallel rank.

Removing the cache reclaimed capacity but dequantizing every layer on every token cut decode almost in half.

The accepted path instead:

1. keeps the original block-FP8 weight and scales;
2. packs the grouped weight for vLLM's existing FP8 Marlin kernel;
3. flattens the two local projection groups into one local projection; and
4. selects the matching block-diagonal outputs.

This removes the 688 MiB BF16 duplicate without repeated dequantization. The path is opt-in with `VLLM_DSV4_WO_A_MARLIN_DIAGONAL=1`; default behavior is unchanged.

## Final measured profile

The checksum-pinned thin image copies only the three production files changed by the Marlin path over the validated DSML-fixed quality image.

| Field | Value |
| --- | --- |
| Model | `deepseek-v4-flash-0731-wna16-quality-12035985` |
| Context | 230,144 |
| Tensor parallel size | 4 |
| `max_num_seqs` | 2 |
| `max_num_batched_tokens` | 256 |
| GPU memory utilization | 0.98 |
| CPU weight offload | 0 |
| KV cache | `fp8_ds_mla` |
| Aggregate KV tokens | 275,238 |
| Maximum full-length concurrency | 1.20× |
| Decode | 61.91 tok/s, CV 0.1% |
| Cache-busted prefill | 875.93 tok/s, CV 0.7% |
| Post-benchmark VRAM growth | 8 MiB aggregate |
| Serving-process swap | 0 |

Final-image API checks passed deterministic generation, automatic tool selection, and post-tool continuation. The final-image long-context request used 211,031 prompt tokens, retrieved the exact needle, stopped normally after 14 completion tokens, and completed in 265.03 seconds.

## Limitations

### Thin physical headroom

After a 211K stress request, only 91–94 MiB of physical VRAM remained per GPU. The service stayed healthy and zero-swap, but it failed the repository's 1 GiB sustained-agent headroom advisory.

Treat 230,144 as a measured single-request capacity profile, not a generous-headroom operating point. Two-request validation covered two 90K prompts, not two simultaneous 230K prompts.

### Specialized cache format

The DeepSeek V4 Ampere path uses `fp8_ds_mla`, a specialized UE8M0 block-scaled cache. Q8-K/Q4-V and Q4 KV are not launch-flag alternatives in this runtime. They require a separate layout and backend implementation.

### Host-specific result

PCIe topology, CUDA driver, JIT cache state, thermal state, and vLLM source all affect the result. Revalidate packaged source, generated device code, numerical correctness, memory, and end-to-end performance before transferring it.

### Model-quality scope

The final artifact passed tool/post-tool canaries, focused quality checks, long reasoning probes, and long-context retrieval. The campaign did not run a broad academic evaluation suite. Future quantization changes should retain an early multi-turn coding-agent gate because reconstruction metrics and one-turn tests missed the DSML failure mode.

## Reproducibility identities

| Component | Identity |
| --- | --- |
| Official model source | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Antirez imatrix content SHA-256 | `02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e` |
| Screen evidence | `2686304a68557827d847e1954050cde6b5e7fd08` |
| Quality artifact | `12035985bf555d0ddc603c6305586a8fa915589c` |
| Canonical vLLM commit | `7b39c93043ffa88729d2cd3dd1f8f482df6ea98c` |
| Canonical vLLM tree | `670643653f99448f90192b79dd0842bcfa073ab8` |
| Mirrored Marlin patch SHA-256 | `bb9fdf4e2452647bccd29934cb2c073a0efa21474a41f0ae659c7be18da4b2fd` |
| Final runtime image | `sha256:f56910530683326051cfdf4e7c8e4d6afc5bace8804cb78b2af9ea799bbba4e6` |
| Club-3090 deployment commit | `e625a892b3af6c32ec22394e2341eed4bb8bdc17` |

## Evidence map

- [`CAPACITY-MARLIN-20260814.md`](CAPACITY-MARLIN-20260814.md): causal capacity campaign.
- [`VLLM-PERFORMANCE-RESEARCH.md`](VLLM-PERFORMANCE-RESEARCH.md): source-grounded vLLM scheduler, graph, KV, attention, and communication research.
- [`VLLM-PERFORMANCE-PLAN.md`](VLLM-PERFORMANCE-PLAN.md): performance campaign and execution ledger.
- [`PLAN.md`](PLAN.md): artifact design, source evidence, rental, conversion, runtime, and acceptance history.
- [`evidence/capacity-marlin-20260814/`](evidence/capacity-marlin-20260814/): final-image startup, API, benchmark, Compose, and long-context evidence with a SHA-256 manifest.
- Hugging Face branch `evidence-quant-frontier-screen-20260813`: immutable full-screen and recipe evidence.
- Hugging Face revision `frontier-20260813/quality` at `12035985bf555d0ddc603c6305586a8fa915589c`: immutable artifact and candidate manifest.

## Safe public claims

The evidence supports these bounded statements:

1. On server60's four PCIe-connected RTX 3090s, the pinned projection-sensitive DeepSeek V4 WNA16 artifact served 230,144 context tokens through the pinned custom vLLM runtime.
2. The final profile measured 61.91 decode tokens/s and 875.93 cache-busted prefill tokens/s under the recorded protocol.
3. Replacing retained BF16 `wo_a` copies with an FP8 Marlin projection reclaimed enough memory to increase advertised context from 131,072 to 230,144 without reducing measured decode speed.
4. The final image retrieved an exact needle from a 211,031-token prompt and served two simultaneous 90,029-token retrieval requests.
5. The profile operates close to the physical VRAM limit and should not be described as having a 1 GiB safety margin.

Do not generalize these numbers to stock vLLM, other DeepSeek revisions, other RTX 3090 hosts, or other quantization formats without fresh evidence.
