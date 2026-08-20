# DwarfStar CUDA tensor parallelism: explored dead end

**Status:** Rejected for the PR #940 four-RTX-3090 target at the audited revisions below. This is a fit failure in the current implementation, not a claim that DwarfStar CUDA tensor parallelism can never support 24 GiB cards.

## Question

Could [`antirez/ds4`](https://github.com/antirez/ds4) replace or outperform PR #940's custom llama.cpp path by running the same Antirez DeepSeek V4 Flash IQ2_XXS GGUF with CUDA tensor parallelism across four RTX 3090s?

## Audited inputs

| Input | Pinned value |
|---|---|
| DwarfStar | [`antirez/ds4@84cc882352757baf628a1776badf7cc54d584e28`](https://github.com/antirez/ds4/tree/84cc882352757baf628a1776badf7cc54d584e28) |
| PR #940 branch | `Whamp/club-3090@79aeb2dca73d6aeefd418c57bb43aad7573e027d` |
| GGUF | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` |
| GGUF revision | `antirez/deepseek-v4-gguf@e7f04037032990db0346398d249baf9fb9df1ccc` |
| GGUF SHA-256 | `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` |
| GGUF file size | 86,720,111,488 bytes (80.764398 GiB) |
| Parsed tensor payload | 86,714,775,900 bytes (80.759428 GiB), 1,328 tensors |
| Target host | 4× RTX 3090, 24 GiB each, PCIe only |

The investigation was source- and CPU-planner-only. It did not start DwarfStar or alter server60.

## Decision

Do not add a DwarfStar CUDA-TP profile to PR #940 at the audited revision. Current DwarfStar cannot load this GGUF on four 24 GiB cards as written, regardless of context length. PR #940's validated llama.cpp layer-split path remains the viable four-card implementation for this artifact.

The decisive blocker is DwarfStar's selective-weight allocation gate. Its hybrid placement produces three 22.09–22.13 GiB slabs. The runtime requires each slab to fit alongside a hard-coded 2 GiB reserve, so three cards exceed 24 GiB before KV cache or other runtime allocations.

## What DwarfStar calls CUDA tensor parallelism

This path is a two-stage pipeline with an expert-parallel pair attached to each stage, not flat tensor parallelism over all four GPUs.

For server60's two close PHB pairs, the topology-correct logical order would be:

```text
--gpu-devices 0,2,1,3
```

GPUs 0 and 2 would be the pipeline-stage homes. GPUs 1 and 3 would be their respective TP partners. The stage boundary would cross the slower NODE path.

At the pinned revision, DwarfStar:

- splits routed experts 50/50 within each pair;
- row-shards the vocabulary head;
- replicates dense attention, router, shared-expert, and other layer tensors within each pair;
- enables pipelined prefill by default in CUDA-TP mode;
- uses exact fallback execution for unsupported grouped Q2 routed-expert shapes.

The ownership rules are documented in DwarfStar's [CUDA-TP guide](https://github.com/antirez/ds4/blob/84cc882352757baf628a1776badf7cc54d584e28/README.md#tensor-parallelism-across-cuda-gpus) and implemented in its [per-device selective-cache installation](https://github.com/antirez/ds4/blob/84cc882352757baf628a1776badf7cc54d584e28/ds4.c#L56447-L56580).

## Exact weight placement

Parsing every tensor header and applying DwarfStar's current ownership rules gives this best contiguous split: layers 0–20 on stage 0 and layers 21–42 plus the output head on stage 1.

| Logical tier | Selective weight slab | Slab plus 2 GiB runtime reserve | Result against 24 GiB |
|---|---:|---:|---|
| Stage 0 home | 22.090907 GiB | 24.090907 GiB | Does not fit |
| Stage 0 partner | 21.104457 GiB | 23.104457 GiB | Fits only before other allocations |
| Stage 1 home | 22.125278 GiB | 24.125278 GiB | Does not fit |
| Stage 1 partner | 22.125263 GiB | 24.125263 GiB | Does not fit |

The GGUF contains 72.562500 GiB of routed-expert tensors and 6.686476 GiB of other layer tensors. Splitting routed experts but duplicating the other layer tensors raises aggregate selective weight residency from 80.759428 GiB of tensor payload to 87.445904 GiB. The 6.686476 GiB increase is duplicated non-routed layer state.

DwarfStar then requires each selective slab to fit with a [hard-coded 2 GiB post-load reserve](https://github.com/antirez/ds4/blob/84cc882352757baf628a1776badf7cc54d584e28/ds4_cuda.cu#L3983-L4023). Three tiers fail even against ideal 24 GiB nameplate capacity. Real free memory is lower after CUDA context creation.

`--gpu-vram auto` is stricter still: it first subtracts [`max(2 GiB, 5% of free VRAM)`](https://github.com/antirez/ds4/blob/84cc882352757baf628a1776badf7cc54d584e28/ds4_cuda.cu#L25994-L26017), after which placement separately charges safety and workspace costs.

## Planner-only context frontier

DwarfStar exposes CPU test hooks for the production placement math. Feeding all 1,328 exact tensor names and sizes through `ds4_test_classify_multi_tier_with_ctx_cuda_tp_prefill()` produced the following minimum equal per-card budgets:

| Context | Prefill chunk 64 | 128 | 256 | 512 | 1,024 | 2,048 default |
|---:|---:|---:|---:|---:|---:|---:|
| 131,072 | 24,228 MiB | 24,308 MiB | 24,477 MiB | 24,805 MiB | 25,462 MiB | 26,774 MiB |
| 150,000 | 24,361 MiB | 24,442 MiB | 24,617 MiB | 24,954 MiB | 25,629 MiB | 26,978 MiB |
| 180,000 | 24,571 MiB | 24,656 MiB | 24,838 MiB | 25,190 MiB | 25,894 MiB | 27,302 MiB |
| 200,000 | 24,711 MiB | 24,799 MiB | 24,985 MiB | 25,347 MiB | 26,071 MiB | 27,518 MiB |
| 430,080 | 26,323 MiB | 26,438 MiB | 26,681 MiB | 27,155 MiB | 28,103 MiB | 30,000 MiB |

If the independent 2 GiB slab gate is ignored, the synthetic four-24-GiB planner frontiers are:

| Prefill chunk | Maximum planned context |
|---:|---:|
| 64 | 180,783 |
| 128 | 168,807 |
| 256 | 144,551 |
| 512 | 101,959 |
| 1,024 | 30,895 |
| 2,048 | 1,280 |

These are dry-fit results, not runtime proof. They show that smaller prefill chunks can satisfy the planner by reducing scratch, but they cannot bypass the stricter slab allocation gate. They would also trade away prefill throughput.

DwarfStar's CUDA persistent compressed-attention cache compounds the fit problem. The compressed cache uses F16 only on Apple; the pinned CUDA path stores it as F32. Source-formula reconstruction at a 64-token prefill chunk gives approximately 1.70 GiB aggregate persistent cache at 131K context, 2.58 GiB at 200K, and 5.53 GiB at 430K, before other runtime allocations.

## Comparison with PR #940

| Property | DwarfStar CUDA-TP at `84cc882` | PR #940 llama.cpp path |
|---|---|---|
| Exact Antirez IQ2_XXS GGUF | Supported format | Validated artifact |
| Four RTX 3090 fit | No, as written | Yes, measured |
| GPU model residency | 87.445904 GiB selective weights before runtime state | About 79.773 GiB logged GPU model buffers plus 0.986 GiB CPU-mapped |
| Persistent cache | Temporally compressed but F32 on CUDA | Q8_0 K and V |
| Long context on four RTX 3090s | Not established | 430,080 reserved; exact recall at 395,282 |
| Decode on four RTX 3090s | Not measured | 33–38 tokens/s across the measured agent curve |
| Deep prefill on four RTX 3090s | Not measured | 1,341 tokens/s at 195K; 913 tokens/s at 395K |

See [`INTERNALS.md`](INTERNALS.md) for the PR #940 runtime, benchmark protocol, and validation evidence.

DwarfStar's published CUDA-TP reference is not transferable to this target. Its recorded result uses eight 48 GiB L40S cards and a Q4_K expert model: 1,524.84 prefill tokens/s, 46.93 single-stream decode tokens/s, and 126 aggregate tokens/s for a 16-row decode oracle. DwarfStar states that Q2 uses exact fallback for unsupported grouped routed shapes and has lower aggregate throughput. Its `ds4-bench` also reports instantaneous frontier throughput by snapshotting and restoring KV state, not the same request-level metric used by PR #940.

## Conditions for revisiting

Reconsider DwarfStar on four 24 GiB cards only after upstream changes materially alter the fit:

1. Non-routed layer tensors are sharded, moved, or otherwise no longer duplicated within each pair.
2. The placement planner and the 2 GiB selective-slab reserve agree on a configuration that fits real free VRAM.
3. CUDA gains a smaller compressed-cache representation, such as a validated F16 or Q8 path.
4. The exact Q2 GGUF fits at useful context without reducing prefill chunks to throughput-damaging sizes.
5. A runtime trial then proves correctness, selected DIRECT-versus-BOUNCE P2P paths, per-card peak VRAM, context capacity, and matched 3-warmup/5-measured throughput.

Until those gates change, further server60 trials would consume downtime without a plausible load path.
