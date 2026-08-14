# DeepSeek V4 Flash internals

This page records the engineering behind
`llamacpp/deepseek-flash-multi4-antirez-iq2-fast-prefill`. The profile is an
incubating specialist, not a replacement for the stock llama.cpp DeepSeek
profiles.

## Validated configuration

| Component | Validated value |
|---|---|
| GPUs | 4× RTX 3090, PCIe only, sm_86 |
| Engine base | [`alesha-pro/llama.cpp` `ds4-longctx` at `b001c8cd7`](https://github.com/alesha-pro/llama.cpp/commit/b001c8cd73855c9c4d5d89226f08179b6f3417d6) |
| Q8 repair | [`Whamp/llama.cpp` `0379cf4bf`](https://github.com/Whamp/llama.cpp/commit/0379cf4bf889f3d28038a005210c4bc193fc8ba1) |
| Published image | `ghcr.io/whamp/llama-cpp-ds4-longctx@sha256:a96bd947d63eb81d8baf9f6f5ecb26669476383976717237450fbb5727b03745` |
| Weights | [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf) IQ2_XXS, revision `e7f04037032990db0346398d249baf9fb9df1ccc` |
| KV cache | matching `q8_0` K and V |
| Context | 430,080 tokens reserved; 395,282-token exact-recall validation |
| Split | layer, `1,1,0.95,1.05` |
| Power | observed at the rig's unchanged 230 W/card setting |
| Reasoning controls | off, or thinking at `low` (default), `high`, or `max` effort |

The engine image is separately named and digest-pinned in
`scripts/lib/profiles/engines/llama-cpp-ds4-longctx.yml`. It does not replace
`llama-cpp-local`. See [`docs/UPSTREAM.md`](../../docs/UPSTREAM.md) for the pin,
source dependency, and retirement trigger.

## Graded reasoning is a profile capability

Most club-3090 profiles expose reasoning as a binary on/off choice. This
profile also preserves DeepSeek V4 Flash 0731's three official thinking levels:
`low`, `high`, and `max`. The registry exports those levels and identifies
`low` as the local default when thinking is enabled.

Pass both controls through `chat_template_kwargs`:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true,
    "reasoning_effort": "max"
  }
}
```

The current llama-server ignores a top-level `reasoning_effort` field. Clients
must use the nested form above. A live prompt-rendering probe on the published
image measured 11 prompt tokens for off and low, 90 for high, and 103 for max.
The distinct high/max lengths confirm that the endpoint injects both official
prefixes instead of silently collapsing the levels.

The effort setting and output budget are separate. `low` adds no prefix;
`high` and `max` prepend different instructions before the system message.
Full 8-pack runs exercised all three levels:

| Effort | Output cap | Pass@1 | Pass@3 | Cap audit |
|---|---:|---:|---:|---|
| `low` | 16,384 | 121/150 | 131/150 | no cap hits |
| `high` | 65,536 | 121/150 | 132/150 | no cap hits |
| `max` | 65,536 | 123/150\* | 130/150 | 4 of 226 API responses hit the cap |

\* The four capped `max` responses ended with `finish_reason=length`; all were
in CLI-40. The score is retained with this limitation rather than rerunning at
an even larger budget.

These scores are regression evidence for the quantized serving path, not a
measure of DeepSeek V4 Flash's intelligence ceiling. The same Antirez weights
and Q8 cache scored 122/150 at stock `b10200`'s default `low` effort, one point
above the fork. The thinking-off comparison was similarly close: 111/150 on
stock versus 109/150 on the fork. Several failed cases also overlap open
benchlocal-cli benchmark-definition problems in DataExtract, StructOutput, and
ReasonMath. DeepSeek's [official model
card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) documents the
three effort levels and reports its code-agent evaluations at `max` with
temperature 1.0 and top-p 0.95.

## Why this fork is faster

The fork's main gain is capture-always CUDA graphs for prefill micro-batches.
The stock path submits roughly thousands of CUDA kernels per micro-batch from
one host thread. The fork captures that work and replays it with one graph
launch. Its stable-topology graph builder, resident MoE scheduler, fused MMQ,
sparse attention, and allocator controls keep the captured graph reusable.

This is a system, not a one-commit optimization. A minimal graph-only port onto
the stock `b10200` image captured and launched graphs but produced no speedup.
Adding only stable-topology padding improved stock prefill by about 22% at 32K,
but the remaining graph churn still prevented the capture-always gain. The
incubating profile therefore uses the separately named external engine rather
than presenting a partial patch as mainline llama.cpp.

The validated compose enables all of the fork's linked `DSV4_*` paths. Do not
remove one as an isolated cleanup: several are prerequisites for reusable graph
capture.

## Q8 repair

The old fork forced DeepSeek V4 to F16 KV because its Q8 path produced corrupt
output. Q8 was not fundamentally incompatible with the architecture. Four
storage-path defects caused the failure.

1. **CUDA concat rejected packed Q8 rows.** DeepSeek V4 concatenates raw and
   compressed cache rows on every attention layer. The fork accepted only F32
   and F16 on CUDA, so Q8 fell back to CPU.
2. **The CPU concat fallback copied logical elements as packed blocks.** It
   advanced by `ggml_type_size` for each logical Q8 element instead of iterating
   block storage. This corrupted the expected-value path as well as runtime
   fallback.
3. **Padding converted F16 to Q8 on CPU.** The fork's CUDA copy kernel supports
   F32-to-Q8, not F16-to-Q8. Filling padding in F32 keeps the conversion on CUDA
   and reduces prefill graph splits from 87 to 5, matching F16.
4. **Long-context gathers forced Q8 rows to F16.** Two compressed-attention paths
   cast gathered rows to F16 and then concatenated them with raw Q8 cache rows.
   Multi-chunk prompts eventually hit a type assertion. Preserving the source
   cache type fixes the boundary.

Commit `0379cf4bf` repairs both concat backends, adds the required strided-view
predicate, keeps padding conversion on CUDA, preserves the cache type in both
gathers, and adds Q8 concat regression cases. The CUDA backend test passes
45/45 concat cases, including packed Q8 and strided layer views.

The validated path still sets `LLAMA_ATTN_ROT_DISABLE=1`. Rotation-enabled Q8
was not established during this work. The compose also requires matching Q8 K
and V; mixed cache types are outside the validated contract.

## Warm-up is part of readiness

The speedup requires one synthetic prefill to `context_size - 512` after every
server start. This does not cache a user's future conversation. It prepares the
allocator and CUDA graphs at the deepest configured shape. Without it, the first
request reaching a new depth pays allocation and graph setup and can run at less
than half the steady-state rate.

At the 430,080-token default, warm-up processed 429,568 tokens at 276.4 prefill
tokens/s and took about 26 minutes. A 200K configuration warmed in about eight
minutes.

The profile-local `serve-fast-prefill.sh` supervisor starts llama-server on a
private internal port, performs the warm-up, and only then starts the TCP
forwarder on the public port. The normal launcher therefore cannot report the
profile ready before graph preparation finishes. The registry exports a
2,400-second readiness timeout for this slug.

This profile suits one long-lived coding-agent session. It is a poor fit for
frequent model switching.

## Performance

All measurements below came from the same 4× RTX 3090 rig at its unchanged
230 W/card setting.

### 430K profile

| Probe | Result |
|---|---:|
| Warm-up | 429,568 tokens at 276.4 prefill tokens/s |
| Fresh prompt at 263K | 1,056 prefill tokens/s, exact needle recall |
| Fresh prompt at 395,282 | 913 prefill tokens/s, exact needle recall |
| Minimum observed free VRAM | 794 MiB |

The 794 MiB floor is 44 MiB above the experiment's accepted 750 MiB floor but
below the repo's normal 1,024 MiB production guard. That narrow margin is one
reason the profile remains incubating. Set `CTX_SIZE=200000` for the higher-margin
fallback.

### Q8 versus F16 at 200K

| Fresh prompt depth | Repaired Q8 prefill | F16 prefill | Q8 difference |
|---:|---:|---:|---:|
| 98K | 1,538 tokens/s | 1,414 tokens/s | +8.8% |
| 130K | 1,475 tokens/s | 1,360 tokens/s | +8.5% |
| 180K | 1,374 tokens/s | 1,334 tokens/s | +3.0% |
| 195K | 1,341 tokens/s | 1,306 tokens/s | +2.7% |

Q8 recovered the full graph path and was 3–9% faster than F16 at depth. The
stock `b10200` Q8 router was much slower on repeated deep prefill; in the
measured agent loop it took 21.8 seconds to first token at 33K, versus about
7.8 seconds on the repaired fork.

### Coding-agent loop

A 15-turn fixture grew from 1,140 to 54,146 prompt tokens. Time to first token
grew from the shallow start to 9.5 seconds at 54K while decode held 33–38
tokens/s.
Across the measured curve, context grew about 40× while TTFT grew 4.2×. The
stock Q8 baseline's measured TTFT grew nearly linearly with context.

## Validation

The clean image built from `0379cf4bf` passed:

- the final catalog launch through `switch.sh --force` pulled the published
  digest, kept its public port closed during warm-up, processed 429,568 tokens
  at 276.28 prefill tokens/s, became ready after 1,628 seconds, and passed `verify-full.sh`
  9/9 on port 8033;
- `verify-full.sh`: 9/9 at 200K and 430K on the pre-publication validation builds;
- `verify-stress.sh`: 8/8 at 200K, including exact recall at 9K, 27K, 55K,
  85K, and the 184K ceiling;
- 430K fast stress gate with a 750 MiB floor: exact recall at 263K and 395,282,
  with 796 MiB free throughout the ceiling ladder;
- `quality-test.sh --medium`: 67/75 pass@1, equal to the stock `b10200` Q8
  baseline;
- full 8-pack thinking-off: 109/150 after replacing an invalid network-blocked
  HermesAgent 0/20 leg with its reachable 14/20 rerun; the same weights and Q8
  cache scored 111/150 on stock `b10200`;
- full 8-pack thinking-on: `low` 121/150 at a 16,384-token output cap, `high`
  121/150 at 65,536, and `max` 123/150 at 65,536; four of 226 `max` API
  responses hit the cap, while `low` and `high` had no cap hits;
- stock `b10200` scored 122/150 at default `low` effort with the same weights
  and Q8 cache; together with the 109/150 fork versus 111/150 stock
  thinking-off comparison, this is the serving/quantization regression check;
- live reasoning-effort rendering: off 11 prompt tokens, low 11, high 90, max
  103;
- `soak-test.sh`: 25/25 responses, zero errors, zero silent-empty responses;
- `bench-agentic.sh`: all 15 turns through 54K;
- Q8 CUDA concat regression: 45/45.

The 430K run used the fast stress mode: the near-ceiling ladder and functional
probes ran, while the redundant 60K/90K intermediate needle section was skipped.
The complete stress gate ran at 200K.

## Quant constraint

The fast MoE capture path is quant-sensitive. The validated Antirez artifact
uses IQ2_XXS for routed-expert gate/up weights and Q2_K for routed-expert down
weights. The locally available Unsloth IQ1_M and role-splice variants contain
IQ1_M expert matrices that failed the MMQ graph-safety check. They are not drop-in
replacements for this profile.

## Rejected paths

- **Minimal graph forward-port to stock `b10200`:** graphs launched but did not
  produce the target gain; incomplete stable topology and scheduler behavior
  kept rebuilding the work.
- **F16-only fork profile:** correct and fast, but Q8 is now equally correct,
  slightly faster, and holds more context.
- **Q8 with concat repair only:** coherent on short prompts but produced 87 graph
  splits and ran at about one-third of F16 speed at depth.
- **Q8 after concat and padding repair only:** fast, but long prompts crossing
  micro-batch boundaries hit a cache-type assertion until the gather fix landed.
- **IQ1_M weights:** incompatible with the validated MMQ capture path.
- **DwarfStar CUDA tensor parallelism:** the audited implementation cannot load
  this GGUF on four 24 GiB cards because three selective weight slabs exceed
  24 GiB after its required 2 GiB runtime reserve. See
  [`DWARFSTAR-CUDA-TP-DEAD-END.md`](DWARFSTAR-CUDA-TP-DEAD-END.md) for the pinned
  source audit, exact tensor accounting, and CPU planner results.

Keep these failures in mind when changing the engine base, cache operations,
quant, or warm-up. Short smoke prompts are insufficient: the bugs appeared only
at depth or across micro-batch boundaries.
