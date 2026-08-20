# GGUF-TP cold-expert offload route study

Status: the static-routing layers already fail the preregistered offload gate. Full dynamic-layer capture remains pending while another user-owned model occupies server60's GPUs.

## Decision so far

Do not implement the proposed `H <= 224` cold-expert cache.

The gate required at least 99% routed-expert visit coverage with no more than 224 of 256 experts resident in every layer. Layers 0–2 use immutable GGUF `tid2eid` tables, so their routes can be derived exactly from rendered token IDs without running the model. All three layers fail on both workloads:

| Workload | Tokens | Layer 0 H99 | Layer 1 H99 | Layer 2 H99 |
| --- | ---: | ---: | ---: | ---: |
| SuperJSON pilot final context | 24,916 | 249 | 248 | 249 |
| 12-task coding-agent corpus | 548,850 | 251 | 251 | 251 |

One failing layer is enough to reject the design. The other 40 layers cannot make the worst-layer result pass.

This is not a full 43-layer route report yet. The dynamic layers still require model execution to capture their selected experts. That missing work does not weaken the `H <= 224` rejection, but it matters if we later study a shallower cache or a different offload policy.

## Workloads and provenance

The pilot replay comes from the passed one-worker SuperJSON DeepSWE run. Pi's active context after compaction contains 55 messages and renders to 24,916 tokens.

The second workload is the completed 12-task GGUF-TP coding-agent campaign. It contains 8.70 aggregate agent-hours across 12 valid cells. Two discarded concurrency probes are excluded. The 12 final active contexts render to 548,850 tokens total; the largest contains 88,022 tokens.

`build_deepswe_route_replay.py` reproduces Pi 0.84's version-3 session tree and compaction ordering. Every derived workload matched its captured second real provider request before rendering. The metadata files bind the source session, both provider-request fixtures, and the derived request by SHA-256.

The CPU-only renderer used the production GGUF-TP image and tokenizer assets with no GPU or network access. `render-manifest.json` binds all 12 request and token-ID files. The three static routing tables are exact byte slices from immutable Antirez GGUF SHA-256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`:

| Layer | Tensor | SHA-256 |
| ---: | --- | --- |
| 0 | `blk.0.ffn_gate_tid2eid.weight` | `da364a46796cbc4a6fb2616272f3f92dcd7fc009c87202a8f25eac22ad5da5c1` |
| 1 | `blk.1.ffn_gate_tid2eid.weight` | `c4bc1f1a1bd00f236a68e898ee0054a989b35e1099928463cbf5802a6be9618c` |
| 2 | `blk.2.ffn_gate_tid2eid.weight` | `505c47590094064f2136f3e347ebe7492fc1e9c6623dd93095e9973a64d07494` |

Each table is little-endian I32 with shape `[6, 129280]`. `build_static_route_workload.py` validates table size, expert range, and six unique experts per token before analysis.

## Coverage and temporal reuse

Static top-224 coverage is well below the 99% target:

| Workload | Layer | Coverage at H=224 | Coverage at H=248 | LRU hit rate at H=224 | LRU hit rate at H=248 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pilot | 0 | 94.418% | 98.900% | 95.544% | 99.039% |
| Pilot | 1 | 94.473% | 99.042% | 95.492% | 99.021% |
| Pilot | 2 | 94.578% | 98.971% | 95.685% | 99.068% |
| 12-task corpus | 0 | 93.034% | 98.560% | 95.264% | 99.035% |
| 12-task corpus | 1 | 93.209% | 98.510% | 95.284% | 99.041% |
| 12-task corpus | 2 | 93.417% | 98.575% | 95.498% | 99.073% |

Consecutive tokens share only 2.2–3.0% of their six-expert sets on average. Exact set repeats occur in 0.052% of pilot transitions and 0.552% of 12-task transitions. A 224-entry LRU still misses about 4.5–4.7% of expert visits.

At six experts per token across 43 layers, that miss rate implies roughly 11–12 cold expert reads per decoded token. The proposed cache therefore misses both requirements: it fails the 99% hit gate and would put frequent transfers on the decode dependency chain.

A 248-entry LRU reaches about 99% in these three layers, but it offloads only eight experts per layer. That frees about one quarter of the planned `H=224` memory and cannot deliver the intended context gain.

The top-expert sets also move between workloads. At H=224, pilot-to-corpus hot-set Jaccard is 0.851–0.859, and a pilot-derived set covers only 91.6–92.0% of corpus visits. A fixed hot set learned from one session would perform worse than each session's own best set.

## Capture implementation and runtime check

Whamp/vLLM branch `research/gguf-tp-route-stats` contains the opt-in capture implementation:

- `e0646f991` adds per-layer expert histograms and an optional decode ring.
- `761b48a44` gates the ring separately because its roughly 34 MiB per rank made the 148K production profile miss the KV fit gate.

The histogram-only image `sha256:5936741ac164bdcce639f728634bf9aa2c1c2c370792122e768d6e9c36fbbd25` reached API health at 148K and emitted one snapshot per TP rank. All four snapshots were identical: 43 layers, 256 experts, and 265 routed token rows per layer. This proves capture dispatch and rank consistency, not workload skew.

The live capture cannot resume while Compose project `autoround-int4` serves Qwen3.8-27B on all four GPUs. That service was healthy on port 8098 when this report was written and was left untouched. The stopped GGUF-TP capture container has restart disabled, so it will not contend for GPUs.

## Files

- `build_deepswe_route_replay.py`: compaction-aware Pi session replay builder
- `build_static_route_workload.py`: exact static-layer route and reuse summarizer
- `analyze_route_skew.py`: coverage, stability, LRU, and decision analysis
- `deepswe-pilot-static-routes.json`: pilot static-layer summary
- `deepswe-12task-static-routes.json`: 12-task static-layer summary
- `static-layers-analysis.json`: full H=1..256 curves and cross-workload metrics
- `static-layers-analysis.md`: generated selected-point table
- `startup-smoke/`: four rank-local runtime capture snapshots

## Remaining work

The approved fusion study still needs its four-GPU Nsight layer-slice trace. No fusion kernel work should start before that trace confirms the launch and dependency-latency hypothesis. The trace and any dynamic-layer capture must wait for an intentional server60 maintenance window after the Qwen service finishes.
