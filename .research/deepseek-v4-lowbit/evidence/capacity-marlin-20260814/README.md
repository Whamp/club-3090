# Final quality-capacity runtime evidence

These files came from server60's clean, checksum-pinned final image run on 2026-08-14. Verify them with:

```bash
sha256sum --check MANIFEST.sha256
```

| File | Evidence |
| --- | --- |
| `startup.log.gz` | Model load, KV allocation, graph memory, and readiness |
| `resolved-compose.yml` | Fully rendered final service contract |
| `contracts.sha256` | Source contract hashes recorded before launch |
| `models.json` | Served model identity and 230,144-token context |
| `canary-basic.json` | Deterministic basic generation |
| `canary-tool-first.json` | Automatic `add` tool selection |
| `canary-tool-second.json` | Normal post-tool continuation after the DSML fix |
| `bench.log` | Fixed 3-warmup/5-measured decode and cache-busted prefill run |
| `performance-summary.json` | Parsed benchmark result and campaign floors |
| `final-longctx-211k.json` | Exact retrieval from a 211,031-token prompt |
| `compose-up.txt.gz` | Final Compose recreation result |

The evidence excludes credentials. Host-specific paths and addresses are expected because club-3090 owns this machine-specific deployment.
