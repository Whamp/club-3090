#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONUTF8="${PYTHONUTF8:-1}"
python3 - <<'PY'
import json
import re
from pathlib import Path

root = Path.cwd()
owner = root / "models/deepseek-v4-flash-0731/vllm/gguf-tp"
compose_dir = root / "models/deepseek-v4-flash-0731/vllm/compose/multi4/gguf-tp"
manifest_path = owner / "FP4-INDEXER-MANIFEST.json"
manifest = json.loads(manifest_path.read_text())
compose = (compose_dir / "fp4-indexer.yml").read_text()
builder = (owner / "build-fp4-indexer-image.sh").read_text()
dockerfile = (owner / "Dockerfile.fp4-indexer").read_text()

sha = re.compile(r"^[0-9a-f]{64}$")
digest = re.compile(r"^sha256:[0-9a-f]{64}$")
assert manifest["schema_version"] == 1
assert manifest["format"] == {
    "name": "mxfp4_indexer_sm86",
    "semantic_values": 128,
    "encoding": "E2M1 packed low-nibble first, group-32 UE8M0 scales",
    "packed_value_bytes": 64,
    "scale_bytes": 4,
    "physical_row_bytes": 68,
}
assert manifest["runtime_base"]["digest"] == (
    "sha256:eb94d5049bf4d8d55c335ac1d2445382a811b7312d28e3e73088011a8103e181"
)
assert manifest["vllm"]["commit"] == "ccd463e6d66f781156352b040da7440db85b2625"
assert manifest["vllm"]["tree"] == "de72d166b53148a65505cd2972d204faaf60ac29"
assert manifest["vllm"]["runtime_source_sha256"] == "5ecd90e95569"
runtime_files = manifest["vllm"]["runtime_files"]
assert len(runtime_files) == 11
assert len({entry["path"] for entry in runtime_files}) == 11
assert all(sha.fullmatch(entry["sha256"]) for entry in runtime_files)
assert all(entry["path"].startswith("vllm/") for entry in runtime_files)
assert digest.fullmatch(manifest["image"]["digest"])

profile = manifest["profile"]
assert profile["max_model_len"] == 200000
assert profile["max_num_seqs"] == 2
assert profile["max_num_batched_tokens"] == 256
assert profile["gpu_memory_utilization"] == 0.98
assert profile["kv_cache_dtype"] == "fp4_ds_mla"
assert profile["use_fp4_indexer_cache"] is True
assert manifest["validation"]["validated_prompt_tokens"] == 195812
assert manifest["validation"]["release_safe"] is False

assert manifest["image"]["digest"] in compose
assert "- \"fp4_ds_mla\"" in compose
assert "- \"200000\"" in compose
assert "- \"2\"" in compose
assert "--attention-config" in compose
assert "use_fp4_indexer_cache" in compose
assert "${CLUB3090_RESTART:-unless-stopped}" in compose
assert "25-26 MiB/card" in compose
assert "capacity experiment" in compose

for sibling in ("base.yml", "fp4.yml", "fp4-indexer.yml"):
    text = (compose_dir / sibling).read_text()
    assert "base.yml  fp8_ds_mla" in text
    assert "fp4.yml   fp4_ds_mla" in text
    assert "fp4-indexer.yml  fp4_ds_mla + MXFP4 indexer" in text

for required in (
    "VLLM_SOURCE_DIR",
    "runtime_files",
    "ACTUAL_COMMIT",
    "ACTUAL_TREE",
    "sha256",
    "EXPECTED_IMAGE_DIGEST",
    "--provenance=false",
):
    assert required in builder
assert "COPY root/ /workspace/vllm/" in dockerfile
assert manifest["vllm"]["commit"] in dockerfile or "VLLM_COMMIT" in dockerfile
assert "TBD" not in manifest_path.read_text()
assert "TBD" not in compose
print("PASS: DeepSeek V4 SM86 MXFP4 indexer delivery contract")
PY

echo "test-deepseek-v4-gguf-tp-fp4-indexer: ok"
