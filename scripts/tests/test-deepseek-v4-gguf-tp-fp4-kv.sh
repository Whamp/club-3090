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
manifest = json.loads((owner / "FP4-MANIFEST.json").read_text())
base = (root / "models/deepseek-v4-flash-0731/vllm/compose/multi4/gguf-tp/base.yml").read_text()
fp4 = (root / "models/deepseek-v4-flash-0731/vllm/compose/multi4/gguf-tp/fp4.yml").read_text()
builder = (owner / "build-fp4-kv-image.sh").read_text()
dockerfile = (owner / "Dockerfile.fp4-kv").read_text()

sha = re.compile(r"^[0-9a-f]{64}$")
digest = re.compile(r"^sha256:[0-9a-f]{64}$")

assert manifest["schema_version"] == 1
layout = manifest["format"]
assert layout == {
    "name": "fp4_ds_mla",
    "nope_values": 448,
    "nope_encoding": "E2M1 packed low-nibble first, group-32 UE8M0 scales",
    "rope_values": 64,
    "rope_encoding": "BF16 unchanged",
    "token_data_bytes": 352,
    "scale_bytes": 16,
    "row_bytes": 368,
}
assert manifest["runtime_base"]["digest"] == (
    "sha256:f91e8283e7ad116b8664b4a936dba88ebafcb8910a968dce2a3c34420f010adf"
)
assert manifest["vllm"]["commit"] == "633815f6889d9d033aefa04bf40cb270d5b6a3f1"
assert manifest["vllm"]["tree"] == "2230f7d43768e45fab2547bea056c9df160aab45"
assert manifest["flash_mla"]["commit"] == "81a06aa6feb608bcba687a40acf60ee87d14f2da"
assert manifest["flash_mla"]["tree"] == "134841dfba0487c3db9f996817c94c16eca972b3"
assert sha.fullmatch(manifest["flash_mla"]["wheel_sha256"])
assert sha.fullmatch(manifest["flash_mla"]["native_library_sha256"])
assert sha.fullmatch(manifest["stable_extension"]["sha256"])
assert digest.fullmatch(manifest["image"]["digest"])

runtime_files = manifest["vllm"]["runtime_files"]
assert len(runtime_files) == 14
assert len({entry["path"] for entry in runtime_files}) == len(runtime_files)
assert all(sha.fullmatch(entry["sha256"]) for entry in runtime_files)
assert all(not entry["path"].startswith("tests/") for entry in runtime_files)

profile = manifest["profile"]
assert profile["kv_cache_dtype"] == "fp4_ds_mla"
assert profile["max_model_len"] == 148000
assert profile["max_num_seqs"] <= 2
assert profile["max_num_batched_tokens"] == 256
assert profile["gpu_memory_utilization"] == 0.98

assert "- \"fp8_ds_mla\"" in base
assert "- \"fp4_ds_mla\"" not in base
assert manifest["runtime_base"]["digest"] in base
assert "fp4.yml   fp4_ds_mla" in base

assert "- \"fp4_ds_mla\"" in fp4
assert "- \"fp8_ds_mla\"" not in fp4
assert manifest["image"]["digest"] in fp4
assert "${CLUB3090_RESTART:-unless-stopped}" in fp4
assert "- \"2\"" in fp4
assert "- \"148000\"" in fp4
assert "31 MiB/card" in fp4
assert "fp4.yml   fp4_ds_mla" in fp4

for required in (
    "VLLM_SOURCE_DIR",
    "GGUF_TP_FP4_EXTENSION",
    "FLASH_MLA_FP4_WHEEL",
    "sha256sum -c",
    "runtime_files",
    "ACTUAL_COMMIT",
    "ACTUAL_TREE",
    "EXPECTED_IMAGE_DIGEST",
):
    assert required in builder
assert "COPY root/ /workspace/vllm/" in dockerfile
assert "--reinstall --no-deps" in dockerfile
assert manifest["vllm"]["commit"] in dockerfile
assert manifest["flash_mla"]["commit"] in dockerfile
assert "TBD" not in fp4
assert "TBD" not in (owner / "FP4-MANIFEST.json").read_text()
print("PASS: DeepSeek V4 GGUF-TP FP4 delivery contract")
PY

echo "test-deepseek-v4-gguf-tp-fp4-kv: ok"
