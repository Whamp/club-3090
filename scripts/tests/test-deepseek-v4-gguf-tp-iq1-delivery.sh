#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
owner="$repo_root/models/deepseek-v4-flash-0731/vllm/gguf-tp"
compose_dir="$repo_root/models/deepseek-v4-flash-0731/vllm/compose/multi4/gguf-tp"

python3 - "$owner" "$compose_dir" <<'PY'
import json
import pathlib
import sys

owner = pathlib.Path(sys.argv[1])
compose_dir = pathlib.Path(sys.argv[2])
manifest = json.loads((owner / "IQ1-MANIFEST.json").read_text())
dockerfile = (owner / "Dockerfile.iq1").read_text()
builder = (owner / "build-iq1-image.sh").read_text()

assert manifest["schema_version"] == 1
assert manifest["runtime_base"] == {
    "ref": "club-3090/deepseek-v4-gguf-tp:fp4-633815f6",
    "digest": "sha256:eb94d5049bf4d8d55c335ac1d2445382a811b7312d28e3e73088011a8103e181",
    "vllm_commit": "633815f6889d9d033aefa04bf40cb270d5b6a3f1",
    "vllm_tree": "2230f7d43768e45fab2547bea056c9df160aab45",
}
assert manifest["vllm"]["commit"] == "e34fb09c712733845f913381e6a284a9c4d5d2a8"
assert manifest["vllm"]["tree"] == "411191a433131f3998c0b3316ff8eeaa4e896fdd"
assert manifest["stable_extension"]["sha256"] == (
    "81d459f2d0560fae37ad8e545876dc34c27b38e5b7608b9d73f1dd2bc132ebb8"
)
assert manifest["stable_extension"]["cuda_arch"] == "sm_86"
assert len(manifest["vllm"]["runtime_files"]) == 8
assert len({entry["path"] for entry in manifest["vllm"]["runtime_files"]}) == 8
assert all(len(entry["sha256"]) == 64 for entry in manifest["vllm"]["runtime_files"])

assert manifest["artifact_repository"] == {
    "repo": "unsloth/DeepSeek-V4-Flash-0731-GGUF",
    "revision": "109848da2469efe1f1aab9e11acea08a065ccd4f",
    "tensor_count": 1328,
}
assert set(manifest["variants"]) == {"UD-IQ1_S", "UD-IQ1_M"}
expected_counts = {"UD-IQ1_S": [0, 812, 516], "UD-IQ1_M": [0, 756, 572]}
for variant, counts in expected_counts.items():
    spec = manifest["variants"][variant]
    assert len(spec["source_quant_types_sha256"]) == 64
    assert len(spec["model_view_config_sha256"]) == 64
    assert len(spec["shards"]) == 3
    assert [shard["tensor_count"] for shard in spec["shards"]] == counts
    assert sum(counts) == 1328
    assert all(len(shard["sha256"]) == 64 for shard in spec["shards"])
    assert all(shard["file_size"] > 0 for shard in spec["shards"])

assert manifest["excluded_overlays"] == {
    "vllm_commit": "ccd463e6d66f781156352b040da7440db85b2625",
    "image_digest": "sha256:c089ee30367a0a38b62fc35943f1d8b6c2e0d131c609f3e27baa9816e57ca53c",
    "reason": "Keep initial IQ1 validation independent of the MXFP4-indexer experiment.",
}
assert manifest["image"]["digest"].startswith("sha256:")
assert manifest["image"]["digest"] != "sha256:TBD"
assert "TBD" not in (owner / "IQ1-MANIFEST.json").read_text()

assert "ARG BASE_IMAGE=club-3090/deepseek-v4-gguf-tp:fp4-633815f6" in dockerfile
assert "COPY root/ /workspace/vllm/" in dockerfile
assert "COPY _C_stable_libtorch.abi3.so" in dockerfile
assert manifest["vllm"]["commit"] in dockerfile
assert manifest["stable_extension"]["sha256"] in dockerfile
assert manifest["runtime_base"]["digest"] in dockerfile

for required in (
    "IQ1-MANIFEST.json",
    "VLLM_SOURCE_DIR",
    "GGUF_TP_IQ1_EXTENSION",
    "runtime_files",
    "stable extension hash mismatch",
    "runtime base",
):
    assert required in builder

for variant, filename, model_id in (
    ("UD-IQ1_S", "iq1-s.yml", "deepseek-v4-flash-0731-gguf-tp-iq1-s"),
    ("UD-IQ1_M", "iq1-m.yml", "deepseek-v4-flash-0731-gguf-tp-iq1-m"),
):
    compose = (compose_dir / filename).read_text()
    assert manifest["image"]["digest"] in compose
    assert model_id in compose
    assert "--max-num-seqs" in compose and '\n      - "2"' in compose
    assert "--max-model-len" in compose and '\n      - "148000"' in compose
    assert "--kv-cache-dtype" in compose and "fp8_ds_mla" in compose
    assert "${CLUB3090_RESTART:-unless-stopped}" in compose
    assert (spec_text := json.dumps(manifest["variants"][variant]["shards"][0]["sha256"]))
    assert spec_text.strip('"') in compose
    assert manifest["variants"][variant]["source_quant_types_sha256"] in compose
    assert compose.count(".gguf:ro") == 3

print("OK DeepSeek V4 GGUF-TP IQ1 delivery contract")
PY
