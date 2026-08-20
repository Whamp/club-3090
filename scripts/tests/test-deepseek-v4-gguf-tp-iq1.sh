#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
owner="$repo_root/models/deepseek-v4-flash-0731/vllm/gguf-tp"
identity="$repo_root/.research/gguf-tp-iq1-unsloth/ARTIFACT-IDENTITIES.json"

python3 - "$owner" "$identity" <<'PY'
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile

owner = pathlib.Path(sys.argv[1])
identity_path = pathlib.Path(sys.argv[2])
profiles_path = owner / "UNSLOTH-IQ1-SOURCE-PROFILES.json"
materializer_path = owner / "materialize-model-view.py"

identity = json.loads(identity_path.read_text())
profiles_bytes = profiles_path.read_bytes()
profiles = json.loads(profiles_bytes)
assert hashlib.sha256(profiles_bytes).hexdigest() == identity["source_profiles"]["sha256"]
assert set(profiles) == {"UD-IQ1_S", "UD-IQ1_M"}
for variant, spec in identity["variants"].items():
    assert profiles[variant]["source_quant_types_sha256"] == spec["source_quant_types_sha256"]
    assert len(profiles[variant]["source_quant_types"]) == 620
    assert sum(shard["tensor_count"] for shard in spec["shards"]) == 1328
    assert [shard["tensor_count"] for shard in spec["shards"]][0] == 0

module_spec = importlib.util.spec_from_file_location("gguf_tp_materializer", materializer_path)
assert module_spec is not None and module_spec.loader is not None
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)

base = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "club_3090_lowbit": {"source_quantization_method": "fp8"},
    "quantization_config": {"quant_method": "compressed-tensors"},
}
base_bytes = json.dumps(base, indent=2).encode()

def derived_bytes(variant: str | None) -> bytes:
    value = dict(base)
    quantization = {"quant_method": "gguf_dsv4"}
    if variant is not None:
        quantization.update(profiles[variant])
    value["quantization_config"] = quantization
    value["quant_method"] = "gguf_dsv4"
    lowbit = value["club_3090_lowbit"]
    lowbit["source_quantization_method"] = "compressed-tensors"
    return json.dumps(value, indent=2).encode()

module.EXPECTED_BASE_CONFIG_SHA256 = hashlib.sha256(base_bytes).hexdigest()
module.EXPECTED_CONFIG_SHA256 = hashlib.sha256(derived_bytes(None)).hexdigest()
module.EXPECTED_VARIANT_CONFIG_SHA256 = {
    variant: hashlib.sha256(derived_bytes(variant)).hexdigest()
    for variant in profiles
}

with tempfile.TemporaryDirectory() as temporary:
    root = pathlib.Path(temporary)
    snapshot = root / "snapshot"
    blobs = root / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    (snapshot / "config.json").write_bytes(base_bytes)
    for blob in module.BLOB_SYMLINKS.values():
        (blobs / blob).touch()

    for variant in (None, "UD-IQ1_S", "UD-IQ1_M"):
        output = root / (variant or "antirez")
        argv = [str(materializer_path), str(snapshot), str(blobs), str(output)]
        if variant is not None:
            argv.append(variant)
        sys.argv = argv
        module.main()
        actual = (output / "config.json").read_bytes()
        assert actual == derived_bytes(variant)
        assert not actual.endswith(b"\n")

    sys.argv = [
        str(materializer_path),
        str(snapshot),
        str(blobs),
        str(root / "bad"),
        "UD-IQ1_X",
    ]
    try:
        module.main()
    except SystemExit as error:
        assert "unsupported GGUF variant" in str(error)
    else:
        raise AssertionError("unsupported variant was accepted")

print("OK DeepSeek V4 GGUF-TP IQ1 identity and model-view contract")
PY
