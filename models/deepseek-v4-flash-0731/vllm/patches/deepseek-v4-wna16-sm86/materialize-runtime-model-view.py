#!/usr/bin/env python3
"""Materialize the validated DeepSeek V4 hybrid-FP8 runtime model view."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

EXPECTED_ARTIFACT_CONFIG_SHA256 = (
    "334bfa9f35a2f05510639538325c20e87a3980b06cabef4d750d3ca8085a0a66"
)
EXPECTED_ARTIFACT_INDEX_SHA256 = (
    "348657275f7e89750555b23b86a117177b487dd8414d00bdd457c67688284735"
)
EXPECTED_RUNTIME_CONFIG_SHA256 = (
    "891883c0c40b28cbec2c9bca6f5e4a8278824fb42ff32695ba4640ebdee7dc91"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, description: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"DeepSeek WNA16 runtime view {description} checksum mismatch: "
            f"got {actual}, expected {expected}"
        )


def require_indexed_model_shards(artifact: Path, index_path: Path) -> None:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("DeepSeek WNA16 runtime view tensor index has no weight_map")
    shard_names = set(weight_map.values())
    for shard_name in shard_names:
        if not isinstance(shard_name, str):
            raise RuntimeError(
                "DeepSeek WNA16 runtime view tensor index has a non-string shard"
            )
        shard_path = Path(shard_name)
        if shard_path.is_absolute() or len(shard_path.parts) != 1:
            raise RuntimeError(
                "DeepSeek WNA16 runtime view tensor index has an unsafe shard path: "
                f"{shard_name!r}"
            )
        if not (artifact / shard_path).is_file():
            raise RuntimeError(
                f"DeepSeek WNA16 runtime view missing indexed shard: {shard_name}"
            )


def clear_runtime_model_view(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for destination in output.iterdir():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        elif destination.is_dir():
            shutil.rmtree(destination)
        else:
            raise RuntimeError(
                f"DeepSeek WNA16 runtime view cannot clear output entry: {destination}"
            )


def materialize_runtime_model_view(artifact: Path, output: Path) -> None:
    if artifact.resolve() == output.resolve():
        raise RuntimeError("DeepSeek WNA16 runtime view output aliases artifact")
    config_path = artifact / "config.json"
    index_path = artifact / "model.safetensors.index.json"
    require_sha256(config_path, EXPECTED_ARTIFACT_CONFIG_SHA256, "config")
    require_sha256(index_path, EXPECTED_ARTIFACT_INDEX_SHA256, "tensor index")
    require_indexed_model_shards(artifact, index_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    provenance = config.get("club_3090_lowbit")
    if not isinstance(quantization, dict) or not isinstance(provenance, dict):
        raise RuntimeError(
            "DeepSeek WNA16 artifact lacks required quantization metadata"
        )
    if quantization.get("quant_method") != "compressed-tensors":
        raise RuntimeError(
            "DeepSeek WNA16 artifact quant_method is not compressed-tensors"
        )
    if list((quantization.get("config_groups") or {}).keys()) != ["group_w2"]:
        raise RuntimeError("DeepSeek WNA16 artifact must contain only group_w2")

    quantization["base_quant_method"] = "deepseek_v4_fp8"
    provenance["source_quantization_method"] = "compressed-tensors"
    rendered = json.dumps(config, indent=2, sort_keys=True) + "\n"
    rendered_sha256 = hashlib.sha256(rendered.encode()).hexdigest()
    if rendered_sha256 != EXPECTED_RUNTIME_CONFIG_SHA256:
        raise RuntimeError(
            "DeepSeek WNA16 runtime config transformation drift: "
            f"got {rendered_sha256}, expected {EXPECTED_RUNTIME_CONFIG_SHA256}"
        )

    clear_runtime_model_view(output)
    temporary_config = output / ".config.json.tmp"
    temporary_config.write_text(rendered, encoding="utf-8")
    os.replace(temporary_config, output / "config.json")

    for source in artifact.iterdir():
        if source.name == "config.json":
            continue
        destination = output / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: materialize-runtime-model-view.py ARTIFACT_DIRECTORY "
            "OUTPUT_DIRECTORY"
        )
    materialize_runtime_model_view(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
