#!/usr/bin/env python3
"""Materialize the immutable projection-sensitive WNA16 quality candidate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

EXPECTED_ARTIFACT_CONFIG_SHA256 = (
    "f29d9d4dfc7d279568922bfa0154959c01640bc97c17aaeaca42a3d812cc1874"
)
EXPECTED_ARTIFACT_INDEX_SHA256 = (
    "a390a5fc8e7884492dcd2d7ee0ea2155ebccdb59f94be84612351649963e4a45"
)
EXPECTED_CONFIG_GROUPS = (
    "group_w2_g128",
    "group_w2_g256",
    "group_w2_g512",
    "group_w4_g128",
)
EXPECTED_INTEGRATION_REVISION = "dd2d1fd6779addccc73094f77fa4ada7d9106a41"
EXPECTED_REQUIRED_TREE = "f73b30cc5a2ed9de200ca2e4de3cdef1a06f6538"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 content identity of one artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, description: str) -> None:
    """Reject an artifact file whose content identity differs from the candidate."""
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"DeepSeek WNA16 quality runtime {description} checksum mismatch: "
            f"got {actual}, expected {expected}"
        )


def require_indexed_model_shards(artifact: Path, index_path: Path) -> None:
    """Require every safe, single-file shard named by the tensor index."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("DeepSeek WNA16 quality tensor index has no weight_map")
    for shard_name in set(weight_map.values()):
        if not isinstance(shard_name, str):
            raise TypeError(
                "DeepSeek WNA16 quality tensor index has a non-string shard"
            )
        shard_path = Path(shard_name)
        if shard_path.is_absolute() or len(shard_path.parts) != 1:
            raise RuntimeError(
                "DeepSeek WNA16 quality tensor index has an unsafe shard path: "
                f"{shard_name!r}"
            )
        if not (artifact / shard_path).is_file():
            raise RuntimeError(
                f"DeepSeek WNA16 quality runtime missing indexed shard: {shard_name}"
            )


def require_quality_candidate_config(config: dict[str, object]) -> None:
    """Require the exact runtime-ready mixed-group quantization contract."""
    quantization = config.get("quantization_config")
    provenance = config.get("club_3090_lowbit")
    if not isinstance(quantization, dict) or not isinstance(provenance, dict):
        raise TypeError("DeepSeek WNA16 quality artifact lacks quantization provenance")
    if quantization.get("quant_method") != "compressed-tensors":
        raise RuntimeError(
            "DeepSeek WNA16 quality quant_method is not compressed-tensors"
        )
    if quantization.get("base_quant_method") != "deepseek_v4_fp8":
        raise RuntimeError(
            "DeepSeek WNA16 quality base_quant_method is not deepseek_v4_fp8"
        )
    if provenance.get("source_quantization_method") != "compressed-tensors":
        raise RuntimeError(
            "DeepSeek WNA16 quality source quantization method is not runtime-ready"
        )
    config_groups = quantization.get("config_groups")
    if (
        not isinstance(config_groups, dict)
        or tuple(config_groups) != EXPECTED_CONFIG_GROUPS
    ):
        raise RuntimeError(
            "DeepSeek WNA16 quality config groups differ from the pinned recipe"
        )
    compatibility = provenance.get("runtime_compatibility")
    if not isinstance(compatibility, dict):
        raise TypeError(
            "DeepSeek WNA16 quality artifact lacks runtime compatibility metadata"
        )
    if compatibility.get("integration_revision") != EXPECTED_INTEGRATION_REVISION:
        raise RuntimeError(
            "DeepSeek WNA16 quality integration revision differs from "
            "the pinned runtime"
        )
    if compatibility.get("required_tree") != EXPECTED_REQUIRED_TREE:
        raise RuntimeError(
            "DeepSeek WNA16 quality required tree differs from the pinned runtime"
        )


def clear_runtime_model_view(output: Path) -> None:
    """Clear only the dedicated tmpfs runtime model view."""
    output.mkdir(parents=True, exist_ok=True)
    for destination in output.iterdir():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        elif destination.is_dir():
            shutil.rmtree(destination)
        else:
            raise RuntimeError(
                f"DeepSeek WNA16 quality runtime cannot clear output: {destination}"
            )


def materialize_quality_candidate_runtime_view(
    artifact: Path,
    output: Path,
) -> None:
    """Validate the immutable candidate and expose it through runtime symlinks."""
    if artifact.resolve() == output.resolve():
        raise RuntimeError("DeepSeek WNA16 quality runtime output aliases artifact")
    config_path = artifact / "config.json"
    index_path = artifact / "model.safetensors.index.json"
    require_sha256(config_path, EXPECTED_ARTIFACT_CONFIG_SHA256, "config")
    require_sha256(index_path, EXPECTED_ARTIFACT_INDEX_SHA256, "tensor index")
    require_indexed_model_shards(artifact, index_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("DeepSeek WNA16 quality config is not an object")
    require_quality_candidate_config(config)

    clear_runtime_model_view(output)
    for source in artifact.iterdir():
        destination = output / source.name
        temporary = output / f".{source.name}.tmp"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(source)
        os.replace(temporary, destination)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: materialize-quality-candidate-runtime-view.py "
            "ARTIFACT_DIRECTORY OUTPUT_DIRECTORY"
        )
    materialize_quality_candidate_runtime_view(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
