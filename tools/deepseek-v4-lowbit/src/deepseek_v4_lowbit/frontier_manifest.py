from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.model_config import mixed_group_runtime_compatibility
from deepseek_v4_lowbit.shard_writer import file_sha256


def write_frontier_candidate_model_card(
    path: Path,
    *,
    candidate: str,
    summary: Mapping[str, Any],
    parent_revision: str,
    recipe_bundle_sha256: str,
) -> None:
    """Write a candidate-specific experimental Hugging Face model card."""
    w13_group128 = ", ".join(map(str, summary["w13_group128_layers"])) or "none"
    w13_group256 = ", ".join(map(str, summary["w13_group256_layers"])) or "none"
    w13_group512 = ", ".join(map(str, summary["w13_group512_layers"])) or "none"
    w2_group128 = ", ".join(map(str, summary["w2_group128_layers"])) or "none"
    w2_group256 = ", ".join(map(str, summary["w2_group256_layers"])) or "none"
    w2_group512 = ", ".join(map(str, summary["w2_group512_layers"])) or "none"
    w4_down = ", ".join(map(str, summary["w4_down_layers"])) or "none"
    runtime = mixed_group_runtime_compatibility()
    rendered = f"""---
license: mit
library_name: transformers
tags:
- deepseek-v4
- compressed-tensors
- humming
- experimental
---

# DeepSeek-V4-Flash-0731 WNA16 frontier: {candidate}

This is an **experimental** mixed-group WNA16 candidate generated from
`deepseek-ai/DeepSeek-V4-Flash-0731`. It is one point on a four-candidate
quantization frontier and is not a stock-vLLM checkpoint.

## Exact recipe

- Routed gate/up: W2.
- Routed down: W2 except the W4 layers listed below.
- Gate/up group-128 layers: {w13_group128}.
- Gate/up group-256 layers: {w13_group256}.
- Gate/up group-512 layers: {w13_group512}.
- Down group-128 layers: {w2_group128}.
- Down group-256 layers: {w2_group256}.
- Down group-512 layers: {w2_group512}.
- W4 down-projection layers: {w4_down}.
- MTP: omitted.
- Raw tensor payload: {int(summary["total_bytes"]):,} bytes
  ({float(summary["total_gib"]):.6f} GiB).
- Whole-model bits per base parameter:
  {float(summary["whole_model_bits_per_parameter"]):.6f}.

The layer allocation comes from a full-expert imatrix-weighted reconstruction
screen under the recorded byte budget. See `frontier-manifest.json` for exact
file hashes and `conversion-metrics.json` for per-tensor errors.

## Runtime

This artifact requires `Whamp/vllm` commit
`{runtime["integration_revision"]}` over `haosdent/vllm` commit
`{runtime["base_revision"]}`. The required Git tree is
`{runtime["required_tree"]}`. Stock vLLM compatibility is not claimed.

This candidate has not passed the single-worker DeepSWE gate. It also requires
the rollback-wrapped SM86 mixed-group numerical/cubin oracle before any server60
runtime test. Its acceptance status is
`{runtime["acceptance_status"]}`; do not promote it as a serving replacement.

## Provenance

- Parent artifact revision: `{parent_revision}`.
- Frontier recipe bundle SHA-256: `{recipe_bundle_sha256}`.
- Quantizer: imatrix-weighted symmetric RTN.
- Source routed weights: official MXFP4/E8M0 representation, dequantized before
  WNA16 fitting.
- Antirez routed-expert imatrix content SHA-256:
  `02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e`.

This candidate preserves the upstream MIT license. DeepSeek, AutoRound,
compressed-tensors, Humming, vLLM, and Antirez retain their respective credit.
"""
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(rendered, encoding="utf-8")
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    os.replace(temporary, path)


def build_frontier_candidate_manifest(
    candidate_directory: Path,
    *,
    candidate: str,
    parent_revision: str,
    recipe_bundle_sha256: str,
    recipe_sha256: str,
) -> dict[str, Any]:
    """Build a hash inventory that survives local candidate deletion."""
    index_path = candidate_directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("frontier candidate tensor index has no weight_map")
    shard_names = sorted(set(weight_map.values()))
    shards: list[dict[str, int | str]] = []
    for shard_name in shard_names:
        shard_path = candidate_directory / shard_name
        if not shard_path.is_file():
            raise ValueError(f"frontier candidate shard is missing: {shard_name}")
        shards.append(
            {
                "path": shard_name,
                "bytes": shard_path.stat().st_size,
                "sha256": file_sha256(shard_path),
            }
        )
    files = {}
    for path in candidate_directory.rglob("*"):
        relative = path.relative_to(candidate_directory)
        if path.name == "frontier-manifest.json" or any(
            part in {".conversion-state", ".cache"} for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"frontier candidate manifest found symlink: {relative}")
        if not path.is_file() or path.name.endswith(".safetensors"):
            continue
        files[relative.as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    for required_file in (
        "README.md",
        "config.json",
        "conversion-metrics.json",
        "frontier-recipe-bundle.json",
        "model.safetensors.index.json",
    ):
        if required_file not in files:
            raise ValueError(
                f"frontier candidate manifest is missing required file: {required_file}"
            )
    return {
        "schema_version": 1,
        "candidate": candidate,
        "parent_revision": parent_revision,
        "recipe_bundle_sha256": recipe_bundle_sha256,
        "recipe_sha256": recipe_sha256,
        "runtime_compatibility": mixed_group_runtime_compatibility(),
        "tensor_count": len(weight_map),
        "shard_count": len(shard_names),
        "model_payload_bytes": sum(int(shard["bytes"]) for shard in shards),
        "files": files,
        "shards": shards,
    }


def write_frontier_candidate_manifest(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Atomically persist one frontier candidate manifest."""
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    os.replace(temporary, path)
