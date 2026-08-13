from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.convert_cli import load_source_weight_map
from deepseek_v4_lowbit.safetensors_header import safetensors_inventory
from deepseek_v4_lowbit.shard_writer import file_sha256


def capture_source_tensor_headers(source_directory: Path) -> dict[str, Any]:
    """Capture validated raw safetensors headers for every indexed source shard."""
    index_path = source_directory / "model.safetensors.index.json"
    weight_map = load_source_weight_map(index_path)
    expected_names_by_shard: dict[str, set[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        expected_names_by_shard.setdefault(shard_name, set()).add(tensor_name)

    captured: dict[str, Any] = {}
    for shard_name in sorted(expected_names_by_shard):
        shard_path = source_directory / shard_name
        raw_header = _read_raw_safetensors_header(shard_path)
        raw_header.pop("__metadata__", None)
        inventory = safetensors_inventory(shard_path)
        if set(raw_header) != expected_names_by_shard[shard_name]:
            raise ValueError(f"source header differs from tensor index: {shard_name}")
        if set(inventory) != set(raw_header):
            raise ValueError(f"source header inventory mismatch: {shard_name}")
        for tensor_name, raw_record in raw_header.items():
            _validate_raw_header_record(
                tensor_name,
                raw_record,
                inventory[tensor_name],
            )
        captured[shard_name] = raw_header
    return captured


def source_tensor_headers_report(
    source_directory: Path,
    captured_headers: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind captured source headers, shards, and copied model assets."""
    index_path = source_directory / "model.safetensors.index.json"
    source_assets = {
        path.name: file_sha256(path)
        for path in source_directory.iterdir()
        if path.is_file()
        and path.name != index_path.name
        and not path.name.endswith(".safetensors")
    }
    if "config.json" not in source_assets:
        raise ValueError("source tensor-header report requires config.json")
    return {
        "schema_version": 1,
        "source_index_sha256": file_sha256(index_path),
        "source_shards": {
            shard_name: file_sha256(source_directory / shard_name)
            for shard_name in captured_headers
        },
        "source_assets": dict(sorted(source_assets.items())),
        "headers": dict(captured_headers),
    }


def write_source_tensor_headers_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist a source tensor-header report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    os.replace(temporary, path)


def extract_captured_headers(report_path: Path, output_path: Path) -> None:
    """Extract the planner-compatible shard map from a bound header report."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    headers = payload.get("headers")
    if not isinstance(headers, dict) or not headers:
        raise ValueError("source tensor-header report contains no headers")
    output_path.write_text(
        json.dumps(headers, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_raw_safetensors_header(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    with path.open("rb") as file_handle:
        encoded_size = file_handle.read(8)
        if len(encoded_size) != 8:
            raise ValueError(f"invalid source safetensors header: {path}")
        header_size = int.from_bytes(encoded_size, byteorder="little")
        if header_size <= 0 or header_size > file_size - 8:
            raise ValueError(f"invalid source safetensors header size: {path}")
        try:
            header = json.loads(file_handle.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid source safetensors header JSON: {path}"
            ) from error
    if not isinstance(header, dict):
        raise ValueError(f"invalid source safetensors tensor map: {path}")
    return header


def _validate_raw_header_record(
    tensor_name: str,
    raw_record: Any,
    inventory_record: Mapping[str, Any],
) -> None:
    if not isinstance(raw_record, dict):
        raise ValueError(f"invalid source tensor header: {tensor_name}")
    dtype = raw_record.get("dtype")
    shape = raw_record.get("shape")
    offsets = raw_record.get("data_offsets")
    if (
        dtype != inventory_record["dtype"]
        or shape != inventory_record["shape"]
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or offsets[1] - offsets[0] != inventory_record["nbytes"]
    ):
        raise ValueError(
            f"source tensor header disagrees with inventory: {tensor_name}"
        )
