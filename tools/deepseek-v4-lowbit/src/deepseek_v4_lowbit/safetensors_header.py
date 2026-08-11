from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SafetensorsHeaderError(ValueError):
    """A safetensors header or data range is structurally invalid."""


def safetensors_inventory(path: Path | str) -> dict[str, dict[str, Any]]:
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as file_handle:
        encoded_header_size = file_handle.read(8)
        if len(encoded_header_size) != 8:
            raise SafetensorsHeaderError(f"invalid safetensors header in {path}")
        header_size = int.from_bytes(encoded_header_size, byteorder="little")
        if header_size <= 0 or header_size > file_size - 8:
            raise SafetensorsHeaderError(f"invalid safetensors header size in {path}")
        try:
            header = json.loads(
                file_handle.read(header_size),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise SafetensorsHeaderError(
                f"invalid safetensors header in {path}"
            ) from error

    if not isinstance(header, dict):
        raise SafetensorsHeaderError(f"invalid safetensors tensor map in {path}")
    payload_bytes = file_size - 8 - header_size
    inventory: dict[str, dict[str, Any]] = {}
    occupied_ranges: list[tuple[int, int, str]] = []
    for tensor_name, raw_record in header.items():
        if tensor_name == "__metadata__":
            continue
        dtype, shape, start, end = _parse_tensor_record(
            tensor_name,
            raw_record,
            payload_bytes,
            path,
        )
        occupied_ranges.append((start, end, tensor_name))
        inventory[tensor_name] = {
            "dtype": dtype,
            "shape": shape,
            "nbytes": end - start,
        }

    previous_end = 0
    for start, end, tensor_name in sorted(occupied_ranges):
        if start != previous_end:
            raise SafetensorsHeaderError(
                f"non-contiguous tensor data before {tensor_name!r} in {path}"
            )
        previous_end = end
    if previous_end != payload_bytes:
        raise SafetensorsHeaderError(f"unaccounted tensor data in {path}")
    return dict(sorted(inventory.items()))


def _parse_tensor_record(
    tensor_name: str,
    raw_record: Any,
    payload_bytes: int,
    path: Path,
) -> tuple[str, list[int], int, int]:
    try:
        dtype = _require_nonempty_string(raw_record["dtype"], "tensor dtype")
        shape = raw_record["shape"]
        offsets = raw_record["data_offsets"]
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int) or dimension < 0 for dimension in shape
        ):
            raise ValueError("invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) for offset in offsets)
        ):
            raise ValueError("invalid offsets")
        start, end = offsets
        if start < 0 or end < start or end > payload_bytes:
            raise ValueError("out-of-range offsets")
    except (KeyError, TypeError, ValueError) as error:
        raise SafetensorsHeaderError(
            f"invalid safetensors record for {tensor_name!r} in {path}"
        ) from error
    return dtype, shape, start, end


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value
