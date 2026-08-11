from __future__ import annotations

import math
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from deepseek_v4_lowbit.artifact_plan import TensorDisposition, classify_tensor

_I32 = struct.Struct("<i")
_ENTRY_LIMIT = 1_000_000
_NAME_BYTE_LIMIT = 4096
_DATASET_BYTE_LIMIT = 1 << 20
_PROJECTION_TO_DS4 = {
    "w1": "ffn_gate_exps",
    "w3": "ffn_up_exps",
    "w2": "ffn_down_exps",
}


@dataclass(frozen=True)
class ImatrixEntry:
    name: str
    calls: int
    value_count: int
    values_offset: int


class ImatrixFile:
    """Memory-mapped index over a legacy llama.cpp imatrix file."""

    def __init__(
        self,
        source: BinaryIO,
        mapping: mmap.mmap,
        entries: dict[str, ImatrixEntry],
        *,
        chunks: int | None,
        dataset: str | None,
    ) -> None:
        self._source = source
        self._mapping = mapping
        self._entries = entries
        self.chunks = chunks
        self.dataset = dataset

    @classmethod
    def open(cls, path: Path) -> ImatrixFile:
        source = path.open("rb")
        try:
            mapping = mmap.mmap(source.fileno(), length=0, access=mmap.ACCESS_READ)
        except Exception:
            source.close()
            raise

        try:
            entries, chunks, dataset = _parse_index(mapping)
            return cls(
                source,
                mapping,
                entries,
                chunks=chunks,
                dataset=dataset,
            )
        except Exception:
            mapping.close()
            source.close()
            raise

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def validate_deepseek_v4_geometry(
        self,
        *,
        layer_count: int = 43,
        expert_count: int = 256,
        hidden_size: int = 4096,
        intermediate_size: int = 2048,
    ) -> None:
        expected: dict[str, int] = {}
        for layer in range(layer_count):
            expected[f"blk.{layer}.ffn_gate_exps.weight"] = expert_count * hidden_size
            expected[f"blk.{layer}.ffn_up_exps.weight"] = expert_count * hidden_size
            expected[f"blk.{layer}.ffn_down_exps.weight"] = (
                expert_count * intermediate_size
            )
        if set(self._entries) != set(expected):
            missing = sorted(set(expected) - set(self._entries))
            extra = sorted(set(self._entries) - set(expected))
            raise ValueError(
                f"DeepSeek V4 imatrix entry mismatch: missing={missing}, extra={extra}"
            )
        for name, value_count in expected.items():
            entry = self._entries[name]
            if entry.calls <= 0:
                raise ValueError(f"DeepSeek V4 imatrix entry has no calls: {name}")
            if entry.value_count != value_count:
                raise ValueError(
                    f"DeepSeek V4 imatrix geometry mismatch for {name}: "
                    f"got {entry.value_count}, expected {value_count}"
                )

    def expert_vector(
        self,
        hf_weight_name: str,
        *,
        expert_count: int,
        input_columns: int,
    ) -> tuple[float, ...]:
        if expert_count <= 0 or input_columns <= 0:
            raise ValueError("expert_count and input_columns must be positive")
        entry_name, expert = map_hf_expert_to_imatrix(hf_weight_name)
        if expert >= expert_count:
            raise ValueError(
                f"expert {expert} is outside configured expert count {expert_count}"
            )
        try:
            entry = self._entries[entry_name]
        except KeyError as error:
            raise KeyError(f"missing imatrix entry: {entry_name}") from error

        expected_values = expert_count * input_columns
        if entry.value_count != expected_values:
            raise ValueError(
                f"imatrix value count mismatch for {entry_name}: "
                f"got {entry.value_count}, expected {expected_values}"
            )
        vector_offset = entry.values_offset + expert * input_columns * 4
        values = struct.unpack_from(f"<{input_columns}f", self._mapping, vector_offset)
        if entry.calls > 0 and entry.calls != 1:
            inverse_calls = 1.0 / entry.calls
            values = tuple(value * inverse_calls for value in values)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"non-finite imatrix value in {entry_name} expert {expert}"
            )
        return values

    def close(self) -> None:
        self._mapping.close()
        self._source.close()

    def __enter__(self) -> ImatrixFile:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


def map_hf_expert_to_imatrix(hf_weight_name: str) -> tuple[str, int]:
    identity = classify_tensor(hf_weight_name)
    if (
        identity.disposition is not TensorDisposition.QUANTIZE
        or identity.layer is None
        or identity.expert is None
        or identity.projection is None
    ):
        raise ValueError(f"not a routed expert weight: {hf_weight_name}")
    ds4_projection = _PROJECTION_TO_DS4[identity.projection]
    return f"blk.{identity.layer}.{ds4_projection}.weight", identity.expert


def _parse_index(
    mapping: mmap.mmap,
) -> tuple[dict[str, ImatrixEntry], int | None, str | None]:
    cursor = 0
    entry_count, cursor = _read_i32(mapping, cursor, "entry count")
    if entry_count < 1 or entry_count > _ENTRY_LIMIT:
        raise ValueError(f"invalid imatrix entry count: {entry_count}")

    entries: dict[str, ImatrixEntry] = {}
    for _ in range(entry_count):
        name_length, cursor = _read_i32(mapping, cursor, "name length")
        if name_length < 1 or name_length > _NAME_BYTE_LIMIT:
            raise ValueError(f"invalid imatrix name length: {name_length}")
        name_bytes, cursor = _read_bytes(mapping, cursor, name_length, "entry name")
        try:
            name = name_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("invalid UTF-8 in imatrix entry name") from error
        if name in entries:
            raise ValueError(f"duplicate imatrix entry: {name}")

        calls, cursor = _read_i32(mapping, cursor, "call count")
        value_count, cursor = _read_i32(mapping, cursor, "value count")
        if calls < 0:
            raise ValueError(f"invalid imatrix call count for {name}: {calls}")
        if value_count < 1:
            raise ValueError(f"invalid imatrix value count for {name}: {value_count}")
        values_offset = cursor
        cursor = _advance(mapping, cursor, value_count * 4, "entry values")
        entries[name] = ImatrixEntry(name, calls, value_count, values_offset)

    if cursor == len(mapping):
        return entries, None, None

    chunks, cursor = _read_i32(mapping, cursor, "chunk count")
    dataset_length, cursor = _read_i32(mapping, cursor, "dataset length")
    if dataset_length < 0 or dataset_length > _DATASET_BYTE_LIMIT:
        raise ValueError(f"invalid imatrix dataset length: {dataset_length}")
    dataset_bytes, cursor = _read_bytes(mapping, cursor, dataset_length, "dataset name")
    if cursor != len(mapping):
        raise ValueError(f"unexpected {len(mapping) - cursor} trailing imatrix bytes")
    try:
        dataset = dataset_bytes.decode("utf-8") if dataset_bytes else None
    except UnicodeDecodeError as error:
        raise ValueError("invalid UTF-8 in imatrix dataset name") from error
    return entries, chunks, dataset


def _read_i32(mapping: mmap.mmap, cursor: int, field: str) -> tuple[int, int]:
    raw, next_cursor = _read_bytes(mapping, cursor, _I32.size, field)
    return _I32.unpack(raw)[0], next_cursor


def _read_bytes(
    mapping: mmap.mmap,
    cursor: int,
    length: int,
    field: str,
) -> tuple[bytes, int]:
    next_cursor = _advance(mapping, cursor, length, field)
    return mapping[cursor:next_cursor], next_cursor


def _advance(mapping: mmap.mmap, cursor: int, length: int, field: str) -> int:
    next_cursor = cursor + length
    if length < 0 or next_cursor > len(mapping):
        raise ValueError(f"truncated imatrix while reading {field}")
    return next_cursor
