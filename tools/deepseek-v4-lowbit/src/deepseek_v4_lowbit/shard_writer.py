from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.safetensors_header import (
    SafetensorsHeaderError,
    safetensors_inventory,
)

_RECEIPT_VERSION = 2
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class ResumeConflictError(RuntimeError):
    """Existing conversion state cannot be safely resumed."""


@dataclass(frozen=True)
class ShardIdentity:
    source_sha256: str
    recipe_sha256: str


@dataclass(frozen=True)
class ShardReceipt:
    shard_name: str
    identity: ShardIdentity
    output_path: Path
    output_sha256: str
    output_bytes: int
    tensors: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


class ResumableSafetensorsWriter:
    def __init__(self, output_directory: Path | str) -> None:
        self.output_directory = Path(output_directory)
        self.state_directory = self.output_directory / ".conversion-state"
        self.receipt_directory = self.state_directory / "receipts"
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.receipt_directory.mkdir(parents=True, exist_ok=True)

    def completed_shard(
        self,
        shard_name: str,
        identity: ShardIdentity,
    ) -> ShardReceipt | None:
        shard_name = _validate_shard_name(shard_name)
        receipt_path = self._receipt_path(shard_name)
        partial_receipt_path = self._partial_receipt_path(shard_name)

        if not receipt_path.exists():
            if partial_receipt_path.exists() and self._output_path(shard_name).exists():
                candidate = self._read_receipt(partial_receipt_path)
                self._require_receipt_shard(candidate, shard_name)
                self._verify_receipt(candidate, identity)
                os.replace(partial_receipt_path, receipt_path)
                _sync_directory(self.receipt_directory)
            else:
                return None

        receipt = self._read_receipt(receipt_path)
        self._require_receipt_shard(receipt, shard_name)
        self._verify_receipt(receipt, identity)
        return receipt

    def write_shard(
        self,
        shard_name: str,
        tensors: Mapping[str, Any],
        identity: ShardIdentity,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ShardReceipt:
        shard_name = _validate_shard_name(shard_name)
        completed = self.completed_shard(shard_name, identity)
        if completed is not None:
            return completed
        if not tensors:
            raise ValueError("cannot write an empty safetensors shard")

        output_path = self._output_path(shard_name)
        temporary_output_path = self._temporary_output_path(shard_name)
        partial_receipt_path = self._partial_receipt_path(shard_name)
        if output_path.exists():
            raise ResumeConflictError(
                f"output shard {shard_name} exists without a verifiable receipt"
            )
        temporary_output_path.unlink(missing_ok=True)
        partial_receipt_path.unlink(missing_ok=True)

        safe_tensors = _prepare_tensors(tensors)
        receipt_metadata = _validate_receipt_metadata(metadata or {})
        save_file = _import_optional("safetensors.torch").save_file
        save_file(safe_tensors, str(temporary_output_path))
        _sync_file(temporary_output_path)

        receipt = ShardReceipt(
            shard_name=shard_name,
            identity=identity,
            output_path=output_path,
            output_sha256=file_sha256(temporary_output_path),
            output_bytes=temporary_output_path.stat().st_size,
            tensors=safetensors_inventory(temporary_output_path),
            metadata=receipt_metadata,
        )
        _write_json_atomic_target(partial_receipt_path, _receipt_payload(receipt))
        os.replace(temporary_output_path, output_path)
        _sync_directory(self.output_directory)
        os.replace(partial_receipt_path, self._receipt_path(shard_name))
        _sync_directory(self.receipt_directory)
        return receipt

    def finalize_index(
        self,
        expected_shards: Iterable[str],
        *,
        expected_weight_map: Mapping[str, str],
    ) -> Path:
        """Publish an index only when produced tensors exactly match expectations."""
        shard_names = [_validate_shard_name(name) for name in expected_shards]
        if not shard_names:
            raise ValueError("at least one expected shard is required")
        if len(set(shard_names)) != len(shard_names):
            raise ValueError("expected shard names must be unique")
        validated_expected_weight_map = _validate_expected_weight_map(
            expected_weight_map,
            set(shard_names),
        )

        weight_map: dict[str, str] = {}
        total_size = 0
        recipe_fingerprints: set[str] = set()
        for shard_name in shard_names:
            receipt_path = self._receipt_path(shard_name)
            if not receipt_path.exists():
                raise ResumeConflictError(
                    f"missing receipt for expected shard {shard_name}"
                )
            receipt = self._read_receipt(receipt_path)
            self._require_receipt_shard(receipt, shard_name)
            self._verify_output(receipt)
            recipe_fingerprints.add(receipt.identity.recipe_sha256)
            for tensor_name, tensor_record in receipt.tensors.items():
                if tensor_name in weight_map:
                    raise ResumeConflictError(
                        f"duplicate tensor {tensor_name!r} across output shards"
                    )
                weight_map[tensor_name] = shard_name
                total_size += int(tensor_record["nbytes"])

        if len(recipe_fingerprints) != 1:
            raise ResumeConflictError(
                "expected shards were produced with different recipe fingerprints"
            )
        if weight_map != validated_expected_weight_map:
            missing = sorted(set(validated_expected_weight_map) - set(weight_map))
            unexpected = sorted(set(weight_map) - set(validated_expected_weight_map))
            wrong_shard = sorted(
                tensor_name
                for tensor_name in set(weight_map) & set(validated_expected_weight_map)
                if weight_map[tensor_name] != validated_expected_weight_map[tensor_name]
            )
            raise ResumeConflictError(
                "final tensor inventory does not match expected output: "
                f"missing={missing[:3]}, unexpected={unexpected[:3]}, "
                f"wrong_shard={wrong_shard[:3]}"
            )

        index_path = self.output_directory / "model.safetensors.index.json"
        _write_json_atomic_target(
            index_path,
            {
                "metadata": {"total_size": total_size},
                "weight_map": dict(sorted(weight_map.items())),
            },
        )
        return index_path

    @staticmethod
    def _require_receipt_shard(receipt: ShardReceipt, expected_name: str) -> None:
        if receipt.shard_name != expected_name:
            raise ResumeConflictError(
                f"receipt for {expected_name} names shard {receipt.shard_name}"
            )

    def _verify_receipt(
        self,
        receipt: ShardReceipt,
        expected_identity: ShardIdentity,
    ) -> None:
        if receipt.identity.source_sha256 != expected_identity.source_sha256:
            raise ResumeConflictError(
                f"source fingerprint changed for shard {receipt.shard_name}"
            )
        if receipt.identity.recipe_sha256 != expected_identity.recipe_sha256:
            raise ResumeConflictError(
                f"recipe fingerprint changed for shard {receipt.shard_name}"
            )
        self._verify_output(receipt)

    def _verify_output(self, receipt: ShardReceipt) -> None:
        expected_path = self._output_path(receipt.shard_name)
        if receipt.output_path != expected_path or not expected_path.is_file():
            raise ResumeConflictError(
                f"output shard {receipt.shard_name} is missing or moved"
            )
        if expected_path.stat().st_size != receipt.output_bytes:
            raise ResumeConflictError(
                f"checksum verification failed for shard {receipt.shard_name}: "
                "size mismatch"
            )
        if file_sha256(expected_path) != receipt.output_sha256:
            raise ResumeConflictError(
                f"checksum verification failed for shard {receipt.shard_name}"
            )
        try:
            inventory = safetensors_inventory(expected_path)
        except SafetensorsHeaderError as error:
            raise ResumeConflictError(
                f"invalid safetensors output for shard {receipt.shard_name}"
            ) from error
        if inventory != receipt.tensors:
            raise ResumeConflictError(
                f"tensor inventory mismatch for shard {receipt.shard_name}"
            )

    def _read_receipt(self, path: Path) -> ShardReceipt:
        try:
            payload = json.loads(path.read_text())
            if payload["version"] != _RECEIPT_VERSION:
                raise ValueError("unsupported receipt version")
            shard_name = _validate_shard_name(payload["shard_name"])
            source_sha256 = _require_nonempty_string(
                payload["source_sha256"], "source_sha256"
            )
            recipe_sha256 = _require_nonempty_string(
                payload["recipe_sha256"], "recipe_sha256"
            )
            output_sha256 = _require_sha256(payload["output_sha256"])
            output_bytes = int(payload["output_bytes"])
            if output_bytes < 0:
                raise ValueError("negative output_bytes")
            tensors = _validate_tensor_records(payload["tensors"])
            metadata = _validate_receipt_metadata(payload.get("metadata", {}))
            return ShardReceipt(
                shard_name=shard_name,
                identity=ShardIdentity(
                    source_sha256=source_sha256,
                    recipe_sha256=recipe_sha256,
                ),
                output_path=self._output_path(shard_name),
                output_sha256=output_sha256,
                output_bytes=output_bytes,
                tensors=tensors,
                metadata=metadata,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ResumeConflictError(f"invalid conversion receipt {path}") from error

    def _output_path(self, shard_name: str) -> Path:
        return self.output_directory / shard_name

    def _temporary_output_path(self, shard_name: str) -> Path:
        return self.state_directory / f"{shard_name}.partial"

    def _receipt_path(self, shard_name: str) -> Path:
        return self.receipt_directory / f"{shard_name}.json"

    def _partial_receipt_path(self, shard_name: str) -> Path:
        return self.receipt_directory / f"{shard_name}.partial.json"


def _validate_shard_name(shard_name: str) -> str:
    if not shard_name or Path(shard_name).name != shard_name:
        raise ValueError(f"shard name must be a basename, got {shard_name!r}")
    if not shard_name.endswith(".safetensors"):
        raise ValueError(f"shard name must end in .safetensors, got {shard_name!r}")
    return shard_name


def _prepare_tensors(tensors: Mapping[str, Any]) -> dict[str, Any]:
    torch = _import_optional("torch")
    prepared: dict[str, Any] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tensor names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"tensor {name!r} is not a torch tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"tensor {name!r} must be on CPU before shard writing")
        prepared[name] = tensor.contiguous()
    return prepared


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: Any) -> str:
    value = _require_nonempty_string(value, "output_sha256")
    valid_characters = "0123456789abcdef"
    if len(value) != 64 or any(
        character not in valid_characters for character in value
    ):
        raise ValueError("output_sha256 must be a lowercase SHA-256 digest")
    return value


def _validate_expected_weight_map(
    value: Mapping[str, str],
    expected_shards: set[str],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("expected weight map must be a mapping")
    validated: dict[str, str] = {}
    for tensor_name, shard_name in value.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("expected tensor names must be non-empty strings")
        if not isinstance(shard_name, str):
            raise ValueError("expected shard names must be strings")
        validated_shard_name = _validate_shard_name(shard_name)
        if validated_shard_name not in expected_shards:
            raise ValueError(
                f"expected tensor {tensor_name!r} references unknown shard "
                f"{validated_shard_name!r}"
            )
        validated[tensor_name] = validated_shard_name
    return validated


def _validate_tensor_records(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("receipt tensors must be an object")
    records: dict[str, dict[str, Any]] = {}
    for tensor_name, raw_record in value.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("receipt tensor names must be non-empty strings")
        if not isinstance(raw_record, dict):
            raise ValueError(f"invalid tensor record for {tensor_name}")
        dtype = _require_nonempty_string(raw_record.get("dtype"), "tensor dtype")
        shape = raw_record.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int) or dimension < 0 for dimension in shape
        ):
            raise ValueError(f"invalid tensor shape for {tensor_name}")
        nbytes = raw_record.get("nbytes")
        if not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError(f"invalid tensor byte count for {tensor_name}")
        records[tensor_name] = {"dtype": dtype, "shape": shape, "nbytes": nbytes}
    return records


def _validate_receipt_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("receipt metadata must be an object")
    try:
        serialized = json.dumps(value, allow_nan=False, sort_keys=True)
        normalized = json.loads(serialized)
    except (TypeError, ValueError) as error:
        raise ValueError("receipt metadata must contain finite JSON values") from error
    if not isinstance(normalized, dict):
        raise ValueError("receipt metadata must normalize to an object")
    return normalized


def _receipt_payload(receipt: ShardReceipt) -> dict[str, Any]:
    return {
        "version": _RECEIPT_VERSION,
        "shard_name": receipt.shard_name,
        "source_sha256": receipt.identity.source_sha256,
        "recipe_sha256": receipt.identity.recipe_sha256,
        "output_sha256": receipt.output_sha256,
        "output_bytes": receipt.output_bytes,
        "tensors": receipt.tensors,
        "metadata": receipt.metadata,
    }


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        while chunk := file_handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _write_json_atomic_target(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.writing")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _sync_file(temporary_path)
    os.replace(temporary_path, path)
    _sync_directory(path.parent)


def _sync_file(path: Path) -> None:
    with path.open("rb") as file_handle:
        os.fsync(file_handle.fileno())


def _sync_directory(path: Path) -> None:
    directory_descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"{module_name} is required for safetensors shard writing"
        ) from error
