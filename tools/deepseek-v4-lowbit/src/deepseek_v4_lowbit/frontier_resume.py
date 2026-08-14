from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.artifact_plan import ArtifactRecipe, LayerQuantization
from deepseek_v4_lowbit.convert_cli import (
    build_expected_output_weight_map,
    load_source_weight_map,
    select_output_shards,
)
from deepseek_v4_lowbit.frontier_convert import (
    artifact_recipe_from_payload,
    load_baseline_artifact_reuse,
    validate_frontier_candidate_names,
)
from deepseek_v4_lowbit.frontier_provenance import (
    validate_frontier_conversion_inputs,
    validate_frontier_recipe_bundle_shape,
)
from deepseek_v4_lowbit.shard_writer import (
    ResumableSafetensorsWriter,
    ShardIdentity,
    canonical_json_sha256,
    file_sha256,
)
from deepseek_v4_lowbit.source_transform import (
    QuantizerKind,
    TransformOptions,
    transform_recipe_sha256,
)

_RESUME_RECEIPT_SCHEMA_VERSION = 1
_RECOVERED_CAMPAIGN = "deepseek-v4-quant-frontier"
_RECOVERED_OUTCOME = "screening-complete-conversion-not-started"
_RECOVERED_REPORT_PATHS = frozenset(
    {
        "frontier-recipe-bundle.json",
        "frontier-screen/frontier-boundary-report.json",
        "frontier-screen/frontier-full-screen-progress.json",
        "frontier-screen/frontier-full-screen.json",
        "frontier-screen/frontier-pilot-screen.json",
        "frontier-screen/frontier-screen-iterations.json",
        "run-verda-quant-frontier.log",
        "source-headers-report.json",
        "source-headers.json",
    }
)
_EVIDENCE_HASH_FIELDS = {
    "pilot_screen_report": "pilot_screen_report_sha256",
    "boundary_report": "boundary_report_sha256",
    "full_screen_report": "screen_report_sha256",
    "source_headers_report": "source_headers_report_sha256",
    "source_headers": "source_headers_sha256",
}


@dataclass(frozen=True)
class FrontierResumeCheckpointRequest:
    """Paths and identities required to validate one recovered checkpoint."""

    source_directory: Path
    baseline_directory: Path
    imatrix_path: Path
    recipe_bundle_path: Path
    rebuilt_recipe_bundle_path: Path
    recovery_manifest_path: Path
    expected_recovery_manifest_sha256: str
    recovery_reports_directory: Path
    output_root: Path
    receipt_path: Path
    volume_id: str
    candidate: str
    evidence_paths: Mapping[str, Path]


def validate_frontier_resume_checkpoint(
    request: FrontierResumeCheckpointRequest,
) -> dict[str, Any]:
    """Validate a recovered conversion checkpoint and atomically bind its volume."""
    if not request.volume_id:
        raise ValueError("frontier resume volume id must be non-empty")
    validate_frontier_candidate_names((request.candidate,))
    recovery_manifest = _validate_recovery_manifest(
        request.recovery_manifest_path,
        expected_manifest_sha256=request.expected_recovery_manifest_sha256,
        reports_directory=request.recovery_reports_directory,
        volume_id=request.volume_id,
    )
    recipe_bundle = _validate_recovered_recipe_bundle(request, recovery_manifest)
    reusable_shard_names, completed_candidate_shards = (
        _validate_resume_conversion_checkpoint(request, recipe_bundle)
    )
    receipt = _build_frontier_resume_receipt(
        request,
        recipe_bundle,
        recovery_manifest,
        reusable_shard_names,
        completed_candidate_shards,
    )
    _write_json_atomic(request.receipt_path, receipt)
    return receipt


def _validate_recovered_recipe_bundle(
    request: FrontierResumeCheckpointRequest,
    recovery_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    recipe_bundle = _load_json_object(request.recipe_bundle_path)
    rebuilt_recipe_bundle = _load_json_object(request.rebuilt_recipe_bundle_path)
    validate_frontier_recipe_bundle_shape(recipe_bundle)
    validate_frontier_recipe_bundle_shape(rebuilt_recipe_bundle)
    recovered_recipe_sha256 = next(
        entry["sha256"]
        for entry in recovery_manifest["files"]
        if entry["path"] == "frontier-recipe-bundle.json"
    )
    if file_sha256(request.recipe_bundle_path) != recovered_recipe_sha256:
        raise ValueError(
            "frontier recovered recipe bundle differs from recovery manifest"
        )
    if rebuilt_recipe_bundle != recipe_bundle or file_sha256(
        request.rebuilt_recipe_bundle_path
    ) != file_sha256(request.recipe_bundle_path):
        raise ValueError(
            "frontier rebuilt recipe bundle is not byte-identical to recovered bundle"
        )
    _validate_frontier_evidence_hashes(recipe_bundle, request.evidence_paths)
    return recipe_bundle


def _validate_resume_conversion_checkpoint(
    request: FrontierResumeCheckpointRequest,
    recipe_bundle: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    validate_frontier_conversion_inputs(
        request.source_directory,
        request.baseline_directory,
        request.imatrix_path,
        recipe_bundle,
    )
    source_weight_map = load_source_weight_map(
        request.source_directory / "model.safetensors.index.json"
    )
    expected_weight_map = build_expected_output_weight_map(source_weight_map)
    baseline_recipe_sha256 = transform_recipe_sha256(
        ArtifactRecipe(default=LayerQuantization(2, 2)),
        _weighted_cuda_options(request.imatrix_path),
    )
    baseline = load_baseline_artifact_reuse(
        request.baseline_directory,
        expected_weight_map,
        baseline_recipe_sha256,
    )
    reusable_shard_names = _require_reusable_baseline_shard_names(recipe_bundle)
    for shard_name in reusable_shard_names:
        receipt = baseline.shard_receipt(
            shard_name,
            request.source_directory / shard_name,
        )
        if receipt is None:
            raise ValueError(
                f"frontier reusable baseline shard is missing: {shard_name}"
            )
    completed_candidate_shards = _validate_partial_candidate_shards(
        request.source_directory,
        request.output_root,
        recipe_bundle,
        candidate=request.candidate,
        imatrix_path=request.imatrix_path,
        expected_weight_map=expected_weight_map,
    )
    return reusable_shard_names, completed_candidate_shards


def _require_reusable_baseline_shard_names(
    recipe_bundle: Mapping[str, Any],
) -> list[str]:
    storage_summary = recipe_bundle.get("storage_summary")
    if not isinstance(storage_summary, Mapping):
        raise ValueError("frontier recipe bundle has no storage summary")
    reusable_shard_names = storage_summary.get("baseline_reused_shard_names")
    if not isinstance(reusable_shard_names, list) or any(
        not isinstance(name, str) for name in reusable_shard_names
    ):
        raise ValueError("frontier baseline reusable shard list is invalid")
    return reusable_shard_names


def _build_frontier_resume_receipt(
    request: FrontierResumeCheckpointRequest,
    recipe_bundle: Mapping[str, Any],
    recovery_manifest: Mapping[str, Any],
    reusable_shard_names: list[str],
    completed_candidate_shards: list[str],
) -> dict[str, Any]:
    evidence_hashes = {
        role: file_sha256(path) for role, path in sorted(request.evidence_paths.items())
    }
    validation_identity = {
        "volume_id": request.volume_id,
        "candidate": request.candidate,
        "recovery_manifest_sha256": file_sha256(request.recovery_manifest_path),
        "recovered_report_sha256": {
            entry["path"]: entry["sha256"] for entry in recovery_manifest["files"]
        },
        "recipe_bundle_sha256": file_sha256(request.recipe_bundle_path),
        "source_index_sha256": recipe_bundle["source_index_sha256"],
        "source_shards_sha256": recipe_bundle["source_shards_sha256"],
        "source_assets_sha256": recipe_bundle["source_assets_sha256"],
        "baseline_metrics_sha256": recipe_bundle["baseline_metrics_sha256"],
        "imatrix_sha256": recipe_bundle["imatrix_sha256"],
        "evidence_sha256": evidence_hashes,
        "reusable_baseline_shard_names": reusable_shard_names,
        "completed_candidate_shard_names": completed_candidate_shards,
    }
    return {
        "schema_version": _RESUME_RECEIPT_SCHEMA_VERSION,
        "validation_identity": validation_identity,
        "validation_identity_sha256": canonical_json_sha256(validation_identity),
        "reusable_baseline_shards": len(reusable_shard_names),
        "completed_candidate_shards": len(completed_candidate_shards),
    }


def require_frontier_resume_validation_receipt(
    receipt_path: Path,
    *,
    volume_id: str,
    candidate: str,
    expected_recovery_manifest_sha256: str,
) -> dict[str, Any]:
    """Reject a stale or edited CPU resume-validation receipt."""
    receipt = _load_json_object(receipt_path)
    identity = receipt.get("validation_identity")
    if (
        receipt.get("schema_version") != _RESUME_RECEIPT_SCHEMA_VERSION
        or not isinstance(identity, Mapping)
        or receipt.get("validation_identity_sha256") != canonical_json_sha256(identity)
        or identity.get("volume_id") != volume_id
        or identity.get("candidate") != candidate
        or identity.get("recovery_manifest_sha256") != expected_recovery_manifest_sha256
    ):
        raise ValueError("frontier resume validation receipt identity mismatch")
    reusable_names = identity.get("reusable_baseline_shard_names")
    completed_names = identity.get("completed_candidate_shard_names")
    if (
        not isinstance(reusable_names, list)
        or not isinstance(completed_names, list)
        or receipt.get("reusable_baseline_shards") != len(reusable_names)
        or receipt.get("completed_candidate_shards") != len(completed_names)
    ):
        raise ValueError("frontier resume validation receipt counts are inconsistent")
    return receipt


def _validate_recovery_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    reports_directory: Path,
    volume_id: str,
) -> dict[str, Any]:
    if (
        len(expected_manifest_sha256) != 64
        or file_sha256(manifest_path) != expected_manifest_sha256
    ):
        raise ValueError("frontier recovery manifest checksum mismatch")
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("frontier recovery manifest schema is unsupported")
    if manifest.get("campaign") != _RECOVERED_CAMPAIGN:
        raise ValueError("frontier recovery manifest campaign mismatch")
    if manifest.get("outcome") != _RECOVERED_OUTCOME:
        raise ValueError("frontier recovery manifest outcome mismatch")
    if manifest.get("recovered_volume_id") != volume_id:
        raise ValueError("frontier recovery manifest volume mismatch")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("frontier recovery manifest file inventory is invalid")
    entries: dict[str, tuple[int, str]] = {}
    for raw_entry in raw_files:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("frontier recovery manifest file entry is invalid")
        relative_path = raw_entry.get("path")
        size = raw_entry.get("size")
        sha256 = raw_entry.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or relative_path in entries
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise ValueError("frontier recovery manifest file entry is invalid")
        entries[relative_path] = (size, sha256)
    if set(entries) != _RECOVERED_REPORT_PATHS:
        raise ValueError("frontier recovery manifest file set is incomplete")
    if reports_directory.is_symlink() or not reports_directory.is_dir():
        raise ValueError("frontier recovered reports directory is missing or unsafe")
    for relative_path, (expected_size, expected_sha256) in entries.items():
        report_path = reports_directory / relative_path
        if report_path.is_symlink() or not report_path.is_file():
            raise ValueError(
                f"frontier recovered report is missing or unsafe: {relative_path}"
            )
        if (
            report_path.stat().st_size != expected_size
            or file_sha256(report_path) != expected_sha256
        ):
            raise ValueError(
                f"frontier recovered report checksum mismatch: {relative_path}"
            )
    return manifest


def _validate_frontier_evidence_hashes(
    recipe_bundle: Mapping[str, Any],
    evidence_paths: Mapping[str, Path],
) -> None:
    if set(evidence_paths) != set(_EVIDENCE_HASH_FIELDS):
        raise ValueError("frontier resume evidence roles are incomplete")
    for role, bundle_field in _EVIDENCE_HASH_FIELDS.items():
        path = evidence_paths[role]
        if not path.is_file() or file_sha256(path) != recipe_bundle[bundle_field]:
            raise ValueError(f"frontier resume evidence checksum mismatch: {role}")


def _validate_partial_candidate_shards(
    source_directory: Path,
    output_root: Path,
    recipe_bundle: Mapping[str, Any],
    *,
    candidate: str,
    imatrix_path: Path,
    expected_weight_map: Mapping[str, str],
) -> list[str]:
    candidates = recipe_bundle.get("candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("frontier recipe bundle has no candidates")
    recipe = artifact_recipe_from_payload(candidates[candidate])
    recipe_sha256 = transform_recipe_sha256(
        recipe,
        _weighted_cuda_options(imatrix_path),
    )
    source_shards = tuple(
        sorted(
            set(
                load_source_weight_map(
                    source_directory / "model.safetensors.index.json"
                ).values()
            )
        )
    )
    output_shards = select_output_shards(source_shards, expected_weight_map)
    output_directory = output_root / candidate
    if not output_directory.exists():
        return []
    writer = ResumableSafetensorsWriter(output_directory)
    completed: list[str] = []
    expected_shard_set = set(output_shards)
    unexpected_outputs = sorted(
        path.name
        for path in output_directory.glob("*.safetensors")
        if path.name not in expected_shard_set
    )
    if unexpected_outputs:
        raise ValueError(
            f"frontier resume has unexpected candidate shards: {unexpected_outputs[:3]}"
        )
    for shard_name in output_shards:
        output_path = output_directory / shard_name
        receipt_path = writer.receipt_directory / f"{shard_name}.json"
        partial_receipt_path = writer.receipt_directory / f"{shard_name}.partial.json"
        has_state = (
            output_path.exists()
            or receipt_path.exists()
            or partial_receipt_path.exists()
        )
        if not has_state:
            continue
        receipt = writer.completed_shard(
            shard_name,
            ShardIdentity(
                source_sha256=file_sha256(source_directory / shard_name),
                recipe_sha256=recipe_sha256,
            ),
        )
        if receipt is None:
            if output_path.exists():
                raise ValueError(
                    f"frontier candidate shard has no verified receipt: {shard_name}"
                )
            continue
        completed.append(shard_name)
    return completed


def _weighted_cuda_options(imatrix_path: Path) -> TransformOptions:
    return TransformOptions(
        group_size=128,
        quantizer=QuantizerKind.IMATRIX_WEIGHTED,
        device="cuda",
        imatrix_sha256=file_sha256(imatrix_path),
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"frontier resume JSON must be an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
