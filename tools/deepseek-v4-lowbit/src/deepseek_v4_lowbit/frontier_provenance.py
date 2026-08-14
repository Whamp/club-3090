from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.convert_cli import load_source_weight_map
from deepseek_v4_lowbit.shard_writer import file_sha256

_FRONTIER_CANDIDATES = ("cliff", "capacity", "balanced", "quality")
_FRONTIER_GROUP_SIZES = (128, 256, 512)
_ROUTED_WEIGHT_NAME = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>w[123])\.weight$"
)


@dataclass(frozen=True)
class FrontierRecipeEvidence:
    """Verified screen-to-recipe evidence and merged screen results."""

    baseline_metrics_sha256: str
    pilot_screen_report_sha256: str
    boundary_report_sha256: str
    full_screen_report_sha256: str
    source_headers_report_sha256: str
    source_headers_sha256: str
    source_index_sha256: str
    source_shards_sha256: dict[str, str]
    source_assets_sha256: dict[str, str]
    imatrix_sha256: str
    boundary_layers: tuple[int, ...]
    merged_screen_results: tuple[dict[str, Any], ...]


def load_verified_frontier_recipe_evidence(
    *,
    baseline_metrics_path: Path,
    pilot_screen_path: Path,
    boundary_report_path: Path,
    full_screen_path: Path,
    source_headers_report_path: Path,
    planner_headers_path: Path,
) -> FrontierRecipeEvidence:
    """Verify every file in the frontier screen-to-recipe handoff."""
    baseline_sha256 = file_sha256(baseline_metrics_path)
    pilot_sha256 = file_sha256(pilot_screen_path)
    boundary_sha256 = file_sha256(boundary_report_path)
    full_sha256 = file_sha256(full_screen_path)
    source_report_sha256 = file_sha256(source_headers_report_path)
    planner_headers_sha256 = file_sha256(planner_headers_path)

    pilot = _load_json_object(pilot_screen_path)
    boundary = _load_json_object(boundary_report_path)
    full = _load_json_object(full_screen_path)
    source_report = _load_json_object(source_headers_report_path)
    planner_headers = _load_json_object(planner_headers_path)

    source_index_sha256 = _require_sha256(
        source_report.get("source_index_sha256"),
        "frontier source-header report source index",
    )
    imatrix_sha256 = _require_sha256(
        pilot.get("imatrix_sha256"),
        "frontier pilot imatrix",
    )
    source_shards = _require_shard_hashes(
        source_report.get("source_shards"),
        "frontier source-header report",
    )
    source_assets = _require_asset_hashes(
        source_report.get("source_assets"),
        "frontier source-header report",
    )
    if source_report.get("headers") != planner_headers:
        raise ValueError("frontier planner headers differ from source-header report")

    if pilot.get("baseline_metrics_sha256") != baseline_sha256:
        raise ValueError("frontier pilot baseline metrics checksum mismatch")
    if pilot.get("source_index_sha256") != source_index_sha256:
        raise ValueError("frontier pilot source index checksum mismatch")
    if pilot.get("imatrix_sha256") != imatrix_sha256:
        raise ValueError("frontier pilot imatrix checksum mismatch")
    _require_matching_shard_subset(
        pilot.get("source_shards"),
        source_shards,
        "frontier pilot",
    )
    pilot_results = validate_frontier_pilot_screen_matrix(pilot)

    if boundary.get("baseline_metrics_sha256") != baseline_sha256:
        raise ValueError("frontier boundary baseline checksum mismatch")
    if boundary.get("screen_report_sha256") != pilot_sha256:
        raise ValueError("frontier boundary pilot checksum mismatch")
    if boundary.get("source_headers_sha256") != planner_headers_sha256:
        raise ValueError("frontier boundary source headers checksum mismatch")
    if boundary.get("source_index_sha256") != source_index_sha256:
        raise ValueError("frontier boundary source index checksum mismatch")
    if boundary.get("imatrix_sha256") != imatrix_sha256:
        raise ValueError("frontier boundary imatrix checksum mismatch")
    _require_matching_shard_subset(
        boundary.get("source_shards"),
        source_shards,
        "frontier boundary",
    )
    boundary_layers = _require_layers(boundary.get("layers"), "frontier boundary")

    if full.get("boundary_report_sha256") != boundary_sha256:
        raise ValueError("frontier full screen boundary checksum mismatch")
    if full.get("source_index_sha256") != source_index_sha256:
        raise ValueError("frontier full screen source index checksum mismatch")
    if full.get("imatrix_sha256") != imatrix_sha256:
        raise ValueError("frontier full screen imatrix checksum mismatch")
    full_layers = _require_layers(full.get("layers"), "frontier full screen")
    if not set(boundary_layers) <= set(full_layers):
        raise ValueError("frontier full screen omits final boundary layers")
    if tuple(full.get("decision_boundary_layers", ())) != boundary_layers:
        raise ValueError("frontier full screen decision boundary differs from report")
    _require_matching_shard_subset(
        full.get("source_shards"),
        source_shards,
        "frontier full screen",
    )
    full_results = validate_frontier_full_screen_matrix(full, full_layers)

    boundary_layer_set = set(full_layers)
    merged_results = (
        tuple(
            result
            for result in pilot_results
            if _result_layer(result) not in boundary_layer_set
        )
        + full_results
    )
    return FrontierRecipeEvidence(
        baseline_metrics_sha256=baseline_sha256,
        pilot_screen_report_sha256=pilot_sha256,
        boundary_report_sha256=boundary_sha256,
        full_screen_report_sha256=full_sha256,
        source_headers_report_sha256=source_report_sha256,
        source_headers_sha256=planner_headers_sha256,
        source_index_sha256=source_index_sha256,
        source_shards_sha256=source_shards,
        source_assets_sha256=source_assets,
        imatrix_sha256=imatrix_sha256,
        boundary_layers=boundary_layers,
        merged_screen_results=merged_results,
    )


def validate_frontier_conversion_inputs(
    source_directory: Path,
    baseline_directory: Path,
    imatrix_path: Path,
    recipe_bundle: Mapping[str, Any],
) -> None:
    """Reject stale or altered conversion inputs before opening output shards."""
    validate_frontier_recipe_bundle_shape(recipe_bundle)
    source_index_path = source_directory / "model.safetensors.index.json"
    if file_sha256(source_index_path) != recipe_bundle["source_index_sha256"]:
        raise ValueError("frontier conversion source index checksum mismatch")
    if file_sha256(imatrix_path) != recipe_bundle["imatrix_sha256"]:
        raise ValueError("frontier conversion imatrix checksum mismatch")
    baseline_metrics_path = baseline_directory / "conversion-metrics.json"
    if file_sha256(baseline_metrics_path) != recipe_bundle["baseline_metrics_sha256"]:
        raise ValueError("frontier conversion baseline metrics checksum mismatch")

    source_weight_map = load_source_weight_map(source_index_path)
    expected_shards = set(source_weight_map.values())
    expected_hashes = _require_shard_hashes(
        recipe_bundle.get("source_shards_sha256"),
        "frontier recipe bundle",
    )
    if set(expected_hashes) != expected_shards:
        raise ValueError("frontier conversion source shard inventory mismatch")
    for shard_name, expected_sha256 in sorted(expected_hashes.items()):
        shard_path = source_directory / shard_name
        if not shard_path.is_file() or file_sha256(shard_path) != expected_sha256:
            raise ValueError(
                f"frontier conversion source shard checksum mismatch: {shard_name}"
            )
    expected_assets = _require_asset_hashes(
        recipe_bundle.get("source_assets_sha256"),
        "frontier recipe bundle",
    )
    actual_asset_names = {
        path.name
        for path in source_directory.iterdir()
        if path.is_file()
        and path.name != source_index_path.name
        and not path.name.endswith(".safetensors")
    }
    if actual_asset_names != set(expected_assets):
        raise ValueError("frontier conversion source asset inventory mismatch")
    for asset_name, expected_sha256 in sorted(expected_assets.items()):
        if file_sha256(source_directory / asset_name) != expected_sha256:
            raise ValueError(
                f"frontier conversion source asset checksum mismatch: {asset_name}"
            )


def validate_frontier_recipe_bundle_shape(recipe_bundle: Mapping[str, Any]) -> None:
    """Validate the durable frontier recipe and provenance envelope."""
    if recipe_bundle.get("schema_version") != 1:
        raise ValueError("frontier recipe bundle schema version is unsupported")
    for key in (
        "baseline_metrics_sha256",
        "pilot_screen_report_sha256",
        "boundary_report_sha256",
        "screen_report_sha256",
        "source_headers_report_sha256",
        "source_headers_sha256",
        "source_index_sha256",
        "imatrix_sha256",
    ):
        _require_sha256(recipe_bundle.get(key), f"frontier recipe bundle {key}")
    _require_shard_hashes(
        recipe_bundle.get("source_shards_sha256"),
        "frontier recipe bundle",
    )
    _require_asset_hashes(
        recipe_bundle.get("source_assets_sha256"),
        "frontier recipe bundle",
    )
    candidates = recipe_bundle.get("candidates")
    if not isinstance(candidates, Mapping) or set(candidates) != set(
        _FRONTIER_CANDIDATES
    ):
        raise ValueError("frontier recipe bundle candidate set is incomplete")
    summaries = recipe_bundle.get("candidate_summaries")
    if (
        not isinstance(summaries, list)
        or tuple(
            summary.get("name") if isinstance(summary, Mapping) else None
            for summary in summaries
        )
        != _FRONTIER_CANDIDATES
    ):
        raise ValueError("frontier recipe bundle summaries are incomplete or unordered")


def validate_frontier_pilot_screen_matrix(
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw_samples = report.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("frontier pilot has no samples")
    samples_per_projection = _require_integer(
        report.get("samples_per_projection"),
        "frontier pilot samples_per_projection",
    )
    if samples_per_projection < 2:
        raise ValueError("frontier pilot requires at least two samples per projection")
    samples: set[tuple[int, int, str]] = set()
    sample_counts = {
        (layer, projection): 0
        for layer in range(43)
        for projection in ("w1", "w2", "w3")
    }
    for sample in raw_samples:
        if not isinstance(sample, Mapping):
            raise ValueError("frontier pilot contains invalid sample")
        identity = (
            _require_integer(sample.get("layer"), "frontier pilot layer"),
            _require_integer(sample.get("expert"), "frontier pilot expert"),
            sample.get("projection"),
        )
        if (
            not 0 <= identity[0] < 43
            or not 0 <= identity[1] < 256
            or identity[2] not in ("w1", "w2", "w3")
            or identity in samples
        ):
            raise ValueError("frontier pilot contains duplicate or invalid sample")
        normalized_identity = (identity[0], identity[1], str(identity[2]))
        samples.add(normalized_identity)
        sample_counts[(normalized_identity[0], normalized_identity[2])] += 1
    invalid_counts = {
        key: count
        for key, count in sample_counts.items()
        if count != samples_per_projection
    }
    if invalid_counts:
        raise ValueError(
            "frontier pilot layer/projection sample matrix is incomplete: "
            f"{list(invalid_counts.items())[:3]}"
        )
    return _validate_screen_results(report, samples)


def validate_frontier_full_screen_matrix(
    report: Mapping[str, Any],
    layers: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    samples = {
        (layer, expert, projection)
        for layer in layers
        for expert in range(256)
        for projection in ("w1", "w2", "w3")
    }
    return _validate_screen_results(report, samples)


def _validate_screen_results(
    report: Mapping[str, Any],
    samples: set[tuple[int, int, str]],
) -> tuple[dict[str, Any], ...]:
    if tuple(report.get("group_sizes", ())) != _FRONTIER_GROUP_SIZES:
        raise ValueError("frontier screen group sizes are incomplete or unordered")
    raw_results = report.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("frontier screen results must be a list")
    expected = {
        (layer, expert, projection, bits, group_size)
        for layer, expert, projection in samples
        for group_size in _FRONTIER_GROUP_SIZES
        for bits in ((2, 4) if projection == "w2" else (2,))
    }
    observed: set[tuple[int, int, str, int, int]] = set()
    normalized: list[dict[str, Any]] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            raise ValueError("frontier screen contains invalid result")
        tensor_name = raw_result.get("tensor_name")
        match = _ROUTED_WEIGHT_NAME.fullmatch(str(tensor_name))
        if match is None:
            raise ValueError("frontier screen contains invalid tensor name")
        key = (
            int(match["layer"]),
            int(match["expert"]),
            match["projection"],
            _require_integer(raw_result.get("bits"), "frontier screen bits"),
            _require_integer(
                raw_result.get("group_size"),
                "frontier screen group size",
            ),
        )
        if key in observed:
            raise ValueError("frontier screen contains duplicate result")
        observed.add(key)
        normalized.append(raw_result)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "frontier screen result matrix is incomplete: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    return tuple(normalized)


def _result_layer(result: Mapping[str, Any]) -> int:
    match = _ROUTED_WEIGHT_NAME.fullmatch(str(result.get("tensor_name")))
    assert match is not None
    return int(match["layer"])


def _require_matching_shard_subset(
    candidate: Any,
    expected: Mapping[str, str],
    label: str,
) -> None:
    candidate_hashes = _require_shard_hashes(candidate, label)
    mismatches = {
        shard: digest
        for shard, digest in candidate_hashes.items()
        if expected.get(shard) != digest
    }
    if mismatches:
        raise ValueError(f"{label} source shard checksums mismatch")


def _require_shard_hashes(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} has no source shard checksums")
    hashes: dict[str, str] = {}
    for shard_name, digest in value.items():
        if (
            not isinstance(shard_name, str)
            or Path(shard_name).name != shard_name
            or not shard_name.endswith(".safetensors")
        ):
            raise ValueError(f"{label} contains invalid source shard name")
        hashes[shard_name] = _require_sha256(digest, f"{label} shard {shard_name}")
    return dict(sorted(hashes.items()))


def _require_asset_hashes(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or "config.json" not in value:
        raise ValueError(f"{label} has no complete source asset checksums")
    hashes: dict[str, str] = {}
    for asset_name, digest in value.items():
        if not isinstance(asset_name, str) or Path(asset_name).name != asset_name:
            raise ValueError(f"{label} contains invalid source asset name")
        hashes[asset_name] = _require_sha256(
            digest,
            f"{label} asset {asset_name}",
        )
    return dict(sorted(hashes.items()))


def _require_layers(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} has no layers")
    if any(not isinstance(layer, int) or isinstance(layer, bool) for layer in value):
        raise ValueError(f"{label} layers must be integers")
    layers = tuple(value)
    if layers != tuple(sorted(set(layers))) or any(
        layer < 0 or layer >= 43 for layer in layers
    ):
        raise ValueError(f"{label} layers are invalid or unordered")
    return layers


def _require_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"frontier JSON input must be an object: {path}")
    return payload
