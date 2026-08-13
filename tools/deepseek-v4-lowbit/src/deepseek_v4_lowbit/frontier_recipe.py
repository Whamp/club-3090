from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from deepseek_v4_lowbit.artifact_plan import (
    ArtifactRecipe,
    LayerQuantization,
    load_tensor_headers,
    plan_artifact,
)

_ROUTED_WEIGHT_NAME = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>w[123])\.weight$"
)
_GROUP_SIZES = (128, 256, 512)
_GIB = 1024**3
_CANDIDATE_TARGET_BYTES = {
    "cliff": 297 * _GIB // 4,
    "capacity": 599 * _GIB // 8,
    "balanced": 305 * _GIB // 4,
    "quality": 315 * _GIB // 4,
}
_UNSLOTH_W4_DOWN_ANCHORS = frozenset({26, 42})
_ANTIREZ_LATE_LAYER_ANCHORS = frozenset(range(37, 43))
_PATTERN_W13_GROUP128_ANCHORS = _UNSLOTH_W4_DOWN_ANCHORS | _ANTIREZ_LATE_LAYER_ANCHORS


@dataclass(frozen=True)
class FrontierLayerSensitivity:
    """Per-layer gate/up and down error effects for precision allocation."""

    layer: int
    baseline_w13_w2_g128_error: float
    baseline_down_w2_g128_error: float
    w13_w2_g128_error_ratio: float
    w13_w2_g256_error_ratio: float
    w13_w2_g512_error_ratio: float
    down_w2_g128_error_ratio: float
    down_w2_g256_error_ratio: float
    down_w2_g512_error_ratio: float
    w4_down_g128_error_ratio: float
    w4_down_g256_error_ratio: float
    w4_down_g512_error_ratio: float
    selected_expert_count: int


@dataclass(frozen=True)
class FrontierCandidateSummary:
    """Exact size and layer assignment for one frontier artifact candidate."""

    name: str
    byte_budget: int
    unused_budget_bytes: int
    total_bytes: int
    total_gib: float
    whole_model_bits_per_parameter: float
    w13_group128_layers: tuple[int, ...]
    w13_group256_layers: tuple[int, ...]
    w13_group512_layers: tuple[int, ...]
    w2_group128_layers: tuple[int, ...]
    w2_group256_layers: tuple[int, ...]
    w2_group512_layers: tuple[int, ...]
    w4_down_layers: tuple[int, ...]


@dataclass(frozen=True)
class FrontierBoundaryLayers:
    """Layers requiring full-expert screening before recipe allocation."""

    layers: tuple[int, ...]
    reasons: dict[int, tuple[str, ...]]


@dataclass(frozen=True)
class FrontierStorageSummary:
    """Exact model-payload storage across baseline and nested candidates."""

    baseline_reused_shards: int
    baseline_reused_shard_names: tuple[str, ...]
    unique_candidate_shards: int
    total_candidate_shard_references: int
    baseline_model_payload_bytes: int
    projected_additional_unique_model_payload_bytes: int
    projected_unique_model_payload_bytes: int
    projected_local_peak_model_payload_bytes: int


@dataclass(frozen=True)
class FrontierRecipeBundle:
    """Deterministic recipes bound to every quantization-frontier input."""

    candidate_summaries: tuple[FrontierCandidateSummary, ...]
    storage_summary: FrontierStorageSummary
    candidates: dict[str, dict[str, Any]]
    layer_sensitivity: tuple[FrontierLayerSensitivity, ...]
    baseline_metrics_sha256: str
    pilot_screen_report_sha256: str
    boundary_report_sha256: str
    screen_report_sha256: str
    source_headers_sha256: str
    source_headers_report_sha256: str
    source_index_sha256: str
    source_shards_sha256: dict[str, str]
    source_assets_sha256: dict[str, str]
    imatrix_sha256: str
    model_parameter_count: int


def select_frontier_boundary_layers(
    baseline_metrics: Iterable[Mapping[str, Any]],
    screen_results: Iterable[Mapping[str, Any]],
    *,
    tensor_headers_path: Path,
    candidate_target_bytes: Mapping[str, int] | None = None,
) -> FrontierBoundaryLayers:
    """Select exact-budget decision boundaries for all-expert confirmation."""
    sensitivity = _summarize_layer_sensitivity(
        _baseline_errors_by_tensor(baseline_metrics),
        _screen_results_by_tensor_schema(screen_results),
    )
    headers = load_tensor_headers(tensor_headers_path)
    target_bytes = dict(candidate_target_bytes or _CANDIDATE_TARGET_BYTES)
    recipes = _build_nested_frontier_recipes(
        sensitivity,
        headers,
        candidate_target_bytes=target_bytes,
    )
    sensitivity_by_layer = {item.layer: item for item in sensitivity}
    reasons: dict[int, set[str]] = {}
    for layer in _UNSLOTH_W4_DOWN_ANCHORS:
        reasons.setdefault(layer, set()).add("unsloth-w4-down-anchor")
    for layer in _ANTIREZ_LATE_LAYER_ANCHORS:
        reasons.setdefault(layer, set()).add("antirez-late-layer-anchor")

    boundary_moves = (
        ("cliff", "w2", ((256, 128),)),
        ("capacity", "w13", ((512, 256), (256, 128))),
        ("balanced", "w13", ((512, 256), (256, 128))),
        ("quality", "w13", ((512, 256), (256, 128))),
    )
    for candidate_name, projection_family, transitions in boundary_moves:
        recipe = recipes[candidate_name]
        for source_group, target_group in transitions:
            selected = []
            unselected = []
            for layer in range(43):
                score = _group_upgrade_score(
                    sensitivity_by_layer[layer],
                    projection_family,
                    source_group,
                    target_group,
                    _group_upgrade_added_bytes(
                        headers,
                        layer,
                        projection_family,
                        source_group,
                        target_group,
                    ),
                )
                group_size = recipe.group_size_for(
                    layer,
                    projection_family,
                    fallback=512,
                )
                if group_size <= target_group:
                    selected.append((score, layer))
                elif group_size == source_group:
                    unselected.append((score, layer))
            _record_decision_boundary(
                reasons,
                selected,
                unselected,
                f"{candidate_name}-{projection_family}-group-boundary",
            )

    for candidate_name in ("balanced", "quality"):
        recipe = recipes[candidate_name]
        selected = []
        unselected = []
        for layer in range(43):
            group_size = recipe.group_size_for(layer, "w2", fallback=128)
            score = _w4_down_error_reduction(
                sensitivity_by_layer[layer],
                group_size,
            )
            destination = selected if recipe.bits_for(layer, "w2") == 4 else unselected
            destination.append((score, layer))
        _record_decision_boundary(
            reasons,
            selected,
            unselected,
            f"{candidate_name}-w4-down-boundary",
        )

    return FrontierBoundaryLayers(
        layers=tuple(sorted(reasons)),
        reasons={
            layer: tuple(sorted(layer_reasons))
            for layer, layer_reasons in sorted(reasons.items())
        },
    )


def _record_decision_boundary(
    reasons: dict[int, set[str]],
    selected: list[tuple[float, int]],
    unselected: list[tuple[float, int]],
    reason: str,
) -> None:
    """Record the weakest accepted and strongest rejected layer for one move."""
    if selected:
        selected_layer = min(selected)[1]
        reasons.setdefault(selected_layer, set()).add(reason)
    if unselected:
        unselected_layer = max(unselected)[1]
        reasons.setdefault(unselected_layer, set()).add(reason)


def build_frontier_recipe_bundle(
    baseline_metrics: Iterable[Mapping[str, Any]],
    screen_results: Iterable[Mapping[str, Any]],
    *,
    tensor_headers_path: Path,
    baseline_metrics_sha256: str,
    screen_report_sha256: str,
    source_headers_sha256: str,
    source_headers_report_sha256: str,
    source_index_sha256: str,
    source_shards_sha256: Mapping[str, str] | None = None,
    source_assets_sha256: Mapping[str, str] | None = None,
    imatrix_sha256: str,
    pilot_screen_report_sha256: str,
    boundary_report_sha256: str,
    model_parameter_count: int,
    candidate_target_bytes: Mapping[str, int] | None = None,
) -> FrontierRecipeBundle:
    """Allocate cliff/capacity/balanced/quality recipes from measured error."""
    if model_parameter_count <= 0:
        raise ValueError("frontier model_parameter_count must be positive")
    baseline_by_tensor = _baseline_errors_by_tensor(baseline_metrics)
    results_by_tensor_schema = _screen_results_by_tensor_schema(screen_results)
    layer_sensitivity = _summarize_layer_sensitivity(
        baseline_by_tensor,
        results_by_tensor_schema,
    )

    headers = load_tensor_headers(tensor_headers_path)
    target_bytes = dict(candidate_target_bytes or _CANDIDATE_TARGET_BYTES)
    if tuple(target_bytes) != ("cliff", "capacity", "balanced", "quality"):
        raise ValueError("frontier candidate byte budgets must be ordered and complete")
    if list(target_bytes.values()) != sorted(target_bytes.values()):
        raise ValueError("frontier candidate byte budgets must be nondecreasing")
    recipes = _build_nested_frontier_recipes(
        layer_sensitivity,
        headers,
        candidate_target_bytes=target_bytes,
    )
    summaries = tuple(
        _summarize_candidate(
            name,
            recipe,
            headers,
            byte_budget=target_bytes[name],
            model_parameter_count=model_parameter_count,
        )
        for name, recipe in recipes.items()
    )
    storage_summary = _summarize_frontier_storage(recipes, headers)
    return FrontierRecipeBundle(
        candidate_summaries=summaries,
        storage_summary=storage_summary,
        candidates={name: _recipe_payload(recipe) for name, recipe in recipes.items()},
        layer_sensitivity=tuple(sorted(layer_sensitivity, key=lambda item: item.layer)),
        baseline_metrics_sha256=baseline_metrics_sha256,
        pilot_screen_report_sha256=pilot_screen_report_sha256,
        boundary_report_sha256=boundary_report_sha256,
        screen_report_sha256=screen_report_sha256,
        source_headers_sha256=source_headers_sha256,
        source_headers_report_sha256=source_headers_report_sha256,
        source_index_sha256=source_index_sha256,
        source_shards_sha256=dict(sorted((source_shards_sha256 or {}).items())),
        source_assets_sha256=dict(sorted((source_assets_sha256 or {}).items())),
        imatrix_sha256=imatrix_sha256,
        model_parameter_count=model_parameter_count,
    )


def frontier_recipe_bundle_payload(bundle: FrontierRecipeBundle) -> dict[str, Any]:
    """Serialize a frontier recipe bundle for durable rental handoff."""
    return {
        "schema_version": 1,
        "baseline_metrics_sha256": bundle.baseline_metrics_sha256,
        "pilot_screen_report_sha256": bundle.pilot_screen_report_sha256,
        "boundary_report_sha256": bundle.boundary_report_sha256,
        "screen_report_sha256": bundle.screen_report_sha256,
        "source_headers_sha256": bundle.source_headers_sha256,
        "source_headers_report_sha256": bundle.source_headers_report_sha256,
        "source_index_sha256": bundle.source_index_sha256,
        "source_shards_sha256": bundle.source_shards_sha256,
        "source_assets_sha256": bundle.source_assets_sha256,
        "imatrix_sha256": bundle.imatrix_sha256,
        "model_parameter_count": bundle.model_parameter_count,
        "candidate_summaries": [
            asdict(summary) for summary in bundle.candidate_summaries
        ],
        "storage_summary": asdict(bundle.storage_summary),
        "candidates": bundle.candidates,
        "layer_sensitivity": [
            asdict(sensitivity) for sensitivity in bundle.layer_sensitivity
        ],
    }


def _baseline_errors_by_tensor(
    metrics: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for metric in metrics:
        tensor_name = _required_tensor_name(metric)
        weighted_error = _required_nonnegative_float(metric, "weighted_error")
        if tensor_name in errors:
            raise ValueError(f"frontier baseline has duplicate tensor: {tensor_name}")
        errors[tensor_name] = weighted_error
    if len(errors) != 43 * 256 * 3:
        raise ValueError(
            f"frontier baseline requires 33024 tensors, found {len(errors)}"
        )
    return errors


def _screen_results_by_tensor_schema(
    results: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, int, int], float]:
    indexed: dict[tuple[str, int, int], float] = {}
    for result in results:
        tensor_name = _required_tensor_name(result)
        bits = _required_integer(result, "bits")
        group_size = _required_integer(result, "group_size")
        weighted_error = _required_nonnegative_float(
            result,
            "selection_error" if "selection_error" in result else "weighted_error",
        )
        key = (tensor_name, bits, group_size)
        if key in indexed:
            raise ValueError(f"frontier screen has duplicate schema result: {key}")
        indexed[key] = weighted_error
    if not indexed:
        raise ValueError("frontier screen has no results")
    return indexed


def _summarize_layer_sensitivity(
    baseline_by_tensor: Mapping[str, float],
    results_by_tensor_schema: Mapping[tuple[str, int, int], float],
) -> tuple[FrontierLayerSensitivity, ...]:
    samples_by_layer: dict[int, set[int]] = {layer: set() for layer in range(43)}
    ratios: dict[tuple[int, str, int, int], list[float]] = {}
    selected_w13_baseline: dict[int, dict[str, float]] = {
        layer: {} for layer in range(43)
    }
    selected_down_baseline: dict[int, dict[str, float]] = {
        layer: {} for layer in range(43)
    }

    for tensor_name, bits, group_size in results_by_tensor_schema:
        match = _ROUTED_WEIGHT_NAME.fullmatch(tensor_name)
        assert match is not None
        layer = int(match["layer"])
        expert = int(match["expert"])
        projection = match["projection"]
        if projection != "w2" and bits != 2:
            raise ValueError("frontier screen may use W4 only for down projections")
        samples_by_layer[layer].add(expert)
        if tensor_name not in baseline_by_tensor:
            raise ValueError(f"frontier tensor is absent from baseline: {tensor_name}")
        screened_baseline_error = results_by_tensor_schema.get((tensor_name, 2, 128))
        if screened_baseline_error is None:
            raise ValueError(
                f"frontier screen lacks W2 group-128 baseline: {tensor_name}"
            )
        selected_baseline = (
            selected_down_baseline if projection == "w2" else selected_w13_baseline
        )
        selected_baseline[layer][tensor_name] = screened_baseline_error
        candidate_error = results_by_tensor_schema[(tensor_name, bits, group_size)]
        ratio = _error_ratio(candidate_error, screened_baseline_error)
        ratios.setdefault((layer, projection, bits, group_size), []).append(ratio)

    summaries: list[FrontierLayerSensitivity] = []
    for layer in range(43):
        w13_w2 = {
            group_size: _w13_w2_ratios(ratios, layer, group_size)
            for group_size in _GROUP_SIZES
        }
        down_w2 = {
            group_size: _require_ratio_values(
                ratios,
                layer,
                "w2",
                2,
                group_size,
            )
            for group_size in _GROUP_SIZES
        }
        w4_down = {
            group_size: _require_ratio_values(
                ratios,
                layer,
                "w2",
                4,
                group_size,
            )
            for group_size in _GROUP_SIZES
        }
        summaries.append(
            FrontierLayerSensitivity(
                layer=layer,
                baseline_w13_w2_g128_error=median(
                    selected_w13_baseline[layer].values()
                ),
                baseline_down_w2_g128_error=median(
                    selected_down_baseline[layer].values()
                ),
                w13_w2_g128_error_ratio=median(w13_w2[128]),
                w13_w2_g256_error_ratio=median(w13_w2[256]),
                w13_w2_g512_error_ratio=median(w13_w2[512]),
                down_w2_g128_error_ratio=median(down_w2[128]),
                down_w2_g256_error_ratio=median(down_w2[256]),
                down_w2_g512_error_ratio=median(down_w2[512]),
                w4_down_g128_error_ratio=median(w4_down[128]),
                w4_down_g256_error_ratio=median(w4_down[256]),
                w4_down_g512_error_ratio=median(w4_down[512]),
                selected_expert_count=len(samples_by_layer[layer]),
            )
        )
    return tuple(summaries)


def _w13_w2_ratios(
    ratios: Mapping[tuple[int, str, int, int], list[float]],
    layer: int,
    group_size: int,
) -> list[float]:
    values = []
    for projection in ("w1", "w3"):
        values.extend(
            _require_ratio_values(
                ratios,
                layer,
                projection,
                2,
                group_size,
            )
        )
    return values


def _require_ratio_values(
    ratios: Mapping[tuple[int, str, int, int], list[float]],
    layer: int,
    projection: str,
    bits: int,
    group_size: int,
) -> list[float]:
    values = ratios.get((layer, projection, bits, group_size))
    if not values:
        raise ValueError(
            f"frontier screen lacks layer {layer} {projection} W{bits} "
            f"group-{group_size} results"
        )
    return values


def _build_nested_frontier_recipes(
    layer_sensitivity: tuple[FrontierLayerSensitivity, ...],
    headers: Any,
    *,
    candidate_target_bytes: Mapping[str, int],
) -> dict[str, ArtifactRecipe]:
    """Allocate a projection-sensitive nested ladder under exact ceilings."""
    w13_group_sizes = {layer: 512 for layer in range(43)}
    w2_group_sizes = {layer: 256 for layer in range(43)}
    w4_down_layers: set[int] = set()
    recipes: dict[str, ArtifactRecipe] = {}
    minimum_recipe = _recipe_from_assignments(
        w13_group_sizes,
        w2_group_sizes,
        w4_down_layers,
    )
    minimum_bytes = plan_artifact(headers, minimum_recipe, group_size=128).total_bytes
    for candidate_name, byte_budget in candidate_target_bytes.items():
        if byte_budget < minimum_bytes:
            raise ValueError(
                f"frontier {candidate_name} byte budget {byte_budget} is below "
                f"the projection-sensitive W2 minimum {minimum_bytes}"
            )

    group_upgrade_order = [
        (
            _group_upgrade_score(
                item,
                projection_family,
                source_group,
                target_group,
                _group_upgrade_added_bytes(
                    headers,
                    item.layer,
                    projection_family,
                    source_group,
                    target_group,
                ),
            ),
            projection_family,
            item.layer,
            source_group,
            target_group,
        )
        for item in layer_sensitivity
        for projection_family, transitions in (
            ("w13", ((512, 256), (256, 128))),
            ("w2", ((256, 128),)),
        )
        for source_group, target_group in transitions
    ]
    group_upgrade_order.sort(reverse=True)

    for candidate_name in ("cliff", "capacity", "balanced", "quality"):
        if candidate_name == "capacity":
            w2_group_sizes.update({layer: 128 for layer in range(43)})
        elif candidate_name == "balanced":
            for layer in _PATTERN_W13_GROUP128_ANCHORS:
                w13_group_sizes[layer] = 128
            w4_down_layers.update(_UNSLOTH_W4_DOWN_ANCHORS)
        elif candidate_name == "quality":
            w4_down_layers.update(_ANTIREZ_LATE_LAYER_ANCHORS)

        _require_projection_precision_hierarchy(w13_group_sizes, w2_group_sizes)
        _apply_group_upgrades(
            w13_group_sizes,
            w2_group_sizes,
            w4_down_layers,
            group_upgrade_order,
            headers,
            allowed_projection_families=(
                frozenset({"w2"}) if candidate_name == "cliff" else frozenset({"w13"})
            ),
            byte_budget=candidate_target_bytes[candidate_name],
        )
        if candidate_name in {"balanced", "quality"}:
            _apply_w4_down_upgrades(
                w13_group_sizes,
                w2_group_sizes,
                w4_down_layers,
                layer_sensitivity,
                headers,
                byte_budget=candidate_target_bytes[candidate_name],
            )
        recipe = _recipe_from_assignments(
            w13_group_sizes,
            w2_group_sizes,
            w4_down_layers,
        )
        _require_recipe_within_budget(
            candidate_name,
            recipe,
            headers,
            candidate_target_bytes[candidate_name],
        )
        recipes[candidate_name] = recipe
    return recipes


def _require_recipe_within_budget(
    candidate_name: str,
    recipe: ArtifactRecipe,
    headers: Any,
    byte_budget: int,
) -> None:
    total_bytes = plan_artifact(headers, recipe, group_size=128).total_bytes
    if total_bytes > byte_budget:
        raise ValueError(
            f"frontier {candidate_name} recipe exceeds its byte budget: "
            f"{total_bytes} > {byte_budget}"
        )


def _require_projection_precision_hierarchy(
    w13_group_sizes: Mapping[int, int],
    w2_group_sizes: Mapping[int, int],
) -> None:
    for layer in range(43):
        if w2_group_sizes[layer] > w13_group_sizes[layer]:
            raise ValueError(
                "frontier down projection cannot use a coarser group size than "
                f"gate/up at layer {layer}"
            )


def _apply_group_upgrades(
    w13_group_sizes: dict[int, int],
    w2_group_sizes: dict[int, int],
    w4_down_layers: set[int],
    upgrade_order: list[tuple[float, str, int, int, int]],
    headers: Any,
    *,
    allowed_projection_families: frozenset[str],
    byte_budget: int,
) -> None:
    group_sizes_by_family = {"w13": w13_group_sizes, "w2": w2_group_sizes}
    while True:
        applied = False
        for score, family, layer, source_group, target_group in upgrade_order:
            group_sizes = group_sizes_by_family[family]
            if (
                family not in allowed_projection_families
                or score <= 0
                or group_sizes[layer] != source_group
            ):
                continue
            trial_w13 = dict(w13_group_sizes)
            trial_w2 = dict(w2_group_sizes)
            trial_group_sizes = trial_w13 if family == "w13" else trial_w2
            trial_group_sizes[layer] = target_group
            try:
                _require_projection_precision_hierarchy(trial_w13, trial_w2)
            except ValueError:
                continue
            trial_recipe = _recipe_from_assignments(
                trial_w13,
                trial_w2,
                w4_down_layers,
            )
            if (
                plan_artifact(headers, trial_recipe, group_size=128).total_bytes
                <= byte_budget
            ):
                group_sizes[layer] = target_group
                applied = True
                break
        if not applied:
            return


def _apply_w4_down_upgrades(
    w13_group_sizes: Mapping[int, int],
    w2_group_sizes: Mapping[int, int],
    w4_down_layers: set[int],
    layer_sensitivity: tuple[FrontierLayerSensitivity, ...],
    headers: Any,
    *,
    byte_budget: int,
) -> None:
    ranked_layers = sorted(
        layer_sensitivity,
        key=lambda item: (
            _w4_down_upgrade_score(
                item,
                w2_group_sizes[item.layer],
                _w4_down_added_bytes(
                    headers,
                    item.layer,
                    w13_group_sizes[item.layer],
                    w2_group_sizes[item.layer],
                ),
            ),
            item.baseline_down_w2_g128_error,
            item.layer,
        ),
        reverse=True,
    )
    for item in ranked_layers:
        if item.layer in w4_down_layers:
            continue
        if _w4_down_error_reduction(item, w2_group_sizes[item.layer]) <= 0:
            continue
        trial_w4_layers = {*w4_down_layers, item.layer}
        trial_recipe = _recipe_from_assignments(
            w13_group_sizes,
            w2_group_sizes,
            trial_w4_layers,
        )
        if (
            plan_artifact(headers, trial_recipe, group_size=128).total_bytes
            <= byte_budget
        ):
            w4_down_layers.add(item.layer)


def _group_upgrade_score(
    item: FrontierLayerSensitivity,
    projection_family: str,
    source_group: int,
    target_group: int,
    added_bytes: int,
) -> float:
    source_ratio = _group_error_ratio(item, projection_family, source_group)
    target_ratio = _group_error_ratio(item, projection_family, target_group)
    baseline_error = (
        item.baseline_w13_w2_g128_error
        if projection_family == "w13"
        else item.baseline_down_w2_g128_error
    )
    value = baseline_error * max(0.0, source_ratio - target_ratio)
    return value / added_bytes if added_bytes > 0 else 0.0


def _group_upgrade_added_bytes(
    headers: Any,
    layer: int,
    projection_family: str,
    source_group: int,
    target_group: int,
) -> int:
    w13_group_sizes = {candidate_layer: 512 for candidate_layer in range(43)}
    w2_group_sizes = {candidate_layer: 256 for candidate_layer in range(43)}
    group_sizes = w13_group_sizes if projection_family == "w13" else w2_group_sizes
    if projection_family == "w13":
        w2_group_sizes[layer] = min(w2_group_sizes[layer], target_group)
    group_sizes[layer] = source_group
    source_recipe = _recipe_from_assignments(
        w13_group_sizes,
        w2_group_sizes,
        set(),
    )
    group_sizes[layer] = target_group
    target_recipe = _recipe_from_assignments(
        w13_group_sizes,
        w2_group_sizes,
        set(),
    )
    return (
        plan_artifact(headers, target_recipe, group_size=128).total_bytes
        - plan_artifact(headers, source_recipe, group_size=128).total_bytes
    )


def _group_error_ratio(
    item: FrontierLayerSensitivity,
    projection_family: str,
    group_size: int,
) -> float:
    ratios = (
        {
            128: item.w13_w2_g128_error_ratio,
            256: item.w13_w2_g256_error_ratio,
            512: item.w13_w2_g512_error_ratio,
        }
        if projection_family == "w13"
        else {
            128: item.down_w2_g128_error_ratio,
            256: item.down_w2_g256_error_ratio,
            512: item.down_w2_g512_error_ratio,
        }
    )
    try:
        return ratios[group_size]
    except KeyError as error:
        raise ValueError(f"unsupported frontier group size: {group_size}") from error


def _recipe_from_assignments(
    w13_group_sizes: Mapping[int, int],
    w2_group_sizes: Mapping[int, int],
    w4_down_layers: set[int],
) -> ArtifactRecipe:
    _require_projection_precision_hierarchy(w13_group_sizes, w2_group_sizes)
    layers = {
        layer: LayerQuantization(
            w13_bits=2,
            w2_bits=4 if layer in w4_down_layers else 2,
            w13_group_size=w13_group_sizes[layer],
            w2_group_size=w2_group_sizes[layer],
        )
        for layer in range(43)
    }
    return ArtifactRecipe(
        default=LayerQuantization(
            w13_bits=2,
            w2_bits=2,
            w13_group_size=512,
            w2_group_size=256,
        ),
        layers=layers,
    )


def _summarize_frontier_storage(
    recipes: Mapping[str, ArtifactRecipe],
    headers: tuple[Any, ...],
) -> FrontierStorageSummary:
    baseline_recipe = ArtifactRecipe(default=LayerQuantization(2, 2, 128))
    all_recipes = {"baseline": baseline_recipe, **recipes}
    headers_by_shard: dict[str, list[Any]] = {}
    for header in headers:
        headers_by_shard.setdefault(header.shard, []).append(header)

    payload_by_shard_signature: dict[tuple[str, tuple[Any, ...]], int] = {}
    signature_by_recipe_shard: dict[tuple[str, str], tuple[Any, ...]] = {}
    output_shards = []
    for shard_name, shard_headers in sorted(headers_by_shard.items()):
        for recipe_name, recipe in all_recipes.items():
            shard_plan = plan_artifact(tuple(shard_headers), recipe, group_size=128)
            if shard_plan.total_bytes == 0:
                continue
            if shard_name not in output_shards:
                output_shards.append(shard_name)
            signature = _frontier_shard_recipe_signature(shard_headers, recipe)
            signature_by_recipe_shard[(recipe_name, shard_name)] = signature
            payload_by_shard_signature.setdefault(
                (shard_name, signature),
                shard_plan.total_bytes,
            )

    baseline_payload = sum(
        payload_by_shard_signature[(shard_name, signature)]
        for (recipe_name, shard_name), signature in signature_by_recipe_shard.items()
        if recipe_name == "baseline"
    )
    baseline_reused_shard_names = tuple(
        shard_name
        for shard_name in output_shards
        if any(
            signature_by_recipe_shard[(candidate, shard_name)]
            == signature_by_recipe_shard[("baseline", shard_name)]
            for candidate in recipes
        )
    )
    baseline_reused_shards = len(baseline_reused_shard_names)
    candidate_unique_keys = {
        (shard_name, signature_by_recipe_shard[(candidate, shard_name)])
        for candidate in recipes
        for shard_name in output_shards
    }
    baseline_keys = {
        (shard_name, signature_by_recipe_shard[("baseline", shard_name)])
        for shard_name in output_shards
    }
    additional_keys = candidate_unique_keys - baseline_keys
    additional_payload = sum(payload_by_shard_signature[key] for key in additional_keys)
    unique_payload = baseline_payload + additional_payload

    candidate_order = tuple(recipes)
    local_peak = 0
    for previous, current in pairwise(candidate_order):
        transition_keys = {
            (shard_name, signature_by_recipe_shard[(candidate, shard_name)])
            for candidate in (previous, current)
            for shard_name in output_shards
        }
        local_peak = max(
            local_peak,
            sum(payload_by_shard_signature[key] for key in transition_keys),
        )
    if not local_peak:
        only_candidate = candidate_order[0]
        local_peak = sum(
            payload_by_shard_signature[
                (shard_name, signature_by_recipe_shard[(only_candidate, shard_name)])
            ]
            for shard_name in output_shards
        )
    return FrontierStorageSummary(
        baseline_reused_shards=baseline_reused_shards,
        baseline_reused_shard_names=baseline_reused_shard_names,
        unique_candidate_shards=len(candidate_unique_keys),
        total_candidate_shard_references=len(output_shards) * len(recipes),
        baseline_model_payload_bytes=baseline_payload,
        projected_additional_unique_model_payload_bytes=additional_payload,
        projected_unique_model_payload_bytes=unique_payload,
        projected_local_peak_model_payload_bytes=local_peak,
    )


def _frontier_shard_recipe_signature(
    shard_headers: list[Any],
    recipe: ArtifactRecipe,
) -> tuple[Any, ...]:
    signature = []
    for header in shard_headers:
        match = _ROUTED_WEIGHT_NAME.fullmatch(header.name)
        if match is None or not header.name.endswith(".weight"):
            continue
        layer = int(match["layer"])
        projection = match["projection"]
        signature.append(
            (
                layer,
                projection,
                recipe.bits_for(layer, projection),
                recipe.group_size_for(layer, projection, fallback=128),
            )
        )
    return tuple(sorted(signature))


def _summarize_candidate(
    name: str,
    recipe: ArtifactRecipe,
    headers: Any,
    *,
    byte_budget: int,
    model_parameter_count: int,
) -> FrontierCandidateSummary:
    plan = plan_artifact(headers, recipe, group_size=128)
    group_layers = {
        projection_family: {
            group_size: tuple(
                layer
                for layer in range(43)
                if recipe.group_size_for(
                    layer,
                    projection_family,
                    fallback=128,
                )
                == group_size
            )
            for group_size in _GROUP_SIZES
        }
        for projection_family in ("w13", "w2")
    }
    w4_down_layers = tuple(
        layer for layer in range(43) if recipe.bits_for(layer, "w2") == 4
    )
    return FrontierCandidateSummary(
        name=name,
        byte_budget=byte_budget,
        unused_budget_bytes=byte_budget - plan.total_bytes,
        total_bytes=plan.total_bytes,
        total_gib=plan.total_bytes / (1024**3),
        whole_model_bits_per_parameter=(plan.total_bytes * 8 / model_parameter_count),
        w13_group128_layers=group_layers["w13"][128],
        w13_group256_layers=group_layers["w13"][256],
        w13_group512_layers=group_layers["w13"][512],
        w2_group128_layers=group_layers["w2"][128],
        w2_group256_layers=group_layers["w2"][256],
        w2_group512_layers=group_layers["w2"][512],
        w4_down_layers=w4_down_layers,
    )


def _recipe_payload(recipe: ArtifactRecipe) -> dict[str, Any]:
    return {
        "default": _layer_payload(recipe.default),
        "layers": {
            str(layer): _layer_payload(quantization)
            for layer, quantization in sorted(recipe.layers.items())
        },
    }


def _layer_payload(quantization: LayerQuantization) -> dict[str, int]:
    payload = {
        "w13_bits": quantization.w13_bits,
        "w2_bits": quantization.w2_bits,
    }
    if quantization.group_size is not None:
        payload["group_size"] = quantization.group_size
    else:
        assert quantization.w13_group_size is not None
        assert quantization.w2_group_size is not None
        payload["w13_group_size"] = quantization.w13_group_size
        payload["w2_group_size"] = quantization.w2_group_size
    return payload


def _down_w2_error_ratio(item: FrontierLayerSensitivity, group_size: int) -> float:
    if group_size == 128:
        return item.down_w2_g128_error_ratio
    if group_size == 256:
        return item.down_w2_g256_error_ratio
    if group_size == 512:
        return item.down_w2_g512_error_ratio
    raise ValueError(f"unsupported frontier group size: {group_size}")


def _w4_down_error_ratio(item: FrontierLayerSensitivity, group_size: int) -> float:
    if group_size == 128:
        return item.w4_down_g128_error_ratio
    if group_size == 256:
        return item.w4_down_g256_error_ratio
    if group_size == 512:
        return item.w4_down_g512_error_ratio
    raise ValueError(f"unsupported frontier group size: {group_size}")


def _w4_down_upgrade_score(
    item: FrontierLayerSensitivity,
    group_size: int,
    added_bytes: int,
) -> float:
    value = _w4_down_error_reduction(item, group_size)
    return value / added_bytes if added_bytes > 0 else 0.0


def _w4_down_added_bytes(
    headers: Any,
    layer: int,
    w13_group_size: int,
    w2_group_size: int,
) -> int:
    w13_group_sizes = {candidate_layer: 512 for candidate_layer in range(43)}
    w2_group_sizes = {candidate_layer: 256 for candidate_layer in range(43)}
    w13_group_sizes[layer] = w13_group_size
    w2_group_sizes[layer] = w2_group_size
    w2_recipe = _recipe_from_assignments(
        w13_group_sizes,
        w2_group_sizes,
        set(),
    )
    w4_recipe = _recipe_from_assignments(
        w13_group_sizes,
        w2_group_sizes,
        {layer},
    )
    return (
        plan_artifact(headers, w4_recipe, group_size=128).total_bytes
        - plan_artifact(headers, w2_recipe, group_size=128).total_bytes
    )


def _w4_down_error_reduction(
    item: FrontierLayerSensitivity,
    group_size: int,
) -> float:
    relative_reduction = max(
        0.0,
        _down_w2_error_ratio(item, group_size) - _w4_down_error_ratio(item, group_size),
    )
    return item.baseline_down_w2_g128_error * relative_reduction


def _error_ratio(candidate_error: float, baseline_error: float) -> float:
    if baseline_error == 0:
        return 1.0 if candidate_error == 0 else math.inf
    return candidate_error / baseline_error


def _required_tensor_name(payload: Mapping[str, Any]) -> str:
    value = payload.get("tensor_name")
    if not isinstance(value, str) or _ROUTED_WEIGHT_NAME.fullmatch(value) is None:
        raise ValueError(f"frontier result has invalid tensor_name: {value!r}")
    return value


def _required_integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"frontier result requires integer {key}")
    return value


def _required_nonnegative_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"frontier result requires numeric {key}")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"frontier result requires finite nonnegative {key}")
    return converted


def load_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object with a path-specific validation error."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"frontier JSON input must be an object: {path}")
    return payload
