from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.artifact_plan import (
    ArtifactRecipe,
    LayerQuantization,
    TensorDisposition,
    classify_tensor,
)
from deepseek_v4_lowbit.convert_cli import (
    build_expected_output_weight_map,
    load_source_weight_map,
    materialize_converted_model_assets,
    select_output_shards,
    write_conversion_metrics_report,
)
from deepseek_v4_lowbit.frontier_provenance import (
    validate_frontier_conversion_inputs,
)
from deepseek_v4_lowbit.imatrix import ImatrixFile
from deepseek_v4_lowbit.safetensors_header import safetensors_inventory
from deepseek_v4_lowbit.shard_writer import (
    ResumableSafetensorsWriter,
    ShardIdentity,
    ShardReceipt,
    file_sha256,
)
from deepseek_v4_lowbit.source_transform import (
    QuantizerKind,
    ShardTransformResult,
    TransformOptions,
    metrics_from_shard_receipt,
    transform_recipe_sha256,
    transform_source_shard,
)

_SOURCE_INDEX_NAME = "model.safetensors.index.json"
_BASELINE_CONFIG_SHA256 = (
    "334bfa9f35a2f05510639538325c20e87a3980b06cabef4d750d3ca8085a0a66"
)
_BASELINE_INDEX_SHA256 = (
    "348657275f7e89750555b23b86a117177b487dd8414d00bdd457c67688284735"
)
_BASELINE_METRICS_SHA256 = (
    "57d77e13fd52901ad386c9aec442f83772beca149f3c25429b8b4588d5da3082"
)
FRONTIER_CANDIDATE_ORDER = ("cliff", "capacity", "balanced", "quality")


def validate_frontier_candidate_names(
    candidate_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Require a unique, nested-order subset of frontier candidates."""
    if not candidate_names:
        raise ValueError("frontier candidate selection must not be empty")
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("frontier candidate selection contains duplicates")
    unknown = set(candidate_names) - set(FRONTIER_CANDIDATE_ORDER)
    if unknown:
        raise ValueError(f"unknown frontier candidates: {sorted(unknown)}")
    positions = tuple(FRONTIER_CANDIDATE_ORDER.index(name) for name in candidate_names)
    if positions != tuple(sorted(positions)):
        raise ValueError("frontier candidates must follow nested frontier order")
    return candidate_names


@dataclass(frozen=True)
class FrontierCandidateConversion:
    """Verified output and reuse accounting for one frontier candidate."""

    name: str
    output_directory: Path
    recipe: ArtifactRecipe
    recipe_sha256: str
    results: tuple[ShardTransformResult, ...]
    reused_shard_count: int
    converted_shard_count: int


@dataclass(frozen=True)
class BaselineArtifactReuse:
    """Hash-bound immutable baseline metadata for reusable output shards."""

    directory: Path
    recipe: ArtifactRecipe
    recipe_sha256: str
    expected_weight_map: dict[str, str]
    output_sha256_by_shard: dict[str, str]
    metrics_by_shard: dict[str, list[dict[str, Any]]]

    def shard_receipt(
        self,
        shard_name: str,
        source_path: Path,
    ) -> ShardReceipt | None:
        """Build a verified receipt when the baseline shard exists locally."""
        output_path = self.directory / shard_name
        if not output_path.is_file():
            return None
        expected_sha256 = self.output_sha256_by_shard[shard_name]
        if file_sha256(output_path) != expected_sha256:
            raise ValueError(f"baseline shard checksum mismatch: {shard_name}")
        inventory = safetensors_inventory(output_path)
        expected_names = {
            tensor_name
            for tensor_name, expected_shard in self.expected_weight_map.items()
            if expected_shard == shard_name
        }
        if set(inventory) != expected_names:
            raise ValueError(f"baseline shard tensor inventory mismatch: {shard_name}")
        return ShardReceipt(
            shard_name=shard_name,
            identity=ShardIdentity(
                source_sha256=file_sha256(source_path),
                recipe_sha256=self.recipe_sha256,
            ),
            output_path=output_path,
            output_sha256=expected_sha256,
            output_bytes=output_path.stat().st_size,
            tensors=inventory,
            metadata={
                "transform_metrics": self.metrics_by_shard[shard_name],
            },
        )


def load_baseline_artifact_reuse(
    baseline_directory: Path,
    expected_weight_map: Mapping[str, str],
    expected_recipe_sha256: str,
) -> BaselineArtifactReuse:
    """Load immutable baseline metadata and reject any provenance drift."""
    config_path = baseline_directory / "config.json"
    metrics_path = baseline_directory / "conversion-metrics.json"
    index_path = baseline_directory / _SOURCE_INDEX_NAME
    expected_checksums = {
        config_path: _BASELINE_CONFIG_SHA256,
        metrics_path: _BASELINE_METRICS_SHA256,
        index_path: _BASELINE_INDEX_SHA256,
    }
    for path, expected_sha256 in expected_checksums.items():
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"baseline metadata checksum mismatch: {path.name}")
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    if metrics_payload.get("recipe_sha256") != expected_recipe_sha256:
        raise ValueError("baseline transform recipe checksum mismatch")
    if index_payload.get("weight_map") != dict(expected_weight_map):
        raise ValueError("baseline tensor index differs from expected output")

    output_sha256_by_shard: dict[str, str] = {}
    metrics_by_shard: dict[str, list[dict[str, Any]]] = {}
    for raw_shard in metrics_payload.get("shards", []):
        if not isinstance(raw_shard, dict):
            raise ValueError("baseline conversion metrics contains invalid shard")
        shard_name = raw_shard.get("shard")
        output_sha256 = raw_shard.get("output_sha256")
        metrics = raw_shard.get("metrics")
        if (
            not isinstance(shard_name, str)
            or shard_name in output_sha256_by_shard
            or not isinstance(output_sha256, str)
            or len(output_sha256) != 64
            or not isinstance(metrics, list)
        ):
            raise ValueError("baseline conversion metrics contains invalid shard")
        normalized_metrics = []
        for raw_metric in metrics:
            if not isinstance(raw_metric, dict):
                raise ValueError("baseline conversion metrics contains invalid metric")
            normalized_metric = dict(raw_metric)
            normalized_metric.setdefault("group_size", 128)
            normalized_metrics.append(normalized_metric)
        output_sha256_by_shard[shard_name] = output_sha256
        metrics_by_shard[shard_name] = normalized_metrics

    expected_shards = set(expected_weight_map.values())
    if set(output_sha256_by_shard) != expected_shards:
        raise ValueError("baseline conversion metrics shard inventory mismatch")
    return BaselineArtifactReuse(
        directory=baseline_directory,
        recipe=ArtifactRecipe(default=LayerQuantization(2, 2)),
        recipe_sha256=expected_recipe_sha256,
        expected_weight_map=dict(expected_weight_map),
        output_sha256_by_shard=output_sha256_by_shard,
        metrics_by_shard=metrics_by_shard,
    )


def load_completed_frontier_candidate(
    source_directory: Path,
    output_root: Path,
    recipe_bundle: Mapping[str, Any],
    *,
    candidate_name: str,
    imatrix_path: Path,
    device: str,
) -> FrontierCandidateConversion:
    """Reconstruct one complete local candidate from checksum-bound receipts."""
    candidates_payload = recipe_bundle.get("candidates")
    if not isinstance(candidates_payload, Mapping):
        raise ValueError("frontier recipe bundle has no candidates")
    if candidate_name not in ("cliff", "capacity", "balanced", "quality"):
        raise ValueError(f"unknown frontier candidate: {candidate_name}")
    recipe = artifact_recipe_from_payload(candidates_payload[candidate_name])
    options = TransformOptions(
        group_size=128,
        quantizer=QuantizerKind.IMATRIX_WEIGHTED,
        device=device,
        imatrix_sha256=file_sha256(imatrix_path),
    )
    recipe_sha256 = transform_recipe_sha256(recipe, options)
    source_weight_map = load_source_weight_map(source_directory / _SOURCE_INDEX_NAME)
    source_shards = tuple(sorted(set(source_weight_map.values())))
    expected_weight_map = build_expected_output_weight_map(source_weight_map)
    output_shards = select_output_shards(source_shards, expected_weight_map)
    output_directory = output_root / candidate_name
    writer = ResumableSafetensorsWriter(output_directory)
    results = []
    for shard_name in output_shards:
        source_path = source_directory / shard_name
        receipt = writer.completed_shard(
            shard_name,
            ShardIdentity(
                source_sha256=file_sha256(source_path),
                recipe_sha256=recipe_sha256,
            ),
        )
        if receipt is None:
            raise ValueError(
                "frontier resume candidate has no receipt: "
                f"{candidate_name}/{shard_name}"
            )
        results.append(
            ShardTransformResult(
                receipt=receipt,
                metrics=metrics_from_shard_receipt(receipt),
                resumed=True,
            )
        )
    writer.finalize_index(output_shards, expected_weight_map=expected_weight_map)
    _verify_conversion_metrics_report(
        output_directory / "conversion-metrics.json",
        results,
        recipe_sha256,
    )
    return FrontierCandidateConversion(
        name=candidate_name,
        output_directory=output_directory,
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        results=tuple(results),
        reused_shard_count=0,
        converted_shard_count=0,
    )


def convert_frontier_candidates(
    source_directory: Path,
    output_root: Path,
    recipe_bundle: Mapping[str, Any],
    *,
    baseline_directory: Path | None,
    imatrix_path: Path,
    device: str,
    candidate_names: tuple[str, ...],
    completed_callback: Callable[[FrontierCandidateConversion], None] | None = None,
    completed_candidate_names: tuple[str, ...] = (),
    reuse_candidate: FrontierCandidateConversion | None = None,
) -> tuple[FrontierCandidateConversion, ...]:
    """Convert selected nested candidates while hardlinking identical shards."""
    if baseline_directory is None:
        raise ValueError(
            "frontier conversion requires the immutable baseline directory"
        )
    validate_frontier_conversion_inputs(
        source_directory,
        baseline_directory,
        imatrix_path,
        recipe_bundle,
    )
    source_weight_map = load_source_weight_map(source_directory / _SOURCE_INDEX_NAME)
    source_shards = tuple(sorted(set(source_weight_map.values())))
    expected_weight_map = build_expected_output_weight_map(source_weight_map)
    output_shards = select_output_shards(source_shards, expected_weight_map)
    source_shard_layers = map_source_shards_to_routed_layers(source_weight_map)
    candidates_payload = recipe_bundle.get("candidates")
    if not isinstance(candidates_payload, Mapping):
        raise ValueError("frontier recipe bundle has no candidates")
    missing_candidates = {"cliff", "capacity", "balanced", "quality"} - set(
        candidates_payload
    )
    if missing_candidates:
        missing = sorted(missing_candidates)
        raise ValueError(f"frontier recipe bundle is missing candidates: {missing}")

    options = TransformOptions(
        group_size=128,
        quantizer=QuantizerKind.IMATRIX_WEIGHTED,
        device=device,
        imatrix_sha256=file_sha256(imatrix_path),
    )
    baseline_recipe = ArtifactRecipe(default=LayerQuantization(2, 2))
    baseline = load_baseline_artifact_reuse(
        baseline_directory,
        expected_weight_map,
        transform_recipe_sha256(baseline_recipe, options),
    )
    candidate_order = validate_frontier_candidate_names(candidate_names)
    if completed_candidate_names != candidate_order[: len(completed_candidate_names)]:
        raise ValueError("frontier completed candidates must prefix the selection")
    if reuse_candidate is not None and (
        not completed_candidate_names
        or reuse_candidate.name != completed_candidate_names[-1]
    ):
        raise ValueError("frontier reuse candidate must end the completed prefix")
    completed = (
        {reuse_candidate.name: reuse_candidate} if reuse_candidate is not None else {}
    )
    newly_completed: list[FrontierCandidateConversion] = []
    with ImatrixFile.open(imatrix_path) as imatrix:
        imatrix.validate_deepseek_v4_geometry()
        for name in candidate_order:
            if name in completed_candidate_names:
                continue
            recipe = artifact_recipe_from_payload(candidates_payload[name])
            output_directory = output_root / name
            writer = ResumableSafetensorsWriter(output_directory)
            recipe_sha256 = transform_recipe_sha256(recipe, options)
            results: list[ShardTransformResult] = []
            reused_shards = converted_shards = 0
            for shard_name in output_shards:
                source_path = source_directory / shard_name
                identity = ShardIdentity(
                    source_sha256=file_sha256(source_path),
                    recipe_sha256=recipe_sha256,
                )
                existing = writer.completed_shard(shard_name, identity)
                if existing is not None:
                    results.append(
                        ShardTransformResult(
                            receipt=existing,
                            metrics=metrics_from_shard_receipt(existing),
                            resumed=True,
                        )
                    )
                    continue

                reusable = _find_reusable_shard(
                    shard_name,
                    recipe,
                    source_shard_layers,
                    completed,
                    baseline=baseline,
                    source_path=source_path,
                )
                if reusable is not None:
                    receipt = writer.reuse_shard(shard_name, reusable, identity)
                    results.append(
                        ShardTransformResult(
                            receipt=receipt,
                            metrics=metrics_from_shard_receipt(receipt),
                            resumed=False,
                        )
                    )
                    reused_shards += 1
                    continue

                result = transform_source_shard(
                    source_path,
                    shard_name,
                    writer=writer,
                    recipe=recipe,
                    options=options,
                    imatrix=imatrix,
                )
                results.append(result)
                converted_shards += 1

            writer.finalize_index(
                output_shards,
                expected_weight_map=expected_weight_map,
            )
            materialize_converted_model_assets(
                source_directory,
                output_directory,
                recipe=recipe,
                group_size=128,
            )
            write_conversion_metrics_report(
                output_directory / "conversion-metrics.json",
                results,
                recipe_sha256,
            )
            completed[name] = FrontierCandidateConversion(
                name=name,
                output_directory=output_directory,
                recipe=recipe,
                recipe_sha256=recipe_sha256,
                results=tuple(results),
                reused_shard_count=reused_shards,
                converted_shard_count=converted_shards,
            )
            newly_completed.append(completed[name])
            if completed_callback is not None:
                completed_callback(completed[name])
    return tuple(newly_completed)


def _verify_conversion_metrics_report(
    path: Path,
    results: list[ShardTransformResult],
    recipe_sha256: str,
) -> None:
    if not path.is_file():
        raise ValueError("frontier resume candidate has no conversion metrics")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("recipe_sha256") != recipe_sha256:
        raise ValueError("frontier resume conversion recipe checksum mismatch")
    expected_shards = []
    for result in results:
        expected_shards.append(
            {
                "shard": result.receipt.shard_name,
                "output_sha256": result.receipt.output_sha256,
                "metrics": [
                    {
                        "tensor_name": metric.tensor_name,
                        "bits": metric.bits,
                        "group_size": metric.group_size,
                        "unweighted_error": metric.unweighted_error,
                        "weighted_error": metric.weighted_error,
                    }
                    for metric in result.metrics
                ],
            }
        )
    if payload.get("shards") != expected_shards:
        raise ValueError("frontier resume conversion metrics differ from receipts")


def map_source_shards_to_routed_layers(
    source_weight_map: Mapping[str, str],
) -> dict[str, frozenset[int]]:
    """Map each source shard to the routed layers whose output it owns."""
    layers: dict[str, set[int]] = {}
    for tensor_name, shard_name in source_weight_map.items():
        identity = classify_tensor(tensor_name)
        if (
            identity.disposition is TensorDisposition.QUANTIZE
            and identity.layer is not None
        ):
            layers.setdefault(shard_name, set()).add(identity.layer)
    return {shard: frozenset(value) for shard, value in layers.items()}


def artifact_recipe_from_payload(payload: Any) -> ArtifactRecipe:
    """Parse an in-memory artifact recipe with per-layer group sizes."""
    if not isinstance(payload, Mapping):
        raise ValueError("frontier candidate recipe must be an object")
    default = _layer_quantization_from_payload(payload.get("default"), "default")
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, Mapping):
        raise ValueError("frontier candidate layers must be an object")
    layers = {}
    for raw_layer, raw_quantization in raw_layers.items():
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid frontier layer number: {raw_layer!r}") from error
        if not 0 <= layer < 43:
            raise ValueError(f"frontier layer is outside [0, 42]: {layer}")
        layers[layer] = _layer_quantization_from_payload(
            raw_quantization,
            f"layers.{layer}",
        )
    if set(layers) != set(range(43)):
        raise ValueError("frontier candidate must explicitly configure all 43 layers")
    return ArtifactRecipe(default=default, layers=layers)


def _find_reusable_shard(
    shard_name: str,
    recipe: ArtifactRecipe,
    source_shard_layers: Mapping[str, frozenset[int]],
    completed: Mapping[str, FrontierCandidateConversion],
    *,
    baseline: BaselineArtifactReuse | None,
    source_path: Path,
) -> ShardReceipt | None:
    layers = source_shard_layers.get(shard_name, frozenset())
    for candidate in reversed(tuple(completed.values())):
        if _recipes_match_layers(recipe, candidate.recipe, layers):
            source_writer = ResumableSafetensorsWriter(candidate.output_directory)
            source_identity = ShardIdentity(
                source_sha256=file_sha256(source_path),
                recipe_sha256=candidate.recipe_sha256,
            )
            return source_writer.completed_shard(shard_name, source_identity)

    if baseline is not None and _recipes_match_layers(
        recipe,
        baseline.recipe,
        layers,
    ):
        return baseline.shard_receipt(shard_name, source_path)
    return None


def _recipes_match_layers(
    left: ArtifactRecipe,
    right: ArtifactRecipe,
    layers: frozenset[int],
) -> bool:
    for layer in layers:
        for projection in ("w1", "w2", "w3"):
            if left.bits_for(layer, projection) != right.bits_for(layer, projection):
                return False
            if left.group_size_for(
                layer,
                projection,
                fallback=128,
            ) != right.group_size_for(layer, projection, fallback=128):
                return False
    return True


def _layer_quantization_from_payload(value: Any, location: str) -> LayerQuantization:
    if not isinstance(value, Mapping):
        raise ValueError(f"frontier {location} must be an object")
    shared_fields = {"w13_bits", "w2_bits", "group_size"}
    projection_fields = {
        "w13_bits",
        "w2_bits",
        "w13_group_size",
        "w2_group_size",
    }
    field_names = frozenset(value)
    if field_names not in {frozenset(shared_fields), frozenset(projection_fields)}:
        raise ValueError(
            f"frontier {location} must contain bit widths and either shared "
            "or projection group sizes"
        )
    return LayerQuantization(
        w13_bits=int(value["w13_bits"]),
        w2_bits=int(value["w2_bits"]),
        group_size=(int(value["group_size"]) if "group_size" in value else None),
        w13_group_size=(
            int(value["w13_group_size"]) if "w13_group_size" in value else None
        ),
        w2_group_size=(
            int(value["w2_group_size"]) if "w2_group_size" in value else None
        ),
    )
