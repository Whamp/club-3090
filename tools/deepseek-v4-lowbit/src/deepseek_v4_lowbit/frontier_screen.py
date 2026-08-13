from __future__ import annotations

import importlib
import math
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from deepseek_v4_lowbit.packing import pack_quantized_tensor
from deepseek_v4_lowbit.quantizer import quantize_symmetric
from deepseek_v4_lowbit.source_dequant import dequantize_routed_expert_weight

_ROUTED_WEIGHT_NAME = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>w[123])\.weight$"
)


@dataclass(frozen=True, order=True)
class FrontierScreenSample:
    """One routed-expert matrix selected for quantization-frontier screening."""

    layer: int
    expert: int
    projection: str

    @property
    def tensor_name(self) -> str:
        return f"layers.{self.layer}.ffn.experts.{self.expert}.{self.projection}.weight"


@dataclass(frozen=True)
class FrontierScreenResult:
    """Measured error and cost for one screened tensor quantization schema."""

    tensor_name: str
    source_shard: str
    bits: int
    group_size: int
    duration_seconds: float
    unweighted_error: float
    weighted_error: float
    normalized_weighted_error: float
    block_output_relative_error: float
    selection_error: float
    packed_bytes: int
    scale_bytes: int


@dataclass(frozen=True)
class FrontierScreenOptions:
    """Candidate schemas and device for the quantization-frontier screen."""

    group_sizes: tuple[int, ...] = (128, 256, 512)
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.group_sizes or any(size <= 0 for size in self.group_sizes):
            raise ValueError("frontier screen group sizes must be positive")
        if not self.device:
            raise ValueError("frontier screen device must be non-empty")


def select_stratified_frontier_samples(
    baseline_metrics: Iterable[Mapping[str, Any]],
    *,
    samples_per_projection: int,
) -> tuple[FrontierScreenSample, ...]:
    """Select per-layer/projection experts across baseline-error quantiles."""
    if samples_per_projection < 2:
        raise ValueError("frontier screen requires at least two samples per projection")

    metrics_by_layer_projection: dict[tuple[int, str], list[tuple[float, int]]] = {}
    for metric in baseline_metrics:
        tensor_name = metric.get("tensor_name")
        weighted_error = metric.get("weighted_error")
        if not isinstance(tensor_name, str):
            raise ValueError("frontier baseline metric has no tensor_name")
        match = _ROUTED_WEIGHT_NAME.fullmatch(tensor_name)
        if match is None:
            raise ValueError(
                f"frontier baseline metric is not a routed weight: {tensor_name}"
            )
        if (
            not isinstance(weighted_error, (int, float))
            or isinstance(weighted_error, bool)
            or not math.isfinite(float(weighted_error))
            or float(weighted_error) < 0
        ):
            raise ValueError(
                f"frontier baseline metric has invalid weighted_error: {tensor_name}"
            )
        key = (int(match["layer"]), match["projection"])
        metrics_by_layer_projection.setdefault(key, []).append(
            (float(weighted_error), int(match["expert"]))
        )

    expected_keys = {
        (layer, projection) for layer in range(43) for projection in ("w1", "w2", "w3")
    }
    if set(metrics_by_layer_projection) != expected_keys:
        missing = sorted(expected_keys - set(metrics_by_layer_projection))
        extra = sorted(set(metrics_by_layer_projection) - expected_keys)
        raise ValueError(
            f"frontier baseline layer/projection mismatch: missing={missing}, "
            f"extra={extra}"
        )

    samples: set[FrontierScreenSample] = set()
    for (layer, projection), values in sorted(metrics_by_layer_projection.items()):
        ordered = sorted(values)
        experts = {expert for _, expert in ordered}
        if len(experts) != 256 or len(ordered) != 256:
            raise ValueError(
                f"frontier baseline requires 256 unique experts for layer {layer} "
                f"projection {projection}"
            )
        for sample_index in range(samples_per_projection):
            quantile_index = round(
                sample_index * (len(ordered) - 1) / (samples_per_projection - 1)
            )
            samples.add(
                FrontierScreenSample(
                    layer=layer,
                    expert=ordered[quantile_index][1],
                    projection=projection,
                )
            )
    return tuple(sorted(samples))


def expand_full_frontier_layers(
    layers: Iterable[int],
) -> tuple[FrontierScreenSample, ...]:
    """Expand selected layers to all 256 experts and three projections."""
    samples = set()
    for layer in layers:
        if not 0 <= layer < 43:
            raise ValueError(f"frontier full-screen layer is outside [0, 42]: {layer}")
        for expert in range(256):
            for projection in ("w1", "w2", "w3"):
                samples.add(FrontierScreenSample(layer, expert, projection))
    return tuple(sorted(samples))


def screen_quantization_frontier(
    source_directory: Path,
    weight_map: Mapping[str, str],
    imatrix: Any,
    samples: Iterable[FrontierScreenSample],
    options: FrontierScreenOptions,
) -> tuple[FrontierScreenResult, ...]:
    """Measure imatrix-weighted W2 tiers and W4 down candidates."""
    torch = _import_optional("torch")
    safe_open = _import_optional("safetensors").safe_open
    samples_by_shard: dict[str, list[FrontierScreenSample]] = {}
    for sample in sorted(set(samples)):
        try:
            shard_name = weight_map[sample.tensor_name]
        except KeyError as error:
            raise ValueError(
                f"frontier sample is absent from source index: {sample.tensor_name}"
            ) from error
        samples_by_shard.setdefault(shard_name, []).append(sample)

    results: list[FrontierScreenResult] = []
    group_sizes = tuple(sorted(set(options.group_sizes)))
    for shard_name, shard_samples in sorted(samples_by_shard.items()):
        shard_path = source_directory / shard_name
        if not shard_path.is_file():
            raise ValueError(f"frontier source shard is missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as source:
            source_names = set(source.keys())
            for sample in shard_samples:
                scale_name = sample.tensor_name.removesuffix(".weight") + ".scale"
                if scale_name not in source_names:
                    raise ValueError(
                        f"frontier sample has no source scale: {sample.tensor_name}"
                    )
                dequantized = dequantize_routed_expert_weight(
                    sample.tensor_name,
                    source.get_tensor(sample.tensor_name),
                    source.get_tensor(scale_name),
                    device=options.device,
                ).to(options.device)
                importance = torch.tensor(
                    imatrix.expert_vector(
                        sample.tensor_name,
                        expert_count=256,
                        input_columns=dequantized.shape[1],
                    ),
                    dtype=torch.float32,
                    device=dequantized.device,
                )
                bit_widths = (2, 4) if sample.projection == "w2" else (2,)
                for group_size in group_sizes:
                    for bits in bit_widths:
                        _synchronize(torch, options.device)
                        started = time.perf_counter()
                        candidate = quantize_symmetric(
                            dequantized,
                            bits=bits,
                            group_size=group_size,
                            importance=importance,
                            optimize_scales=True,
                        )
                        packed = pack_quantized_tensor(
                            candidate,
                            bits=bits,
                            group_size=group_size,
                        )
                        _synchronize(torch, options.device)
                        normalized_error, block_output_error = (
                            _normalized_frontier_errors(
                                dequantized,
                                candidate.dequantized,
                                importance,
                                torch,
                            )
                        )
                        results.append(
                            FrontierScreenResult(
                                tensor_name=sample.tensor_name,
                                source_shard=shard_name,
                                bits=bits,
                                group_size=group_size,
                                duration_seconds=time.perf_counter() - started,
                                unweighted_error=candidate.unweighted_error,
                                weighted_error=candidate.weighted_error,
                                normalized_weighted_error=normalized_error,
                                block_output_relative_error=block_output_error,
                                selection_error=max(
                                    normalized_error,
                                    block_output_error,
                                ),
                                packed_bytes=packed.weight_packed.numel()
                                * packed.weight_packed.element_size(),
                                scale_bytes=packed.weight_scale.numel()
                                * packed.weight_scale.element_size(),
                            )
                        )
    return tuple(results)


def _normalized_frontier_errors(
    source_weight: Any,
    reconstructed_weight: Any,
    importance: Any,
    torch: Any,
) -> tuple[float, float]:
    """Return global and per-output relative error under diagonal activations."""
    source = source_weight.to(torch.float32)
    reconstruction = reconstructed_weight.to(torch.float32)
    weighted_squared_error = (reconstruction - source).square() * importance
    weighted_source_energy = source.square() * importance
    total_source_energy = weighted_source_energy.sum()
    if float(total_source_energy.item()) <= 0.0:
        raise ValueError("frontier source has no weighted output energy")
    normalized_weighted_error = float(
        (weighted_squared_error.sum() / total_source_energy).item()
    )
    output_source_energy = weighted_source_energy.sum(dim=1)
    output_error_energy = weighted_squared_error.sum(dim=1)
    positive_energy = output_source_energy > 0
    if not bool(positive_energy.any()):
        raise ValueError("frontier source has no positive output-row energy")
    block_output_relative_error = float(
        (output_error_energy[positive_energy] / output_source_energy[positive_energy])
        .mean()
        .item()
    )
    if not math.isfinite(normalized_weighted_error) or not math.isfinite(
        block_output_relative_error
    ):
        raise ValueError("frontier normalized output error is non-finite")
    return normalized_weighted_error, block_output_relative_error


def frontier_screen_results_payload(
    results: Iterable[FrontierScreenResult],
) -> list[dict[str, Any]]:
    """Serialize frontier-screen results in deterministic input order."""
    return [asdict(result) for result in results]


def baseline_metrics_from_conversion_report(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Flatten and validate all baseline conversion metrics."""
    shards = payload.get("shards")
    if not isinstance(shards, list):
        raise ValueError("frontier baseline conversion report has no shards")
    metrics: list[dict[str, Any]] = []
    for shard in shards:
        if not isinstance(shard, Mapping) or not isinstance(shard.get("metrics"), list):
            raise ValueError("frontier baseline conversion report has invalid shard")
        for metric in shard["metrics"]:
            if not isinstance(metric, dict):
                raise ValueError("frontier baseline conversion metric is not an object")
            metrics.append(metric)
    if len(metrics) != 43 * 256 * 3:
        raise ValueError(
            f"frontier baseline requires 33024 metrics, found {len(metrics)}"
        )
    return tuple(metrics)


def median_frontier_screen_duration(
    results: Iterable[FrontierScreenResult],
) -> float:
    """Return the median candidate fit duration for rental projection."""
    durations = [result.duration_seconds for result in results]
    if not durations:
        raise ValueError("frontier screen contains no results")
    return median(durations)


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"{module_name} is required for quantization-frontier screening"
        ) from error
