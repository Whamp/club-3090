from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

_EXPECTED_QUANTIZERS = {"plain-rtn", "imatrix-weighted-rtn"}
_EXPERTS_PER_LAYER = 256
_LAYER_COUNT = 43
_PROJECTION_NAMES = ("w1", "w2", "w3")


@dataclass(frozen=True)
class PilotPairComparison:
    """Paired plain-versus-weighted result for one tensor and bit width."""

    tensor_name: str
    bits: int
    projection: str
    weighted_error_improvement_fraction: float
    weighted_duration_ratio: float
    plain_duration_seconds: float
    weighted_duration_seconds: float


@dataclass(frozen=True)
class PilotProjectionSummary:
    """Pilot evidence and projected quantize-and-pack time for one projection."""

    projection: str
    pair_count: int
    improved_count: int
    tied_count: int
    worsened_count: int
    median_weighted_error_improvement_fraction: float
    plain_mean_duration_seconds: float
    weighted_mean_duration_seconds: float
    plain_projected_quantize_pack_seconds: float
    weighted_projected_quantize_pack_seconds: float


@dataclass(frozen=True)
class PilotDecisionSummary:
    """Descriptive pilot summary that deliberately makes no quantizer decision."""

    pair_count: int
    improved_count: int
    tied_count: int
    worsened_count: int
    median_weighted_error_improvement_fraction: float
    plain_projected_quantize_pack_seconds: float
    weighted_projected_quantize_pack_seconds: float
    projection_summaries: tuple[PilotProjectionSummary, ...]
    pair_comparisons: tuple[PilotPairComparison, ...]
    estimate_scope: str = (
        "quantize-and-pack only; excludes checkpoint download, source dequantization, "
        "shard writing, finalization, and upload"
    )
    decision: str | None = None


def summarize_quantizer_pilot(
    results: Iterable[dict[str, Any]],
) -> PilotDecisionSummary:
    """Summarize paired pilot metrics without selecting the winning quantizer."""
    candidates: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for result in results:
        tensor_name = _required_string(result, "tensor_name")
        quantizer = _required_string(result, "quantizer")
        if quantizer not in _EXPECTED_QUANTIZERS:
            raise ValueError(f"Pilot summary found unexpected quantizer: {quantizer}")
        bits = _required_integer(result, "bits")
        _required_positive_float(result, "duration_seconds")
        _required_nonnegative_float(result, "weighted_error")
        key = (tensor_name, bits)
        pair = candidates.setdefault(key, {})
        if quantizer in pair:
            raise ValueError(
                f"Pilot summary found duplicate candidate: {tensor_name} W{bits} "
                f"{quantizer}"
            )
        pair[quantizer] = result

    comparisons: list[PilotPairComparison] = []
    for (tensor_name, bits), pair in sorted(candidates.items()):
        if set(pair) != _EXPECTED_QUANTIZERS:
            raise ValueError(
                f"Pilot summary found incomplete pair: {tensor_name} W{bits}"
            )
        plain = pair["plain-rtn"]
        weighted = pair["imatrix-weighted-rtn"]
        plain_error = _required_nonnegative_float(plain, "weighted_error")
        weighted_error = _required_nonnegative_float(weighted, "weighted_error")
        plain_duration = _required_positive_float(plain, "duration_seconds")
        weighted_duration = _required_positive_float(weighted, "duration_seconds")
        if plain_error == 0 and weighted_error != 0:
            raise ValueError(
                f"Pilot summary cannot calculate improvement from zero baseline: "
                f"{tensor_name} W{bits}"
            )
        improvement = (
            0.0 if plain_error == 0 else (plain_error - weighted_error) / plain_error
        )
        duration_ratio = weighted_duration / plain_duration
        comparisons.append(
            PilotPairComparison(
                tensor_name=tensor_name,
                bits=bits,
                projection=_projection_from_tensor_name(tensor_name),
                weighted_error_improvement_fraction=improvement,
                weighted_duration_ratio=duration_ratio,
                plain_duration_seconds=plain_duration,
                weighted_duration_seconds=weighted_duration,
            )
        )
    if not comparisons:
        raise ValueError("Pilot summary found no paired candidates")

    projection_summaries = tuple(
        _summarize_projection(projection, comparisons)
        for projection in _PROJECTION_NAMES
        if any(item.projection == projection for item in comparisons)
    )
    improvement_counts = _improvement_counts(comparisons)
    return PilotDecisionSummary(
        pair_count=len(comparisons),
        improved_count=improvement_counts[0],
        tied_count=improvement_counts[1],
        worsened_count=improvement_counts[2],
        median_weighted_error_improvement_fraction=median(
            item.weighted_error_improvement_fraction for item in comparisons
        ),
        plain_projected_quantize_pack_seconds=sum(
            item.plain_projected_quantize_pack_seconds for item in projection_summaries
        ),
        weighted_projected_quantize_pack_seconds=sum(
            item.weighted_projected_quantize_pack_seconds
            for item in projection_summaries
        ),
        projection_summaries=projection_summaries,
        pair_comparisons=tuple(comparisons),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize paired DeepSeek V4 quantizer pilot measurements."
    )
    parser.add_argument("pilot_report", type=Path)
    parser.add_argument("summary_report", type=Path)
    arguments = parser.parse_args(argv)
    payload = json.loads(arguments.pilot_report.read_text(encoding="utf-8"))
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        parser.error("pilot report results must be a list")
    summary = summarize_quantizer_pilot(raw_results)
    _write_json_atomic(arguments.summary_report.resolve(), asdict(summary))
    print(
        f"summarized pilot: pairs={summary.pair_count} "
        f"improved={summary.improved_count} tied={summary.tied_count} "
        f"worsened={summary.worsened_count} "
        f"plain_projected_seconds={summary.plain_projected_quantize_pack_seconds:.1f} "
        f"weighted_projected_seconds="
        f"{summary.weighted_projected_quantize_pack_seconds:.1f}"
    )
    return 0


def _summarize_projection(
    projection: str,
    comparisons: list[PilotPairComparison],
) -> PilotProjectionSummary:
    selected = [item for item in comparisons if item.projection == projection]
    counts = _improvement_counts(selected)
    full_projection_count = _LAYER_COUNT * _EXPERTS_PER_LAYER
    plain_mean = mean(item.plain_duration_seconds for item in selected)
    weighted_mean = mean(item.weighted_duration_seconds for item in selected)
    return PilotProjectionSummary(
        projection=projection,
        pair_count=len(selected),
        improved_count=counts[0],
        tied_count=counts[1],
        worsened_count=counts[2],
        median_weighted_error_improvement_fraction=median(
            item.weighted_error_improvement_fraction for item in selected
        ),
        plain_mean_duration_seconds=plain_mean,
        weighted_mean_duration_seconds=weighted_mean,
        plain_projected_quantize_pack_seconds=plain_mean * full_projection_count,
        weighted_projected_quantize_pack_seconds=weighted_mean * full_projection_count,
    )


def _improvement_counts(
    comparisons: Iterable[PilotPairComparison],
) -> tuple[int, int, int]:
    improved = tied = worsened = 0
    for comparison in comparisons:
        improvement = comparison.weighted_error_improvement_fraction
        if improvement > 0:
            improved += 1
        elif improvement == 0:
            tied += 1
        else:
            worsened += 1
    return improved, tied, worsened


def _projection_from_tensor_name(tensor_name: str) -> str:
    parts = tensor_name.split(".")
    if len(parts) < 2 or parts[-1] != "weight" or parts[-2] not in _PROJECTION_NAMES:
        raise ValueError(
            f"Pilot summary cannot identify projection from tensor: {tensor_name}"
        )
    return parts[-2]


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Pilot summary requires non-empty string field: {key}")
    return value


def _required_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Pilot summary requires integer field: {key}")
    return value


def _required_nonnegative_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Pilot summary requires numeric field: {key}")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"Pilot summary requires finite nonnegative field: {key}")
    return converted


def _required_positive_float(payload: dict[str, Any], key: str) -> float:
    converted = _required_nonnegative_float(payload, key)
    if converted == 0:
        raise ValueError(f"Pilot summary requires positive field: {key}")
    return converted


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
