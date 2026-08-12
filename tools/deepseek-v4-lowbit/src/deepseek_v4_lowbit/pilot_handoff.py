from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.pilot import PilotSample
from deepseek_v4_lowbit.pilot_cli import expand_pilot_samples
from deepseek_v4_lowbit.pilot_report_summary import (
    PilotDecisionSummary,
    summarize_quantizer_pilot,
)
from deepseek_v4_lowbit.shard_writer import file_sha256

_REPORT_SCHEMA_VERSION = 1
_EXPECTED_QUANTIZERS = ("imatrix-weighted-rtn", "plain-rtn")


def validate_quantizer_pilot_handoff(
    pilot_report_path: Path,
    summary_report_path: Path,
    source_directory: Path,
    imatrix_path: Path,
    *,
    expected_samples: tuple[PilotSample, ...],
    expected_bits: tuple[int, ...] = (2,),
    expected_group_size: int = 128,
    expected_device: str = "cuda",
) -> PilotDecisionSummary:
    """Validate that resumed full conversion uses one exact quantizer pilot.

    Raises:
        ValueError: If inputs, candidates, metrics, or report bindings differ.
    """
    report = _load_json_object(pilot_report_path, "pilot report")
    summary_report = _load_json_object(summary_report_path, "pilot summary")
    source_index_path = source_directory / "model.safetensors.index.json"
    source_index = _load_json_object(source_index_path, "source index")
    source_weight_map = _validated_source_weight_map(source_index)

    _validate_report_provenance(
        report,
        source_index_path,
        imatrix_path,
        expected_group_size=expected_group_size,
        expected_device=expected_device,
    )
    canonical_samples = _canonical_expected_samples(expected_samples)
    if _validated_report_samples(report.get("samples")) != canonical_samples:
        raise ValueError("Pilot handoff sample set mismatch")
    bit_widths = _canonical_expected_bits(expected_bits)
    expected_tensor_shards = _expected_tensor_shards(
        canonical_samples,
        source_weight_map,
    )
    _validate_source_shard_checksums(
        report.get("source_shards"),
        source_directory,
        set(expected_tensor_shards.values()),
    )
    raw_results = _validated_candidate_results(
        report.get("results"),
        expected_tensor_shards,
        bit_widths,
    )
    return _validated_bound_summary(
        pilot_report_path,
        summary_report,
        raw_results,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a quantizer pilot before resumed full conversion."
    )
    parser.add_argument("pilot_report", type=Path)
    parser.add_argument("summary_report", type=Path)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("imatrix", type=Path)
    parser.add_argument(
        "--sample",
        action="append",
        required=True,
        help="Expected sample LAYER:EXPERT; repeat for multiple experts.",
    )
    parser.add_argument("--projection", action="append", choices=["w1", "w2", "w3"])
    parser.add_argument("--bits", action="append", type=int)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    samples = expand_pilot_samples(arguments.sample, arguments.projection)
    summary = validate_quantizer_pilot_handoff(
        arguments.pilot_report.resolve(),
        arguments.summary_report.resolve(),
        arguments.source_directory.resolve(),
        arguments.imatrix.resolve(),
        expected_samples=samples,
        expected_bits=tuple(sorted(set(arguments.bits or [2]))),
        expected_group_size=arguments.group_size,
        expected_device=arguments.device,
    )
    print(
        f"validated pilot handoff: pairs={summary.pair_count} "
        f"improved={summary.improved_count} tied={summary.tied_count} "
        f"worsened={summary.worsened_count}"
    )
    return 0


def _validate_report_provenance(
    report: dict[str, Any],
    source_index_path: Path,
    imatrix_path: Path,
    *,
    expected_group_size: int,
    expected_device: str,
) -> None:
    schema_version = report.get("report_schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != _REPORT_SCHEMA_VERSION
    ):
        raise ValueError("Pilot handoff report schema version mismatch")
    _require_file_checksum(
        report,
        "source_index_sha256",
        source_index_path,
        "source index",
    )
    _require_file_checksum(report, "imatrix_sha256", imatrix_path, "imatrix")
    if report.get("group_size") != expected_group_size:
        raise ValueError("Pilot handoff group size mismatch")
    if report.get("device") != expected_device:
        raise ValueError("Pilot handoff device mismatch")


def _canonical_expected_samples(
    expected_samples: tuple[PilotSample, ...],
) -> tuple[PilotSample, ...]:
    canonical_samples = tuple(sorted(set(expected_samples)))
    if not canonical_samples or canonical_samples != expected_samples:
        raise ValueError("Pilot handoff expected samples must be sorted and unique")
    return canonical_samples


def _canonical_expected_bits(expected_bits: tuple[int, ...]) -> tuple[int, ...]:
    bit_widths = tuple(sorted(set(expected_bits)))
    if not bit_widths or bit_widths != expected_bits:
        raise ValueError("Pilot handoff expected bits must be sorted and unique")
    return bit_widths


def _expected_tensor_shards(
    samples: tuple[PilotSample, ...],
    source_weight_map: dict[str, str],
) -> dict[str, str]:
    tensor_shards: dict[str, str] = {}
    for sample in samples:
        try:
            tensor_shards[sample.tensor_name] = source_weight_map[sample.tensor_name]
        except KeyError as error:
            raise ValueError(
                "Pilot handoff tensor is absent from source index: "
                f"{sample.tensor_name}"
            ) from error
    return tensor_shards


def _validated_candidate_results(
    raw_results: Any,
    expected_tensor_shards: dict[str, str],
    bit_widths: tuple[int, ...],
) -> list[dict[str, Any]]:
    if not isinstance(raw_results, list):
        raise ValueError("Pilot handoff results must be a list")
    expected_candidates = {
        (tensor_name, bits, quantizer)
        for tensor_name in expected_tensor_shards
        for bits in bit_widths
        for quantizer in _EXPECTED_QUANTIZERS
    }
    actual_candidates: set[tuple[str, int, str]] = set()
    validated_results: list[dict[str, Any]] = []
    for result in raw_results:
        if not isinstance(result, dict):
            raise ValueError("Pilot handoff candidate must be an object")
        tensor_name = _required_string(result, "tensor_name")
        bits = _required_integer(result, "bits")
        quantizer = _required_string(result, "quantizer")
        candidate_key = (tensor_name, bits, quantizer)
        if candidate_key in actual_candidates:
            raise ValueError(f"Pilot handoff duplicate candidate: {candidate_key}")
        actual_candidates.add(candidate_key)
        expected_shard = expected_tensor_shards.get(tensor_name)
        if expected_shard is None or result.get("source_shard") != expected_shard:
            raise ValueError(
                f"Pilot handoff candidate source shard mismatch: {tensor_name}"
            )
        for metric_name in (
            "duration_seconds",
            "unweighted_error",
            "weighted_error",
        ):
            _required_finite_nonnegative(result, metric_name)
        validated_results.append(result)
    if actual_candidates != expected_candidates:
        missing = sorted(expected_candidates - actual_candidates)
        unexpected = sorted(actual_candidates - expected_candidates)
        raise ValueError(
            f"Pilot handoff candidate set mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return validated_results


def _validated_bound_summary(
    pilot_report_path: Path,
    summary_report: dict[str, Any],
    raw_results: list[dict[str, Any]],
) -> PilotDecisionSummary:
    report_checksum = summary_report.pop("pilot_report_sha256", None)
    if report_checksum != file_sha256(pilot_report_path):
        raise ValueError("Pilot handoff summary is not bound to the pilot report")
    computed_summary = summarize_quantizer_pilot(raw_results)
    normalized_computed_summary = json.loads(json.dumps(asdict(computed_summary)))
    if summary_report != normalized_computed_summary:
        raise ValueError("Pilot handoff summary does not match recomputed metrics")
    return computed_summary


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Pilot handoff cannot read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Pilot handoff {description} must be an object")
    return payload


def _validated_source_weight_map(source_index: dict[str, Any]) -> dict[str, str]:
    raw_weight_map = source_index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise ValueError("Pilot handoff source index weight_map must be an object")
    weight_map: dict[str, str] = {}
    for tensor_name, shard_name in raw_weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("Pilot handoff source index has an invalid tensor name")
        if (
            not isinstance(shard_name, str)
            or Path(shard_name).name != shard_name
            or not shard_name.endswith(".safetensors")
        ):
            raise ValueError("Pilot handoff source index has an invalid shard name")
        weight_map[tensor_name] = shard_name
    return weight_map


def _validated_report_samples(raw_samples: Any) -> tuple[PilotSample, ...]:
    if not isinstance(raw_samples, list):
        raise ValueError("Pilot handoff samples must be a list")
    samples: list[PilotSample] = []
    for sample in raw_samples:
        if not isinstance(sample, dict):
            raise ValueError("Pilot handoff sample must be an object")
        layer = _required_integer(sample, "layer")
        expert = _required_integer(sample, "expert")
        projection = _required_string(sample, "projection")
        if not 0 <= layer < 43 or not 0 <= expert < 256:
            raise ValueError("Pilot handoff sample geometry is invalid")
        if projection not in {"w1", "w2", "w3"}:
            raise ValueError("Pilot handoff sample projection is invalid")
        samples.append(PilotSample(layer, expert, projection))
    canonical_samples = tuple(sorted(set(samples)))
    if len(canonical_samples) != len(samples):
        raise ValueError("Pilot handoff samples contain duplicates")
    return canonical_samples


def _validate_source_shard_checksums(
    raw_source_shards: Any,
    source_directory: Path,
    expected_shards: set[str],
) -> None:
    if not isinstance(raw_source_shards, dict):
        raise ValueError("Pilot handoff source_shards must be an object")
    if set(raw_source_shards) != expected_shards:
        raise ValueError("Pilot handoff source shard set mismatch")
    for shard_name, expected_checksum in raw_source_shards.items():
        if not isinstance(expected_checksum, str):
            raise ValueError("Pilot handoff source shard checksum must be a string")
        shard_path = source_directory / shard_name
        if file_sha256(shard_path) != expected_checksum:
            raise ValueError(
                f"Pilot handoff source shard checksum mismatch: {shard_name}"
            )


def _require_file_checksum(
    payload: dict[str, Any],
    field_name: str,
    path: Path,
    description: str,
) -> None:
    expected_checksum = payload.get(field_name)
    if not isinstance(expected_checksum, str) or file_sha256(path) != expected_checksum:
        raise ValueError(f"Pilot handoff {description} checksum mismatch")


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Pilot handoff requires non-empty string field: {key}")
    return value


def _required_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Pilot handoff requires integer field: {key}")
    return value


def _required_finite_nonnegative(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Pilot handoff requires numeric field: {key}")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"Pilot handoff requires finite nonnegative field: {key}")
    return converted


if __name__ == "__main__":
    raise SystemExit(main())
