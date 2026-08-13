from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.convert_cli import load_source_weight_map
from deepseek_v4_lowbit.frontier_provenance import (
    validate_frontier_full_screen_matrix,
    validate_frontier_pilot_screen_matrix,
)
from deepseek_v4_lowbit.frontier_recipe import select_frontier_boundary_layers
from deepseek_v4_lowbit.frontier_screen import (
    FrontierScreenOptions,
    baseline_metrics_from_conversion_report,
    expand_full_frontier_layers,
    screen_quantization_frontier,
    select_stratified_frontier_samples,
)
from deepseek_v4_lowbit.imatrix import ImatrixFile
from deepseek_v4_lowbit.shard_writer import file_sha256


class FrontierScreenCampaign:
    """Run and resume stratified plus exact-boundary quantization screening."""

    def __init__(
        self,
        source_directory: Path,
        imatrix_path: Path,
        baseline_metrics_path: Path,
        source_headers_report_path: Path,
        planner_headers_path: Path,
        output_directory: Path,
        *,
        samples_per_projection: int,
        device: str,
    ) -> None:
        self.source_directory = source_directory
        self.imatrix_path = imatrix_path
        self.baseline_metrics_path = baseline_metrics_path
        self.source_headers_report_path = source_headers_report_path
        self.planner_headers_path = planner_headers_path
        self.output_directory = output_directory
        self.samples_per_projection = samples_per_projection
        self.device = device
        self.source_index_path = source_directory / "model.safetensors.index.json"
        self.pilot_report_path = output_directory / "frontier-pilot-screen.json"
        self.boundary_report_path = output_directory / "frontier-boundary-report.json"
        self.full_screen_report_path = output_directory / "frontier-full-screen.json"
        self.iteration_report_path = (
            output_directory / "frontier-screen-iterations.json"
        )

    def run(self) -> tuple[Path, Path, Path]:
        """Return pilot, final boundary, and final full-screen report paths."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        weight_map = load_source_weight_map(self.source_index_path)
        baseline_payload = _load_json_object(self.baseline_metrics_path)
        baseline_metrics = baseline_metrics_from_conversion_report(baseline_payload)
        source_evidence = self._verify_source_evidence(weight_map)
        pilot = self._load_or_run_pilot(weight_map, baseline_metrics, source_evidence)
        pilot_results = _require_result_list(pilot, "frontier pilot")
        full_results_by_layer = self._load_full_screen_progress(source_evidence)
        iterations: list[dict[str, Any]] = []

        with ImatrixFile.open(self.imatrix_path) as imatrix:
            imatrix.validate_deepseek_v4_geometry()
            for iteration in range(44):
                merged_results = merge_frontier_screen_results(
                    pilot_results,
                    full_results_by_layer,
                )
                selection = select_frontier_boundary_layers(
                    baseline_metrics,
                    merged_results,
                    tensor_headers_path=self.planner_headers_path,
                )
                missing_layers = tuple(
                    layer
                    for layer in selection.layers
                    if layer not in full_results_by_layer
                )
                iterations.append(
                    {
                        "iteration": iteration,
                        "selected_layers": list(selection.layers),
                        "new_layers": list(missing_layers),
                        "reasons": {
                            str(layer): list(reasons)
                            for layer, reasons in selection.reasons.items()
                        },
                    }
                )
                self._write_iteration_report(iterations, source_evidence)
                if not missing_layers:
                    self._write_final_reports(
                        selection.layers,
                        selection.reasons,
                        full_results_by_layer,
                        source_evidence,
                    )
                    return (
                        self.pilot_report_path,
                        self.boundary_report_path,
                        self.full_screen_report_path,
                    )

                results = screen_quantization_frontier(
                    self.source_directory,
                    weight_map,
                    imatrix,
                    expand_full_frontier_layers(missing_layers),
                    FrontierScreenOptions(device=self.device),
                )
                grouped_results: dict[int, list[dict[str, Any]]] = {
                    layer: [] for layer in missing_layers
                }
                for result in results:
                    layer = _result_layer(asdict(result))
                    grouped_results[layer].append(asdict(result))
                for layer in missing_layers:
                    full_results_by_layer[layer] = grouped_results[layer]
                self._write_full_screen_progress(
                    full_results_by_layer,
                    source_evidence,
                )
        raise RuntimeError(
            "frontier boundary screen did not stabilize in 44 iterations"
        )

    def _verify_source_evidence(
        self,
        weight_map: Mapping[str, str],
    ) -> dict[str, Any]:
        report = _load_json_object(self.source_headers_report_path)
        planner_headers = _load_json_object(self.planner_headers_path)
        if report.get("headers") != planner_headers:
            raise ValueError("frontier campaign planner headers differ from report")
        if report.get("source_index_sha256") != file_sha256(self.source_index_path):
            raise ValueError("frontier campaign source index checksum mismatch")
        source_shards = report.get("source_shards")
        if not isinstance(source_shards, dict) or set(source_shards) != set(
            weight_map.values()
        ):
            raise ValueError("frontier campaign source shard inventory mismatch")
        for shard_name, expected_sha256 in sorted(source_shards.items()):
            if file_sha256(self.source_directory / shard_name) != expected_sha256:
                raise ValueError(
                    f"frontier campaign source shard checksum mismatch: {shard_name}"
                )
        source_assets = report.get("source_assets")
        if not isinstance(source_assets, dict) or "config.json" not in source_assets:
            raise ValueError("frontier campaign source asset inventory is invalid")
        for asset_name, expected_sha256 in sorted(source_assets.items()):
            if file_sha256(self.source_directory / asset_name) != expected_sha256:
                raise ValueError(
                    f"frontier campaign source asset checksum mismatch: {asset_name}"
                )
        return report

    def _load_or_run_pilot(
        self,
        weight_map: Mapping[str, str],
        baseline_metrics: tuple[dict[str, Any], ...],
        source_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_identity = {
            "source_index_sha256": file_sha256(self.source_index_path),
            "baseline_metrics_sha256": file_sha256(self.baseline_metrics_path),
            "imatrix_sha256": file_sha256(self.imatrix_path),
            "device": self.device,
            "group_sizes": [128, 256, 512],
            "samples_per_projection": self.samples_per_projection,
        }
        if self.pilot_report_path.exists():
            report = _load_json_object(self.pilot_report_path)
            _require_report_identity(report, expected_identity, "frontier pilot")
            validate_frontier_pilot_screen_matrix(report)
            return report

        samples = select_stratified_frontier_samples(
            baseline_metrics,
            samples_per_projection=self.samples_per_projection,
        )
        with ImatrixFile.open(self.imatrix_path) as imatrix:
            imatrix.validate_deepseek_v4_geometry()
            results = screen_quantization_frontier(
                self.source_directory,
                weight_map,
                imatrix,
                samples,
                FrontierScreenOptions(device=self.device),
            )
        used_shards = sorted({result.source_shard for result in results})
        report = {
            "report_schema_version": 1,
            **expected_identity,
            "source_shards": {
                shard_name: source_evidence["source_shards"][shard_name]
                for shard_name in used_shards
            },
            "samples": [asdict(sample) for sample in samples],
            "results": [asdict(result) for result in results],
        }
        _write_json_atomic(self.pilot_report_path, report)
        return report

    def _load_full_screen_progress(
        self,
        source_evidence: Mapping[str, Any],
    ) -> dict[int, list[dict[str, Any]]]:
        progress_path = self.output_directory / "frontier-full-screen-progress.json"
        if not progress_path.exists():
            return {}
        payload = _load_json_object(progress_path)
        expected_identity = self._full_screen_identity(source_evidence)
        _require_report_identity(payload, expected_identity, "frontier full screen")
        raw_layers = payload.get("results_by_layer")
        if not isinstance(raw_layers, dict):
            raise ValueError("frontier full-screen progress has invalid layers")
        results_by_layer: dict[int, list[dict[str, Any]]] = {}
        for raw_layer, raw_results in raw_layers.items():
            layer = int(raw_layer)
            if not isinstance(raw_results, list):
                raise ValueError("frontier full-screen progress has invalid results")
            layer_report = {
                "group_sizes": [128, 256, 512],
                "results": raw_results,
            }
            validate_frontier_full_screen_matrix(layer_report, (layer,))
            results_by_layer[layer] = _require_result_list(
                layer_report,
                f"frontier full-screen layer {layer}",
            )
        return results_by_layer

    def _write_full_screen_progress(
        self,
        results_by_layer: Mapping[int, list[dict[str, Any]]],
        source_evidence: Mapping[str, Any],
    ) -> None:
        progress_path = self.output_directory / "frontier-full-screen-progress.json"
        _write_json_atomic(
            progress_path,
            {
                "report_schema_version": 1,
                **self._full_screen_identity(source_evidence),
                "results_by_layer": {
                    str(layer): results
                    for layer, results in sorted(results_by_layer.items())
                },
            },
        )

    def _full_screen_identity(
        self,
        source_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "source_index_sha256": file_sha256(self.source_index_path),
            "baseline_metrics_sha256": file_sha256(self.baseline_metrics_path),
            "imatrix_sha256": file_sha256(self.imatrix_path),
            "pilot_screen_report_sha256": file_sha256(self.pilot_report_path),
            "source_headers_report_sha256": file_sha256(
                self.source_headers_report_path
            ),
            "source_headers_sha256": file_sha256(self.planner_headers_path),
            "source_shards_sha256": source_evidence["source_shards"],
            "device": self.device,
            "group_sizes": [128, 256, 512],
        }

    def _write_final_reports(
        self,
        boundary_layers: tuple[int, ...],
        reasons: Mapping[int, tuple[str, ...]],
        results_by_layer: Mapping[int, list[dict[str, Any]]],
        source_evidence: Mapping[str, Any],
    ) -> None:
        boundary_payload = {
            "schema_version": 1,
            "baseline_metrics_sha256": file_sha256(self.baseline_metrics_path),
            "screen_report_sha256": file_sha256(self.pilot_report_path),
            "source_headers_sha256": file_sha256(self.planner_headers_path),
            "source_index_sha256": file_sha256(self.source_index_path),
            "imatrix_sha256": file_sha256(self.imatrix_path),
            "source_shards": source_evidence["source_shards"],
            "layers": list(boundary_layers),
            "reasons": {
                str(layer): list(layer_reasons)
                for layer, layer_reasons in reasons.items()
            },
        }
        _write_json_atomic(self.boundary_report_path, boundary_payload)
        screened_layers = tuple(sorted(results_by_layer))
        selected_results = [
            result for layer in screened_layers for result in results_by_layer[layer]
        ]
        used_shards = sorted({result["source_shard"] for result in selected_results})
        full_payload = {
            "report_schema_version": 1,
            "source_index_sha256": file_sha256(self.source_index_path),
            "imatrix_sha256": file_sha256(self.imatrix_path),
            "boundary_report_sha256": file_sha256(self.boundary_report_path),
            "device": self.device,
            "group_sizes": [128, 256, 512],
            "layers": list(screened_layers),
            "decision_boundary_layers": list(boundary_layers),
            "source_shards": {
                shard_name: source_evidence["source_shards"][shard_name]
                for shard_name in used_shards
            },
            "results": selected_results,
        }
        _write_json_atomic(self.full_screen_report_path, full_payload)

    def _write_iteration_report(
        self,
        iterations: list[dict[str, Any]],
        source_evidence: Mapping[str, Any],
    ) -> None:
        _write_json_atomic(
            self.iteration_report_path,
            {
                "schema_version": 1,
                **self._full_screen_identity(source_evidence),
                "iterations": iterations,
            },
        )


def merge_frontier_screen_results(
    pilot_results: list[dict[str, Any]],
    full_results_by_layer: Mapping[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Replace pilot rows with all-expert rows for fully screened layers."""
    full_layers = set(full_results_by_layer)
    merged = [
        result for result in pilot_results if _result_layer(result) not in full_layers
    ]
    for layer in sorted(full_results_by_layer):
        merged.extend(full_results_by_layer[layer])
    return merged


def _result_layer(result: Mapping[str, Any]) -> int:
    tensor_name = result.get("tensor_name")
    if not isinstance(tensor_name, str):
        raise ValueError("frontier screen result has no tensor name")
    try:
        return int(tensor_name.split(".", 2)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"frontier screen result has invalid tensor name: {tensor_name}"
        ) from error


def _require_result_list(
    payload: Mapping[str, Any], label: str
) -> list[dict[str, Any]]:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list) or not all(
        isinstance(result, dict) for result in raw_results
    ):
        raise ValueError(f"{label} results are invalid")
    return raw_results


def _require_report_identity(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    mismatches = [
        key
        for key, expected_value in expected.items()
        if report.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(f"{label} identity mismatch: {mismatches}")


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"frontier JSON input must be an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    os.replace(temporary, path)
