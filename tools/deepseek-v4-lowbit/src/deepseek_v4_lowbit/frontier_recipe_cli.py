from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from deepseek_v4_lowbit.frontier_provenance import (
    load_verified_frontier_recipe_evidence,
)
from deepseek_v4_lowbit.frontier_recipe import (
    build_frontier_recipe_bundle,
    frontier_recipe_bundle_payload,
    load_json_object,
    select_frontier_boundary_layers,
)
from deepseek_v4_lowbit.frontier_screen import baseline_metrics_from_conversion_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build checksum-bound DeepSeek V4 frontier recipes."
    )
    parser.add_argument("baseline_metrics", type=Path)
    parser.add_argument("pilot_screen_report", type=Path)
    parser.add_argument("boundary_report", type=Path)
    parser.add_argument("full_screen_report", type=Path)
    parser.add_argument("source_headers_report", type=Path)
    parser.add_argument("tensor_headers", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-parameter-count", type=int, required=True)
    arguments = parser.parse_args(argv)

    baseline_path = arguments.baseline_metrics.resolve()
    pilot_path = arguments.pilot_screen_report.resolve()
    boundary_path = arguments.boundary_report.resolve()
    full_screen_path = arguments.full_screen_report.resolve()
    source_headers_report_path = arguments.source_headers_report.resolve()
    headers_path = arguments.tensor_headers.resolve()
    evidence = load_verified_frontier_recipe_evidence(
        baseline_metrics_path=baseline_path,
        pilot_screen_path=pilot_path,
        boundary_report_path=boundary_path,
        full_screen_path=full_screen_path,
        source_headers_report_path=source_headers_report_path,
        planner_headers_path=headers_path,
    )
    baseline_payload = load_json_object(baseline_path)
    baseline_metrics = baseline_metrics_from_conversion_report(baseline_payload)
    stabilized_boundary = select_frontier_boundary_layers(
        baseline_metrics,
        evidence.merged_screen_results,
        tensor_headers_path=headers_path,
    )
    if stabilized_boundary.layers != evidence.boundary_layers:
        raise ValueError(
            "frontier full screen moved the exact recipe boundary: "
            f"screened={evidence.boundary_layers}, "
            f"required={stabilized_boundary.layers}"
        )
    bundle = build_frontier_recipe_bundle(
        baseline_metrics,
        evidence.merged_screen_results,
        tensor_headers_path=headers_path,
        baseline_metrics_sha256=evidence.baseline_metrics_sha256,
        pilot_screen_report_sha256=evidence.pilot_screen_report_sha256,
        boundary_report_sha256=evidence.boundary_report_sha256,
        screen_report_sha256=evidence.full_screen_report_sha256,
        source_headers_sha256=evidence.source_headers_sha256,
        source_headers_report_sha256=evidence.source_headers_report_sha256,
        source_index_sha256=evidence.source_index_sha256,
        source_shards_sha256=evidence.source_shards_sha256,
        source_assets_sha256=evidence.source_assets_sha256,
        imatrix_sha256=evidence.imatrix_sha256,
        model_parameter_count=arguments.model_parameter_count,
    )
    output_path = arguments.output.resolve()
    _write_json_atomic(output_path, frontier_recipe_bundle_payload(bundle))
    for summary in bundle.candidate_summaries:
        print(
            f"candidate={summary.name} gib={summary.total_gib:.6f} "
            f"bpw={summary.whole_model_bits_per_parameter:.6f} "
            f"w13_g128={len(summary.w13_group128_layers)} "
            f"w13_g256={len(summary.w13_group256_layers)} "
            f"w13_g512={len(summary.w13_group512_layers)} "
            f"w2_g128={len(summary.w2_group128_layers)} "
            f"w2_g256={len(summary.w2_group256_layers)} "
            f"w2_g512={len(summary.w2_group512_layers)} "
            f"w4_down={len(summary.w4_down_layers)}"
        )
    print(f"frontier recipe bundle={output_path}")
    return 0


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
