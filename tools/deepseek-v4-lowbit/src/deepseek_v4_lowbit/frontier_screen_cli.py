from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.convert_cli import load_source_weight_map
from deepseek_v4_lowbit.frontier_screen import (
    FrontierScreenOptions,
    baseline_metrics_from_conversion_report,
    frontier_screen_results_payload,
    median_frontier_screen_duration,
    screen_quantization_frontier,
    select_stratified_frontier_samples,
)
from deepseek_v4_lowbit.imatrix import ImatrixFile
from deepseek_v4_lowbit.shard_writer import file_sha256


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    source_directory = arguments.source_directory.resolve()
    source_index_path = source_directory / "model.safetensors.index.json"
    baseline_path = arguments.baseline_metrics.resolve()
    imatrix_path = arguments.imatrix.resolve()
    weight_map = load_source_weight_map(source_index_path)
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    samples = select_stratified_frontier_samples(
        baseline_metrics_from_conversion_report(baseline_payload),
        samples_per_projection=arguments.samples_per_projection,
    )

    with ImatrixFile.open(imatrix_path) as imatrix:
        imatrix.validate_deepseek_v4_geometry()
        results = screen_quantization_frontier(
            source_directory,
            weight_map,
            imatrix,
            samples,
            FrontierScreenOptions(
                group_sizes=tuple(arguments.group_size or [128, 256, 512]),
                device=arguments.device,
            ),
        )

    used_shards = sorted({result.source_shard for result in results})
    report = {
        "report_schema_version": 1,
        "source_index_sha256": file_sha256(source_index_path),
        "baseline_metrics_sha256": file_sha256(baseline_path),
        "imatrix_sha256": file_sha256(imatrix_path),
        "source_shards": {
            shard_name: file_sha256(source_directory / shard_name)
            for shard_name in used_shards
        },
        "device": arguments.device,
        "group_sizes": sorted(set(arguments.group_size or [128, 256, 512])),
        "samples_per_projection": arguments.samples_per_projection,
        "samples": [
            {
                "layer": sample.layer,
                "expert": sample.expert,
                "projection": sample.projection,
            }
            for sample in samples
        ],
        "results": frontier_screen_results_payload(results),
    }
    _write_json_atomic(arguments.output.resolve(), report)
    print(
        f"frontier screen complete: samples={len(samples)} "
        f"candidates={len(results)} "
        f"median_candidate_seconds={median_frontier_screen_duration(results):.6f} "
        f"report={arguments.output.resolve()}"
    )
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Screen stratified DeepSeek V4 experts across W2 group-size tiers and "
            "W4 down candidates."
        )
    )
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("imatrix", type=Path)
    parser.add_argument("baseline_metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples-per-projection", type=int, default=8)
    parser.add_argument(
        "--group-size",
        type=int,
        action="append",
        default=None,
        help="Candidate group size; repeat for multiple tiers.",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def _write_json_atomic(path: Path, payload: Any) -> None:
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
