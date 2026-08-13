from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from deepseek_v4_lowbit.frontier_recipe import (
    load_json_object,
    select_frontier_boundary_layers,
)
from deepseek_v4_lowbit.frontier_screen import baseline_metrics_from_conversion_report
from deepseek_v4_lowbit.shard_writer import file_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select DeepSeek V4 layers requiring all-expert frontier screening."
    )
    parser.add_argument("baseline_metrics", type=Path)
    parser.add_argument("screen_report", type=Path)
    parser.add_argument("tensor_headers", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args(argv)
    baseline_path = arguments.baseline_metrics.resolve()
    screen_path = arguments.screen_report.resolve()
    headers_path = arguments.tensor_headers.resolve()
    baseline = load_json_object(baseline_path)
    screen = load_json_object(screen_path)
    results = screen.get("results")
    if not isinstance(results, list):
        parser.error("frontier screen report results must be a list")
    if screen.get("baseline_metrics_sha256") != file_sha256(baseline_path):
        parser.error("frontier screen report baseline checksum mismatch")

    selection = select_frontier_boundary_layers(
        baseline_metrics_from_conversion_report(baseline),
        results,
        tensor_headers_path=headers_path,
    )
    payload = {
        "schema_version": 1,
        "baseline_metrics_sha256": file_sha256(baseline_path),
        "screen_report_sha256": file_sha256(screen_path),
        "source_headers_sha256": file_sha256(headers_path),
        "source_index_sha256": screen.get("source_index_sha256"),
        "imatrix_sha256": screen.get("imatrix_sha256"),
        "source_shards": screen.get("source_shards"),
        "layers": list(selection.layers),
        "reasons": {
            str(layer): reasons for layer, reasons in selection.reasons.items()
        },
    }
    output_path = arguments.output.resolve()
    _write_json_atomic(output_path, payload)
    selected_layers = ",".join(str(layer) for layer in selection.layers)
    print(f"frontier boundary layers={selected_layers} report={output_path}")
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
