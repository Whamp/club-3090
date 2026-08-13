from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.convert_cli import load_source_weight_map
from deepseek_v4_lowbit.frontier_recipe import load_json_object
from deepseek_v4_lowbit.frontier_screen import (
    FrontierScreenOptions,
    expand_full_frontier_layers,
    frontier_screen_results_payload,
    screen_quantization_frontier,
)
from deepseek_v4_lowbit.imatrix import ImatrixFile
from deepseek_v4_lowbit.shard_writer import file_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Screen every expert in frontier boundary layers."
    )
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("imatrix", type=Path)
    parser.add_argument("boundary_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    source_directory = arguments.source_directory.resolve()
    source_index_path = source_directory / "model.safetensors.index.json"
    imatrix_path = arguments.imatrix.resolve()
    boundary_path = arguments.boundary_report.resolve()
    boundary = load_json_object(boundary_path)
    layers = boundary.get("layers")
    if not isinstance(layers, list) or not all(
        isinstance(layer, int) for layer in layers
    ):
        parser.error("frontier boundary report layers must be integer list")
    samples = expand_full_frontier_layers(layers)
    weight_map = load_source_weight_map(source_index_path)
    with ImatrixFile.open(imatrix_path) as imatrix:
        imatrix.validate_deepseek_v4_geometry()
        results = screen_quantization_frontier(
            source_directory,
            weight_map,
            imatrix,
            samples,
            FrontierScreenOptions(device=arguments.device),
        )
    used_shards = sorted({result.source_shard for result in results})
    report = {
        "report_schema_version": 1,
        "source_index_sha256": file_sha256(source_index_path),
        "imatrix_sha256": file_sha256(imatrix_path),
        "boundary_report_sha256": file_sha256(boundary_path),
        "device": arguments.device,
        "group_sizes": [128, 256, 512],
        "layers": layers,
        "source_shards": {
            shard: file_sha256(source_directory / shard) for shard in used_shards
        },
        "results": frontier_screen_results_payload(results),
    }
    output_path = arguments.output.resolve()
    _write_json_atomic(output_path, report)
    print(
        f"frontier full screen complete: layers={len(layers)} samples={len(samples)} "
        f"candidates={len(results)} report={output_path}"
    )
    return 0


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
