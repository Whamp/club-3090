from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.imatrix import ImatrixFile
from deepseek_v4_lowbit.pilot import (
    PilotOptions,
    PilotSample,
    compare_quantizers,
    pilot_results_payload,
)
from deepseek_v4_lowbit.shard_writer import file_sha256


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    source_directory = arguments.source_directory.resolve()
    imatrix_path = arguments.imatrix.resolve()
    source_index_path = source_directory / "model.safetensors.index.json"
    weight_map = _load_weight_map(source_index_path)
    samples = expand_pilot_samples(arguments.sample, arguments.projection)

    with ImatrixFile.open(imatrix_path) as imatrix:
        imatrix.validate_deepseek_v4_geometry()
        results = compare_quantizers(
            source_directory,
            weight_map,
            imatrix,
            samples,
            PilotOptions(
                bits=tuple(arguments.bits or [2]),
                group_size=arguments.group_size,
                device=arguments.device,
            ),
        )

    used_shards = sorted({result.source_shard for result in results})
    report = {
        "report_schema_version": 1,
        "source_index_sha256": file_sha256(source_index_path),
        "imatrix_sha256": file_sha256(imatrix_path),
        "source_shards": {
            shard_name: file_sha256(source_directory / shard_name)
            for shard_name in used_shards
        },
        "device": arguments.device,
        "group_size": arguments.group_size,
        "samples": [
            {
                "layer": sample.layer,
                "expert": sample.expert,
                "projection": sample.projection,
            }
            for sample in samples
        ],
        "results": pilot_results_payload(results),
    }
    _write_json_atomic(arguments.output.resolve(), report)
    print(
        f"pilot complete: {len(samples)} weights, {len(results)} candidates, "
        f"report={arguments.output.resolve()}"
    )
    return 0


def expand_pilot_samples(
    raw_samples: list[str],
    projections: list[str] | None,
) -> tuple[PilotSample, ...]:
    """Expand LAYER:EXPERT pilot samples into sorted projection samples."""
    projection_names = projections or ["w1", "w2", "w3"]
    samples: set[PilotSample] = set()
    for raw_sample in raw_samples:
        parts = raw_sample.split(":")
        if len(parts) != 2:
            raise ValueError(f"pilot sample must be LAYER:EXPERT, got {raw_sample!r}")
        try:
            layer, expert = (int(part) for part in parts)
        except ValueError as error:
            raise ValueError(
                f"pilot sample must contain integers, got {raw_sample!r}"
            ) from error
        if not 0 <= layer < 43:
            raise ValueError(f"pilot layer is outside [0, 42]: {layer}")
        if not 0 <= expert < 256:
            raise ValueError(f"pilot expert is outside [0, 255]: {expert}")
        samples.update(
            PilotSample(layer, expert, projection) for projection in projection_names
        )
    return tuple(sorted(samples))


def _load_weight_map(index_path: Path) -> dict[str, str]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        raw_weight_map = payload["weight_map"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors index: {index_path}") from error
    if not isinstance(raw_weight_map, dict):
        raise ValueError("safetensors weight_map must be an object")
    weight_map: dict[str, str] = {}
    for tensor_name, shard_name in raw_weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("invalid tensor name in source index")
        if (
            not isinstance(shard_name, str)
            or Path(shard_name).name != shard_name
            or not shard_name.endswith(".safetensors")
        ):
            raise ValueError(f"invalid shard name in source index: {shard_name!r}")
        weight_map[tensor_name] = shard_name
    return weight_map


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    os.replace(temporary, path)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare plain and imatrix-weighted RTN on selected experts."
    )
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("imatrix", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--sample",
        action="append",
        required=True,
        help="Sample LAYER:EXPERT; repeat for multiple experts.",
    )
    parser.add_argument(
        "--projection",
        action="append",
        choices=["w1", "w2", "w3"],
    )
    parser.add_argument("--bits", action="append", type=int)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
