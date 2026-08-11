from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.artifact_plan import load_artifact_recipe
from deepseek_v4_lowbit.imatrix import ImatrixFile
from deepseek_v4_lowbit.model_config import materialize_model_config
from deepseek_v4_lowbit.shard_writer import (
    ResumableSafetensorsWriter,
    file_sha256,
)
from deepseek_v4_lowbit.source_transform import (
    QuantizerKind,
    ShardTransformResult,
    TransformOptions,
    transform_recipe_sha256,
    transform_source_shard,
)

_SOURCE_INDEX_NAME = "model.safetensors.index.json"
_OUTPUT_METRICS_NAME = "conversion-metrics.json"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)

    source_directory = arguments.source_directory.resolve()
    output_directory = arguments.output_directory.resolve()
    if source_directory == output_directory:
        parser.error("source and output directories must differ")

    recipe = load_artifact_recipe(arguments.recipe)
    source_shards = load_source_shards(source_directory / _SOURCE_INDEX_NAME)
    selected_shards = _select_shards(source_shards, arguments.shard)
    quantizer = QuantizerKind(arguments.quantizer)

    if quantizer is QuantizerKind.IMATRIX_WEIGHTED:
        if arguments.imatrix is None:
            parser.error("--imatrix is required for imatrix-weighted-rtn")
        imatrix_path = arguments.imatrix.resolve()
        imatrix_checksum = file_sha256(imatrix_path)
        imatrix_context = ImatrixFile.open(imatrix_path)
    else:
        if arguments.imatrix is not None:
            parser.error("--imatrix is valid only for imatrix-weighted-rtn")
        imatrix_checksum = None
        imatrix_context = nullcontext(None)

    options = TransformOptions(
        group_size=arguments.group_size,
        quantizer=quantizer,
        device=arguments.device,
        imatrix_sha256=imatrix_checksum,
    )
    writer = ResumableSafetensorsWriter(output_directory)
    results: list[ShardTransformResult] = []
    with imatrix_context as imatrix:
        for shard_name in selected_shards:
            result = transform_source_shard(
                source_directory / shard_name,
                shard_name,
                writer=writer,
                recipe=recipe,
                options=options,
                imatrix=imatrix,
            )
            results.append(result)
            status = "resumed" if result.resumed else "written"
            print(f"{status}: {shard_name} ({len(result.metrics)} quantized weights)")

    all_shards_selected = selected_shards == source_shards
    if all_shards_selected:
        writer.finalize_index(source_shards)
        _materialize_model_assets(
            source_directory,
            output_directory,
            recipe=recipe,
            group_size=arguments.group_size,
        )
        _write_metrics_report(
            output_directory / _OUTPUT_METRICS_NAME,
            results,
            transform_recipe_sha256(recipe, options),
        )
        print(f"finalized: {output_directory}")
    else:
        print(
            f"pilot subset complete: {len(selected_shards)}/{len(source_shards)} "
            "shards; model index and config were not finalized"
        )
    return 0


def load_source_shards(index_path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload["weight_map"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors index: {index_path}") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"safetensors index has no weight map: {index_path}")

    shard_names: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("safetensors index contains an invalid tensor name")
        if (
            not isinstance(shard_name, str)
            or Path(shard_name).name != shard_name
            or not shard_name.endswith(".safetensors")
        ):
            raise ValueError(f"invalid source shard name: {shard_name!r}")
        shard_names.add(shard_name)

    ordered = tuple(sorted(shard_names))
    missing = [name for name in ordered if not (index_path.parent / name).is_file()]
    if missing:
        raise ValueError(f"source index references missing shards: {missing}")
    return ordered


def _select_shards(
    source_shards: tuple[str, ...],
    requested_shards: list[str] | None,
) -> tuple[str, ...]:
    if not requested_shards:
        return source_shards
    if len(set(requested_shards)) != len(requested_shards):
        raise ValueError("--shard values must be unique")
    unknown = sorted(set(requested_shards) - set(source_shards))
    if unknown:
        raise ValueError(f"requested shards are not in the source index: {unknown}")
    requested = set(requested_shards)
    return tuple(name for name in source_shards if name in requested)


def _materialize_model_assets(
    source_directory: Path,
    output_directory: Path,
    *,
    recipe: Any,
    group_size: int,
) -> None:
    source_config_path = source_directory / "config.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    output_config = materialize_model_config(
        source_config,
        recipe,
        group_size=group_size,
    )

    for source_path in source_directory.iterdir():
        if not source_path.is_file():
            continue
        if source_path.name in {"config.json", _SOURCE_INDEX_NAME}:
            continue
        if source_path.name.endswith(_WEIGHT_SUFFIXES):
            continue
        _copy_file_atomic(source_path, output_directory / source_path.name)
    _write_json_atomic(output_directory / "config.json", output_config)


def _write_metrics_report(
    path: Path,
    results: Iterable[ShardTransformResult],
    recipe_sha256: str,
) -> None:
    shards = []
    for result in results:
        shards.append(
            {
                "shard": result.receipt.shard_name,
                "output_sha256": result.receipt.output_sha256,
                "metrics": [
                    {
                        "tensor_name": metric.tensor_name,
                        "bits": metric.bits,
                        "unweighted_error": metric.unweighted_error,
                        "weighted_error": metric.weighted_error,
                    }
                    for metric in result.metrics
                ],
            }
        )
    _write_json_atomic(
        path,
        {
            "recipe_sha256": recipe_sha256,
            "shards": shards,
        },
    )


def _copy_file_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.writing")
    shutil.copy2(source, temporary)
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    os.replace(temporary, destination)


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    os.replace(temporary, path)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DeepSeek V4 routed experts to resumable Humming WNA16 shards."
        )
    )
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--quantizer",
        choices=[kind.value for kind in QuantizerKind],
        default=QuantizerKind.PLAIN_RTN.value,
    )
    parser.add_argument("--imatrix", type=Path)
    parser.add_argument(
        "--shard",
        action="append",
        help="Convert only this indexed shard; repeat for a pilot subset.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
