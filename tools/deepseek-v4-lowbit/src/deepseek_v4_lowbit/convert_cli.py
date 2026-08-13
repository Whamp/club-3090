from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.artifact_plan import (
    TensorDisposition,
    classify_tensor,
    load_artifact_recipe,
)
from deepseek_v4_lowbit.imatrix import ImatrixFile
from deepseek_v4_lowbit.model_config import materialize_model_config
from deepseek_v4_lowbit.packing import packed_checkpoint_tensor_names
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
    source_weight_map = load_source_weight_map(source_directory / _SOURCE_INDEX_NAME)
    source_shards = tuple(sorted(set(source_weight_map.values())))
    selected_source_shards = _select_shards(source_shards, arguments.shard)
    expected_weight_map = build_expected_output_weight_map(source_weight_map)
    output_shards = select_output_shards(source_shards, expected_weight_map)
    selected_output_shards = select_output_shards(
        selected_source_shards,
        expected_weight_map,
    )
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
        for shard_name in selected_output_shards:
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

    omitted_only_shard_count = len(selected_source_shards) - len(selected_output_shards)
    if omitted_only_shard_count:
        print(f"skipped: {omitted_only_shard_count} MTP-only source shards")

    all_shards_selected = selected_source_shards == source_shards
    if all_shards_selected:
        writer.finalize_index(
            output_shards,
            expected_weight_map=expected_weight_map,
        )
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
            f"pilot subset complete: {len(selected_source_shards)}/"
            f"{len(source_shards)} source shards, {len(selected_output_shards)} "
            "output shards; model index and config were not finalized"
        )
    return 0


def load_source_weight_map(index_path: Path) -> dict[str, str]:
    """Load and validate the complete source tensor-to-shard mapping."""
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload["weight_map"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors index: {index_path}") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"safetensors index has no weight map: {index_path}")

    validated_weight_map: dict[str, str] = {}
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("safetensors index contains an invalid tensor name")
        if (
            not isinstance(shard_name, str)
            or Path(shard_name).name != shard_name
            or not shard_name.endswith(".safetensors")
        ):
            raise ValueError(f"invalid source shard name: {shard_name!r}")
        validated_weight_map[tensor_name] = shard_name

    shard_names = sorted(set(validated_weight_map.values()))
    missing = [name for name in shard_names if not (index_path.parent / name).is_file()]
    if missing:
        raise ValueError(f"source index references missing shards: {missing}")
    return validated_weight_map


def load_source_shards(index_path: Path) -> tuple[str, ...]:
    """Load sorted unique shard names from a validated source index."""
    return tuple(sorted(set(load_source_weight_map(index_path).values())))


def build_expected_output_weight_map(
    source_weight_map: Mapping[str, str],
) -> dict[str, str]:
    """Derive the exact MTP-free WNA16 output inventory from the source index."""
    expected_weight_map: dict[str, str] = {}
    for source_tensor_name, shard_name in source_weight_map.items():
        identity = classify_tensor(source_tensor_name)
        if identity.disposition in {
            TensorDisposition.OMIT,
            TensorDisposition.REPLACE_SOURCE_SCALE,
        }:
            continue
        if identity.disposition is TensorDisposition.QUANTIZE:
            output_tensor_names = packed_checkpoint_tensor_names(source_tensor_name)
        else:
            output_tensor_names = (source_tensor_name,)
        for output_tensor_name in output_tensor_names:
            if output_tensor_name in expected_weight_map:
                raise ValueError(
                    f"source index maps multiple tensors to output "
                    f"{output_tensor_name!r}"
                )
            expected_weight_map[output_tensor_name] = shard_name
    return expected_weight_map


def select_output_shards(
    source_shards: tuple[str, ...],
    expected_weight_map: Mapping[str, str],
) -> tuple[str, ...]:
    """Keep source shards that own at least one expected output tensor."""
    output_shard_names = set(expected_weight_map.values())
    return tuple(shard for shard in source_shards if shard in output_shard_names)


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
                        "group_size": metric.group_size,
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


# Reuse seams for frontier conversion. Keep the original private names so
# existing callers and structural checks retain their established contract.
materialize_converted_model_assets = _materialize_model_assets
write_conversion_metrics_report = _write_metrics_report


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
