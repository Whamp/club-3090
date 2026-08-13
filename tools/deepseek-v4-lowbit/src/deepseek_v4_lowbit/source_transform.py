from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.artifact_plan import (
    ArtifactRecipe,
    TensorDisposition,
    TensorIdentity,
    classify_tensor,
)
from deepseek_v4_lowbit.packing import (
    pack_quantized_tensor,
    packed_checkpoint_tensors,
)
from deepseek_v4_lowbit.quantizer import quantize_symmetric
from deepseek_v4_lowbit.shard_writer import (
    ResumableSafetensorsWriter,
    ResumeConflictError,
    ShardIdentity,
    ShardReceipt,
    canonical_json_sha256,
    file_sha256,
)
from deepseek_v4_lowbit.source_dequant import dequantize_routed_expert_weight

_TRANSFORM_SCHEMA_VERSION = 1
_AUTO_ROUND_REVISION = "f17d9cd4b36982006bad21ff87127aac739072e3"
_COMPRESSED_TENSORS_VERSION = "0.17.0"


class QuantizerKind(Enum):
    PLAIN_RTN = "plain-rtn"
    IMATRIX_WEIGHTED = "imatrix-weighted-rtn"


@dataclass(frozen=True)
class TransformOptions:
    group_size: int = 128
    quantizer: QuantizerKind = QuantizerKind.PLAIN_RTN
    device: str = "cpu"
    imatrix_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if not self.device:
            raise ValueError("device must be non-empty")
        if self.quantizer is QuantizerKind.IMATRIX_WEIGHTED:
            if not self.imatrix_sha256:
                raise ValueError("weighted quantization requires an imatrix checksum")
        elif self.imatrix_sha256 is not None:
            raise ValueError("plain RTN must not include an imatrix checksum")


@dataclass(frozen=True)
class TensorTransformMetric:
    tensor_name: str
    bits: int
    group_size: int
    unweighted_error: float
    weighted_error: float


@dataclass(frozen=True)
class ShardTransformResult:
    receipt: ShardReceipt
    metrics: tuple[TensorTransformMetric, ...]
    resumed: bool


@dataclass(frozen=True)
class _TransformContext:
    recipe: ArtifactRecipe
    options: TransformOptions
    imatrix: Any | None
    torch: Any


def transform_source_shard(
    source_path: Path | str,
    output_shard_name: str,
    *,
    writer: ResumableSafetensorsWriter,
    recipe: ArtifactRecipe,
    options: TransformOptions,
    imatrix: Any | None = None,
) -> ShardTransformResult:
    """Stream one official source shard into one resumable output shard."""
    if options.quantizer is QuantizerKind.IMATRIX_WEIGHTED and imatrix is None:
        raise ValueError("weighted quantization requires an open imatrix")
    if options.quantizer is QuantizerKind.PLAIN_RTN and imatrix is not None:
        raise ValueError("plain RTN must not receive an imatrix")

    source_path = Path(source_path)
    source_snapshot = _file_snapshot(source_path)
    identity = ShardIdentity(
        source_sha256=file_sha256(source_path),
        recipe_sha256=transform_recipe_sha256(recipe, options),
    )
    completed = writer.completed_shard(output_shard_name, identity)
    if completed is not None:
        return ShardTransformResult(
            receipt=completed,
            metrics=metrics_from_shard_receipt(completed),
            resumed=True,
        )

    torch = _import_optional("torch")
    safe_open = _import_optional("safetensors").safe_open
    output_tensors: dict[str, Any] = {}
    metrics: list[TensorTransformMetric] = []
    context = _TransformContext(recipe, options, imatrix, torch)

    with safe_open(source_path, framework="pt", device="cpu") as source:
        source_names = set(source.keys())
        _validate_source_pairs(source_names)
        for tensor_name in sorted(source_names):
            identity_record = classify_tensor(tensor_name)
            if identity_record.disposition is TensorDisposition.OMIT:
                continue
            if identity_record.disposition is TensorDisposition.REPLACE_SOURCE_SCALE:
                continue
            if identity_record.disposition is TensorDisposition.PRESERVE:
                output_tensors[tensor_name] = source.get_tensor(tensor_name)
                continue

            checkpoint_tensors, metric = _transform_routed_weight(
                source,
                tensor_name,
                identity_record,
                context,
            )
            output_tensors.update(checkpoint_tensors)
            metrics.append(metric)

    if _file_snapshot(source_path) != source_snapshot:
        raise RuntimeError(f"source shard changed during transformation: {source_path}")
    receipt = writer.write_shard(
        output_shard_name,
        output_tensors,
        identity,
        metadata={"transform_metrics": [_metric_payload(metric) for metric in metrics]},
    )
    return ShardTransformResult(
        receipt=receipt,
        metrics=tuple(metrics),
        resumed=False,
    )


def _transform_routed_weight(
    source: Any,
    tensor_name: str,
    identity: TensorIdentity,
    context: _TransformContext,
) -> tuple[dict[str, Any], TensorTransformMetric]:
    assert identity.layer is not None
    assert identity.projection is not None
    scale_name = tensor_name.removesuffix(".weight") + ".scale"
    dequantized = dequantize_routed_expert_weight(
        tensor_name,
        source.get_tensor(tensor_name),
        source.get_tensor(scale_name),
        device=context.options.device,
    ).to(context.options.device)
    importance = None
    optimize_scales = context.options.quantizer is QuantizerKind.IMATRIX_WEIGHTED
    if optimize_scales:
        assert context.imatrix is not None
        values = context.imatrix.expert_vector(
            tensor_name,
            expert_count=256,
            input_columns=dequantized.shape[1],
        )
        importance = context.torch.tensor(
            values,
            dtype=context.torch.float32,
            device=dequantized.device,
        )

    bits = context.recipe.bits_for(identity.layer, identity.projection)
    group_size = context.recipe.group_size_for(
        identity.layer,
        identity.projection,
        fallback=context.options.group_size,
    )
    candidate = quantize_symmetric(
        dequantized,
        bits=bits,
        group_size=group_size,
        importance=importance,
        optimize_scales=optimize_scales,
    )
    packed = pack_quantized_tensor(
        candidate,
        bits=bits,
        group_size=group_size,
    )
    checkpoint_tensors = {
        name: tensor.to("cpu")
        for name, tensor in packed_checkpoint_tensors(tensor_name, packed).items()
    }
    metric = TensorTransformMetric(
        tensor_name=tensor_name,
        bits=bits,
        group_size=group_size,
        unweighted_error=candidate.unweighted_error,
        weighted_error=candidate.weighted_error,
    )
    return checkpoint_tensors, metric


def _metric_payload(metric: TensorTransformMetric) -> dict[str, Any]:
    if not math.isfinite(metric.unweighted_error) or not math.isfinite(
        metric.weighted_error
    ):
        raise ValueError(f"non-finite transform metric for {metric.tensor_name}")
    return {
        "tensor_name": metric.tensor_name,
        "bits": metric.bits,
        "group_size": metric.group_size,
        "unweighted_error": metric.unweighted_error,
        "weighted_error": metric.weighted_error,
    }


def metrics_from_shard_receipt(
    receipt: ShardReceipt,
) -> tuple[TensorTransformMetric, ...]:
    """Recover finite per-tensor transform metrics from a verified receipt."""
    raw_metrics = receipt.metadata.get("transform_metrics")
    if not isinstance(raw_metrics, list):
        raise ResumeConflictError(
            f"shard {receipt.shard_name} receipt has no transform metrics"
        )
    try:
        metrics = tuple(
            TensorTransformMetric(
                tensor_name=raw["tensor_name"],
                bits=int(raw["bits"]),
                group_size=_metric_group_size(receipt, raw),
                unweighted_error=float(raw["unweighted_error"]),
                weighted_error=float(raw["weighted_error"]),
            )
            for raw in raw_metrics
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ResumeConflictError(
            f"shard {receipt.shard_name} has invalid transform metrics"
        ) from error
    for metric in metrics:
        if (
            not metric.tensor_name
            or metric.bits not in {2, 3, 4, 8}
            or metric.group_size <= 0
            or not math.isfinite(metric.unweighted_error)
            or not math.isfinite(metric.weighted_error)
        ):
            raise ResumeConflictError(
                f"shard {receipt.shard_name} has invalid transform metrics"
            )
    return metrics


def _metric_group_size(receipt: ShardReceipt, raw_metric: Any) -> int:
    raw_group_size = raw_metric.get("group_size")
    if raw_group_size is not None:
        return int(raw_group_size)
    tensor_name = raw_metric["tensor_name"]
    bits = int(raw_metric["bits"])
    prefix = tensor_name.removesuffix(".weight")
    packed_record = receipt.tensors[f"{prefix}.weight_packed"]
    scale_record = receipt.tensors[f"{prefix}.weight_scale"]
    packed_shape = packed_record["shape"]
    scale_shape = scale_record["shape"]
    if len(packed_shape) != 2 or len(scale_shape) != 2 or scale_shape[1] <= 0:
        raise ValueError("cannot infer legacy metric group size")
    input_width = packed_shape[1] * (32 // bits)
    if input_width % scale_shape[1]:
        raise ValueError("cannot infer legacy metric group size")
    return input_width // scale_shape[1]


def transform_recipe_sha256(
    recipe: ArtifactRecipe,
    options: TransformOptions,
) -> str:
    layer_payload = {
        str(layer): _layer_quantization_payload(quantization)
        for layer, quantization in sorted(recipe.layers.items())
    }
    return canonical_json_sha256(
        {
            "schema_version": _TRANSFORM_SCHEMA_VERSION,
            "artifact": {
                "default": _layer_quantization_payload(recipe.default),
                "layers": layer_payload,
                "mtp": "omit",
            },
            "quantization": {
                "group_size": options.group_size,
                "quantizer": options.quantizer.value,
                "imatrix_sha256": options.imatrix_sha256,
                "device": options.device,
            },
            "toolchain": {
                "auto_round_revision": _AUTO_ROUND_REVISION,
                "compressed_tensors_version": _COMPRESSED_TENSORS_VERSION,
            },
        }
    )


def _layer_quantization_payload(quantization: Any) -> dict[str, Any]:
    payload = {
        "w13_bits": quantization.w13_bits,
        "w2_bits": quantization.w2_bits,
    }
    if quantization.group_size is not None:
        payload["group_size"] = quantization.group_size
    elif quantization.w13_group_size is not None:
        payload["w13_group_size"] = quantization.w13_group_size
        payload["w2_group_size"] = quantization.w2_group_size
    return payload


def _validate_source_pairs(source_names: set[str]) -> None:
    for tensor_name in source_names:
        identity = classify_tensor(tensor_name)
        if identity.disposition is TensorDisposition.QUANTIZE:
            scale_name = tensor_name.removesuffix(".weight") + ".scale"
            if scale_name not in source_names:
                raise ValueError(
                    f"routed expert weight has no source scale: {tensor_name}"
                )
        elif identity.disposition is TensorDisposition.REPLACE_SOURCE_SCALE:
            weight_name = tensor_name.removesuffix(".scale") + ".weight"
            if weight_name not in source_names:
                raise ValueError(
                    f"routed expert scale has no source weight: {tensor_name}"
                )


def _file_snapshot(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"{module_name} is required for source shard transformation"
        ) from error
