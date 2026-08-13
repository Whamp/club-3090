from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

_ROUTED_EXPERT_NAME = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>w[123])\.(?P<kind>weight|scale)$"
)
_SUPPORTED_BITS = frozenset({2, 3, 4, 8})
_WNA16_WEIGHT_SHAPE_BYTES = 2 * 8


class TensorDisposition(Enum):
    PRESERVE = "preserve"
    QUANTIZE = "quantize"
    REPLACE_SOURCE_SCALE = "replace-source-scale"
    OMIT = "omit"


@dataclass(frozen=True)
class TensorIdentity:
    disposition: TensorDisposition
    layer: int | None = None
    expert: int | None = None
    projection: str | None = None


@dataclass(frozen=True)
class TensorHeader:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    byte_count: int


@dataclass(frozen=True)
class LayerQuantization:
    """Routed-expert bit widths and shared or projection scale group sizes."""

    w13_bits: int
    w2_bits: int
    group_size: int | None = None
    w13_group_size: int | None = None
    w2_group_size: int | None = None

    def __post_init__(self) -> None:
        for projection, bits in (("w13", self.w13_bits), ("w2", self.w2_bits)):
            if bits not in _SUPPORTED_BITS:
                supported = sorted(_SUPPORTED_BITS)
                raise ValueError(
                    f"{projection}_bits must be one of {supported}, got {bits}"
                )
        group_sizes = {
            "group_size": self.group_size,
            "w13_group_size": self.w13_group_size,
            "w2_group_size": self.w2_group_size,
        }
        for field_name, value in group_sizes.items():
            if value is not None and value <= 0:
                raise ValueError(f"layer {field_name} must be positive, got {value}")
        has_w13_group_size = self.w13_group_size is not None
        has_w2_group_size = self.w2_group_size is not None
        if has_w13_group_size != has_w2_group_size:
            raise ValueError(
                "projection group sizes require both w13_group_size and w2_group_size"
            )
        if self.group_size is not None and has_w13_group_size:
            raise ValueError(
                "shared group_size cannot be combined with projection group sizes"
            )

    def bits_for(self, projection: str) -> int:
        return self.w2_bits if projection == "w2" else self.w13_bits

    def group_size_for(self, projection: str, *, fallback: int) -> int:
        """Resolve the scale group size for one routed-expert projection."""
        if fallback <= 0:
            raise ValueError(f"fallback group size must be positive, got {fallback}")
        if projection == "w2":
            projection_group_size = self.w2_group_size
        elif projection in {"w1", "w3", "w13"}:
            projection_group_size = self.w13_group_size
        else:
            raise ValueError(f"unsupported routed-expert projection: {projection}")
        return projection_group_size or self.group_size or fallback


@dataclass(frozen=True)
class ArtifactRecipe:
    """Resolve layer quantization while preserving a caller-supplied fallback."""

    default: LayerQuantization
    layers: Mapping[int, LayerQuantization] = field(default_factory=dict)

    def layer_quantization(self, layer: int) -> LayerQuantization:
        return self.layers.get(layer, self.default)

    def bits_for(self, layer: int, projection: str) -> int:
        return self.layer_quantization(layer).bits_for(projection)

    def group_size_for(self, layer: int, projection: str, *, fallback: int) -> int:
        return self.layer_quantization(layer).group_size_for(
            projection,
            fallback=fallback,
        )


@dataclass(frozen=True)
class ArtifactPlan:
    total_bytes: int
    preserved_bytes: int
    quantized_weight_bytes: int
    quantized_scale_bytes: int
    quantized_shape_bytes: int
    replaced_source_bytes: int
    omitted_bytes: int
    preserved_tensor_count: int
    quantized_tensor_count: int
    omitted_tensor_count: int


@dataclass
class _PlanAccumulator:
    preserved_bytes: int = 0
    quantized_weight_bytes: int = 0
    quantized_scale_bytes: int = 0
    quantized_shape_bytes: int = 0
    replaced_source_bytes: int = 0
    omitted_bytes: int = 0
    preserved_tensor_count: int = 0
    quantized_tensor_count: int = 0
    omitted_tensor_count: int = 0

    def finish(self) -> ArtifactPlan:
        total_bytes = (
            self.preserved_bytes
            + self.quantized_weight_bytes
            + self.quantized_scale_bytes
            + self.quantized_shape_bytes
        )
        return ArtifactPlan(total_bytes=total_bytes, **vars(self))


def classify_tensor(name: str) -> TensorIdentity:
    if name.startswith("mtp."):
        return TensorIdentity(TensorDisposition.OMIT)

    match = _ROUTED_EXPERT_NAME.fullmatch(name)
    if match is None:
        return TensorIdentity(TensorDisposition.PRESERVE)

    disposition = (
        TensorDisposition.QUANTIZE
        if match["kind"] == "weight"
        else TensorDisposition.REPLACE_SOURCE_SCALE
    )
    return TensorIdentity(
        disposition=disposition,
        layer=int(match["layer"]),
        expert=int(match["expert"]),
        projection=match["projection"],
    )


def load_artifact_recipe(path: Path) -> ArtifactRecipe:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("recipe must be a JSON object")
    default = _load_layer_quantization(raw.get("default"), "default")
    raw_layers = raw.get("layers", {})
    if not isinstance(raw_layers, dict):
        raise ValueError("recipe layers must be a JSON object")

    layers: dict[int, LayerQuantization] = {}
    for layer, value in raw_layers.items():
        try:
            layer_number = int(layer)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid layer number: {layer!r}") from error
        if layer_number < 0:
            raise ValueError(f"layer number must be non-negative: {layer_number}")
        layers[layer_number] = _load_layer_quantization(value, f"layers.{layer}")
    return ArtifactRecipe(default=default, layers=layers)


def load_tensor_headers(path: Path) -> tuple[TensorHeader, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("tensor header file must contain a shard-to-tensors object")

    headers: list[TensorHeader] = []
    seen_names: set[str] = set()
    for shard, shard_tensors in raw.items():
        if not isinstance(shard, str) or not isinstance(shard_tensors, dict):
            raise ValueError(
                "each tensor-header entry must map a shard name to tensors"
            )
        for name, metadata in shard_tensors.items():
            if name in seen_names:
                raise ValueError(f"duplicate tensor header: {name}")
            seen_names.add(name)
            headers.append(_parse_header(shard, name, metadata))
    return tuple(headers)


def plan_artifact(
    headers: tuple[TensorHeader, ...],
    recipe: ArtifactRecipe,
    *,
    group_size: int,
) -> ArtifactPlan:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    source_scales = {
        header.name.removesuffix(".scale")
        for header in headers
        if classify_tensor(header.name).disposition
        is TensorDisposition.REPLACE_SOURCE_SCALE
    }
    accumulator = _PlanAccumulator()

    for header in headers:
        identity = classify_tensor(header.name)
        if identity.disposition is TensorDisposition.OMIT:
            accumulator.omitted_bytes += header.byte_count
            accumulator.omitted_tensor_count += 1
        elif identity.disposition is TensorDisposition.PRESERVE:
            accumulator.preserved_bytes += header.byte_count
            accumulator.preserved_tensor_count += 1
        elif identity.disposition is TensorDisposition.REPLACE_SOURCE_SCALE:
            accumulator.replaced_source_bytes += header.byte_count
        else:
            assert identity.layer is not None
            assert identity.projection is not None
            source_scale_stem = header.name.removesuffix(".weight")
            if source_scale_stem not in source_scales:
                raise ValueError(
                    f"routed expert weight has no source scale: {header.name}"
                )
            weight_bytes, scale_bytes, shape_bytes = _planned_wna16_bytes(
                header,
                bits=recipe.bits_for(identity.layer, identity.projection),
                group_size=recipe.group_size_for(
                    identity.layer,
                    identity.projection,
                    fallback=group_size,
                ),
            )
            accumulator.quantized_weight_bytes += weight_bytes
            accumulator.quantized_scale_bytes += scale_bytes
            accumulator.quantized_shape_bytes += shape_bytes
            accumulator.quantized_tensor_count += 1

    return accumulator.finish()


def _load_layer_quantization(value: Any, location: str) -> LayerQuantization:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    required_fields = {"w13_bits", "w2_bits"}
    group_size_fields = {"group_size", "w13_group_size", "w2_group_size"}
    allowed_fields = required_fields | group_size_fields
    if not required_fields <= set(value) or not set(value) <= allowed_fields:
        raise ValueError(
            f"{location} must contain w13_bits and w2_bits, with optional "
            "shared or projection group sizes"
        )
    w13_bits = value["w13_bits"]
    w2_bits = value["w2_bits"]
    group_size = value.get("group_size")
    w13_group_size = value.get("w13_group_size")
    w2_group_size = value.get("w2_group_size")
    if not isinstance(w13_bits, int) or not isinstance(w2_bits, int):
        raise ValueError(f"{location} bit widths must be integers")
    for field_name, field_value in (
        ("group_size", group_size),
        ("w13_group_size", w13_group_size),
        ("w2_group_size", w2_group_size),
    ):
        if field_value is not None and not isinstance(field_value, int):
            raise ValueError(f"{location} {field_name} must be an integer")
    return LayerQuantization(
        w13_bits=w13_bits,
        w2_bits=w2_bits,
        group_size=group_size,
        w13_group_size=w13_group_size,
        w2_group_size=w2_group_size,
    )


def _parse_header(shard: str, name: str, metadata: Any) -> TensorHeader:
    if not isinstance(name, str) or not isinstance(metadata, dict):
        raise ValueError(f"invalid tensor metadata in {shard}")
    shape = metadata.get("shape")
    offsets = metadata.get("data_offsets")
    dtype = metadata.get("dtype")
    if (
        not isinstance(dtype, str)
        or not isinstance(shape, list)
        or not shape
        or not all(isinstance(size, int) and size > 0 for size in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(offset, int) for offset in offsets)
        or offsets[1] < offsets[0]
    ):
        raise ValueError(f"invalid tensor header for {name} in {shard}")
    return TensorHeader(name, shard, dtype, tuple(shape), offsets[1] - offsets[0])


def _planned_wna16_bytes(
    header: TensorHeader,
    *,
    bits: int,
    group_size: int,
) -> tuple[int, int, int]:
    if header.dtype != "I8" or len(header.shape) != 2:
        raise ValueError(
            f"expected packed I8 matrix for routed expert weight {header.name}, "
            f"got {header.dtype} {header.shape}"
        )

    output_features, packed_source_input = header.shape
    input_features = packed_source_input * 2
    pack_factor = 32 // bits
    if input_features % pack_factor:
        raise ValueError(
            f"{header.name} input dimension {input_features} is not divisible by "
            f"Humming pack factor {pack_factor} for W{bits}"
        )
    if input_features % group_size:
        raise ValueError(
            f"{header.name} input dimension {input_features} is not divisible by "
            f"group size {group_size}"
        )

    weight_bytes = output_features * (input_features // pack_factor) * 4
    scale_bytes = output_features * (input_features // group_size) * 2
    return weight_bytes, scale_bytes, _WNA16_WEIGHT_SHAPE_BYTES
