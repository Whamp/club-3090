from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from deepseek_v4_lowbit.quantizer import QuantizedTensor

_SUPPORTED_BITS = frozenset({2, 3, 4, 8})


@dataclass(frozen=True)
class PackedQuantizedTensor:
    weight_packed: Any
    weight_scale: Any
    weight_shape: Any


def pack_quantized_tensor(
    candidate: QuantizedTensor,
    *,
    bits: int,
    group_size: int,
) -> PackedQuantizedTensor:
    """Pack one WNA16 weight using compressed-tensors checkpoint layout."""
    torch = _import_optional("torch")
    packing_helpers = _import_optional(
        "compressed_tensors.compressors.pack_quantized.helpers"
    )

    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {sorted(_SUPPORTED_BITS)}, got {bits}")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if not isinstance(candidate.codes, torch.Tensor) or candidate.codes.dim() != 2:
        raise ValueError("quantized codes must be a two-dimensional torch tensor")
    if candidate.codes.dtype is not torch.int8:
        raise ValueError("quantized codes must have torch.int8 dtype")

    output_features, input_features = candidate.codes.shape
    pack_factor = 32 // bits
    if input_features % pack_factor:
        raise ValueError(
            f"input width {input_features} is not divisible by Humming pack "
            f"factor {pack_factor} for W{bits}"
        )
    if input_features % group_size:
        raise ValueError(
            f"input width {input_features} is not divisible by group size {group_size}"
        )

    expected_scale_shape = (output_features, input_features // group_size)
    if tuple(candidate.scales.shape) != expected_scale_shape:
        raise ValueError(
            f"scale shape {tuple(candidate.scales.shape)} does not match "
            f"expected {expected_scale_shape}"
        )
    if candidate.scales.dtype is not torch.float16:
        raise ValueError("weight scales must have torch.float16 dtype")

    minimum_code = -(1 << (bits - 1))
    maximum_code = (1 << (bits - 1)) - 1
    if (
        int(candidate.codes.min().item()) < minimum_code
        or int(candidate.codes.max().item()) > maximum_code
    ):
        raise ValueError(
            f"quantized codes exceed signed W{bits} range "
            f"[{minimum_code}, {maximum_code}]"
        )

    weight_packed = packing_helpers.pack_to_int32(candidate.codes, bits)
    return PackedQuantizedTensor(
        weight_packed=weight_packed.contiguous(),
        weight_scale=candidate.scales.contiguous(),
        weight_shape=torch.tensor(
            [output_features, input_features],
            dtype=torch.int64,
            device=candidate.codes.device,
        ),
    )


def packed_checkpoint_tensors(
    source_weight_name: str,
    packed: PackedQuantizedTensor,
) -> dict[str, Any]:
    """Expand one logical expert weight into compressed-tensors checkpoint keys."""
    suffix = ".weight"
    if not source_weight_name.endswith(suffix):
        raise ValueError(
            f"source weight name must end in {suffix!r}, got {source_weight_name!r}"
        )
    prefix = source_weight_name[: -len(suffix)]
    return {
        f"{prefix}.weight_packed": packed.weight_packed,
        f"{prefix}.weight_scale": packed.weight_scale,
        f"{prefix}.weight_shape": packed.weight_shape,
    }


def _import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"{module_name} is required for checkpoint packing; use the "
            "pinned vLLM conversion environment"
        ) from error
