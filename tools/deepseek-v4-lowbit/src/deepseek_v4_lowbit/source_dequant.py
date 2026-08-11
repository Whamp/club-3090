from __future__ import annotations

import importlib
from typing import Any

from deepseek_v4_lowbit.artifact_plan import (
    TensorDisposition,
    classify_tensor,
)


def dequantize_routed_expert_weight(
    weight_name: str,
    packed_weight: Any,
    coarse_scale: Any,
    *,
    device: str = "cpu",
) -> Any:
    """Dequantize one official DeepSeek V4 MXFP4 routed-expert weight.

    This intentionally delegates source-format normalization and FP4 E2M1
    decoding to the pinned AutoRound model-free implementation.
    """
    torch = _import_optional("torch")
    model_free = _import_optional("auto_round.compressors.model_free")

    identity = classify_tensor(weight_name)
    if identity.disposition is not TensorDisposition.QUANTIZE:
        raise ValueError(f"not a routed expert weight: {weight_name}")
    if not isinstance(packed_weight, torch.Tensor) or packed_weight.dim() != 2:
        raise ValueError("packed routed-expert weight must be a two-dimensional tensor")
    if packed_weight.dtype not in (torch.int8, torch.uint8):
        raise ValueError(
            "official routed-expert MXFP4 weight must have int8 or uint8 dtype"
        )
    if not isinstance(coarse_scale, torch.Tensor) or coarse_scale.dim() != 2:
        raise ValueError("routed-expert E8M0 scale must be a two-dimensional tensor")
    if coarse_scale.element_size() != 1:
        raise ValueError("routed-expert E8M0 scale must use one byte per value")

    scale_name = weight_name.removesuffix(".weight") + ".scale"
    raw_tensors = {
        weight_name: packed_weight,
        scale_name: coarse_scale,
    }
    normalized, source_state = model_free._preprocess_model_type_source_tensors(
        raw_tensors,
        model_type="deepseek_v4",
        group_size=32,
    )
    layer_name = weight_name.removesuffix(".weight")
    if source_state != {layer_name: 4}:
        raise RuntimeError(
            f"AutoRound did not recognize {weight_name} as DeepSeek V4 MXFP4"
        )

    dequantized_tensors = model_free._dequant_mxfp_tensors(
        normalized,
        device=device,
        shard_name=None,
    )
    if set(dequantized_tensors) != {weight_name}:
        unexpected = sorted(dequantized_tensors)
        raise RuntimeError(
            f"unexpected tensors after dequantizing {weight_name}: {unexpected}"
        )
    dequantized = dequantized_tensors[weight_name]
    expected_shape = (packed_weight.shape[0], packed_weight.shape[1] * 2)
    if tuple(dequantized.shape) != expected_shape:
        raise RuntimeError(
            f"dequantized shape {tuple(dequantized.shape)} does not match "
            f"expected {expected_shape}"
        )
    if dequantized.dtype is not torch.bfloat16:
        raise RuntimeError(
            f"dequantized dtype must be bfloat16, got {dequantized.dtype}"
        )
    return dequantized.to("cpu")


def _import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"{module_name} is required for DeepSeek source dequantization; "
            "use the pinned AutoRound environment"
        ) from error
