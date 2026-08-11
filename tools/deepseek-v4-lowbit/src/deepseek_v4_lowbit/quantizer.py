from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

_SUPPORTED_BITS = frozenset({2, 3, 4, 8})


@dataclass(frozen=True)
class QuantizedTensor:
    codes: Any
    scales: Any
    dequantized: Any
    unweighted_error: float
    weighted_error: float


def quantize_symmetric(
    weight: Any,
    *,
    bits: int,
    group_size: int,
    importance: Any | None = None,
    optimize_scales: bool = False,
) -> QuantizedTensor:
    """Fit an AutoRound-compatible symmetric WNA16 candidate.

    ``importance`` contains one activation-importance value per input column.
    Supplying it always affects the reported weighted error; it affects scale
    selection only when ``optimize_scales`` is true.
    """
    torch = _import_optional("torch")
    int_quantization = _import_optional("auto_round.data_type.int")

    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {sorted(_SUPPORTED_BITS)}, got {bits}")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        raise ValueError("weight must be a two-dimensional torch tensor")

    output_features, input_features = weight.shape
    if input_features % group_size:
        raise ValueError(
            f"input width {input_features} is not divisible by group size {group_size}"
        )
    if importance is not None:
        if not isinstance(importance, torch.Tensor) or importance.dim() != 1:
            raise ValueError("importance must be a one-dimensional torch tensor")
        if importance.numel() != input_features:
            raise ValueError(
                f"importance length {importance.numel()} does not match "
                f"input width {input_features}"
            )
        if not bool(torch.isfinite(importance).all()):
            raise ValueError("importance contains non-finite values")
        if bool((importance < 0).any()):
            raise ValueError("importance values must be non-negative")
        importance = importance.to(device=weight.device, dtype=torch.float32)

    source = weight.to(dtype=torch.float32)
    if optimize_scales:
        if importance is None:
            raise ValueError("optimized scale search requires importance")
        _, raw_scales, maxq = int_quantization.quant_tensor_opt_rtn_sym(
            source,
            bits=bits,
            group_size=group_size,
            imatrix=importance,
        )
    else:
        _, raw_scales, maxq = int_quantization.quant_tensor_rtn_sym(
            source,
            bits=bits,
            group_size=group_size,
        )

    group_count = input_features // group_size
    scales = raw_scales.reshape(output_features, group_count).to(torch.float16)
    grouped_source = source.reshape(output_features, group_count, group_size)
    codes = (
        (grouped_source / raw_scales.reshape(output_features, group_count, 1))
        .round()
        .clamp(-maxq, maxq - 1)
        .to(torch.int8)
        .reshape_as(source)
    )
    dequantized = (
        codes.reshape(output_features, group_count, group_size).to(torch.float32)
        * scales.to(torch.float32).unsqueeze(-1)
    ).reshape_as(source)

    squared_error = (dequantized - source).square()
    unweighted_error = float(squared_error.mean().item())
    weighted_error = _weighted_mean_error(squared_error, importance, torch)
    return QuantizedTensor(
        codes=codes,
        scales=scales,
        dequantized=dequantized,
        unweighted_error=unweighted_error,
        weighted_error=weighted_error,
    )


def _weighted_mean_error(
    squared_error: Any,
    importance: Any | None,
    torch: Any,
) -> float:
    if importance is None:
        return float(squared_error.mean().item())
    total_importance = importance.sum()
    if float(total_importance.item()) == 0.0:
        return float(squared_error.mean().item())
    weighted_sum = (squared_error * importance.reshape(1, -1)).sum()
    denominator = total_importance * squared_error.shape[0]
    return float((weighted_sum / denominator).item())


def _import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"{module_name} is required for quantization; use the pinned "
            "AutoRound environment"
        ) from error
