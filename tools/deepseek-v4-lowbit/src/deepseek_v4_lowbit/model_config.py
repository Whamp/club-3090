from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from deepseek_v4_lowbit.artifact_plan import ArtifactRecipe

_COMPRESSED_TENSORS_VERSION = "0.17.0"
_PROJECTION_NAMES = {
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}


def build_compressed_tensors_config(
    recipe: ArtifactRecipe,
    *,
    layer_count: int,
    group_size: int,
) -> dict[str, Any]:
    if layer_count <= 0:
        raise ValueError(f"layer_count must be positive, got {layer_count}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    targets_by_bits: dict[int, list[str]] = {}
    for layer in range(layer_count):
        for projection, runtime_projection in _PROJECTION_NAMES.items():
            bits = recipe.bits_for(layer, projection)
            targets_by_bits.setdefault(bits, []).append(
                f"model.layers.{layer}.ffn.experts.0.{runtime_projection}"
            )

    config_groups = {
        f"group_w{bits}": {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": sorted(targets),
            "weights": _weight_quantization_args(bits, group_size),
        }
        for bits, targets in sorted(targets_by_bits.items())
    }
    return {
        "config_groups": config_groups,
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": [],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
        "version": _COMPRESSED_TENSORS_VERSION,
    }


def materialize_model_config(
    source_config: Mapping[str, Any],
    recipe: ArtifactRecipe,
    *,
    group_size: int,
) -> dict[str, Any]:
    if source_config.get("model_type") != "deepseek_v4":
        raise ValueError("source config must describe a deepseek_v4 model")
    layer_count = source_config.get("num_hidden_layers")
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError("source config has no valid num_hidden_layers")

    output = copy.deepcopy(dict(source_config))
    source_quantization = source_config.get("quantization_config")
    output["quantization_config"] = build_compressed_tensors_config(
        recipe,
        layer_count=layer_count,
        group_size=group_size,
    )
    output["num_nextn_predict_layers"] = 0
    output["club_3090_lowbit"] = {
        "format": "symmetric-group-wna16",
        "group_size": group_size,
        "mtp": "omitted",
        "recipe": _recipe_payload(recipe),
        "source_quantization_method": (
            source_quantization.get("quant_method")
            if isinstance(source_quantization, Mapping)
            else None
        ),
    }
    return output


def _weight_quantization_args(bits: int, group_size: int) -> dict[str, Any]:
    return {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": group_size,
        "num_bits": bits,
        "observer": "memoryless_minmax",
        "observer_kwargs": {},
        "scale_dtype": None,
        "strategy": "group",
        "symmetric": True,
        "type": "int",
        "zp_dtype": None,
    }


def _recipe_payload(recipe: ArtifactRecipe) -> dict[str, Any]:
    return {
        "default": {
            "w13_bits": recipe.default.w13_bits,
            "w2_bits": recipe.default.w2_bits,
        },
        "layers": {
            str(layer): {
                "w13_bits": quantization.w13_bits,
                "w2_bits": quantization.w2_bits,
            }
            for layer, quantization in sorted(recipe.layers.items())
        },
    }
