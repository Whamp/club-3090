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


def mixed_group_runtime_compatibility() -> dict[str, str]:
    """Return the exact unpromoted vLLM runtime required by frontier artifacts."""
    return {
        "acceptance_status": "pending-sm86-oracle-and-deepswe",
        "base_repository": "haosdent/vllm",
        "base_revision": "12810046c799cbe874967e19b1c0fa134ab7b209",
        "integration_repository": "Whamp/vllm",
        "integration_revision": "dd2d1fd6779addccc73094f77fa4ada7d9106a41",
        "required_tree": "f73b30cc5a2ed9de200ca2e4de3cdef1a06f6538",
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

    targets_by_schema: dict[tuple[int, int], list[str]] = {}
    for layer in range(layer_count):
        for projection, runtime_projection in _PROJECTION_NAMES.items():
            bits = recipe.bits_for(layer, projection)
            projection_group_size = recipe.group_size_for(
                layer,
                projection,
                fallback=group_size,
            )
            targets_by_schema.setdefault((bits, projection_group_size), []).append(
                f"model.layers.{layer}.ffn.experts.0.{runtime_projection}"
            )

    uses_mixed_group_sizes = (
        len({schema_group_size for _, schema_group_size in targets_by_schema}) > 1
    )
    config_groups = {}
    for (bits, schema_group_size), targets in sorted(targets_by_schema.items()):
        group_name = (
            f"group_w{bits}_g{schema_group_size}"
            if uses_mixed_group_sizes
            else f"group_w{bits}"
        )
        config_groups[group_name] = {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": sorted(targets),
            "weights": _weight_quantization_args(bits, schema_group_size),
        }
    return {
        "base_quant_method": "deepseek_v4_fp8",
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
    resolved_group_sizes = {
        (layer, projection): recipe.group_size_for(
            layer,
            projection,
            fallback=group_size,
        )
        for layer in range(layer_count)
        for projection in ("w13", "w2")
    }
    unique_group_sizes = set(resolved_group_sizes.values())
    projection_specific_groups = any(
        resolved_group_sizes[(layer, "w13")] != resolved_group_sizes[(layer, "w2")]
        for layer in range(layer_count)
    )
    group_size_label: int | str
    if len(unique_group_sizes) == 1:
        group_size_label = unique_group_sizes.pop()
    elif projection_specific_groups:
        group_size_label = "mixed-by-layer-and-projection"
    else:
        group_size_label = "mixed-by-layer"

    output["club_3090_lowbit"] = {
        "format": "symmetric-group-wna16",
        "group_size": group_size_label,
        "mtp": "omitted",
        "recipe": _recipe_payload(recipe),
        "runtime_compatibility": mixed_group_runtime_compatibility(),
        "source_quantization_method": "compressed-tensors",
        "source_checkpoint_quantization_method": (
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
        "default": _layer_quantization_payload(recipe.default),
        "layers": {
            str(layer): _layer_quantization_payload(quantization)
            for layer, quantization in sorted(recipe.layers.items())
        },
    }


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
