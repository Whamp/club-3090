from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.artifact_plan import (
    ArtifactRecipe,
    LayerQuantization,
    load_tensor_headers,
    plan_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan an MTP-free DeepSeek V4 Humming WNA16 artifact."
    )
    parser.add_argument("headers", type=Path, help="Captured safetensors headers JSON")
    parser.add_argument("recipe", type=Path, help="Mixed-projection recipe JSON")
    parser.add_argument("--group-size", type=int, default=128)
    arguments = parser.parse_args(argv)

    recipe = _load_recipe(arguments.recipe)
    plan = plan_artifact(
        load_tensor_headers(arguments.headers),
        recipe,
        group_size=arguments.group_size,
    )
    output = asdict(plan)
    output["total_gib"] = plan.total_bytes / 2**30
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _load_recipe(path: Path) -> ArtifactRecipe:
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


def _load_layer_quantization(value: Any, location: str) -> LayerQuantization:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    if set(value) != {"w13_bits", "w2_bits"}:
        raise ValueError(f"{location} must contain only w13_bits and w2_bits")
    w13_bits = value["w13_bits"]
    w2_bits = value["w2_bits"]
    if not isinstance(w13_bits, int) or not isinstance(w2_bits, int):
        raise ValueError(f"{location} bit widths must be integers")
    return LayerQuantization(w13_bits=w13_bits, w2_bits=w2_bits)


if __name__ == "__main__":
    raise SystemExit(main())
