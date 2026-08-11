from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from deepseek_v4_lowbit.artifact_plan import (
    load_artifact_recipe,
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

    recipe = load_artifact_recipe(arguments.recipe)
    plan = plan_artifact(
        load_tensor_headers(arguments.headers),
        recipe,
        group_size=arguments.group_size,
    )
    output = asdict(plan)
    output["total_gib"] = plan.total_bytes / 2**30
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
