from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepseek_v4_lowbit.frontier_convert import convert_frontier_candidates
from deepseek_v4_lowbit.frontier_recipe import load_json_object


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert nested DeepSeek V4 quantization-frontier candidates."
    )
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("recipe_bundle", type=Path)
    parser.add_argument("imatrix", type=Path)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=("cliff", "capacity", "balanced", "quality"),
    )
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)

    candidates = convert_frontier_candidates(
        arguments.source_directory.resolve(),
        arguments.output_root.resolve(),
        load_json_object(arguments.recipe_bundle.resolve()),
        baseline_directory=arguments.baseline_directory.resolve(),
        imatrix_path=arguments.imatrix.resolve(),
        device=arguments.device,
        candidate_names=(arguments.candidate,),
    )
    for candidate in candidates:
        print(
            json.dumps(
                {
                    "candidate": candidate.name,
                    "output_directory": str(candidate.output_directory),
                    "recipe_sha256": candidate.recipe_sha256,
                    "reused_shards": candidate.reused_shard_count,
                    "converted_shards": candidate.converted_shard_count,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
