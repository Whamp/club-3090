from __future__ import annotations

import argparse
from pathlib import Path

from deepseek_v4_lowbit.frontier_campaign import FrontierScreenCampaign


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the resumable DeepSeek V4 quantization-frontier screen."
    )
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("imatrix", type=Path)
    parser.add_argument("baseline_metrics", type=Path)
    parser.add_argument("source_headers_report", type=Path)
    parser.add_argument("tensor_headers", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--samples-per-projection", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    campaign = FrontierScreenCampaign(
        arguments.source_directory.resolve(),
        arguments.imatrix.resolve(),
        arguments.baseline_metrics.resolve(),
        arguments.source_headers_report.resolve(),
        arguments.tensor_headers.resolve(),
        arguments.output_directory.resolve(),
        samples_per_projection=arguments.samples_per_projection,
        device=arguments.device,
    )
    pilot, boundary, full_screen = campaign.run()
    print(f"frontier pilot report={pilot}")
    print(f"frontier boundary report={boundary}")
    print(f"frontier full-screen report={full_screen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
