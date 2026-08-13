from __future__ import annotations

import argparse
import os
from pathlib import Path

from deepseek_v4_lowbit.frontier_publish import (
    frontier_publish_report_payload,
    publish_frontier_candidate,
    write_frontier_publish_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish and verify one DeepSeek V4 frontier candidate branch."
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument("repository")
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--baseline-revision",
        default="75d9286c37f3037f3ab390cfbc10747466eac714",
    )
    parser.add_argument("--branch-prefix", default="frontier")
    parser.add_argument(
        "--candidate",
        required=True,
        choices=("cliff", "capacity", "balanced", "quality"),
    )
    arguments = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("frontier publication requires HF_TOKEN")

    output_root = arguments.output_root.resolve()
    candidate_directory = output_root / arguments.candidate
    if not candidate_directory.is_dir():
        parser.error(f"frontier candidate directory is missing: {candidate_directory}")
    branch = f"{arguments.branch_prefix}/{arguments.candidate}"
    published = publish_frontier_candidate(
        candidate_directory,
        arguments.repository,
        candidate=arguments.candidate,
        branch=branch,
        parent_revision=arguments.baseline_revision,
        token=token,
    )
    report_path = arguments.report.resolve()
    write_frontier_publish_report(
        report_path,
        frontier_publish_report_payload(
            (published,),
            repository=arguments.repository,
            parent_revision=arguments.baseline_revision,
        ),
    )
    print(
        f"published candidate={published.candidate} branch={published.branch} "
        f"revision={published.revision} files={published.verification.file_count} "
        f"bytes={published.verification.total_bytes}"
    )
    print(f"frontier publish report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
