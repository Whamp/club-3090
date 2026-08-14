from __future__ import annotations

import argparse
from pathlib import Path

from deepseek_v4_lowbit.frontier_resume import (
    FrontierResumeCheckpointRequest,
    validate_frontier_resume_checkpoint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a recovered DeepSeek V4 frontier checkpoint before GPU resume."
        )
    )
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("baseline_directory", type=Path)
    parser.add_argument("imatrix", type=Path)
    parser.add_argument("recipe_bundle", type=Path)
    parser.add_argument("rebuilt_recipe_bundle", type=Path)
    parser.add_argument("recovery_manifest", type=Path)
    parser.add_argument("recovery_reports_directory", type=Path)
    parser.add_argument("--recovery-manifest-sha256", required=True)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--volume-id", required=True)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=("cliff", "capacity", "balanced", "quality"),
    )
    parser.add_argument("--pilot-screen-report", type=Path, required=True)
    parser.add_argument("--boundary-report", type=Path, required=True)
    parser.add_argument("--full-screen-report", type=Path, required=True)
    parser.add_argument("--source-headers-report", type=Path, required=True)
    parser.add_argument("--source-headers", type=Path, required=True)
    arguments = parser.parse_args(argv)

    receipt = validate_frontier_resume_checkpoint(
        FrontierResumeCheckpointRequest(
            source_directory=arguments.source_directory.resolve(),
            baseline_directory=arguments.baseline_directory.resolve(),
            imatrix_path=arguments.imatrix.resolve(),
            recipe_bundle_path=arguments.recipe_bundle.resolve(),
            rebuilt_recipe_bundle_path=arguments.rebuilt_recipe_bundle.resolve(),
            recovery_manifest_path=arguments.recovery_manifest.resolve(),
            expected_recovery_manifest_sha256=arguments.recovery_manifest_sha256,
            recovery_reports_directory=(arguments.recovery_reports_directory.resolve()),
            output_root=arguments.output_root.resolve(),
            receipt_path=arguments.receipt.resolve(),
            volume_id=arguments.volume_id,
            candidate=arguments.candidate,
            evidence_paths={
                "pilot_screen_report": arguments.pilot_screen_report.resolve(),
                "boundary_report": arguments.boundary_report.resolve(),
                "full_screen_report": arguments.full_screen_report.resolve(),
                "source_headers_report": arguments.source_headers_report.resolve(),
                "source_headers": arguments.source_headers.resolve(),
            },
        )
    )
    identity = receipt["validation_identity"]
    print(
        "frontier_resume_validated=true "
        f"volume_id={identity['volume_id']} "
        f"candidate={identity['candidate']} "
        f"reusable_baseline_shards={receipt['reusable_baseline_shards']} "
        f"completed_candidate_shards={receipt['completed_candidate_shards']} "
        f"identity={receipt['validation_identity_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
