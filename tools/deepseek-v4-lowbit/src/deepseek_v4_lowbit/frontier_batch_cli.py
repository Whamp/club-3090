from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from deepseek_v4_lowbit.frontier_batch import (
    FrontierBatchPublisher,
    run_frontier_conversion_batch,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert, publish, verify, and optionally delete frontier candidates."
        )
    )
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("recipe_bundle", type=Path)
    parser.add_argument("imatrix", type=Path)
    parser.add_argument("baseline_directory", type=Path)
    parser.add_argument("repository")
    parser.add_argument("report", type=Path)
    parser.add_argument("--parent-revision", required=True)
    parser.add_argument("--branch-prefix", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-device",
        action="append",
        default=[],
        help=(
            "Physical GPU selector for one spawned conversion worker; repeat "
            "for fixed multi-GPU conversion."
        ),
    )
    parser.add_argument(
        "--candidate",
        required=True,
        choices=("cliff", "capacity", "balanced", "quality"),
        help="Publish exactly one candidate; quality is the first campaign gate.",
    )
    parser.add_argument(
        "--evidence-file",
        type=Path,
        action="append",
        default=[],
        help="Checksum-preserved campaign evidence copied into every candidate.",
    )
    parser.add_argument("--delete-local-after-verify", action="store_true")
    arguments = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("frontier batch publication requires HF_TOKEN")

    recipe_bundle_path = arguments.recipe_bundle.resolve()
    recipe_bundle = json.loads(recipe_bundle_path.read_text(encoding="utf-8"))
    raw_summaries = recipe_bundle.get("candidate_summaries")
    if not isinstance(raw_summaries, list):
        parser.error("frontier recipe bundle has no candidate summaries")
    candidate_summaries = {
        summary["name"]: summary
        for summary in raw_summaries
        if isinstance(summary, dict) and isinstance(summary.get("name"), str)
    }
    if set(candidate_summaries) != {"cliff", "capacity", "balanced", "quality"}:
        parser.error("frontier candidate summaries are incomplete")

    publisher = FrontierBatchPublisher(
        arguments.repository,
        parent_revision=arguments.parent_revision,
        branch_prefix=arguments.branch_prefix,
        recipe_bundle_path=recipe_bundle_path,
        report_path=arguments.report.resolve(),
        token=token,
        delete_local_after_verify=arguments.delete_local_after_verify,
        candidate_summaries=candidate_summaries,
        candidate_names=(arguments.candidate,),
        evidence_files=tuple(
            evidence_file.resolve() for evidence_file in arguments.evidence_file
        ),
    )
    conversions = run_frontier_conversion_batch(
        arguments.source_directory.resolve(),
        arguments.output_root.resolve(),
        recipe_bundle_path,
        arguments.imatrix.resolve(),
        arguments.baseline_directory.resolve(),
        device=arguments.device,
        publisher=publisher,
        gpu_devices=tuple(arguments.gpu_device),
    )
    if tuple(item.candidate for item in publisher.published) != (arguments.candidate,):
        raise RuntimeError("frontier batch did not publish the selected candidate")
    print(f"frontier converted_this_run={len(conversions)}")
    for published in publisher.published:
        print(
            f"frontier candidate={published.candidate} "
            f"revision={published.revision} "
            f"bytes={published.verification.total_bytes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
