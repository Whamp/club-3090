from __future__ import annotations

import argparse
from pathlib import Path

from deepseek_v4_lowbit.frontier_resume import (
    require_frontier_resume_validation_receipt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Require an exact CPU-validated frontier resume receipt."
    )
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--volume-id", required=True)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=("cliff", "capacity", "balanced", "quality"),
    )
    parser.add_argument("--recovery-manifest-sha256", required=True)
    arguments = parser.parse_args(argv)

    receipt = require_frontier_resume_validation_receipt(
        arguments.receipt.resolve(),
        volume_id=arguments.volume_id,
        candidate=arguments.candidate,
        expected_recovery_manifest_sha256=arguments.recovery_manifest_sha256,
    )
    print(
        "frontier_resume_receipt_verified=true "
        f"volume_id={arguments.volume_id} "
        f"candidate={arguments.candidate} "
        f"identity={receipt['validation_identity_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
