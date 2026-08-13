from __future__ import annotations

import argparse
import json

from deepseek_v4_lowbit.frontier_gpu_workers import inspect_fixed_gpu_runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify fixed-GPU frontier workers before expensive rental work."
    )
    parser.add_argument(
        "--gpu-device",
        action="append",
        required=True,
        help="Physical GPU selector; repeat once per dedicated worker.",
    )
    arguments = parser.parse_args(argv)
    inspections = inspect_fixed_gpu_runtime(tuple(arguments.gpu_device))
    print(
        json.dumps(
            [
                {
                    "physical_device": item.physical_device,
                    "device_name": item.device_name,
                    "compute_capability": list(item.compute_capability),
                    "total_memory_bytes": item.total_memory_bytes,
                }
                for item in inspections
            ],
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
