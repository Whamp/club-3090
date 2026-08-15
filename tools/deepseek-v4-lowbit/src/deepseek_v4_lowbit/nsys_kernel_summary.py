from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _number(value: str) -> float:
    return float(value.strip().replace(",", ""))


def summarize_nsys_cuda_kernels(path: Path) -> dict[str, Any]:
    """Summarize NCCL share from an Nsight CUDA GPU kernel summary CSV."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Total Time" in line and "Name" in line
        ),
        None,
    )
    if header_index is None:
        raise ValueError("Nsight kernel summary has no Total Time/Name header")
    rows = list(csv.DictReader(lines[header_index:]))
    if not rows:
        raise ValueError("Nsight kernel summary contains no kernels")
    time_column = next(
        (column for column in rows[0] if column and "Total Time" in column),
        None,
    )
    name_column = next(
        (column for column in rows[0] if column and column.strip() == "Name"),
        None,
    )
    if time_column is None or name_column is None:
        raise ValueError("Nsight kernel summary columns are incomplete")

    total_time = 0.0
    nccl_time = 0.0
    nccl_rows = []
    for row in rows:
        kernel_time = _number(row[time_column])
        kernel_name = row[name_column]
        total_time += kernel_time
        lowered = kernel_name.lower()
        if "nccl" in lowered or "allreduce" in lowered or "all_reduce" in lowered:
            nccl_time += kernel_time
            nccl_rows.append({"name": kernel_name, "total_time": kernel_time})
    if total_time <= 0:
        raise ValueError("Nsight kernel summary total time must be positive")
    return {
        "schema_version": 1,
        "scope": "summed_cuda_kernel_time_not_critical_path",
        "total_kernel_time": total_time,
        "nccl_kernel_time": nccl_time,
        "nccl_kernel_time_fraction": nccl_time / total_time,
        "nccl_kernels": nccl_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize NCCL kernel time from Nsight Systems CSV."
    )
    parser.add_argument("kernel_summary_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(
        summarize_nsys_cuda_kernels(args.kernel_summary_csv),
        indent=2,
        sort_keys=True,
    )
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
