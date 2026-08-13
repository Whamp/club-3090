from __future__ import annotations

import argparse
from pathlib import Path

from deepseek_v4_lowbit.source_headers import (
    capture_source_tensor_headers,
    extract_captured_headers,
    source_tensor_headers_report,
    write_source_tensor_headers_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture checksum-bound DeepSeek V4 safetensors headers."
    )
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("planner_headers", type=Path)
    arguments = parser.parse_args(argv)
    source_directory = arguments.source_directory.resolve()
    report_path = arguments.report.resolve()
    planner_headers_path = arguments.planner_headers.resolve()
    headers = capture_source_tensor_headers(source_directory)
    write_source_tensor_headers_report(
        report_path,
        source_tensor_headers_report(source_directory, headers),
    )
    extract_captured_headers(report_path, planner_headers_path)
    tensor_count = sum(len(shard_headers) for shard_headers in headers.values())
    print(
        f"captured source headers: shards={len(headers)} tensors={tensor_count} "
        f"report={report_path} planner_headers={planner_headers_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
