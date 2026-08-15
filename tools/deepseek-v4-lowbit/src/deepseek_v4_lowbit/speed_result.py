from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_DECODE_FLOOR = 46.17
_PREFILL_FLOOR = 552.546


def _required_match(pattern: str, text: str, field: str) -> float:
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"DeepSeek V4 speed result is missing {field}")
    return float(match.group(1))


def parse_speed_measurement(directory: Path) -> dict[str, Any]:
    bench_log = (directory / "bench.log").read_text(encoding="utf-8")
    decode_tps = _required_match(
        r"summary \[code\].*?decode_TPS\s+mean=\s*([0-9.]+)",
        bench_log,
        "decode mean",
    )
    prefill_tps = _required_match(
        r"summary \[prefill-[^]]+\].*?prefill tok/s\s+mean=\s*([0-9.]+)",
        bench_log,
        "prefill mean",
    )

    swap_values: list[int] = []
    for line in (directory / "worker-swap-kib.txt").read_text().splitlines():
        _pid, swap_kib = line.split()
        swap_values.append(int(swap_kib))
    if not swap_values:
        raise ValueError("DeepSeek V4 speed result has no worker swap inventory")

    free_values: list[int] = []
    for line in (directory / "gpu-after.csv").read_text().splitlines():
        fields = [field.strip() for field in line.split(",")]
        free_values.append(int(fields[2]))
    if len(free_values) != 4:
        raise ValueError("DeepSeek V4 speed result must report four GPUs")

    startup = (directory / "startup.log").read_text(encoding="utf-8")
    cache_token_values = [
        int(value.replace(",", ""))
        for value in re.findall(r"GPU KV cache size:\s*([0-9,]+) tokens", startup)
    ]
    return {
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "minimum_free_mib": min(free_values),
        "maximum_worker_swap_kib": max(swap_values),
        "kv_cache_tokens": min(cache_token_values) if cache_token_values else None,
    }


def compare_speed_measurements(
    baseline_directory: Path,
    candidate_directory: Path,
) -> dict[str, Any]:
    baseline = parse_speed_measurement(baseline_directory)
    candidate = parse_speed_measurement(candidate_directory)
    decode_ratio = candidate["decode_tps"] / baseline["decode_tps"]
    prefill_ratio = candidate["prefill_tps"] / baseline["prefill_tps"]
    capacity_delta = None
    if (
        baseline["kv_cache_tokens"] is not None
        and candidate["kv_cache_tokens"] is not None
    ):
        capacity_delta = candidate["kv_cache_tokens"] - baseline["kv_cache_tokens"]
    gates = {
        "decode_floor": candidate["decode_tps"] >= _DECODE_FLOOR,
        "prefill_floor": candidate["prefill_tps"] >= _PREFILL_FLOOR,
        "zero_worker_swap": candidate["maximum_worker_swap_kib"] == 0,
    }
    return {
        "schema_version": 1,
        "baseline": baseline,
        "candidate": candidate,
        "decode_ratio": decode_ratio,
        "decode_delta_fraction": decode_ratio - 1.0,
        "prefill_ratio": prefill_ratio,
        "prefill_delta_fraction": prefill_ratio - 1.0,
        "kv_cache_token_delta": capacity_delta,
        "gates": gates,
        "passes_hard_gates": all(gates.values()),
        "operator_note": (
            "Hard gates do not establish a speed win. Review run variance, "
            "dispatch evidence, realized KV capacity, and the predicted mediator."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare matched DeepSeek V4 SM86 speed measurements."
    )
    parser.add_argument("baseline_directory", type=Path)
    parser.add_argument("candidate_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_speed_measurements(
        args.baseline_directory,
        args.candidate_directory,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
