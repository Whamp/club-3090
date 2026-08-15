from __future__ import annotations

from pathlib import Path

import pytest

from deepseek_v4_lowbit.speed_result import compare_speed_measurements


def _write_measurement(
    directory: Path,
    *,
    decode_tps: float,
    prefill_tps: float,
    free_mib: int,
    swap_kib: int,
    cache_tokens: int,
) -> None:
    directory.mkdir()
    (directory / "bench.log").write_text(
        f"""
=== summary [code] (n=5) ===
  decode_TPS     mean= {decode_tps:.2f} std=0.1
=== summary [prefill-8k] (n=3) ===
  prefill tok/s  mean= {prefill_tps:.2f} std=1.0
"""
    )
    (directory / "worker-swap-kib.txt").write_text(
        "\n".join(f"{1000 + rank} {swap_kib}" for rank in range(4)) + "\n"
    )
    (directory / "gpu-after.csv").write_text(
        "\n".join(f"{rank}, 24000, {free_mib}, 0, 120, 60" for rank in range(4)) + "\n"
    )
    (directory / "startup.log").write_text(
        f"GPU KV cache size: {cache_tokens:,} tokens\n"
    )


def test_compare_speed_measurements_reports_ratios_and_capacity(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_measurement(
        baseline,
        decode_tps=60,
        prefill_tps=900,
        free_mib=100,
        swap_kib=0,
        cache_tokens=275_000,
    )
    _write_measurement(
        candidate,
        decode_tps=66,
        prefill_tps=930,
        free_mib=96,
        swap_kib=0,
        cache_tokens=274_000,
    )

    comparison = compare_speed_measurements(baseline, candidate)

    assert comparison["decode_ratio"] == pytest.approx(1.1)
    assert comparison["prefill_ratio"] == pytest.approx(930 / 900)
    assert comparison["kv_cache_token_delta"] == -1_000
    assert comparison["passes_hard_gates"] is True


def test_compare_speed_measurements_fails_swap_and_performance_gates(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_measurement(
        baseline,
        decode_tps=60,
        prefill_tps=900,
        free_mib=100,
        swap_kib=0,
        cache_tokens=275_000,
    )
    _write_measurement(
        candidate,
        decode_tps=40,
        prefill_tps=500,
        free_mib=200,
        swap_kib=12,
        cache_tokens=280_000,
    )

    comparison = compare_speed_measurements(baseline, candidate)

    assert comparison["gates"] == {
        "decode_floor": False,
        "prefill_floor": False,
        "zero_worker_swap": False,
    }
    assert comparison["passes_hard_gates"] is False
