from __future__ import annotations

import pytest

from deepseek_v4_lowbit.speed_trace_gate import validate_speed_trace_gate


def valid_gate() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_id": "deepseek-v4-flash-0731-wna16-quality-12035985",
        "vllm_tree": "6354125afd1306c9286f734d1c47c23c767d77a9",
        "profile_sha256": "a" * 64,
        "request_sha256": "b" * 64,
        "all_reduce_critical_path_fraction": 0.15,
        "timeline_reviewed": True,
        "reviewer_note": "NCCL kernels serialize the measured decode interval.",
    }


def test_trace_gate_accepts_reviewed_material_all_reduce() -> None:
    payload = valid_gate()
    assert validate_speed_trace_gate(payload) is payload


def test_trace_gate_rejects_small_all_reduce_fraction() -> None:
    payload = valid_gate()
    payload["all_reduce_critical_path_fraction"] = 0.09
    with pytest.raises(ValueError, match="does not justify"):
        validate_speed_trace_gate(payload)


def test_trace_gate_rejects_unreviewed_timeline() -> None:
    payload = valid_gate()
    payload["timeline_reviewed"] = False
    with pytest.raises(ValueError, match="timeline_reviewed"):
        validate_speed_trace_gate(payload)
