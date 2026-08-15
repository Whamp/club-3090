from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

_MINIMUM_ALL_REDUCE_FRACTION = 0.10
_EXPECTED_MODEL_ID = "deepseek-v4-flash-0731-wna16-quality-12035985"
_EXPECTED_VLLM_TREE = "5238d1e4148bc747e122b9bc19bb1562a05b3207"


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"DeepSeek V4 trace gate {field} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"DeepSeek V4 trace gate {field} must be hexadecimal"
        ) from error
    return value


def validate_speed_trace_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate operator-reviewed Nsight evidence before hierarchical AR."""
    if payload.get("schema_version") != 1:
        raise ValueError("DeepSeek V4 trace gate schema_version must be 1")
    if payload.get("model_id") != _EXPECTED_MODEL_ID:
        raise ValueError("DeepSeek V4 trace gate model identity mismatch")
    if payload.get("vllm_tree") != _EXPECTED_VLLM_TREE:
        raise ValueError("DeepSeek V4 trace gate vLLM tree mismatch")
    _require_sha256(payload.get("profile_sha256"), "profile_sha256")
    _require_sha256(payload.get("request_sha256"), "request_sha256")
    fraction = payload.get("all_reduce_critical_path_fraction")
    if not isinstance(fraction, int | float) or not 0 <= fraction <= 1:
        raise ValueError(
            "DeepSeek V4 trace gate all_reduce_critical_path_fraction must be 0..1"
        )
    if fraction < _MINIMUM_ALL_REDUCE_FRACTION:
        raise ValueError(
            "DeepSeek V4 trace gate does not justify hierarchical all-reduce: "
            f"{fraction:.3f} < {_MINIMUM_ALL_REDUCE_FRACTION:.3f}"
        )
    if payload.get("timeline_reviewed") is not True:
        raise ValueError("DeepSeek V4 trace gate requires timeline_reviewed=true")
    if not str(payload.get("reviewer_note", "")).strip():
        raise ValueError("DeepSeek V4 trace gate requires a reviewer_note")
    return payload


def validate_speed_trace_gate_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as trace_file:
        payload = json.load(trace_file)
    if not isinstance(payload, dict):
        raise ValueError("DeepSeek V4 trace gate must be a JSON object")
    validate_speed_trace_gate(payload)
    return {
        "trace_gate_path": str(path.resolve()),
        "trace_gate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "all_reduce_critical_path_fraction": payload[
            "all_reduce_critical_path_fraction"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate reviewed Nsight evidence for hierarchical all-reduce."
    )
    parser.add_argument("trace_gate_json", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_speed_trace_gate_file(args.trace_gate_json),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
