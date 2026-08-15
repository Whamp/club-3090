from __future__ import annotations

import json

import pytest

from deepseek_v4_lowbit.speed_experiment import (
    BASELINE_PROFILE,
    EXPERIMENT_ARMS,
    ExperimentArm,
    build_speed_experiment_manifest,
)

EXPECTED_CHANGES = {
    "prefill-block2": {"VLLM_SPARSE_DENSE_QUERY_BLOCK": "2"},
    "flashmla-decode": {"VLLM_DSV4_FLASH_MLA_DECODE": "1"},
    "hier-allreduce": {"VLLM_HIER_ALL_REDUCE": "0,1;2,3"},
    "indexer96": {"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "96"},
    "batched320": {"MAX_NUM_BATCHED_TOKENS": "320"},
}


def test_speed_experiment_arms_change_one_variable() -> None:
    assert set(EXPERIMENT_ARMS) == {"baseline", "trace-baseline", *EXPECTED_CHANGES}
    for name, expected_change in EXPECTED_CHANGES.items():
        arm = EXPERIMENT_ARMS[name]
        assert arm.changed_values == expected_change
        full_profile = arm.full_profile()
        changed = {
            key: value
            for key, value in full_profile.items()
            if BASELINE_PROFILE.get(key) != value
        }
        assert changed == expected_change


def test_speed_experiment_baseline_is_an_exact_control() -> None:
    baseline = EXPERIMENT_ARMS["baseline"]
    assert baseline.changed_values == {}
    assert baseline.full_profile() == BASELINE_PROFILE


def test_speed_experiment_trace_arm_disables_unused_host_kv_tier() -> None:
    trace = EXPERIMENT_ARMS["trace-baseline"]
    assert trace.observational is True
    assert trace.changed_values == {"KV_OFFLOADING_SIZE": "0.001"}
    assert trace.full_profile() == {
        **BASELINE_PROFILE,
        "KV_OFFLOADING_SIZE": "0.001",
    }
    assert BASELINE_PROFILE["KV_OFFLOADING_SIZE"] == "16"


def test_speed_experiment_rejects_more_than_one_change() -> None:
    arm = ExperimentArm(
        name="bundled",
        outcome="invalid",
        changed_values={
            "VLLM_SPARSE_DENSE_QUERY_BLOCK": "2",
            "MAX_NUM_BATCHED_TOKENS": "320",
        },
    )
    with pytest.raises(ValueError, match="exactly one"):
        arm.validate()


def test_speed_experiment_manifest_records_deferred_gpu_evidence() -> None:
    manifest = build_speed_experiment_manifest("flashmla-decode")

    assert manifest["arm"] == "flashmla-decode"
    assert manifest["server_changed"] is False
    assert manifest["gpu_evidence_status"] == "deferred"
    assert manifest["required_gates"] == [
        "image_identity",
        "model_identity",
        "startup_dispatch",
        "numerical_correctness",
        "deterministic_api",
        "tool_and_post_tool",
        "zero_worker_swap",
        "kv_capacity",
        "matched_decode",
        "cache_busted_prefill",
        "long_context_recall",
        "rollback_health",
    ]
    json.dumps(manifest, sort_keys=True)
