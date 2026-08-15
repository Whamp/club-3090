from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

BASELINE_PROFILE: dict[str, str] = {
    "MAX_MODEL_LEN": "230144",
    "GPU_MEMORY_UTILIZATION": "0.98",
    "MAX_NUM_SEQS": "2",
    "MAX_NUM_BATCHED_TOKENS": "256",
    "KV_OFFLOADING_SIZE": "16",
    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "64",
    "VLLM_SPARSE_DENSE_QUERY_BLOCK": "0",
    "VLLM_DSV4_FLASH_MLA_DECODE": "0",
    "VLLM_HIER_ALL_REDUCE": "",
}

REQUIRED_GATES = [
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


@dataclass(frozen=True)
class ExperimentArm:
    """One DeepSeek V4 speed hypothesis or explicit observation profile."""

    name: str
    outcome: str
    changed_values: dict[str, str] = field(default_factory=dict)
    precondition: str = "baseline-parity"
    predicted_mediator: str = ""
    lose_condition: str = ""
    observational: bool = False

    def validate(self) -> None:
        unknown = set(self.changed_values) - set(BASELINE_PROFILE)
        if unknown:
            raise ValueError(f"unknown speed experiment variables: {sorted(unknown)}")
        if self.name == "baseline":
            if self.changed_values:
                raise ValueError("the baseline speed arm must not change variables")
            return
        if self.observational:
            return
        if len(self.changed_values) != 1:
            raise ValueError(
                f"speed experiment arm {self.name!r} must change exactly one variable"
            )
        key, value = next(iter(self.changed_values.items()))
        if BASELINE_PROFILE[key] == value:
            raise ValueError(
                f"speed experiment arm {self.name!r} does not change {key}"
            )

    def full_profile(self) -> dict[str, str]:
        self.validate()
        return {**BASELINE_PROFILE, **self.changed_values}


EXPERIMENT_ARMS: dict[str, ExperimentArm] = {
    "baseline": ExperimentArm(
        name="baseline",
        outcome="Control for the speed-candidate image with every optimization off.",
        predicted_mediator="Matches the promoted profile before experiments.",
        lose_condition="Any material performance, capacity, or output drift.",
    ),
    "trace-baseline": ExperimentArm(
        name="trace-baseline",
        outcome="Attribute one baseline decode interval with Nsight Systems.",
        precondition="The unprofiled baseline passes and profiling is not benchmarked.",
        changed_values={"KV_OFFLOADING_SIZE": "0.001"},
        predicted_mediator="Reports sparse MLA, NCCL, MoE, and host-gap time shares.",
        lose_condition="Trace misses CUDA graphs/NCCL or changes request behavior.",
        observational=True,
    ),
    "prefill-block2": ExperimentArm(
        name="prefill-block2",
        outcome="Increase cache-busted prefill throughput without persistent memory.",
        changed_values={"VLLM_SPARSE_DENSE_QUERY_BLOCK": "2"},
        precondition="SM86 kernel compiles and direct prefill timing is lower.",
        predicted_mediator="Fewer repeated sparse-KV row loads in ratio-128 layers.",
        lose_condition="Compile failure, numerical mismatch, or lower macro prefill.",
    ),
    "flashmla-decode": ExperimentArm(
        name="flashmla-decode",
        outcome="Increase single-stream decode throughput without changing KV layout.",
        changed_values={"VLLM_DSV4_FLASH_MLA_DECODE": "1"},
        precondition="Pinned SM86 FlashMLA oracle and packaged cubin checks pass.",
        predicted_mediator="Lower fp8_ds_mla sparse-decode kernel time per layer.",
        lose_condition=(
            "Oracle mismatch, wrong dispatch, graph failure, or macro slowdown."
        ),
    ),
    "hier-allreduce": ExperimentArm(
        name="hier-allreduce",
        outcome="Reduce tensor-parallel decode coordination time.",
        changed_values={"VLLM_HIER_ALL_REDUCE": "0,1;2,3"},
        precondition=(
            "Exact backend oracle passes and an unprofiled baseline timeline shows "
            "all-reduce on at least 10% of the decode critical path."
        ),
        predicted_mediator="Lower small-message all-reduce latency across PHB islands.",
        lose_condition=(
            "IPC/P2P mismatch, hang, extra serialization, or macro slowdown."
        ),
    ),
    "indexer96": ExperimentArm(
        name="indexer96",
        outcome=(
            "Increase cache-busted prefill throughput with bounded workspace growth."
        ),
        changed_values={"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "96"},
        precondition="A baseline trace reports more than one sparse-indexer chunk.",
        predicted_mediator="Fewer sparse-indexer chunks and launches.",
        lose_condition=(
            "One baseline chunk, lower prefill, or unsafe physical headroom."
        ),
    ),
    "batched320": ExperimentArm(
        name="batched320",
        outcome=(
            "Increase cache-busted prefill throughput through larger scheduler chunks."
        ),
        changed_values={"MAX_NUM_BATCHED_TOKENS": "320"},
        precondition=(
            "Post-KV-work baseline has at least 256 MiB physical headroom per GPU "
            "and smaller-memory prefill arms are insufficient."
        ),
        predicted_mediator="Fewer scheduler chunks for the matched 9K prompt.",
        lose_condition=(
            "KV/capacity loss, activation OOM, or prefill below the baseline."
        ),
    ),
}


def build_speed_experiment_manifest(arm_name: str) -> dict[str, Any]:
    """Build a deferred experiment manifest without contacting server60."""
    try:
        arm = EXPERIMENT_ARMS[arm_name]
    except KeyError as error:
        raise ValueError(f"unknown speed experiment arm: {arm_name}") from error
    return {
        "schema_version": 1,
        "arm": arm.name,
        "outcome": arm.outcome,
        "precondition": arm.precondition,
        "predicted_mediator": arm.predicted_mediator,
        "lose_condition": arm.lose_condition,
        "changed_values": dict(sorted(arm.changed_values.items())),
        "observational": arm.observational,
        "full_profile": dict(sorted(arm.full_profile().items())),
        "required_gates": REQUIRED_GATES,
        "server_changed": False,
        "gpu_evidence_status": "deferred",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one deferred DeepSeek V4 speed experiment arm."
    )
    parser.add_argument("arm", choices=tuple(EXPERIMENT_ARMS))
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = json.dumps(
        build_speed_experiment_manifest(args.arm),
        indent=2,
        sort_keys=True,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
