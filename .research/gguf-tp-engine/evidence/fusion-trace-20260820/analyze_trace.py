#!/usr/bin/env python3
"""Analyze stable TP=4 CUDA Graph replays from the GGUF layer-slice trace."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

GRAPH_NODES_PER_REPLAY = 23
CAPTURED_REPLAYS_PER_RANK = 50

NODE_LABELS = (
    "attention_fused_wqa_wkv",
    "attention_wq_b",
    "attention_wo_a",
    "attention_pointwise_copy",
    "attention_wo_b",
    "attention_all_reduce",
    "routed_input_quantize",
    "routed_iq2_gate_up",
    "routed_swiglu_weighted_quantize",
    "routed_q2_down",
    "routed_topk_sum",
    "shared_gate_up",
    "shared_bf16_to_fp32",
    "shared_gate_clamp",
    "shared_up_clamp",
    "shared_silu",
    "shared_multiply",
    "shared_fp32_to_bf16",
    "shared_down",
    "shared_output_to_fp32",
    "routed_shared_add",
    "ffn_fp32_to_bf16",
    "ffn_all_reduce",
)

JsonObject = dict[str, Any]
KernelRow = tuple[int, int, int, str]


def summarize(values: Iterable[float]) -> JsonObject:
    """Return stable descriptive statistics in microseconds."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    return {
        "count": len(ordered),
        "min_us": ordered[0],
        "median_us": statistics.median(ordered),
        "mean_us": statistics.fmean(ordered),
        "max_us": ordered[-1],
    }


def interval_union_ns(rows: list[KernelRow]) -> int:
    """Measure covered GPU time without double-counting stream overlap."""
    intervals = sorted((start, end) for start, end, _, _ in rows)
    covered = 0
    active_start, active_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= active_end:
            active_end = max(active_end, end)
        else:
            covered += active_end - active_start
            active_start, active_end = start, end
    return covered + active_end - active_start


def load_graph_replays(database: Path) -> dict[int, list[list[KernelRow]]]:
    """Load graph kernels by device and split the repeated node sequence."""
    connection = sqlite3.connect(database)
    rows = connection.execute(
        """
        SELECT kernel.deviceId, kernel.start, kernel.end, kernel.graphNodeId,
               strings.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
        JOIN StringIds AS strings ON strings.id = kernel.shortName
        WHERE kernel.graphNodeId IS NOT NULL
        ORDER BY kernel.deviceId, kernel.start
        """
    ).fetchall()
    connection.close()

    by_device: dict[int, list[KernelRow]] = {}
    for device, start, end, graph_node, name in rows:
        by_device.setdefault(device, []).append((start, end, graph_node, name))
    if sorted(by_device) != [0, 1, 2, 3]:
        raise ValueError(f"expected devices 0..3, found {sorted(by_device)}")

    replays: dict[int, list[list[KernelRow]]] = {}
    for device, kernels in by_device.items():
        expected = GRAPH_NODES_PER_REPLAY * CAPTURED_REPLAYS_PER_RANK
        if len(kernels) != expected:
            raise ValueError(
                f"device {device}: expected {expected} graph kernels, found {len(kernels)}"
            )
        chunks = [
            kernels[offset : offset + GRAPH_NODES_PER_REPLAY]
            for offset in range(0, len(kernels), GRAPH_NODES_PER_REPLAY)
        ]
        reference_nodes = [row[2] for row in chunks[0]]
        for replay_index, chunk in enumerate(chunks):
            if [row[2] for row in chunk] != reference_nodes:
                raise ValueError(
                    f"device {device} replay {replay_index}: graph node order changed"
                )
        replays[device] = chunks
    return replays


def analyze(database: Path) -> JsonObject:
    """Analyze stable replays, excluding capture-start synchronization."""
    replays = load_graph_replays(database)
    stable_replays = [chunk for chunks in replays.values() for chunk in chunks[1:]]

    graph_spans = []
    kernel_sums = []
    busy_times = []
    internal_idle = []
    launch_gaps = []
    periods = []
    rank_summaries = []
    for device, chunks in replays.items():
        rank_spans = []
        rank_launch_gaps = []
        for replay_index, chunk in enumerate(chunks):
            span_ns = chunk[-1][1] - chunk[0][0]
            busy_ns = interval_union_ns(chunk)
            if replay_index > 0:
                kernel_sum_ns = sum(end - start for start, end, _, _ in chunk)
                launch_gap_ns = chunk[0][0] - chunks[replay_index - 1][-1][1]
                graph_spans.append(span_ns / 1000)
                kernel_sums.append(kernel_sum_ns / 1000)
                busy_times.append(busy_ns / 1000)
                internal_idle.append((span_ns - busy_ns) / 1000)
                launch_gaps.append(launch_gap_ns / 1000)
                rank_spans.append(span_ns / 1000)
                rank_launch_gaps.append(launch_gap_ns / 1000)
            if 0 < replay_index < len(chunks) - 1:
                periods.append((chunks[replay_index + 1][0][0] - chunk[0][0]) / 1000)
        rank_summaries.append(
            {
                "device": device,
                "stable_graph_span": summarize(rank_spans),
                "preceding_launch_gap": summarize(rank_launch_gaps),
            }
        )

    nodes: list[JsonObject] = []
    for position, label in enumerate(NODE_LABELS):
        durations = [
            (replay[position][1] - replay[position][0]) / 1000
            for replay in stable_replays
        ]
        short_names = sorted({replay[position][3] for replay in stable_replays})
        if len(short_names) != 1:
            raise ValueError(f"node {position}: kernel name changed: {short_names}")
        nodes.append(
            {
                "position": position,
                "label": label,
                "kernel": short_names[0],
                "duration": summarize(durations),
            }
        )

    median_node_us = [node["duration"]["median_us"] for node in nodes]
    groups = {
        "dense_marlin_projections": (0, 1, 2, 4, 11, 18),
        "hierarchical_all_reduce": (5, 22),
        "routed_major_matvecs": (7, 9),
        "original_f1_f2_removable_nodes": (6, 8, 10, 20, 21),
        "shared_swiglu_pointwise_chain": (12, 13, 14, 15, 16, 17),
        "final_add_cast_pointwise_chain": (19, 20, 21),
    }
    group_median_sums = {
        name: {
            "positions": list(positions),
            "median_duration_sum_us": sum(median_node_us[index] for index in positions),
        }
        for name, positions in groups.items()
    }

    return {
        "schema_version": 1,
        "database": database.name,
        "captured_replays_per_rank": CAPTURED_REPLAYS_PER_RANK,
        "stable_replays_per_rank": CAPTURED_REPLAYS_PER_RANK - 1,
        "excluded_replays": "replay 0 on each rank; capture-start barrier perturbs its first collective",
        "rank_summaries": rank_summaries,
        "all_stable_replays": {
            "graph_span": summarize(graph_spans),
            "kernel_duration_sum": summarize(kernel_sums),
            "gpu_busy_union": summarize(busy_times),
            "internal_idle": summarize(internal_idle),
            "preceding_graph_launch_gap": summarize(launch_gaps),
            "first_start_to_next_start_period": summarize(periods),
        },
        "node_groups": group_median_sums,
        "nodes": nodes,
        "decision": "FALSIFIED",
        "decision_rationale": (
            "Stable graph replays are kernel/collective dominated. The median internal idle "
            "and preceding launch gap are far below the preregistered 60-100 us/layer gap."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.database)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
