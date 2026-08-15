from __future__ import annotations

import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.hier_all_reduce import (
    HierarchicalAllReduce,
)

_TEST_NUMELS = (4096, 8192, 32768, 65536, 262144)
_WARMUP = 10
_ITERATIONS = 50


def _measure_us(function) -> float:
    samples = []
    for _ in range(_WARMUP):
        function()
    torch.cuda.synchronize()
    for _ in range(_ITERATIONS):
        start = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return statistics.median(samples)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 4:
        raise RuntimeError(
            f"hierarchical all-reduce gate requires TP=4, got {world_size}"
        )
    torch.cuda.set_device(local_rank)
    capability = torch.cuda.get_device_capability(local_rank)
    if capability != (8, 6):
        raise RuntimeError(
            f"hierarchical all-reduce gate requires SM86, got {capability}"
        )

    device = torch.device("cuda", local_rank)
    backend = HierarchicalAllReduce(dist.group.WORLD, device, [[0, 1], [2, 3]])
    results = []
    for numel in _TEST_NUMELS:
        generator = torch.Generator(device=device).manual_seed(1000 + rank + numel)
        source = torch.randn(
            numel,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        reference = source.clone()
        dist.all_reduce(reference)
        dist.barrier()
        actual = backend.all_reduce(source)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)

        nccl_input = torch.empty_like(source)

        def run_nccl(
            input_tensor: torch.Tensor = nccl_input,
            source_tensor: torch.Tensor = source,
        ) -> None:
            input_tensor.copy_(source_tensor)
            dist.all_reduce(input_tensor)

        def run_hierarchical(source_tensor: torch.Tensor = source) -> None:
            backend.all_reduce(source_tensor)

        nccl_us = _measure_us(run_nccl)
        hierarchical_us = _measure_us(run_hierarchical)
        results.append(
            {
                "numel": numel,
                "bytes": numel * source.element_size(),
                "nccl_median_us": nccl_us,
                "hierarchical_median_us": hierarchical_us,
                "hierarchical_over_nccl": hierarchical_us / nccl_us,
            }
        )
        dist.barrier()

    if rank == 0:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "world_size": world_size,
                    "islands": [[0, 1], [2, 3]],
                    "capability": "8.6",
                    "warmup": _WARMUP,
                    "iterations": _ITERATIONS,
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
