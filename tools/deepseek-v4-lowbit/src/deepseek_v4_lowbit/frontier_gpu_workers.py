from __future__ import annotations

import importlib
import multiprocessing
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.artifact_plan import ArtifactRecipe
from deepseek_v4_lowbit.frontier_screen import (
    FrontierScreenOptions,
    FrontierScreenResult,
    FrontierScreenSample,
    screen_quantization_frontier,
)
from deepseek_v4_lowbit.imatrix import ImatrixFile
from deepseek_v4_lowbit.shard_writer import ResumableSafetensorsWriter
from deepseek_v4_lowbit.source_transform import (
    ShardTransformResult,
    TransformOptions,
    transform_source_shard,
)


@dataclass(frozen=True)
class FrontierScreenGpuBatch:
    """Own complete screen layers on one fixed physical GPU selector."""

    physical_device: str
    samples: tuple[FrontierScreenSample, ...]
    workload_units: int


@dataclass(frozen=True)
class FrontierConversionGpuBatch:
    """Own disjoint source shards on one fixed physical GPU selector."""

    physical_device: str
    shard_names: tuple[str, ...]


@dataclass(frozen=True)
class FixedGpuRuntimeInspection:
    """Report one spawned worker's physical selector and CUDA runtime identity."""

    physical_device: str
    device_name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int


@dataclass(frozen=True)
class _FrontierScreenGpuTask:
    source_directory: Path
    weight_map: dict[str, str]
    imatrix_path: Path
    options: FrontierScreenOptions
    batch: FrontierScreenGpuBatch


@dataclass(frozen=True)
class _FrontierConversionGpuTask:
    source_directory: Path
    output_directory: Path
    recipe: ArtifactRecipe
    options: TransformOptions
    imatrix_path: Path
    batch: FrontierConversionGpuBatch


def validate_fixed_gpu_devices(gpu_devices: Sequence[str]) -> tuple[str, ...]:
    """Require unique single-device selectors suitable for CUDA visibility."""
    devices = tuple(gpu_devices)
    if not devices:
        raise ValueError("fixed GPU worker pool requires at least one GPU device")
    if any(
        not device or device != device.strip() or "," in device for device in devices
    ):
        raise ValueError("fixed GPU worker selectors must each name one device")
    if len(set(devices)) != len(devices):
        raise ValueError("fixed GPU worker selectors must be unique")
    return devices


def partition_frontier_screen_samples(
    samples: Iterable[FrontierScreenSample],
    gpu_devices: Sequence[str],
    *,
    group_sizes: Sequence[int],
) -> tuple[FrontierScreenGpuBatch, ...]:
    """Balance screen work by layer without splitting one layer across GPUs."""
    devices = validate_fixed_gpu_devices(gpu_devices)
    unique_group_sizes = tuple(sorted(set(group_sizes)))
    if not unique_group_sizes or any(size <= 0 for size in unique_group_sizes):
        raise ValueError("frontier screen group sizes must be positive")

    samples_by_layer: dict[int, list[FrontierScreenSample]] = {}
    for sample in sorted(set(samples)):
        samples_by_layer.setdefault(sample.layer, []).append(sample)
    assignments = {device: [] for device in devices}
    workload_by_device = {device: 0 for device in devices}
    device_order = {device: index for index, device in enumerate(devices)}
    layers = sorted(
        samples_by_layer.items(),
        key=lambda item: (
            -_screen_workload_units(item[1], unique_group_sizes),
            item[0],
        ),
    )
    for _, layer_samples in layers:
        device = min(
            devices,
            key=lambda candidate: (
                workload_by_device[candidate],
                device_order[candidate],
            ),
        )
        assignments[device].extend(layer_samples)
        workload_by_device[device] += _screen_workload_units(
            layer_samples,
            unique_group_sizes,
        )
    return tuple(
        FrontierScreenGpuBatch(
            physical_device=device,
            samples=tuple(sorted(assignments[device])),
            workload_units=workload_by_device[device],
        )
        for device in devices
        if assignments[device]
    )


def partition_frontier_conversion_shards(
    shard_names: Iterable[str],
    gpu_devices: Sequence[str],
) -> tuple[FrontierConversionGpuBatch, ...]:
    """Assign every conversion shard exactly once across fixed GPU workers."""
    devices = validate_fixed_gpu_devices(gpu_devices)
    assignments = {device: [] for device in devices}
    for index, shard_name in enumerate(sorted(set(shard_names))):
        assignments[devices[index % len(devices)]].append(shard_name)
    return tuple(
        FrontierConversionGpuBatch(device, tuple(assignments[device]))
        for device in devices
        if assignments[device]
    )


def inspect_fixed_gpu_runtime(
    gpu_devices: Sequence[str],
) -> tuple[FixedGpuRuntimeInspection, ...]:
    """Spawn one child per selector and prove a real CUDA operation succeeds."""
    devices = validate_fixed_gpu_devices(gpu_devices)
    inspections = _run_fixed_gpu_batches(_inspect_fixed_gpu_runtime_task, devices)
    observed_devices = tuple(item.physical_device for item in inspections)
    if observed_devices != devices:
        raise RuntimeError(
            "fixed GPU runtime inspection returned unexpected selectors: "
            f"expected={devices!r} observed={observed_devices!r}"
        )
    return inspections


def screen_quantization_frontier_on_fixed_gpus(
    source_directory: Path,
    weight_map: Mapping[str, str],
    imatrix_path: Path,
    samples: Iterable[FrontierScreenSample],
    options: FrontierScreenOptions,
    gpu_devices: Sequence[str],
) -> tuple[FrontierScreenResult, ...]:
    """Run unchanged frontier screens in spawned, one-GPU child processes."""
    _require_canonical_cuda_device(options.device)
    ordered_samples = tuple(sorted(set(samples)))
    batches = partition_frontier_screen_samples(
        ordered_samples,
        gpu_devices,
        group_sizes=options.group_sizes,
    )
    tasks = tuple(
        _FrontierScreenGpuTask(
            source_directory=source_directory,
            weight_map={
                sample.tensor_name: weight_map[sample.tensor_name]
                for sample in batch.samples
            },
            imatrix_path=imatrix_path,
            options=options,
            batch=batch,
        )
        for batch in batches
    )
    worker_results = _run_fixed_gpu_batches(_run_frontier_screen_gpu_task, tasks)
    results = tuple(
        result for worker_result in worker_results for result in worker_result
    )
    sample_order = {
        sample.tensor_name: index for index, sample in enumerate(ordered_samples)
    }
    group_order = {
        group_size: index
        for index, group_size in enumerate(sorted(set(options.group_sizes)))
    }
    return tuple(
        sorted(
            results,
            key=lambda result: (
                sample_order[result.tensor_name],
                group_order[result.group_size],
                result.bits,
            ),
        )
    )


def transform_frontier_shards_on_fixed_gpus(
    source_directory: Path,
    output_directory: Path,
    shard_names: Iterable[str],
    *,
    recipe: ArtifactRecipe,
    options: TransformOptions,
    imatrix_path: Path,
    gpu_devices: Sequence[str],
) -> tuple[ShardTransformResult, ...]:
    """Transform disjoint output shards in spawned, one-GPU child processes."""
    _require_canonical_cuda_device(options.device)
    ordered_shards = tuple(sorted(set(shard_names)))
    batches = partition_frontier_conversion_shards(ordered_shards, gpu_devices)
    tasks = tuple(
        _FrontierConversionGpuTask(
            source_directory=source_directory,
            output_directory=output_directory,
            recipe=recipe,
            options=options,
            imatrix_path=imatrix_path,
            batch=batch,
        )
        for batch in batches
    )
    worker_results = _run_fixed_gpu_batches(_run_frontier_conversion_gpu_task, tasks)
    flat_results = tuple(
        result for worker_result in worker_results for result in worker_result
    )
    results_by_shard = {result.receipt.shard_name: result for result in flat_results}
    if len(results_by_shard) != len(flat_results):
        raise RuntimeError("fixed GPU conversion workers returned duplicate shards")
    if set(results_by_shard) != set(ordered_shards):
        raise RuntimeError(
            "fixed GPU conversion workers returned an incomplete shard set"
        )
    return tuple(results_by_shard[shard_name] for shard_name in ordered_shards)


def _inspect_fixed_gpu_runtime_task(
    physical_device: str,
) -> FixedGpuRuntimeInspection:
    _select_one_physical_gpu(physical_device)
    torch = importlib.import_module("torch")
    smoke_value = float(torch.ones(1, device="cuda").sum().item())
    if smoke_value != 1.0:
        raise RuntimeError(
            f"fixed GPU CUDA operation failed for selector {physical_device!r}"
        )
    properties = torch.cuda.get_device_properties(0)
    return FixedGpuRuntimeInspection(
        physical_device=physical_device,
        device_name=str(properties.name),
        compute_capability=(int(properties.major), int(properties.minor)),
        total_memory_bytes=int(properties.total_memory),
    )


def _run_frontier_screen_gpu_task(
    task: _FrontierScreenGpuTask,
) -> tuple[FrontierScreenResult, ...]:
    _select_one_physical_gpu(task.batch.physical_device)
    with ImatrixFile.open(task.imatrix_path) as imatrix:
        imatrix.validate_deepseek_v4_geometry()
        return screen_quantization_frontier(
            task.source_directory,
            task.weight_map,
            imatrix,
            task.batch.samples,
            task.options,
        )


def _run_frontier_conversion_gpu_task(
    task: _FrontierConversionGpuTask,
) -> tuple[ShardTransformResult, ...]:
    _select_one_physical_gpu(task.batch.physical_device)
    writer = ResumableSafetensorsWriter(task.output_directory)
    results = []
    with ImatrixFile.open(task.imatrix_path) as imatrix:
        imatrix.validate_deepseek_v4_geometry()
        for shard_name in task.batch.shard_names:
            results.append(
                transform_source_shard(
                    task.source_directory / shard_name,
                    shard_name,
                    writer=writer,
                    recipe=task.recipe,
                    options=task.options,
                    imatrix=imatrix,
                )
            )
    return tuple(results)


def _select_one_physical_gpu(physical_device: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "fixed GPU worker did not resolve exactly one CUDA device: "
            f"selector={physical_device!r} count={torch.cuda.device_count()}"
        )


def _screen_workload_units(
    samples: Iterable[FrontierScreenSample],
    group_sizes: Sequence[int],
) -> int:
    return len(group_sizes) * sum(
        2 if sample.projection == "w2" else 1 for sample in samples
    )


def _require_canonical_cuda_device(device: str) -> None:
    if device != "cuda":
        raise ValueError(
            "fixed GPU workers require canonical device='cuda'; physical GPU "
            "selection belongs in the worker visibility contract"
        )


def _spawn_fixed_gpu_pool(process_count: int) -> Any:
    return multiprocessing.get_context("spawn").Pool(processes=process_count)


def _run_fixed_gpu_batches(
    worker: Callable[[Any], Any],
    batches: Sequence[Any],
    *,
    pool_factory: Callable[[int], Any] = _spawn_fixed_gpu_pool,
) -> tuple[Any, ...]:
    """Run each fixed-GPU batch in its own child and fail the whole cohort."""
    if not batches:
        return ()
    pools = []
    try:
        for _ in batches:
            pools.append(pool_factory(1))
        pending_results = [
            pool.apply_async(worker, (batch,))
            for pool, batch in zip(pools, batches, strict=True)
        ]
        for pool in pools:
            pool.close()
        completed: list[Any | None] = [None] * len(pending_results)
        pending_indexes = set(range(len(pending_results)))
        while pending_indexes:
            made_progress = False
            for index in tuple(sorted(pending_indexes)):
                if not pending_results[index].ready():
                    continue
                completed[index] = pending_results[index].get()
                pending_indexes.remove(index)
                made_progress = True
            if pending_indexes and not made_progress:
                time.sleep(0.05)
        for pool in pools:
            pool.join()
        return tuple(completed)
    except BaseException as error:
        _terminate_fixed_gpu_pools(pools, error)
        raise


def _terminate_fixed_gpu_pools(pools: Sequence[Any], error: BaseException) -> None:
    """Attempt every pool cleanup action without masking the primary failure."""
    for action_name in ("terminate", "join"):
        for pool in pools:
            try:
                getattr(pool, action_name)()
            except BaseException as cleanup_error:
                error.add_note(
                    f"fixed GPU pool {action_name} failed: {cleanup_error!r}"
                )
