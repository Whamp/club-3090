from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from deepseek_v4_lowbit.artifact_plan import ArtifactRecipe, LayerQuantization
from deepseek_v4_lowbit.frontier_batch import run_frontier_conversion_batch
from deepseek_v4_lowbit.frontier_campaign_cli import main as frontier_campaign_main
from deepseek_v4_lowbit.frontier_gpu_inspect_cli import main as gpu_inspect_main
from deepseek_v4_lowbit.frontier_gpu_workers import (
    FixedGpuRuntimeInspection,
    _run_fixed_gpu_batches,
    _select_one_physical_gpu,
    partition_frontier_conversion_shards,
    partition_frontier_screen_samples,
    transform_frontier_shards_on_fixed_gpus,
    validate_fixed_gpu_devices,
)
from deepseek_v4_lowbit.frontier_screen import FrontierScreenSample
from deepseek_v4_lowbit.shard_writer import ShardIdentity, ShardReceipt
from deepseek_v4_lowbit.source_transform import ShardTransformResult, TransformOptions


class _ReadyResult:
    def __init__(self, *, value: object = None, error: Exception | None = None) -> None:
        self.value = value
        self.error = error

    def ready(self) -> bool:
        return True

    def get(self) -> object:
        if self.error is not None:
            raise self.error
        return self.value


class _FakeProcessPool:
    def __init__(self, results: list[_ReadyResult]) -> None:
        self.results = iter(results)
        self.close = Mock()
        self.terminate = Mock()
        self.join = Mock()

    def apply_async(self, _worker: object, _arguments: object) -> _ReadyResult:
        return next(self.results)


class FrontierGpuWorkerTests(unittest.TestCase):
    def test_partitions_screen_without_splitting_a_layer_across_gpus(self) -> None:
        samples = tuple(
            FrontierScreenSample(layer, expert, projection)
            for layer in range(8)
            for expert in range(2)
            for projection in ("w1", "w2", "w3")
        )

        batches = partition_frontier_screen_samples(
            samples,
            ("0", "1", "2", "3"),
            group_sizes=(128, 256, 512),
        )

        self.assertEqual(
            tuple(batch.physical_device for batch in batches), ("0", "1", "2", "3")
        )
        observed = [sample for batch in batches for sample in batch.samples]
        self.assertEqual(sorted(observed), sorted(samples))
        layer_owners: dict[int, set[str]] = {}
        for batch in batches:
            for sample in batch.samples:
                layer_owners.setdefault(sample.layer, set()).add(batch.physical_device)
        self.assertTrue(all(len(owners) == 1 for owners in layer_owners.values()))
        workload_units = [batch.workload_units for batch in batches]
        self.assertEqual(max(workload_units), min(workload_units))

    def test_partitions_conversion_shards_exactly_once(self) -> None:
        shards = tuple(f"model-{index:05d}.safetensors" for index in range(11))

        batches = partition_frontier_conversion_shards(
            shards,
            ("0", "1", "2", "3"),
        )

        observed = [shard for batch in batches for shard in batch.shard_names]
        self.assertEqual(sorted(observed), sorted(shards))
        self.assertEqual(len(observed), len(set(observed)))
        shard_counts = [len(batch.shard_names) for batch in batches]
        self.assertLessEqual(max(shard_counts) - min(shard_counts), 1)

    def test_rejects_ambiguous_or_duplicate_gpu_selectors(self) -> None:
        for devices in ((), ("0", "0"), ("0,1",), ("",)):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                validate_fixed_gpu_devices(devices)

    def test_sets_gpu_visibility_before_torch_import(self) -> None:
        observed_visibility: list[str | None] = []
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 1

        def import_torch(_module_name: str) -> Mock:
            observed_visibility.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
            return torch

        with patch(
            "deepseek_v4_lowbit.frontier_gpu_workers.importlib.import_module",
            side_effect=import_torch,
        ):
            _select_one_physical_gpu("GPU-fixed-selector")

        self.assertEqual(observed_visibility, ["GPU-fixed-selector"])

    def test_gpu_inspect_cli_forwards_every_fixed_selector(self) -> None:
        inspections = tuple(
            FixedGpuRuntimeInspection(
                physical_device=str(index),
                device_name="NVIDIA A100-SXM4-80GB",
                compute_capability=(8, 0),
                total_memory_bytes=80_000_000_000,
            )
            for index in range(4)
        )
        with patch(
            "deepseek_v4_lowbit.frontier_gpu_inspect_cli.inspect_fixed_gpu_runtime",
            return_value=inspections,
        ) as inspect:
            status = gpu_inspect_main(
                [
                    "--gpu-device",
                    "0",
                    "--gpu-device",
                    "1",
                    "--gpu-device",
                    "2",
                    "--gpu-device",
                    "3",
                ]
            )

        self.assertEqual(status, 0)
        inspect.assert_called_once_with(("0", "1", "2", "3"))

    def test_screen_cli_forwards_every_fixed_gpu_selector(self) -> None:
        campaign = Mock()
        campaign.run.return_value = (
            Path("pilot.json"),
            Path("boundary.json"),
            Path("full.json"),
        )
        with patch(
            "deepseek_v4_lowbit.frontier_campaign_cli.FrontierScreenCampaign",
            return_value=campaign,
        ) as campaign_type:
            status = frontier_campaign_main(
                [
                    "source",
                    "imatrix.dat",
                    "baseline.json",
                    "source-report.json",
                    "headers.json",
                    "output",
                    "--gpu-device",
                    "0",
                    "--gpu-device",
                    "1",
                    "--gpu-device",
                    "2",
                    "--gpu-device",
                    "3",
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(
            campaign_type.call_args.kwargs["gpu_devices"], ("0", "1", "2", "3")
        )

    def test_conversion_batch_forwards_every_fixed_gpu_selector(self) -> None:
        publisher = Mock()
        publisher.candidate_names = ("quality",)
        publisher.resume_conversion_state.return_value = ((), None)
        with tempfile.TemporaryDirectory() as raw:
            recipe_bundle_path = Path(raw) / "recipe.json"
            recipe_bundle_path.write_text(json.dumps({}), encoding="utf-8")
            with patch(
                "deepseek_v4_lowbit.frontier_batch.convert_frontier_candidates",
                return_value=(),
            ) as convert:
                converted = run_frontier_conversion_batch(
                    Path("source"),
                    Path("output"),
                    recipe_bundle_path,
                    Path("imatrix.dat"),
                    Path("baseline"),
                    device="cuda",
                    publisher=publisher,
                    gpu_devices=("0", "1", "2", "3"),
                )

        self.assertEqual(converted, ())
        self.assertEqual(
            convert.call_args.kwargs["gpu_devices"],
            ("0", "1", "2", "3"),
        )

    def test_rejects_duplicate_conversion_worker_shards(self) -> None:
        receipt = ShardReceipt(
            shard_name="model-00001.safetensors",
            identity=ShardIdentity(source_sha256="s", recipe_sha256="r"),
            output_path=Path("output"),
            output_sha256="a" * 64,
            output_bytes=1,
            tensors={},
            metadata={},
        )
        result = ShardTransformResult(receipt=receipt, metrics=(), resumed=False)
        with (
            patch(
                "deepseek_v4_lowbit.frontier_gpu_workers._run_fixed_gpu_batches",
                return_value=((result, result),),
            ),
            self.assertRaisesRegex(RuntimeError, "duplicate shards"),
        ):
            transform_frontier_shards_on_fixed_gpus(
                Path("source"),
                Path("output"),
                ("model-00001.safetensors",),
                recipe=ArtifactRecipe(default=LayerQuantization(2, 2, 128)),
                options=TransformOptions(device="cuda"),
                imatrix_path=Path("imatrix.dat"),
                gpu_devices=("0",),
            )

    def test_cleans_created_pools_when_later_pool_construction_fails(self) -> None:
        first_pool = _FakeProcessPool([])
        calls = 0

        def pool_factory(_size: int) -> _FakeProcessPool:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("pool construction failed")
            return first_pool

        with self.assertRaisesRegex(RuntimeError, "pool construction failed"):
            _run_fixed_gpu_batches(
                lambda task: task,
                ("first", "second"),
                pool_factory=pool_factory,
            )

        first_pool.terminate.assert_called_once_with()
        first_pool.join.assert_called_once_with()

    def test_terminates_remaining_workers_after_one_failure(self) -> None:
        pools = [
            _FakeProcessPool([_ReadyResult(value="first")]),
            _FakeProcessPool([_ReadyResult(error=RuntimeError("gpu worker failed"))]),
        ]
        pending_pools = iter(pools)
        requested_pool_sizes: list[int] = []

        def pool_factory(size: int) -> _FakeProcessPool:
            requested_pool_sizes.append(size)
            return next(pending_pools)

        with self.assertRaisesRegex(RuntimeError, "gpu worker failed"):
            _run_fixed_gpu_batches(
                lambda task: task,
                ("first", "second"),
                pool_factory=pool_factory,
            )

        self.assertEqual(requested_pool_sizes, [1, 1])
        for pool in pools:
            pool.close.assert_called_once_with()
            pool.terminate.assert_called_once_with()
            pool.join.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
