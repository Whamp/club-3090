from __future__ import annotations

import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_HAS_WRITER_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("safetensors") is not None
)


@unittest.skipUnless(_HAS_WRITER_DEPS, "requires torch and safetensors")
class ResumableShardWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.output_directory = Path(self.temporary_directory.name)

    def test_recipe_fingerprint_is_canonical(self) -> None:
        from deepseek_v4_lowbit.shard_writer import canonical_json_sha256

        first = canonical_json_sha256({"default": {"w2_bits": 4, "w13_bits": 2}})
        second = canonical_json_sha256({"default": {"w13_bits": 2, "w2_bits": 4}})

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_writes_verified_shard_receipt_and_final_index(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.shard_writer import (
            ResumableSafetensorsWriter,
            ShardIdentity,
        )

        writer = ResumableSafetensorsWriter(self.output_directory)
        identity = ShardIdentity(source_sha256="source-a", recipe_sha256="recipe-a")
        tensors = {
            "model.layers.0.weight_packed": torch.tensor(
                [[0x12345678]], dtype=torch.int32
            ),
            "model.layers.0.weight_scale": torch.tensor([[0.5]], dtype=torch.float16),
        }

        receipt = writer.write_shard(
            "model-00001-of-00001.safetensors", tensors, identity
        )
        resumed = writer.completed_shard("model-00001-of-00001.safetensors", identity)
        index_path = writer.finalize_index(
            ["model-00001-of-00001.safetensors"],
            expected_weight_map={
                "model.layers.0.weight_packed": "model-00001-of-00001.safetensors",
                "model.layers.0.weight_scale": "model-00001-of-00001.safetensors",
            },
        )

        self.assertEqual(resumed, receipt)
        self.assertEqual(receipt.output_bytes, receipt.output_path.stat().st_size)
        self.assertEqual(len(receipt.output_sha256), 64)
        index = json.loads(index_path.read_text())
        self.assertEqual(index["metadata"]["total_size"], 6)
        self.assertEqual(
            index["weight_map"],
            {
                "model.layers.0.weight_packed": "model-00001-of-00001.safetensors",
                "model.layers.0.weight_scale": "model-00001-of-00001.safetensors",
            },
        )

    def test_recovers_final_shard_from_receipt_rename_crash_window(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.shard_writer import (
            ResumableSafetensorsWriter,
            ShardIdentity,
        )

        writer = ResumableSafetensorsWriter(self.output_directory)
        shard_name = "model-00001-of-00001.safetensors"
        identity = ShardIdentity(source_sha256="source-a", recipe_sha256="recipe-a")
        expected = writer.write_shard(shard_name, {"weight": torch.ones(1)}, identity)
        receipt = (
            self.output_directory
            / ".conversion-state"
            / "receipts"
            / f"{shard_name}.json"
        )
        partial_receipt = receipt.with_name(f"{shard_name}.partial.json")
        receipt.rename(partial_receipt)

        recovered = writer.completed_shard(shard_name, identity)

        self.assertEqual(recovered, expected)
        self.assertTrue(receipt.is_file())
        self.assertFalse(partial_receipt.exists())

    def test_reuses_verified_shard_with_new_recipe_identity(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.shard_writer import (
            ResumableSafetensorsWriter,
            ShardIdentity,
        )

        source_directory = self.output_directory / "source"
        candidate_directory = self.output_directory / "candidate"
        shard_name = "model-00001-of-00001.safetensors"
        source_writer = ResumableSafetensorsWriter(source_directory)
        source_receipt = source_writer.write_shard(
            shard_name,
            {"weight": torch.ones(1)},
            ShardIdentity(source_sha256="source-a", recipe_sha256="recipe-a"),
        )
        candidate_writer = ResumableSafetensorsWriter(candidate_directory)
        candidate_identity = ShardIdentity(
            source_sha256="source-a",
            recipe_sha256="recipe-b",
        )

        receipt = candidate_writer.reuse_shard(
            shard_name,
            source_receipt,
            candidate_identity,
        )

        self.assertEqual(receipt.identity, candidate_identity)
        self.assertEqual(receipt.output_sha256, source_receipt.output_sha256)
        self.assertEqual(
            receipt.output_path.stat().st_ino,
            source_receipt.output_path.stat().st_ino,
        )
        self.assertEqual(
            candidate_writer.completed_shard(shard_name, candidate_identity),
            receipt,
        )

    def test_rejects_recipe_change_for_completed_shard(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.shard_writer import (
            ResumableSafetensorsWriter,
            ResumeConflictError,
            ShardIdentity,
        )

        writer = ResumableSafetensorsWriter(self.output_directory)
        shard_name = "model-00001-of-00001.safetensors"
        writer.write_shard(
            shard_name,
            {"weight": torch.ones(1)},
            ShardIdentity(source_sha256="source-a", recipe_sha256="recipe-a"),
        )

        with self.assertRaisesRegex(ResumeConflictError, "recipe fingerprint"):
            writer.completed_shard(
                shard_name,
                ShardIdentity(source_sha256="source-a", recipe_sha256="recipe-b"),
            )

    def test_rejects_corrupted_completed_shard(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.shard_writer import (
            ResumableSafetensorsWriter,
            ResumeConflictError,
            ShardIdentity,
        )

        writer = ResumableSafetensorsWriter(self.output_directory)
        shard_name = "model-00001-of-00001.safetensors"
        identity = ShardIdentity(source_sha256="source-a", recipe_sha256="recipe-a")
        receipt = writer.write_shard(shard_name, {"weight": torch.ones(1)}, identity)
        receipt.output_path.write_bytes(b"corrupt")

        with self.assertRaisesRegex(ResumeConflictError, "checksum"):
            writer.completed_shard(shard_name, identity)

    def test_rejects_receipt_inventory_that_disagrees_with_shard_header(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.shard_writer import (
            ResumableSafetensorsWriter,
            ResumeConflictError,
            ShardIdentity,
        )

        writer = ResumableSafetensorsWriter(self.output_directory)
        shard_name = "model-00001-of-00001.safetensors"
        identity = ShardIdentity(source_sha256="source-a", recipe_sha256="recipe-a")
        writer.write_shard(shard_name, {"weight": torch.ones(1)}, identity)
        receipt_path = (
            self.output_directory
            / ".conversion-state"
            / "receipts"
            / f"{shard_name}.json"
        )
        receipt = json.loads(receipt_path.read_text())
        receipt["tensors"]["weight"]["shape"] = [999]
        receipt_path.write_text(json.dumps(receipt))

        with self.assertRaisesRegex(ResumeConflictError, "tensor inventory"):
            writer.completed_shard(shard_name, identity)

    def test_rejects_final_inventory_that_differs_from_expected_mapping(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.shard_writer import (
            ResumableSafetensorsWriter,
            ResumeConflictError,
            ShardIdentity,
        )

        writer = ResumableSafetensorsWriter(self.output_directory)
        shard_name = "model-00001-of-00001.safetensors"
        identity = ShardIdentity(source_sha256="source-a", recipe_sha256="recipe-a")
        writer.write_shard(shard_name, {"weight": torch.ones(1)}, identity)

        mismatches = {
            "missing tensor": {"weight": shard_name, "missing": shard_name},
            "unexpected tensor": {},
        }
        for mismatch, expected_weight_map in mismatches.items():
            with self.subTest(mismatch=mismatch):
                with self.assertRaisesRegex(
                    ResumeConflictError,
                    "final tensor inventory does not match expected output",
                ):
                    writer.finalize_index(
                        [shard_name],
                        expected_weight_map=expected_weight_map,
                    )
                self.assertFalse(
                    (self.output_directory / "model.safetensors.index.json").exists()
                )

    def test_rejects_tensor_written_to_wrong_expected_shard(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.shard_writer import (
            ResumableSafetensorsWriter,
            ResumeConflictError,
            ShardIdentity,
        )

        writer = ResumableSafetensorsWriter(self.output_directory)
        identity = ShardIdentity(source_sha256="source-a", recipe_sha256="recipe-a")
        first = "model-00001-of-00002.safetensors"
        second = "model-00002-of-00002.safetensors"
        writer.write_shard(first, {"first_weight": torch.ones(1)}, identity)
        writer.write_shard(second, {"second_weight": torch.zeros(1)}, identity)

        with self.assertRaisesRegex(
            ResumeConflictError,
            "final tensor inventory does not match expected output",
        ):
            writer.finalize_index(
                [first, second],
                expected_weight_map={
                    "first_weight": second,
                    "second_weight": first,
                },
            )

    def test_rejects_duplicate_tensor_names_across_shards(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.shard_writer import (
            ResumableSafetensorsWriter,
            ResumeConflictError,
            ShardIdentity,
        )

        writer = ResumableSafetensorsWriter(self.output_directory)
        identity = ShardIdentity(source_sha256="source-a", recipe_sha256="recipe-a")
        first = "model-00001-of-00002.safetensors"
        second = "model-00002-of-00002.safetensors"
        writer.write_shard(first, {"weight": torch.ones(1)}, identity)
        writer.write_shard(second, {"weight": torch.zeros(1)}, identity)

        with self.assertRaisesRegex(ResumeConflictError, "duplicate tensor"):
            writer.finalize_index(
                [first, second],
                expected_weight_map={"weight": first},
            )


if __name__ == "__main__":
    unittest.main()
