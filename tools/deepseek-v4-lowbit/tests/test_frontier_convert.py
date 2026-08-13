from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepseek_v4_lowbit.frontier_convert import (
    artifact_recipe_from_payload,
    load_baseline_artifact_reuse,
    map_source_shards_to_routed_layers,
    validate_frontier_candidate_names,
)
from deepseek_v4_lowbit.frontier_convert_cli import main as convert_frontier_main
from deepseek_v4_lowbit.shard_writer import file_sha256

_HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


class FrontierConvertContractTests(unittest.TestCase):
    def test_accepts_one_quality_first_candidate(self) -> None:
        self.assertEqual(
            validate_frontier_candidate_names(("quality",)),
            ("quality",),
        )

    def test_rejects_non_nested_candidate_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "nested frontier order"):
            validate_frontier_candidate_names(("quality", "balanced"))

    def test_rejects_empty_or_unknown_candidate_selection(self) -> None:
        for names in ((), ("quality", "quality"), ("unknown",)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_frontier_candidate_names(names)

    def test_standalone_converter_requires_one_explicit_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bundle_path = root / "recipe.json"
            bundle_path.write_text("{}\n", encoding="utf-8")
            imatrix_path = root / "imatrix.dat"
            imatrix_path.write_bytes(b"imatrix")
            with (
                patch(
                    "deepseek_v4_lowbit.frontier_convert_cli.load_json_object",
                    return_value={},
                ),
                patch(
                    "deepseek_v4_lowbit.frontier_convert_cli.convert_frontier_candidates",
                    return_value=(),
                ) as convert,
            ):
                status = convert_frontier_main(
                    [
                        str(root / "source"),
                        str(root / "output"),
                        str(bundle_path),
                        str(imatrix_path),
                        "--baseline-directory",
                        str(root / "baseline"),
                        "--candidate",
                        "quality",
                    ]
                )

        self.assertEqual(status, 0)
        self.assertEqual(convert.call_args.kwargs["candidate_names"], ("quality",))

    def test_maps_one_routed_layer_to_its_source_shard(self) -> None:
        source_weight_map = {
            "layers.0.ffn.experts.0.w1.weight": "model-00002.safetensors",
            "layers.0.ffn.experts.0.w1.scale": "model-00002.safetensors",
            "layers.1.ffn.experts.0.w2.weight": "model-00003.safetensors",
            "layers.0.input_layernorm.weight": "model-00002.safetensors",
        }

        self.assertEqual(
            map_source_shards_to_routed_layers(source_weight_map),
            {
                "model-00002.safetensors": frozenset({0}),
                "model-00003.safetensors": frozenset({1}),
            },
        )

    def test_parses_explicit_43_layer_frontier_recipe(self) -> None:
        payload = {
            "default": {"w13_bits": 2, "w2_bits": 2, "group_size": 512},
            "layers": {
                str(layer): {
                    "w13_bits": 2,
                    "w2_bits": 4 if layer == 42 else 2,
                    "group_size": 128 if layer == 42 else 512,
                }
                for layer in range(43)
            },
        }

        recipe = artifact_recipe_from_payload(payload)

        self.assertEqual(recipe.group_size_for(0, "w1", fallback=128), 512)
        self.assertEqual(recipe.group_size_for(0, "w2", fallback=128), 512)
        self.assertEqual(recipe.group_size_for(42, "w1", fallback=128), 128)
        self.assertEqual(recipe.group_size_for(42, "w2", fallback=128), 128)
        self.assertEqual(recipe.bits_for(42, "w2"), 4)

    def test_parses_projection_specific_frontier_recipe(self) -> None:
        payload = {
            "default": {
                "w13_bits": 2,
                "w2_bits": 2,
                "w13_group_size": 512,
                "w2_group_size": 256,
            },
            "layers": {
                str(layer): {
                    "w13_bits": 2,
                    "w2_bits": 4 if layer == 42 else 2,
                    "w13_group_size": 128 if layer == 42 else 512,
                    "w2_group_size": 128,
                }
                for layer in range(43)
            },
        }

        recipe = artifact_recipe_from_payload(payload)

        self.assertEqual(recipe.group_size_for(0, "w1", fallback=128), 512)
        self.assertEqual(recipe.group_size_for(0, "w2", fallback=128), 128)
        self.assertEqual(recipe.group_size_for(42, "w1", fallback=128), 128)
        self.assertEqual(recipe.bits_for(42, "w2"), 4)

    @unittest.skipUnless(_HAS_SAFETENSORS, "requires safetensors")
    def test_imports_only_checksum_bound_baseline_shard(self) -> None:
        import torch
        from safetensors.torch import save_file

        shard_name = "model-00001-of-00001.safetensors"
        tensor_name = "layers.0.ffn.experts.0.w1.weight_packed"
        expected_weight_map = {tensor_name: shard_name}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.safetensors"
            save_file({"source": torch.ones(1)}, source_path)
            shard_path = root / shard_name
            save_file({tensor_name: torch.ones((1, 1), dtype=torch.int32)}, shard_path)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": expected_weight_map}),
                encoding="utf-8",
            )
            recipe_sha256 = "a" * 64
            (root / "conversion-metrics.json").write_text(
                json.dumps(
                    {
                        "recipe_sha256": recipe_sha256,
                        "shards": [
                            {
                                "shard": shard_name,
                                "output_sha256": file_sha256(shard_path),
                                "metrics": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "deepseek_v4_lowbit.frontier_convert._BASELINE_CONFIG_SHA256",
                    file_sha256(root / "config.json"),
                ),
                patch(
                    "deepseek_v4_lowbit.frontier_convert._BASELINE_INDEX_SHA256",
                    file_sha256(root / "model.safetensors.index.json"),
                ),
                patch(
                    "deepseek_v4_lowbit.frontier_convert._BASELINE_METRICS_SHA256",
                    file_sha256(root / "conversion-metrics.json"),
                ),
            ):
                baseline = load_baseline_artifact_reuse(
                    root,
                    expected_weight_map,
                    recipe_sha256,
                )
                receipt = baseline.shard_receipt(shard_name, source_path)

            self.assertIsNotNone(receipt)
            assert receipt is not None
            self.assertEqual(receipt.output_sha256, file_sha256(shard_path))
            self.assertEqual(set(receipt.tensors), {tensor_name})

    def test_rejects_implicit_missing_layers(self) -> None:
        payload = {
            "default": {"w13_bits": 2, "w2_bits": 2, "group_size": 512},
            "layers": {},
        }

        with self.assertRaisesRegex(ValueError, "explicitly configure all 43"):
            artifact_recipe_from_payload(payload)


if __name__ == "__main__":
    unittest.main()
