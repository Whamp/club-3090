from __future__ import annotations

import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from deepseek_v4_lowbit.convert_cli import (
    build_expected_output_weight_map,
    load_source_shards,
    main,
    select_output_shards,
)

_HAS_FULL_TOOLCHAIN = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in (
        "torch",
        "safetensors",
        "compressed_tensors",
        "auto_round",
        "auto_round_extension",
    )
)


class ConvertCliTests(unittest.TestCase):
    def test_builds_exact_transformed_output_mapping(self) -> None:
        shard_name = "model-00001-of-00001.safetensors"
        routed_weight = "layers.0.ffn.experts.7.w2.weight"
        routed_scale = "layers.0.ffn.experts.7.w2.scale"
        source_weight_map = {
            "layers.0.ffn.shared_experts.w1.weight": shard_name,
            "mtp.0.weight": shard_name,
            routed_scale: shard_name,
            routed_weight: shard_name,
        }

        self.assertEqual(
            build_expected_output_weight_map(source_weight_map),
            {
                "layers.0.ffn.experts.7.w2.weight_packed": shard_name,
                "layers.0.ffn.experts.7.w2.weight_scale": shard_name,
                "layers.0.ffn.experts.7.w2.weight_shape": shard_name,
                "layers.0.ffn.shared_experts.w1.weight": shard_name,
            },
        )

    def test_excludes_mtp_only_source_shards_from_output_conversion(self) -> None:
        source_shards = (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        )
        expected_weight_map = {
            "layers.0.input_layernorm.weight": source_shards[0],
        }

        self.assertEqual(
            select_output_shards(source_shards, expected_weight_map),
            (source_shards[0],),
        )

    def test_loads_sorted_existing_shards_from_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in ("model-00002.safetensors", "model-00001.safetensors"):
                (root / name).touch()
            index_path = root / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "weight_b": "model-00002.safetensors",
                            "weight_a": "model-00001.safetensors",
                        }
                    }
                )
            )

            self.assertEqual(
                load_source_shards(index_path),
                ("model-00001.safetensors", "model-00002.safetensors"),
            )

    @unittest.skipUnless(
        _HAS_FULL_TOOLCHAIN,
        "requires the combined pinned conversion toolchain",
    )
    def test_finalizes_tiny_mtp_free_artifact(self) -> None:
        torch = importlib.import_module("torch")
        safetensors_torch = importlib.import_module("safetensors.torch")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            shard_name = "model-00001-of-00002.safetensors"
            mtp_shard_name = "model-00002-of-00002.safetensors"
            weight_name = "layers.0.ffn.experts.0.w1.weight"
            scale_name = weight_name.removesuffix(".weight") + ".scale"
            safetensors_torch.save_file(
                {
                    weight_name: torch.full((2, 16), 0x11, dtype=torch.uint8),
                    scale_name: torch.full((1, 1), 127, dtype=torch.uint8),
                },
                source / shard_name,
            )
            safetensors_torch.save_file(
                {"mtp.0.weight": torch.ones(1)},
                source / mtp_shard_name,
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            weight_name: shard_name,
                            scale_name: shard_name,
                            "mtp.0.weight": mtp_shard_name,
                        }
                    }
                )
            )
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "deepseek_v4",
                        "num_hidden_layers": 1,
                        "num_nextn_predict_layers": 1,
                        "quantization_config": {"quant_method": "fp8"},
                    }
                )
            )
            (source / "tokenizer.json").write_text("{}")
            recipe = root / "recipe.json"
            recipe.write_text(json.dumps({"default": {"w13_bits": 2, "w2_bits": 2}}))

            exit_code = main(
                [
                    str(source),
                    str(output),
                    str(recipe),
                    "--group-size",
                    "16",
                ]
            )

            self.assertEqual(exit_code, 0)
            output_config = json.loads((output / "config.json").read_text())
            self.assertEqual(output_config["num_nextn_predict_layers"], 0)
            self.assertEqual(
                output_config["quantization_config"]["quant_method"],
                "compressed-tensors",
            )
            self.assertTrue((output / "model.safetensors.index.json").is_file())
            self.assertTrue((output / "conversion-metrics.json").is_file())
            self.assertEqual((output / "tokenizer.json").read_text(), "{}")
            output_weights = safetensors_torch.load_file(output / shard_name)
            self.assertFalse((output / mtp_shard_name).exists())
            self.assertNotIn("mtp.0.weight", output_weights)
            self.assertIn(
                "layers.0.ffn.experts.0.w1.weight_packed",
                output_weights,
            )


if __name__ == "__main__":
    unittest.main()
