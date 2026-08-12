from __future__ import annotations

import unittest

from deepseek_v4_lowbit.artifact_plan import ArtifactRecipe, LayerQuantization
from deepseek_v4_lowbit.model_config import (
    build_compressed_tensors_config,
    materialize_model_config,
)


class ModelConfigTests(unittest.TestCase):
    def test_emits_projection_specific_exact_vllm_targets(self) -> None:
        recipe = ArtifactRecipe(
            default=LayerQuantization(2, 2),
            layers={0: LayerQuantization(2, 4)},
        )

        config = build_compressed_tensors_config(
            recipe,
            layer_count=2,
            group_size=128,
        )

        groups = config["config_groups"]
        self.assertEqual(config["quant_method"], "compressed-tensors")
        self.assertEqual(config["format"], "pack-quantized")
        self.assertEqual(groups["group_w2"]["weights"]["num_bits"], 2)
        self.assertEqual(groups["group_w4"]["weights"]["num_bits"], 4)
        self.assertEqual(
            groups["group_w4"]["targets"],
            ["model.layers.0.ffn.experts.0.down_proj"],
        )
        self.assertIn(
            "model.layers.0.ffn.experts.0.gate_proj",
            groups["group_w2"]["targets"],
        )
        self.assertIn(
            "model.layers.1.ffn.experts.0.down_proj",
            groups["group_w2"]["targets"],
        )

    def test_routes_preserved_fp8_linears_through_deepseek_native_config(
        self,
    ) -> None:
        recipe = ArtifactRecipe(default=LayerQuantization(2, 2))

        config = build_compressed_tensors_config(
            recipe,
            layer_count=2,
            group_size=128,
        )

        groups = config["config_groups"]
        self.assertEqual(list(groups), ["group_w2"])
        self.assertEqual(config["base_quant_method"], "deepseek_v4_fp8")
        self.assertNotIn("Linear", groups["group_w2"]["targets"])

    def test_replaces_source_fp8_metadata_and_disables_mtp(self) -> None:
        source = {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "quantization_config": {"quant_method": "fp8"},
        }
        recipe = ArtifactRecipe(default=LayerQuantization(2, 2))

        output = materialize_model_config(source, recipe, group_size=128)

        self.assertEqual(source["num_nextn_predict_layers"], 1)
        self.assertEqual(output["num_nextn_predict_layers"], 0)
        self.assertEqual(
            output["quantization_config"]["quant_method"],
            "compressed-tensors",
        )
        self.assertEqual(
            output["club_3090_lowbit"]["mtp"],
            "omitted",
        )


if __name__ == "__main__":
    unittest.main()
