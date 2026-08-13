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

    def test_emits_groups_for_distinct_bit_and_group_size_pairs(self) -> None:
        recipe = ArtifactRecipe(
            default=LayerQuantization(2, 2, group_size=512),
            layers={0: LayerQuantization(2, 4, group_size=128)},
        )

        config = build_compressed_tensors_config(
            recipe,
            layer_count=2,
            group_size=256,
        )

        groups = config["config_groups"]
        self.assertEqual(
            set(groups),
            {"group_w2_g128", "group_w2_g512", "group_w4_g128"},
        )
        self.assertEqual(groups["group_w2_g128"]["weights"]["group_size"], 128)
        self.assertEqual(groups["group_w4_g128"]["weights"]["group_size"], 128)
        self.assertEqual(groups["group_w2_g512"]["weights"]["group_size"], 512)
        self.assertEqual(
            groups["group_w4_g128"]["targets"],
            ["model.layers.0.ffn.experts.0.down_proj"],
        )

    def test_emits_projection_specific_group_sizes(self) -> None:
        recipe = ArtifactRecipe(
            default=LayerQuantization(
                2,
                4,
                w13_group_size=512,
                w2_group_size=128,
            )
        )

        config = build_compressed_tensors_config(
            recipe,
            layer_count=1,
            group_size=256,
        )

        groups = config["config_groups"]
        self.assertEqual(set(groups), {"group_w2_g512", "group_w4_g128"})
        self.assertEqual(
            groups["group_w2_g512"]["targets"],
            [
                "model.layers.0.ffn.experts.0.gate_proj",
                "model.layers.0.ffn.experts.0.up_proj",
            ],
        )
        self.assertEqual(
            groups["group_w4_g128"]["targets"],
            ["model.layers.0.ffn.experts.0.down_proj"],
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
        self.assertEqual(
            output["club_3090_lowbit"]["source_quantization_method"],
            "compressed-tensors",
        )
        self.assertEqual(
            output["club_3090_lowbit"]["source_checkpoint_quantization_method"],
            "fp8",
        )

    def test_pins_the_mixed_group_runtime_tree_in_model_metadata(self) -> None:
        source = {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 1,
        }

        output = materialize_model_config(
            source,
            ArtifactRecipe(default=LayerQuantization(2, 2)),
            group_size=128,
        )

        self.assertEqual(
            output["club_3090_lowbit"]["runtime_compatibility"],
            {
                "acceptance_status": "pending-sm86-oracle-and-deepswe",
                "base_repository": "haosdent/vllm",
                "base_revision": "12810046c799cbe874967e19b1c0fa134ab7b209",
                "integration_repository": "Whamp/vllm",
                "integration_revision": ("dd2d1fd6779addccc73094f77fa4ada7d9106a41"),
                "required_tree": "f73b30cc5a2ed9de200ca2e4de3cdef1a06f6538",
            },
        )


if __name__ == "__main__":
    unittest.main()
