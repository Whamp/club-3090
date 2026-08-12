from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deepseek_v4_lowbit.artifact_plan import (
    ArtifactRecipe,
    LayerQuantization,
    TensorDisposition,
    classify_tensor,
    load_tensor_headers,
    plan_artifact,
)


class TensorClassificationTests(unittest.TestCase):
    def test_classifies_main_routed_expert_projection(self) -> None:
        tensor = classify_tensor("layers.26.ffn.experts.17.w2.weight")

        self.assertEqual(tensor.disposition, TensorDisposition.QUANTIZE)
        self.assertEqual(tensor.layer, 26)
        self.assertEqual(tensor.expert, 17)
        self.assertEqual(tensor.projection, "w2")

    def test_omits_every_mtp_tensor(self) -> None:
        tensor = classify_tensor("mtp.2.ffn.experts.17.w2.weight")

        self.assertEqual(tensor.disposition, TensorDisposition.OMIT)

    def test_preserves_shared_experts(self) -> None:
        tensor = classify_tensor("layers.26.ffn.shared_experts.w2.weight")

        self.assertEqual(tensor.disposition, TensorDisposition.PRESERVE)


class ArtifactPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.headers = load_tensor_headers(
            _write_headers(
                Path(self.temp_dir.name),
                {
                    "model-00001-of-00001.safetensors": {
                        "layers.0.ffn.experts.0.w1.scale": _header(
                            "F8_E8M0", [2, 1], 2
                        ),
                        "layers.0.ffn.experts.0.w1.weight": _header("I8", [2, 64], 128),
                        "layers.0.ffn.experts.0.w2.scale": _header(
                            "F8_E8M0", [4, 1], 4
                        ),
                        "layers.0.ffn.experts.0.w2.weight": _header("I8", [4, 64], 256),
                        "layers.0.ffn.experts.0.w3.scale": _header(
                            "F8_E8M0", [2, 1], 2
                        ),
                        "layers.0.ffn.experts.0.w3.weight": _header("I8", [2, 64], 128),
                        "layers.0.attn.q_norm.weight": _header("BF16", [4], 8),
                        "mtp.1.norm.weight": _header("BF16", [4], 8),
                    }
                },
            )
        )

    def test_replaces_source_expert_weights_and_scales(self) -> None:
        recipe = ArtifactRecipe(default=LayerQuantization(w13_bits=2, w2_bits=4))

        plan = plan_artifact(self.headers, recipe, group_size=128)

        # w1/w3: each 2 rows * eight int32 words + one FP16 scale per row.
        # w2: 4 rows * sixteen int32 words + one FP16 scale per row.
        # Every replacement also stores its logical [output, input] shape as INT64.
        self.assertEqual(plan.quantized_weight_bytes, 384)
        self.assertEqual(plan.quantized_scale_bytes, 16)
        self.assertEqual(plan.quantized_shape_bytes, 48)
        self.assertEqual(plan.preserved_bytes, 8)
        self.assertEqual(plan.omitted_bytes, 8)
        self.assertEqual(plan.total_bytes, 456)
        self.assertEqual(plan.quantized_tensor_count, 3)

    def test_layer_override_changes_only_selected_layer(self) -> None:
        recipe = ArtifactRecipe(
            default=LayerQuantization(w13_bits=2, w2_bits=2),
            layers={0: LayerQuantization(w13_bits=4, w2_bits=2)},
        )

        plan = plan_artifact(self.headers, recipe, group_size=128)

        self.assertEqual(plan.quantized_weight_bytes, 384)

    def test_rejects_w3_when_runtime_dimension_is_not_packable(self) -> None:
        recipe = ArtifactRecipe(default=LayerQuantization(w13_bits=3, w2_bits=2))

        message = "not divisible by Humming pack factor 10"
        with self.assertRaisesRegex(ValueError, message):
            plan_artifact(self.headers, recipe, group_size=128)


def _header(dtype: str, shape: list[int], byte_count: int) -> dict[str, object]:
    return {"dtype": dtype, "shape": shape, "data_offsets": [0, byte_count]}


def _write_headers(directory: Path, headers: dict[str, object]) -> Path:
    path = directory / "headers.json"
    path.write_text(json.dumps(headers), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
