from __future__ import annotations

import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_HAS_TRANSFORM_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("safetensors") is not None
    and importlib.util.find_spec("compressed_tensors") is not None
)
_HAS_FULL_TOOLCHAIN = (
    _HAS_TRANSFORM_DEPS
    and importlib.util.find_spec("auto_round") is not None
    and importlib.util.find_spec("auto_round_extension") is not None
)


@unittest.skipUnless(
    _HAS_TRANSFORM_DEPS,
    "requires torch, safetensors, and pinned compressed-tensors",
)
class SourceShardTransformTests(unittest.TestCase):
    def test_streams_preserved_and_quantized_tensors_into_verified_shard(self) -> None:
        torch = importlib.import_module("torch")
        safetensors_torch = importlib.import_module("safetensors.torch")

        from deepseek_v4_lowbit.artifact_plan import (
            ArtifactRecipe,
            LayerQuantization,
        )
        from deepseek_v4_lowbit.quantizer import QuantizedTensor
        from deepseek_v4_lowbit.shard_writer import ResumableSafetensorsWriter
        from deepseek_v4_lowbit.source_transform import (
            QuantizerKind,
            TransformOptions,
            transform_source_shard,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.safetensors"
            output_directory = root / "output"
            weight_name = "layers.0.ffn.experts.7.w1.weight"
            scale_name = "layers.0.ffn.experts.7.w1.scale"
            safetensors_torch.save_file(
                {
                    weight_name: torch.zeros((1, 8), dtype=torch.int8),
                    scale_name: torch.full((1, 1), 127, dtype=torch.uint8),
                    "layers.0.input_layernorm.weight": torch.tensor([1.25]),
                    "mtp.layers.0.weight": torch.tensor([9.0]),
                },
                source_path,
            )
            recipe = ArtifactRecipe(default=LayerQuantization(2, 2))
            options = TransformOptions(
                group_size=16,
                quantizer=QuantizerKind.PLAIN_RTN,
            )
            writer = ResumableSafetensorsWriter(output_directory)

            def fake_dequantize(*_args, **_kwargs):
                return torch.tensor(
                    [[-2.0, -1.0, 0.0, 1.0] * 4],
                    dtype=torch.bfloat16,
                )

            def fake_quantize(weight, **_kwargs):
                codes = torch.tensor(
                    [[-2, -1, 0, 1] * 4],
                    dtype=torch.int8,
                    device=weight.device,
                )
                scales = torch.tensor(
                    [[1.0]], dtype=torch.float16, device=weight.device
                )
                return QuantizedTensor(
                    codes=codes,
                    scales=scales,
                    dequantized=codes.to(torch.float32),
                    unweighted_error=0.0,
                    weighted_error=0.0,
                )

            with (
                patch(
                    "deepseek_v4_lowbit.source_transform.dequantize_routed_expert_weight",
                    side_effect=fake_dequantize,
                ),
                patch(
                    "deepseek_v4_lowbit.source_transform.quantize_symmetric",
                    side_effect=fake_quantize,
                ),
            ):
                result = transform_source_shard(
                    source_path,
                    "model-00001-of-00001.safetensors",
                    writer=writer,
                    recipe=recipe,
                    options=options,
                )

            output = safetensors_torch.load_file(result.receipt.output_path)
            self.assertFalse(result.resumed)
            self.assertEqual(len(result.metrics), 1)
            self.assertEqual(result.metrics[0].tensor_name, weight_name)
            self.assertEqual(result.metrics[0].group_size, 16)
            self.assertEqual(
                set(output),
                {
                    "layers.0.ffn.experts.7.w1.weight_packed",
                    "layers.0.ffn.experts.7.w1.weight_scale",
                    "layers.0.ffn.experts.7.w1.weight_shape",
                    "layers.0.input_layernorm.weight",
                },
            )
            torch.testing.assert_close(
                output["layers.0.input_layernorm.weight"],
                torch.tensor([1.25]),
            )

            resumed = transform_source_shard(
                source_path,
                "model-00001-of-00001.safetensors",
                writer=writer,
                recipe=recipe,
                options=options,
            )
            self.assertTrue(resumed.resumed)
            self.assertEqual(resumed.metrics, result.metrics)

            receipt_path = (
                output_directory
                / ".conversion-state"
                / "receipts"
                / "model-00001-of-00001.safetensors.json"
            )
            receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            del receipt_payload["metadata"]["transform_metrics"][0]["group_size"]
            receipt_path.write_text(json.dumps(receipt_payload), encoding="utf-8")
            legacy_resumed = transform_source_shard(
                source_path,
                "model-00001-of-00001.safetensors",
                writer=writer,
                recipe=recipe,
                options=options,
            )
            self.assertEqual(legacy_resumed.metrics[0].group_size, 16)

    @unittest.skipUnless(
        _HAS_FULL_TOOLCHAIN,
        "requires the combined pinned conversion toolchain",
    )
    def test_real_toolchain_transforms_synthetic_mxfp4_weight(self) -> None:
        torch = importlib.import_module("torch")
        safetensors_torch = importlib.import_module("safetensors.torch")

        from deepseek_v4_lowbit.artifact_plan import (
            ArtifactRecipe,
            LayerQuantization,
        )
        from deepseek_v4_lowbit.shard_writer import ResumableSafetensorsWriter
        from deepseek_v4_lowbit.source_transform import (
            QuantizerKind,
            TransformOptions,
            transform_source_shard,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.safetensors"
            weight_name = "layers.0.ffn.experts.7.w1.weight"
            safetensors_torch.save_file(
                {
                    weight_name: torch.full((2, 16), 0x11, dtype=torch.uint8),
                    weight_name.removesuffix(".weight") + ".scale": torch.full(
                        (1, 1), 127, dtype=torch.uint8
                    ),
                },
                source_path,
            )

            result = transform_source_shard(
                source_path,
                "model-00001-of-00001.safetensors",
                writer=ResumableSafetensorsWriter(root / "output"),
                recipe=ArtifactRecipe(default=LayerQuantization(2, 2)),
                options=TransformOptions(
                    group_size=16,
                    quantizer=QuantizerKind.PLAIN_RTN,
                ),
            )

            output = safetensors_torch.load_file(result.receipt.output_path)
            self.assertEqual(output[f"{weight_name[:-7]}.weight_packed"].shape, (2, 2))
            self.assertEqual(output[f"{weight_name[:-7]}.weight_scale"].shape, (2, 2))
            self.assertEqual(len(result.metrics), 1)
            self.assertTrue(result.metrics[0].unweighted_error >= 0.0)


if __name__ == "__main__":
    unittest.main()
