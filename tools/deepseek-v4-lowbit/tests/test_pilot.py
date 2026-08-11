from __future__ import annotations

import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepseek_v4_lowbit.pilot import PilotOptions, PilotSample, compare_quantizers
from deepseek_v4_lowbit.pilot_cli import _expand_samples

_HAS_PILOT_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("torch", "safetensors", "compressed_tensors", "auto_round")
)


class PilotCliTests(unittest.TestCase):
    def test_expands_layer_expert_samples_across_projections(self) -> None:
        samples = _expand_samples(["26:17", "0:0"], ["w1", "w2"])

        self.assertEqual(
            samples,
            (
                PilotSample(0, 0, "w1"),
                PilotSample(0, 0, "w2"),
                PilotSample(26, 17, "w1"),
                PilotSample(26, 17, "w2"),
            ),
        )

    @unittest.skipUnless(_HAS_PILOT_DEPS, "requires combined conversion toolchain")
    def test_compares_both_quantizers_for_selected_weight(self) -> None:
        torch = importlib.import_module("torch")
        safetensors_torch = importlib.import_module("safetensors.torch")

        class FakeImatrix:
            @staticmethod
            def expert_vector(*_args, input_columns, **_kwargs):
                return (1.0,) * input_columns

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shard_name = "model-00001.safetensors"
            tensor_name = "layers.0.ffn.experts.0.w1.weight"
            scale_name = tensor_name.removesuffix(".weight") + ".scale"
            safetensors_torch.save_file(
                {
                    tensor_name: torch.zeros((1, 8), dtype=torch.uint8),
                    scale_name: torch.full((1, 1), 127, dtype=torch.uint8),
                },
                root / shard_name,
            )

            with patch(
                "deepseek_v4_lowbit.pilot.dequantize_routed_expert_weight",
                return_value=torch.tensor(
                    [[-2.0, -1.0, 0.0, 1.0] * 4],
                    dtype=torch.bfloat16,
                ),
            ):
                results = compare_quantizers(
                    root,
                    {tensor_name: shard_name},
                    FakeImatrix(),
                    [PilotSample(0, 0, "w1")],
                    PilotOptions(bits=(2,), group_size=16, device="cpu"),
                )

            self.assertEqual(len(results), 2)
            self.assertEqual(
                {result.quantizer for result in results},
                {"plain-rtn", "imatrix-weighted-rtn"},
            )
            self.assertTrue(all(result.duration_seconds >= 0.0 for result in results))


if __name__ == "__main__":
    unittest.main()
