from __future__ import annotations

import importlib
import importlib.util
import unittest

_HAS_DEQUANTIZATION_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("auto_round") is not None
    and importlib.util.find_spec("auto_round_extension") is not None
)


@unittest.skipUnless(
    _HAS_DEQUANTIZATION_DEPS,
    "requires torch and pinned AutoRound extension",
)
class DeepSeekSourceDequantizationTests(unittest.TestCase):
    def test_dequantizes_one_official_mxfp4_expert_weight(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.source_dequant import dequantize_routed_expert_weight

        packed_weight = torch.full((2, 16), 0x11, dtype=torch.uint8)
        coarse_scale = torch.full((1, 1), 127, dtype=torch.uint8)

        dequantized = dequantize_routed_expert_weight(
            "layers.0.ffn.experts.7.w1.weight",
            packed_weight,
            coarse_scale,
            device="cpu",
        )

        self.assertEqual(dequantized.shape, (2, 32))
        self.assertEqual(dequantized.dtype, torch.bfloat16)
        self.assertEqual(dequantized.device.type, "cpu")
        self.assertTrue(bool(torch.isfinite(dequantized).all()))
        self.assertTrue(bool((dequantized != 0).any()))

    def test_rejects_non_routed_weight_name(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.source_dequant import dequantize_routed_expert_weight

        with self.assertRaisesRegex(ValueError, "routed expert"):
            dequantize_routed_expert_weight(
                "layers.0.self_attn.q_proj.weight",
                torch.zeros((2, 16), dtype=torch.uint8),
                torch.full((1, 1), 127, dtype=torch.uint8),
            )


if __name__ == "__main__":
    unittest.main()
