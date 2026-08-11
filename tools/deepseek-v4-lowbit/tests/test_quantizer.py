from __future__ import annotations

import importlib
import importlib.util
import unittest

_HAS_QUANTIZATION_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("auto_round") is not None
)


@unittest.skipUnless(_HAS_QUANTIZATION_DEPS, "requires torch and pinned AutoRound")
class SymmetricQuantizerTests(unittest.TestCase):
    def test_weighted_scale_search_improves_weighted_error(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.quantizer import quantize_symmetric

        weight = torch.tensor(
            [
                [-4.0, -1.8, -0.2, 0.1, 0.3, 0.8, 1.7, 3.9],
                [-3.1, -1.4, -0.1, 0.2, 0.4, 1.1, 2.2, 3.0],
            ],
            dtype=torch.float32,
        )
        importance = torch.tensor(
            [20.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 20.0],
            dtype=torch.float32,
        )

        plain = quantize_symmetric(
            weight,
            bits=2,
            group_size=4,
            importance=importance,
        )
        weighted = quantize_symmetric(
            weight,
            bits=2,
            group_size=4,
            importance=importance,
            optimize_scales=True,
        )

        self.assertLess(weighted.weighted_error, plain.weighted_error)
        self.assertEqual(weighted.codes.shape, weight.shape)
        self.assertEqual(weighted.scales.shape, (2, 2))
        self.assertEqual(weighted.codes.dtype, torch.int8)
        self.assertEqual(weighted.scales.dtype, torch.float16)
        self.assertGreaterEqual(int(weighted.codes.min()), -2)
        self.assertLessEqual(int(weighted.codes.max()), 1)
        reconstructed = (
            weighted.codes.reshape(2, 2, 4).to(torch.float32)
            * weighted.scales.to(torch.float32).unsqueeze(-1)
        ).reshape_as(weight)
        torch.testing.assert_close(weighted.dequantized, reconstructed)

    def test_rejects_importance_with_wrong_input_width(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.quantizer import quantize_symmetric

        weight = torch.ones((2, 8), dtype=torch.float32)
        importance = torch.ones(4, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "importance length"):
            quantize_symmetric(
                weight,
                bits=4,
                group_size=4,
                importance=importance,
                optimize_scales=True,
            )


if __name__ == "__main__":
    unittest.main()
