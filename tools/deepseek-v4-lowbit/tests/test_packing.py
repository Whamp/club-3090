from __future__ import annotations

import importlib
import importlib.util
import unittest

_HAS_PACKING_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("compressed_tensors") is not None
)


@unittest.skipUnless(
    _HAS_PACKING_DEPS,
    "requires torch and pinned compressed-tensors",
)
class CompressedTensorsPackingTests(unittest.TestCase):
    def test_packs_low_bits_in_compressed_tensors_order(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.packing import (
            pack_quantized_tensor,
            packed_checkpoint_tensors,
        )
        from deepseek_v4_lowbit.quantizer import QuantizedTensor

        codes = torch.tensor(
            [[-2, -1, 0, 1] * 4],
            dtype=torch.int8,
        )
        scales = torch.tensor([[0.5]], dtype=torch.float16)
        candidate = QuantizedTensor(
            codes=codes,
            scales=scales,
            dequantized=codes.to(torch.float32) * 0.5,
            unweighted_error=0.0,
            weighted_error=0.0,
        )

        packed = pack_quantized_tensor(candidate, bits=2, group_size=16)

        self.assertEqual(packed.weight_packed.dtype, torch.int32)
        self.assertEqual(packed.weight_packed.shape, (1, 1))
        self.assertEqual(
            int(packed.weight_packed[0, 0].item()) & 0xFFFFFFFF,
            0xE4E4E4E4,
        )
        torch.testing.assert_close(packed.weight_scale, scales)
        torch.testing.assert_close(
            packed.weight_shape,
            torch.tensor([1, 16], dtype=torch.int64),
        )
        checkpoint_tensors = packed_checkpoint_tensors(
            "model.layers.0.ffn.experts.7.w1.weight",
            packed,
        )
        self.assertEqual(
            set(checkpoint_tensors),
            {
                "model.layers.0.ffn.experts.7.w1.weight_packed",
                "model.layers.0.ffn.experts.7.w1.weight_scale",
                "model.layers.0.ffn.experts.7.w1.weight_shape",
            },
        )

    def test_rejects_width_that_humming_allocation_would_truncate(self) -> None:
        torch = importlib.import_module("torch")

        from deepseek_v4_lowbit.packing import pack_quantized_tensor
        from deepseek_v4_lowbit.quantizer import QuantizedTensor

        codes = torch.zeros((1, 32), dtype=torch.int8)
        candidate = QuantizedTensor(
            codes=codes,
            scales=torch.ones((1, 1), dtype=torch.float16),
            dequantized=codes.to(torch.float32),
            unweighted_error=0.0,
            weighted_error=0.0,
        )

        with self.assertRaisesRegex(ValueError, "pack factor 10"):
            pack_quantized_tensor(candidate, bits=3, group_size=32)


if __name__ == "__main__":
    unittest.main()
