from __future__ import annotations

import importlib.util
import unittest

from deepseek_v4_lowbit.frontier_screen import (
    _normalized_frontier_errors,
    expand_full_frontier_layers,
    select_stratified_frontier_samples,
)


class FrontierScreenSampleTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_normalized_errors_match_relative_output_energy(self) -> None:
        import torch

        source = torch.ones((2, 4), dtype=torch.float32)
        importance = torch.ones(4, dtype=torch.float32)

        perfect = _normalized_frontier_errors(
            source,
            source,
            importance,
            torch,
        )
        doubled = _normalized_frontier_errors(
            source,
            source * 2,
            importance,
            torch,
        )

        self.assertEqual(perfect, (0.0, 0.0))
        self.assertEqual(doubled, (1.0, 1.0))

    def test_selects_error_quantiles_for_every_layer_and_projection(self) -> None:
        metrics = []
        for layer in range(43):
            for projection in ("w1", "w2", "w3"):
                for expert in range(256):
                    metrics.append(
                        {
                            "tensor_name": (
                                f"layers.{layer}.ffn.experts.{expert}."
                                f"{projection}.weight"
                            ),
                            "weighted_error": float(expert + 1),
                        }
                    )

        samples = select_stratified_frontier_samples(
            metrics,
            samples_per_projection=8,
        )

        self.assertEqual(len(samples), 43 * 3 * 8)
        layer_zero_w2 = [
            sample.expert
            for sample in samples
            if sample.layer == 0 and sample.projection == "w2"
        ]
        self.assertEqual(layer_zero_w2, [0, 36, 73, 109, 146, 182, 219, 255])

    def test_expands_boundary_layers_to_all_experts(self) -> None:
        samples = expand_full_frontier_layers([42, 26])

        self.assertEqual(len(samples), 2 * 256 * 3)
        self.assertEqual(samples[0].layer, 26)
        self.assertEqual(samples[-1].layer, 42)

    def test_requires_complete_256_expert_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer/projection mismatch"):
            select_stratified_frontier_samples(
                [
                    {
                        "tensor_name": "layers.0.ffn.experts.0.w1.weight",
                        "weighted_error": 1.0,
                    }
                ],
                samples_per_projection=8,
            )


if __name__ == "__main__":
    unittest.main()
