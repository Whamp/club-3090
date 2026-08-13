from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deepseek_v4_lowbit.frontier_recipe import (
    build_frontier_recipe_bundle,
    select_frontier_boundary_layers,
)


class FrontierRecipeTests(unittest.TestCase):
    def test_builds_nested_cliff_capacity_balanced_quality_recipes(self) -> None:
        baseline_metrics, screen_results = _screen_fixture()

        with tempfile.TemporaryDirectory() as temporary_directory:
            headers_path = Path(temporary_directory) / "headers.json"
            headers_path.write_text(
                json.dumps(
                    {
                        "model.safetensors": {
                            "layers.0.ffn.experts.0.w1.scale": _header(
                                "F8_E8M0", [2, 1], 2
                            ),
                            "layers.0.ffn.experts.0.w1.weight": _header(
                                "I8", [2, 256], 512
                            ),
                        }
                    }
                ),
                encoding="utf-8",
            )
            bundle = build_frontier_recipe_bundle(
                baseline_metrics,
                screen_results,
                tensor_headers_path=headers_path,
                baseline_metrics_sha256="a" * 64,
                screen_report_sha256="b" * 64,
                source_headers_sha256="c" * 64,
                source_headers_report_sha256="2" * 64,
                model_parameter_count=284_334_567_511,
                source_index_sha256="d" * 64,
                source_assets_sha256={"config.json": "3" * 64},
                imatrix_sha256="e" * 64,
                pilot_screen_report_sha256="f" * 64,
                boundary_report_sha256="1" * 64,
                candidate_target_bytes={
                    "cliff": 580,
                    "capacity": 586,
                    "balanced": 590,
                    "quality": 594,
                },
            )

        summaries = {summary.name: summary for summary in bundle.candidate_summaries}
        self.assertEqual(list(summaries), ["cliff", "capacity", "balanced", "quality"])
        for summary in summaries.values():
            self.assertLessEqual(summary.total_bytes, summary.byte_budget)
            self.assertEqual(
                summary.unused_budget_bytes,
                summary.byte_budget - summary.total_bytes,
            )
        candidates = bundle.candidates
        for earlier, later in (
            ("cliff", "capacity"),
            ("capacity", "balanced"),
            ("balanced", "quality"),
        ):
            for layer in map(str, range(43)):
                earlier_layer = candidates[earlier]["layers"][layer]
                later_layer = candidates[later]["layers"][layer]
                self.assertLessEqual(
                    later_layer["w13_group_size"],
                    earlier_layer["w13_group_size"],
                )
                self.assertLessEqual(
                    later_layer["w2_group_size"],
                    earlier_layer["w2_group_size"],
                )
                self.assertGreaterEqual(
                    later_layer["w2_bits"],
                    earlier_layer["w2_bits"],
                )
        for candidate in candidates.values():
            for layer in candidate["layers"].values():
                self.assertLessEqual(
                    layer["w2_group_size"],
                    layer["w13_group_size"],
                )
        self.assertEqual(candidates["balanced"]["layers"]["26"]["w2_bits"], 4)
        self.assertEqual(candidates["balanced"]["layers"]["42"]["w2_bits"], 4)
        for layer in range(37, 43):
            self.assertEqual(
                candidates["quality"]["layers"][str(layer)]["w2_bits"],
                4,
            )
        self.assertEqual(bundle.source_index_sha256, "d" * 64)
        self.assertEqual(bundle.imatrix_sha256, "e" * 64)

    def test_selects_boundaries_from_exact_byte_recipes(self) -> None:
        baseline_metrics, screen_results = _screen_fixture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            headers_path = Path(temporary_directory) / "headers.json"
            headers_path.write_text(
                json.dumps(
                    {
                        "model.safetensors": {
                            "layers.0.ffn.experts.0.w1.scale": _header(
                                "F8_E8M0", [2, 1], 2
                            ),
                            "layers.0.ffn.experts.0.w1.weight": _header(
                                "I8", [2, 256], 512
                            ),
                        }
                    }
                ),
                encoding="utf-8",
            )
            selection = select_frontier_boundary_layers(
                baseline_metrics,
                screen_results,
                tensor_headers_path=headers_path,
                candidate_target_bytes={
                    "cliff": 580,
                    "capacity": 586,
                    "balanced": 590,
                    "quality": 594,
                },
            )

        self.assertTrue(selection.layers)
        self.assertLessEqual(len(selection.layers), 16)
        reason_names = {
            reason for reasons in selection.reasons.values() for reason in reasons
        }
        self.assertIn("cliff-w2-group-boundary", reason_names)
        self.assertIn("antirez-late-layer-anchor", reason_names)
        self.assertIn("unsloth-w4-down-anchor", reason_names)


def _screen_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline_metrics: list[dict[str, object]] = []
    screen_results: list[dict[str, object]] = []
    for layer in range(43):
        sensitivity = layer + 1
        for projection in ("w1", "w2", "w3"):
            for expert in range(256):
                tensor_name = f"layers.{layer}.ffn.experts.{expert}.{projection}.weight"
                baseline_metrics.append(
                    {"tensor_name": tensor_name, "weighted_error": 1.0}
                )
            for expert in range(8):
                tensor_name = f"layers.{layer}.ffn.experts.{expert}.{projection}.weight"
                for group_size, ratio in (
                    (128, 1.0),
                    (256, 1.0 + sensitivity / 1000),
                    (512, 1.0 + sensitivity / 100),
                ):
                    screen_results.append(
                        {
                            "tensor_name": tensor_name,
                            "bits": 2,
                            "group_size": group_size,
                            "weighted_error": ratio,
                        }
                    )
                if projection == "w2":
                    for group_size in (128, 256, 512):
                        screen_results.append(
                            {
                                "tensor_name": tensor_name,
                                "bits": 4,
                                "group_size": group_size,
                                "weighted_error": 0.1,
                            }
                        )
    return baseline_metrics, screen_results


def _header(dtype: str, shape: list[int], byte_count: int) -> dict[str, object]:
    return {"dtype": dtype, "shape": shape, "data_offsets": [0, byte_count]}


if __name__ == "__main__":
    unittest.main()
