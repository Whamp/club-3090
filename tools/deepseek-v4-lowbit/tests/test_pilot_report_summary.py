from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deepseek_v4_lowbit.pilot_report_summary import main, summarize_quantizer_pilot
from deepseek_v4_lowbit.shard_writer import file_sha256


def candidate(
    tensor_name: str,
    quantizer: str,
    *,
    duration: float,
    weighted_error: float,
) -> dict[str, object]:
    return {
        "tensor_name": tensor_name,
        "bits": 2,
        "quantizer": quantizer,
        "duration_seconds": duration,
        "weighted_error": weighted_error,
    }


class PilotReportSummaryTests(unittest.TestCase):
    def test_summarizes_paired_error_and_projection_runtime(self) -> None:
        results = [
            candidate(
                "layers.0.ffn.experts.0.w1.weight",
                "plain-rtn",
                duration=1.0,
                weighted_error=10.0,
            ),
            candidate(
                "layers.0.ffn.experts.0.w1.weight",
                "imatrix-weighted-rtn",
                duration=2.0,
                weighted_error=9.0,
            ),
            candidate(
                "layers.0.ffn.experts.0.w2.weight",
                "plain-rtn",
                duration=3.0,
                weighted_error=5.0,
            ),
            candidate(
                "layers.0.ffn.experts.0.w2.weight",
                "imatrix-weighted-rtn",
                duration=6.0,
                weighted_error=5.0,
            ),
            candidate(
                "layers.0.ffn.experts.0.w3.weight",
                "plain-rtn",
                duration=4.0,
                weighted_error=8.0,
            ),
            candidate(
                "layers.0.ffn.experts.0.w3.weight",
                "imatrix-weighted-rtn",
                duration=8.0,
                weighted_error=10.0,
            ),
        ]

        summary = summarize_quantizer_pilot(results)

        self.assertEqual(summary.pair_count, 3)
        self.assertEqual(
            (summary.improved_count, summary.tied_count, summary.worsened_count),
            (1, 1, 1),
        )
        self.assertEqual(summary.median_weighted_error_improvement_fraction, 0.0)
        self.assertEqual(
            summary.plain_projected_quantize_pack_seconds,
            (1.0 + 3.0 + 4.0) * 43 * 256,
        )
        self.assertEqual(
            summary.weighted_projected_quantize_pack_seconds,
            (2.0 + 6.0 + 8.0) * 43 * 256,
        )
        self.assertIsNone(summary.decision)
        self.assertIn("excludes checkpoint download", summary.estimate_scope)

    def test_cli_binds_summary_to_pilot_report_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pilot_report = root / "pilot.json"
            summary_report = root / "summary.json"
            pilot_report.write_text(
                json.dumps(
                    {
                        "results": [
                            candidate(
                                "layers.0.ffn.experts.0.w1.weight",
                                "plain-rtn",
                                duration=1.0,
                                weighted_error=1.0,
                            ),
                            candidate(
                                "layers.0.ffn.experts.0.w1.weight",
                                "imatrix-weighted-rtn",
                                duration=2.0,
                                weighted_error=0.5,
                            ),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(main([str(pilot_report), str(summary_report)]), 0)

            summary = json.loads(summary_report.read_text(encoding="utf-8"))
            self.assertEqual(summary["pilot_report_sha256"], file_sha256(pilot_report))

    def test_rejects_incomplete_duplicate_and_zero_baseline_pairs(self) -> None:
        plain = candidate(
            "layers.0.ffn.experts.0.w1.weight",
            "plain-rtn",
            duration=1.0,
            weighted_error=1.0,
        )
        with self.assertRaisesRegex(ValueError, "incomplete pair"):
            summarize_quantizer_pilot([plain])
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            summarize_quantizer_pilot([plain, plain])
        with self.assertRaisesRegex(ValueError, "zero baseline"):
            summarize_quantizer_pilot(
                [
                    {**plain, "weighted_error": 0.0},
                    candidate(
                        "layers.0.ffn.experts.0.w1.weight",
                        "imatrix-weighted-rtn",
                        duration=2.0,
                        weighted_error=1.0,
                    ),
                ]
            )


if __name__ == "__main__":
    unittest.main()
