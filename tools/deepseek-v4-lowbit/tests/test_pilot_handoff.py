from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from deepseek_v4_lowbit.pilot import PilotSample
from deepseek_v4_lowbit.pilot_handoff import validate_quantizer_pilot_handoff
from deepseek_v4_lowbit.pilot_report_summary import summarize_quantizer_pilot
from deepseek_v4_lowbit.shard_writer import file_sha256


class PilotHandoffTests(unittest.TestCase):
    def test_validates_bound_report_summary_and_input_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            source_directory.mkdir()
            tensor_name = "layers.0.ffn.experts.0.w1.weight"
            shard_name = "model-00001-of-00001.safetensors"
            shard_path = source_directory / shard_name
            shard_path.write_bytes(b"source-shard")
            index_path = source_directory / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps({"weight_map": {tensor_name: shard_name}}),
                encoding="utf-8",
            )
            imatrix_path = root / "imatrix.dat"
            imatrix_path.write_bytes(b"imatrix")
            results = _candidate_pair(tensor_name, shard_name)
            report_path = root / "pilot.json"
            report_path.write_text(
                json.dumps(
                    {
                        "report_schema_version": 1,
                        "source_index_sha256": file_sha256(index_path),
                        "imatrix_sha256": file_sha256(imatrix_path),
                        "source_shards": {shard_name: file_sha256(shard_path)},
                        "device": "cuda",
                        "group_size": 128,
                        "samples": [{"layer": 0, "expert": 0, "projection": "w1"}],
                        "results": results,
                    }
                ),
                encoding="utf-8",
            )
            summary_path = root / "summary.json"
            summary_payload = asdict(summarize_quantizer_pilot(results))
            summary_payload["pilot_report_sha256"] = file_sha256(report_path)
            summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

            summary = validate_quantizer_pilot_handoff(
                report_path,
                summary_path,
                source_directory,
                imatrix_path,
                expected_samples=(PilotSample(0, 0, "w1"),),
                expected_bits=(2,),
                expected_group_size=128,
                expected_device="cuda",
            )

            self.assertEqual(summary.pair_count, 1)

    def test_rejects_stale_summary_and_changed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            source_directory.mkdir()
            tensor_name = "layers.0.ffn.experts.0.w1.weight"
            shard_name = "model-00001-of-00001.safetensors"
            shard_path = source_directory / shard_name
            shard_path.write_bytes(b"source-shard")
            index_path = source_directory / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps({"weight_map": {tensor_name: shard_name}}),
                encoding="utf-8",
            )
            imatrix_path = root / "imatrix.dat"
            imatrix_path.write_bytes(b"imatrix")
            results = _candidate_pair(tensor_name, shard_name)
            report_path = root / "pilot.json"
            report_path.write_text(
                json.dumps(
                    {
                        "report_schema_version": 1,
                        "source_index_sha256": file_sha256(index_path),
                        "imatrix_sha256": file_sha256(imatrix_path),
                        "source_shards": {shard_name: file_sha256(shard_path)},
                        "device": "cuda",
                        "group_size": 128,
                        "samples": [{"layer": 0, "expert": 0, "projection": "w1"}],
                        "results": results,
                    }
                ),
                encoding="utf-8",
            )
            summary_path = root / "summary.json"
            summary_payload = asdict(summarize_quantizer_pilot(results))
            summary_payload["pilot_report_sha256"] = "0" * 64
            summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "summary is not bound"):
                validate_quantizer_pilot_handoff(
                    report_path,
                    summary_path,
                    source_directory,
                    imatrix_path,
                    expected_samples=(PilotSample(0, 0, "w1"),),
                )

            summary_payload["pilot_report_sha256"] = file_sha256(report_path)
            summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
            shard_path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source shard checksum mismatch"):
                validate_quantizer_pilot_handoff(
                    report_path,
                    summary_path,
                    source_directory,
                    imatrix_path,
                    expected_samples=(PilotSample(0, 0, "w1"),),
                )


def _candidate_pair(tensor_name: str, shard_name: str) -> list[dict[str, object]]:
    return [
        {
            "tensor_name": tensor_name,
            "source_shard": shard_name,
            "bits": 2,
            "quantizer": "plain-rtn",
            "duration_seconds": 1.0,
            "unweighted_error": 2.0,
            "weighted_error": 2.0,
        },
        {
            "tensor_name": tensor_name,
            "source_shard": shard_name,
            "bits": 2,
            "quantizer": "imatrix-weighted-rtn",
            "duration_seconds": 2.0,
            "unweighted_error": 2.0,
            "weighted_error": 1.0,
        },
    ]


if __name__ == "__main__":
    unittest.main()
