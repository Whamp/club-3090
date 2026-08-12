from __future__ import annotations

import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepseek_v4_lowbit.pilot import (
    PilotCandidateResult,
    PilotOptions,
    PilotSample,
    compare_quantizers,
)
from deepseek_v4_lowbit.pilot_cli import expand_pilot_samples, main
from deepseek_v4_lowbit.shard_writer import file_sha256

_HAS_PILOT_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("torch", "safetensors", "compressed_tensors", "auto_round")
)


class PilotCliTests(unittest.TestCase):
    def test_expands_layer_expert_samples_across_projections(self) -> None:
        samples = expand_pilot_samples(["26:17", "0:0"], ["w1", "w2"])

        self.assertEqual(
            samples,
            (
                PilotSample(0, 0, "w1"),
                PilotSample(0, 0, "w2"),
                PilotSample(26, 17, "w1"),
                PilotSample(26, 17, "w2"),
            ),
        )

    def test_cli_binds_report_to_source_index_and_inputs(self) -> None:
        class FakeImatrix:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            @staticmethod
            def validate_deepseek_v4_geometry() -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source"
            source_directory.mkdir()
            shard_name = "model-00001.safetensors"
            tensor_name = "layers.0.ffn.experts.0.w1.weight"
            shard_path = source_directory / shard_name
            shard_path.write_bytes(b"source")
            index_path = source_directory / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps({"weight_map": {tensor_name: shard_name}}),
                encoding="utf-8",
            )
            imatrix_path = root / "imatrix.dat"
            imatrix_path.write_bytes(b"imatrix")
            report_path = root / "pilot.json"
            results = (
                PilotCandidateResult(
                    tensor_name=tensor_name,
                    source_shard=shard_name,
                    bits=2,
                    quantizer="plain-rtn",
                    duration_seconds=1.0,
                    unweighted_error=1.0,
                    weighted_error=1.0,
                ),
                PilotCandidateResult(
                    tensor_name=tensor_name,
                    source_shard=shard_name,
                    bits=2,
                    quantizer="imatrix-weighted-rtn",
                    duration_seconds=2.0,
                    unweighted_error=1.0,
                    weighted_error=0.5,
                ),
            )
            with (
                patch(
                    "deepseek_v4_lowbit.pilot_cli.ImatrixFile.open",
                    return_value=FakeImatrix(),
                ),
                patch(
                    "deepseek_v4_lowbit.pilot_cli.compare_quantizers",
                    return_value=results,
                ),
            ):
                self.assertEqual(
                    main(
                        [
                            str(source_directory),
                            str(imatrix_path),
                            str(report_path),
                            "--sample",
                            "0:0",
                            "--projection",
                            "w1",
                        ]
                    ),
                    0,
                )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["report_schema_version"], 1)
            self.assertEqual(report["source_index_sha256"], file_sha256(index_path))
            self.assertEqual(report["imatrix_sha256"], file_sha256(imatrix_path))
            self.assertEqual(
                report["source_shards"], {shard_name: file_sha256(shard_path)}
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
