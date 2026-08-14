from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepseek_v4_lowbit.artifact_plan import ArtifactRecipe, LayerQuantization
from deepseek_v4_lowbit.frontier_resume import (
    FrontierResumeCheckpointRequest,
    require_frontier_resume_validation_receipt,
    validate_frontier_resume_checkpoint,
)
from deepseek_v4_lowbit.shard_writer import file_sha256
from deepseek_v4_lowbit.source_transform import (
    QuantizerKind,
    TransformOptions,
    transform_recipe_sha256,
)

_HAS_SAFETENSORS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("safetensors") is not None
)


@unittest.skipUnless(_HAS_SAFETENSORS, "requires torch and safetensors")
class FrontierResumeCheckpointTests(unittest.TestCase):
    def test_writes_atomic_receipt_for_verified_conversion_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _write_resume_fixture(Path(temporary_directory))
            with _patch_baseline_hashes(fixture):
                receipt = validate_frontier_resume_checkpoint(_resume_request(fixture))

            persisted = json.loads(fixture["receipt"].read_text(encoding="utf-8"))

        self.assertEqual(receipt, persisted)
        self.assertEqual(receipt["validation_identity"]["candidate"], "quality")
        self.assertEqual(
            receipt["validation_identity"]["volume_id"],
            "volume-recovered",
        )
        self.assertEqual(receipt["reusable_baseline_shards"], 1)
        self.assertEqual(receipt["completed_candidate_shards"], 0)
        self.assertEqual(len(receipt["validation_identity_sha256"]), 64)

    def test_rejects_edited_validation_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _write_resume_fixture(Path(temporary_directory))
            with _patch_baseline_hashes(fixture):
                validate_frontier_resume_checkpoint(_resume_request(fixture))
                receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
                receipt["validation_identity"]["candidate"] = "cliff"
                fixture["receipt"].write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    require_frontier_resume_validation_receipt(
                        fixture["receipt"],
                        volume_id="volume-recovered",
                        candidate="quality",
                        expected_recovery_manifest_sha256=file_sha256(
                            fixture["recovery_manifest"]
                        ),
                    )

    def test_rejects_source_drift_without_replacing_prior_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _write_resume_fixture(Path(temporary_directory))
            with _patch_baseline_hashes(fixture):
                validate_frontier_resume_checkpoint(_resume_request(fixture))
                original_receipt = fixture["receipt"].read_bytes()
                fixture["source_shard"].write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "source shard checksum"):
                    validate_frontier_resume_checkpoint(_resume_request(fixture))

            self.assertEqual(fixture["receipt"].read_bytes(), original_receipt)


def _write_resume_fixture(root: Path) -> dict[str, object]:
    import torch
    from safetensors.torch import save_file

    source = root / "source"
    baseline = root / "baseline"
    output = root / "output"
    source.mkdir()
    baseline.mkdir()
    output.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    tensor_name = "layers.0.input_layernorm.weight"
    source_shard = source / shard_name
    save_file({tensor_name: torch.ones(1)}, source_shard)
    source_index = source / "model.safetensors.index.json"
    source_index.write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_config = source / "config.json"
    source_config.write_text('{"model_type":"deepseek_v4"}\n', encoding="utf-8")

    imatrix = root / "imatrix.dat"
    imatrix.write_bytes(b"test-imatrix")
    transform_options = TransformOptions(
        group_size=128,
        quantizer=QuantizerKind.IMATRIX_WEIGHTED,
        device="cuda",
        imatrix_sha256=file_sha256(imatrix),
    )
    baseline_recipe_sha256 = transform_recipe_sha256(
        ArtifactRecipe(default=LayerQuantization(2, 2)),
        transform_options,
    )
    baseline_shard = baseline / shard_name
    save_file({tensor_name: torch.ones(1)}, baseline_shard)
    baseline_config = baseline / "config.json"
    baseline_config.write_text("{}\n", encoding="utf-8")
    baseline_index = baseline / "model.safetensors.index.json"
    baseline_index.write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    baseline_metrics = baseline / "conversion-metrics.json"
    baseline_metrics.write_text(
        json.dumps(
            {
                "recipe_sha256": baseline_recipe_sha256,
                "shards": [
                    {
                        "shard": shard_name,
                        "output_sha256": file_sha256(baseline_shard),
                        "metrics": [],
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    evidence_directory = root / "evidence"
    evidence_directory.mkdir()
    evidence_names = {
        "pilot_screen_report": "frontier-pilot-screen.json",
        "boundary_report": "frontier-boundary-report.json",
        "full_screen_report": "frontier-full-screen.json",
        "source_headers_report": "source-headers-report.json",
        "source_headers": "source-headers.json",
    }
    evidence: dict[str, Path] = {}
    for role, name in evidence_names.items():
        path = evidence_directory / name
        path.write_text(json.dumps({"role": role}) + "\n", encoding="utf-8")
        evidence[role] = path

    candidate = {
        "default": {
            "w13_bits": 2,
            "w2_bits": 2,
            "w13_group_size": 512,
            "w2_group_size": 128,
        },
        "layers": {
            str(layer): {
                "w13_bits": 2,
                "w2_bits": 4 if layer in {26, 37, 38, 39, 40, 41, 42} else 2,
                "w13_group_size": 128 if layer in {26, 37, 38, 39, 40, 41, 42} else 512,
                "w2_group_size": 128,
            }
            for layer in range(43)
        },
    }
    candidates = {
        name: candidate for name in ("cliff", "capacity", "balanced", "quality")
    }
    bundle = {
        "schema_version": 1,
        "baseline_metrics_sha256": file_sha256(baseline_metrics),
        "pilot_screen_report_sha256": file_sha256(evidence["pilot_screen_report"]),
        "boundary_report_sha256": file_sha256(evidence["boundary_report"]),
        "screen_report_sha256": file_sha256(evidence["full_screen_report"]),
        "source_headers_report_sha256": file_sha256(evidence["source_headers_report"]),
        "source_headers_sha256": file_sha256(evidence["source_headers"]),
        "source_index_sha256": file_sha256(source_index),
        "source_shards_sha256": {shard_name: file_sha256(source_shard)},
        "source_assets_sha256": {"config.json": file_sha256(source_config)},
        "imatrix_sha256": file_sha256(imatrix),
        "candidates": candidates,
        "candidate_summaries": [
            {"name": name} for name in ("cliff", "capacity", "balanced", "quality")
        ],
        "storage_summary": {"baseline_reused_shard_names": [shard_name]},
    }
    recipe = root / "frontier-recipe-bundle.json"
    rebuilt_recipe = root / "frontier-recipe-bundle.rebuilt.json"
    serialized = json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    recipe.write_text(serialized, encoding="utf-8")
    rebuilt_recipe.write_text(serialized, encoding="utf-8")

    recovery_reports = root / "recovery-reports"
    (recovery_reports / "frontier-screen").mkdir(parents=True)
    recovered_report_sources = {
        "frontier-recipe-bundle.json": recipe,
        "frontier-screen/frontier-boundary-report.json": evidence["boundary_report"],
        "frontier-screen/frontier-full-screen.json": evidence["full_screen_report"],
        "frontier-screen/frontier-pilot-screen.json": evidence["pilot_screen_report"],
        "source-headers-report.json": evidence["source_headers_report"],
        "source-headers.json": evidence["source_headers"],
    }
    for relative_path, source_path in recovered_report_sources.items():
        destination = recovery_reports / relative_path
        destination.write_bytes(source_path.read_bytes())
    for relative_path in (
        "frontier-screen/frontier-full-screen-progress.json",
        "frontier-screen/frontier-screen-iterations.json",
        "run-verda-quant-frontier.log",
    ):
        destination = recovery_reports / relative_path
        destination.write_text(f"fixture={relative_path}\n", encoding="utf-8")
    recovered_files = []
    for path in sorted(recovery_reports.rglob("*")):
        if path.is_file():
            recovered_files.append(
                {
                    "path": path.relative_to(recovery_reports).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    recovery_manifest = root / "manifest.json"
    recovery_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign": "deepseek-v4-quant-frontier",
                "recovered_volume_id": "volume-recovered",
                "outcome": "screening-complete-conversion-not-started",
                "files": recovered_files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "source": source,
        "baseline": baseline,
        "output": output,
        "source_shard": source_shard,
        "baseline_config": baseline_config,
        "baseline_index": baseline_index,
        "baseline_metrics": baseline_metrics,
        "imatrix": imatrix,
        "recipe": recipe,
        "rebuilt_recipe": rebuilt_recipe,
        "recovery_manifest": recovery_manifest,
        "recovery_reports": recovery_reports,
        "receipt": root / "frontier-resume-validation.json",
        "evidence": evidence,
    }


def _resume_request(fixture: dict[str, object]) -> FrontierResumeCheckpointRequest:
    return FrontierResumeCheckpointRequest(
        source_directory=fixture["source"],
        baseline_directory=fixture["baseline"],
        imatrix_path=fixture["imatrix"],
        recipe_bundle_path=fixture["recipe"],
        rebuilt_recipe_bundle_path=fixture["rebuilt_recipe"],
        recovery_manifest_path=fixture["recovery_manifest"],
        expected_recovery_manifest_sha256=file_sha256(fixture["recovery_manifest"]),
        recovery_reports_directory=fixture["recovery_reports"],
        output_root=fixture["output"],
        receipt_path=fixture["receipt"],
        volume_id="volume-recovered",
        candidate="quality",
        evidence_paths=fixture["evidence"],
    )


def _patch_baseline_hashes(fixture: dict[str, object]):
    return patch.multiple(
        "deepseek_v4_lowbit.frontier_convert",
        _BASELINE_CONFIG_SHA256=file_sha256(fixture["baseline_config"]),
        _BASELINE_INDEX_SHA256=file_sha256(fixture["baseline_index"]),
        _BASELINE_METRICS_SHA256=file_sha256(fixture["baseline_metrics"]),
    )


if __name__ == "__main__":
    unittest.main()
