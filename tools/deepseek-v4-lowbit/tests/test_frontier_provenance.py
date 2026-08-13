from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.frontier_provenance import (
    load_verified_frontier_recipe_evidence,
    validate_frontier_recipe_bundle_shape,
)
from deepseek_v4_lowbit.shard_writer import file_sha256


class FrontierProvenanceTests(unittest.TestCase):
    def test_verifies_complete_screen_to_recipe_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = _write_evidence_fixture(Path(temporary_directory))

            evidence = load_verified_frontier_recipe_evidence(**paths)

        self.assertEqual(evidence.boundary_layers, (42,))
        self.assertEqual(
            len(evidence.merged_screen_results),
            42 * 2 * 12 + 256 * 12,
        )
        self.assertEqual(evidence.source_index_sha256, "a" * 64)

    def test_rejects_full_screen_from_stale_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = _write_evidence_fixture(Path(temporary_directory))
            full_screen_path = paths["full_screen_path"]
            full = json.loads(full_screen_path.read_text(encoding="utf-8"))
            full["boundary_report_sha256"] = "f" * 64
            _write_json(full_screen_path, full)

            with self.assertRaisesRegex(ValueError, "boundary checksum mismatch"):
                load_verified_frontier_recipe_evidence(**paths)

    def test_rejects_incomplete_recipe_bundle(self) -> None:
        bundle = _recipe_bundle_fixture()
        bundle["candidates"].pop("quality")

        with self.assertRaisesRegex(ValueError, "incomplete or unordered"):
            validate_frontier_recipe_bundle_shape(bundle)


def _write_evidence_fixture(root: Path) -> dict[str, Path]:
    baseline_path = root / "baseline.json"
    pilot_path = root / "pilot.json"
    boundary_path = root / "boundary.json"
    full_path = root / "full.json"
    source_report_path = root / "source-headers-report.json"
    planner_headers_path = root / "source-headers.json"
    shard_name = "model-00001-of-00001.safetensors"
    source_headers = {shard_name: {"tensor": {"dtype": "I8"}}}
    source_report = {
        "schema_version": 1,
        "source_index_sha256": "a" * 64,
        "source_shards": {shard_name: "b" * 64},
        "source_assets": {"config.json": "d" * 64},
        "headers": source_headers,
    }
    _write_json(baseline_path, {"shards": []})
    _write_json(source_report_path, source_report)
    _write_json(planner_headers_path, source_headers)

    pilot_samples = [
        {"layer": layer, "expert": 0, "projection": projection}
        for layer in range(43)
        for projection in ("w1", "w2", "w3")
    ]
    pilot = {
        "report_schema_version": 1,
        "source_index_sha256": "a" * 64,
        "baseline_metrics_sha256": file_sha256(baseline_path),
        "imatrix_sha256": "c" * 64,
        "source_shards": {shard_name: "b" * 64},
        "group_sizes": [128, 256, 512],
        "samples_per_projection": 1,
        "samples": pilot_samples,
        "results": _screen_results(
            (layer, 0, projection)
            for layer in range(43)
            for projection in ("w1", "w2", "w3")
        ),
    }
    # The production default is >=2, but one complete expert per projection keeps this
    # synthetic handoff small. Patch the declared count to two by adding expert 1.
    pilot["samples_per_projection"] = 2
    pilot["samples"].extend(
        {"layer": layer, "expert": 1, "projection": projection}
        for layer in range(43)
        for projection in ("w1", "w2", "w3")
    )
    pilot["results"].extend(
        _screen_results(
            (layer, 1, projection)
            for layer in range(43)
            for projection in ("w1", "w2", "w3")
        )
    )
    _write_json(pilot_path, pilot)

    boundary = {
        "schema_version": 1,
        "baseline_metrics_sha256": file_sha256(baseline_path),
        "screen_report_sha256": file_sha256(pilot_path),
        "source_headers_sha256": file_sha256(planner_headers_path),
        "source_index_sha256": "a" * 64,
        "imatrix_sha256": "c" * 64,
        "source_shards": {shard_name: "b" * 64},
        "layers": [42],
        "reasons": {"42": ["quality-w4-boundary"]},
    }
    _write_json(boundary_path, boundary)
    full = {
        "report_schema_version": 1,
        "source_index_sha256": "a" * 64,
        "imatrix_sha256": "c" * 64,
        "boundary_report_sha256": file_sha256(boundary_path),
        "source_shards": {shard_name: "b" * 64},
        "group_sizes": [128, 256, 512],
        "layers": [42],
        "decision_boundary_layers": [42],
        "results": _screen_results(
            (42, expert, projection)
            for expert in range(256)
            for projection in ("w1", "w2", "w3")
        ),
    }
    _write_json(full_path, full)
    return {
        "baseline_metrics_path": baseline_path,
        "pilot_screen_path": pilot_path,
        "boundary_report_path": boundary_path,
        "full_screen_path": full_path,
        "source_headers_report_path": source_report_path,
        "planner_headers_path": planner_headers_path,
    }


def _screen_results(
    samples: Iterable[tuple[int, int, str]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for layer, expert, projection in samples:
        for group_size in (128, 256, 512):
            for bits in (2, 4) if projection == "w2" else (2,):
                results.append(
                    {
                        "tensor_name": (
                            f"layers.{layer}.ffn.experts.{expert}.{projection}.weight"
                        ),
                        "bits": bits,
                        "group_size": group_size,
                        "selection_error": 0.1,
                    }
                )
    return results


def _recipe_bundle_fixture() -> dict[str, Any]:
    candidates = {
        name: {
            "default": {"w13_bits": 2, "w2_bits": 2, "group_size": 512},
            "layers": {},
        }
        for name in ("cliff", "capacity", "balanced", "quality")
    }
    return {
        "schema_version": 1,
        "baseline_metrics_sha256": "a" * 64,
        "pilot_screen_report_sha256": "b" * 64,
        "boundary_report_sha256": "c" * 64,
        "screen_report_sha256": "d" * 64,
        "source_headers_report_sha256": "e" * 64,
        "source_headers_sha256": "f" * 64,
        "source_index_sha256": "1" * 64,
        "source_shards_sha256": {"model.safetensors": "2" * 64},
        "source_assets_sha256": {"config.json": "4" * 64},
        "imatrix_sha256": "3" * 64,
        "candidates": candidates,
        "candidate_summaries": [
            {"name": name} for name in ("cliff", "capacity", "balanced", "quality")
        ],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
