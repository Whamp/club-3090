from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[3]
PATCH_DIRECTORY = (
    REPOSITORY_ROOT / "models/deepseek-v4-flash-0731/vllm/patches/"
    "deepseek-v4-sm86-speed-experiments"
)
EXPERIMENT_DIRECTORY = (
    REPOSITORY_ROOT / "models/deepseek-v4-flash-0731/vllm/experiments/sm86-speed"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_speed_patch_and_image_contract() -> None:
    assert (
        _sha256(PATCH_DIRECTORY / "0011-perf-add-opt-in-Ampere-FlashMLA-decode.patch")
        == "329e25afcc8293040d899c4dc42b2ecdc345563134571761b3db40ce816c9dc6"
    )
    assert (
        _sha256(
            PATCH_DIRECTORY / "0012-chore-report-hierarchical-all-reduce-dispatch.patch"
        )
        == "ddc4ab71a16f9ee2f36faebdb551b98923d9bdf4d281e68a86b222648d68ef4e"
    )

    builder = (PATCH_DIRECTORY / "build-flash-mla-decode-image.sh").read_text()
    assert 'EXPECTED_VLLM_COMMIT="1d6b37c8eb904bb2d1db7ddd05b002157d5e9f26"' in builder
    assert 'EXPECTED_VLLM_TREE="1260b4aba8fb5bf92e6632882326eb2b800ff3df"' in builder
    assert (
        'EXPECTED_DOCKERFILE_SHA256="'
        '6014fd3703132fc9e06f27dcf3ac0e6c8ac129b5aee3046f405054fd6de738dc"'
    ) in builder
    assert (
        'FLASH_MLA_WHEEL_SHA256="'
        '1e750446aa04b1f325fd1ca29be5d6b3e62f69df69e7ccd4b45df2c267b694d3"'
    ) in builder

    dockerfile = (PATCH_DIRECTORY / "Dockerfile.flash-mla-decode").read_text()
    assert "vllm/envs.py" in dockerfile
    assert "vllm/models/deepseek_v4/ampere/ampere_sparse.py" in dockerfile
    assert "vllm/distributed/device_communicators/cuda_communicator.py" in dockerfile
    assert (
        "flash_mla-2.0.0-cp39-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
    ) in dockerfile


def test_every_speed_arm_renders_without_runtime_side_effects(tmp_path: Path) -> None:
    trace_gate = tmp_path / "trace-gate.json"
    trace_gate.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "deepseek-v4-flash-0731-wna16-quality-12035985",
                "vllm_tree": "1260b4aba8fb5bf92e6632882326eb2b800ff3df",
                "profile_sha256": "a" * 64,
                "request_sha256": "b" * 64,
                "all_reduce_critical_path_fraction": 0.2,
                "timeline_reviewed": True,
                "reviewer_note": "test-reviewed trace",
            }
        )
        + "\n"
    )
    runner = EXPERIMENT_DIRECTORY / "run-speed-arm-with-rollback.sh"
    environment = os.environ | {
        "APPROVED_PRODUCTION_IMAGE_ID": "sha256:" + "1" * 64,
        "SPEED_IMAGE": "example.invalid/speed:test",
        "SPEED_IMAGE_ID": "sha256:" + "2" * 64,
        "MODEL_SNAPSHOT": "/nonexistent/model",
        "MODEL_BLOBS": "/nonexistent/blobs",
        "RUNTIME_CACHE_ROOT": "/nonexistent/cache",
        "HIER_TRACE_GATE_JSON": str(trace_gate),
    }
    for arm in (
        "baseline",
        "trace-baseline",
        "prefill-block2",
        "flashmla-decode",
        "hier-allreduce",
        "indexer96",
        "batched320",
    ):
        output_directory = tmp_path / arm
        completed = subprocess.run(
            [
                str(runner),
                "--dry-run",
                arm,
                str(output_directory),
                "--",
                "true",
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        manifest = json.loads((output_directory / "plan.json").read_text())
        assert manifest["arm"] == arm
        assert manifest["expected_speed_tree"] == (
            "1260b4aba8fb5bf92e6632882326eb2b800ff3df"
        )
        assert "plan_sha256=" in completed.stdout


def test_rollback_runner_never_recreates_production() -> None:
    runner = (EXPERIMENT_DIRECTORY / "run-speed-arm-with-rollback.sh").read_text()
    assert 'docker start "$PRODUCTION_CONTAINER"' in runner
    assert 'docker stop --time 180 "$PRODUCTION_CONTAINER"' in runner
    assert "docker compose up" not in runner
    assert 'docker rm -f "$PRODUCTION_CONTAINER"' not in runner
    assert (
        'restart: "no"' in (EXPERIMENT_DIRECTORY / "compose.override.yml").read_text()
    )
