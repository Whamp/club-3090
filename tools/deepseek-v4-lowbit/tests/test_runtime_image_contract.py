from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

_REPOSITORY_ROOT = Path(__file__).parents[3]
_RUNTIME_PATCH_DIRECTORY = (
    _REPOSITORY_ROOT
    / "models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86"
)
_DOCKERFILE = _RUNTIME_PATCH_DIRECTORY / "Dockerfile.runtime-cu130"
_BUILD_SCRIPT = _RUNTIME_PATCH_DIRECTORY / "build-runtime-image.sh"
_FINAL_DOCKERFILE = _RUNTIME_PATCH_DIRECTORY / "Dockerfile.final-overlay"
_FINAL_BUILD_SCRIPT = _RUNTIME_PATCH_DIRECTORY / "build-final-overlay-image.sh"
_FINAL_COMPOSE = (
    _REPOSITORY_ROOT
    / "models/deepseek-v4-flash-0731/vllm/compose/multi4/wna16/base.yml"
)
_PATCH_4 = _RUNTIME_PATCH_DIRECTORY / "0004-fix-load-hybrid-DeepSeek-FP8-linears.patch"
_PATCH_5 = (
    _RUNTIME_PATCH_DIRECTORY / "0005-fix-forward-layer-to-Humming-MoE-kernel.patch"
)
_PATCH_6 = (
    _RUNTIME_PATCH_DIRECTORY / "0006-fix-compose-DeepSeek-FP8-with-WNA16-experts.patch"
)
_PATCH_7 = (
    _RUNTIME_PATCH_DIRECTORY
    / "0007-fix-gate-sparse-split-K-decode-by-shared-memory.patch"
)
_PATCH_8 = (
    _RUNTIME_PATCH_DIRECTORY
    / "0008-fix-bound-DeepSeek-V4-RoPE-cache-to-runtime-context.patch"
)
_SM86_ORACLE_SCRIPT = _RUNTIME_PATCH_DIRECTORY / "run-sm86-oracle.sh"
_SERVER60_ROLLBACK_SCRIPT = (
    _RUNTIME_PATCH_DIRECTORY / "run-server60-oracle-with-rollback.sh"
)
_MATERIALIZER = _RUNTIME_PATCH_DIRECTORY / "materialize-runtime-model-view.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_materializer_module():
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_runtime_view_materializer", _MATERIALIZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {_MATERIALIZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeImageContractTests(unittest.TestCase):
    def test_runtime_image_pins_rental_proven_software_contract(self) -> None:
        dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn(
            "nvidia/cuda:13.0.2-devel-ubuntu24.04@"
            "sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c",
            dockerfile,
        )
        self.assertIn(
            "VLLM_PRECOMPILED_WHEEL_COMMIT=62195e9784ebec1ece42b88a861734e0702cc2d5",
            dockerfile,
        )
        self.assertIn("ARG TORCH_VERSION=2.13.0", dockerfile)
        self.assertIn('== "0.1.10"', dockerfile)
        self.assertIn('"vllm._C_stable_libtorch"', dockerfile)
        self.assertIn('"vllm._moe_C_stable_libtorch"', dockerfile)
        self.assertIn("importlib.util.find_spec(module_name)", dockerfile)
        self.assertIn('ENTRYPOINT ["vllm", "serve"]', dockerfile)

    def test_builder_rejects_drifted_or_dirty_vllm_source(self) -> None:
        script = _BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'EXPECTED_VLLM_TREE="aeb62948e33074514a742d19c2f9a1a3c2ee3e1f"',
            script,
        )
        self.assertIn("status --porcelain --untracked-files=all", script)
        self.assertIn("exit 2", script)
        self.assertIn("org.opencontainers.image.revision", script)

    def test_final_overlay_image_pins_base_tree_and_production_sources(self) -> None:
        dockerfile = _FINAL_DOCKERFILE.read_text(encoding="utf-8")
        script = _FINAL_BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ARG VERIFIED_BASE_IMAGE=", dockerfile)
        self.assertIn("FROM ${VERIFIED_BASE_IMAGE}", dockerfile)
        self.assertIn(
            'EXPECTED_BASE_IMAGE_DIGEST="'
            "sha256:0e8cc6dc48081e907d553febc8002b1f6d61298454340840f27f18b3a2e66c6c",
            script,
        )
        self.assertIn('BASE_IMAGE="${3:-$EXPECTED_BASE_IMAGE_DIGEST}"', script)
        self.assertIn('docker image inspect "$BASE_IMAGE"', script)
        self.assertIn("EXPECTED_RUNTIME_CONTRACT_SHA256=", script)
        self.assertIn("org.club3090.runtime.contract-sha256", script)
        self.assertIn('--build-arg "VERIFIED_BASE_IMAGE=$verified_base_tag"', script)
        self.assertIn(
            'EXPECTED_VLLM_TREE="aeb62948e33074514a742d19c2f9a1a3c2ee3e1f"',
            script,
        )
        for digest in (
            "07e06cb5489f02f761b99422235014bc6f1cab8c1f799ea2bf7855112dd68910",
            "880bf06530aab3bf8c7b60a8a125663e9c145a2a9ad27ac99cbe0b27cda50b62",
            "ffdb2abe98456d8b1601bbac51cb113d7018bd3db0296ed65e51cf459cf6923a",
            "973692c269a16f2f9791867aa07aab7ad328b26b38f1be6cd5054a43d15eb23b",
            "59c6cce38f43d214c1cde9f26d3287ab4eb1fee13978a32d846add2b85a815db",
            "e0da11160d84fdf9c56ad0848f77372ac81d7b089753b06213ce7b9dac224091",
            "6180c64a7e6caad5a3d887fcf4cecada11122cba60eda5339a66c563a130ba21",
        ):
            self.assertIn(digest, script)
        for source_path in (
            "vllm/model_executor/layers/rotary_embedding/__init__.py",
            "vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py",
            "vllm/models/deepseek_v4/attention.py",
            "vllm/models/deepseek_v4/common/rope.py",
        ):
            self.assertIn(f"COPY {source_path}", dockerfile)
        self.assertIn("status --porcelain --untracked-files=all", script)
        self.assertIn("org.opencontainers.image.revision", script)
        self.assertIn("org.club3090.runtime.base-digest", script)

    def test_full_runtime_builder_pins_fresh_host_contract(self) -> None:
        script = _BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'EXPECTED_RUNTIME_DOCKERFILE_SHA256="'
            "7d4ab7f124d1ca5fc68facaafec8c55b98683e249cf669a2c102ac8ba6013838",
            script,
        )
        self.assertIn(
            "org.club3090.runtime.contract-sha256=",
            script,
        )
        self.assertIn("status --porcelain --untracked-files=all", script)

    def test_final_compose_encodes_measured_graph_profile(self) -> None:
        compose = _FINAL_COMPOSE.read_text(encoding="utf-8")
        self.assertIn("${MODEL_SNAPSHOT:?", compose)
        self.assertIn("${MODEL_BLOBS:?", compose)
        self.assertIn("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB: 64", compose)
        self.assertIn("VLLM_SPARSE_DENSE_QUERY_BLOCK: 0", compose)
        self.assertIn(
            "club-3090/deepseek-v4-wna16-sm86:aeb62948-rope-cu130@"
            "sha256:0beb1f0cba2e41837f4ba5af01cc5c4686afde4f40ab1df5147a6ad945b0af1f",
            compose,
        )
        self.assertIn("MAX_MODEL_LEN:-215000", compose)
        self.assertIn("MAX_NUM_SEQS:-4", compose)
        self.assertIn("MAX_NUM_BATCHED_TOKENS:-256", compose)
        self.assertIn("--enable-auto-tool-choice", compose)
        self.assertIn("--tool-call-parser", compose)
        self.assertIn("--reasoning-parser", compose)
        self.assertNotIn("--enforce-eager", compose)
        self.assertNotIn("--cpu-offload-gb", compose)
        self.assertNotIn("kv-cache-memory-bytes", compose)

    def test_runtime_model_view_is_hash_bound_and_reproducible(self) -> None:
        materializer = _load_materializer_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "artifact"
            output = root / "runtime-model"
            artifact.mkdir()
            output.mkdir()
            (output / "stale-file").write_text("stale", encoding="utf-8")
            config = {
                "club_3090_lowbit": {"source_quantization_method": "fp8"},
                "quantization_config": {
                    "config_groups": {"group_w2": {}},
                    "quant_method": "compressed-tensors",
                },
            }
            (artifact / "config.json").write_text(json.dumps(config), encoding="utf-8")
            index = {"weight_map": {"model.weight": "model-00001.safetensors"}}
            (artifact / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            (artifact / "model-00001.safetensors").write_bytes(b"weights")
            (artifact / "tokenizer.json").write_text("{}", encoding="utf-8")

            expected_config = json.loads((artifact / "config.json").read_text())
            expected_config["quantization_config"]["base_quant_method"] = (
                "deepseek_v4_fp8"
            )
            expected_config["club_3090_lowbit"]["source_quantization_method"] = (
                "compressed-tensors"
            )
            expected_rendering = (
                json.dumps(expected_config, indent=2, sort_keys=True) + "\n"
            )
            with ExitStack() as patches:
                patches.enter_context(
                    patch.object(
                        materializer,
                        "EXPECTED_ARTIFACT_CONFIG_SHA256",
                        _sha256(artifact / "config.json"),
                    )
                )
                patches.enter_context(
                    patch.object(
                        materializer,
                        "EXPECTED_ARTIFACT_INDEX_SHA256",
                        _sha256(artifact / "model.safetensors.index.json"),
                    )
                )
                patches.enter_context(
                    patch.object(
                        materializer,
                        "EXPECTED_RUNTIME_CONFIG_SHA256",
                        hashlib.sha256(expected_rendering.encode()).hexdigest(),
                    )
                )
                materializer.materialize_runtime_model_view(artifact, output)

            self.assertEqual(
                (output / "config.json").read_text(encoding="utf-8"),
                expected_rendering,
            )
            self.assertFalse((output / "stale-file").exists())
            self.assertTrue((output / "model-00001.safetensors").is_symlink())
            self.assertEqual(
                (output / "model-00001.safetensors").resolve().read_bytes(),
                b"weights",
            )

    def test_runtime_model_view_rejects_missing_indexed_shard(self) -> None:
        materializer = _load_materializer_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "artifact"
            artifact.mkdir()
            (artifact / "config.json").write_text(
                json.dumps(
                    {
                        "club_3090_lowbit": {"source_quantization_method": "fp8"},
                        "quantization_config": {
                            "config_groups": {"group_w2": {}},
                            "quant_method": "compressed-tensors",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (artifact / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.weight": "missing.safetensors"}}),
                encoding="utf-8",
            )
            with ExitStack() as patches:
                patches.enter_context(
                    patch.object(
                        materializer,
                        "EXPECTED_ARTIFACT_CONFIG_SHA256",
                        _sha256(artifact / "config.json"),
                    )
                )
                patches.enter_context(
                    patch.object(
                        materializer,
                        "EXPECTED_ARTIFACT_INDEX_SHA256",
                        _sha256(artifact / "model.safetensors.index.json"),
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "missing indexed shard"):
                    materializer.materialize_runtime_model_view(
                        artifact, root / "runtime-model"
                    )

    def test_hybrid_fp8_loader_patch_is_checksum_pinned(self) -> None:
        self.assertEqual(
            hashlib.sha256(_PATCH_4.read_bytes()).hexdigest(),
            "3be16754f61170ff2da57a1c64edcd7c524ed6ad9b10c5189d3661e6f55ffc8f",
        )
        readme = (_RUNTIME_PATCH_DIRECTORY / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "3be16754f61170ff2da57a1c64edcd7c524ed6ad9b10c5189d3661e6f55ffc8f"
            "  0004-fix-load-hybrid-DeepSeek-FP8-linears.patch",
            readme,
        )

    def test_humming_layer_forwarding_patch_is_checksum_pinned(self) -> None:
        self.assertEqual(
            hashlib.sha256(_PATCH_5.read_bytes()).hexdigest(),
            "f446a73a37b7715023f05aeec526b714fdadbefa80772268e242218c69efc34e",
        )
        readme = (_RUNTIME_PATCH_DIRECTORY / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "f446a73a37b7715023f05aeec526b714fdadbefa80772268e242218c69efc34e"
            "  0005-fix-forward-layer-to-Humming-MoE-kernel.patch",
            readme,
        )

    def test_hybrid_deepseek_quant_config_patch_is_checksum_pinned(self) -> None:
        self.assertEqual(
            hashlib.sha256(_PATCH_6.read_bytes()).hexdigest(),
            "9af88957c5900e741794002907183a324510bcc7ebb7dd60fef22d66cd5ac005",
        )
        readme = (_RUNTIME_PATCH_DIRECTORY / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "9af88957c5900e741794002907183a324510bcc7ebb7dd60fef22d66cd5ac005"
            "  0006-fix-compose-DeepSeek-FP8-with-WNA16-experts.patch",
            readme,
        )

    def test_sm86_sparse_decode_dispatch_patch_is_checksum_pinned(self) -> None:
        self.assertEqual(
            hashlib.sha256(_PATCH_7.read_bytes()).hexdigest(),
            "f4dec6b898ec327a06b8bd85841ad9e662eb9be7ab59a6cd3a75f60e4c0bc672",
        )
        readme = (_RUNTIME_PATCH_DIRECTORY / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "f4dec6b898ec327a06b8bd85841ad9e662eb9be7ab59a6cd3a75f60e4c0bc672"
            "  0007-fix-gate-sparse-split-K-decode-by-shared-memory.patch",
            readme,
        )

    def test_runtime_bounded_rope_cache_patch_is_checksum_pinned(self) -> None:
        self.assertEqual(
            hashlib.sha256(_PATCH_8.read_bytes()).hexdigest(),
            "173dc71a669f1ab7cbffd19256b4eb2dd30329597bf9de54b7f95cec8dc76c52",
        )
        readme = (_RUNTIME_PATCH_DIRECTORY / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "173dc71a669f1ab7cbffd19256b4eb2dd30329597bf9de54b7f95cec8dc76c52"
            "  0008-fix-bound-DeepSeek-V4-RoPE-cache-to-runtime-context.patch",
            readme,
        )

    def test_sm86_oracle_requires_authorization_and_idle_gpu(self) -> None:
        script = _SM86_ORACLE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("I_AUTHORIZE_SERVER60_GPU_ORACLE", script)
        self.assertIn("query-compute-apps=pid,process_name", script)
        self.assertIn("refuses to share GPUs with active processes", script)
        self.assertIn("capability != (8, 6)", script)
        self.assertIn("test_humming_w2_group128_indexed_numerical_oracle", script)
        self.assertIn("grep -q 'sm_86'", script)
        self.assertIn("--gpus", script)
        self.assertIn("device=0", script)

    def test_server60_wrapper_restores_exact_llama_service(self) -> None:
        script = _SERVER60_ROLLBACK_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("I_AUTHORIZE_LLAMA_STOP_FOR_SM86_ORACLE", script)
        self.assertIn("trap restore_llama_service EXIT INT TERM HUP", script)
        self.assertIn("sha256:a96bd947d63eb81d8baf9f6f5ecb266", script)
        self.assertIn("stop --timeout 120", script)
        self.assertIn("up --detach", script)
        self.assertIn("I_AUTHORIZE_SERVER60_GPU_ORACLE", script)
        self.assertIn("timeout --signal=TERM --kill-after=2m", script)
        self.assertIn("CRITICAL: failed to restore healthy llama.cpp service", script)


if __name__ == "__main__":
    unittest.main()
