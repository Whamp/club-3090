from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).parents[3]
_RUNTIME_PATCH_DIRECTORY = (
    _REPOSITORY_ROOT
    / "models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86"
)
_DOCKERFILE = _RUNTIME_PATCH_DIRECTORY / "Dockerfile.runtime-cu130"
_BUILD_SCRIPT = _RUNTIME_PATCH_DIRECTORY / "build-runtime-image.sh"
_PATCH_4 = _RUNTIME_PATCH_DIRECTORY / "0004-fix-load-hybrid-DeepSeek-FP8-linears.patch"
_SM86_ORACLE_SCRIPT = _RUNTIME_PATCH_DIRECTORY / "run-sm86-oracle.sh"
_SERVER60_ROLLBACK_SCRIPT = (
    _RUNTIME_PATCH_DIRECTORY / "run-server60-oracle-with-rollback.sh"
)


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
            'EXPECTED_VLLM_TREE="7f4c19003f808a28ec5adcb5675468c5d34af97b"',
            script,
        )
        self.assertIn("status --porcelain --untracked-files=all", script)
        self.assertIn("exit 2", script)
        self.assertIn("org.opencontainers.image.revision", script)

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
