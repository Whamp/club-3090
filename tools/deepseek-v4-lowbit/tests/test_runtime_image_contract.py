from __future__ import annotations

import unittest
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).parents[3]
_RUNTIME_PATCH_DIRECTORY = (
    _REPOSITORY_ROOT
    / "models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86"
)
_DOCKERFILE = _RUNTIME_PATCH_DIRECTORY / "Dockerfile.runtime-cu130"
_BUILD_SCRIPT = _RUNTIME_PATCH_DIRECTORY / "build-runtime-image.sh"


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
        self.assertIn("import vllm._C_stable_libtorch", dockerfile)
        self.assertIn("import vllm._moe_C_stable_libtorch", dockerfile)
        self.assertIn('ENTRYPOINT ["vllm", "serve"]', dockerfile)

    def test_builder_rejects_drifted_or_dirty_vllm_source(self) -> None:
        script = _BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'EXPECTED_VLLM_TREE="97a21943d9a68bcf1ef4ac3319d0a6e3e1c66267"',
            script,
        )
        self.assertIn("status --porcelain --untracked-files=all", script)
        self.assertIn("exit 2", script)
        self.assertIn("org.opencontainers.image.revision", script)


if __name__ == "__main__":
    unittest.main()
