from __future__ import annotations

import unittest
from pathlib import Path

_RENTAL_DIRECTORY = Path(__file__).parents[1] / "rental"
_PILOT_SCRIPT = _RENTAL_DIRECTORY / "run-verda-quantizer-pilot.sh"
_FULL_CONVERSION_SCRIPT = _RENTAL_DIRECTORY / "run-verda-full-conversion.sh"
_ORACLE_SCRIPT = _RENTAL_DIRECTORY / "run-verda-vllm-w2-oracle.sh"


class RentalScriptContractTests(unittest.TestCase):
    def test_capacity_checks_measure_the_rental_root(self) -> None:
        for script_path in (_PILOT_SCRIPT, _FULL_CONVERSION_SCRIPT):
            script = script_path.read_text(encoding="utf-8")
            with self.subTest(script=script_path.name):
                self.assertIn("- \"$RENTAL_ROOT\" <<'PY'", script)
                self.assertIn("shutil.disk_usage(sys.argv[1])", script)
                self.assertNotIn('shutil.disk_usage(".")', script)

    def test_rental_workloads_require_the_selected_a100_capability(self) -> None:
        for script_path in (_PILOT_SCRIPT, _FULL_CONVERSION_SCRIPT):
            script = script_path.read_text(encoding="utf-8")
            with self.subTest(script=script_path.name):
                self.assertIn("torch.cuda.get_device_capability() != (8, 0)", script)
                self.assertIn("requires compute capability 8.0", script)

    def test_full_conversion_requires_private_write_target(self) -> None:
        script = _FULL_CONVERSION_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('HfApi(token=os.environ["HF_TOKEN"])', script)
        self.assertIn('"repo.write"', script)
        self.assertIn("if not repository.private", script)
        self.assertNotIn('"$PYTHON_ENVIRONMENT/bin/hf" auth whoami', script)

    def test_mutable_checkouts_fail_closed_on_dirty_trees(self) -> None:
        for script_path in (_PILOT_SCRIPT, _FULL_CONVERSION_SCRIPT, _ORACLE_SCRIPT):
            script = script_path.read_text(encoding="utf-8")
            with self.subTest(script=script_path.name):
                self.assertIn(
                    "status --porcelain --untracked-files=all",
                    script,
                )
                self.assertGreaterEqual(script.count("require_clean_checkout"), 3)

    def test_clone_capable_checkouts_validate_after_first_checkout(self) -> None:
        for script_path in (_PILOT_SCRIPT, _ORACLE_SCRIPT):
            script = script_path.read_text(encoding="utf-8")
            with self.subTest(script=script_path.name):
                self.assertIn(
                    'if [[ -d "$destination/.git" ]]; then\n'
                    '        require_clean_checkout "$destination"\n'
                    "    else\n"
                    "        git clone --filter=blob:none --no-checkout "
                    '"$repository_url" "$destination"\n'
                    "    fi",
                    script,
                )

    def test_oracle_discovers_pinned_cuda_tools(self) -> None:
        script = _ORACLE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('export PATH="/usr/local/cuda/bin:$PATH"', script)

    def test_oracle_installs_vllm_test_and_jit_dependencies(self) -> None:
        script = _ORACLE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"ninja==1.13.0"', script)
        self.assertIn('"tblib==3.1.0"', script)


if __name__ == "__main__":
    unittest.main()
