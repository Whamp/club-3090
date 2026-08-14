from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

from deepseek_v4_lowbit.shard_writer import file_sha256

_RENTAL_DIRECTORY = Path(__file__).parents[1] / "rental"
_PILOT_SCRIPT = _RENTAL_DIRECTORY / "run-verda-quantizer-pilot.sh"
_FULL_CONVERSION_SCRIPT = _RENTAL_DIRECTORY / "run-verda-full-conversion.sh"
_ORACLE_SCRIPT = _RENTAL_DIRECTORY / "run-verda-vllm-w2-oracle.sh"
_FRONTIER_SCRIPT = _RENTAL_DIRECTORY / "run-verda-quant-frontier.sh"
_FRONTIER_HOST_SCRIPT = _RENTAL_DIRECTORY / "run-verda-frontier-host.sh"
_FRONTIER_DELETE_SCRIPT = _RENTAL_DIRECTORY / "delete-verda-frontier-vm.sh"
_FRONTIER_RECOVERY_MANIFEST = (
    _RENTAL_DIRECTORY / "frontier-recovery-manifest-20260813.json"
)
_RECOVERY_MANIFEST_SHA256 = (
    "0788715e8373daed48bbedaee11fedee217c798a2f189fee00473c9a9913a480"
)

_FAKE_FRONTIER_HOST_VERDA_CLI = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json
    import os
    import sys
    from pathlib import Path

    root = Path(os.environ["FAKE_VERDA_ROOT"])
    arguments = sys.argv[1:]
    if not arguments or arguments[0] != "--agent" or "json" not in arguments:
        raise SystemExit(f"unsafe fake Verda invocation: {arguments}")
    with (root / "calls.log").open("a", encoding="utf-8") as handle:
        handle.write(" ".join(arguments) + "\\n")
    resource, action = arguments[1:3]
    volume = {
        "id": "volume-1",
        "name": "deepseek-v4-quant-frontier-os",
        "size": 350,
        "type": "NVMe",
        "status": "detached",
        "location": "FIN-03",
        "contract": "PAY_AS_YOU_GO",
        "is_os_volume": True,
        "instance_id": None,
        "instances": [],
        "base_hourly_cost": 0.0959,
    }
    if (resource, action) == ("vm", "list"):
        response = []
    elif (resource, action) == ("volume", "list"):
        response = [volume]
    elif (resource, action) == ("volume", "describe"):
        requested_id = arguments[3]
        if requested_id != volume["id"]:
            raise SystemExit("unknown volume")
        response = volume
    elif (resource, action) == ("vm", "availability"):
        response = [
            {
                "location": "FIN-03",
                "instance_type": "CPU.4V.16G",
                "price_per_hour": 0.0279,
            }
        ]
    elif (resource, action) == ("ssh-key", "list"):
        response = [{"id": "ssh-key-1"}]
    elif (resource, action) == ("cost", "balance"):
        response = {"amount": 24.0, "currency": "usd"}
    elif (resource, action) == ("cost", "estimate"):
        response = {
            "instance_type": "CPU.4V.16G",
            "instance": {"hourly": 0.0279},
            "total": {"hourly": 0.0279},
        }
    else:
        raise SystemExit(f"unsupported fake Verda invocation: {arguments}")
    print(json.dumps(response))
    """
)

_FAKE_VERDA_CLI = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json
    import os
    import sys
    from pathlib import Path

    root = Path(os.environ["FAKE_VERDA_ROOT"])
    arguments = sys.argv[1:]
    if not arguments or arguments[0] != "--agent" or "json" not in arguments:
        raise SystemExit(f"unsafe fake Verda invocation: {arguments}")
    resource, action = arguments[1:3]
    vm_path = root / "vm-present"
    volume_path = root / "volume-present"
    vm = {
        "id": "vm-1",
        "hostname": "deepseek-v4-quant-frontier",
        "instance_type": "2A100.44V",
        "location": "FIN-01",
        "os_volume_id": "volume-1",
        "volume_ids": [],
    }
    volume = {
        "id": "volume-1",
        "name": "deepseek-v4-quant-frontier-os",
        "size": 350,
        "location": "FIN-01",
        "is_os_volume": True,
        "instance_id": "vm-1" if vm_path.exists() else None,
    }
    if (resource, action) == ("vm", "list"):
        print(json.dumps([vm] if vm_path.exists() else []))
    elif (resource, action) == ("vm", "describe") and vm_path.exists():
        print(json.dumps(vm))
    elif (resource, action) == ("volume", "list"):
        print(json.dumps([volume] if volume_path.exists() else []))
    elif (resource, action) == ("vm", "delete"):
        with (root / "deletions.log").open("a", encoding="utf-8") as handle:
            handle.write(f"vm {arguments[3]} {' '.join(arguments)}\\n")
        vm_path.unlink(missing_ok=True)
        print(json.dumps({"status": "completed"}))
    elif (resource, action) == ("volume", "delete"):
        with (root / "deletions.log").open("a", encoding="utf-8") as handle:
            handle.write(f"volume {arguments[3]} {' '.join(arguments)}\\n")
        volume_path.unlink(missing_ok=True)
        print(json.dumps({"status": "completed"}))
    else:
        raise SystemExit(f"unsupported fake Verda invocation: {arguments}")
    """
)


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

    def test_recovery_manifest_is_checksum_pinned(self) -> None:
        self.assertEqual(
            file_sha256(_FRONTIER_RECOVERY_MANIFEST),
            _RECOVERY_MANIFEST_SHA256,
        )
        manifest = json.loads(_FRONTIER_RECOVERY_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["recovered_volume_id"],
            "9a7105b5-3c04-4bd7-b9fb-84c7be98c961",
        )
        self.assertEqual(len(manifest["files"]), 9)

    def test_frontier_runner_is_pinned_and_low_disk(self) -> None:
        script = _FRONTIER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'CLUB_3090_REVISION="2a6d6a4b07907e39d0fe76c59290e5dbb4589839"',
            script,
        )
        self.assertNotIn("__CLUB_3090_REVISION__", script)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"', script)
        self.assertIn("requires at least 110 GiB host RAM", script)
        self.assertIn("minimum_free_gib=280", script)
        self.assertIn("minimum_free_gib=88", script)
        self.assertIn("requires exactly one A100 80GB GPU", script)
        self.assertIn("requires compute capability 8.0", script)
        self.assertIn('export HF_HOME="$RENTAL_ROOT/huggingface-cache"', script)
        self.assertIn('export HF_HUB_CACHE="$HF_HOME/hub"', script)
        self.assertIn('export HF_XET_CACHE="$HF_HOME/xet"', script)
        self.assertIn("export HF_XET_CHUNK_CACHE_SIZE_BYTES=0", script)
        self.assertIn("Hugging Face cache unexpectedly consumed", script)
        self.assertIn("baseline_reused_shard_names", script)
        self.assertIn('FRONTIER_CANDIDATE="${4:-quality}"', script)
        self.assertIn('--candidate "$FRONTIER_CANDIDATE"', script)
        self.assertNotIn("reclaim four nested candidates", script)
        self.assertIn("--delete-local-after-verify", script)
        self.assertIn("Verify the immutable published revision", script)
        self.assertIn(
            'COMPLETION_RECEIPT="$REPORT_DIRECTORY/frontier-complete.json"', script
        )
        self.assertIn('"publication_report_sha256"', script)
        self.assertIn("os.replace(temporary, receipt_path)", script)
        self.assertIn("os.fsync(directory_fd)", script)
        self.assertIn('RUN_MODE="${5:-fresh}"', script)
        self.assertIn('if [[ "$RUN_MODE" == "validate-resume" ]]', script)
        self.assertIn('if [[ "$RUN_MODE" == "resume-conversion" ]]', script)
        self.assertIn("deepseek-v4-require-frontier-resume", script)
        self.assertIn(
            'if [[ "$RUN_MODE" == "fresh" ]]; then\n'
            '    log_frontier_step "Download pinned official checkpoint',
            script,
        )
        self.assertEqual(script.count("deepseek-v4-run-frontier-screen"), 1)
        self.assertEqual(script.count("deepseek-v4-build-frontier-recipes"), 2)

    def test_frontier_host_orchestrator_guards_cost_and_exact_cleanup(self) -> None:
        script = _FRONTIER_HOST_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("VERDA_FRONTIER_PROFILE:-main", script)
        self.assertIn("export VERDA_PROFILE", script)
        self.assertNotIn("auth use", script)
        self.assertIn('RUN_MODE="${VERDA_FRONTIER_RUN_MODE:-fresh}"', script)
        self.assertIn('default_instance_type="1A100.22V"', script)
        self.assertIn('default_instance_type="CPU.4V.16G"', script)
        self.assertIn("VERDA_FRONTIER_LOCATION:-FIN-03", script)
        self.assertIn('default_max_hours="16"', script)
        self.assertIn('default_max_cost_usd="30.25"', script)
        self.assertNotIn("VERDA_FRONTIER_INSTANCE_TYPE:-2RTXPRO6000.60V", script)
        self.assertIn('verda --agent "$@" -o json', script)
        self.assertIn("require_account_contract", script)
        self.assertIn("require_reusable_state_file", script)
        self.assertIn("projected > max_cost", script)
        self.assertIn("projected > float(balance", script)
        pending_state = "write_state pending"
        create_command = 'create_response="$(verda_json "${create_arguments[@]}")"'
        arm_invocation = "arm_watchdog\ncreate_started=1"
        self.assertLess(script.index(pending_state), script.index(arm_invocation))
        self.assertLess(script.index(arm_invocation), script.index(create_command))
        self.assertIn("--on-calendar", script)
        self.assertIn("--timer-property=Persistent=true", script)
        self.assertIn("--service-type=exec", script)
        self.assertIn("--property=Restart=on-failure", script)
        self.assertIn("--property=RestartSec=60s", script)
        self.assertIn('--setenv="VERDA_PROFILE=$VERDA_PROFILE"', script)
        self.assertIn('systemctl --user start "$WATCHDOG_UNIT.service"', script)
        self.assertIn(
            '"$DELETE_SCRIPT" "$STATE_FILE" --delete-volume',
            script,
        )
        self.assertIn(
            '"$DELETE_SCRIPT" "$STATE_FILE" --preserve-volume',
            script,
        )
        self.assertIn("extract_volume_ids", script)
        self.assertIn('source.get("os_volume_id")', script)
        self.assertIn('"pending_cleanup_not_before_epoch"', script)
        self.assertNotIn("resolve_volume_ids", script)
        self.assertIn("local command_status=$?", script)
        self.assertIn("trap - EXIT", script)
        self.assertIn('exit "$command_status"', script)
        self.assertIn("systemctl --user is-active --quiet", script)
        self.assertIn('UserKnownHostsFile="$SSH_KNOWN_HOSTS_FILE"', script)
        self.assertIn('install -m 0600 /dev/null "$SSH_KNOWN_HOSTS_FILE"', script)
        self.assertIn("fetch_failure_evidence", script)
        cleanup_body = script[script.index("cleanup_on_exit() {") :]
        self.assertLess(
            cleanup_body.index('fetch_failure_evidence "$vm_id"'),
            cleanup_body.index('"$DELETE_SCRIPT" "$STATE_FILE" --preserve-volume'),
        )
        self.assertIn("trap cleanup_on_exit EXIT", script)
        self.assertIn("trap 'exit 130' INT", script)
        self.assertIn("trap 'exit 143' TERM", script)
        self.assertIn("trap 'exit 129' HUP", script)
        self.assertNotIn("trap cleanup_on_exit EXIT INT TERM HUP", script)
        self.assertIn("VERDA_FRONTIER_DRY_RUN", script)
        self.assertIn("VERDA_FRONTIER_CANDIDATE:-quality", script)
        self.assertIn("frontier-complete.json", script)
        self.assertIn("publication_report_sha256", script)
        self.assertIn("/root/frontier.exit", script)
        self.assertIn("rm -f /root/frontier.pid", script)
        self.assertIn('[[ "$(cat /root/frontier.exit)" != 0 ]]', script)
        self.assertIn("assert json.load(sys.stdin) == []", script)
        self.assertIn('["total"]["hourly"]', script)

    def test_frontier_watchdog_deletes_only_state_file_vm(self) -> None:
        script = _FRONTIER_DELETE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('state="$(cat "$STATE_FILE")"', script)
        self.assertIn('state.get("vm_id")', script)
        self.assertIn('state["hostname"]', script)
        self.assertIn('description.get("os_volume_id")', script)
        self.assertIn("expected_name = f\"{state['hostname']}-os\"", script)
        self.assertIn('volume.get("name") == expected_name', script)
        self.assertIn("pending create is still inside its discovery window", script)
        self.assertIn('vm delete "$vm_id"', script)
        self.assertIn('CLEANUP_MODE="${2:-}"', script)
        self.assertIn("--preserve-volume | --delete-volume", script)
        self.assertIn('if [[ "$CLEANUP_MODE" == "--delete-volume" ]]', script)
        self.assertIn("--with-volumes", script)
        self.assertIn('volume delete "$volume_id"', script)
        self.assertIn("--yes", script)
        self.assertNotIn("--all", script)

    def test_frontier_watchdog_preserves_pending_state_during_discovery_window(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_verda = fake_bin / "verda"
            fake_verda.write_text(_FAKE_VERDA_CLI, encoding="utf-8")
            fake_verda.chmod(0o755)
            state_path = root / "frontier-state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "phase": "pending",
                        "vm_id": None,
                        "volume_ids": [],
                        "create_response": {},
                        "hostname": "deepseek-v4-quant-frontier",
                        "instance_type": "2A100.44V",
                        "location": "FIN-01",
                        "os_volume_size_gib": 350,
                        "pending_cleanup_not_before_epoch": 4_102_444_800,
                    }
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["FAKE_VERDA_ROOT"] = str(root)
            result = subprocess.run(
                [
                    "bash",
                    str(_FRONTIER_DELETE_SCRIPT),
                    str(state_path),
                    "--preserve-volume",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            state_preserved = state_path.exists()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inside its discovery window", result.stderr)
        self.assertTrue(state_preserved)

    def test_frontier_resume_dry_run_requires_exact_detached_volume(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_verda = fake_bin / "verda"
            fake_verda.write_text(
                _FAKE_FRONTIER_HOST_VERDA_CLI,
                encoding="utf-8",
            )
            fake_verda.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "HOME": str(root / "home"),
                    "FAKE_VERDA_ROOT": str(root),
                    "VERDA_FRONTIER_DRY_RUN": "1",
                    "VERDA_FRONTIER_RUN_MODE": "validate-resume",
                    "VERDA_FRONTIER_RESUME_VOLUME_ID": "volume-1",
                    "VERDA_FRONTIER_INSTANCE_TYPE": "CPU.4V.16G",
                    "VERDA_FRONTIER_LOCATION": "FIN-03",
                    "VERDA_FRONTIER_SSH_KEY_ID": "ssh-key-1",
                    "VERDA_FRONTIER_MAX_HOURS": "2",
                    "VERDA_FRONTIER_MAX_COST_USD": "0.25",
                    "VERDA_FRONTIER_EVIDENCE_DIRECTORY": str(root / "evidence"),
                }
            )

            result = subprocess.run(
                ["bash", str(_FRONTIER_HOST_SCRIPT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            calls = (root / "calls.log").read_text(encoding="utf-8")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("frontier_resume_volume_verified=true", result.stdout)
            self.assertNotIn("vm create", calls)

            environment["VERDA_FRONTIER_RESUME_VOLUME_ID"] = "wrong-volume"
            rejected = subprocess.run(
                ["bash", str(_FRONTIER_HOST_SCRIPT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_frontier_failure_cleanup_stops_compute_and_preserves_volume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_verda = fake_bin / "verda"
            fake_verda.write_text(_FAKE_VERDA_CLI, encoding="utf-8")
            fake_verda.chmod(0o755)
            (root / "vm-present").touch()
            (root / "volume-present").touch()
            state_path = root / "frontier-state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "phase": "active",
                        "vm_id": "vm-1",
                        "volume_ids": ["volume-1"],
                        "create_response": {},
                        "hostname": "deepseek-v4-quant-frontier",
                        "instance_type": "2A100.44V",
                        "location": "FIN-01",
                        "os_volume_size_gib": 350,
                        "pending_cleanup_not_before_epoch": 0,
                    }
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["FAKE_VERDA_ROOT"] = str(root)

            result = subprocess.run(
                [
                    "bash",
                    str(_FRONTIER_DELETE_SCRIPT),
                    str(state_path),
                    "--preserve-volume",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            preserved_state = json.loads(state_path.read_text(encoding="utf-8"))
            deletion_log = (root / "deletions.log").read_text(encoding="utf-8")

            self.assertFalse((root / "vm-present").exists())
            self.assertTrue((root / "volume-present").exists())
            self.assertEqual(preserved_state["phase"], "preserved")
            self.assertIsNone(preserved_state["vm_id"])
            self.assertEqual(preserved_state["volume_ids"], ["volume-1"])
            self.assertIn("vm vm-1", deletion_log)
            self.assertNotIn("--with-volumes", deletion_log)
            self.assertNotIn("volume volume-1", deletion_log)

    def test_frontier_watchdog_recovers_every_provisioning_crash_window(
        self,
    ) -> None:
        scenarios = {
            "active": {"state_vm_id": "vm-1", "state_volumes": ["volume-1"]},
            "pending-vm": {"state_vm_id": None, "state_volumes": []},
            "pending-detached": {"state_vm_id": None, "state_volumes": []},
        }
        for scenario, state_values in scenarios.items():
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                fake_bin = root / "bin"
                fake_bin.mkdir()
                fake_verda = fake_bin / "verda"
                fake_verda.write_text(_FAKE_VERDA_CLI, encoding="utf-8")
                fake_verda.chmod(0o755)
                if scenario != "pending-detached":
                    (root / "vm-present").touch()
                (root / "volume-present").touch()
                state_path = root / "frontier-state.json"
                state_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "phase": scenario,
                            "vm_id": state_values["state_vm_id"],
                            "volume_ids": state_values["state_volumes"],
                            "create_response": {},
                            "hostname": "deepseek-v4-quant-frontier",
                            "instance_type": "2A100.44V",
                            "location": "FIN-01",
                            "os_volume_size_gib": 350,
                            "pending_cleanup_not_before_epoch": 0,
                        }
                    ),
                    encoding="utf-8",
                )
                environment = os.environ.copy()
                environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
                environment["FAKE_VERDA_ROOT"] = str(root)
                result = subprocess.run(
                    [
                        "bash",
                        str(_FRONTIER_DELETE_SCRIPT),
                        str(state_path),
                        "--delete-volume",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(state_path.exists())
                self.assertFalse((root / "vm-present").exists())
                self.assertFalse((root / "volume-present").exists())
                deletion_log = (root / "deletions.log").read_text(encoding="utf-8")
                if scenario != "pending-detached":
                    self.assertIn("vm vm-1", deletion_log)
                self.assertIn("volume volume-1", deletion_log)
                self.assertNotIn("--all", deletion_log)

    def test_oracle_installs_vllm_test_and_jit_dependencies(self) -> None:
        script = _ORACLE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('PYTHON_DEV_VERSION="3.12.3-1ubuntu0.15"', script)
        self.assertIn('"python3.12-dev=$PYTHON_DEV_VERSION"', script)
        self.assertIn("/usr/include/python3.12/Python.h", script)
        self.assertIn('"ninja==1.13.0"', script)
        self.assertIn('"tblib==3.1.0"', script)
        self.assertIn('export PATH="$ORACLE_ENVIRONMENT/bin:$PATH"', script)
        self.assertIn("ninja --version", script)

    def test_frontier_runner_uses_fixed_gpu_workers_and_selected_disk_bound(
        self,
    ) -> None:
        script = _FRONTIER_SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(script.count('gpu_arguments+=(--gpu-device "$device")'), 1)
        self.assertEqual(script.count('"${gpu_arguments[@]}"'), 3)
        self.assertIn("deepseek-v4-inspect-frontier-gpus", script)
        self.assertLess(
            script.index("deepseek-v4-inspect-frontier-gpus"),
            script.index("Download pinned official checkpoint"),
        )
        self.assertIn("FRONTIER_GPU_DEVICES=(0)", script)
        self.assertIn('recipe["candidate_summaries"]', script)
        self.assertIn('summary.get("name") == selected_candidate', script)
        self.assertNotIn(
            'recipe["storage_summary"]["projected_local_peak_model_payload_bytes"]',
            script,
        )


if __name__ == "__main__":
    unittest.main()
