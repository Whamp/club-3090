#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
readonly STATE_FILE="${1:?usage: delete-verda-frontier-vm.sh STATE_FILE (--preserve-volume|--delete-volume)}"
readonly CLEANUP_MODE="${2:-}"
case "$CLEANUP_MODE" in
    --preserve-volume | --delete-volume) ;;
    *)
        echo "Frontier cleanup requires --preserve-volume or --delete-volume" >&2
        exit 2
        ;;
esac
[[ -f "$STATE_FILE" ]] || exit 0
state="$(cat "$STATE_FILE")"

vm_list="$(verda --agent vm list -o json)"
vm_id="$(python3 - "$state" "$vm_list" <<'PY'
import json
import sys
import time

state = json.loads(sys.argv[1])
vms = json.loads(sys.argv[2])
if state.get("schema_version") != 1:
    raise SystemExit("unsupported frontier cleanup state schema")
for key in ("hostname", "instance_type", "location", "os_volume_size_gib"):
    if not state.get(key):
        raise SystemExit(f"frontier cleanup state is missing {key}")
if not isinstance(vms, list):
    raise SystemExit("invalid Verda VM list response")

recorded_vm_id = state.get("vm_id")
if recorded_vm_id:
    matches = [vm for vm in vms if vm.get("id") == recorded_vm_id]
else:
    matches = [
        vm
        for vm in vms
        if vm.get("hostname") == state["hostname"]
        and vm.get("instance_type") == state["instance_type"]
        and vm.get("location") == state["location"]
    ]
if len(matches) > 1:
    raise SystemExit("multiple matching frontier VMs found")
if not matches:
    not_before = state.get("pending_cleanup_not_before_epoch")
    if state.get("phase") == "pending" and (
        not isinstance(not_before, int) or time.time() < not_before
    ):
        raise SystemExit("frontier pending create is still inside its discovery window")
    print("")
    raise SystemExit(0)
vm = matches[0]
for key in ("hostname", "instance_type", "location"):
    if vm.get(key) != state[key]:
        raise SystemExit(f"frontier VM identity mismatch for {key}")
vm_id = vm.get("id")
if not isinstance(vm_id, str) or not vm_id:
    raise SystemExit("matching frontier VM has no id")
print(vm_id)
PY
)"

if [[ -n "$vm_id" ]]; then
    vm_description="$(verda --agent vm describe "$vm_id" -o json)"
else
    vm_description='{}'
fi
volume_list="$(verda --agent volume list -o json)"
mapfile -t volume_ids < <(python3 - \
    "$state" "$vm_description" "$volume_list" "$vm_id" <<'PY'
import json
import sys

state = json.loads(sys.argv[1])
description = json.loads(sys.argv[2])
volumes = json.loads(sys.argv[3])
vm_id = sys.argv[4]
if not isinstance(volumes, list):
    raise SystemExit("invalid Verda volume list response")

expected_name = f"{state['hostname']}-os"
expected_location = state["location"]
expected_size = int(state["os_volume_size_gib"])
volume_ids = set()
for volume_id in state.get("volume_ids", []):
    if not isinstance(volume_id, str) or not volume_id:
        raise SystemExit("frontier cleanup state contains an invalid volume id")
    volume_ids.add(volume_id)

if description:
    for key, expected in (
        ("id", vm_id),
        ("hostname", state["hostname"]),
        ("instance_type", state["instance_type"]),
        ("location", expected_location),
    ):
        if description.get(key) != expected:
            raise SystemExit(f"frontier VM description mismatch for {key}")
    os_volume_id = description.get("os_volume_id")
    if os_volume_id is not None:
        volume_ids.add(os_volume_id)
    attached_volume_ids = description.get("volume_ids", [])
    if not isinstance(attached_volume_ids, list):
        raise SystemExit("frontier VM description has invalid volume_ids")
    volume_ids.update(attached_volume_ids)

matching_named_volumes = [
    volume
    for volume in volumes
    if volume.get("name") == expected_name
    and volume.get("location") == expected_location
    and volume.get("size") == expected_size
    and volume.get("is_os_volume") is True
    and volume.get("instance_id") in {None, vm_id or None}
]
if len(matching_named_volumes) > 1:
    raise SystemExit("multiple exact frontier OS volumes found")
for volume in matching_named_volumes:
    volume_id = volume.get("id")
    if not isinstance(volume_id, str) or not volume_id:
        raise SystemExit("matching frontier OS volume has no id")
    volume_ids.add(volume_id)

if any(not isinstance(value, str) or not value for value in volume_ids):
    raise SystemExit("frontier resource metadata contains an invalid volume id")
if len(volume_ids) > 1:
    raise SystemExit(f"frontier state identifies multiple volumes: {sorted(volume_ids)}")
if vm_id and not volume_ids:
    raise SystemExit("running frontier VM has no exact OS-volume identity")

listed_by_id = {
    volume.get("id"): volume
    for volume in volumes
    if isinstance(volume.get("id"), str)
}
for volume_id in volume_ids:
    volume = listed_by_id.get(volume_id)
    if volume is None:
        continue
    if (
        volume.get("name") != expected_name
        or volume.get("location") != expected_location
        or volume.get("size") != expected_size
        or volume.get("is_os_volume") is not True
        or volume.get("instance_id") not in {None, vm_id or None}
    ):
        raise SystemExit(f"frontier volume identity mismatch: {volume_id}")

for volume_id in sorted(volume_ids):
    print(volume_id)
PY
)

for _ in $(seq 1 60); do
    if [[ -n "$vm_id" ]]; then
        vm_delete_arguments=(
            --agent vm delete "$vm_id"
            --yes
            --wait
            --wait-timeout 10m
            -o json
        )
        if [[ "$CLEANUP_MODE" == "--delete-volume" ]]; then
            vm_delete_arguments+=(--with-volumes)
        fi
        verda "${vm_delete_arguments[@]}" >/dev/null 2>&1 || true
    fi
    if [[ "$CLEANUP_MODE" == "--delete-volume" ]]; then
        for volume_id in "${volume_ids[@]}"; do
            verda --agent volume delete "$volume_id" --yes -o json \
                >/dev/null 2>&1 || true
        done
    fi
    vms="$(verda --agent vm list -o json)"
    volumes="$(verda --agent volume list -o json)"
    if python3 - \
        "$STATE_FILE" "$state" "$vms" "$volumes" "$CLEANUP_MODE" \
        "$vm_id" "${volume_ids[*]}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

state_path = Path(sys.argv[1])
state = json.loads(sys.argv[2])
vms = json.loads(sys.argv[3])
volumes = json.loads(sys.argv[4])
cleanup_mode = sys.argv[5]
vm_id = sys.argv[6]
volume_ids = tuple(value for value in sys.argv[7].split() if value)
if not isinstance(vms, list) or not isinstance(volumes, list):
    raise SystemExit(1)
if vm_id and any(vm.get("id") == vm_id for vm in vms):
    raise SystemExit(1)
volumes_by_id = {
    volume.get("id"): volume
    for volume in volumes
    if isinstance(volume.get("id"), str)
}
if cleanup_mode == "--delete-volume":
    if any(volume_id in volumes_by_id for volume_id in volume_ids):
        raise SystemExit(1)
    state_path.unlink(missing_ok=True)
    raise SystemExit(0)
if not volume_ids:
    state_path.unlink(missing_ok=True)
    raise SystemExit(0)
for volume_id in volume_ids:
    volume = volumes_by_id.get(volume_id)
    if volume is None:
        raise SystemExit(1)
    if volume.get("instance_id") not in (None, "") or volume.get("instances"):
        raise SystemExit(1)
state["phase"] = "preserved"
state["vm_id"] = None
state["volume_ids"] = list(volume_ids)
state["preserved_compute_instance_id"] = vm_id or None
state["preserved_at_epoch"] = int(time.time())
temporary = state_path.with_name(f".{state_path.name}.writing")
temporary.write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
os.replace(temporary, state_path)
directory_fd = os.open(state_path.parent, os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
    then
        exit 0
    fi
    sleep 10
done

echo "Frontier cleanup could not prove the requested VM and volume state" >&2
exit 1
