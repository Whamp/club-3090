#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
export VERDA_PROFILE="${VERDA_FRONTIER_PROFILE:-main}"

readonly RUN_MODE="${VERDA_FRONTIER_RUN_MODE:-fresh}"
case "$RUN_MODE" in
    fresh)
        default_instance_kind="gpu"
        default_instance_type="1A100.22V"
        default_max_hours="16"
        default_max_cost_usd="30.25"
        ;;
    validate-resume)
        default_instance_kind="cpu"
        default_instance_type="CPU.4V.16G"
        default_max_hours="2"
        default_max_cost_usd="0.25"
        ;;
    resume-conversion)
        default_instance_kind="gpu"
        default_instance_type="1A100.22V"
        default_max_hours="12"
        default_max_cost_usd="22.64"
        ;;
    *)
        echo "Unknown frontier run mode: $RUN_MODE" >&2
        exit 2
        ;;
esac
readonly INSTANCE_KIND="${VERDA_FRONTIER_INSTANCE_KIND:-$default_instance_kind}"
readonly INSTANCE_TYPE="${VERDA_FRONTIER_INSTANCE_TYPE:-$default_instance_type}"
readonly LOCATION="${VERDA_FRONTIER_LOCATION:-FIN-03}"
readonly OS_IMAGE="${VERDA_FRONTIER_OS_IMAGE:-ubuntu-24.04-cuda-13.0-open-docker}"
readonly OS_VOLUME_GIB="${VERDA_FRONTIER_OS_VOLUME_GIB:-350}"
readonly RESUME_VOLUME_ID="${VERDA_FRONTIER_RESUME_VOLUME_ID:-}"
readonly RESUME_VOLUME_NAME="${VERDA_FRONTIER_RESUME_VOLUME_NAME:-deepseek-v4-quant-frontier-os}"
readonly RECOVERY_MANIFEST_SHA256="0788715e8373daed48bbedaee11fedee217c798a2f189fee00473c9a9913a480"
readonly SSH_KEY_ID="${VERDA_FRONTIER_SSH_KEY_ID:-2a039811-2dc0-4785-9fb5-2694c9d98f1b}"
readonly SSH_IDENTITY="${VERDA_FRONTIER_SSH_IDENTITY:-$HOME/.ssh/id_ed25519}"
readonly -a SSH_OPTIONS=(
    -i "$SSH_IDENTITY"
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -o StrictHostKeyChecking=accept-new
)
readonly HOSTNAME="${VERDA_FRONTIER_HOSTNAME:-deepseek-v4-quant-frontier}"
readonly MAX_HOURS="${VERDA_FRONTIER_MAX_HOURS:-$default_max_hours}"
readonly MAX_COST_USD="${VERDA_FRONTIER_MAX_COST_USD:-$default_max_cost_usd}"
readonly REMOTE_ROOT="${VERDA_FRONTIER_REMOTE_ROOT:-/root/deepseek-v4-quant-frontier}"
readonly REPOSITORY="${VERDA_FRONTIER_REPOSITORY:-hampsonw/DeepSeek-V4-Flash-0731-WNA16}"
readonly BRANCH_PREFIX="${VERDA_FRONTIER_BRANCH_PREFIX:-frontier-20260813}"
readonly CANDIDATE="${VERDA_FRONTIER_CANDIDATE:-quality}"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIRECTORY/../../.." && pwd)"
readonly REPOSITORY_ROOT
readonly STATE_DIRECTORY="${VERDA_FRONTIER_STATE_DIRECTORY:-$HOME/.local/state/club-3090}"
readonly STATE_FILE="$STATE_DIRECTORY/deepseek-v4-quant-frontier-verda.json"
readonly SSH_KNOWN_HOSTS_FILE="$STATE_DIRECTORY/deepseek-v4-quant-frontier-known-hosts"
readonly DELETE_SCRIPT="$SCRIPT_DIRECTORY/delete-verda-frontier-vm.sh"
readonly WATCHDOG_UNIT="deepseek-v4-quant-frontier-delete"
readonly LOCAL_EVIDENCE_DIRECTORY="${VERDA_FRONTIER_EVIDENCE_DIRECTORY:-$PWD/.research/deepseek-v4-lowbit/frontier-verda-evidence}"
readonly LOCAL_RECOVERY_MANIFEST="${VERDA_FRONTIER_RECOVERY_MANIFEST:-$SCRIPT_DIRECTORY/frontier-recovery-manifest-20260813.json}"
readonly DRY_RUN="${VERDA_FRONTIER_DRY_RUN:-0}"

mkdir -p "$STATE_DIRECTORY" "$LOCAL_EVIDENCE_DIRECTORY"
install -m 0600 /dev/null "$SSH_KNOWN_HOSTS_FILE"

verda_json() {
    verda --agent "$@" -o json
}

require_account_contract() {
    local vms volumes
    vms="$(verda_json vm list)"
    volumes="$(verda_json volume list)"
    python3 - "$vms" "$volumes" "$RUN_MODE" "$RESUME_VOLUME_ID" <<'PY'
import json
import sys
vms, volumes = map(json.loads, sys.argv[1:3])
run_mode, resume_volume_id = sys.argv[3:5]
if vms:
    raise SystemExit("Frontier provisioning requires zero existing Verda VMs")
if run_mode == "fresh":
    if volumes:
        raise SystemExit("Fresh frontier provisioning requires zero existing volumes")
    raise SystemExit(0)
if not resume_volume_id:
    raise SystemExit("Frontier resume requires VERDA_FRONTIER_RESUME_VOLUME_ID")
if not isinstance(volumes, list) or len(volumes) != 1:
    raise SystemExit("Frontier resume requires exactly one existing Verda volume")
if volumes[0].get("id") != resume_volume_id:
    raise SystemExit("Frontier resume volume id differs from the only existing volume")
PY
}

require_reusable_state_file() {
    [[ ! -e "$STATE_FILE" ]] && return 0
    [[ "$RUN_MODE" != "fresh" ]] || {
        echo "Frontier Verda state file already exists: $STATE_FILE" >&2
        return 2
    }
    python3 - "$STATE_FILE" "$RESUME_VOLUME_ID" <<'PY'
import json
import sys
from pathlib import Path
state_path = Path(sys.argv[1])
resume_volume_id = sys.argv[2]
state = json.loads(state_path.read_text(encoding="utf-8"))
if (
    state.get("schema_version") != 1
    or state.get("phase") != "preserved"
    or state.get("vm_id") is not None
    or state.get("volume_ids") != [resume_volume_id]
):
    raise SystemExit("Frontier resume state does not identify the preserved volume")
PY
}

verify_live_contract() {
    local availability images keys balance estimate volume
    availability="$(verda_json vm availability --type "$INSTANCE_TYPE")"
    keys="$(verda_json ssh-key list)"
    balance="$(verda_json cost balance)"
    if [[ "$RUN_MODE" == "fresh" ]]; then
        images="$(verda_json images --type "$INSTANCE_TYPE")"
        estimate="$(verda_json cost estimate \
            --type "$INSTANCE_TYPE" --os-volume "$OS_VOLUME_GIB")"
        volume='{}'
    else
        images='[]'
        estimate="$(verda_json cost estimate --type "$INSTANCE_TYPE")"
        volume="$(verda_json volume describe "$RESUME_VOLUME_ID")"
    fi
    python3 - \
        "$availability" "$images" "$keys" "$balance" "$estimate" "$volume" \
        "$RUN_MODE" "$INSTANCE_TYPE" "$LOCATION" "$OS_IMAGE" "$SSH_KEY_ID" \
        "$MAX_HOURS" "$MAX_COST_USD" "$RESUME_VOLUME_ID" \
        "$RESUME_VOLUME_NAME" "$OS_VOLUME_GIB" <<'PY'
import json
import sys
availability, images, keys, balance, estimate, volume = map(json.loads, sys.argv[1:7])
(
    run_mode,
    instance_type,
    location,
    image,
    key_id,
    max_hours,
    max_cost,
    resume_volume_id,
    resume_volume_name,
    os_volume_gib,
) = sys.argv[7:17]
max_hours = float(max_hours)
max_cost = float(max_cost)
os_volume_gib = int(os_volume_gib)
matching = [
    item
    for item in (availability or [])
    if item.get("location") == location
    and item.get("instance_type") == instance_type
]
if len(matching) != 1:
    raise SystemExit("Frontier instance is not uniquely available at the pinned location")
if not any(item.get("id") == key_id for item in keys):
    raise SystemExit("Frontier SSH key is unavailable")
if run_mode == "fresh":
    if not any(item.get("image_type") == image for item in images):
        raise SystemExit("Frontier OS image is unavailable for the selected instance")
    storage_hourly = 0.0
else:
    expected_volume = {
        "id": resume_volume_id,
        "name": resume_volume_name,
        "size": os_volume_gib,
        "type": "NVMe",
        "status": "detached",
        "location": location,
        "contract": "PAY_AS_YOU_GO",
        "is_os_volume": True,
        "instance_id": None,
        "instances": [],
    }
    for field, expected in expected_volume.items():
        if volume.get(field) != expected:
            raise SystemExit(
                f"Frontier resume volume identity mismatch: "
                f"{field}={volume.get(field)!r}, expected={expected!r}"
            )
    storage_hourly = float(volume.get("base_hourly_cost", -1))
    if storage_hourly < 0:
        raise SystemExit("Frontier resume volume has no valid hourly cost")
    print(
        f"frontier_resume_volume_verified=true id={resume_volume_id} "
        f"size_gib={os_volume_gib} location={location}"
    )
compute_hourly = float(estimate["total"]["hourly"])
hourly = compute_hourly + storage_hourly
projected = hourly * max_hours
if projected > max_cost:
    raise SystemExit(
        f"Frontier projected cost ${projected:.2f} exceeds cap ${max_cost:.2f}"
    )
if projected > float(balance["amount"]):
    raise SystemExit("Frontier projected cost exceeds account balance")
print(
    f"frontier_available=true hourly_usd={hourly:.4f} "
    f"compute_hourly_usd={compute_hourly:.4f} "
    f"storage_hourly_usd={storage_hourly:.4f} "
    f"max_hours={max_hours:g} projected_usd={projected:.2f} "
    f"balance_usd={float(balance['amount']):.2f}"
)
PY
}

write_state() {
    local phase="$1"
    local vm_id="$2"
    local volume_ids_json="$3"
    local create_response="$4"
    python3 - \
        "$STATE_FILE" "$phase" "$vm_id" "$volume_ids_json" \
        "$create_response" "$HOSTNAME" "$INSTANCE_TYPE" "$LOCATION" \
        "$OS_VOLUME_GIB" "$RUN_MODE" <<'PY'
import json
import os
import sys
import time
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "phase": sys.argv[2],
    "vm_id": sys.argv[3] or None,
    "volume_ids": json.loads(sys.argv[4]),
    "create_response": json.loads(sys.argv[5]),
    "hostname": sys.argv[6],
    "instance_type": sys.argv[7],
    "location": sys.argv[8],
    "os_volume_size_gib": int(sys.argv[9]),
    "run_mode": sys.argv[10],
    "pending_cleanup_not_before_epoch": int(time.time()) + 30 * 60,
}
temporary = path.with_name(f".{path.name}.writing")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
os.replace(temporary, path)
os.sync()
PY
}

extract_volume_ids() {
    local create_response="$1"
    python3 - "$create_response" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
sources = [payload]
for key in ("instance", "vm"):
    nested = payload.get(key)
    if isinstance(nested, dict):
        sources.append(nested)
volume_ids = []
for source in sources:
    os_volume_id = source.get("os_volume_id")
    if os_volume_id is not None:
        volume_ids.append(os_volume_id)
    attached_volume_ids = source.get("volume_ids", [])
    if not isinstance(attached_volume_ids, list):
        raise SystemExit("Frontier create response has invalid volume_ids")
    volume_ids.extend(attached_volume_ids)
if any(not isinstance(value, str) or not value for value in volume_ids):
    raise SystemExit("Frontier create response contains an invalid volume id")
unique_volume_ids = sorted(set(volume_ids))
if len(unique_volume_ids) != 1:
    raise SystemExit(
        "Frontier create response must identify exactly one OS volume, "
        f"found {unique_volume_ids}"
    )
print(json.dumps(unique_volume_ids))
PY
}

extract_vm_id() {
    local create_response="$1"
    python3 - "$create_response" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
candidates = [
    payload.get("id"),
    payload.get("instance_id"),
    (payload.get("instance") or {}).get("id") if isinstance(payload.get("instance"), dict) else None,
    (payload.get("vm") or {}).get("id") if isinstance(payload.get("vm"), dict) else None,
]
values = [value for value in candidates if isinstance(value, str) and value]
if len(set(values)) != 1:
    raise SystemExit(f"Cannot extract unique Verda VM id from create response: {payload}")
print(values[0])
PY
}

vm_public_ip() {
    local vm_id="$1"
    local description
    description="$(verda_json vm describe "$vm_id")"
    python3 - "$description" "$vm_id" "$HOSTNAME" "$INSTANCE_TYPE" "$LOCATION" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
vm_id, hostname, instance_type, location = sys.argv[2:]
for key, expected in (
    ("id", vm_id),
    ("hostname", hostname),
    ("instance_type", instance_type),
    ("location", location),
):
    actual = payload.get(key)
    if actual is not None and actual != expected:
        raise SystemExit(f"Frontier VM description mismatch: {key}={actual!r}")
for key in ("public_ip", "ip_address", "ip"):
    value = payload.get(key)
    if isinstance(value, str) and value:
        print(value)
        break
else:
    raise SystemExit(f"No public IP in Verda VM description: {payload}")
PY
}

arm_watchdog() {
    local deadline
    deadline="$(python3 - "$MAX_HOURS" <<'PY'
from datetime import datetime, timedelta, timezone
import sys
deadline = datetime.now(timezone.utc) + timedelta(hours=float(sys.argv[1]))
print(deadline.strftime("%Y-%m-%d %H:%M:%S UTC"))
PY
)"
    systemd-run --user \
        --unit "$WATCHDOG_UNIT" \
        --on-calendar "$deadline" \
        --timer-property=Persistent=true \
        --service-type=exec \
        --property=Restart=on-failure \
        --property=RestartSec=60s \
        --setenv="VERDA_PROFILE=$VERDA_PROFILE" \
        "$DELETE_SCRIPT" "$STATE_FILE" --preserve-volume
    systemctl --user is-active --quiet "$WATCHDOG_UNIT.timer"
}

cancel_watchdog() {
    systemctl --user stop "$WATCHDOG_UNIT.timer" "$WATCHDOG_UNIT.service" \
        >/dev/null 2>&1 || true
    systemctl --user reset-failed "$WATCHDOG_UNIT.service" >/dev/null 2>&1 || true
    if systemctl --user is-active --quiet "$WATCHDOG_UNIT.timer" || \
        systemctl --user is-active --quiet "$WATCHDOG_UNIT.service"; then
        echo "Frontier deletion watchdog remained active after cancellation" >&2
        return 1
    fi
}

run_remote_ssh() {
    local address="$1"
    shift
    # Callers pass literals or commands assembled with printf %q for remote parsing.
    # shellcheck disable=SC2029
    ssh "${SSH_OPTIONS[@]}" \
        -o UserKnownHostsFile="$SSH_KNOWN_HOSTS_FILE" \
        "root@$address" "$@"
}

wait_for_ssh() {
    local vm_id="$1"
    local address
    address="$(vm_public_ip "$vm_id")"
    for _ in $(seq 1 60); do
        if run_remote_ssh "$address" true >/dev/null 2>&1; then
            return 0
        fi
        sleep 10
    done
    return 1
}

copy_runner() {
    local vm_id="$1"
    local address
    address="$(vm_public_ip "$vm_id")"
    run_remote_ssh "$address" \
        'cat > /root/run-verda-quant-frontier.sh && chmod 0700 /root/run-verda-quant-frontier.sh' \
        < "$SCRIPT_DIRECTORY/run-verda-quant-frontier.sh"
    if [[ "$RUN_MODE" != "fresh" ]]; then
        printf '%s  %s\n' "$RECOVERY_MANIFEST_SHA256" "$LOCAL_RECOVERY_MANIFEST" | \
            sha256sum --check --strict
        run_remote_ssh "$address" \
            'cat > /root/frontier-recovery-manifest.json && chmod 0400 /root/frontier-recovery-manifest.json' \
            < "$LOCAL_RECOVERY_MANIFEST"
    fi
}

start_remote_campaign() {
    local vm_id="$1"
    local address remote_command stage_command token_file
    address="$(vm_public_ip "$vm_id")"
    printf -v stage_command '%q ' \
        /root/run-verda-quant-frontier.sh \
        "$REMOTE_ROOT" "$REPOSITORY" "$BRANCH_PREFIX" "$CANDIDATE" \
        "$RUN_MODE" "$RESUME_VOLUME_ID" /root/frontier-recovery-manifest.json
    # The remote bash process expands status; the host must preserve these literals.
    # shellcheck disable=SC2016
    stage_command+='; status=$?; printf "%s\n" "$status" > /root/frontier.exit; rm -f /root/frontier.pid; exit "$status"'
    printf -v remote_command \
        'rm -f %q; nohup bash -c %q > %q 2>&1 & echo $! > %q' \
        /root/frontier.exit "$stage_command" \
        /root/frontier-nohup.log /root/frontier.pid
    if [[ "$RUN_MODE" == "validate-resume" ]]; then
        run_remote_ssh "$address" "$remote_command"
        return
    fi
    token_file="$(mktemp)"
    chmod 0600 "$token_file"
    trap 'rm -f "$token_file"' RETURN
    env -u HF_TOKEN hf auth token > "$token_file"
    remote_command="read -r HF_TOKEN; export HF_TOKEN; $remote_command"
    {
        cat "$token_file"
        printf '\n'
    } | run_remote_ssh "$address" "$remote_command"
}

wait_remote_campaign() {
    local vm_id="$1"
    local address deadline_epoch remote_command remote_state
    address="$(vm_public_ip "$vm_id")"
    printf -v remote_command '%q ' env \
        "REMOTE_ROOT=$REMOTE_ROOT" "RUN_MODE=$RUN_MODE" bash -s
    deadline_epoch="$(python3 - "$MAX_HOURS" <<'PY'
import sys
import time
print(int(time.time() + float(sys.argv[1]) * 3600 + 900))
PY
)"
    while (("$(date +%s)" < deadline_epoch)); do
        if ! verda_json vm describe "$vm_id" >/dev/null; then
            sleep 30
            continue
        fi
        if ! remote_state="$(run_remote_ssh "$address" \
            "$remote_command" <<'EOF'
set -euo pipefail
completion="$REMOTE_ROOT/reports/frontier-complete.json"
publication="$REMOTE_ROOT/reports/frontier-publication.json"
resume_validation="$REMOTE_ROOT/reports/frontier-resume-validation.json"
if [[ -f /root/frontier.pid ]] && \
    kill -0 "$(cat /root/frontier.pid)" 2>/dev/null; then
    echo running
elif [[ ! -f /root/frontier.exit ]] || \
    [[ "$(cat /root/frontier.exit)" != 0 ]]; then
    echo failed
elif [[ "$RUN_MODE" == "validate-resume" && -f "$resume_validation" ]]; then
    echo complete
elif [[ -f "$completion" ]]; then
    if python3 - "$completion" "$publication" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
completion_path, publication_path = map(Path, sys.argv[1:])
receipt = json.loads(completion_path.read_text(encoding="utf-8"))
publication_bytes = publication_path.read_bytes()
publication = json.loads(publication_bytes)
valid = (
    receipt.get("schema_version") == 1
    and len(publication.get("candidates", [])) == 1
    and receipt.get("candidate") == publication["candidates"][0].get("candidate")
    and receipt.get("revision") == publication["candidates"][0].get("revision")
    and receipt.get("publication_report_sha256")
    == hashlib.sha256(publication_bytes).hexdigest()
)
raise SystemExit(0 if valid else 1)
PY
    then
        echo complete
    else
        echo failed
    fi
else
    echo failed
fi
EOF
)"; then
            sleep 30
            continue
        fi
        case "$remote_state" in
            complete) return 0 ;;
            running) sleep 60 ;;
            *)
                run_remote_ssh "$address" \
                    "$remote_command" <<'EOF'
tail -200 /root/frontier-nohup.log 2>/dev/null || true
tail -200 "$REMOTE_ROOT/reports/run-verda-frontier-resume-validation.log" 2>/dev/null || true
tail -200 "$REMOTE_ROOT/reports/run-verda-frontier-resume-conversion.log" 2>/dev/null || true
tail -200 "$REMOTE_ROOT/reports/run-verda-quant-frontier.log" 2>/dev/null || true
EOF
                return 1
                ;;
        esac
    done
    echo "Frontier host wait exceeded the guarded rental deadline" >&2
    return 1
}

fetch_reports_archive() {
    local vm_id="$1"
    local destination="$2"
    local address archive remote_command temporary_archive
    address="$(vm_public_ip "$vm_id")"
    mkdir -p "$destination"
    archive="$destination/frontier-reports.tar.gz"
    temporary_archive="$archive.writing"
    rm -f "$temporary_archive"
    printf -v remote_command '%q ' tar -C "$REMOTE_ROOT" -czf - reports
    if ! run_remote_ssh "$address" "$remote_command" > "$temporary_archive"; then
        rm -f "$temporary_archive"
        return 1
    fi
    mv "$temporary_archive" "$archive"
    tar -C "$destination" -xzf "$archive"
}

fetch_evidence() {
    fetch_reports_archive "$1" "$LOCAL_EVIDENCE_DIRECTORY"
}

fetch_failure_evidence() {
    fetch_reports_archive "$1" "$LOCAL_EVIDENCE_DIRECTORY/failure"
}

verify_fetched_evidence() {
    if [[ "$RUN_MODE" == "validate-resume" ]]; then
        PYTHONPATH="$REPOSITORY_ROOT/tools/deepseek-v4-lowbit/src" \
            python3 -m deepseek_v4_lowbit.frontier_resume_receipt_cli \
            "$LOCAL_EVIDENCE_DIRECTORY/reports/frontier-resume-validation.json" \
            --volume-id "$RESUME_VOLUME_ID" \
            --candidate "$CANDIDATE" \
            --recovery-manifest-sha256 "$RECOVERY_MANIFEST_SHA256"
        return
    fi
    python3 - \
        "$LOCAL_EVIDENCE_DIRECTORY/reports/frontier-complete.json" \
        "$LOCAL_EVIDENCE_DIRECTORY/reports/frontier-publication.json" \
        "$CANDIDATE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
completion_path, publication_path = map(Path, sys.argv[1:3])
expected_candidate = sys.argv[3]
receipt = json.loads(completion_path.read_text(encoding="utf-8"))
publication_bytes = publication_path.read_bytes()
publication = json.loads(publication_bytes)
if (
    receipt.get("schema_version") != 1
    or receipt.get("candidate") != expected_candidate
    or publication.get("selected_candidate_names") != [expected_candidate]
    or len(publication.get("candidates", [])) != 1
    or receipt.get("revision") != publication["candidates"][0].get("revision")
    or receipt.get("publication_report_sha256")
    != hashlib.sha256(publication_bytes).hexdigest()
):
    raise SystemExit("Fetched frontier completion evidence is inconsistent")
print(
    f"frontier_evidence_verified=true candidate={expected_candidate} "
    f"revision={receipt['revision']}"
)
PY
}

case "$CANDIDATE" in
    quality) ;;
    cliff | capacity | balanced)
        echo "Frontier recovery is quality-first; lower candidates require a later campaign" >&2
        exit 2
        ;;
    *)
        echo "Unknown frontier candidate: $CANDIDATE" >&2
        exit 2
        ;;
esac
require_account_contract
require_reusable_state_file
verify_live_contract
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Frontier Verda dry-run complete; no resource created"
    exit 0
fi
cleanup_needed=1
create_started=0
vm_id=""
cleanup_on_exit() {
    local command_status=$?
    trap - EXIT INT TERM HUP
    if [[ "$cleanup_needed" == 1 ]]; then
        if [[ "$create_started" == 0 ]]; then
            rm -f "$STATE_FILE"
            if ! cancel_watchdog; then
                command_status=1
            fi
        else
            if [[ -n "$vm_id" ]] && ! fetch_failure_evidence "$vm_id"; then
                echo "Could not capture frontier failure reports before cleanup" >&2
            fi
            if "$DELETE_SCRIPT" "$STATE_FILE" --preserve-volume; then
                if ! cancel_watchdog; then
                    command_status=1
                fi
            else
                echo "Immediate cleanup failed; starting retrying watchdog service" >&2
                systemctl --user start "$WATCHDOG_UNIT.service" || true
                command_status=1
            fi
        fi
    fi
    exit "$command_status"
}
trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if [[ "$RUN_MODE" == "fresh" ]]; then
    rm -f "$STATE_FILE"
    pending_volume_ids='[]'
else
    pending_volume_ids="[\"$RESUME_VOLUME_ID\"]"
fi
write_state pending "" "$pending_volume_ids" '{}'
arm_watchdog
create_started=1
create_arguments=(
    vm create
    --kind "$INSTANCE_KIND"
    --instance-type "$INSTANCE_TYPE"
    --location "$LOCATION"
    --contract pay_as_go
    --hostname "$HOSTNAME"
    --ssh-key "$SSH_KEY_ID"
    --wait
    --wait-timeout 10m
)
if [[ "$RUN_MODE" == "fresh" ]]; then
    create_arguments+=(
        --os "$OS_IMAGE"
        --os-volume-size "$OS_VOLUME_GIB"
    )
else
    create_arguments+=(--os "$RESUME_VOLUME_ID")
fi
create_response="$(verda_json "${create_arguments[@]}")"
vm_id="$(extract_vm_id "$create_response")"
volume_ids_json="$(extract_volume_ids "$create_response")"
write_state active "$vm_id" "$volume_ids_json" "$create_response"

wait_for_ssh "$vm_id"
copy_runner "$vm_id"
start_remote_campaign "$vm_id"
wait_remote_campaign "$vm_id"
fetch_evidence "$vm_id"
verify_fetched_evidence
if [[ "$RUN_MODE" == "validate-resume" ]]; then
    "$DELETE_SCRIPT" "$STATE_FILE" --preserve-volume
else
    "$DELETE_SCRIPT" "$STATE_FILE" --delete-volume
fi
cancel_watchdog
cleanup_needed=0

verda_json vm list | python3 -c 'import json,sys; assert json.load(sys.stdin) == []'
if [[ "$RUN_MODE" == "validate-resume" ]]; then
    final_volumes="$(verda_json volume list)"
    python3 - "$final_volumes" "$RESUME_VOLUME_ID" <<'PY'
import json
import sys
volumes = json.loads(sys.argv[1])
expected_volume_id = sys.argv[2]
assert len(volumes) == 1
assert volumes[0]["id"] == expected_volume_id
assert volumes[0]["status"] == "detached"
assert volumes[0]["instance_id"] is None
PY
else
    verda_json volume list | python3 -c 'import json,sys; assert json.load(sys.stdin) == []'
fi
verda_json cost running | python3 -c '
import json,sys
assert float(json.load(sys.stdin)["total"]["hourly"]) == 0.0
'
echo "Frontier Verda $RUN_MODE campaign complete; evidence downloaded"
