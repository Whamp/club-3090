#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
export VERDA_PROFILE="${VERDA_FRONTIER_PROFILE:-main}"

readonly INSTANCE_TYPE="${VERDA_FRONTIER_INSTANCE_TYPE:-2A100.44V}"
readonly LOCATION="${VERDA_FRONTIER_LOCATION:-FIN-01}"
readonly OS_IMAGE="${VERDA_FRONTIER_OS_IMAGE:-ubuntu-24.04-cuda-13.0-open-docker}"
readonly OS_VOLUME_GIB="${VERDA_FRONTIER_OS_VOLUME_GIB:-350}"
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
readonly MAX_HOURS="${VERDA_FRONTIER_MAX_HOURS:-8}"
readonly MAX_COST_USD="${VERDA_FRONTIER_MAX_COST_USD:-29.50}"
readonly REMOTE_ROOT="${VERDA_FRONTIER_REMOTE_ROOT:-/root/deepseek-v4-quant-frontier}"
readonly REPOSITORY="${VERDA_FRONTIER_REPOSITORY:-hampsonw/DeepSeek-V4-Flash-0731-WNA16}"
readonly BRANCH_PREFIX="${VERDA_FRONTIER_BRANCH_PREFIX:-frontier-20260813}"
readonly CANDIDATE="${VERDA_FRONTIER_CANDIDATE:-quality}"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly STATE_DIRECTORY="${VERDA_FRONTIER_STATE_DIRECTORY:-$HOME/.local/state/club-3090}"
readonly STATE_FILE="$STATE_DIRECTORY/deepseek-v4-quant-frontier-verda.json"
readonly DELETE_SCRIPT="$SCRIPT_DIRECTORY/delete-verda-frontier-vm.sh"
readonly WATCHDOG_UNIT="deepseek-v4-quant-frontier-delete"
readonly LOCAL_EVIDENCE_DIRECTORY="${VERDA_FRONTIER_EVIDENCE_DIRECTORY:-$PWD/.research/deepseek-v4-lowbit/frontier-verda-evidence}"
readonly DRY_RUN="${VERDA_FRONTIER_DRY_RUN:-0}"

mkdir -p "$STATE_DIRECTORY" "$LOCAL_EVIDENCE_DIRECTORY"

verda_json() {
    verda --agent "$@" -o json
}

require_empty_account() {
    local vms volumes
    vms="$(verda_json vm list)"
    volumes="$(verda_json volume list)"
    python3 - "$vms" "$volumes" <<'PY'
import json
import sys
if json.loads(sys.argv[1]):
    raise SystemExit("Frontier provisioning requires zero existing Verda VMs")
if json.loads(sys.argv[2]):
    raise SystemExit("Frontier provisioning requires zero existing Verda volumes")
PY
}

verify_live_contract() {
    local availability images keys balance estimate
    availability="$(verda_json vm availability --type "$INSTANCE_TYPE")"
    images="$(verda_json images --type "$INSTANCE_TYPE")"
    keys="$(verda_json ssh-key list)"
    balance="$(verda_json cost balance)"
    estimate="$(verda_json cost estimate --type "$INSTANCE_TYPE" --os-volume "$OS_VOLUME_GIB")"
    python3 - \
        "$availability" "$images" "$keys" "$balance" "$estimate" \
        "$LOCATION" "$OS_IMAGE" "$SSH_KEY_ID" "$MAX_HOURS" "$MAX_COST_USD" <<'PY'
import json
import sys
availability, images, keys, balance, estimate = map(json.loads, sys.argv[1:6])
location, image, key_id = sys.argv[6:9]
max_hours, max_cost = map(float, sys.argv[9:11])
matching = [item for item in (availability or []) if item["location"] == location]
if len(matching) != 1:
    raise SystemExit("Frontier instance is not uniquely available at the pinned location")
if not any(item["image_type"] == image for item in images):
    raise SystemExit("Frontier OS image is unavailable for the selected instance")
if not any(item["id"] == key_id for item in keys):
    raise SystemExit("Frontier SSH key is unavailable")
hourly = float(estimate["total"]["hourly"])
projected = hourly * max_hours
if projected > max_cost:
    raise SystemExit(
        f"Frontier projected cost ${projected:.2f} exceeds cap ${max_cost:.2f}"
    )
if projected > float(balance["amount"]):
    raise SystemExit("Frontier projected cost exceeds account balance")
print(
    f"frontier_available=true hourly_usd={hourly:.4f} "
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
        "$OS_VOLUME_GIB" <<'PY'
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
        "$DELETE_SCRIPT" "$STATE_FILE" --finalize-state
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
    ssh "${SSH_OPTIONS[@]}" "root@$address" "$@"
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
}

start_remote_campaign() {
    local vm_id="$1"
    local address remote_command token_file
    address="$(vm_public_ip "$vm_id")"
    token_file="$(mktemp)"
    chmod 0600 "$token_file"
    trap 'rm -f "$token_file"' RETURN
    env -u HF_TOKEN hf auth token > "$token_file"
    printf -v remote_command \
        'read -r HF_TOKEN; export HF_TOKEN; nohup %q %q %q %q %q > %q 2>&1 & echo $! > %q' \
        /root/run-verda-quant-frontier.sh \
        "$REMOTE_ROOT" "$REPOSITORY" "$BRANCH_PREFIX" "$CANDIDATE" \
        /root/frontier-nohup.log /root/frontier.pid
    {
        cat "$token_file"
        printf '\n'
    } | run_remote_ssh "$address" "$remote_command"
}

wait_remote_campaign() {
    local vm_id="$1"
    local address deadline_epoch remote_command remote_state
    address="$(vm_public_ip "$vm_id")"
    printf -v remote_command '%q ' env "REMOTE_ROOT=$REMOTE_ROOT" bash -s
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
if [[ -f "$completion" ]]; then
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
        if [[ -f /root/frontier.pid ]] && \
            kill -0 "$(cat /root/frontier.pid)" 2>/dev/null; then
            echo running
        else
            echo complete
        fi
    else
        echo failed
    fi
elif [[ -f /root/frontier.pid ]] && kill -0 "$(cat /root/frontier.pid)" 2>/dev/null; then
    echo running
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
tail -200 "$REMOTE_ROOT/reports/run-verda-quant-frontier.log" 2>/dev/null || true
EOF
                return 1
                ;;
        esac
    done
    echo "Frontier host wait exceeded the guarded rental deadline" >&2
    return 1
}

fetch_evidence() {
    local vm_id="$1"
    local address remote_command
    address="$(vm_public_ip "$vm_id")"
    printf -v remote_command '%q ' tar -C "$REMOTE_ROOT" -czf - reports
    run_remote_ssh "$address" "$remote_command" \
        > "$LOCAL_EVIDENCE_DIRECTORY/frontier-reports.tar.gz"
    tar -C "$LOCAL_EVIDENCE_DIRECTORY" -xzf \
        "$LOCAL_EVIDENCE_DIRECTORY/frontier-reports.tar.gz"
}

verify_fetched_evidence() {
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
    cliff | capacity | balanced | quality) ;;
    *)
        echo "Unknown frontier candidate: $CANDIDATE" >&2
        exit 2
        ;;
esac
require_empty_account
verify_live_contract
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Frontier Verda dry-run complete; no resource created"
    exit 0
fi
[[ ! -e "$STATE_FILE" ]] || {
    echo "Frontier Verda state file already exists: $STATE_FILE" >&2
    exit 2
}

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
        elif "$DELETE_SCRIPT" "$STATE_FILE" --finalize-state; then
            if ! cancel_watchdog; then
                command_status=1
            fi
        else
            echo "Immediate cleanup failed; starting retrying watchdog service" >&2
            systemctl --user start "$WATCHDOG_UNIT.service" || true
            command_status=1
        fi
    fi
    exit "$command_status"
}
trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

write_state pending "" '[]' '{}'
arm_watchdog
create_started=1
create_response="$(verda_json vm create \
    --kind gpu \
    --instance-type "$INSTANCE_TYPE" \
    --location "$LOCATION" \
    --contract pay_as_go \
    --os "$OS_IMAGE" \
    --os-volume-size "$OS_VOLUME_GIB" \
    --hostname "$HOSTNAME" \
    --ssh-key "$SSH_KEY_ID" \
    --wait \
    --wait-timeout 10m)"
vm_id="$(extract_vm_id "$create_response")"
volume_ids_json="$(extract_volume_ids "$create_response")"
write_state active "$vm_id" "$volume_ids_json" "$create_response"

wait_for_ssh "$vm_id"
copy_runner "$vm_id"
start_remote_campaign "$vm_id"
wait_remote_campaign "$vm_id"
fetch_evidence "$vm_id"
verify_fetched_evidence
"$DELETE_SCRIPT" "$STATE_FILE" --finalize-state
cancel_watchdog
cleanup_needed=0

verda_json vm list | python3 -c 'import json,sys; assert json.load(sys.stdin) == []'
verda_json volume list | python3 -c 'import json,sys; assert json.load(sys.stdin) == []'
verda_json cost running | python3 -c '
import json,sys
assert float(json.load(sys.stdin)["total"]["hourly"]) == 0.0
'
echo "Frontier Verda campaign complete; resources deleted and evidence downloaded"
