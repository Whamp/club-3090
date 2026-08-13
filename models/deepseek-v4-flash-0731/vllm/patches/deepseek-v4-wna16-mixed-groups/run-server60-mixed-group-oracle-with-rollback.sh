#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZATION_PHRASE="I_AUTHORIZE_SERVER60_MIXED_GROUP_ORACLE_STOP"
readonly AUTHORIZATION="${1:-}"
readonly SERVICE_CONTAINER="vllm-deepseek-v4-wna16-sm86"
readonly COMPOSE_PROJECT="dsv4-wna16-prod"
readonly COMPOSE_SERVICE="deepseek-v4-wna16-sm86"
readonly COMPOSE_FILE="${PROMOTED_COMPOSE_FILE:-/home/will/inference/runtime/deepseek-v4-wna16-sm86/canonical-215k-26ae767a/models/deepseek-v4-flash-0731/vllm/compose/multi4/wna16/base.yml}"
readonly COMPOSE_ENV_FILE="${PROMOTED_COMPOSE_ENV_FILE:-/home/will/inference/runtime/deepseek-v4-wna16-sm86/canonical-promotion-20260812/compose.env}"
readonly EXPECTED_PRODUCTION_IMAGE_ID="sha256:0beb1f0cba2e41837f4ba5af01cc5c4686afde4f40ab1df5147a6ad945b0af1f"
readonly EXPECTED_MODEL="deepseek-v4-flash-0731-wna16"
readonly HEALTH_URL="${PROMOTED_HEALTH_URL:-http://100.92.238.117:8034}"
readonly ORACLE_RUNNER="${ORACLE_RUNNER:-$PWD/run-mixed-group-sm86-oracle.sh}"
readonly ORACLE_IMAGE="${MIXED_GROUP_ORACLE_IMAGE:-club-3090/deepseek-v4-wna16-sm86:mixed-group-oracle-f73b30cc}"
readonly ORACLE_REPORT_DIRECTORY="${ORACLE_REPORT_DIRECTORY:-$PWD/mixed-group-sm86-oracle-report}"
readonly ORACLE_TIMEOUT="${ORACLE_TIMEOUT:-20m}"
service_stopped=0

[[ "$AUTHORIZATION" == "$AUTHORIZATION_PHRASE" ]] || {
    echo "Server60 mixed-group rollback requires: $AUTHORIZATION_PHRASE" >&2
    exit 2
}
[[ -x "$ORACLE_RUNNER" ]] || {
    echo "Server60 mixed-group oracle runner is not executable: $ORACLE_RUNNER" >&2
    exit 2
}
[[ -f "$COMPOSE_FILE" && -f "$COMPOSE_ENV_FILE" ]] || {
    echo "Server60 promoted Compose contract is missing" >&2
    exit 2
}

require_promoted_service_health() {
    local actual_health actual_image actual_model
    actual_health="$(
        docker inspect "$SERVICE_CONTAINER" \
            --format '{{.State.Health.Status}}' 2>/dev/null || true
    )"
    actual_image="$(
        docker inspect "$SERVICE_CONTAINER" --format '{{.Image}}' 2>/dev/null || true
    )"
    actual_model="$(
        curl --fail --silent --show-error "$HEALTH_URL/v1/models" | \
            python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
    )" || return 1
    [[ "$actual_health" == "healthy" ]] || return 1
    [[ "$actual_image" == "$EXPECTED_PRODUCTION_IMAGE_ID" ]] || return 1
    [[ "$actual_model" == "$EXPECTED_MODEL" ]] || return 1
}

restore_promoted_service() {
    local restore_status=0
    ((service_stopped == 1)) || return 0
    env -u VLLM_IMAGE docker compose \
        --project-name "$COMPOSE_PROJECT" \
        --env-file "$COMPOSE_ENV_FILE" \
        --profile authorized-gpu-test \
        --file "$COMPOSE_FILE" \
        up --detach --no-deps "$COMPOSE_SERVICE" || restore_status=$?
    if ((restore_status == 0)); then
        for _ in $(seq 1 240); do
            if require_promoted_service_health; then
                service_stopped=0
                return 0
            fi
            sleep 5
        done
        restore_status=1
    fi
    echo "CRITICAL: failed to restore exact healthy promoted WNA16 service" >&2
    return "$restore_status"
}

finish_server60_mixed_group_oracle() {
    local command_status=$?
    trap - EXIT INT TERM HUP
    if ! restore_promoted_service; then
        exit 1
    fi
    exit "$command_status"
}
trap finish_server60_mixed_group_oracle EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

require_promoted_service_health || {
    echo "Server60 promoted WNA16 service is not the expected healthy baseline" >&2
    exit 2
}
resolved_production_image="$(
    env -u VLLM_IMAGE docker compose \
        --project-name "$COMPOSE_PROJECT" \
        --env-file "$COMPOSE_ENV_FILE" \
        --profile authorized-gpu-test \
        --file "$COMPOSE_FILE" \
        config --images
)"
[[ "$resolved_production_image" == *"@$EXPECTED_PRODUCTION_IMAGE_ID" ]] || {
    echo "Production Compose image was overridden: $resolved_production_image" >&2
    exit 2
}

env -u VLLM_IMAGE docker compose \
    --project-name "$COMPOSE_PROJECT" \
    --env-file "$COMPOSE_ENV_FILE" \
    --profile authorized-gpu-test \
    --file "$COMPOSE_FILE" \
    stop --timeout 120 "$COMPOSE_SERVICE"
service_stopped=1
for _ in $(seq 1 120); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]] && \
        break
    sleep 1
done
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]] || {
    echo "Server60 GPUs remained busy after stopping the promoted service" >&2
    exit 1
}

REPORT_DIRECTORY="$ORACLE_REPORT_DIRECTORY" \
    VLLM_IMAGE="$ORACLE_IMAGE" \
    timeout --signal=TERM --kill-after=2m "$ORACLE_TIMEOUT" \
    "$ORACLE_RUNNER" I_AUTHORIZE_SERVER60_MIXED_GROUP_ORACLE
