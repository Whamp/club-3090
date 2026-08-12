#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZATION_PHRASE="I_AUTHORIZE_LLAMA_STOP_FOR_SM86_ORACLE"
readonly AUTHORIZATION="${1:-}"
readonly LLAMA_PROJECT="antirez-iq2-xxs"
readonly LLAMA_SERVICE="llama-cpp-deepseek-v4-fast-prefill"
readonly LLAMA_COMPOSE_FILE="/home/will/inference/serving/club-3090/.worktrees/deepseek-v4-q8-fast-prefill/models/deepseek-v4-flash-0731/llama-cpp/compose/multi4/antirez-iq2-xxs/fast-prefill.yml"
readonly LLAMA_IMAGE_ID="sha256:a96bd947d63eb81d8baf9f6f5ecb26669476383976717237450fbb5727b03745"
readonly ORACLE_RUNNER="${ORACLE_RUNNER:-$PWD/run-sm86-oracle.sh}"
readonly ORACLE_REPORT_DIRECTORY="${ORACLE_REPORT_DIRECTORY:-$PWD/sm86-oracle-report}"
readonly ORACLE_TIMEOUT="${ORACLE_TIMEOUT:-20m}"

[[ "$AUTHORIZATION" == "$AUTHORIZATION_PHRASE" ]] || {
    echo "Server60 rollback wrapper requires literal authorization: $AUTHORIZATION_PHRASE" >&2
    exit 2
}
[[ -x "$ORACLE_RUNNER" ]] || {
    echo "Server60 rollback wrapper cannot execute oracle runner: $ORACLE_RUNNER" >&2
    exit 2
}
[[ -f "$LLAMA_COMPOSE_FILE" ]] || {
    echo "Server60 rollback wrapper cannot find compose file: $LLAMA_COMPOSE_FILE" >&2
    exit 2
}

actual_image_id="$(docker inspect "$LLAMA_SERVICE" --format '{{.Image}}')"
actual_health="$(docker inspect "$LLAMA_SERVICE" --format '{{.State.Health.Status}}')"
[[ "$actual_image_id" == "$LLAMA_IMAGE_ID" && "$actual_health" == "healthy" ]] || {
    echo "Server60 rollback baseline mismatch: image=$actual_image_id health=$actual_health" >&2
    exit 2
}

restore_llama_service() {
    local restore_status=0
    docker compose \
        --project-name "$LLAMA_PROJECT" \
        --file "$LLAMA_COMPOSE_FILE" \
        up --detach "$LLAMA_SERVICE" || restore_status=$?
    if ((restore_status == 0)); then
        for _ in $(seq 1 120); do
            if [[ "$(docker inspect "$LLAMA_SERVICE" --format '{{.State.Health.Status}}' 2>/dev/null || true)" == "healthy" ]]; then
                break
            fi
            sleep 5
        done
        [[ "$(docker inspect "$LLAMA_SERVICE" --format '{{.State.Health.Status}}' 2>/dev/null || true)" == "healthy" ]] || restore_status=1
        [[ "$(docker inspect "$LLAMA_SERVICE" --format '{{.Image}}' 2>/dev/null || true)" == "$LLAMA_IMAGE_ID" ]] || restore_status=1
    fi
    if ((restore_status != 0)); then
        echo "CRITICAL: failed to restore healthy llama.cpp service" >&2
    fi
    return "$restore_status"
}
trap restore_llama_service EXIT INT TERM HUP

docker compose \
    --project-name "$LLAMA_PROJECT" \
    --file "$LLAMA_COMPOSE_FILE" \
    stop --timeout 120 "$LLAMA_SERVICE"

for _ in $(seq 1 120); do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
        break
    fi
    sleep 1
done
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]] || {
    echo "Server60 GPUs remained busy after stopping llama.cpp" >&2
    exit 1
}

REPORT_DIRECTORY="$ORACLE_REPORT_DIRECTORY" \
    timeout --signal=TERM --kill-after=2m "$ORACLE_TIMEOUT" \
    "$ORACLE_RUNNER" I_AUTHORIZE_SERVER60_GPU_ORACLE
