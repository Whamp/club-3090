#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIRECTORY/../../../../.." && pwd)"
readonly REPOSITORY_ROOT
readonly COMPOSE_FILE="$REPOSITORY_ROOT/models/deepseek-v4-flash-0731/vllm/compose/multi4/wna16/base.yml"
readonly COMPOSE_OVERRIDE="$SCRIPT_DIRECTORY/compose.override.yml"
readonly COMPOSE_NSYS_OVERRIDE="$SCRIPT_DIRECTORY/compose.nsight.override.yml"
readonly TOOL_PROJECT="$REPOSITORY_ROOT/tools/deepseek-v4-lowbit"
readonly EXPECTED_SPEED_COMMIT="b7766cfe4d15d9b68acea43097ceff221e8a739f"
readonly EXPECTED_SPEED_TREE="6354125afd1306c9286f734d1c47c23c767d77a9"
readonly EXPECTED_MODEL_ID="deepseek-v4-flash-0731-wna16-quality-12035985"
readonly PRODUCTION_CONTAINER="${PRODUCTION_CONTAINER:-vllm-deepseek-v4-wna16-sm86}"

usage() {
    cat >&2 <<'EOF'
usage: run-speed-arm-with-rollback.sh [--dry-run] ARM RESULT_DIRECTORY [-- COMMAND ...]

ARM is one of baseline, prefill-block2, flashmla-decode, hier-allreduce,
indexer96, or batched320. Dry-run never calls Docker or contacts server60.
Actual execution requires these identity-bound variables:

  APPROVED_PRODUCTION_IMAGE_ID=sha256:...
  SPEED_IMAGE=repository:tag-or-id
  SPEED_IMAGE_ID=sha256:...
  MODEL_SNAPSHOT=/immutable/snapshot
  MODEL_BLOBS=/huggingface/blobs
  RUNTIME_CACHE_ROOT=/runtime/cache
  BIND_HOST=server-address
  HEALTH_URL=http://server-address:8034
  SERVER60_SPEED_PLAN_SHA256=<hash printed by --dry-run with the same inputs>
EOF
    exit 2
}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
[[ $# -ge 2 ]] || usage
readonly ARM="$1"
readonly RESULT_DIRECTORY="$2"
shift 2
if [[ "${1:-}" == "--" ]]; then
    shift
fi
COMMAND=("$@")

case "$ARM" in
    baseline | trace-baseline | prefill-block2 | flashmla-decode | hier-allreduce | indexer96 | batched320) ;;
    *) echo "Unknown DeepSeek V4 speed arm: $ARM" >&2; exit 2 ;;
esac

readonly APPROVED_PRODUCTION_IMAGE_ID="${APPROVED_PRODUCTION_IMAGE_ID:-UNSET}"
readonly SPEED_IMAGE="${SPEED_IMAGE:-UNSET}"
readonly SPEED_IMAGE_ID="${SPEED_IMAGE_ID:-UNSET}"
readonly MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-UNSET}"
readonly MODEL_BLOBS="${MODEL_BLOBS:-UNSET}"
readonly RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-UNSET}"
readonly BIND_HOST="${BIND_HOST:-UNSET}"
readonly PORT="${PORT:-8034}"
readonly HEALTH_URL="${HEALTH_URL:-UNSET}"
readonly NSYS_OUTPUT_DIRECTORY="${NSYS_OUTPUT_DIRECTORY:-UNSET}"
readonly NSYS_REPORT_BASENAME="${NSYS_REPORT_BASENAME:-deepseek-v4-decode}"
readonly HIER_TRACE_GATE_JSON="${HIER_TRACE_GATE_JSON:-UNSET}"
HARNESS_COMMIT="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD)"
readonly HARNESS_COMMIT
HARNESS_TREE="$(git -C "$REPOSITORY_ROOT" rev-parse 'HEAD^{tree}')"
readonly HARNESS_TREE
HARNESS_SHA256="$(
    cd "$SCRIPT_DIRECTORY"
    find . -maxdepth 1 -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | cut -d' ' -f1
)"
readonly HARNESS_SHA256
readonly EXPERIMENT_PROJECT="dsv4-wna16-speed-${ARM}"
readonly EXPERIMENT_CONTAINER="vllm-deepseek-v4-speed-${ARM}"
MEASUREMENT_COMMAND="$(printf '%q ' "${COMMAND[@]}")"
readonly MEASUREMENT_COMMAND

arm_manifest="$(
    cd "$TOOL_PROJECT"
    uv run --python 3.12 deepseek-v4-speed-experiment "$ARM"
)"
trace_gate_summary='{}'
if [[ "$ARM" == hier-allreduce ]]; then
    [[ "$HIER_TRACE_GATE_JSON" != UNSET ]] || {
        echo "hier-allreduce requires HIER_TRACE_GATE_JSON" >&2
        exit 2
    }
    trace_gate_summary="$(
        cd "$TOOL_PROJECT"
        uv run --python 3.12 deepseek-v4-validate-speed-trace \
            "$HIER_TRACE_GATE_JSON"
    )"
fi
plan_json="$(
    ARM_MANIFEST="$arm_manifest" \
    PLAN_PRODUCTION_IMAGE_ID="$APPROVED_PRODUCTION_IMAGE_ID" \
    PLAN_SPEED_IMAGE="$SPEED_IMAGE" \
    PLAN_SPEED_IMAGE_ID="$SPEED_IMAGE_ID" \
    PLAN_MODEL_SNAPSHOT="$MODEL_SNAPSHOT" \
    PLAN_MODEL_BLOBS="$MODEL_BLOBS" \
    PLAN_RUNTIME_CACHE_ROOT="$RUNTIME_CACHE_ROOT" \
    PLAN_BIND_HOST="$BIND_HOST" PLAN_PORT="$PORT" \
    PLAN_HEALTH_URL="$HEALTH_URL" \
    PLAN_NSYS_OUTPUT_DIRECTORY="$NSYS_OUTPUT_DIRECTORY" \
    PLAN_NSYS_REPORT_BASENAME="$NSYS_REPORT_BASENAME" \
    PLAN_EXPECTED_SPEED_COMMIT="$EXPECTED_SPEED_COMMIT" \
    PLAN_EXPECTED_SPEED_TREE="$EXPECTED_SPEED_TREE" \
    PLAN_HARNESS_COMMIT="$HARNESS_COMMIT" \
    PLAN_HARNESS_TREE="$HARNESS_TREE" \
    PLAN_HARNESS_SHA256="$HARNESS_SHA256" \
    PLAN_MEASUREMENT_COMMAND="$MEASUREMENT_COMMAND" \
    PLAN_TRACE_GATE_SUMMARY="$trace_gate_summary" \
    python3 - <<'PY'
import json
import os

manifest = json.loads(os.environ["ARM_MANIFEST"])
manifest["approved_production_image_id"] = os.environ["PLAN_PRODUCTION_IMAGE_ID"]
manifest["speed_image"] = os.environ["PLAN_SPEED_IMAGE"]
manifest["speed_image_id"] = os.environ["PLAN_SPEED_IMAGE_ID"]
manifest["model_snapshot"] = os.environ["PLAN_MODEL_SNAPSHOT"]
manifest["model_blobs"] = os.environ["PLAN_MODEL_BLOBS"]
manifest["runtime_cache_root"] = os.environ["PLAN_RUNTIME_CACHE_ROOT"]
manifest["bind_host"] = os.environ["PLAN_BIND_HOST"]
manifest["port"] = os.environ["PLAN_PORT"]
manifest["health_url"] = os.environ["PLAN_HEALTH_URL"]
manifest["nsys_output_directory"] = os.environ["PLAN_NSYS_OUTPUT_DIRECTORY"]
manifest["nsys_report_basename"] = os.environ["PLAN_NSYS_REPORT_BASENAME"]
manifest["expected_speed_commit"] = os.environ["PLAN_EXPECTED_SPEED_COMMIT"]
manifest["expected_speed_tree"] = os.environ["PLAN_EXPECTED_SPEED_TREE"]
manifest["harness_commit"] = os.environ["PLAN_HARNESS_COMMIT"]
manifest["harness_tree"] = os.environ["PLAN_HARNESS_TREE"]
manifest["harness_sha256"] = os.environ["PLAN_HARNESS_SHA256"]
manifest["measurement_command"] = os.environ["PLAN_MEASUREMENT_COMMAND"]
manifest["trace_gate"] = json.loads(os.environ["PLAN_TRACE_GATE_SUMMARY"])
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
)"
plan_sha256="$(printf '%s\n' "$plan_json" | sha256sum | cut -d' ' -f1)"
mkdir -p "$RESULT_DIRECTORY"
printf '%s\n' "$plan_json" > "$RESULT_DIRECTORY/plan.json"
printf '%s\n' "$plan_sha256" > "$RESULT_DIRECTORY/plan.sha256"

if [[ "$DRY_RUN" == 1 ]]; then
    printf '%s\n' "$plan_json"
    printf 'plan_sha256=%s\n' "$plan_sha256"
    exit 0
fi

[[ ${#COMMAND[@]} -gt 0 ]] || {
    echo "Actual speed-arm execution requires a measurement COMMAND after --" >&2
    exit 2
}
for value_name in APPROVED_PRODUCTION_IMAGE_ID SPEED_IMAGE SPEED_IMAGE_ID \
    MODEL_SNAPSHOT MODEL_BLOBS RUNTIME_CACHE_ROOT BIND_HOST HEALTH_URL; do
    [[ "${!value_name}" != "UNSET" ]] || {
        echo "Missing required speed experiment identity: $value_name" >&2
        exit 2
    }
done
[[ "${SERVER60_SPEED_PLAN_SHA256:-}" == "$plan_sha256" ]] || {
    echo "DeepSeek V4 speed plan is not approved: expected $plan_sha256" >&2
    exit 2
}
[[ -d "$MODEL_SNAPSHOT" && -d "$MODEL_BLOBS" && -d "$RUNTIME_CACHE_ROOT" ]] || {
    echo "DeepSeek V4 speed experiment paths are not mounted directories" >&2
    exit 2
}
command -v fuser >/dev/null || {
    echo "DeepSeek V4 speed cleanup requires fuser" >&2
    exit 2
}
sudo -n true || {
    echo "DeepSeek V4 speed cleanup requires passwordless sudo" >&2
    exit 2
}
if [[ "$ARM" == trace-baseline ]]; then
    [[ "$NSYS_OUTPUT_DIRECTORY" != UNSET ]] || {
        echo "trace-baseline requires NSYS_OUTPUT_DIRECTORY" >&2
        exit 2
    }
    mkdir -p "$NSYS_OUTPUT_DIRECTORY"
fi
COMPOSE_FILES=(--file "$COMPOSE_FILE" --file "$COMPOSE_OVERRIDE")
if [[ "$ARM" == trace-baseline ]]; then
    COMPOSE_FILES+=(--file "$COMPOSE_NSYS_OVERRIDE")
fi

production_image_id="$(docker inspect "$PRODUCTION_CONTAINER" --format '{{.Image}}')"
[[ "$production_image_id" == "$APPROVED_PRODUCTION_IMAGE_ID" ]] || {
    echo "Production image mismatch: $production_image_id" >&2
    exit 2
}
[[ "$(docker inspect "$PRODUCTION_CONTAINER" --format '{{.State.Running}}')" == true ]] || {
    echo "Production container is not running before the experiment" >&2
    exit 2
}
[[ "$(docker inspect "$PRODUCTION_CONTAINER" --format '{{.State.Health.Status}}')" == healthy ]] || {
    echo "Production container is not healthy before the experiment" >&2
    exit 2
}
actual_speed_image_id="$(docker image inspect "$SPEED_IMAGE" --format '{{.Id}}')"
[[ "$actual_speed_image_id" == "$SPEED_IMAGE_ID" ]] || {
    echo "Speed image identity mismatch: $actual_speed_image_id" >&2
    exit 2
}
[[ "$(docker image inspect "$SPEED_IMAGE" --format '{{index .Config.Labels "org.club3090.runtime.canonical-commit"}}')" == "$EXPECTED_SPEED_COMMIT" ]] || {
    echo "Speed image canonical commit mismatch" >&2
    exit 2
}
[[ "$(docker image inspect "$SPEED_IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')" == "$EXPECTED_SPEED_TREE" ]] || {
    echo "Speed image source tree mismatch" >&2
    exit 2
}

cleanup_stale_kv_offload_files() {
    local file
    shopt -s nullglob
    for file in /dev/shm/vllm_offload_*.mmap; do
        if fuser "$file" >/dev/null 2>&1; then
            continue
        fi
        printf 'remove_unreferenced_kv_offload_file=%s\n' "$file" \
            >> "$RESULT_DIRECTORY/cleanup.log"
        sudo -n rm -f -- "$file"
    done
    shopt -u nullglob
}

rollback_started=0
restore_production() {
    local status=$?
    trap - EXIT INT TERM
    if [[ "$rollback_started" == 1 ]]; then
        docker logs "$EXPERIMENT_CONTAINER" > "$RESULT_DIRECTORY/container.log" 2>&1 || true
        env VLLM_IMAGE="$SPEED_IMAGE" \
            CONTAINER_NAME="$EXPERIMENT_CONTAINER" \
            CLUB3090_RESTART=no \
            BIND_HOST="$BIND_HOST" PORT="$PORT" \
            MODEL_SNAPSHOT="$MODEL_SNAPSHOT" MODEL_BLOBS="$MODEL_BLOBS" \
            RUNTIME_CACHE_ROOT="$RUNTIME_CACHE_ROOT" \
            docker compose --profile authorized-gpu-test \
            --project-name "$EXPERIMENT_PROJECT" "${COMPOSE_FILES[@]}" down \
            --remove-orphans >/dev/null 2>&1 || true
        cleanup_stale_kv_offload_files || status=1
        docker start "$PRODUCTION_CONTAINER" >/dev/null
        for _ in $(seq 1 180); do
            if [[ "$(docker inspect "$PRODUCTION_CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null || true)" == healthy ]]; then
                break
            fi
            sleep 5
        done
        [[ "$(docker inspect "$PRODUCTION_CONTAINER" --format '{{.Image}}')" == "$APPROVED_PRODUCTION_IMAGE_ID" ]] || status=1
        [[ "$(docker inspect "$PRODUCTION_CONTAINER" --format '{{.State.Health.Status}}')" == healthy ]] || status=1
    fi
    exit "$status"
}
trap restore_production EXIT INT TERM

# Every arm uses the same speed image. Only these exported profile values differ.
export MAX_MODEL_LEN=230144 GPU_MEMORY_UTILIZATION=0.98 MAX_NUM_SEQS=2
export MAX_NUM_BATCHED_TOKENS=256 KV_OFFLOADING_SIZE=16
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
export VLLM_SPARSE_DENSE_QUERY_BLOCK=0 VLLM_DSV4_FLASH_MLA_DECODE=0
export VLLM_HIER_ALL_REDUCE=""
export NSYS_OUTPUT_DIRECTORY NSYS_REPORT_BASENAME
case "$ARM" in
    baseline) ;;
    trace-baseline) export KV_OFFLOADING_SIZE=0 ;;
    prefill-block2) export VLLM_SPARSE_DENSE_QUERY_BLOCK=2 ;;
    flashmla-decode) export VLLM_DSV4_FLASH_MLA_DECODE=1 ;;
    hier-allreduce) export VLLM_HIER_ALL_REDUCE="0,1;2,3" ;;
    indexer96) export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=96 ;;
    batched320) export MAX_NUM_BATCHED_TOKENS=320 ;;
esac

# Arm rollback before the first production-state mutation. Stop, but never
# remove or recreate, the exact production container.
rollback_started=1
docker stop --time 180 "$PRODUCTION_CONTAINER" >/dev/null
cleanup_stale_kv_offload_files

case "$ARM" in
    flashmla-decode)
        "$SCRIPT_DIRECTORY/run-flash-mla-sm86-gate.sh" \
            "$SPEED_IMAGE" "$RESULT_DIRECTORY/flash-mla-gate"
        ;;
    hier-allreduce)
        "$SCRIPT_DIRECTORY/run-hier-all-reduce-sm86-gate.sh" \
            "$SPEED_IMAGE" "$RESULT_DIRECTORY/hier-all-reduce-gate"
        ;;
esac

env VLLM_IMAGE="$SPEED_IMAGE" \
    CONTAINER_NAME="$EXPERIMENT_CONTAINER" \
    CLUB3090_RESTART=no \
    BIND_HOST="$BIND_HOST" PORT="$PORT" \
    MODEL_SNAPSHOT="$MODEL_SNAPSHOT" MODEL_BLOBS="$MODEL_BLOBS" \
    RUNTIME_CACHE_ROOT="$RUNTIME_CACHE_ROOT" \
    docker compose --profile authorized-gpu-test \
    --project-name "$EXPERIMENT_PROJECT" "${COMPOSE_FILES[@]}" up --detach

for _ in $(seq 1 180); do
    if [[ "$(docker inspect "$EXPERIMENT_CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null || true)" == healthy ]]; then
        break
    fi
    if [[ "$(docker inspect "$EXPERIMENT_CONTAINER" --format '{{.State.Running}}' 2>/dev/null || true)" == false ]]; then
        echo "Speed experiment container exited before readiness" >&2
        exit 1
    fi
    sleep 5
done
[[ "$(docker inspect "$EXPERIMENT_CONTAINER" --format '{{.State.Health.Status}}')" == healthy ]] || {
    echo "Speed experiment did not reach health" >&2
    exit 1
}

served_model="$(curl --fail --silent "$HEALTH_URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
[[ "$served_model" == "$EXPECTED_MODEL_ID" ]] || {
    echo "Speed experiment served model mismatch: $served_model" >&2
    exit 1
}

export URL="$HEALTH_URL" MODEL="$EXPECTED_MODEL_ID" CONTAINER="$EXPERIMENT_CONTAINER"
export SPEED_ARM="$ARM"
"${COMMAND[@]}" > >(tee "$RESULT_DIRECTORY/measurement.stdout.log") \
    2> >(tee "$RESULT_DIRECTORY/measurement.stderr.log" >&2)
