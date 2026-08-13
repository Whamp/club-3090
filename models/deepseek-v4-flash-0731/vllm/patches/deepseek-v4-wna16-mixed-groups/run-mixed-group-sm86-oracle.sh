#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZATION_PHRASE="I_AUTHORIZE_SERVER60_MIXED_GROUP_ORACLE"
readonly EXPECTED_IMAGE_TREE="f73b30cc5a2ed9de200ca2e4de3cdef1a06f6538"
readonly EXPECTED_IMAGE_SCOPE="mixed-group-oracle-only"
readonly ORACLE_TEST="tests/kernels/moe/test_moe.py::test_humming_wna16_grouped_indexed_numerical_oracle"
readonly IMAGE="${VLLM_IMAGE:-club-3090/deepseek-v4-wna16-sm86:mixed-group-oracle-f73b30cc}"
readonly REPORT_DIRECTORY="${REPORT_DIRECTORY:-$PWD/mixed-group-sm86-oracle-report}"
readonly AUTHORIZATION="${1:-}"

[[ "$AUTHORIZATION" == "$AUTHORIZATION_PHRASE" ]] || {
    echo "Mixed-group SM86 oracle requires literal authorization: $AUTHORIZATION_PHRASE" >&2
    exit 2
}
actual_tree="$(
    docker image inspect "$IMAGE" \
        --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
)"
actual_scope="$(
    docker image inspect "$IMAGE" \
        --format '{{index .Config.Labels "org.club3090.runtime.scope"}}'
)"
[[ "$actual_tree" == "$EXPECTED_IMAGE_TREE" && "$actual_scope" == "$EXPECTED_IMAGE_SCOPE" ]] || {
    echo "Mixed-group SM86 oracle image contract mismatch" >&2
    exit 2
}

mapfile -t gpu_processes < <(
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
)
((${#gpu_processes[@]} == 0)) || {
    printf 'Mixed-group SM86 oracle refuses active GPU processes:\n%s\n' \
        "${gpu_processes[*]}" >&2
    exit 2
}
mkdir -p "$REPORT_DIRECTORY"
[[ -z "$(find "$REPORT_DIRECTORY" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "Mixed-group SM86 oracle report directory must start empty" >&2
    exit 2
}

nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
    --format=csv,noheader > "$REPORT_DIRECTORY/gpu-before.csv"
set +e
docker run --rm \
    --gpus '"device=0"' \
    --ipc=host \
    --shm-size=4g \
    --entrypoint bash \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env HUMMING_CACHE_DIR=/report/humming-cache \
    --env HUMMING_COMPILER=nvrtc \
    --env TORCH_EXTENSIONS_DIR=/report/torch-extensions \
    --volume "$REPORT_DIRECTORY:/report" \
    "$IMAGE" \
    -lc '
        set -euo pipefail
        uv pip install --python /opt/venv/bin/python \
            "pytest==9.1.1" \
            "pytest-asyncio==1.4.0" \
            "pytest-rerunfailures==16.4" \
            "pytest-shard==0.1.2" \
            "pytest-timeout==2.4.0" \
            "tblib==3.1.0"
        /opt/venv/bin/python - <<"PY"
import torch
import vllm._C_stable_libtorch
import vllm._moe_C_stable_libtorch
capability = torch.cuda.get_device_capability()
if capability != (8, 6):
    raise SystemExit(f"Mixed-group oracle requires SM86, got {capability}")
print(f"gpu={torch.cuda.get_device_name(0)} capability={capability}")
PY
        /opt/venv/bin/python -m pytest \
            "/workspace/vllm/'"$ORACLE_TEST"'" -vv
    ' 2>&1 | tee "$REPORT_DIRECTORY/oracle.log"
status=${PIPESTATUS[0]}
set -e
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
    --format=csv,noheader > "$REPORT_DIRECTORY/gpu-after.csv"
((status == 0)) || exit "$status"
readonly -a ORACLE_CASES=(
    w2-g128-g128
    w2-g256-g128
    w2-g512-g256
    w2-g512-g512
    w4-g512-g128
    w4-g512-g256
    w4-g512-g512
)
for oracle_case in "${ORACLE_CASES[@]}"; do
    grep -q "\[$oracle_case\].*PASSED" "$REPORT_DIRECTORY/oracle.log" || {
        echo "Mixed-group SM86 oracle did not pass case $oracle_case" >&2
        exit 1
    }
done

mapfile -d '' cubins < <(
    find "$REPORT_DIRECTORY/humming-cache" -name kernel.cubin -type f -print0
)
((${#cubins[@]} > 0)) || {
    echo "Mixed-group SM86 oracle generated no Humming cubin" >&2
    exit 1
}
: > "$REPORT_DIRECTORY/cubin-sha256.txt"
for cubin in "${cubins[@]}"; do
    relative_path="${cubin#"$REPORT_DIRECTORY/"}"
    report_name="$(printf '%s' "$relative_path" | tr '/ ' '__').cuobjdump.txt"
    sha256sum "$cubin" >> "$REPORT_DIRECTORY/cubin-sha256.txt"
    docker run --rm --entrypoint cuobjdump \
        --volume "$REPORT_DIRECTORY:/report:ro" \
        "$IMAGE" \
        --dump-elf "/report/$relative_path" \
        > "$REPORT_DIRECTORY/$report_name"
    grep -q 'sm_86' "$REPORT_DIRECTORY/$report_name" || {
        echo "Mixed-group Humming cubin is not sm_86: $relative_path" >&2
        exit 1
    }
done
printf 'oracle=passed cases=%s cubin_count=%s report=%s\n' \
    "${ORACLE_CASES[*]}" \
    "${#cubins[@]}" "$REPORT_DIRECTORY"
