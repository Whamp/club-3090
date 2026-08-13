#!/usr/bin/env bash
set -euo pipefail

readonly AUTHORIZATION_PHRASE="I_AUTHORIZE_SERVER60_GPU_ORACLE"
readonly EXPECTED_IMAGE_TREE="aeb62948e33074514a742d19c2f9a1a3c2ee3e1f"
readonly ORACLE_TEST="tests/kernels/moe/test_moe.py::test_humming_w2_group128_indexed_numerical_oracle"
readonly IMAGE="${VLLM_IMAGE:-club-3090/deepseek-v4-wna16-sm86:aeb62948-rope-cu130}"
readonly REPORT_DIRECTORY="${REPORT_DIRECTORY:-$PWD/sm86-oracle-report}"
readonly AUTHORIZATION="${1:-}"

[[ "$AUTHORIZATION" == "$AUTHORIZATION_PHRASE" ]] || {
    echo "SM86 oracle requires literal authorization: $AUTHORIZATION_PHRASE" >&2
    exit 2
}

actual_tree="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
[[ "$actual_tree" == "$EXPECTED_IMAGE_TREE" ]] || {
    echo "SM86 oracle image tree mismatch: got $actual_tree, expected $EXPECTED_IMAGE_TREE" >&2
    exit 2
}

mapfile -t gpu_processes < <(
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
)
((${#gpu_processes[@]} == 0)) || {
    printf 'SM86 oracle refuses to share GPUs with active processes:\n%s\n' \
        "${gpu_processes[*]}" >&2
    exit 2
}

mkdir -p "$REPORT_DIRECTORY"
[[ -z "$(find "$REPORT_DIRECTORY" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "SM86 oracle report directory must start empty: $REPORT_DIRECTORY" >&2
    exit 2
}

before_gpu_state="$REPORT_DIRECTORY/gpu-before.csv"
after_gpu_state="$REPORT_DIRECTORY/gpu-after.csv"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
    --format=csv,noheader > "$before_gpu_state"

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
        python - <<"PY"
import torch
import vllm._C_stable_libtorch
import vllm._moe_C_stable_libtorch
capability = torch.cuda.get_device_capability()
if capability != (8, 6):
    raise SystemExit(f"SM86 oracle requires compute capability 8.6, got {capability}")
print(f"gpu={torch.cuda.get_device_name(0)} capability={capability}")
PY
        python -m pytest "/workspace/vllm/'"$ORACLE_TEST"'" -vv
    ' 2>&1 | tee "$REPORT_DIRECTORY/oracle.log"
status=${PIPESTATUS[0]}
set -e

nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
    --format=csv,noheader > "$after_gpu_state"
((status == 0)) || exit "$status"

mapfile -d '' cubins < <(
    find "$REPORT_DIRECTORY/humming-cache" -name kernel.cubin -type f -print0
)
((${#cubins[@]} > 0)) || {
    echo "SM86 oracle passed without a generated Humming cubin" >&2
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
        echo "Generated Humming cubin does not report sm_86: $relative_path" >&2
        exit 1
    }
done

printf 'oracle=passed cubin_count=%s report=%s\n' \
    "${#cubins[@]}" \
    "$REPORT_DIRECTORY"
