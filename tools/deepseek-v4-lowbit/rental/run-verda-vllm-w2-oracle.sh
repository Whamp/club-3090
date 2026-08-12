#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"

readonly CLUB_3090_REVISION="934d405a7bb6c3dacd8bc6c2b9ffaff0ad87757a"
readonly CLUB_3090_REF="refs/heads/feat/deepseek-v4-lowbit-vllm"
readonly HAOSDENT_REVISION="12810046c799cbe874967e19b1c0fa134ab7b209"
readonly HAOSDENT_REF="refs/heads/dsv4-flash-a100"
readonly EXPECTED_PATCHED_TREE="97a21943d9a68bcf1ef4ac3319d0a6e3e1c66267"
readonly TORCH_VERSION="2.13.0"
readonly HUMMING_VERSION="0.1.10"
readonly RENTAL_ROOT="${1:-$HOME/deepseek-v4-lowbit-rental}"
readonly CLUB_3090_DIRECTORY="$RENTAL_ROOT/club-3090"
readonly VLLM_DIRECTORY="$RENTAL_ROOT/vllm-sm8x"
readonly ORACLE_ENVIRONMENT="$RENTAL_ROOT/vllm-oracle-environment"
readonly ORACLE_REPORT_DIRECTORY="$RENTAL_ROOT/reports/vllm-w2-oracle"
readonly HUMMING_CACHE_DIRECTORY="$ORACLE_REPORT_DIRECTORY/humming-cache"
readonly ORACLE_LOG="$ORACLE_REPORT_DIRECTORY/run-verda-vllm-w2-oracle.log"
readonly PATCH_INSTALLER="$CLUB_3090_DIRECTORY/models/deepseek-v4-flash-0731/vllm/patches/deepseek-v4-wna16-sm86/install.sh"
readonly ORACLE_TEST="tests/kernels/moe/test_moe.py::test_humming_w2_group128_indexed_numerical_oracle"

mkdir -p "$ORACLE_REPORT_DIRECTORY"
exec > >(tee -a "$ORACLE_LOG") 2>&1

log_oracle_step() {
    printf '\n[%s] %s\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

checkout_pinned_ref() {
    local repository_url="$1"
    local repository_ref="$2"
    local revision="$3"
    local destination="$4"
    if [[ ! -d "$destination/.git" ]]; then
        git clone --filter=blob:none --no-checkout "$repository_url" "$destination"
    fi
    [[ -z "$(git -C "$destination" status --porcelain)" ]] || {
        echo "Pinned oracle checkout must be clean: $destination" >&2
        return 2
    }
    git -C "$destination" fetch --depth 1 origin "$repository_ref"
    git -C "$destination" checkout --detach "$revision"
    test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
}

log_oracle_step "Verify A100 scope, CUDA tools, and storage"
command -v cuobjdump >/dev/null || {
    echo "A100 oracle requires cuobjdump" >&2
    exit 2
}
command -v nvidia-smi >/dev/null || {
    echo "A100 oracle requires nvidia-smi" >&2
    exit 2
}
nvidia-smi
available_kib="$(df --output=avail -k "$RENTAL_ROOT" | tail -1)"
minimum_kib=$((50 * 1024 * 1024))
((available_kib >= minimum_kib)) || {
    echo "vLLM oracle requires at least 50 GiB free" >&2
    exit 2
}
printf 'free_disk_gib=%s\n' "$((available_kib / 1024 / 1024))"

log_oracle_step "Reconstruct pinned haosdent vLLM patch tree"
checkout_pinned_ref \
    "https://github.com/Whamp/club-3090.git" \
    "$CLUB_3090_REF" \
    "$CLUB_3090_REVISION" \
    "$CLUB_3090_DIRECTORY"
checkout_pinned_ref \
    "https://github.com/haosdent/vllm.git" \
    "$HAOSDENT_REF" \
    "$HAOSDENT_REVISION" \
    "$VLLM_DIRECTORY"
"$PATCH_INSTALLER" "$VLLM_DIRECTORY"
test "$(git -C "$VLLM_DIRECTORY" rev-parse 'HEAD^{tree}')" = "$EXPECTED_PATCHED_TREE"

log_oracle_step "Install isolated precompiled-extension vLLM test environment"
uv venv --allow-existing --python 3.12 "$ORACLE_ENVIRONMENT"
uv pip install --python "$ORACLE_ENVIRONMENT/bin/python" \
    "torch==$TORCH_VERSION" \
    --torch-backend=cu130
VLLM_USE_PRECOMPILED=1 uv pip install \
    --python "$ORACLE_ENVIRONMENT/bin/python" \
    --editable "$VLLM_DIRECTORY" \
    --torch-backend=cu130
uv pip install --python "$ORACLE_ENVIRONMENT/bin/python" \
    "pytest==9.1.1" \
    "pytest-asyncio==1.4.0" \
    "pytest-rerunfailures==16.4" \
    "pytest-shard==0.1.2" \
    "pytest-timeout==2.4.0"
uv pip check --python "$ORACLE_ENVIRONMENT/bin/python"

log_oracle_step "Verify exact GPU and Humming environment"
"$ORACLE_ENVIRONMENT/bin/python" - "$HUMMING_VERSION" <<'PY'
import importlib.metadata
import sys
import torch
from humming.utils.cuda import filter_cuda_paths
expected_humming = sys.argv[1]
if torch.cuda.get_device_capability() != (8, 0):
    raise SystemExit(
        "A100 oracle requires compute capability 8.0, got "
        f"{torch.cuda.get_device_capability()}"
    )
if not torch.__version__.startswith("2.13.0+cu130"):
    raise SystemExit(f"Torch version mismatch: {torch.__version__}")
actual_humming = importlib.metadata.version("humming-kernels")
if actual_humming != expected_humming:
    raise SystemExit(
        f"Humming version mismatch: got {actual_humming}, expected {expected_humming}"
    )
cuda_environment = filter_cuda_paths(
    target_version=(13, 0),
    required_headers=["cuda_runtime.h", "nvrtc.h"],
    source="system",
)
if cuda_environment["minor"] != 0:
    raise SystemExit(f"System CUDA minor mismatch: {cuda_environment}")
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)} cc=8.0 humming={actual_humming} "
    f"compiler_cuda={cuda_environment['path']}"
)
PY

log_oracle_step "Run W2 group-128 BF16 indexed-MoE numerical oracle"
mkdir -p "$HUMMING_CACHE_DIRECTORY"
HUMMING_CACHE_DIR="$HUMMING_CACHE_DIRECTORY" \
HUMMING_COMPILER=nvrtc \
PYTHONPATH="$VLLM_DIRECTORY" \
"$ORACLE_ENVIRONMENT/bin/python" -m pytest \
    "$VLLM_DIRECTORY/$ORACLE_TEST" \
    -vv

log_oracle_step "Inspect generated SM80 Humming cubins"
mapfile -d '' cubins < <(find "$HUMMING_CACHE_DIRECTORY" -name kernel.cubin -type f -print0)
((${#cubins[@]} > 0)) || {
    echo "W2 oracle passed without a generated Humming cubin" >&2
    exit 1
}
: > "$ORACLE_REPORT_DIRECTORY/cubin-sha256.txt"
for cubin in "${cubins[@]}"; do
    relative_path="${cubin#"$ORACLE_REPORT_DIRECTORY"/}"
    report_name="$(printf '%s' "$relative_path" | tr '/ ' '__').cuobjdump.txt"
    sha256sum "$cubin" >> "$ORACLE_REPORT_DIRECTORY/cubin-sha256.txt"
    cuobjdump --dump-elf "$cubin" > "$ORACLE_REPORT_DIRECTORY/$report_name"
    grep -q 'sm_80' "$ORACLE_REPORT_DIRECTORY/$report_name" || {
        echo "Generated Humming cubin does not report sm_80: $cubin" >&2
        exit 1
    }
done

log_oracle_step "A100 W2 Humming oracle complete"
sha256sum "$ORACLE_REPORT_DIRECTORY/cubin-sha256.txt"
printf 'oracle_log=%s\ncubin_checksums=%s\ncubin_count=%s\n' \
    "$ORACLE_LOG" \
    "$ORACLE_REPORT_DIRECTORY/cubin-sha256.txt" \
    "${#cubins[@]}"
