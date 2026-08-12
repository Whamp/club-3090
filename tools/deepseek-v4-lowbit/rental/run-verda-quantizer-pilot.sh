#!/usr/bin/env bash
set -euo pipefail

readonly CLUB_3090_REVISION="e8599bad2ac721bdf0650f4a36aa71bc4137f15d"
readonly AUTO_ROUND_REVISION="f17d9cd4b36982006bad21ff87127aac739072e3"
readonly DEEPSEEK_REVISION="7872f01b1d1fe23eabc4c98b48bffcef5a386062"
readonly IMATRIX_REVISION="e7f04037032990db0346398d249baf9fb9df1ccc"
readonly IMATRIX_SHA256="02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e"
readonly IMATRIX_REPOSITORY="antirez/deepseek-v4-gguf"
readonly IMATRIX_FILENAME="imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat"
readonly SOURCE_REPOSITORY="deepseek-ai/DeepSeek-V4-Flash-0731"
readonly UV_VERSION="0.9.28"
readonly TORCH_VERSION="2.13.0"
readonly COMPRESSED_TENSORS_VERSION="0.17.0"
readonly RENTAL_ROOT="${1:-$HOME/deepseek-v4-lowbit-rental}"
readonly SOURCE_DIRECTORY="$RENTAL_ROOT/source/DeepSeek-V4-Flash-0731"
readonly IMATRIX_DIRECTORY="$RENTAL_ROOT/imatrix"
readonly IMATRIX_PATH="$IMATRIX_DIRECTORY/$IMATRIX_FILENAME"
readonly REPORT_DIRECTORY="$RENTAL_ROOT/reports"
readonly PILOT_REPORT="$REPORT_DIRECTORY/w2-quantizer-comparison.json"
readonly PILOT_SUMMARY="$REPORT_DIRECTORY/w2-quantizer-summary.json"
readonly RENTAL_LOG="$REPORT_DIRECTORY/run-verda-quantizer-pilot.log"
readonly CLUB_3090_DIRECTORY="$RENTAL_ROOT/club-3090"
readonly AUTO_ROUND_DIRECTORY="$RENTAL_ROOT/auto-round"
readonly PYTHON_ENVIRONMENT="$RENTAL_ROOT/python-environment"

mkdir -p "$REPORT_DIRECTORY"
exec > >(tee -a "$RENTAL_LOG") 2>&1

log_pilot_step() {
    printf '\n[%s] %s\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

checkout_pinned_repository() {
    local repository_url="$1"
    local repository_ref="$2"
    local revision="$3"
    local destination="$4"
    if [[ ! -d "$destination/.git" ]]; then
        git clone --filter=blob:none --no-checkout "$repository_url" "$destination"
    fi
    git -C "$destination" fetch --depth 1 origin "$repository_ref"
    git -C "$destination" checkout --detach "$revision"
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision"* ]] || {
        echo "Pinned repository checkout mismatch: $destination" >&2
        return 1
    }
}

log_pilot_step "Verify rental hardware and storage"
nvidia-smi
python3 - <<'PY'
import shutil
free_gib = shutil.disk_usage(".").free / (1024**3)
if free_gib < 50:
    raise SystemExit(f"Rental pilot requires at least 50 GiB free, found {free_gib:.1f}")
print(f"free_disk_gib={free_gib:.1f}")
PY

log_pilot_step "Install pinned uv"
if ! command -v uv >/dev/null || [[ "$(uv --version)" != "uv $UV_VERSION" ]]; then
    curl --fail --location --silent --show-error \
        "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

log_pilot_step "Checkout pinned conversion sources"
checkout_pinned_repository \
    "https://github.com/Whamp/club-3090.git" \
    "refs/heads/feat/deepseek-v4-lowbit-vllm" \
    "$CLUB_3090_REVISION" \
    "$CLUB_3090_DIRECTORY"
checkout_pinned_repository \
    "https://github.com/intel/auto-round.git" \
    "refs/heads/main" \
    "$AUTO_ROUND_REVISION" \
    "$AUTO_ROUND_DIRECTORY"

log_pilot_step "Create pinned CUDA quantization environment"
uv venv --allow-existing --python 3.12 "$PYTHON_ENVIRONMENT"
uv pip install --python "$PYTHON_ENVIRONMENT/bin/python" \
    "torch==$TORCH_VERSION" \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$PYTHON_ENVIRONMENT/bin/python" \
    --no-build-isolation \
    --editable "$AUTO_ROUND_DIRECTORY"
uv pip install --python "$PYTHON_ENVIRONMENT/bin/python" \
    "compressed-tensors==$COMPRESSED_TENSORS_VERSION" \
    "huggingface_hub[cli]" \
    --editable "$CLUB_3090_DIRECTORY/tools/deepseek-v4-lowbit"
uv pip check --python "$PYTHON_ENVIRONMENT/bin/python"
"$PYTHON_ENVIRONMENT/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Rental pilot CUDA check failed: torch.cuda.is_available() is false")
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

log_pilot_step "Download four pinned representative source shards"
mkdir -p "$SOURCE_DIRECTORY"
"$PYTHON_ENVIRONMENT/bin/hf" download \
    "$SOURCE_REPOSITORY" \
    model.safetensors.index.json \
    model-00002-of-00048.safetensors \
    model-00028-of-00048.safetensors \
    model-00039-of-00048.safetensors \
    model-00044-of-00048.safetensors \
    --revision "$DEEPSEEK_REVISION" \
    --local-dir "$SOURCE_DIRECTORY" \
    --max-workers 4

log_pilot_step "Download and verify pinned routed-expert imatrix"
mkdir -p "$IMATRIX_DIRECTORY"
"$PYTHON_ENVIRONMENT/bin/hf" download \
    "$IMATRIX_REPOSITORY" \
    "$IMATRIX_FILENAME" \
    --revision "$IMATRIX_REVISION" \
    --local-dir "$IMATRIX_DIRECTORY" \
    --max-workers 1
printf '%s  %s\n' "$IMATRIX_SHA256" "$IMATRIX_PATH" | sha256sum --check --strict

log_pilot_step "Compare 24 matrices and 48 W2 quantizer candidates"
"$PYTHON_ENVIRONMENT/bin/deepseek-v4-pilot" \
    "$SOURCE_DIRECTORY" \
    "$IMATRIX_PATH" \
    "$PILOT_REPORT" \
    --sample 0:0 --sample 0:127 \
    --sample 26:0 --sample 26:127 \
    --sample 37:0 --sample 37:127 \
    --sample 42:0 --sample 42:127 \
    --bits 2 \
    --device cuda

log_pilot_step "Summarize paired error and projected quantize-and-pack time"
"$PYTHON_ENVIRONMENT/bin/deepseek-v4-summarize-pilot" \
    "$PILOT_REPORT" \
    "$PILOT_SUMMARY"

log_pilot_step "Pilot complete"
sha256sum "$PILOT_REPORT" "$PILOT_SUMMARY"
printf 'pilot_report=%s\npilot_summary=%s\nrental_log=%s\n' \
    "$PILOT_REPORT" \
    "$PILOT_SUMMARY" \
    "$RENTAL_LOG"
