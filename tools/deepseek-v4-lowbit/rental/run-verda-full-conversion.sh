#!/usr/bin/env bash
set -euo pipefail

readonly CLUB_3090_REVISION="e8599bad2ac721bdf0650f4a36aa71bc4137f15d"
readonly CLUB_3090_REF="refs/heads/feat/deepseek-v4-lowbit-vllm"
readonly DEEPSEEK_REVISION="7872f01b1d1fe23eabc4c98b48bffcef5a386062"
readonly SOURCE_REPOSITORY="deepseek-ai/DeepSeek-V4-Flash-0731"
readonly HUGGINGFACE_REPOSITORY="${3:-hampsonw/DeepSeek-V4-Flash-0731-WNA16}"
readonly QUANTIZER="${1:?usage: run-verda-full-conversion.sh QUANTIZER [RENTAL_ROOT] [HF_REPOSITORY]}"
readonly RENTAL_ROOT="${2:-$HOME/deepseek-v4-lowbit-rental}"
readonly SOURCE_DIRECTORY="$RENTAL_ROOT/source/DeepSeek-V4-Flash-0731"
readonly IMATRIX_PATH="$RENTAL_ROOT/imatrix/imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat"
readonly PILOT_REPORT="$RENTAL_ROOT/reports/w2-quantizer-comparison.json"
readonly PILOT_SUMMARY="$RENTAL_ROOT/reports/w2-quantizer-summary.json"
readonly FULL_RUN_LOG="$RENTAL_ROOT/reports/run-verda-full-conversion.log"
readonly UPLOAD_REPORT="$RENTAL_ROOT/reports/huggingface-upload-verification.json"
readonly CLUB_3090_DIRECTORY="$RENTAL_ROOT/club-3090"
readonly PYTHON_ENVIRONMENT="$RENTAL_ROOT/python-environment"
readonly RECIPE_PATH="$RENTAL_ROOT/recipes/all-w2.json"
readonly OUTPUT_DIRECTORY="$RENTAL_ROOT/output/DeepSeek-V4-Flash-0731-W2A16-$QUANTIZER"

mkdir -p "$RENTAL_ROOT/reports" "$RENTAL_ROOT/recipes"
exec > >(tee -a "$FULL_RUN_LOG") 2>&1

log_full_run_step() {
    printf '\n[%s] %s\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

case "$QUANTIZER" in
    plain-rtn | imatrix-weighted-rtn) ;;
    *)
        echo "Full conversion quantizer must be plain-rtn or imatrix-weighted-rtn" >&2
        exit 2
        ;;
esac
[[ -n "${HF_TOKEN:-}" ]] || {
    echo "Full conversion requires HF_TOKEN for durable private upload" >&2
    exit 2
}
[[ -x "$PYTHON_ENVIRONMENT/bin/python" ]] || {
    echo "Full conversion requires the completed rental pilot environment" >&2
    exit 2
}
[[ -f "$PILOT_REPORT" && -f "$PILOT_SUMMARY" ]] || {
    echo "Full conversion requires the completed pilot report and summary" >&2
    exit 2
}
if [[ "$QUANTIZER" == "imatrix-weighted-rtn" && ! -f "$IMATRIX_PATH" ]]; then
    echo "Weighted full conversion requires the verified routed-expert imatrix" >&2
    exit 2
fi

log_full_run_step "Verify pilot report, CUDA, authentication, and free storage"
"$PYTHON_ENVIRONMENT/bin/python" - "$PILOT_REPORT" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
results = report.get("results", [])
if len(results) != 48:
    raise SystemExit(f"Full conversion requires 48 pilot candidates, found {len(results)}")
if {item.get("quantizer") for item in results} != {"plain-rtn", "imatrix-weighted-rtn"}:
    raise SystemExit("Full conversion pilot report has unexpected quantizer candidates")
print(f"pilot_candidates={len(results)}")
PY
"$PYTHON_ENVIRONMENT/bin/python" - <<'PY'
import shutil
import torch
if not torch.cuda.is_available():
    raise SystemExit("Full conversion CUDA check failed")
free_gib = shutil.disk_usage(".").free / (1024**3)
if free_gib < 260:
    raise SystemExit(f"Full conversion requires at least 260 GiB free, found {free_gib:.1f}")
print(f"free_disk_gib={free_gib:.1f} gpu={torch.cuda.get_device_name(0)}")
PY
"$PYTHON_ENVIRONMENT/bin/hf" auth whoami

log_full_run_step "Update conversion tooling to pinned upload-verifier revision"
git -C "$CLUB_3090_DIRECTORY" fetch --depth 1 origin "$CLUB_3090_REF"
git -C "$CLUB_3090_DIRECTORY" checkout --detach "$CLUB_3090_REVISION"
test "$(git -C "$CLUB_3090_DIRECTORY" rev-parse HEAD)" = "$CLUB_3090_REVISION"
uv pip install --python "$PYTHON_ENVIRONMENT/bin/python" \
    --editable "$CLUB_3090_DIRECTORY/tools/deepseek-v4-lowbit"
uv pip check --python "$PYTHON_ENVIRONMENT/bin/python"

log_full_run_step "Resume full pinned official-checkpoint download"
"$PYTHON_ENVIRONMENT/bin/hf" download \
    "$SOURCE_REPOSITORY" \
    --revision "$DEEPSEEK_REVISION" \
    --local-dir "$SOURCE_DIRECTORY" \
    --max-workers 8

log_full_run_step "Materialize exact all-W2 MTP-free recipe"
printf '%s\n' '{"default":{"w13_bits":2,"w2_bits":2},"layers":{}}' > "$RECIPE_PATH"

log_full_run_step "Resume all-W2 MTP-free conversion with $QUANTIZER"
conversion_arguments=(
    "$SOURCE_DIRECTORY"
    "$OUTPUT_DIRECTORY"
    "$RECIPE_PATH"
    --device cuda
    --quantizer "$QUANTIZER"
)
if [[ "$QUANTIZER" == "imatrix-weighted-rtn" ]]; then
    conversion_arguments+=(--imatrix "$IMATRIX_PATH")
fi
"$PYTHON_ENVIRONMENT/bin/deepseek-v4-convert" "${conversion_arguments[@]}"

log_full_run_step "Upload artifact to private Hugging Face repository"
"$PYTHON_ENVIRONMENT/bin/hf" upload-large-folder \
    "$HUGGINGFACE_REPOSITORY" \
    "$OUTPUT_DIRECTORY" \
    --private \
    --exclude ".conversion-state/**" \
    --exclude ".cache/**" \
    --num-workers 4 \
    --no-bars

log_full_run_step "Cryptographically verify remote artifact inventory"
"$PYTHON_ENVIRONMENT/bin/deepseek-v4-verify-upload" \
    "$OUTPUT_DIRECTORY" \
    "$HUGGINGFACE_REPOSITORY" \
    "$UPLOAD_REPORT"

log_full_run_step "Full conversion and durable upload complete"
sha256sum \
    "$OUTPUT_DIRECTORY/model.safetensors.index.json" \
    "$OUTPUT_DIRECTORY/config.json" \
    "$OUTPUT_DIRECTORY/conversion-metrics.json" \
    "$UPLOAD_REPORT"
printf 'artifact=%s\nrepository=%s\nupload_report=%s\nfull_run_log=%s\n' \
    "$OUTPUT_DIRECTORY" \
    "$HUGGINGFACE_REPOSITORY" \
    "$UPLOAD_REPORT" \
    "$FULL_RUN_LOG"
