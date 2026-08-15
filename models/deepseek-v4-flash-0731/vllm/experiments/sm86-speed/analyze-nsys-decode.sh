#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIRECTORY/../../../../.." && pwd)"
readonly REPOSITORY_ROOT
readonly TOOL_PROJECT="$REPOSITORY_ROOT/tools/deepseek-v4-lowbit"
readonly NSIGHT_IMAGE="${1:?usage: analyze-nsys-decode.sh NSIGHT_IMAGE PROFILE.nsys-rep OUTPUT_DIRECTORY}"
readonly PROFILE="${2:?usage: analyze-nsys-decode.sh NSIGHT_IMAGE PROFILE.nsys-rep OUTPUT_DIRECTORY}"
readonly OUTPUT_DIRECTORY="${3:?usage: analyze-nsys-decode.sh NSIGHT_IMAGE PROFILE.nsys-rep OUTPUT_DIRECTORY}"

[[ -f "$PROFILE" ]] || { echo "Nsight profile is missing: $PROFILE" >&2; exit 2; }
mkdir -p "$OUTPUT_DIRECTORY"
profile_directory="$(cd -- "$(dirname -- "$PROFILE")" && pwd)"
profile_name="$(basename -- "$PROFILE")"

# This summary is a screening aid only. Summed kernel time is not critical-path
# attribution; the timeline still requires operator review before hier-allreduce.
docker run --rm --entrypoint nsys \
    --volume "$profile_directory:/profiles:ro" \
    "$NSIGHT_IMAGE" stats --report cuda_gpu_kern_sum --format csv \
    "/profiles/$profile_name" > "$OUTPUT_DIRECTORY/cuda-gpu-kernel-summary.csv"
(
    cd "$TOOL_PROJECT"
    uv run --python 3.12 deepseek-v4-summarize-nsys-kernels \
        "$OUTPUT_DIRECTORY/cuda-gpu-kernel-summary.csv" \
        --output "$OUTPUT_DIRECTORY/cuda-gpu-kernel-summary.json"
)
sha256sum "$PROFILE" > "$OUTPUT_DIRECTORY/profile.sha256"
sha256sum "$OUTPUT_DIRECTORY"/*.csv "$OUTPUT_DIRECTORY"/*.json \
    > "$OUTPUT_DIRECTORY/SHA256SUMS"

cat <<EOF
Nsight summary written to $OUTPUT_DIRECTORY.
Review $PROFILE in the Nsight/Perfetto timeline before creating trace-gate.json.
The trace gate requires an operator-reviewed critical-path fraction; do not copy
nccl_kernel_time_fraction blindly because summed GPU kernel time can overlap.
EOF
