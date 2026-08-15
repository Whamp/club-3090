#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIRECTORY/../../../../.." && pwd)"
readonly REPOSITORY_ROOT
readonly URL="${URL:?set URL to the active experiment endpoint}"
readonly MODEL="${MODEL:?set MODEL to the active experiment model}"
readonly CONTAINER="${CONTAINER:?set CONTAINER to the active experiment container}"
readonly OUTPUT_DIRECTORY="${1:?usage: run-long-context-gate.sh OUTPUT_DIRECTORY}"
mkdir -p "$OUTPUT_DIRECTORY"

# Run the functional ladder independently from the documented 1 GiB advisory.
# The promoted 230,144 profile already carries an explicit 91-94 MiB warning;
# this gate records actual free memory and never relabels that ceiling as safe.
(
    cd "$REPOSITORY_ROOT"
    URL="$URL" MODEL="$MODEL" CONTAINER="$CONTAINER" \
    VRAM_MARGIN_MB=0 STRESS_LONGCTX_TIMEOUT_S=900 \
    bash scripts/verify-stress.sh
) | tee "$OUTPUT_DIRECTORY/verify-stress-functional.log"

nvidia-smi --query-gpu=index,memory.used,memory.free \
    --format=csv,noheader,nounits > "$OUTPUT_DIRECTORY/gpu-after-long-context.csv"
minimum_free_mib="$(awk -F, '{gsub(/ /, "", $3); if (NR == 1 || $3 < min) min=$3} END {print min}' "$OUTPUT_DIRECTORY/gpu-after-long-context.csv")"
printf 'minimum_free_mib=%s\n' "$minimum_free_mib" \
    > "$OUTPUT_DIRECTORY/headroom-advisory.txt"
if (( minimum_free_mib < 1024 )); then
    printf 'release_advisory=FAIL (<1024 MiB; capacity ceiling only)\n' \
        >> "$OUTPUT_DIRECTORY/headroom-advisory.txt"
else
    printf 'release_advisory=PASS\n' >> "$OUTPUT_DIRECTORY/headroom-advisory.txt"
fi

docker top "$CONTAINER" -eo pid | tail -n +2 > "$OUTPUT_DIRECTORY/worker-pids.txt"
: > "$OUTPUT_DIRECTORY/worker-swap-kib.txt"
while read -r pid; do
    swap_kib="$(awk '/^VmSwap:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)"
    printf '%s %s\n' "$pid" "${swap_kib:-0}" >> "$OUTPUT_DIRECTORY/worker-swap-kib.txt"
    [[ "${swap_kib:-0}" == 0 ]] || {
        echo "Serving process $pid has ${swap_kib} KiB swap" >&2
        exit 1
    }
done < "$OUTPUT_DIRECTORY/worker-pids.txt"
sha256sum "$OUTPUT_DIRECTORY"/* > "$OUTPUT_DIRECTORY/SHA256SUMS"
