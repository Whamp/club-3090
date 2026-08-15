#!/usr/bin/env bash
set -euo pipefail

readonly CONTAINER="${CONTAINER:?set CONTAINER to the active experiment container}"
[[ $# -gt 0 ]] || {
    echo "usage: normalize-swap-then-measure.sh COMMAND [ARGUMENT ...]" >&2
    exit 2
}

mem_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
swap_total_kib="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
swap_free_kib="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
swap_used_kib="$((swap_total_kib - swap_free_kib))"
readonly reserve_kib=$((8 * 1024 * 1024))
if (( mem_available_kib < swap_used_kib + reserve_kib )); then
    echo "Refusing swap normalization: MemAvailable=${mem_available_kib} KiB, swap_used=${swap_used_kib} KiB, reserve=${reserve_kib} KiB" >&2
    exit 1
fi

sudo -n swapoff -a
sudo -n swapon -a
[[ "$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)" == \
    "$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)" ]] || {
    echo "Swap normalization did not return to zero used swap" >&2
    exit 1
}

while read -r pid; do
    swap_kib="$(awk '/^VmSwap:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)"
    [[ "${swap_kib:-0}" == 0 ]] || {
        echo "Serving process $pid still has ${swap_kib} KiB swap" >&2
        exit 1
    }
done < <(docker top "$CONTAINER" -eo pid | tail -n +2)

exec "$@"
