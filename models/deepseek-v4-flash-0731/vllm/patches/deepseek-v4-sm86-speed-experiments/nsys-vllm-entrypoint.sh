#!/usr/bin/env bash
set -euo pipefail

readonly REPORT_DIRECTORY="${NSYS_REPORT_DIRECTORY:-/profiles}"
readonly REPORT_BASENAME="${NSYS_REPORT_BASENAME:-deepseek-v4-decode}"
mkdir -p "$REPORT_DIRECTORY"

exec nsys profile \
    --output "$REPORT_DIRECTORY/$REPORT_BASENAME" \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt,cublas,nccl \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    /opt/venv/bin/vllm serve "$@" \
    --profiler-config.profiler cuda
