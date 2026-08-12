#!/usr/bin/env bash
set -euo pipefail

readonly ARTIFACT_DIRECTORY="${DEEPSEEK_WNA16_ARTIFACT_DIRECTORY:-/artifact}"
readonly RUNTIME_MODEL_DIRECTORY="${DEEPSEEK_WNA16_RUNTIME_MODEL_DIRECTORY:-/runtime-model}"

/usr/local/bin/materialize-deepseek-v4-wna16-runtime-view \
    "$ARTIFACT_DIRECTORY" \
    "$RUNTIME_MODEL_DIRECTORY"

exec vllm serve "$@"
