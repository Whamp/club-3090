#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_VLLM_TREE="9a54d487051e937a9dd6c146b971d93ff422eb30"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly DOCKERFILE="$SCRIPT_DIRECTORY/Dockerfile.runtime-cu130"
readonly VLLM_DIRECTORY="${1:?usage: build-runtime-image.sh VLLM_DIRECTORY [IMAGE_TAG]}"
readonly IMAGE_TAG="${2:-club-3090/deepseek-v4-wna16-sm86:9a54d487-cu130}"

actual_tree="$(git -C "$VLLM_DIRECTORY" rev-parse 'HEAD^{tree}')"
[[ "$actual_tree" == "$EXPECTED_VLLM_TREE" ]] || {
    echo "DeepSeek V4 runtime build tree mismatch: got $actual_tree, expected $EXPECTED_VLLM_TREE" >&2
    exit 2
}
[[ -z "$(git -C "$VLLM_DIRECTORY" status --porcelain --untracked-files=all)" ]] || {
    echo "DeepSeek V4 runtime build requires a clean vLLM checkout: $VLLM_DIRECTORY" >&2
    exit 2
}

docker build \
    --file "$DOCKERFILE" \
    --label "org.opencontainers.image.revision=$EXPECTED_VLLM_TREE" \
    --tag "$IMAGE_TAG" \
    "$VLLM_DIRECTORY"

docker image inspect "$IMAGE_TAG" \
    --format 'image={{.Id}} tree={{index .Config.Labels "org.opencontainers.image.revision"}}'
