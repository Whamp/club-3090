#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_VLLM_TREE="12b87bcd52bb2973685fa8f38b5fc8bbbfe7519c"
readonly EXPECTED_RUNTIME_DOCKERFILE_SHA256="7d4ab7f124d1ca5fc68facaafec8c55b98683e249cf669a2c102ac8ba6013838"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly DOCKERFILE="$SCRIPT_DIRECTORY/Dockerfile.runtime-cu130"
readonly VLLM_DIRECTORY="${1:?usage: build-runtime-image.sh VLLM_DIRECTORY [IMAGE_TAG]}"
readonly IMAGE_TAG="${2:-club-3090/deepseek-v4-wna16-sm86:runtime-12b87bcd-cu130}"

actual_dockerfile_sha256="$(sha256sum "$DOCKERFILE" | awk '{print $1}')"
[[ "$actual_dockerfile_sha256" == "$EXPECTED_RUNTIME_DOCKERFILE_SHA256" ]] || {
    echo "DeepSeek V4 runtime Dockerfile checksum mismatch: got $actual_dockerfile_sha256" >&2
    exit 2
}

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
    --label "org.club3090.runtime.contract-sha256=$EXPECTED_RUNTIME_DOCKERFILE_SHA256" \
    --tag "$IMAGE_TAG" \
    "$VLLM_DIRECTORY"

docker image inspect "$IMAGE_TAG" \
    --format 'image={{.Id}} tree={{index .Config.Labels "org.opencontainers.image.revision"}} contract={{index .Config.Labels "org.club3090.runtime.contract-sha256"}}'
