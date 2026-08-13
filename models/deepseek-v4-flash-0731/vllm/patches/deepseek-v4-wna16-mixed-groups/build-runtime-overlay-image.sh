#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_VLLM_TREE="f73b30cc5a2ed9de200ca2e4de3cdef1a06f6538"
readonly EXPECTED_PARENT_TREE="aeb62948e33074514a742d19c2f9a1a3c2ee3e1f"
readonly EXPECTED_CANONICAL_COMMIT="dd2d1fd6779addccc73094f77fa4ada7d9106a41"
readonly EXPECTED_PRODUCTION_IMAGE_ID="sha256:0beb1f0cba2e41837f4ba5af01cc5c4686afde4f40ab1df5147a6ad945b0af1f"
readonly EXPECTED_RUNTIME_BASE_ID="sha256:0e8cc6dc48081e907d553febc8002b1f6d61298454340840f27f18b3a2e66c6c"
readonly EXPECTED_DOCKERFILE_SHA256="b0f8ab5993254b1a8e7ee79c78d0eba56e8f9d4de0bfcb9bf0c1f0928b425e9c"
readonly EXPECTED_LOADER_SHA256="fd512829989af7d86f39a618990d52916aab6ae4b4d70259523c340b2574a830"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly DOCKERFILE="$SCRIPT_DIRECTORY/Dockerfile.runtime-overlay"
readonly VLLM_DIRECTORY="${1:?usage: build-runtime-overlay-image.sh VLLM_DIRECTORY [IMAGE_TAG] [PRODUCTION_IMAGE]}"
readonly IMAGE_TAG="${2:-club-3090/deepseek-v4-wna16-sm86:f73b30cc-mixed-groups-cu130}"
readonly PRODUCTION_IMAGE="${3:-$EXPECTED_PRODUCTION_IMAGE_ID}"
readonly LOADER_PATH="vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py"

actual_dockerfile_sha256="$(sha256sum "$DOCKERFILE" | awk '{print $1}')"
[[ "$actual_dockerfile_sha256" == "$EXPECTED_DOCKERFILE_SHA256" ]] || {
    echo "DeepSeek V4 mixed-group runtime Dockerfile checksum mismatch: $actual_dockerfile_sha256" >&2
    exit 2
}
actual_tree="$(git -C "$VLLM_DIRECTORY" rev-parse 'HEAD^{tree}')"
[[ "$actual_tree" == "$EXPECTED_VLLM_TREE" ]] || {
    echo "DeepSeek V4 mixed-group runtime tree mismatch: got $actual_tree" >&2
    exit 2
}
[[ -z "$(git -C "$VLLM_DIRECTORY" status --porcelain --untracked-files=all)" ]] || {
    echo "DeepSeek V4 mixed-group runtime requires a clean vLLM checkout" >&2
    exit 2
}
actual_loader_sha256="$(sha256sum "$VLLM_DIRECTORY/$LOADER_PATH" | awk '{print $1}')"
[[ "$actual_loader_sha256" == "$EXPECTED_LOADER_SHA256" ]] || {
    echo "DeepSeek V4 mixed-group loader checksum mismatch: $actual_loader_sha256" >&2
    exit 2
}

production_image_id="$(docker image inspect "$PRODUCTION_IMAGE" --format '{{.Id}}')" || {
    echo "DeepSeek V4 accepted production image is unavailable: $PRODUCTION_IMAGE" >&2
    exit 2
}
if [[ "$production_image_id" != "$EXPECTED_PRODUCTION_IMAGE_ID" ]]; then
    production_tree="$(
        docker image inspect "$PRODUCTION_IMAGE" \
            --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
    )"
    runtime_base_id="$(
        docker image inspect "$PRODUCTION_IMAGE" \
            --format '{{index .Config.Labels "org.club3090.runtime.base-digest"}}'
    )"
    [[ "$production_tree" == "$EXPECTED_PARENT_TREE" ]] || {
        echo "DeepSeek V4 mixed-group production tree mismatch: $production_tree" >&2
        exit 2
    }
    [[ "$runtime_base_id" == "$EXPECTED_RUNTIME_BASE_ID" ]] || {
        echo "DeepSeek V4 mixed-group runtime base mismatch: $runtime_base_id" >&2
        exit 2
    }
fi

verified_base_tag="club-3090/deepseek-v4-wna16-sm86:mixed-group-base-${BASHPID}"
temporary_context="$(mktemp -d)"
cleanup_mixed_group_runtime_build() {
    rm -rf "$temporary_context"
    docker image rm "$verified_base_tag" >/dev/null 2>&1 || true
}
trap cleanup_mixed_group_runtime_build EXIT

docker tag "$production_image_id" "$verified_base_tag"
mkdir -p "$temporary_context/$(dirname -- "$LOADER_PATH")"
cp -- "$VLLM_DIRECTORY/$LOADER_PATH" "$temporary_context/$LOADER_PATH"
docker build \
    --pull=false \
    --build-arg "VERIFIED_PRODUCTION_IMAGE=$verified_base_tag" \
    --file "$DOCKERFILE" \
    --label "org.opencontainers.image.revision=$EXPECTED_VLLM_TREE" \
    --label "org.club3090.runtime.canonical-commit=$EXPECTED_CANONICAL_COMMIT" \
    --label "org.club3090.runtime.production-base=$production_image_id" \
    --label "org.club3090.runtime.scope=mixed-projection-groups-candidate" \
    --tag "$IMAGE_TAG" \
    "$temporary_context"

docker image inspect "$IMAGE_TAG" --format \
    'image={{.Id}} tree={{index .Config.Labels "org.opencontainers.image.revision"}} scope={{index .Config.Labels "org.club3090.runtime.scope"}} base={{index .Config.Labels "org.club3090.runtime.production-base"}}'
