#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_SWIGLU_IMAGE_ID="sha256:a31c73626c16ed758dd33ac5c411b8f520b10c5843ddac35875d2b380e6eb185"
readonly EXPECTED_SWIGLU_COMMIT="a7758f7436a713f042e245b3e0aaab64b3a2f2c6"
readonly EXPECTED_SWIGLU_TREE="7f70947987c406d1e9fd0155a7fd5aa597520bb1"
readonly EXPECTED_DOCKERFILE_SHA256="a818836e0faa6163810d5595b89fc5dfd4d6940b418968bc5f59bb6bc5c6d8aa"
readonly EXPECTED_MATERIALIZER_SHA256="cc365ebe3949a8b2160a38cdadc8d0c14be55a6dc38fbc31a9c74dd6c6ec2439"
readonly CANDIDATE_REVISION="12035985bf555d0ddc603c6305586a8fa915589c"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly DOCKERFILE="$SCRIPT_DIRECTORY/Dockerfile.quality-candidate"
readonly MATERIALIZER="$SCRIPT_DIRECTORY/materialize-quality-candidate-runtime-view.py"
readonly IMAGE_TAG="${1:-club-3090/deepseek-v4-wna16-sm86:quality-12035985-a7758f74}"
readonly SWIGLU_RUNTIME_IMAGE="${2:-$EXPECTED_SWIGLU_IMAGE_ID}"

printf '%s  %s\n' "$EXPECTED_DOCKERFILE_SHA256" "$DOCKERFILE" |
    sha256sum --check --strict
printf '%s  %s\n' "$EXPECTED_MATERIALIZER_SHA256" "$MATERIALIZER" |
    sha256sum --check --strict

swiglu_image_id="$(
    docker image inspect "$SWIGLU_RUNTIME_IMAGE" --format '{{.Id}}'
)" || {
    echo "DeepSeek WNA16 SwiGLU runtime image is unavailable" >&2
    exit 2
}
[[ "$swiglu_image_id" == "$EXPECTED_SWIGLU_IMAGE_ID" ]] || {
    echo "DeepSeek WNA16 SwiGLU runtime image identity mismatch: $swiglu_image_id" >&2
    exit 2
}
swiglu_commit="$(
    docker image inspect "$SWIGLU_RUNTIME_IMAGE" \
        --format '{{index .Config.Labels "org.club3090.runtime.commit"}}'
)"
[[ "$swiglu_commit" == "$EXPECTED_SWIGLU_COMMIT" ]] || {
    echo "DeepSeek WNA16 SwiGLU runtime commit mismatch: $swiglu_commit" >&2
    exit 2
}
swiglu_tree="$(
    docker image inspect "$SWIGLU_RUNTIME_IMAGE" \
        --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
)"
[[ "$swiglu_tree" == "$EXPECTED_SWIGLU_TREE" ]] || {
    echo "DeepSeek WNA16 SwiGLU runtime tree mismatch: $swiglu_tree" >&2
    exit 2
}

verified_base_tag="club-3090/deepseek-v4-wna16-sm86:quality-base-${BASHPID}"
temporary_context="$(mktemp -d)"
cleanup_quality_candidate_build() {
    rm -rf "$temporary_context"
    docker image rm "$verified_base_tag" >/dev/null 2>&1 || true
}
trap cleanup_quality_candidate_build EXIT

docker tag "$swiglu_image_id" "$verified_base_tag"
cp -- "$MATERIALIZER" "$temporary_context/"
docker build \
    --pull=false \
    --build-arg "VERIFIED_SWIGLU_RUNTIME_IMAGE=$verified_base_tag" \
    --file "$DOCKERFILE" \
    --label "org.opencontainers.image.revision=$EXPECTED_SWIGLU_TREE" \
    --label "org.club3090.runtime.commit=$EXPECTED_SWIGLU_COMMIT" \
    --label "org.club3090.runtime.candidate-revision=$CANDIDATE_REVISION" \
    --label "org.club3090.runtime.production-base=$swiglu_image_id" \
    --label "org.club3090.runtime.scope=quality-candidate-deepswe-gate" \
    --tag "$IMAGE_TAG" \
    "$temporary_context"

docker image inspect "$IMAGE_TAG" --format \
    'image={{.Id}} tree={{index .Config.Labels "org.opencontainers.image.revision"}} candidate={{index .Config.Labels "org.club3090.runtime.candidate-revision"}} scope={{index .Config.Labels "org.club3090.runtime.scope"}} base={{index .Config.Labels "org.club3090.runtime.production-base"}}'
