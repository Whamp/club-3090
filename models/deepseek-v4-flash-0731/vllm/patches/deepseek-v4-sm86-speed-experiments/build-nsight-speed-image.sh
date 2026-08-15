#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_VLLM_COMMIT="91a39786d48f48efb45fbe3a160d448c783b0131"
readonly EXPECTED_VLLM_TREE="5238d1e4148bc747e122b9bc19bb1562a05b3207"
readonly EXPECTED_DOCKERFILE_SHA256="c858ec541b5ee817c3d5e7b9e2968ff3cbd60829befe091145866d015ee802c1"
readonly EXPECTED_ENTRYPOINT_SHA256="001d59d535180af11262f902728de3aecfe875e4e4101fd109994a311aea4f6c"
readonly NSIGHT_VERSION="2026.4.1.191-3860507"
readonly NSIGHT_DEB_SHA256="b896cb2b9586ddf617c363a43bababad0a015dff4c77d8f0fbb9c26144056a69"
readonly NSIGHT_DEB_NAME="NsightSystems-linux-cli-public-${NSIGHT_VERSION}.deb"
readonly NSIGHT_DEB_URL="https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/$NSIGHT_DEB_NAME"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly DOCKERFILE="$SCRIPT_DIRECTORY/Dockerfile.nsight"
readonly ENTRYPOINT_PATH="$SCRIPT_DIRECTORY/nsys-vllm-entrypoint.sh"
readonly SPEED_IMAGE="${1:?usage: build-nsight-speed-image.sh SPEED_IMAGE SPEED_IMAGE_ID [OUTPUT_IMAGE]}"
readonly SPEED_IMAGE_ID="${2:?usage: build-nsight-speed-image.sh SPEED_IMAGE SPEED_IMAGE_ID [OUTPUT_IMAGE]}"
readonly OUTPUT_IMAGE="${3:-club-3090/deepseek-v4-wna16-sm86:quality-12035985-nsight-91a39786}"

verify_file_sha256() {
    local expected_sha256="$1"
    local path="$2"
    printf '%s  %s\n' "$expected_sha256" "$path" | sha256sum --check --strict
}

verify_file_sha256 "$EXPECTED_DOCKERFILE_SHA256" "$DOCKERFILE"
verify_file_sha256 "$EXPECTED_ENTRYPOINT_SHA256" "$ENTRYPOINT_PATH"
actual_speed_image_id="$(docker image inspect "$SPEED_IMAGE" --format '{{.Id}}')"
[[ "$actual_speed_image_id" == "$SPEED_IMAGE_ID" ]] || {
    echo "DeepSeek V4 speed image identity mismatch: $actual_speed_image_id" >&2
    exit 2
}
[[ "$(docker image inspect "$SPEED_IMAGE" --format '{{index .Config.Labels "org.club3090.runtime.canonical-commit"}}')" == "$EXPECTED_VLLM_COMMIT" ]] || {
    echo "DeepSeek V4 speed image commit mismatch" >&2
    exit 2
}
[[ "$(docker image inspect "$SPEED_IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')" == "$EXPECTED_VLLM_TREE" ]] || {
    echo "DeepSeek V4 speed image tree mismatch" >&2
    exit 2
}

temporary_context="$(mktemp -d)"
verified_base_tag="club-3090/deepseek-v4-wna16-sm86:nsight-base-${BASHPID}"
cleanup_nsight_build() {
    rm -rf "$temporary_context"
    docker image rm "$verified_base_tag" >/dev/null 2>&1 || true
}
trap cleanup_nsight_build EXIT

docker tag "$actual_speed_image_id" "$verified_base_tag"
cp -- "$ENTRYPOINT_PATH" "$temporary_context/nsys-vllm-entrypoint.sh"
curl --fail --location --silent --show-error --retry 3 \
    --output "$temporary_context/$NSIGHT_DEB_NAME" "$NSIGHT_DEB_URL"
verify_file_sha256 "$NSIGHT_DEB_SHA256" "$temporary_context/$NSIGHT_DEB_NAME"

docker build --pull=false \
    --build-arg "VERIFIED_SPEED_RUNTIME_IMAGE=$verified_base_tag" \
    --file "$DOCKERFILE" \
    --label "org.opencontainers.image.revision=$EXPECTED_VLLM_TREE" \
    --label "org.club3090.runtime.canonical-commit=$EXPECTED_VLLM_COMMIT" \
    --label "org.club3090.runtime.production-base=$actual_speed_image_id" \
    --label "org.club3090.runtime.nsight-version=$NSIGHT_VERSION" \
    --label "org.club3090.runtime.nsight-deb-sha256=$NSIGHT_DEB_SHA256" \
    --label "org.club3090.runtime.scope=deepseek-v4-sm86-nsight" \
    --tag "$OUTPUT_IMAGE" \
    "$temporary_context"

docker image inspect "$OUTPUT_IMAGE" --format \
    'image={{.Id}} tree={{index .Config.Labels "org.opencontainers.image.revision"}} commit={{index .Config.Labels "org.club3090.runtime.canonical-commit"}} nsys={{index .Config.Labels "org.club3090.runtime.nsight-version"}} base={{index .Config.Labels "org.club3090.runtime.production-base"}}'
