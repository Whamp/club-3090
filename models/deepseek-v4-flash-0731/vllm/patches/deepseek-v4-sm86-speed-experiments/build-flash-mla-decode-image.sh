#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_VLLM_COMMIT="b7766cfe4d15d9b68acea43097ceff221e8a739f"
readonly EXPECTED_VLLM_TREE="6354125afd1306c9286f734d1c47c23c767d77a9"
readonly EXPECTED_CAPACITY_IMAGE_ID="sha256:f56910530683326051cfdf4e7c8e4d6afc5bace8804cb78b2af9ea799bbba4e6"
readonly EXPECTED_DOCKERFILE_SHA256="f66da13f324bd6092b72396b7feb1a5ed7352b4b99969f3dd9242226ca30b160"
readonly EXPECTED_ENVS_SHA256="a1e3fe3c67169a013c429cc3754fcd0432da475686c05f73ad88b5c8d9a2fa69"
readonly EXPECTED_AMPERE_SPARSE_SHA256="219fc766a7da71871fde8628d671eebdf84a5de3d9e83fcb6a716308ffcddab0"
readonly EXPECTED_CUDA_COMMUNICATOR_SHA256="d32440190e4007824c3043e8aa94bdfdbbf04c354a1bd2f7ebea527e25bc6cdf"
readonly EXPECTED_KV_OFFLOAD_GPU_WORKER_SHA256="bb3704cd412bb9ae93dc4dbcb1e2d6a38c18cfc895c6cd55f176ffe17c98ac96"
readonly EXPECTED_SHARED_OFFLOAD_REGION_SHA256="1a784349d8a8d970442ef12360da2ea1fd9b3bf31a4206a82840bd91e3afc12f"
readonly FLASH_MLA_SOURCE_COMMIT="7f41a5baa5cf57bfbce06458794b4b05737a162a"
readonly FLASH_MLA_WHEEL_SHA256="1e750446aa04b1f325fd1ca29be5d6b3e62f69df69e7ccd4b45df2c267b694d3"
readonly FLASH_MLA_WHEEL_NAME="flash_mla-2.0.0-cp39-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
readonly FLASH_MLA_WHEEL_URL="https://github.com/AppMana/forks-flash-mla-int/releases/download/v2.0.0/$FLASH_MLA_WHEEL_NAME"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly DOCKERFILE="$SCRIPT_DIRECTORY/Dockerfile.flash-mla-decode"
readonly VLLM_DIRECTORY="${1:?usage: build-flash-mla-decode-image.sh VLLM_DIRECTORY [IMAGE_TAG] [CAPACITY_IMAGE]}"
readonly IMAGE_TAG="${2:-club-3090/deepseek-v4-wna16-sm86:quality-12035985-speed-b7766cfe}"
readonly CAPACITY_IMAGE="${3:-$EXPECTED_CAPACITY_IMAGE_ID}"
readonly ENVS_PATH="vllm/envs.py"
readonly CUDA_COMMUNICATOR_PATH="vllm/distributed/device_communicators/cuda_communicator.py"
readonly AMPERE_SPARSE_PATH="vllm/models/deepseek_v4/ampere/ampere_sparse.py"
readonly KV_OFFLOAD_GPU_WORKER_PATH="vllm/v1/kv_offload/cpu/gpu_worker.py"
readonly SHARED_OFFLOAD_REGION_PATH="vllm/v1/kv_offload/cpu/shared_offload_region.py"

verify_file_sha256() {
    local expected_sha256="$1"
    local path="$2"
    printf '%s  %s\n' "$expected_sha256" "$path" | sha256sum --check --strict
}

verify_file_sha256 "$EXPECTED_DOCKERFILE_SHA256" "$DOCKERFILE"
actual_commit="$(git -C "$VLLM_DIRECTORY" rev-parse HEAD)"
[[ "$actual_commit" == "$EXPECTED_VLLM_COMMIT" ]] || {
    echo "DeepSeek V4 speed runtime commit mismatch: $actual_commit" >&2
    exit 2
}
actual_tree="$(git -C "$VLLM_DIRECTORY" rev-parse 'HEAD^{tree}')"
[[ "$actual_tree" == "$EXPECTED_VLLM_TREE" ]] || {
    echo "DeepSeek V4 speed runtime tree mismatch: $actual_tree" >&2
    exit 2
}
[[ -z "$(git -C "$VLLM_DIRECTORY" status --porcelain --untracked-files=all)" ]] || {
    echo "DeepSeek V4 speed runtime requires a clean vLLM checkout" >&2
    exit 2
}
verify_file_sha256 "$EXPECTED_ENVS_SHA256" "$VLLM_DIRECTORY/$ENVS_PATH"
verify_file_sha256 "$EXPECTED_AMPERE_SPARSE_SHA256" \
    "$VLLM_DIRECTORY/$AMPERE_SPARSE_PATH"
verify_file_sha256 "$EXPECTED_CUDA_COMMUNICATOR_SHA256" \
    "$VLLM_DIRECTORY/$CUDA_COMMUNICATOR_PATH"
verify_file_sha256 "$EXPECTED_KV_OFFLOAD_GPU_WORKER_SHA256" \
    "$VLLM_DIRECTORY/$KV_OFFLOAD_GPU_WORKER_PATH"
verify_file_sha256 "$EXPECTED_SHARED_OFFLOAD_REGION_SHA256" \
    "$VLLM_DIRECTORY/$SHARED_OFFLOAD_REGION_PATH"

capacity_image_id="$(docker image inspect "$CAPACITY_IMAGE" --format '{{.Id}}')" || {
    echo "DeepSeek V4 validated capacity image is unavailable" >&2
    exit 2
}
[[ "$capacity_image_id" == "$EXPECTED_CAPACITY_IMAGE_ID" ]] || {
    echo "DeepSeek V4 capacity image identity mismatch: $capacity_image_id" >&2
    exit 2
}

verified_base_tag="club-3090/deepseek-v4-wna16-sm86:speed-base-${BASHPID}"
temporary_context="$(mktemp -d)"
cleanup_flash_mla_build() {
    rm -rf "$temporary_context"
    docker image rm "$verified_base_tag" >/dev/null 2>&1 || true
}
trap cleanup_flash_mla_build EXIT

docker tag "$capacity_image_id" "$verified_base_tag"
for path in "$ENVS_PATH" "$CUDA_COMMUNICATOR_PATH" "$AMPERE_SPARSE_PATH" \
    "$KV_OFFLOAD_GPU_WORKER_PATH" "$SHARED_OFFLOAD_REGION_PATH"; do
    mkdir -p "$temporary_context/$(dirname -- "$path")"
    cp -- "$VLLM_DIRECTORY/$path" "$temporary_context/$path"
done
curl --fail --location --silent --show-error --retry 3 \
    --output "$temporary_context/$FLASH_MLA_WHEEL_NAME" \
    "$FLASH_MLA_WHEEL_URL"
verify_file_sha256 "$FLASH_MLA_WHEEL_SHA256" \
    "$temporary_context/$FLASH_MLA_WHEEL_NAME"

docker build \
    --pull=false \
    --build-arg "VERIFIED_CAPACITY_RUNTIME_IMAGE=$verified_base_tag" \
    --file "$DOCKERFILE" \
    --label "org.opencontainers.image.revision=$EXPECTED_VLLM_TREE" \
    --label "org.club3090.runtime.canonical-commit=$EXPECTED_VLLM_COMMIT" \
    --label "org.club3090.runtime.production-base=$capacity_image_id" \
    --label "org.club3090.runtime.flash-mla-source=$FLASH_MLA_SOURCE_COMMIT" \
    --label "org.club3090.runtime.flash-mla-wheel-sha256=$FLASH_MLA_WHEEL_SHA256" \
    --label "org.club3090.runtime.scope=deepseek-v4-sm86-speed-experiments" \
    --tag "$IMAGE_TAG" \
    "$temporary_context"

docker image inspect "$IMAGE_TAG" --format \
    'image={{.Id}} tree={{index .Config.Labels "org.opencontainers.image.revision"}} commit={{index .Config.Labels "org.club3090.runtime.canonical-commit"}} flash_mla={{index .Config.Labels "org.club3090.runtime.flash-mla-source"}} wheel={{index .Config.Labels "org.club3090.runtime.flash-mla-wheel-sha256"}} base={{index .Config.Labels "org.club3090.runtime.production-base"}}'
