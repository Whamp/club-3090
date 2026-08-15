#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_CANONICAL_COMMIT="7b39c93043ffa88729d2cd3dd1f8f482df6ea98c"
readonly EXPECTED_CANONICAL_TREE="670643653f99448f90192b79dd0842bcfa073ab8"
readonly EXPECTED_DSML_STOP_COMMIT="9a2ffbb4534400064e645cb4fef8ab2f2a987f11"
readonly EXPECTED_CANDIDATE_REVISION="12035985bf555d0ddc603c6305586a8fa915589c"
readonly EXPECTED_QUALITY_RUNTIME_IMAGE_ID="sha256:ed16ef3a2eadd6898d6c675a523efb455977d260324c97b8fd829830f8ab7d64"
readonly EXPECTED_DOCKERFILE_SHA256="0ac5486afceb5abec0e4714a831d40498411c6f7103500e945785a2e5bc67b4c"
readonly EXPECTED_ENVS_SHA256="0e0715e7d8f5c7b7bbfa62ceedd71c7706d49d24f8c3a74cf35f529b4d4078f1"
readonly EXPECTED_MARLIN_SHA256="e63a28ec1306cbc9e2b0b0c6000e58a8f51d9fad7128c5b22d3d57957c3de910"
readonly EXPECTED_SPARSE_MLA_SHA256="cb31aed4fc477edf325ed3a63279dca1d245a336eb82c180c46337cceddd2a15"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly DOCKERFILE="$SCRIPT_DIRECTORY/Dockerfile.quality-capacity"
readonly VLLM_DIRECTORY="${1:?usage: build-quality-capacity-image.sh VLLM_DIRECTORY [IMAGE_TAG] [QUALITY_RUNTIME_IMAGE]}"
readonly IMAGE_TAG="${2:-club-3090/deepseek-v4-wna16-sm86:quality-12035985-marlin-7b39c930}"
readonly QUALITY_RUNTIME_IMAGE="${3:-$EXPECTED_QUALITY_RUNTIME_IMAGE_ID}"
readonly ENVS_PATH="vllm/envs.py"
readonly MARLIN_PATH="vllm/model_executor/kernels/linear/scaled_mm/marlin.py"
readonly SPARSE_MLA_PATH="vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"

verify_file_sha256() {
    local expected_sha256="$1"
    local path="$2"
    printf '%s  %s\n' "$expected_sha256" "$path" | sha256sum --check --strict
}

verify_file_sha256 "$EXPECTED_DOCKERFILE_SHA256" "$DOCKERFILE"
actual_commit="$(git -C "$VLLM_DIRECTORY" rev-parse HEAD)"
[[ "$actual_commit" == "$EXPECTED_CANONICAL_COMMIT" ]] || {
    echo "DeepSeek V4 capacity runtime commit mismatch: $actual_commit" >&2
    exit 2
}
actual_tree="$(git -C "$VLLM_DIRECTORY" rev-parse 'HEAD^{tree}')"
[[ "$actual_tree" == "$EXPECTED_CANONICAL_TREE" ]] || {
    echo "DeepSeek V4 capacity runtime tree mismatch: $actual_tree" >&2
    exit 2
}
[[ -z "$(git -C "$VLLM_DIRECTORY" status --porcelain --untracked-files=all)" ]] || {
    echo "DeepSeek V4 capacity runtime requires a clean vLLM checkout" >&2
    exit 2
}
verify_file_sha256 "$EXPECTED_ENVS_SHA256" "$VLLM_DIRECTORY/$ENVS_PATH"
verify_file_sha256 "$EXPECTED_MARLIN_SHA256" "$VLLM_DIRECTORY/$MARLIN_PATH"
verify_file_sha256 "$EXPECTED_SPARSE_MLA_SHA256" "$VLLM_DIRECTORY/$SPARSE_MLA_PATH"

quality_runtime_image_id="$(
    docker image inspect "$QUALITY_RUNTIME_IMAGE" --format '{{.Id}}'
)" || {
    echo "DeepSeek V4 DSML-fixed quality runtime image is unavailable" >&2
    exit 2
}
[[ "$quality_runtime_image_id" == "$EXPECTED_QUALITY_RUNTIME_IMAGE_ID" ]] || {
    echo "DeepSeek V4 quality runtime image identity mismatch: $quality_runtime_image_id" >&2
    exit 2
}
base_dsml_commit="$(
    docker image inspect "$QUALITY_RUNTIME_IMAGE" \
        --format '{{index .Config.Labels "org.club3090.runtime.dsml-stop-commit"}}'
)"
[[ "$base_dsml_commit" == "$EXPECTED_DSML_STOP_COMMIT" ]] || {
    echo "DeepSeek V4 quality runtime DSML commit mismatch: $base_dsml_commit" >&2
    exit 2
}
base_candidate_revision="$(
    docker image inspect "$QUALITY_RUNTIME_IMAGE" \
        --format '{{index .Config.Labels "org.club3090.runtime.candidate-revision"}}'
)"
[[ "$base_candidate_revision" == "$EXPECTED_CANDIDATE_REVISION" ]] || {
    echo "DeepSeek V4 quality runtime candidate mismatch: $base_candidate_revision" >&2
    exit 2
}

verified_base_tag="club-3090/deepseek-v4-wna16-sm86:quality-capacity-base-${BASHPID}"
temporary_context="$(mktemp -d)"
cleanup_quality_capacity_build() {
    rm -rf "$temporary_context"
    docker image rm "$verified_base_tag" >/dev/null 2>&1 || true
}
trap cleanup_quality_capacity_build EXIT

docker tag "$quality_runtime_image_id" "$verified_base_tag"
for path in "$ENVS_PATH" "$MARLIN_PATH" "$SPARSE_MLA_PATH"; do
    mkdir -p "$temporary_context/$(dirname -- "$path")"
    cp -- "$VLLM_DIRECTORY/$path" "$temporary_context/$path"
done
docker build \
    --pull=false \
    --build-arg "VERIFIED_QUALITY_RUNTIME_IMAGE=$verified_base_tag" \
    --file "$DOCKERFILE" \
    --label "org.opencontainers.image.revision=$EXPECTED_CANONICAL_TREE" \
    --label "org.club3090.runtime.canonical-commit=$EXPECTED_CANONICAL_COMMIT" \
    --label "org.club3090.runtime.dsml-stop-commit=$EXPECTED_DSML_STOP_COMMIT" \
    --label "org.club3090.runtime.candidate-revision=$EXPECTED_CANDIDATE_REVISION" \
    --label "org.club3090.runtime.production-base=$quality_runtime_image_id" \
    --label "org.club3090.runtime.scope=quality-capacity-marlin" \
    --tag "$IMAGE_TAG" \
    "$temporary_context"

docker image inspect "$IMAGE_TAG" --format \
    'image={{.Id}} tree={{index .Config.Labels "org.opencontainers.image.revision"}} commit={{index .Config.Labels "org.club3090.runtime.canonical-commit"}} candidate={{index .Config.Labels "org.club3090.runtime.candidate-revision"}} scope={{index .Config.Labels "org.club3090.runtime.scope"}} base={{index .Config.Labels "org.club3090.runtime.production-base"}}'
