#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_ACCEPTANCE_TREE="f73b30cc5a2ed9de200ca2e4de3cdef1a06f6538"
readonly EXPECTED_PRODUCTION_TREE="aeb62948e33074514a742d19c2f9a1a3c2ee3e1f"
readonly EXPECTED_PRODUCTION_IMAGE_ID="sha256:0beb1f0cba2e41837f4ba5af01cc5c4686afde4f40ab1df5147a6ad945b0af1f"
readonly EXPECTED_LOADER_SHA256="fd512829989af7d86f39a618990d52916aab6ae4b4d70259523c340b2574a830"
readonly EXPECTED_MOE_TEST_SHA256="975297b7c4404fc770097f9b0d3bf43a45e90c212d621867800ddd6b6347d4c4"
readonly EXPECTED_MAPPER_TEST_SHA256="7e46912885783b317b4634d8a03931749b1640372c8fc21f20c01dc7dfe6a334"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly DOCKERFILE="$SCRIPT_DIRECTORY/Dockerfile.mixed-group-oracle"
readonly VLLM_DIRECTORY="${1:?usage: build-mixed-group-oracle-image.sh VLLM_DIRECTORY [IMAGE_TAG]}"
readonly IMAGE_TAG="${2:-club-3090/deepseek-v4-wna16-sm86:mixed-group-oracle-f73b30cc}"
readonly PRODUCTION_IMAGE="${VLLM_PRODUCTION_IMAGE:-$EXPECTED_PRODUCTION_IMAGE_ID}"

actual_tree="$(git -C "$VLLM_DIRECTORY" rev-parse 'HEAD^{tree}')"
[[ "$actual_tree" == "$EXPECTED_ACCEPTANCE_TREE" ]] || {
    echo "Mixed-group oracle source tree mismatch: got $actual_tree" >&2
    exit 2
}
[[ -z "$(git -C "$VLLM_DIRECTORY" status --porcelain --untracked-files=all)" ]] || {
    echo "Mixed-group oracle requires a clean vLLM checkout" >&2
    exit 2
}

for record in \
    "$EXPECTED_LOADER_SHA256 vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py" \
    "$EXPECTED_MOE_TEST_SHA256 tests/kernels/moe/test_moe.py" \
    "$EXPECTED_MAPPER_TEST_SHA256 tests/models/test_deepseek_v4_weight_mapper.py"; do
    expected="${record%% *}"
    relative_path="${record#* }"
    actual="$(sha256sum "$VLLM_DIRECTORY/$relative_path" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "Mixed-group oracle test checksum mismatch: $relative_path" >&2
        exit 2
    }
done

production_image_id="$(docker image inspect "$PRODUCTION_IMAGE" --format '{{.Id}}')"
[[ "$production_image_id" == "$EXPECTED_PRODUCTION_IMAGE_ID" ]] || {
    echo "Mixed-group oracle production image mismatch: $production_image_id" >&2
    exit 2
}
production_tree="$(
    docker image inspect "$PRODUCTION_IMAGE" \
        --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
)"
[[ "$production_tree" == "$EXPECTED_PRODUCTION_TREE" ]] || {
    echo "Mixed-group oracle production tree mismatch: $production_tree" >&2
    exit 2
}

verified_base_tag="club-3090/deepseek-v4-wna16-sm86:mixed-group-oracle-base-${BASHPID}"
temporary_context="$(mktemp -d)"
cleanup_mixed_group_oracle_build() {
    rm -rf "$temporary_context"
    docker image rm "$verified_base_tag" >/dev/null 2>&1 || true
}
trap cleanup_mixed_group_oracle_build EXIT

docker tag "$production_image_id" "$verified_base_tag"
mkdir -p \
    "$temporary_context/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe" \
    "$temporary_context/tests/kernels/moe" \
    "$temporary_context/tests/models"
cp -- \
    "$VLLM_DIRECTORY/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py" \
    "$temporary_context/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py"
cp -- \
    "$VLLM_DIRECTORY/tests/kernels/moe/test_moe.py" \
    "$temporary_context/tests/kernels/moe/test_moe.py"
cp -- \
    "$VLLM_DIRECTORY/tests/models/test_deepseek_v4_weight_mapper.py" \
    "$temporary_context/tests/models/test_deepseek_v4_weight_mapper.py"

docker build \
    --pull=false \
    --build-arg "VERIFIED_PRODUCTION_IMAGE=$verified_base_tag" \
    --file "$DOCKERFILE" \
    --label "org.opencontainers.image.revision=$EXPECTED_ACCEPTANCE_TREE" \
    --label "org.club3090.runtime.production-base=$EXPECTED_PRODUCTION_IMAGE_ID" \
    --label "org.club3090.runtime.scope=mixed-group-oracle-only" \
    --tag "$IMAGE_TAG" \
    "$temporary_context"

docker image inspect "$IMAGE_TAG" --format \
    'image={{.Id}} tree={{index .Config.Labels "org.opencontainers.image.revision"}} scope={{index .Config.Labels "org.club3090.runtime.scope"}}'
