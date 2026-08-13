#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_VLLM_TREE="aeb62948e33074514a742d19c2f9a1a3c2ee3e1f"
readonly EXPECTED_BASE_IMAGE_DIGEST="sha256:0e8cc6dc48081e907d553febc8002b1f6d61298454340840f27f18b3a2e66c6c"
readonly EXPECTED_RUNTIME_CONTRACT_SHA256="7d4ab7f124d1ca5fc68facaafec8c55b98683e249cf669a2c102ac8ba6013838"
readonly EXPECTED_MATERIALIZER_SHA256="8aff33ec192e0a67203c974918ad4c74bb875ec6d320ede6276077cce872d80c"
readonly EXPECTED_STARTER_SHA256="6bf86f1e90c58bd3e5964bca8938515dc73073b3d27fe9a41f6ba67b7ed0eeaa"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly DOCKERFILE="$SCRIPT_DIRECTORY/Dockerfile.final-overlay"
readonly VLLM_DIRECTORY="${1:?usage: build-final-overlay-image.sh VLLM_DIRECTORY [IMAGE_TAG] [BASE_IMAGE]}"
readonly IMAGE_TAG="${2:-club-3090/deepseek-v4-wna16-sm86:aeb62948-rope-cu130}"
readonly BASE_IMAGE="${3:-$EXPECTED_BASE_IMAGE_DIGEST}"

readonly -a PINNED_SOURCE_HASHES=(
    "973692c269a16f2f9791867aa07aab7ad328b26b38f1be6cd5054a43d15eb23b  vllm/model_executor/layers/rotary_embedding/__init__.py"
    "59c6cce38f43d214c1cde9f26d3287ab4eb1fee13978a32d846add2b85a815db  vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py"
    "07e06cb5489f02f761b99422235014bc6f1cab8c1f799ea2bf7855112dd68910  vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py"
    "e0da11160d84fdf9c56ad0848f77372ac81d7b089753b06213ce7b9dac224091  vllm/models/deepseek_v4/attention.py"
    "6180c64a7e6caad5a3d887fcf4cecada11122cba60eda5339a66c563a130ba21  vllm/models/deepseek_v4/common/rope.py"
    "880bf06530aab3bf8c7b60a8a125663e9c145a2a9ad27ac99cbe0b27cda50b62  vllm/models/deepseek_v4/quant_config.py"
    "ffdb2abe98456d8b1601bbac51cb113d7018bd3db0296ed65e51cf459cf6923a  vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"
)

grep -Fqx "FROM \${VERIFIED_BASE_IMAGE}" "$DOCKERFILE" || {
    echo "DeepSeek V4 final overlay Dockerfile does not use VERIFIED_BASE_IMAGE" >&2
    exit 2
}

actual_tree="$(git -C "$VLLM_DIRECTORY" rev-parse 'HEAD^{tree}')"
[[ "$actual_tree" == "$EXPECTED_VLLM_TREE" ]] || {
    echo "DeepSeek V4 final overlay tree mismatch: got $actual_tree, expected $EXPECTED_VLLM_TREE" >&2
    exit 2
}
[[ -z "$(git -C "$VLLM_DIRECTORY" status --porcelain --untracked-files=all)" ]] || {
    echo "DeepSeek V4 final overlay requires a clean vLLM checkout: $VLLM_DIRECTORY" >&2
    exit 2
}

for checksum_record in "${PINNED_SOURCE_HASHES[@]}"; do
    expected="${checksum_record%%  *}"
    relative_path="${checksum_record#*  }"
    actual="$(sha256sum "$VLLM_DIRECTORY/$relative_path" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "DeepSeek V4 final overlay source mismatch: $relative_path" >&2
        echo "got $actual, expected $expected" >&2
        exit 2
    }
done

for pinned_script in \
    "$EXPECTED_MATERIALIZER_SHA256 materialize-runtime-model-view.py" \
    "$EXPECTED_STARTER_SHA256 start-final-runtime.sh"; do
    expected="${pinned_script%% *}"
    script_name="${pinned_script#* }"
    actual="$(sha256sum "$SCRIPT_DIRECTORY/$script_name" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "DeepSeek V4 final overlay script mismatch: $script_name" >&2
        echo "got $actual, expected $expected" >&2
        exit 2
    }
done

base_image_id="$(docker image inspect "$BASE_IMAGE" --format '{{.Id}}')" || {
    echo "DeepSeek V4 final overlay base image is unavailable locally: $BASE_IMAGE" >&2
    exit 2
}
if [[ "$base_image_id" != "$EXPECTED_BASE_IMAGE_DIGEST" ]]; then
    base_tree="$(
        docker image inspect "$BASE_IMAGE" \
            --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
    )"
    base_contract="$(
        docker image inspect "$BASE_IMAGE" \
            --format '{{index .Config.Labels "org.club3090.runtime.contract-sha256"}}'
    )"
    [[ "$base_tree" == "$EXPECTED_VLLM_TREE" ]] || {
        echo "DeepSeek V4 final overlay base tree mismatch: got $base_tree" >&2
        exit 2
    }
    [[ "$base_contract" == "$EXPECTED_RUNTIME_CONTRACT_SHA256" ]] || {
        echo "DeepSeek V4 final overlay runtime contract mismatch: got $base_contract" >&2
        exit 2
    }
fi

verified_base_tag="club-3090/deepseek-v4-wna16-sm86:verified-base-${BASHPID}"
temporary_context=""
cleanup_final_overlay_build() {
    if [[ -n "$temporary_context" ]]; then
        rm -rf "$temporary_context"
    fi
    docker image rm "$verified_base_tag" >/dev/null 2>&1 || true
}
trap cleanup_final_overlay_build EXIT

docker tag "$base_image_id" "$verified_base_tag"
temporary_context="$(mktemp -d)"
mkdir -p "$temporary_context/vllm"
for checksum_record in "${PINNED_SOURCE_HASHES[@]}"; do
    relative_path="${checksum_record#*  }"
    mkdir -p "$temporary_context/$(dirname -- "$relative_path")"
    cp -- "$VLLM_DIRECTORY/$relative_path" "$temporary_context/$relative_path"
done
cp -- \
    "$SCRIPT_DIRECTORY/materialize-runtime-model-view.py" \
    "$SCRIPT_DIRECTORY/start-final-runtime.sh" \
    "$temporary_context/"

docker build \
    --pull=false \
    --build-arg "VERIFIED_BASE_IMAGE=$verified_base_tag" \
    --file "$DOCKERFILE" \
    --label "org.opencontainers.image.revision=$EXPECTED_VLLM_TREE" \
    --label "org.club3090.runtime.base-digest=$base_image_id" \
    --tag "$IMAGE_TAG" \
    "$temporary_context"

docker image inspect "$IMAGE_TAG" \
    --format 'image={{.Id}} tree={{index .Config.Labels "org.opencontainers.image.revision"}} base={{index .Config.Labels "org.club3090.runtime.base-digest"}}'
