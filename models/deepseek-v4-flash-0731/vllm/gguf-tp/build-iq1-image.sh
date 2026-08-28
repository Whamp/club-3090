#!/usr/bin/env bash
# Build the checksum-pinned native Unsloth IQ1 GGUF-TP overlay.
#
# Required inputs:
#   VLLM_SOURCE_DIR          Whamp/vLLM checkout at IQ1-MANIFEST.json commit.
#   GGUF_TP_IQ1_EXTENSION    SM86 stable extension named in the manifest.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MANIFEST="${IQ1_MANIFEST:-IQ1-MANIFEST.json}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-}"
EXTENSION="${GGUF_TP_IQ1_EXTENSION:-}"

[[ -f "$MANIFEST" ]] || { echo "FATAL: missing IQ1-MANIFEST.json" >&2; exit 1; }
[[ -d "$VLLM_SOURCE_DIR" ]] || {
  echo "FATAL: set VLLM_SOURCE_DIR to the pinned Whamp/vLLM source tree" >&2
  exit 1
}
[[ -f "$EXTENSION" ]] || {
  echo "FATAL: set GGUF_TP_IQ1_EXTENSION to the compiled stable extension" >&2
  exit 1
}

CONTEXT="$(mktemp -d)"
BASE_TAG="club-3090/iq1-build-base:verified"
cleanup() {
  docker image rm "$BASE_TAG" >/dev/null 2>&1 || true
  rm -rf "$CONTEXT"
}
trap cleanup EXIT

readarray -t VALUES < <(python3 - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["runtime_base"]["digest"])
print(manifest["vllm"]["commit"])
print(manifest["vllm"]["tree"])
print(manifest["stable_extension"]["sha256"])
print(manifest["image"]["tag"])
print(manifest["image"]["digest"])
PY
)
BASE_DIGEST="${VALUES[0]}"
VLLM_COMMIT="${VALUES[1]}"
VLLM_TREE="${VALUES[2]}"
EXT_SHA="${VALUES[3]}"
IMAGE_TAG="${VALUES[4]}"
EXPECTED_IMAGE_DIGEST="${VALUES[5]}"

echo "$EXT_SHA  $EXTENSION" | sha256sum -c - >/dev/null || {
  echo "FATAL: stable extension hash mismatch" >&2
  exit 1
}

if git -C "$VLLM_SOURCE_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  ACTUAL_COMMIT="$(git -C "$VLLM_SOURCE_DIR" rev-parse HEAD)"
  ACTUAL_TREE="$(git -C "$VLLM_SOURCE_DIR" rev-parse 'HEAD^{tree}')"
  [[ "$ACTUAL_COMMIT" == "$VLLM_COMMIT" ]] || {
    echo "FATAL: vLLM commit $ACTUAL_COMMIT != $VLLM_COMMIT" >&2
    exit 1
  }
  [[ "$ACTUAL_TREE" == "$VLLM_TREE" ]] || {
    echo "FATAL: vLLM tree $ACTUAL_TREE != $VLLM_TREE" >&2
    exit 1
  }
fi

mkdir -p "$CONTEXT/root"
python3 - "$MANIFEST" "$VLLM_SOURCE_DIR" "$CONTEXT/root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
source = Path(sys.argv[2])
destination = Path(sys.argv[3])
for entry in manifest["vllm"]["runtime_files"]:
    relative = Path(entry["path"])
    source_path = source / relative
    if not source_path.is_file():
        raise SystemExit(f"FATAL: missing vLLM runtime file {relative}")
    payload = source_path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != entry["sha256"]:
        raise SystemExit(
            f"FATAL: vLLM runtime hash {relative}: {actual} != {entry['sha256']}"
        )
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
print(f"OK: {len(manifest['vllm']['runtime_files'])} runtime_files verified")
PY

cp "$EXTENSION" "$CONTEXT/_C_stable_libtorch.abi3.so"
cp Dockerfile.iq1 "$CONTEXT/Dockerfile.iq1"

docker image inspect "$BASE_DIGEST" >/dev/null 2>&1 || {
  echo "FATAL: runtime base $BASE_DIGEST is not present" >&2
  exit 1
}
docker tag "$BASE_DIGEST" "$BASE_TAG"

docker build \
  --provenance=false \
  --build-arg "BASE_IMAGE=$BASE_TAG" \
  --file "$CONTEXT/Dockerfile.iq1" \
  --tag "$IMAGE_TAG" \
  "$CONTEXT"

IMAGE_ID="$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}')"
if [[ "$EXPECTED_IMAGE_DIGEST" != "TBD" && "$IMAGE_ID" != "$EXPECTED_IMAGE_DIGEST" ]]; then
  echo "FATAL: built image $IMAGE_ID != $EXPECTED_IMAGE_DIGEST" >&2
  exit 1
fi
printf 'OK: built %s\nimage=%s\n' "$IMAGE_TAG" "$IMAGE_ID"
