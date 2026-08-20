#!/usr/bin/env bash
# Build the checksum-pinned SM86 MXFP4 sparse-indexer overlay.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MANIFEST="${FP4_INDEXER_MANIFEST:-FP4-INDEXER-MANIFEST.json}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-}"
[[ -f "$MANIFEST" ]] || { echo "FATAL: missing manifest $MANIFEST" >&2; exit 1; }
[[ -d "$VLLM_SOURCE_DIR" ]] || { echo "FATAL: set VLLM_SOURCE_DIR" >&2; exit 1; }
CONTEXT="$(mktemp -d)"
BASE_TAG="club-3090/fp4-indexer-build-base:verified"
cleanup() { docker image rm "$BASE_TAG" >/dev/null 2>&1 || true; rm -rf "$CONTEXT"; }
trap cleanup EXIT
readarray -t VALUES < <(python3 - "$MANIFEST" <<'PY'
import json, sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
print(m["runtime_base"]["digest"])
print(m["vllm"]["commit"])
print(m["vllm"]["tree"])
print(m["vllm"]["runtime_source_sha256"])
print(m["image"]["tag"])
print(m["image"]["digest"])
PY
)
BASE_DIGEST="${VALUES[0]}"; VLLM_COMMIT="${VALUES[1]}"; VLLM_TREE="${VALUES[2]}"
SOURCE_SHA="${VALUES[3]}"; IMAGE_TAG="${VALUES[4]}"; EXPECTED_IMAGE_DIGEST="${VALUES[5]}"
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
import hashlib,json,sys
from pathlib import Path
m=json.load(open(sys.argv[1],encoding="utf-8")); src=Path(sys.argv[2]); dst=Path(sys.argv[3])
for entry in m["vllm"]["runtime_files"]:
 p=Path(entry["path"]); payload=(src/p).read_bytes(); actual=hashlib.sha256(payload).hexdigest()
 if actual != entry["sha256"]: raise SystemExit(f"FATAL: runtime hash {p}: {actual} != {entry['sha256']}")
 target=dst/p; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(payload)
print(f"OK: {len(m['vllm']['runtime_files'])} runtime files verified")
PY
cp Dockerfile.fp4-indexer "$CONTEXT/Dockerfile"
docker image inspect "$BASE_DIGEST" >/dev/null 2>&1 || { echo "FATAL: base image absent" >&2; exit 1; }
docker tag "$BASE_DIGEST" "$BASE_TAG"
docker build --provenance=false --build-arg "BASE_IMAGE=$BASE_TAG" --build-arg "VLLM_COMMIT=$VLLM_COMMIT" --build-arg "VLLM_TREE=$VLLM_TREE" --build-arg "SOURCE_SHA=$SOURCE_SHA" --tag "$IMAGE_TAG" "$CONTEXT"
IMAGE_ID="$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}')"
if [[ "$EXPECTED_IMAGE_DIGEST" != "TBD" && "$IMAGE_ID" != "$EXPECTED_IMAGE_DIGEST" ]]; then echo "FATAL: image $IMAGE_ID != $EXPECTED_IMAGE_DIGEST" >&2; exit 1; fi
printf 'OK: built %s\nimage=%s\n' "$IMAGE_TAG" "$IMAGE_ID"
