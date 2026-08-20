#!/usr/bin/env bash
# ===========================================================================
# Fail-closed reproducible builder for the GGUF-TP engine image.
#
# Produces club-3090/deepseek-v4-gguf-tp:3ec20ceb, byte-identical in content
# to the promoted image (digest sha256:f91e8283… — note: Docker image IDs are
# not reproducible across builder versions, so the build verifies content,
# not the final ID; MANIFEST.json pins every content hash instead).
#
# Requirements:
#   - a clone of Whamp/vLLM checked out at 3ec20cebe (or the 21 files already
#     extracted under root/ — see below)
#   - the compiled stable extension _C_stable_libtorch.abi3.so (sha256 pinned;
#     normally carried from the rig that built it — see README "Rebuilding the
#     native extension")
#   - base image club-3090/gguf-tp-base:b7766cfe present
#
# Usage:  VLLM_CLONE=/path/to/vllm@3ec20cebe ./build-image.sh
# (With VLLM_CLONE set, root/ is re-extracted and hash-verified from the clone.
#  Without it, an existing root/ is hash-verified in place. Fails otherwise.)
# ===========================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MANIFEST=MANIFEST.json
CONTEXT="$(mktemp -d)"
trap 'rm -rf "$CONTEXT"' EXIT

python3 - <<'PY' "$MANIFEST"
import json, sys
m = json.load(open(sys.argv[1]))
print(m["images"]["base"]["digest"])
print(m["stable_extension"]["sha256"])
print(m["images"]["repack"]["vllm_revision"])
PY

BASE_DIGEST="$(python3 -c 'import json;print(json.load(open("MANIFEST.json"))["images"]["base"]["digest"])')"
EXT_SHA="$(python3 -c 'import json;print(json.load(open("MANIFEST.json"))["stable_extension"]["sha256"])')"
REV="$(python3 -c 'import json;print(json.load(open("MANIFEST.json"))["images"]["repack"]["vllm_revision"])')"

# 1. Base image must exist with the pinned digest.
docker image inspect "$BASE_DIGEST" >/dev/null 2>&1 || {
  echo "FATAL: base image $BASE_DIGEST not present" >&2; exit 1; }
echo "OK: base image $BASE_DIGEST present"

# 2. Assemble root/ (21 files) and hash-verify each against MANIFEST.
if [[ -n "${VLLM_CLONE:-}" ]]; then
  echo "Extracting 21 engine files from $VLLM_CLONE @ $REV ..."
  pushd "$VLLM_CLONE" >/dev/null
  git cat-file -e "$REV" 2>/dev/null || { echo "FATAL: commit $REV not in clone" >&2; exit 1; }
  mkdir -p "$CONTEXT/root"
  python3 - "$CONTEXT/root" <<'PY'
import json, os, subprocess, sys
root = sys.argv[1]
files = json.load(open("MANIFEST.json"))["overlaid_sources"]["files"]
for f in files:
    dst = os.path.join(root, f["path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as out:
        subprocess.run(["git", "show", f"3ec20cebe:{f['path']}"], check=True, stdout=out)
print(f"extracted {len(files)} files")
PY
  popd >/dev/null
elif [[ -d root ]]; then
  echo "Using existing root/ (will hash-verify)..."
  cp -r root "$CONTEXT/root"
else
  echo "FATAL: set VLLM_CLONE or provide root/ with the 21 engine files" >&2; exit 1
fi

python3 - "$CONTEXT/root" <<'PY'
import hashlib, json, os, sys
root = sys.argv[1]
fail = 0
for f in json.load(open("MANIFEST.json"))["overlaid_sources"]["files"]:
    p = os.path.join(root, f["path"])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != f["sha256"]:
        print(f"FAIL hash {f['path']}: {h} != {f['sha256']}"); fail += 1
if fail:
    sys.exit(f"FATAL: {fail} overlaid files failed hash verification")
print(f"OK: {len(json.load(open('MANIFEST.json'))['overlaid_sources']['files'])} overlaid files hash-verified")
PY

# 3. Stable extension: require the artifact, verify hash.
EXT="${GGUF_TP_EXTENSION:-/home/will/inference/runtime/gguf-tp-m5-image/_C_stable_libtorch.abi3.so}"
[[ -f "$EXT" ]] || { echo "FATAL: extension not at $EXT (set GGUF_TP_EXTENSION)" >&2; exit 1; }
echo "$EXT_SHA  $EXT" | sha256sum -c - >/dev/null || { echo "FATAL: extension hash mismatch" >&2; exit 1; }
echo "OK: extension hash verified"

cp "$EXT" "$CONTEXT/_C_stable_libtorch.abi3.so"

# 4. Build layer 1 (engine sources + extension).
docker build -f Dockerfile.gguf-tp -t "club-3090/deepseek-v4-gguf-tp:$REV" "$CONTEXT"
LAYER1="$(docker inspect --format '{{index .RepoDigests 0}}' "club-3090/deepseek-v4-gguf-tp:$REV" 2>/dev/null || echo "(no repo digest)")"
echo "Layer 1 built: club-3090/deepseek-v4-gguf-tp:$REV ($LAYER1)"

# 5. Build layer 2 (repack) — context only needs the one file.
mkdir -p "$CONTEXT/repack/root/vllm/model_executor/layers/quantization/gguf_dsv4"
cp "$CONTEXT/root/vllm/model_executor/layers/quantization/gguf_dsv4/q8_0_marlin.py" \
   "$CONTEXT/repack/root/vllm/model_executor/layers/quantization/gguf_dsv4/q8_0_marlin.py"
docker build -f Dockerfile.repack -t "club-3090/deepseek-v4-gguf-tp:3ec20ceb" "$CONTEXT/repack"
echo "=== done: club-3090/deepseek-v4-gguf-tp:3ec20ceb ==="
docker inspect "club-3090/deepseek-v4-gguf-tp:3ec20ceb" --format 'image={{.Id}} digest={{index .RepoDigests 0}}'
