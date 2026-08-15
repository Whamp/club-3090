#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_COMMIT="7f41a5baa5cf57bfbce06458794b4b05737a162a"
readonly SOURCE_ARCHIVE_SHA256="bf21d9c33530a718d1988f24dff1f3600c7715fde6e9f5c73824a4c364831268"
readonly SOURCE_ARCHIVE_URL="https://github.com/AppMana/forks-flash-mla-int/archive/${SOURCE_COMMIT}.tar.gz"
readonly IMAGE="${1:?usage: run-flash-mla-sm86-gate.sh IMAGE REPORT_DIRECTORY}"
readonly REPORT_DIRECTORY="${2:?usage: run-flash-mla-sm86-gate.sh IMAGE REPORT_DIRECTORY}"

mkdir -p "$REPORT_DIRECTORY"
temporary_directory="$(mktemp -d)"
cleanup_flash_mla_gate() {
    rm -rf "$temporary_directory"
}
trap cleanup_flash_mla_gate EXIT

archive="$temporary_directory/flash-mla-source.tar.gz"
curl --fail --location --silent --show-error --retry 3 \
    --output "$archive" "$SOURCE_ARCHIVE_URL"
printf '%s  %s\n' "$SOURCE_ARCHIVE_SHA256" "$archive" | \
    sha256sum --check --strict
tar --extract --gzip --file "$archive" --directory "$temporary_directory"
source_directory="$temporary_directory/forks-flash-mla-int-${SOURCE_COMMIT}"

image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
wheel_sha256="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.club3090.runtime.flash-mla-wheel-sha256"}}')"
cat > "$REPORT_DIRECTORY/identity.json" <<EOF
{
  "image": "$IMAGE",
  "image_id": "$image_id",
  "source_commit": "$SOURCE_COMMIT",
  "source_archive_sha256": "$SOURCE_ARCHIVE_SHA256",
  "wheel_sha256": "$wheel_sha256"
}
EOF

# The single-quoted script is interpreted inside the container.
# shellcheck disable=SC2016
timeout 900 docker run --rm --gpus 'device=0' --ipc host \
    --entrypoint /bin/bash \
    --volume "$source_directory:/flash-mla-source:ro" \
    --volume "$REPORT_DIRECTORY:/report" \
    "$IMAGE" -lc '
        set -euo pipefail
        test "$(/opt/venv/bin/python - <<'"'"'PY'"'"'
import torch
print(".".join(map(str, torch.cuda.get_device_capability(0))))
PY
)" = "8.6"
        extension="$(find /opt/venv -name '"'"'flash_mla_cuda*.so'"'"' -type f -print -quit)"
        test -n "$extension"
        cuobjdump --list-elf "$extension" | tee /report/cuobjdump-list-elf.txt
        grep -F '"'"'sm_86'"'"' /report/cuobjdump-list-elf.txt
        /opt/venv/bin/python -m pytest --version | grep -F "pytest 9.1.1"
        cd /flash-mla-source
        /opt/venv/bin/python -m pytest -q -x \
            tests/test_sparse_mla_decode_sm86.py -k decode \
            | tee /report/decode-oracle.log
    '
sha256sum "$REPORT_DIRECTORY"/* > "$REPORT_DIRECTORY/SHA256SUMS"
