#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly GATE_SCRIPT="$SCRIPT_DIRECTORY/hier_all_reduce_sm86_gate.py"
readonly IMAGE="${1:?usage: run-hier-all-reduce-sm86-gate.sh IMAGE REPORT_DIRECTORY}"
readonly REPORT_DIRECTORY="${2:?usage: run-hier-all-reduce-sm86-gate.sh IMAGE REPORT_DIRECTORY}"
readonly EXPECTED_GATE_SHA256="9689ab217010785b88232acd0ce2abf35a26ab9f2152f70353302c6ec4fe0751"

mkdir -p "$REPORT_DIRECTORY"
printf '%s  %s\n' "$EXPECTED_GATE_SHA256" "$GATE_SCRIPT" | \
    sha256sum --check --strict
image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"

cat > "$REPORT_DIRECTORY/identity.json" <<EOF
{
  "image": "$IMAGE",
  "image_id": "$image_id",
  "gate_sha256": "$EXPECTED_GATE_SHA256"
}
EOF

timeout 600 docker run --rm --gpus all --ipc host --network host \
    --entrypoint /opt/venv/bin/python \
    --volume "$GATE_SCRIPT:/gate/hier_all_reduce_sm86_gate.py:ro" \
    --volume "$REPORT_DIRECTORY:/report" \
    "$IMAGE" -m torch.distributed.run --standalone --nproc-per-node=4 \
        /gate/hier_all_reduce_sm86_gate.py \
    | tee "$REPORT_DIRECTORY/oracle-and-timing.json"
sha256sum "$REPORT_DIRECTORY"/* > "$REPORT_DIRECTORY/SHA256SUMS"
