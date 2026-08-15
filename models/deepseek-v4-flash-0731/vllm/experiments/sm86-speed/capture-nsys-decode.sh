#!/usr/bin/env bash
set -euo pipefail

readonly URL="${URL:?set URL to the active experiment endpoint}"
readonly MODEL="${MODEL:?set MODEL to the active experiment model}"
readonly OUTPUT_DIRECTORY="${1:?usage: capture-nsys-decode.sh OUTPUT_DIRECTORY}"
mkdir -p "$OUTPUT_DIRECTORY"

request_file="$OUTPUT_DIRECTORY/request.json"
python - "$MODEL" > "$request_file" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "messages": [{
        "role": "user",
        "content": (
            "Implement an iterative quicksort in Python and explain the loop "
            "invariants briefly. This is a deterministic profiling request."
        ),
    }],
    "temperature": 0,
    "max_tokens": 256,
    "stream": False,
}))
PY

# Warm graph and JIT state before the capture range.
curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    --data-binary "@$request_file" \
    "$URL/v1/chat/completions" > "$OUTPUT_DIRECTORY/warmup-response.json"

curl --fail --silent --show-error --request POST \
    "$URL/start_profile" > "$OUTPUT_DIRECTORY/start-profile.json"
curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    --data-binary "@$request_file" \
    "$URL/v1/chat/completions" > "$OUTPUT_DIRECTORY/profiled-response.json"
curl --fail --silent --show-error --request POST \
    "$URL/stop_profile" > "$OUTPUT_DIRECTORY/stop-profile.json"

python - "$OUTPUT_DIRECTORY/profiled-response.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as response_file:
    response = json.load(response_file)
choice = response["choices"][0]
if choice["finish_reason"] not in {"stop", "length"}:
    raise SystemExit(f"unexpected finish reason: {choice['finish_reason']!r}")
if not choice["message"].get("content"):
    raise SystemExit("profiled response had no content")
PY
