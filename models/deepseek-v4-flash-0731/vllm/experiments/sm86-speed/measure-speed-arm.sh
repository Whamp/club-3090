#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIRECTORY/../../../../.." && pwd)"
readonly REPOSITORY_ROOT
readonly URL="${URL:?set URL to the active experiment endpoint}"
readonly MODEL="${MODEL:?set MODEL to the active experiment model}"
readonly CONTAINER="${CONTAINER:?set CONTAINER to the active experiment container}"
readonly SPEED_ARM="${SPEED_ARM:?set SPEED_ARM to the active experiment arm}"
readonly OUTPUT_DIRECTORY="${1:?usage: measure-speed-arm.sh OUTPUT_DIRECTORY}"
mkdir -p "$OUTPUT_DIRECTORY"

curl --fail --silent --show-error "$URL/v1/models" > "$OUTPUT_DIRECTORY/models.json"
docker inspect "$CONTAINER" > "$OUTPUT_DIRECTORY/container-inspect.json"
docker logs "$CONTAINER" > "$OUTPUT_DIRECTORY/startup.log" 2>&1
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu \
    --format=csv,noheader,nounits > "$OUTPUT_DIRECTORY/gpu-before.csv"

env SPEED_ARM="$SPEED_ARM" OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY" python3 - <<'PY'
import json
import os

with open(f"{os.environ['OUTPUT_DIRECTORY']}/container-inspect.json", encoding="utf-8") as source:
    inspect = json.load(source)[0]
environment = dict(item.split("=", 1) for item in inspect["Config"]["Env"] if "=" in item)
arguments = inspect["Args"]
arm = os.environ["SPEED_ARM"]
expected_environment = {
    "baseline": {},
    "prefill-block2": {"VLLM_SPARSE_DENSE_QUERY_BLOCK": "2"},
    "flashmla-decode": {"VLLM_DSV4_FLASH_MLA_DECODE": "1"},
    "hier-allreduce": {"VLLM_HIER_ALL_REDUCE": "0,1;2,3"},
    "flashmla-hier": {
        "VLLM_DSV4_FLASH_MLA_DECODE": "1",
        "VLLM_HIER_ALL_REDUCE": "0,1;2,3",
    },
    "indexer96": {"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "96"},
    "batched320": {},
}[arm]
for key, expected in expected_environment.items():
    actual = environment.get(key)
    if actual != expected:
        raise SystemExit(f"{arm} dispatch environment mismatch: {key}={actual!r}")
if arm == "batched320":
    index = arguments.index("--max-num-batched-tokens")
    if arguments[index + 1] != "320":
        raise SystemExit("batched320 did not set --max-num-batched-tokens 320")
PY
case "$SPEED_ARM" in
    flashmla-decode)
        grep -F "Using native Ampere FlashMLA sparse decode" \
            "$OUTPUT_DIRECTORY/startup.log" >/dev/null
        ;;
    hier-allreduce)
        grep -F "'HIERARCHICAL'" "$OUTPUT_DIRECTORY/startup.log" >/dev/null
        ;;
    flashmla-hier)
        grep -F "Using native Ampere FlashMLA sparse decode" \
            "$OUTPUT_DIRECTORY/startup.log" >/dev/null
        grep -F "'HIERARCHICAL'" "$OUTPUT_DIRECTORY/startup.log" >/dev/null
        ;;
esac

env URL="$URL" MODEL="$MODEL" OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY" python3 - <<'PY'
import json
import os
import urllib.request

url = os.environ["URL"]
model = os.environ["MODEL"]
out = os.environ["OUTPUT_DIRECTORY"]

def request(name, payload):
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        value = json.load(response)
    with open(f"{out}/{name}.json", "w", encoding="utf-8") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")
    return value

basic = request("canary-basic", {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly: capacity-ok"}],
    "temperature": 0,
    "max_tokens": 32,
})
content = basic["choices"][0]["message"].get("content", "").strip()
if content != "capacity-ok":
    raise SystemExit(f"deterministic canary mismatch: {content!r}")

tools = [{
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}]
messages = [{"role": "user", "content": "Use the add tool exactly once for 17 + 25."}]
first = request("canary-tool-first", {
    "model": model,
    "messages": messages,
    "tools": tools,
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 256,
})
choice = first["choices"][0]
if choice["finish_reason"] != "tool_calls":
    raise SystemExit(f"tool canary finish mismatch: {choice['finish_reason']!r}")
tool_calls = choice["message"].get("tool_calls") or []
if len(tool_calls) != 1 or tool_calls[0]["function"]["name"] != "add":
    raise SystemExit(f"tool canary call mismatch: {tool_calls!r}")
arguments = json.loads(tool_calls[0]["function"]["arguments"])
if arguments != {"a": 17, "b": 25}:
    raise SystemExit(f"tool canary arguments mismatch: {arguments!r}")
messages.extend([
    choice["message"],
    {"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "42"},
])
second = request("canary-tool-second", {
    "model": model,
    "messages": messages,
    "tools": tools,
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 128,
})
second_choice = second["choices"][0]
second_content = second_choice["message"].get("content", "")
if second_choice["finish_reason"] != "stop" or "42" not in second_content:
    raise SystemExit(
        f"post-tool canary mismatch: finish={second_choice['finish_reason']!r} "
        f"content={second_content!r}"
    )
PY

(
    cd "$REPOSITORY_ROOT"
    env URL="$URL" MODEL="$MODEL" CONTAINER="$CONTAINER" \
    WARMUPS=3 RUNS=5 ONLY=code MAX_TOKENS_CODE=512 \
    PREFILL_PROBE=1 PREFILL_DEPTHS=8984 PREFILL_RUNS=3 \
    bash scripts/bench.sh
) | tee "$OUTPUT_DIRECTORY/bench.log"

docker logs "$CONTAINER" > "$OUTPUT_DIRECTORY/post-bench.log" 2>&1
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu \
    --format=csv,noheader,nounits > "$OUTPUT_DIRECTORY/gpu-after.csv"
docker top "$CONTAINER" -eo pid | tail -n +2 > "$OUTPUT_DIRECTORY/worker-pids.txt"
: > "$OUTPUT_DIRECTORY/worker-swap-kib.txt"
while read -r pid; do
    swap_kib="$(awk '/^VmSwap:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)"
    printf '%s %s\n' "$pid" "${swap_kib:-0}" >> "$OUTPUT_DIRECTORY/worker-swap-kib.txt"
    [[ "${swap_kib:-0}" == 0 ]] || {
        echo "Serving process $pid has ${swap_kib} KiB swap" >&2
        exit 1
    }
done < "$OUTPUT_DIRECTORY/worker-pids.txt"
sha256sum "$OUTPUT_DIRECTORY"/* > "$OUTPUT_DIRECTORY/SHA256SUMS"
