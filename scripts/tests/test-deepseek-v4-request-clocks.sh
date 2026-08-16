#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTROLLER="$ROOT/scripts/deepseek-v4-request-clocks.py"
TMP="$(mktemp -d)"
SERVER_PID=""
CONTROLLER_PID=""

cleanup() {
    if [[ -n "$CONTROLLER_PID" ]] && kill -0 "$CONTROLLER_PID" 2>/dev/null; then
        kill "$CONTROLLER_PID" 2>/dev/null || true
        wait "$CONTROLLER_PID" 2>/dev/null || true
    fi
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$TMP"
}
trap cleanup EXIT

cat > "$TMP/fake-nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$CLOCK_COMMAND_LOG"
EOF
chmod +x "$TMP/fake-nvidia-smi"
printf 'idle' > "$TMP/slot-state"

cat > "$TMP/slot-server.py" <<'PY'
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

state_path = Path(sys.argv[1])
port_path = Path(sys.argv[2])

class SlotHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/slots":
            self.send_error(404)
            return
        busy = state_path.read_text(encoding="utf-8").strip() == "busy"
        payload = json.dumps([{"id": 0, "is_processing": busy}]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass

server = ThreadingHTTPServer(("127.0.0.1", 0), SlotHandler)
port_path.write_text(str(server.server_port), encoding="utf-8")
server.serve_forever()
PY
python3 "$TMP/slot-server.py" "$TMP/slot-state" "$TMP/port" &
SERVER_PID=$!
for _ in $(seq 1 100); do
    [[ -s "$TMP/port" ]] && break
    sleep 0.01
done
[[ -s "$TMP/port" ]]

export CLOCK_COMMAND_LOG="$TMP/clock-commands"

start_controller() {
    python3 "$CONTROLLER" \
        --slots-url "http://127.0.0.1:$(cat "$TMP/port")/slots" \
        --nvidia-smi "$TMP/fake-nvidia-smi" \
        --poll-interval-seconds 0.02 \
        --idle-reset-seconds 0.10 \
        --endpoint-failure-reset-seconds 0.10 \
        > "$TMP/controller.log" 2>&1 &
    CONTROLLER_PID=$!
}

start_controller

wait_for_line() {
    local expected="$1"
    for _ in $(seq 1 200); do
        grep -Fxq -- "$expected" "$CLOCK_COMMAND_LOG" 2>/dev/null && return 0
        sleep 0.01
    done
    echo "missing clock command: $expected" >&2
    cat "$TMP/controller.log" >&2 || true
    cat "$CLOCK_COMMAND_LOG" >&2 || true
    return 1
}

wait_for_count() {
    local expected="$1"
    local count="$2"
    for _ in $(seq 1 200); do
        [[ "$(grep -Fxc -- "$expected" "$CLOCK_COMMAND_LOG" 2>/dev/null || true)" == "$count" ]] && return 0
        sleep 0.01
    done
    echo "clock command count mismatch: $expected expected=$count" >&2
    cat "$CLOCK_COMMAND_LOG" >&2 || true
    return 1
}

wait_for_line "-rgc"
printf 'busy' > "$TMP/slot-state"
wait_for_line "-lgc 1995"
printf 'idle' > "$TMP/slot-state"
wait_for_count "-rgc" 2

kill "$CONTROLLER_PID"
wait "$CONTROLLER_PID" || true
CONTROLLER_PID=""
wait_for_count "-rgc" 2

grep -Fq "DeepSeek V4 request clocks: locked SM clocks at 1995 MHz" "$TMP/controller.log"
grep -Fq "DeepSeek V4 request clocks: reset SM clocks after idle" "$TMP/controller.log"

: > "$CLOCK_COMMAND_LOG"
printf 'busy' > "$TMP/slot-state"
start_controller
wait_for_line "-lgc 1995"
kill "$CONTROLLER_PID"
wait "$CONTROLLER_PID" || true
CONTROLLER_PID=""
wait_for_count "-rgc" 2
grep -Fq "DeepSeek V4 request clocks: reset SM clocks during shutdown" "$TMP/controller.log"

: > "$CLOCK_COMMAND_LOG"
start_controller
wait_for_line "-lgc 1995"
kill "$SERVER_PID"
wait "$SERVER_PID" || true
SERVER_PID=""
wait_for_count "-rgc" 2
kill "$CONTROLLER_PID"
wait "$CONTROLLER_PID" || true
CONTROLLER_PID=""
grep -Fq "DeepSeek V4 request clocks: reset SM clocks after endpoint failure" "$TMP/controller.log"

UNIT="$ROOT/scripts/systemd/deepseek-v4-request-clocks.service"
grep -Fq "EnvironmentFile=-/etc/club-3090/deepseek-v4-request-clocks.env" "$UNIT"
grep -Fq "ExecStart=/usr/local/libexec/club-3090/deepseek-v4-request-clocks --slots-url \${DEEPSEEK_V4_SLOTS_URL} --clock-mhz \${DEEPSEEK_V4_CLOCK_MHZ}" "$UNIT"
grep -Fq "Restart=on-failure" "$UNIT"

echo "PASS: DeepSeek V4 request-aware GPU clocks lock on busy and reset on idle"
