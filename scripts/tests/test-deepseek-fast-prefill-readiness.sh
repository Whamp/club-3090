#!/usr/bin/env bash
set -euo pipefail

# Force Python's UTF-8 mode for the fake HTTP server on non-UTF-8 locales.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVE_SCRIPT="${ROOT_DIR}/models/deepseek-v4-flash-0731/llama-cpp/recipes/serve-fast-prefill.sh"
FORWARD_SCRIPT="${ROOT_DIR}/models/deepseek-v4-flash-0731/llama-cpp/recipes/tcp-forward.pl"
TMP="$(mktemp -d)"
ENTRY_PID=""

cleanup() {
  if [[ -n "${ENTRY_PID}" ]]; then
    kill "${ENTRY_PID}" 2>/dev/null || true
    wait "${ENTRY_PID}" 2>/dev/null || true
  fi
  rm -rf "${TMP}"
}
trap cleanup EXIT

fail() {
  printf 'test-deepseek-fast-prefill-readiness: FAIL: %s\n' "$*" >&2
  [[ -f "${TMP}/serve.log" ]] && tail -n 80 "${TMP}/serve.log" >&2
  exit 1
}

[[ -x "${SERVE_SCRIPT}" ]] || fail "missing executable serve-fast-prefill.sh"
[[ -f "${FORWARD_SCRIPT}" ]] || fail "missing tcp-forward.pl"

read -r INTERNAL_PORT PUBLIC_PORT < <(python3 - <<'PY'
import socket
ports = []
for _ in range(2):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    ports.append(sock.getsockname()[1])
    sock.close()
print(*ports)
PY
)

cat > "${TMP}/fake-llama-server.py" <<'PY'
#!/usr/bin/env python3
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok"})
        elif self.path == "/v1/models":
            self.send_json({"data": [{"id": "deepseek-v4-test"}]})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/tokenize":
            self.send_json({"tokens": list(range(1, 13))})
            return
        if self.path == "/completion":
            prompt = payload.get("prompt") or []
            with open(os.environ["FAKE_CAPTURE"], "w", encoding="utf-8") as handle:
                handle.write(str(len(prompt)))
            time.sleep(1.0)
            if len(prompt) != 8:
                self.send_json({"error": f"expected 8 warm-up tokens, got {len(prompt)}"}, 400)
                return
            self.send_json({"timings": {"prompt_n": len(prompt), "prompt_per_second": 1234.5}})
            return
        self.send_json({"error": "not found"}, 404)

ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
PY
chmod +x "${TMP}/fake-llama-server.py"

FAKE_CAPTURE="${TMP}/prompt-count" \
SERVER_BIN="${TMP}/fake-llama-server.py" \
TCP_FORWARD_SCRIPT="${FORWARD_SCRIPT}" \
FAST_PREFILL_INTERNAL_PORT="${INTERNAL_PORT}" \
FAST_PREFILL_PUBLIC_PORT="${PUBLIC_PORT}" \
FAST_PREFILL_CONTEXT_SIZE=520 \
FAST_PREFILL_WARMUP_MARGIN=512 \
FAST_PREFILL_READY_FILE="${TMP}/ready" \
  "${SERVE_SCRIPT}" --model /unused.gguf >"${TMP}/serve.log" 2>&1 &
ENTRY_PID=$!

for _ in $(seq 1 100); do
  curl -fsS "http://127.0.0.1:${INTERNAL_PORT}/health" >/dev/null 2>&1 && break
  kill -0 "${ENTRY_PID}" 2>/dev/null || fail "entrypoint exited before the internal server became healthy"
  sleep 0.05
done
curl -fsS "http://127.0.0.1:${INTERNAL_PORT}/health" >/dev/null 2>&1 \
  || fail "internal server never became healthy"

if curl -fsS --max-time 0.2 "http://127.0.0.1:${PUBLIC_PORT}/v1/models" >/dev/null 2>&1; then
  fail "public endpoint became reachable before warm-up completed"
fi

for _ in $(seq 1 100); do
  [[ -f "${TMP}/ready" ]] && curl -fsS "http://127.0.0.1:${PUBLIC_PORT}/v1/models" >/dev/null 2>&1 && break
  kill -0 "${ENTRY_PID}" 2>/dev/null || fail "entrypoint exited before publishing the warmed endpoint"
  sleep 0.05
done

[[ -f "${TMP}/ready" ]] || fail "ready marker was not written"
[[ "$(cat "${TMP}/prompt-count")" == "8" ]] || fail "warm-up did not use context_size - margin tokens"
curl -fsS "http://127.0.0.1:${PUBLIC_PORT}/v1/models" | grep -q 'deepseek-v4-test' \
  || fail "TCP forwarder did not preserve the OpenAI response"
grep -q 'Full-context warm-up complete: 8 tokens at 1234.50 tok/s' "${TMP}/serve.log" \
  || fail "warm-up completion was not logged"
grep -q 'DeepSeek V4 fast-prefill endpoint is warm and ready' "${TMP}/serve.log" \
  || fail "public readiness was not logged"

printf 'test-deepseek-fast-prefill-readiness: PASS\n'
