#!/usr/bin/env bash
# Keep DeepSeek V4 private until its mandatory full-context graph warm-up finishes.

set -euo pipefail

SERVER_BIN="${SERVER_BIN:-/app/llama-server}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TCP_FORWARD_SCRIPT="${TCP_FORWARD_SCRIPT:-${SCRIPT_DIR}/tcp-forward.pl}"
FAST_PREFILL_INTERNAL_PORT="${FAST_PREFILL_INTERNAL_PORT:-8081}"
FAST_PREFILL_PUBLIC_PORT="${FAST_PREFILL_PUBLIC_PORT:-8080}"
FAST_PREFILL_CONTEXT_SIZE="${FAST_PREFILL_CONTEXT_SIZE:-430080}"
FAST_PREFILL_WARMUP_MARGIN="${FAST_PREFILL_WARMUP_MARGIN:-512}"
FAST_PREFILL_READY_FILE="${FAST_PREFILL_READY_FILE:-/tmp/deepseek-v4-fast-prefill-ready}"
FAST_PREFILL_BOOT_TIMEOUT="${FAST_PREFILL_BOOT_TIMEOUT:-600}"
FAST_PREFILL_WARMUP_TIMEOUT="${FAST_PREFILL_WARMUP_TIMEOUT:-3600}"

SERVER_PID=""
FORWARD_PID=""
WORK_DIR=""

cleanup() {
  local pid
  trap - EXIT TERM INT
  for pid in "${FORWARD_PID}" "${SERVER_PID}"; do
    [[ -n "${pid}" ]] || continue
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${FORWARD_PID}" "${SERVER_PID}"; do
    [[ -n "${pid}" ]] || continue
    wait "${pid}" 2>/dev/null || true
  done
  [[ -z "${WORK_DIR}" ]] || rm -rf "${WORK_DIR}"
  rm -f "${FAST_PREFILL_READY_FILE}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

require_positive_integer() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    printf 'DeepSeek V4 fast-prefill configuration error: %s must be a positive integer, got %q.\n' \
      "${name}" "${value}" >&2
    exit 2
  fi
}

require_positive_integer FAST_PREFILL_CONTEXT_SIZE "${FAST_PREFILL_CONTEXT_SIZE}"
require_positive_integer FAST_PREFILL_WARMUP_MARGIN "${FAST_PREFILL_WARMUP_MARGIN}"
require_positive_integer FAST_PREFILL_BOOT_TIMEOUT "${FAST_PREFILL_BOOT_TIMEOUT}"
require_positive_integer FAST_PREFILL_WARMUP_TIMEOUT "${FAST_PREFILL_WARMUP_TIMEOUT}"

WARMUP_TARGET=$((FAST_PREFILL_CONTEXT_SIZE - FAST_PREFILL_WARMUP_MARGIN))
if (( WARMUP_TARGET <= 0 )); then
  printf 'DeepSeek V4 fast-prefill configuration error: context %d must exceed warm-up margin %d.\n' \
    "${FAST_PREFILL_CONTEXT_SIZE}" "${FAST_PREFILL_WARMUP_MARGIN}" >&2
  exit 2
fi
if [[ ! -x "${SERVER_BIN}" ]]; then
  printf 'DeepSeek V4 fast-prefill startup error: server binary is not executable: %s\n' "${SERVER_BIN}" >&2
  exit 2
fi
if [[ ! -f "${TCP_FORWARD_SCRIPT}" ]]; then
  printf 'DeepSeek V4 fast-prefill startup error: TCP forwarder is missing: %s\n' "${TCP_FORWARD_SCRIPT}" >&2
  exit 2
fi

rm -f "${FAST_PREFILL_READY_FILE}"
WORK_DIR="$(mktemp -d)"
INTERNAL_URL="http://127.0.0.1:${FAST_PREFILL_INTERNAL_PORT}"
PUBLIC_URL="http://127.0.0.1:${FAST_PREFILL_PUBLIC_PORT}"

printf 'DeepSeek V4 fast-prefill: starting the internal server on %s; public port %s remains closed during warm-up.\n' \
  "${INTERNAL_URL}" "${FAST_PREFILL_PUBLIC_PORT}"
"${SERVER_BIN}" --host 127.0.0.1 --port "${FAST_PREFILL_INTERNAL_PORT}" "$@" &
SERVER_PID=$!

boot_deadline=$((SECONDS + FAST_PREFILL_BOOT_TIMEOUT))
until curl -fsS --max-time 3 "${INTERNAL_URL}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" || true
    printf 'DeepSeek V4 fast-prefill startup error: internal server exited before health became ready.\n' >&2
    exit 1
  fi
  if (( SECONDS >= boot_deadline )); then
    printf 'DeepSeek V4 fast-prefill startup error: internal health did not become ready within %d seconds.\n' \
      "${FAST_PREFILL_BOOT_TIMEOUT}" >&2
    exit 1
  fi
  sleep 2
done

printf 'DeepSeek V4 fast-prefill: warming %d tokens; the public endpoint remains closed.\n' "${WARMUP_TARGET}"
{
  printf '{"content":"'
  perl -e '
    my ($context_size) = @ARGV;
    my $paragraph = "The quick brown fox jumps over the lazy dog while seventeen engineers benchmark a mixture-of-experts transformer on four consumer graphics cards at increasing context depths. ";
    my $copies = int($context_size * 7 / length($paragraph)) + 1;
    print $paragraph x $copies;
  ' "${FAST_PREFILL_CONTEXT_SIZE}"
  printf '"}'
} > "${WORK_DIR}/tokenize-request.json"

curl -fsS \
  --connect-timeout 5 \
  --max-time 120 \
  -H 'Content-Type: application/json' \
  --data-binary "@${WORK_DIR}/tokenize-request.json" \
  "${INTERNAL_URL}/tokenize" > "${WORK_DIR}/tokenize-response.json"

{
  printf '{"prompt":['
  perl -0777 -e '
    my ($target) = @ARGV;
    my $body = do { local $/; <STDIN> };
    $body =~ /"tokens"\s*:\s*\[(.*?)\]/s
      or die "DeepSeek V4 fast-prefill warm-up error: /tokenize response has no tokens array\n";
    my @tokens = split /\s*,\s*/, $1;
    die "DeepSeek V4 fast-prefill warm-up error: tokenizer returned " . scalar(@tokens) . " tokens, need $target\n"
      if @tokens < $target;
    splice @tokens, $target;
    print join q{,}, @tokens;
  ' "${WARMUP_TARGET}" < "${WORK_DIR}/tokenize-response.json"
  printf '],"n_predict":1,"temperature":0,"cache_prompt":false,"stream":false}'
} > "${WORK_DIR}/completion-request.json"

curl -fsS \
  --connect-timeout 5 \
  --max-time "${FAST_PREFILL_WARMUP_TIMEOUT}" \
  -H 'Content-Type: application/json' \
  --data-binary "@${WORK_DIR}/completion-request.json" \
  "${INTERNAL_URL}/completion" > "${WORK_DIR}/completion-response.json"

perl -0777 -e '
  my $body = do { local $/; <STDIN> };
  $body =~ /"prompt_n"\s*:\s*(\d+)/
    or die "DeepSeek V4 fast-prefill warm-up error: completion response has no prompt_n timing\n";
  my $prompt_n = $1;
  $body =~ /"prompt_per_second"\s*:\s*([0-9]+(?:\.[0-9]+)?)/
    or die "DeepSeek V4 fast-prefill warm-up error: completion response has no prompt_per_second timing\n";
  printf "Full-context warm-up complete: %d tokens at %.2f tok/s.\n", $prompt_n, $1;
' < "${WORK_DIR}/completion-response.json"

perl "${TCP_FORWARD_SCRIPT}" 0.0.0.0 "${FAST_PREFILL_PUBLIC_PORT}" 127.0.0.1 "${FAST_PREFILL_INTERNAL_PORT}" &
FORWARD_PID=$!
forward_deadline=$((SECONDS + 30))
until curl -fsS --max-time 3 "${PUBLIC_URL}/health" >/dev/null 2>&1; do
  if ! kill -0 "${FORWARD_PID}" 2>/dev/null; then
    wait "${FORWARD_PID}" || true
    printf 'DeepSeek V4 fast-prefill startup error: TCP forwarder exited before public health became ready.\n' >&2
    exit 1
  fi
  if (( SECONDS >= forward_deadline )); then
    printf 'DeepSeek V4 fast-prefill startup error: public endpoint did not become ready within 30 seconds.\n' >&2
    exit 1
  fi
  sleep 1
done

touch "${FAST_PREFILL_READY_FILE}"
printf 'DeepSeek V4 fast-prefill endpoint is warm and ready on 0.0.0.0:%s.\n' "${FAST_PREFILL_PUBLIC_PORT}"

set +e
wait -n "${SERVER_PID}" "${FORWARD_PID}"
status=$?
set -e
printf 'DeepSeek V4 fast-prefill runtime error: a supervised process exited with status %d.\n' "${status}" >&2
exit "${status}"
