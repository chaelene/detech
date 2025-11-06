#!/usr/bin/env bash

# Demo helper for recording the DETECH end-to-end flow.
#
# The script performs the following steps:
#   1. Ensures the backend health endpoint is reachable.
#   2. Starts a mock stream session against the backend (using the CPU jetson simulator).
#   3. Records the browser dashboard using either ffmpeg or OBS (via obs-cli).
#
# Environment variables / flags:
#   BACKEND_URL      – default http://localhost:8000
#   FRONTEND_URL     – default http://localhost:3000
#   MODE             – ffmpeg (default) or obs
#   OUTPUT           – output file path (default ./docs/demo-<timestamp>.mp4)
#   DURATION         – capture duration in seconds (default 45)
#   SESSION_ID       – optional session id override
#   WALLET_PUBKEY    – wallet identifier for the mock stream
#   FFMPEG_INPUT_ARGS – required when MODE=ffmpeg (e.g. "-f x11grab -video_size 1920x1080 -i :0.0+0,0")
#   OBS_PORT / OBS_PASSWORD – forwarded to obs-cli when MODE=obs
#
# Usage examples:
#   MODE=ffmpeg FFMPEG_INPUT_ARGS="-f x11grab -video_size 1920x1080 -i :0.0" ./docs/demo.sh
#   MODE=obs OBS_PASSWORD=secret ./docs/demo.sh --duration 90 --output demo.mp4

set -euo pipefail

MODE=${MODE:-ffmpeg}
OUTPUT=${OUTPUT:-"$(pwd)/docs/demo-$(date +%Y%m%d-%H%M%S).mp4"}
DURATION=${DURATION:-45}
BACKEND_URL=${BACKEND_URL:-"http://localhost:8000"}
FRONTEND_URL=${FRONTEND_URL:-"http://localhost:3000"}
SESSION_ID=${SESSION_ID:-"demo-$(date +%s)"}
WALLET_PUBKEY=${WALLET_PUBKEY:-"DemoWallet111111111111111111111111111111111"}

usage() {
  cat <<EOF
Usage: ${0##*/} [--duration seconds] [--output file] [--mode ffmpeg|obs]

Environment overrides:
  BACKEND_URL, FRONTEND_URL, SESSION_ID, WALLET_PUBKEY
  MODE (ffmpeg|obs), OUTPUT, DURATION
  FFMPEG_INPUT_ARGS (required for MODE=ffmpeg)
  OBS_PORT, OBS_PASSWORD (used when MODE=obs)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

log() {
  printf '[demo] %s\n' "$*"
}

ensure_backend() {
  log "Checking backend at ${BACKEND_URL}/health"
  for attempt in {1..10}; do
    if curl -sf "${BACKEND_URL}/health" >/dev/null; then
      log "Backend healthy"
      return 0
    fi
    log "Backend not ready (attempt ${attempt}), retrying..."
    sleep 2
  done
  log "Backend did not become healthy. Aborting."
  return 1
}

start_mock_stream() {
  local payload
  payload=$(cat <<JSON
{
  "session_id": "${SESSION_ID}",
  "wallet_pubkey": "${WALLET_PUBKEY}",
  "sdp": "$(printf 'v=0\no=- 0 0 IN IP4 127.0.0.1\n' | base64)",
  "ice_candidates": [],
  "metadata": {
    "demo": true,
    "x402": {
      "balance_usdc": 5.0,
      "user_secret": "demo-user-secret"
    }
  }
}
JSON
)

  log "Triggering mock stream session ${SESSION_ID}"
  curl -sf -X POST "${BACKEND_URL}/stream" \
    -H "Content-Type: application/json" \
    -d "${payload}" >/dev/null
  log "Stream negotiated; open ${FRONTEND_URL} to observe alerts"
}

record_with_ffmpeg() {
  if [[ -z "${FFMPEG_INPUT_ARGS:-}" ]]; then
    cat <<'EOS' >&2
FFMPEG_INPUT_ARGS is required when MODE=ffmpeg.
Examples:
  # macOS (screen capture)
  export FFMPEG_INPUT_ARGS="-f avfoundation -i 1:none"

  # Linux (X11 display)
  export FFMPEG_INPUT_ARGS="-f x11grab -video_size 1920x1080 -i :0.0"

  # Windows (desktop capture)
  export FFMPEG_INPUT_ARGS="-f gdigrab -i desktop"
EOS
    exit 1
  fi

  log "Recording with ffmpeg for ${DURATION}s -> ${OUTPUT}"
  ffmpeg -y ${FFMPEG_INPUT_ARGS} -t "${DURATION}" -pix_fmt yuv420p "${OUTPUT}"
  log "ffmpeg capture complete"
}

record_with_obs() {
  if ! command -v obs-cli >/dev/null 2>&1; then
    log "obs-cli not available; install it from https://github.com/muesli/obs-cli"
    exit 1
  fi

  log "Starting OBS recording via obs-cli"
  obs-cli --port "${OBS_PORT:-4455}" --password "${OBS_PASSWORD:-}" StartRecording >/dev/null
  trap 'log "Stopping OBS recording"; obs-cli --port "${OBS_PORT:-4455}" --password "${OBS_PASSWORD:-}" StopRecording >/dev/null' EXIT

  log "OBS recording running; sleeping for ${DURATION}s"
  sleep "${DURATION}"
  # Trap handles StopRecording
}

ensure_backend
start_mock_stream

case "${MODE}" in
  ffmpeg)
    record_with_ffmpeg
    ;;
  obs)
    record_with_obs
    ;;
  *)
    log "Unsupported MODE=${MODE}"
    exit 1
    ;;
esac

log "Demo complete. Session ${SESSION_ID}"

