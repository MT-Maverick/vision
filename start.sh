#!/usr/bin/env bash
# Keep llama-server private to the container; Render provides external HTTPS.
set -u
PORT="${PORT:-10000}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
NUM_CTX="${NUM_CTX:-4096}"
LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://127.0.0.1:$LLAMA_PORT}"
VISION_MODEL="$(cat /models/name.txt)"
export LLAMA_SERVER_URL VISION_MODEL

LLAMA_SERVER_BIN="$(command -v llama-server || true)"
if [[ -z "$LLAMA_SERVER_BIN" && -x /app/llama-server ]]; then
  LLAMA_SERVER_BIN=/app/llama-server
fi
if [[ -z "$LLAMA_SERVER_BIN" ]]; then
  echo "Could not find the llama-server executable."
  exit 127
fi

"$LLAMA_SERVER_BIN" \
  --model /models/model.gguf \
  --mmproj /models/mmproj.gguf \
  --alias "$VISION_MODEL" \
  --host 127.0.0.1 \
  --port "$LLAMA_PORT" \
  --ctx-size "$NUM_CTX" &
LLAMA_PID=$!

uvicorn main:app --host 0.0.0.0 --port "$PORT" \
  --proxy-headers --forwarded-allow-ips='*' &
APP_PID=$!

trap 'kill "$LLAMA_PID" "$APP_PID" 2>/dev/null' TERM INT
wait -n "$LLAMA_PID" "$APP_PID"
code=$?
echo "A process exited (code $code). Stopping container."
kill "$LLAMA_PID" "$APP_PID" 2>/dev/null
exit "$code"
