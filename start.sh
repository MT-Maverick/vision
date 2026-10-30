#!/usr/bin/env bash
# Container launcher: starts Ollama, makes sure the model exists, then starts the web app.
# Render provides HTTPS, so the app serves plain HTTP here.
# If either process stops, the container exits so Render restarts it.
set -u
PORT="${PORT:-10000}"

# 1. Ollama (listens inside the container only)
ollama serve &
OLLAMA_PID=$!
for _ in $(seq 1 60); do
  ollama list >/dev/null 2>&1 && break
  kill -0 "$OLLAMA_PID" 2>/dev/null || { echo "Ollama failed to start"; exit 1; }
  sleep 1
done
echo "Ollama is running."

# 2. Model: baked into the image, so this only downloads if VISION_MODEL was changed
want="$VISION_MODEL"
case "$want" in *:*) ;; *) want="$want:latest" ;; esac
if ! ollama list | awk 'NR>1{print $1}' | grep -qx "$want"; then
  echo "Model '$want' is not in the image. Downloading it ..."
  ollama pull "$VISION_MODEL" || echo "Could not download '$VISION_MODEL'. Analysis will fail until it exists."
fi

# 3. Web app
uvicorn main:app --host 0.0.0.0 --port "$PORT" \
  --proxy-headers --forwarded-allow-ips='*' &
APP_PID=$!

trap 'kill "$OLLAMA_PID" "$APP_PID" 2>/dev/null' TERM INT
wait -n "$OLLAMA_PID" "$APP_PID"
code=$?
echo "A process exited (code $code). Stopping container."
kill "$OLLAMA_PID" "$APP_PID" 2>/dev/null
exit "$code"
