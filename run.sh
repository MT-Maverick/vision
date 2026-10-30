#!/usr/bin/env bash
# Vision Snapshot launcher: sets up the virtual environment, dependencies,
# HTTPS certificate and Ollama, then starts the server.
set -e
cd "$(dirname "$0")"

PORT="${PORT:-8443}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_CONTAINER="${OLLAMA_CONTAINER:-model}"

# 1. Virtual environment: create it if it doesn't exist
if [ ! -f .venv/bin/activate ]; then
  PY=$(command -v python3 || command -v python || true)
  [ -z "$PY" ] && { echo "❌ Python 3 is not installed."; exit 1; }
  echo "📦 Creating virtual environment in .venv ..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate
echo "✅ Virtual environment active ($(python --version))."

# 2. Dependencies: reinstall only when the requirements files change
STAMP=".venv/.requirements.stamp"
SUM=$(cat requirements.txt requirements-optional.txt 2>/dev/null | cksum)
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$SUM" ]; then
  echo "📥 Installing dependencies ..."
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
  if [ -f requirements-optional.txt ]; then
    python -m pip install --quiet -r requirements-optional.txt \
      || echo "ℹ️  Optional packages skipped (the app works without them)."
  fi
  echo "$SUM" > "$STAMP"
fi

# 3. HTTPS certificate (phones only allow the camera over HTTPS)
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$IP" ] && IP=$(ipconfig getifaddr en0 2>/dev/null || true)
if [ ! -f certs/cert.pem ]; then
  command -v openssl >/dev/null || { echo "❌ openssl is required to create the certificate."; exit 1; }
  echo "🔐 Creating a self-signed certificate ..."
  mkdir -p certs
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout certs/key.pem -out certs/cert.pem -subj "/CN=vision.local" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1${IP:+,IP:$IP}" 2>/dev/null
fi

# 4. Ollama: start the Docker container if it exists but is stopped
ollama_up() { curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }
if ! ollama_up; then
  if command -v docker >/dev/null 2>&1 \
     && docker inspect "$OLLAMA_CONTAINER" >/dev/null 2>&1; then
    echo "🤖 Starting Docker container '$OLLAMA_CONTAINER' ..."
    docker start "$OLLAMA_CONTAINER" >/dev/null
    for _ in $(seq 1 20); do ollama_up && break; sleep 1; done
  fi
fi
if ollama_up; then
  echo "🤖 Ollama is reachable at $OLLAMA_URL."
else
  echo "⚠️  Ollama is not reachable at $OLLAMA_URL. The app will start, but analysis will fail until it is running."
fi

# 5. Start the server
echo
echo "  On this computer:  https://localhost:$PORT"
[ -n "$IP" ] && echo "  On your phone:     https://$IP:$PORT   (same Wi-Fi)"
echo "  Accept the certificate warning once, then allow camera access."
echo
exec python -m uvicorn main:app --host 0.0.0.0 --port "$PORT" \
  --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
