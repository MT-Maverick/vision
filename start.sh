#!/usr/bin/env bash
# Runs the model server (inside the container only) and the web app (the public part).
# If either stops, the container exits so Render restarts it.
set -u
export PORT="${PORT:-10000}"

# Fail loudly if something is missing (for example if a folder was mounted over /app).
for f in /app/llama-server /models/model.gguf /models/mmproj.gguf; do
  [ -e "$f" ] || { echo "ERROR: $f is missing, so the model server can't start."; exit 1; }
done

# llama.cpp has moved its shared libraries around between releases, so find wherever this
# image keeps them and add those folders to the library search path.
libdirs=$(find /app /usr/local /opt \( -type f -o -type l \) \
  \( -name 'libllama*.so*' -o -name 'libggml*.so*' -o -name 'libmtmd*.so*' \) \
  -printf '%h\n' 2>/dev/null | sort -u | paste -sd: -)
export LD_LIBRARY_PATH="${libdirs:+$libdirs:}/app:${LD_LIBRARY_PATH:-}"

# Check before starting, so a problem is explained instead of just exiting with code 127.
if command -v ldd >/dev/null && ldd /app/llama-server 2>&1 | grep -q "not found"; then
  echo "ERROR: the model server is missing library files:"
  ldd /app/llama-server 2>&1 | grep "not found"
  echo "Library search path: $LD_LIBRARY_PATH"
  echo "Files in /app:"; ls -la /app
  echo "This usually means this llama.cpp image version is laid out differently."
  echo "Rebuild with an older image: docker build --build-arg LLAMA_IMAGE=<older tag> ."
  exit 1
fi

echo "Starting the model server (threads=${LLAMA_THREADS:-1}, context=${LLAMA_CTX:-4096}) ..."
# Run from /app like the official image does; the subshell keeps the web app's folder unchanged.
( cd /app && exec /app/llama-server -m /models/model.gguf --mmproj /models/mmproj.gguf \
    --port 8080 \
    -c "${LLAMA_CTX:-4096}" -t "${LLAMA_THREADS:-1}" -np 1 ${LLAMA_EXTRA_ARGS:-} ) &
LLAMA_PID=$!

export LLM_MODEL="$(cat /models/name.txt 2>/dev/null || echo model)"

# Web app starts straight away so Render sees the port open; it reports "loading" until the model is ready.
/usr/local/bin/vision-snapshot &
APP_PID=$!

STOP_SIGNAL=
trap 'STOP_SIGNAL=TERM' TERM
trap 'STOP_SIGNAL=INT' INT

EXITED_PID=
wait -n -p EXITED_PID "$LLAMA_PID" "$APP_PID"
code=$?
if [ -n "$STOP_SIGNAL" ]; then
  echo "Received SIG$STOP_SIGNAL; stopping child processes."
  kill -TERM "$LLAMA_PID" "$APP_PID" 2>/dev/null || :
  wait "$LLAMA_PID" 2>/dev/null || :
  wait "$APP_PID" 2>/dev/null || :
  if [ "$STOP_SIGNAL" = TERM ]; then exit 143; fi
  exit 130
fi

if [ "$EXITED_PID" = "$LLAMA_PID" ]; then
  echo "ERROR: llama-server exited with code $code. Stopping the web app."
elif [ "$EXITED_PID" = "$APP_PID" ]; then
  echo "ERROR: Vision Snapshot exited with code $code. Stopping llama-server."
else
  echo "ERROR: A child process exited with code $code. Stopping the container."
fi

kill -TERM "$LLAMA_PID" "$APP_PID" 2>/dev/null || :
wait "$LLAMA_PID" 2>/dev/null || :
wait "$APP_PID" 2>/dev/null || :
exit "$code"
