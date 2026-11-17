#!/usr/bin/env bash
set -u
export PORT="${PORT:-10000}"
available_cpus=$(nproc)
export LLAMA_THREADS="${LLAMA_THREADS:-$available_cpus}"
export LLAMA_THREADS_BATCH="${LLAMA_THREADS_BATCH:-$available_cpus}"

for f in /app/llama-server /models/model.gguf /models/mmproj.gguf; do
  [ -e "$f" ] || { echo "ERROR: $f is missing, so the model server can't start."; exit 1; }
done

libdirs=$(find /app /usr/local /opt \( -type f -o -type l \) \
  \( -name 'libllama*.so*' -o -name 'libggml*.so*' -o -name 'libmtmd*.so*' \) \
  -printf '%h\n' 2>/dev/null | sort -u | paste -sd: -)
export LD_LIBRARY_PATH="${libdirs:+$libdirs:}/app:${LD_LIBRARY_PATH:-}"

if command -v ldd >/dev/null && ldd /app/llama-server 2>&1 | grep -q "not found"; then
  echo "ERROR: the model server is missing library files:"
  ldd /app/llama-server 2>&1 | grep "not found"
  echo "Library search path: $LD_LIBRARY_PATH"
  echo "Files in /app:"; ls -la /app
  echo "Rebuild with an older llama.cpp image if its library layout has changed."
  exit 1
fi

echo "Starting the model server (threads=$LLAMA_THREADS, batch_threads=$LLAMA_THREADS_BATCH, context=${LLAMA_CTX:-4096}) ..."
( cd /app && exec /app/llama-server -m /models/model.gguf --mmproj /models/mmproj.gguf \
    --host 127.0.0.1 --port 8080 \
    -c "${LLAMA_CTX:-4096}" -t "$LLAMA_THREADS" -tb "$LLAMA_THREADS_BATCH" \
    -np 1 ${LLAMA_EXTRA_ARGS:-} ) &
LLAMA_PID=$!

export LLM_MODEL="$(cat /models/name.txt 2>/dev/null || echo model)"
vision-snapshot &
APP_PID=$!

trap 'kill "$LLAMA_PID" "$APP_PID" 2>/dev/null' TERM INT
wait -n "$LLAMA_PID" "$APP_PID"
code=$?
echo "A process exited (code $code). Stopping the other process."
kill "$LLAMA_PID" "$APP_PID" 2>/dev/null
exit "$code"
