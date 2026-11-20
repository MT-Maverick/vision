#!/usr/bin/env bash
# Container launcher for Vision Snapshot.
#
#   1. checks that everything needed is present (explains problems instead of exiting with 127),
#   2. starts the llama.cpp model server, listening on this container's loopback only,
#   3. starts the Rust web app, pointed at exactly that server,
#   4. confirms the app can actually reach the model, and logs when it can,
#   5. supervises both: if either stops, the other is stopped and the container exits,
#      so the platform (Render, Docker restart policy, ...) restarts it.
#
# The web app starts right after the model server instead of waiting for it. The model takes a
# while to load, and the public port needs to be open straight away; until the model is ready
# the app reports "loading" to the page.
set -u

# ---- settings (every one can be overridden with an environment variable) ---------------------
PORT="${PORT:-10000}"                          # the public web app
LLAMA_PORT="${LLAMA_PORT:-8080}"               # the model server (never published)
LLAMA_BIN="${LLAMA_BIN:-/app/llama-server}"
MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/model.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/mmproj.gguf}"
APP_BIN="${APP_BIN:-vision-snapshot}"
LLAMA_CTX="${LLAMA_CTX:-4096}"
available_cpus="$(nproc 2>/dev/null || echo 1)"
LLAMA_THREADS="${LLAMA_THREADS:-$available_cpus}"
LLAMA_THREADS_BATCH="${LLAMA_THREADS_BATCH:-$available_cpus}"
LLAMA_READY_TIMEOUT="${LLAMA_READY_TIMEOUT:-300}"   # seconds before warning that the model still isn't ready

# The model server is private to the container, so it is always bound to loopback.
LLAMA_HOST="127.0.0.1"
LLM_MODEL="${LLM_MODEL:-$(cat "$MODEL_DIR/name.txt" 2>/dev/null || echo model)}"   # label shown in the app
LLM_BASE_URL="http://$LLAMA_HOST:$LLAMA_PORT/v1"
LLM_HEALTH_URL="http://$LLAMA_HOST:$LLAMA_PORT/health"
# The web page and its assets live next to this script's folder (web/), in the repo and in the image.
WEB_DIR="${WEB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/web}"
# Hand the app the address of the server started below, so the two can't disagree.
export PORT LLM_MODEL LLM_BASE_URL LLM_HEALTH_URL WEB_DIR

# ---- logging, in the same "date  message" style as the app ------------------------------------
log() { printf '%s  [launcher] %s\n' "$(date '+%Y-%m-%d %H:%M:%S,%3N')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# ---- checks before anything starts ------------------------------------------------------------
for f in "$LLAMA_BIN" "$MODEL_FILE" "$MMPROJ_FILE"; do
  [ -e "$f" ] || die "$f is missing, so the model server can't start (was a folder mounted over it?)."
done
[ -x "$LLAMA_BIN" ] || die "$LLAMA_BIN is not executable."
command -v "$APP_BIN" >/dev/null 2>&1 || die "the web app '$APP_BIN' was not found on PATH."
command -v curl >/dev/null 2>&1 || die "curl is needed to confirm the model server is reachable, but it isn't installed."
[ "$PORT" != "$LLAMA_PORT" ] || die "PORT and LLAMA_PORT are both $PORT; the web app and the model server need different ports."

# llama.cpp has moved its shared libraries around between releases, so find wherever this
# image keeps them and add those folders to the library search path.
libdirs=$(find "$(dirname "$LLAMA_BIN")" /app /usr/local /opt \( -type f -o -type l \) \
  \( -name 'libllama*.so*' -o -name 'libggml*.so*' -o -name 'libmtmd*.so*' \) \
  -printf '%h\n' 2>/dev/null | sort -u | paste -sd: -)
llama_dir="$(dirname "$LLAMA_BIN")"
export LD_LIBRARY_PATH="${libdirs:+$libdirs:}$llama_dir:${LD_LIBRARY_PATH:-}"

# Explain a missing library now, instead of the server just dying with exit code 127.
if command -v ldd >/dev/null 2>&1 && ldd "$LLAMA_BIN" 2>&1 | grep -q "not found"; then
  log "ERROR: the model server is missing library files:"
  ldd "$LLAMA_BIN" 2>&1 | grep "not found" >&2
  log "Library search path: $LD_LIBRARY_PATH"
  log "Files in $llama_dir:"; ls -la "$llama_dir" >&2
  log "This usually means this llama.cpp image version is laid out differently."
  die "Rebuild with an older image: docker build --build-arg LLAMA_IMAGE=<older tag> ."
fi

# Extra llama-server flags, split on spaces (LLAMA_EXTRA_ARGS="--flash-attn on --no-mmap").
read -r -a extra_args <<< "${LLAMA_EXTRA_ARGS:-}"

LLAMA_PID=""
APP_PID=""
WATCH_PID=""

stop_all() {
  trap '' TERM INT
  [ -n "$WATCH_PID" ] && kill "$WATCH_PID" 2>/dev/null
  # Stop the web app first so no new requests arrive while the model server shuts down.
  [ -n "$APP_PID" ] && kill "$APP_PID" 2>/dev/null
  [ -n "$LLAMA_PID" ] && kill "$LLAMA_PID" 2>/dev/null
  wait 2>/dev/null
}
trap 'log "Stop signal received. Stopping the web app and the model server."; stop_all; exit 143' TERM INT

# ---- 1. the model server ----------------------------------------------------------------------
log "Starting the model server '$LLM_MODEL' on $LLAMA_HOST:$LLAMA_PORT (threads=$LLAMA_THREADS, batch_threads=$LLAMA_THREADS_BATCH, context=$LLAMA_CTX) ..."
# Run from the binary's own folder like the official image does; the subshell leaves this script's folder alone.
( cd "$llama_dir" && exec "$LLAMA_BIN" \
    -m "$MODEL_FILE" --mmproj "$MMPROJ_FILE" --alias "$LLM_MODEL" \
    --host "$LLAMA_HOST" --port "$LLAMA_PORT" \
    -c "$LLAMA_CTX" -t "$LLAMA_THREADS" -tb "$LLAMA_THREADS_BATCH" \
    -np 1 ${extra_args[@]+"${extra_args[@]}"} ) &
LLAMA_PID=$!

# Bad arguments or a corrupt model file make the server exit within a second or two. Catch that
# here so the cause is clear, rather than starting a web app that can never reach a model.
sleep 2
if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
  wait "$LLAMA_PID"; code=$?
  die "the model server exited right after starting (code $code). The reason is in the llama-server output above."
fi
log "Model server is running (pid $LLAMA_PID); loading the model in the background."

# ---- 2. the web app ---------------------------------------------------------------------------
log "Starting the web app '$APP_BIN' on port $PORT (model server: $LLM_BASE_URL) ..."
"$APP_BIN" &
APP_PID=$!

# ---- 3. confirm the app can reach the model ---------------------------------------------------
# Waits for the model server's own health check, then asks the *app* whether it sees the model as
# ready (its /api/status probes the same URL the page depends on), so a success message here
# means the whole path works, not just that the processes exist.
confirm_model_reachable() {
  local started=$SECONDS
  until curl -fsS -o /dev/null --max-time 2 "$LLM_HEALTH_URL" 2>/dev/null; do
    if (( SECONDS - started >= LLAMA_READY_TIMEOUT )); then
      log "WARNING: the model server is still not ready after ${LLAMA_READY_TIMEOUT}s ($LLM_HEALTH_URL). The app keeps running and tells the page the model is loading."
      return 1
    fi
    sleep 2
  done
  local loaded=$((SECONDS - started))
  until curl -fsS --max-time 2 "http://127.0.0.1:$PORT/api/status" 2>/dev/null | grep -q '"ready":true'; do
    if (( SECONDS - started >= LLAMA_READY_TIMEOUT )); then
      log "WARNING: the model server is ready, but the web app on port $PORT does not see it ($LLM_HEALTH_URL). Check the app's log lines above."
      return 1
    fi
    sleep 1
  done
  log "Ready: the model loaded in ${loaded}s and the web app can reach '$LLM_MODEL' at $LLM_BASE_URL."
}
confirm_model_reachable &
WATCH_PID=$!

# ---- 4. supervise both ------------------------------------------------------------------------
# Bash drops finished background jobs, so kill -0 fails as soon as either process has stopped.
while kill -0 "$LLAMA_PID" 2>/dev/null && kill -0 "$APP_PID" 2>/dev/null; do
  sleep 1
done

if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
  wait "$LLAMA_PID"; code=$?
  log "ERROR: the model server (pid $LLAMA_PID) exited with code $code. Stopping the web app."
else
  wait "$APP_PID"; code=$?
  log "ERROR: the web app (pid $APP_PID) exited with code $code. Stopping the model server."
fi
stop_all
# Neither process is meant to stop on its own, so never report success from here.
[ "$code" -ne 0 ] || code=1
exit "$code"
