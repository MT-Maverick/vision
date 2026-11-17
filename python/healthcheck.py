#!/usr/bin/env python3
"""Docker health check for Vision Snapshot (standard library only).

Checks the web app and the model server, writes one line per run to the container's
log (so it shows up in `docker logs`), and exits 0 if both are healthy, 1 otherwise.

  HEALTH_LOG=all      log every check (default)
  HEALTH_LOG=changes  log only when the result changes
  HEALTH_LOG=off      don't log, just report to Docker
"""
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

PORT = os.getenv("PORT", "10000")
CHECKS = [
    ("app", f"http://127.0.0.1:{PORT}/healthz"),
    ("model", os.getenv("LLM_HEALTH_URL", "http://127.0.0.1:8080/health")),
]
MODE = os.getenv("HEALTH_LOG", "all").lower()
LOG_TARGET = os.getenv("HEALTH_LOG_TARGET", "/proc/1/fd/1")  # the container's main output
STATE_FILE = "/tmp/healthcheck.last"


def probe(url: str) -> tuple[bool, str]:
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            ok, what = r.status == 200, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        ok, what = False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        ok, what = False, type(e.reason).__name__ if not isinstance(e.reason, str) else e.reason
    except Exception as e:
        ok, what = False, type(e).__name__
    return ok, f"{what}, {(time.perf_counter() - start) * 1000:.0f} ms"


def main() -> int:
    results = [(name, *probe(url)) for name, url in CHECKS]
    healthy = all(ok for _, ok, _ in results)
    parts = " ".join(f"{name}={'ok' if ok else 'FAIL'} ({detail})" for name, ok, detail in results)
    line = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}  "
            f"healthcheck: {parts} -> {'healthy' if healthy else 'UNHEALTHY'}")

    print(line)  # also kept by Docker: `docker inspect` shows the last few results

    if MODE != "off":
        try:
            previous = open(STATE_FILE).read()
        except OSError:
            previous = ""
        state = "healthy" if healthy else "unhealthy"
        try:
            open(STATE_FILE, "w").write(state)
        except OSError:
            pass
        if MODE == "all" or state != previous:
            try:
                with open(LOG_TARGET, "a") as log:
                    log.write(line + "\n")
            except OSError:
                pass  # not allowed to write there; the line above still reaches Docker
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())