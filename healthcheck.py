"""Check that both the web app and its llama.cpp model server are healthy."""
import os
import sys
from urllib.error import URLError
from urllib.request import urlopen


def check(url: str) -> None:
    with urlopen(url, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")


port = os.getenv("PORT", "10000")
llama_url = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")

try:
    check(f"http://127.0.0.1:{port}/healthz")
    check(f"{llama_url}/health")
except (OSError, RuntimeError, URLError) as error:
    print(f"Health check failed: {error}", file=sys.stderr)
    sys.exit(1)