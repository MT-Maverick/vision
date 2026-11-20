# Vision Snapshot

Vision Snapshot uses a local vision model served by [llama.cpp](https://github.com/ggml-org/llama.cpp) to identify or describe what your camera sees. Use a live camera, choose a photo, or let Watch mode look for people and landmarks

The web app is written in Rust. The Docker image runs it next to llama.cpp, started by one launcher script.

## Project layout

| Path | Contents |
| --- | --- |
| `src/` | Rust web app: `main.rs` (server, WebSocket, prompts), `logging.rs` (log format), `healthcheck.rs` (Docker health check). |
| `web/` | The web interface: `index.html` and `static/` (CSS, JavaScript, icon). |
| `scripts/start.sh` | Launcher: starts llama.cpp, then the app, and checks that the app can reach the model. |
| `Dockerfile` | Builds the app, downloads the model, and sets up the launcher and health check. |
| `Cargo.toml`, `Cargo.lock` | Rust package configuration. |
| `render.yaml` | Render blueprint. |
| `.github/workflows/` | Builds and publishes the Docker image. |

## Contents

- [Quick start](#quick-start)
- [How it starts](#how-it-starts)
- [Choose a model](#choose-a-model)
- [Run locally](#run-locally)
- [Use the app](#use-the-app)
- [Logs and health checks](#logs-and-health-checks)
- [Configuration](#configuration)
- [Location and privacy](#location-and-privacy)

## Quick start

You need Docker. From the project directory, build and run the image:

```sh
docker build -t vision-snapshot .
docker run --rm -p 10000:10000 vision-snapshot
```

The default `balanced` preset downloads Qwen3-VL-2B and its vision projector during the build. The first build can take a few minutes. For higher throughput on limited CPU, build with the smaller SmolVLM-500M model instead (`--build-arg PRESET=fast`); it trades some answer quality for lower compute and memory use. Open [http://localhost:10000](http://localhost:10000) after the container starts.

Camera access works on `localhost` over HTTP. For other devices or hostnames, use HTTPS. If live camera access is unavailable, choose **Options** and take or select a photo.

## How it starts

The container runs `scripts/start.sh`, which:

1. Checks that llama-server, the model files, the app and `curl` are present, and explains any problem (such as missing llama.cpp libraries) instead of exiting with a bare error code.
2. Starts llama-server on `127.0.0.1:8080`. It is only reachable from inside the container. If it exits right away (bad arguments, a damaged model file), the launcher says so and stops.
3. Starts the web app, pointed at that same server through `LLM_BASE_URL` and `LLM_HEALTH_URL`, so the two can't disagree about the address.
4. Waits for the model to load, then asks the app whether it can see the model, and logs `Ready: ...` only when it can. The app starts without waiting, so the public port is open straight away. Until the model is ready, the page shows that it is loading.
5. Keeps watching both processes. If either stops, it stops the other and exits with a non-zero code, so Docker or Render restarts the container. A `docker stop` shuts both down cleanly.

## Choose a model

The Docker image downloads a preset model and its matching projector at build time:

| Preset | Model | Build command |
| --- | --- | --- |
| `balanced` (default) | Qwen3-VL-2B, larger and more capable | `docker build -t vision-snapshot .` |
| `fast` | SmolVLM-500M, smaller and quicker | `docker build --build-arg PRESET=fast -t vision-snapshot .` |

The preset also determines the model name the app shows. Changing `LLM_MODEL` at runtime only changes that label; it does not load a different model.

## Run locally

Install a llama.cpp build with multimodal support, a Rust toolchain, and download a matching model GGUF and projector GGUF. Then build and start everything with the same launcher the container uses, from the project directory:

```sh
cargo build --release

LLAMA_BIN=$(command -v llama-server) \
MODEL_FILE=/path/to/SmolVLM-500M-Instruct-Q8_0.gguf \
MMPROJ_FILE=/path/to/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
LLM_MODEL=SmolVLM-500M \
APP_BIN=target/release/vision-snapshot \
scripts/start.sh
```

Open [http://localhost:10000](http://localhost:10000).

To run the pieces yourself instead, start `llama-server` with `--host 127.0.0.1 --port 8080 --alias <name>`, then run `cargo run --release` from the project directory. The app's defaults already point at that address.

## Use the app

1. Allow camera access when prompted. Use **Flip** to switch between front and rear cameras.
2. Press the shutter to analyze the current view. In **Options**, choose an answer style.
3. Choose **Burst of 4** to analyze several frames from a short camera clip.
4. Turn on **Watch** to check for selected people or landmarks and analyze the view when one is spotted.
5. Location is optional. Enable it for browser GPS, or enter a place in **Options**. A typed place overrides GPS.

Recent results appear along the bottom of the screen.

## Logs and health checks

Everything is written to `docker logs`: the launcher (`[launcher]`), llama-server's own output, and the app. App and launcher lines start with a timestamp, and problems are marked `WARNING:` or `ERROR:`. The app logs:

- its settings at startup, and when the model becomes ready (or hasn't after about three minutes);
- one line per HTTP request, with the client address (the real one behind a proxy, from `X-Forwarded-For`);
- each camera client connecting and disconnecting;
- for every analysis: a numbered request with its mode, frame count, place, timing (reading the photo, writing the answer, tokens per second), the model's answer, and the outcome. Errors include the reason, such as the model still loading.

Docker runs `vision-snapshot healthcheck` every 30 seconds. It checks the app and the model server, prints one line (for example `healthcheck: app=ok (HTTP 200, 1 ms) model=ok (HTTP 200, 1 ms) -> healthy`) and also writes it to the container log. Set `HEALTH_LOG=changes` to log only when the result changes, or `HEALTH_LOG=off` to stop logging it.

## Configuration

### App settings

The app requires a positive whole number for `MAX_DIM`, `INFERENCE_TIMEOUT`, and `MAX_CONCURRENT`.

| Setting | Default | Description |
| --- | --- | --- |
| `PORT` | `10000` | Port used by the web app. |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | llama.cpp OpenAI-compatible API URL. The launcher sets this. |
| `LLM_HEALTH_URL` | `http://127.0.0.1:8080/health` | llama.cpp health-check URL. The launcher sets this. |
| `LLM_MODEL` | `SmolVLM-256M` (the Docker image uses the preset's name) | Model label returned by the app; does not load or switch models. |
| `WEB_DIR` | `web` (the launcher uses the repo's `web/`) | Folder holding `index.html` and `static/`. |
| `MAX_DIM` | `512` | Maximum frame dimension. Larger frames are resized. |
| `INFERENCE_TIMEOUT` | `300` seconds | Maximum time allowed for a model request. |
| `MAX_CONCURRENT` | `1` | Maximum simultaneous model requests. |
| `HEALTH_LOG` | `all` | `all`, `changes` or `off`: how health check results are logged. |

### Docker build and launcher settings

| Setting | Default | Description |
| --- | --- | --- |
| `PRESET` | `balanced` | Docker build argument selecting the model files (`fast` or `balanced`). |
| `LLAMA_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server` | Docker build argument selecting the llama.cpp image. |
| `LLAMA_CTX` | `4096` | Context size passed to llama.cpp. |
| `LLAMA_THREADS` | CPUs available to the container | CPU threads used for generation. |
| `LLAMA_THREADS_BATCH` | CPUs available to the container | CPU threads used for prompt/batch processing. |
| `LLAMA_EXTRA_ARGS` | unset | Additional arguments passed to llama.cpp, separated by spaces. |
| `LLAMA_PORT` | `8080` | Port of the model server inside the container. Must differ from `PORT`. |
| `LLAMA_READY_TIMEOUT` | `300` seconds | How long the launcher waits before warning that the model still isn't ready. The app keeps running. |
| `LLAMA_BIN`, `MODEL_FILE`, `MMPROJ_FILE`, `APP_BIN` | `/app/llama-server`, `/models/model.gguf`, `/models/mmproj.gguf`, `vision-snapshot` | Locations used by the launcher; change them to run it outside Docker. |

## Location and privacy

Camera frames are sent to the configured llama.cpp server. In the Docker setup, llama.cpp runs locally in the same container. Location is off by default; enabling it asks the browser for GPS permission and uses OpenStreetMap's Nominatim service to turn coordinates into a place name. A manually entered place is sent to the model as context.
