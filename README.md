# Vision Snapshot

Vision Snapshot uses a local vision model served by [llama.cpp](https://github.com/ggml-org/llama.cpp) to identify or describe what your camera sees. Use a live camera, choose a photo, or let Watch mode look for people and landmarks.

## Project layout

| Path | Contents |
| --- | --- |
| `python/` | Python backend, health check, and Python dependency lists. |
| `src/` | Rust backend. |
| `index.html`, `static/` | Shared web interface and assets. |
| `Cargo.toml` | Rust package configuration. |

## Contents

- [Quick start](#quick-start)
- [Rust Docker image](#rust-docker-image)
- [Choose a model](#choose-a-model)
- [Run locally](#run-locally)
- [Use the app](#use-the-app)
- [Configuration](#configuration)
- [Location and privacy](#location-and-privacy)

## Quick start

You need Docker. From the project directory, build and run the image:

```sh
docker build -t vision-snapshot .
docker run --rm -p 10000:10000 vision-snapshot
```

The default `balanced` preset downloads Qwen3-VL-2B and its vision projector during the build. The first build can take a few minutes. To use the smaller SmolVLM-500M model instead, build with `--build-arg PRESET=fast`. Open [http://localhost:10000](http://localhost:10000) after the container starts.

Camera access works on `localhost` over HTTP. For other devices or hostnames, use HTTPS. If live camera access is unavailable, choose **Options** and take or select a photo.

## Rust Docker image

The Rust and Python images use the same local tag and run command. Build the Rust variant with its dedicated Dockerfile:

```sh
docker buildx build --load -f Dockerfile.rust -t vision-snapshot .
docker run --rm -p 10000:10000 vision-snapshot
```

This builds the Rust server in a separate stage, starts llama.cpp with the selected model, and runs the Rust app. Build the Python variant instead with the `Dockerfile` shown in [Quick start](#quick-start). Whichever image you build last is tagged `vision-snapshot` and runs with the same command.

For higher throughput on limited CPU, use the smaller `fast` model preset. It uses SmolVLM-500M instead of Qwen3-VL-2B, trading some response quality for lower compute and memory use:

```sh
docker buildx build --load -f Dockerfile.rust -t vision-snapshot --build-arg PRESET=fast .
docker run --rm -p 10000:10000 vision-snapshot
```

Increasing image-token or context settings can increase computation, so leave them at their defaults when throughput is the priority.

## Choose a model

The Docker image downloads a preset model and its matching projector at build time:

| Preset | Model | Build command |
| --- | --- | --- |
| `balanced` (default) | Qwen3-VL-2B, larger and more capable | `docker build -t vision-snapshot .` |
| `fast` | SmolVLM-500M, smaller and quicker | `docker build --build-arg PRESET=fast -t vision-snapshot .` |

The preset also determines the model alias reported by the app. Changing `LLM_MODEL` at runtime does not replace the model loaded by llama.cpp.

## Run locally

For either backend, install a llama.cpp build with multimodal support, download a matching model GGUF and projector GGUF, then start llama.cpp. This example uses the `fast` model:

```sh
llama-server \
  --model /path/to/SmolVLM-500M-Instruct-Q8_0.gguf \
  --mmproj /path/to/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  --alias SmolVLM-500M \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096
```

Choose **one** backend in a second terminal. Run commands from the project directory so the app can find `index.html` and `static/`.

### Python backend

Install Python 3.10 or later, then install and start the app:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

Start the app:

```sh
LLM_BASE_URL=http://127.0.0.1:8080/v1 \
LLM_HEALTH_URL=http://127.0.0.1:8080/health \
LLM_MODEL=SmolVLM-500M \
uvicorn main:app --app-dir python --host 0.0.0.0 --port 10000
```

### Rust backend

Install Rust with Cargo, then run:

```sh
LLM_BASE_URL=http://127.0.0.1:8080/v1 \
LLM_HEALTH_URL=http://127.0.0.1:8080/health \
LLM_MODEL=SmolVLM-500M \
cargo run --release
```

Open [http://localhost:10000](http://localhost:10000).

## Use the app

1. Allow camera access when prompted. Use **Flip** to switch between front and rear cameras.
2. Press the shutter to analyze the current view. In **Options**, choose an answer style.
3. Choose **Burst of 4** to analyze several frames from a short camera clip.
4. Turn on **Watch** to check for selected people or landmarks and analyze the view when one is spotted.
5. Location is optional. Enable it for browser GPS, or enter a place in **Options**. A typed place overrides GPS.

Recent results appear along the bottom of the screen.

## Configuration

### App settings

These settings apply when running either backend. The Rust server requires a positive value for `MAX_DIM`, `INFERENCE_TIMEOUT`, and `MAX_CONCURRENT`.

| Setting | Default | Description |
| --- | --- | --- |
| `PORT` | `10000` | Port used by the web app. |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | llama.cpp OpenAI-compatible API URL. |
| `LLM_HEALTH_URL` | `http://127.0.0.1:8080/health` | llama.cpp health-check URL. |
| `LLM_MODEL` | Backend-specific | Model label returned by the app; does not load or switch models. |
| `MAX_DIM` | `512` | Maximum frame dimension. Rust always resizes oversized images; Python resizes when optional OpenCV is installed. |
| `INFERENCE_TIMEOUT` | `300` seconds | Maximum time allowed for a model request. |
| `MAX_CONCURRENT` | `1` | Maximum simultaneous model requests. |

The Python backend's default `LLM_MODEL` is `SmolVLM-500M-Instruct`; the Rust backend's is `SmolVLM-256M`. Set `LLM_MODEL` to match the model alias configured for your llama.cpp server.

### Docker build and model-server settings

| Setting | Default | Description |
| --- | --- | --- |
| `PRESET` | `balanced` | Docker build argument selecting the model files (`fast` or `balanced`). |
| `LLAMA_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server` | Docker build argument selecting the llama.cpp image. |
| `LLAMA_CTX` | `4096` | Context size passed to llama.cpp in Docker. |
| `LLAMA_THREADS` | CPUs available to the container | CPU threads used for generation; override with this variable. |
| `LLAMA_THREADS_BATCH` | CPUs available to the container | CPU threads used for prompt/batch processing; override with this variable. |
| `LLAMA_EXTRA_ARGS` | unset | Additional arguments passed to llama.cpp in Docker. |

## Location and privacy

Camera frames are sent to the configured llama.cpp server. In the Docker setup, llama.cpp runs locally in the same container. Location is off by default; enabling it asks the browser for GPS permission and uses OpenStreetMap's Nominatim service to turn coordinates into a place name. A manually entered place is sent to the model as context.
