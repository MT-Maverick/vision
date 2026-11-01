# Vision-Snapshot

Vision Snapshot uses a local vision model served by [llama.cpp](https://github.com/ggml-org/llama.cpp) to identify or describe what your camera sees. Use a live camera, choose a photo, or let Watch mode look for people and landmarks.

## Quick start

You need Docker. From the project directory, build and run the image:

```sh
docker build -t vision-snapshot .
docker run --rm -p 10000:10000 vision-snapshot
```

The default `fast` preset downloads SmolVLM-500M and its vision projector during the build. The first build can take a few minutes. Open [http://localhost:10000](http://localhost:10000) after the container starts.

Camera access works on `localhost` over HTTP. For other devices or hostnames, use HTTPS. If live camera access is unavailable, choose **Options** and take or select a photo.

## Choose a model

The image includes one preset model and its matching projector:

| Preset | Model | Build command |
| --- | --- | --- |
| `fast` (default) | SmolVLM-500M, smaller and quicker | `docker build -t vision-snapshot .` |
| `balanced` | Qwen3-VL-2B, larger and more capable | `docker build --build-arg PRESET=balanced -t vision-snapshot .` |

The model files are selected at build time. `LLM_MODEL` is the request alias exposed by llama.cpp; changing it at runtime does not replace the model baked into the image.

## Use the app

1. Allow camera access when prompted. Use **Flip** to switch between front and rear cameras.
2. Press the shutter to analyze the current view. In **Options**, choose **Name it** for a short label or **Describe it** for a brief description.
3. Choose **Burst of 4** to analyze several frames from a short camera clip.
4. Turn on **Watch** to check for selected people or landmarks and analyze the view when one is spotted.
5. Location is optional. Enable it for browser GPS, or enter a place in **Options**. A typed place overrides GPS.

Recent results appear along the bottom of the screen.

## Run without Docker

Install Rust and a llama.cpp build with multimodal support. Download a matching model GGUF and projector GGUF from one of the model repositories listed in the Dockerfile. Start the server with the matching alias; for the default preset:

```sh
llama-server \
	--model /path/to/SmolVLM-500M-Instruct-Q8_0.gguf \
	--mmproj /path/to/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
	--alias SmolVLM-500M \
	--host 127.0.0.1 \
	--port 8080 \
	--ctx-size 4096
```

In a second terminal, start the Rust web app:

```sh
LLM_BASE_URL=http://127.0.0.1:8080/v1 LLM_MODEL=SmolVLM-500M \
	cargo run --release
```

Open [http://localhost:10000](http://localhost:10000).

## Configuration

| Setting | Default | Description |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | llama.cpp OpenAI-compatible API URL. |
| `LLM_HEALTH_URL` | `http://127.0.0.1:8080/health` | llama.cpp health endpoint. |
| `LLM_MODEL` | Set from the baked preset | Model alias used for llama.cpp requests. |
| `PORT` | `10000` | Web app port. |
| `MAX_DIM` | `512` | Maximum server-side frame dimension. |
| `INFERENCE_TIMEOUT` | `300` seconds | Maximum time allowed for an inference request. |
| `MAX_CONCURRENT` | `1` | Maximum number of simultaneous model requests. |

## Location and privacy

Camera frames are sent to the llama.cpp server configured for the app. In the Docker setup, llama.cpp runs locally in the same container. Location is off by default; enabling it asks the browser for GPS permission and uses a geocoding service to turn coordinates into a place name. A manually entered place is sent to the model as context.
