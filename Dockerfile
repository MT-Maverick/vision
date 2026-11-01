# Which llama.cpp image to build on. If a newer image breaks, pin an older tag:
#   docker build --build-arg LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:<older tag> .
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server

# Stage 1: download the model files (kept out of the final image's build tooling)
FROM debian:stable-slim AS models
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Pick the model with --build-arg PRESET=fast|balanced (default: balanced).
#   fast      SmolVLM-500M, about 0.55 GB. Quickest, but short and shallow answers.
#   balanced  Qwen3-VL-2B (4-bit), about 1.55 GB. Follows instructions much better; slower and needs more memory.
ARG PRESET=fast
WORKDIR /models
RUN case "$PRESET" in \
      fast)     REPO=ggml-org/SmolVLM-500M-Instruct-GGUF; LM=SmolVLM-500M-Instruct-Q8_0.gguf; PROJ=mmproj-SmolVLM-500M-Instruct-Q8_0.gguf; NAME=SmolVLM-500M ;; \
      balanced) REPO=Qwen/Qwen3-VL-2B-Instruct-GGUF; LM=Qwen3VL-2B-Instruct-Q4_K_M.gguf; PROJ=mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf; NAME=Qwen3-VL-2B ;; \
      *) echo "Unknown PRESET '$PRESET' (use fast or balanced)"; exit 1 ;; \
    esac \
 && curl -fL --retry 3 -o model.gguf  "https://huggingface.co/$REPO/resolve/main/$LM" \
 && curl -fL --retry 3 -o mmproj.gguf "https://huggingface.co/$REPO/resolve/main/$PROJ" \
 && echo "$NAME" > name.txt

# Stage 2: llama.cpp's official CPU server image + our Python app
FROM ${LLAMA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    MAX_DIM=512 \
    PORT=10000

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /srv/app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --from=models /models /models

COPY main.py index.html start.sh healthcheck.py ./
COPY static ./static
RUN chmod +x start.sh

# Docker runs this every 30 s. It logs each result to `docker logs` and marks the container healthy/unhealthy.
# The start period gives the model time to load before failures count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD ["python", "/srv/app/healthcheck.py"]

EXPOSE 10000
ENTRYPOINT ["/srv/app/start.sh"]
CMD []