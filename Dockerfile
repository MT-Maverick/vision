# Which llama.cpp image to build on. If a newer image breaks, pin an older tag:
#   docker build --build-arg LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:<older tag> .
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server

# Stage 1: compile the Rust app for the runtime image.
FROM rust:1-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

# Stage 2: download the model files (kept out of the final image's build tooling)
FROM debian:stable-slim AS models
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Pick the model with --build-arg PRESET=fast|balanced (default: balanced).
#   fast      SmolVLM-500M, about 0.55 GB. Quickest, but short and shallow answers.
#   balanced  Qwen3-VL-2B (4-bit), about 1.55 GB. Follows instructions much better; slower and needs more memory.
ARG PRESET=balanced
WORKDIR /models
RUN case "$PRESET" in \
      fast)     REPO=ggml-org/SmolVLM-500M-Instruct-GGUF; LM=SmolVLM-500M-Instruct-Q8_0.gguf; PROJ=mmproj-SmolVLM-500M-Instruct-Q8_0.gguf; NAME=SmolVLM-500M ;; \
      balanced) REPO=Qwen/Qwen3-VL-2B-Instruct-GGUF; LM=Qwen3VL-2B-Instruct-Q4_K_M.gguf; PROJ=mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf; NAME=Qwen3-VL-2B ;; \
      *) echo "Unknown PRESET '$PRESET' (use tiny, fast, or balanced)"; exit 1 ;; \
    esac \
 && curl -fL --retry 3 -o model.gguf  "https://huggingface.co/$REPO/resolve/main/$LM" \
 && curl -fL --retry 3 -o mmproj.gguf "https://huggingface.co/$REPO/resolve/main/$PROJ" \
 && echo "$NAME" > name.txt

# Stage 3: llama.cpp's official CPU server image + our Rust app
FROM ${LLAMA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    MAX_DIM=512 \
    LLAMA_CTX=8192 \
    PORT=10000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app
COPY --from=builder /build/target/release/vision-snapshot /usr/local/bin/vision-snapshot
COPY --from=builder /build/target/release/vision-healthcheck /usr/local/bin/vision-healthcheck

COPY --from=models /models /models

COPY index.html start.sh ./
COPY static ./static
RUN chmod +x start.sh

# Docker runs this every 30 s. It logs each result to `docker logs` and marks the container healthy/unhealthy.
# The start period gives the model time to load before failures count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD ["/usr/local/bin/vision-healthcheck"]

EXPOSE 10000
ENTRYPOINT ["/srv/app/start.sh"]
CMD []
