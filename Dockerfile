# Which llama.cpp image to build on. If a newer image breaks, pin an older tag:
#   docker build --build-arg LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:<older tag> .
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server

# Stage 1: download the model files (kept out of the final image's build tooling)
FROM debian:stable-slim AS models
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Pick the model with --build-arg PRESET=fast|balanced (default: fast).
#   fast      SmolVLM-256M, about 0.30 GB. Quickest and lightest.
#   balanced  Qwen3-VL-2B (4-bit), about 1.55 GB. Follows instructions much better; slower and needs more memory.
ARG PRESET=fast
WORKDIR /models
RUN case "$PRESET" in \
      fast)     REPO=ggml-org/SmolVLM-256M-Instruct-GGUF; LM=SmolVLM-256M-Instruct-Q8_0.gguf; PROJ=mmproj-SmolVLM-256M-Instruct-Q8_0.gguf; NAME=SmolVLM-256M ;; \
      balanced) REPO=Qwen/Qwen3-VL-2B-Instruct-GGUF; LM=Qwen3VL-2B-Instruct-Q4_K_M.gguf; PROJ=mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf; NAME=Qwen3-VL-2B ;; \
      *) echo "Unknown PRESET '$PRESET' (use fast or balanced)"; exit 1 ;; \
    esac \
 && curl -fL --retry 3 -o model.gguf  "https://huggingface.co/$REPO/resolve/main/$LM" \
 && curl -fL --retry 3 -o mmproj.gguf "https://huggingface.co/$REPO/resolve/main/$PROJ" \
 && echo "$NAME" > name.txt

# Stage 2: build the Rust web app
FROM rust:1-bookworm AS rust-build
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

# Stage 3: llama.cpp's official CPU server image + the Rust app
FROM ${LLAMA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PORT=10000 \
    MAX_DIM=512

# curl is used by the launcher script to confirm the app can reach the model.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=models /models /models
COPY --from=rust-build /build/target/release/vision-snapshot /usr/local/bin/vision-snapshot

# The repo's own layout: scripts/start.sh finds the page in ../web.
WORKDIR /srv/app
COPY web ./web
COPY scripts ./scripts
# The binary was built on a different base image than this one. Running it once here makes a
# mismatch (for example in the C library) fail the build, not the first start of the container.
RUN chmod +x scripts/start.sh && vision-snapshot --version

# Docker runs this every 30 s. It logs each result to `docker logs` and marks the container healthy/unhealthy.
# The start period gives the model time to load before failures count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD ["vision-snapshot", "healthcheck"]

EXPOSE 10000
ENTRYPOINT ["/srv/app/scripts/start.sh"]
CMD []
