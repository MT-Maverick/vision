# One container: Ollama + the vision model + the web app.
FROM ollama/ollama:latest

# Change this to bake in a different model (e.g. gemma3:4b), then rebuild.
ARG MODEL=qwen3-vl:2b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    OLLAMA_MODELS=/models \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_URL=http://127.0.0.1:11434 \
    OLLAMA_KEEP_ALIVE=24h \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    VISION_MODEL=${MODEL} \
    MAX_DIM=768 \
    NUM_CTX=8192 \
    PORT=10000

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the model into the image so the container never has to fetch it at startup.
# This sits before the app files so editing code doesn't re-download several GB.
RUN mkdir -p /models && bash -c 'ollama serve > /tmp/ollama.log 2>&1 & pid=$!; \
    until ollama list >/dev/null 2>&1; do sleep 1; done; \
    ollama pull "$MODEL"; \
    kill $pid; wait $pid || true'

COPY main.py index.html start.sh ./
COPY static ./static
RUN chmod +x start.sh

EXPOSE 10000
ENTRYPOINT ["/app/start.sh"]
CMD []
