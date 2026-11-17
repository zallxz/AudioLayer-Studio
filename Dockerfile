# syntax=docker/dockerfile:1
# AudioLayer Studio — API + web UI (+ optional remote MCP at /mcp).
#   docker build -t audiolayer .                      # CPU image with ML models
#   docker build --build-arg ML=false -t audiolayer:lite .   # no PyTorch (detection-only, heuristic)

FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS app
ARG ML=true
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    AUDIOLAYER_WORKSPACE=/data/workspace AUDIOLAYER_MODEL_DIR=/data/models TORCH_HOME=/data/models/torch \
    HF_HOME=/data/models/hf AUDIOLAYER_FRONTEND_DIST=/app/frontend/dist
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY backend ./backend
RUN if [ "$ML" = "true" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install ".[ml]"; \
    else pip install . ; fi
COPY scripts ./scripts
COPY --from=ui /ui/dist ./frontend/dist
RUN useradd -m -u 10001 audiolayer && mkdir -p /data && chown -R audiolayer /data
USER audiolayer
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["audiolayer-api", "--host", "0.0.0.0", "--port", "8000"]
