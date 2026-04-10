FROM python:3.11-slim

# OpenEnv's LocalDockerProvider maps host_port:8000 inside the container.
# HF Spaces injects PORT=7860. We default to 8000 for OpenEnv compatibility
# and let HF override it via the PORT env var.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md openenv.yaml uv.lock ./
COPY server ./server
COPY support_triage_env ./support_triage_env

RUN pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null || exit 1

CMD sh -c 'uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}'
