FROM python:3.11-slim

# HF Spaces expects port 7860 (default). OpenEnv's LocalDockerProvider
# maps host_port:8000; inference.py passes PORT=8000 via env_vars so
# uvicorn listens on 8000 inside the container during grading.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md openenv.yaml uv.lock ./
COPY server ./server
COPY support_triage_env ./support_triage_env

RUN pip install --no-cache-dir .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
