# Tricura incident-risk API — uv-based image
FROM python:3.12-slim

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models \
    ARTIFACTS_DIR=/app/artifacts \
    PREDICTION_LOG=/app/logs/predictions.jsonl

WORKDIR /app

# install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# app code + serving artifacts (models/ and artifacts/ are produced by modeling.ipynb)
COPY serving/ ./serving/
COPY models/ ./models/
COPY artifacts/ ./artifacts/

RUN mkdir -p /app/logs

EXPOSE 8000

# healthcheck hits the API's own endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uv", "run", "--no-dev", "uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
