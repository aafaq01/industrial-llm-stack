# Build: docker compose build rag-api (context is rag/)
FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

FROM python:3.12.15-slim-trixie AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Models are baked in at pinned revisions so the runtime needs no Hugging Face access.
RUN uv sync --frozen --no-default-groups --no-cache \
    && .venv/bin/python -c "from huggingface_hub import snapshot_download as d; \
d('BAAI/bge-small-en-v1.5', revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a', local_dir='/opt/models/bge-small-en-v1.5', allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Pooling/*']); \
d('cross-encoder/ms-marco-MiniLM-L6-v2', revision='233902d25c440f23af6f7d6e94d2946bac0bee0a', local_dir='/opt/models/ms-marco-MiniLM-L6-v2', allow_patterns=['*.json', '*.txt', 'model.safetensors'])"

FROM python:3.12.15-slim-trixie
# Any UID works (OpenShift): files are group 0 and /data/corpus is group-writable.
RUN useradd --uid 1001 --gid 0 --no-create-home --home-dir /tmp app \
    && mkdir -p /data/corpus && chown 1001:0 /data/corpus && chmod 2775 /data/corpus
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /opt/models /opt/models
COPY alembic.ini ./
COPY migrations ./migrations
COPY data/sources.json ./data/sources.json
COPY app ./app
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBED_MODEL=/opt/models/bge-small-en-v1.5 \
    RERANK_MODEL=/opt/models/ms-marco-MiniLM-L6-v2
USER 1001
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
