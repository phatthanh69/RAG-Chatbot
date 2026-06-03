# RAG Chatbot Flask API - Docker Configuration (uv-based)

FROM python:3.13-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5099 \
    # uv: compile bytecode for faster startup, copy (don't hardlink) into the venv,
    # and place the venv OUTSIDE /app so a compose bind-mount of . doesn't shadow it.
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# uv binary (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from the lockfile first (cached unless deps change).
# --frozen: fail if the lock is stale; --no-dev: skip the dev group (pytest, ...).
# Graph-RAG extra is intentionally NOT installed (heavy torch deps); add it with
# `uv sync --frozen --extra graph-rag` if that feature is needed.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code
COPY . .

# Writable runtime directories (app also ensures these at startup).
# Listed explicitly because /bin/sh does not expand brace patterns.
RUN mkdir -p \
    data/raw_data data/processed data/uploads data/embeddings \
    output/results chat_sessions token_logs logs temp cache

# Non-root user (owns both the code and the venv)
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app /opt/venv
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Expose port
EXPOSE $PORT

# Run application
CMD ["python", "run.py"]
