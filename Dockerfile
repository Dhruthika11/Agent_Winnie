# Stage 1: Build & wheels
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && pip wheel --no-cache-dir --wheel-dir /app/wheels -e .

# Stage 2: Runtime (Minimal attack surface, unprivileged user)
FROM python:3.11-slim AS runtime
WORKDIR /app
RUN groupadd -r hermes && useradd -r -g hermes -s /sbin/nologin -d /app hermes
RUN apt-get update && apt-get install -y --no-install-recommends curl docker.io && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY --chown=hermes:hermes src ./src
COPY --chown=hermes:hermes infrastructure ./infrastructure
RUN mkdir -p /app/workspace /app/.mem0 && chown -R hermes:hermes /app/workspace /app/.mem0 /app

USER hermes
EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=5s --retries=3 CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
