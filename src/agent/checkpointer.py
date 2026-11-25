"""
Distributed Checkpointer Factory for LangGraph State Machine.

Supports:
1. PostgresSaver (langgraph-checkpoint-postgres) for horizontally scaled multi-pod production.
2. MemorySaver (in-memory) for lightweight local testing or offline fallback.
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from src.config import settings

logger = logging.getLogger("hermes.agent.checkpointer")

_ACTIVE_CHECKPOINTER: BaseCheckpointSaver | None = None


def get_checkpointer() -> BaseCheckpointSaver:
    """Return a singleton distributed checkpointer (PostgresSaver) or MemorySaver."""
    global _ACTIVE_CHECKPOINTER
    if _ACTIVE_CHECKPOINTER is not None:
        return _ACTIVE_CHECKPOINTER

    backend = (settings.checkpointer_backend or "auto").lower()

    if backend in ("postgres", "auto") and settings.postgres_url:
        try:
            import psycopg
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool

            # Quick probe with 1s timeout to ensure Postgres is running
            with psycopg.connect(settings.postgres_url, connect_timeout=1) as probe_conn:
                pass

            logger.info("initializing_postgres_checkpointer", extra={"backend": "postgres"})
            pool = ConnectionPool(settings.postgres_url, max_size=10, kwargs={"autocommit": True})
            saver = PostgresSaver(pool)
            saver.setup()
            _ACTIVE_CHECKPOINTER = saver
            logger.info("postgres_checkpointer_ready")
            return _ACTIVE_CHECKPOINTER
        except Exception as exc:
            logger.warning(
                f"[Checkpointer] Postgres initialization failed ({exc.__class__.__name__}: {exc}). "
                "Falling back to MemorySaver for local state."
            )

    logger.info("initializing_memory_checkpointer", extra={"backend": "memory"})
    _ACTIVE_CHECKPOINTER = MemorySaver()
    return _ACTIVE_CHECKPOINTER


_ACTIVE_ASYNC_CHECKPOINTER: Any | None = None


async def get_async_checkpointer() -> Any:
    """Return an asynchronous singleton checkpointer (AsyncPostgresSaver) or AsyncMemorySaver."""
    global _ACTIVE_ASYNC_CHECKPOINTER
    if _ACTIVE_ASYNC_CHECKPOINTER is not None:
        return _ACTIVE_ASYNC_CHECKPOINTER

    backend = (settings.checkpointer_backend or "auto").lower()

    if backend in ("postgres", "auto") and settings.postgres_url:
        try:
            import psycopg
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool

            async with await psycopg.AsyncConnection.connect(settings.postgres_url, connect_timeout=1) as probe_conn:
                pass

            pool = AsyncConnectionPool(settings.postgres_url, max_size=10, kwargs={"autocommit": True})
            saver = AsyncPostgresSaver(pool)
            await saver.asetup()
            _ACTIVE_ASYNC_CHECKPOINTER = saver
            return _ACTIVE_ASYNC_CHECKPOINTER
        except Exception as exc:
            logger.warning(
                f"[Checkpointer] Async Postgres initialization failed ({exc.__class__.__name__}: {exc}). "
                "Falling back to AsyncMemorySaver."
            )

    from langgraph.checkpoint.memory import MemorySaver
    _ACTIVE_ASYNC_CHECKPOINTER = MemorySaver()
    return _ACTIVE_ASYNC_CHECKPOINTER
