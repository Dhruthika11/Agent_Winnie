"""
Durable Execution and State Checkpointing Engine for Hermes Agent Platform.

Guarantees crash-resilient multi-turn agent execution:
1. Authoritative PostgreSQL persistence for full lifecycle transitions:
   REQUEST_RECEIVED -> AGENT_RUNNING -> WAITING_FOR_TOOL -> TOOL_EXECUTED ->
   WAITING_FOR_APPROVAL -> WAITING_FOR_VERIFICATION -> COMPLETED / FAILED / INTERRUPTED
2. Atomic step-by-step checkpointing allowing resume from intermediate states.
3. Distinguishes between:
   - RETRY_FAILED: Retrying an operation that definitely failed.
   - RECOVER_UNKNOWN: Recovering an operation whose outcome is unknown (crash mid-flight).
   - RESUME_COMPLETED: Resuming an operation that completed but whose result was not committed.
4. In-memory graceful fallback for hermetic offline test suites.
"""
from __future__ import annotations

import datetime
import enum
import json
import logging
import threading
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger("hermes.agent.durable")


class ExecutionLifecycleState(str, enum.Enum):
    REQUEST_RECEIVED = "REQUEST_RECEIVED"
    AGENT_RUNNING = "AGENT_RUNNING"
    WAITING_FOR_TOOL = "WAITING_FOR_TOOL"
    TOOL_EXECUTED = "TOOL_EXECUTED"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    WAITING_FOR_VERIFICATION = "WAITING_FOR_VERIFICATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class RecoveryDisposition(str, enum.Enum):
    """Classification of interrupted agent state to guide safe recovery without duplicate side effects."""
    RETRY_FAILED = "RETRY_FAILED"          # Operation definitely failed; safe to retry with fresh attempt
    RECOVER_UNKNOWN = "RECOVER_UNKNOWN"    # Outcome unknown (backend crashed mid-tool); inspect idempotency key
    RESUME_COMPLETED = "RESUME_COMPLETED"  # Tool/step finished; resume next phase without re-executing


class ExecutionCheckpoint(BaseModel):
    checkpoint_id: str
    execution_id: str
    step_index: int
    state_from: Optional[str] = None
    state_to: str
    checkpoint_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "execution_id": self.execution_id,
            "step_index": self.step_index,
            "state_from": self.state_from,
            "state_to": self.state_to,
            "checkpoint_payload": self.checkpoint_payload,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else str(self.created_at),
        }


class DurableExecutionRecord(BaseModel):
    execution_id: str
    session_id: str
    user_id: str
    tenant_id: str = "default"
    status: ExecutionLifecycleState = ExecutionLifecycleState.REQUEST_RECEIVED
    current_step: int = 0
    state_payload: dict[str, Any] = Field(default_factory=dict)
    last_heartbeat: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    error: Optional[str] = None
    recovery_disposition: Optional[RecoveryDisposition] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "status": self.status.value if isinstance(self.status, ExecutionLifecycleState) else str(self.status),
            "current_step": self.current_step,
            "state_payload": self.state_payload,
            "last_heartbeat": self.last_heartbeat.isoformat() if hasattr(self.last_heartbeat, "isoformat") else str(self.last_heartbeat),
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else str(self.created_at),
            "updated_at": self.updated_at.isoformat() if hasattr(self.updated_at, "isoformat") else str(self.updated_at),
            "error": self.error,
            "recovery_disposition": self.recovery_disposition.value if self.recovery_disposition else None,
        }


class DurableExecutionStore:
    """Authoritative storage manager for agent executions and step checkpoints."""

    def __init__(self):
        self._pool = None
        self._is_postgres_ready = False
        self._lock = threading.RLock()

        # In-memory structures for testing and offline fallback
        self._mem_executions: dict[str, dict[str, Any]] = {}
        self._mem_checkpoints: dict[str, list[dict[str, Any]]] = {}

    def init_db(self) -> None:
        """Initialize PostgreSQL tables for executions and state checkpoints."""
        try:
            import psycopg
            from psycopg_pool import ConnectionPool

            pg_url = settings.postgres_url
            if not pg_url:
                logger.warning("No POSTGRES_URL configured; using in-memory durable store.")
                return

            self._pool = ConnectionPool(
                conninfo=pg_url,
                min_size=1,
                max_size=10,
                timeout=5.0,
                open=False,
            )
            self._pool.open()

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS hermes_executions (
                            execution_id VARCHAR(128) PRIMARY KEY,
                            session_id VARCHAR(128) NOT NULL,
                            user_id VARCHAR(128) NOT NULL,
                            tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                            status VARCHAR(64) NOT NULL DEFAULT 'REQUEST_RECEIVED',
                            current_step INT NOT NULL DEFAULT 0,
                            last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            state_payload JSONB,
                            error TEXT,
                            recovery_disposition VARCHAR(64),
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_hermes_exec_status ON hermes_executions(status, last_heartbeat);
                        CREATE INDEX IF NOT EXISTS idx_hermes_exec_tenant ON hermes_executions(tenant_id, session_id);

                        CREATE TABLE IF NOT EXISTS hermes_execution_checkpoints (
                            checkpoint_id VARCHAR(128) PRIMARY KEY,
                            execution_id VARCHAR(128) NOT NULL REFERENCES hermes_executions(execution_id) ON DELETE CASCADE,
                            step_index INT NOT NULL,
                            state_from VARCHAR(64) NOT NULL,
                            state_to VARCHAR(64) NOT NULL,
                            checkpoint_payload JSONB NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_hermes_checkpoints_exec ON hermes_execution_checkpoints(execution_id, step_index DESC);
                    """)
                conn.commit()

            self._is_postgres_ready = True
            logger.info("Hermes Durable Execution & Checkpoint tables initialized successfully.")
        except Exception as exc:
            logger.warning(f"PostgreSQL init for durable execution store failed: {exc}. Operating with in-memory store.")

    def close(self) -> None:
        if self._pool:
            try:
                self._pool.close()
            except Exception:
                pass

    def create_execution(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str = "default",
        initial_payload: Optional[dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> DurableExecutionRecord:
        """Create and register an authoritative execution record."""
        now = datetime.datetime.now(datetime.timezone.utc)
        exec_id = execution_id or f"exec_{uuid.uuid4().hex[:12]}"
        payload = initial_payload or {}

        record = DurableExecutionRecord(
            execution_id=exec_id,
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            status=ExecutionLifecycleState.REQUEST_RECEIVED,
            current_step=0,
            state_payload=payload,
            last_heartbeat=now,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._mem_executions[exec_id] = record.to_dict()
            self._mem_checkpoints[exec_id] = []

        if self._is_postgres_ready and self._pool:
            try:
                clean_payload = json.dumps(payload).replace("\\u0000", "").replace("\x00", "")
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO hermes_executions (
                                execution_id, session_id, user_id, tenant_id,
                                status, current_step, last_heartbeat, state_payload, created_at, updated_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                            ON CONFLICT (execution_id) DO UPDATE SET
                                status = EXCLUDED.status,
                                state_payload = EXCLUDED.state_payload,
                                updated_at = EXCLUDED.updated_at
                            """,
                            (
                                record.execution_id,
                                record.session_id,
                                record.user_id,
                                record.tenant_id,
                                record.status.value,
                                record.current_step,
                                record.last_heartbeat,
                                clean_payload,
                                record.created_at,
                                record.updated_at,
                            ),
                        )
                    conn.commit()
            except Exception as exc:
                logger.error("Error creating execution in PostgreSQL: %s", exc)

        return record

    def record_transition(
        self,
        execution_id: str,
        state_from: Optional[str],
        state_to: str,
        state_payload: dict[str, Any],
        error: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> ExecutionCheckpoint:
        """Record an atomic state transition and write a point-in-time checkpoint."""
        now = datetime.datetime.now(datetime.timezone.utc)
        checkpoint_id = f"chk_{uuid.uuid4().hex[:10]}"

        with self._lock:
            exec_dict = self._mem_executions.get(execution_id)
            if not exec_dict:
                # Auto-create if missing
                new_rec = self.create_execution(
                    session_id=session_id or "unknown",
                    user_id=user_id or "unknown",
                    tenant_id=tenant_id or "default",
                    execution_id=execution_id,
                )
                exec_dict = self._mem_executions.get(execution_id)

            step_idx = exec_dict.get("current_step", 0) + 1
            exec_dict["current_step"] = step_idx
            exec_dict["status"] = state_to
            exec_dict["state_payload"] = state_payload
            exec_dict["last_heartbeat"] = now.isoformat()
            exec_dict["updated_at"] = now.isoformat()
            if error:
                exec_dict["error"] = error

            checkpoint = ExecutionCheckpoint(
                checkpoint_id=checkpoint_id,
                execution_id=execution_id,
                step_index=step_idx,
                state_from=state_from,
                state_to=state_to,
                checkpoint_payload=state_payload,
                created_at=now,
            )
            self._mem_checkpoints.setdefault(execution_id, []).append(checkpoint.to_dict())

        if self._is_postgres_ready and self._pool:
            try:
                clean_payload = json.dumps(state_payload).replace("\\u0000", "").replace("\x00", "")
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE hermes_executions
                            SET status = %s,
                                current_step = %s,
                                state_payload = %s::jsonb,
                                last_heartbeat = %s,
                                updated_at = %s,
                                error = %s
                            WHERE execution_id = %s
                            """,
                            (state_to, step_idx, clean_payload, now, now, error, execution_id),
                        )
                        cur.execute(
                            """
                            INSERT INTO hermes_execution_checkpoints (
                                checkpoint_id, execution_id, step_index, state_from, state_to, checkpoint_payload, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                            """,
                            (checkpoint_id, execution_id, step_idx, state_from, state_to, clean_payload, now),
                        )
                    conn.commit()
            except Exception as exc:
                logger.error("Error recording transition in PostgreSQL: %s", exc)

        return checkpoint

    def heartbeat(self, execution_id: str) -> None:
        """Update last heartbeat timestamp for active execution."""
        now = datetime.datetime.now(datetime.timezone.utc)
        with self._lock:
            if execution_id in self._mem_executions:
                self._mem_executions[execution_id]["last_heartbeat"] = now.isoformat()

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE hermes_executions SET last_heartbeat = %s WHERE execution_id = %s",
                            (now, execution_id),
                        )
                    conn.commit()
            except Exception as exc:
                logger.debug("Heartbeat update failed: %s", exc)

    def get_execution(self, execution_id: str) -> Optional[DurableExecutionRecord]:
        """Fetch execution record by ID."""
        with self._lock:
            d = self._mem_executions.get(execution_id)
            if d:
                return DurableExecutionRecord(**d)

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT execution_id, session_id, user_id, tenant_id, status,
                                   current_step, state_payload, last_heartbeat, created_at, updated_at,
                                   error, recovery_disposition
                            FROM hermes_executions WHERE execution_id = %s
                            """,
                            (execution_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            return DurableExecutionRecord(
                                execution_id=row[0],
                                session_id=row[1],
                                user_id=row[2],
                                tenant_id=row[3],
                                status=ExecutionLifecycleState(row[4]),
                                current_step=row[5],
                                state_payload=row[6] or {},
                                last_heartbeat=row[7],
                                created_at=row[8],
                                updated_at=row[9],
                                error=row[10],
                                recovery_disposition=RecoveryDisposition(row[11]) if row[11] else None,
                            )
            except Exception as exc:
                logger.error("Error fetching execution from PostgreSQL: %s", exc)
        return None

    def list_executions(
        self,
        status: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[DurableExecutionRecord]:
        """List execution records with optional status and tenant filters."""
        res: list[DurableExecutionRecord] = []
        with self._lock:
            for d in self._mem_executions.values():
                if status and d.get("status") != status:
                    continue
                if tenant_id and d.get("tenant_id") != tenant_id:
                    continue
                res.append(DurableExecutionRecord(**d))
        return res[:limit]

    def get_checkpoints(self, execution_id: str) -> list[ExecutionCheckpoint]:
        """Fetch all chronological checkpoints for an execution."""
        with self._lock:
            chk_list = self._mem_checkpoints.get(execution_id, [])
            if chk_list:
                return [ExecutionCheckpoint(**c) for c in chk_list]

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT checkpoint_id, execution_id, step_index, state_from, state_to,
                                   checkpoint_payload, created_at
                            FROM hermes_execution_checkpoints
                            WHERE execution_id = %s
                            ORDER BY step_index ASC
                            """,
                            (execution_id,),
                        )
                        rows = cur.fetchall()
                        return [
                            ExecutionCheckpoint(
                                checkpoint_id=r[0],
                                execution_id=r[1],
                                step_index=r[2],
                                state_from=r[3],
                                state_to=r[4],
                                checkpoint_payload=r[5] or {},
                                created_at=r[6],
                            )
                            for r in rows
                        ]
            except Exception as exc:
                logger.error("Error fetching checkpoints from PostgreSQL: %s", exc)
        return []

    def get_latest_checkpoint(self, execution_id: str) -> Optional[ExecutionCheckpoint]:
        """Fetch the most recent checkpoint for an execution."""
        with self._lock:
            chk_list = self._mem_checkpoints.get(execution_id, [])
            if chk_list:
                return ExecutionCheckpoint(**chk_list[-1])

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT checkpoint_id, execution_id, step_index, state_from, state_to,
                                   checkpoint_payload, created_at
                            FROM hermes_execution_checkpoints
                            WHERE execution_id = %s
                            ORDER BY step_index DESC LIMIT 1
                            """,
                            (execution_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            return ExecutionCheckpoint(
                                checkpoint_id=row[0],
                                execution_id=row[1],
                                step_index=row[2],
                                state_from=row[3],
                                state_to=row[4],
                                checkpoint_payload=row[5] or {},
                                created_at=row[6],
                            )
            except Exception as exc:
                logger.error("Error fetching checkpoint from PostgreSQL: %s", exc)
        return None

    def determine_recovery_disposition(self, execution: DurableExecutionRecord) -> RecoveryDisposition:
        """Classify the interrupted state to choose the exact safe recovery strategy."""
        status = execution.status

        # 1. Definite failure: safe to retry with fresh attempt
        if status == ExecutionLifecycleState.FAILED:
            return RecoveryDisposition.RETRY_FAILED

        # 2. Crash during tool execution: outcome is unknown
        if status == ExecutionLifecycleState.WAITING_FOR_TOOL:
            return RecoveryDisposition.RECOVER_UNKNOWN

        # 3. Step was executed or ready for verification: resume without re-running side effects
        if status in (
            ExecutionLifecycleState.TOOL_EXECUTED,
            ExecutionLifecycleState.WAITING_FOR_VERIFICATION,
            ExecutionLifecycleState.WAITING_FOR_APPROVAL,
        ):
            return RecoveryDisposition.RESUME_COMPLETED

        # Default for running/interrupted
        return RecoveryDisposition.RECOVER_UNKNOWN


durable_store = DurableExecutionStore()
