"""
Crash Recovery and State Reconciliation Manager for Hermes Agent Platform.

Detects abandoned, interrupted, or stale agent executions and executes safe recovery policies:
1. Periodic or on-demand scan of stale executions (heartbeat timeout).
2. Triage into exact recovery classifications:
   - RETRY_FAILED: Safely retry an operation that definitively errored out.
   - RECOVER_UNKNOWN: Operation stopped mid-tool; check idempotency keys before executing.
   - RESUME_COMPLETED: Tool/step finished; resume next phase directly from checkpoint.
3. Resumes execution via LangGraph checkpoint restore or gracefully marks interrupted.
"""
from __future__ import annotations

import datetime
import logging
from typing import Any, Optional

from src.agent.durable import (
    DurableExecutionRecord,
    ExecutionLifecycleState,
    RecoveryDisposition,
    durable_store,
)
from src.tools.execution_engine import tool_execution_engine

logger = logging.getLogger("hermes.agent.recovery")


class RecoveryManager:
    """Manages detection, triage, and reconciliation of interrupted agent workflows."""

    def reconcile_stale_executions(
        self,
        stale_threshold_seconds: float = 60.0,
        tenant_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Scan active executions, detect stalled instances, and apply appropriate recovery strategies."""
        now = datetime.datetime.now(datetime.timezone.utc)
        active_states = [
            ExecutionLifecycleState.AGENT_RUNNING.value,
            ExecutionLifecycleState.WAITING_FOR_TOOL.value,
            ExecutionLifecycleState.WAITING_FOR_VERIFICATION.value,
        ]

        stale_records: list[dict[str, Any]] = []
        candidates = durable_store.list_executions(tenant_id=tenant_id, limit=200)

        for rec in candidates:
            if rec.status.value not in active_states:
                continue

            hb = rec.last_heartbeat
            # Handle tz-aware comparison
            if hb.tzinfo is None:
                hb = hb.replace(tzinfo=datetime.timezone.utc)
            delta = (now - hb).total_seconds()

            if delta >= stale_threshold_seconds:
                disposition = durable_store.determine_recovery_disposition(rec)

                # If outcome was unknown during tool wait, check if tool succeeded in idempotency store
                if disposition == RecoveryDisposition.RECOVER_UNKNOWN:
                    pending_calls = rec.state_payload.get("pending_tool_calls", [])
                    if pending_calls:
                        tool_call_id = pending_calls[0].get("id") or rec.execution_id
                        with tool_execution_engine._lock:
                            cached = tool_execution_engine._idempotency_store.get(tool_call_id)
                        if cached and cached.status == "success":
                            disposition = RecoveryDisposition.RESUME_COMPLETED

                # Transition to INTERRUPTED state with classified recovery disposition
                durable_store.record_transition(
                    execution_id=rec.execution_id,
                    state_from=rec.status.value,
                    state_to=ExecutionLifecycleState.INTERRUPTED.value,
                    state_payload=rec.state_payload,
                    error=f"Stale execution detected: no heartbeat for {round(delta, 1)}s. Triage: {disposition.value}.",
                )

                stale_records.append({
                    "execution_id": rec.execution_id,
                    "session_id": rec.session_id,
                    "user_id": rec.user_id,
                    "tenant_id": rec.tenant_id,
                    "previous_status": rec.status.value,
                    "disposition": disposition.value,
                    "seconds_since_heartbeat": round(delta, 1),
                })

        logger.info("recovery_reconciliation_cycle_completed", extra={"stale_found": len(stale_records)})
        return stale_records

    def resume_execution(self, execution_id: str) -> dict[str, Any]:
        """
        Resume an interrupted execution from its latest checkpoint.
        Reconstructs the AgentState and invokes the next graph step.
        """
        rec = durable_store.get_execution(execution_id)
        if not rec:
            return {"success": False, "error": f"Execution '{execution_id}' not found"}

        latest_chk = durable_store.get_latest_checkpoint(execution_id)
        if not latest_chk:
            return {"success": False, "error": f"No checkpoint available for '{execution_id}'"}

        disposition = durable_store.determine_recovery_disposition(rec)
        checkpoint_payload = latest_chk.checkpoint_payload

        # Transition back to AGENT_RUNNING
        durable_store.record_transition(
            execution_id=execution_id,
            state_from=rec.status.value,
            state_to=ExecutionLifecycleState.AGENT_RUNNING.value,
            state_payload=checkpoint_payload,
        )

        try:
            from src.agent.graph import agent_graph

            res = agent_graph.invoke(checkpoint_payload)

            durable_store.record_transition(
                execution_id=execution_id,
                state_from=ExecutionLifecycleState.AGENT_RUNNING.value,
                state_to=ExecutionLifecycleState.COMPLETED.value,
                state_payload={"messages_count": len(res.get("messages", []))},
            )
            return {
                "success": True,
                "execution_id": execution_id,
                "disposition": disposition.value,
                "status": "COMPLETED",
                "result": res,
            }
        except Exception as exc:
            logger.exception("resume_execution_failed", extra={"execution_id": execution_id})
            durable_store.record_transition(
                execution_id=execution_id,
                state_from=ExecutionLifecycleState.AGENT_RUNNING.value,
                state_to=ExecutionLifecycleState.FAILED.value,
                state_payload=checkpoint_payload,
                error=str(exc),
            )
            return {"success": False, "execution_id": execution_id, "error": str(exc)}

    def cancel_execution(self, execution_id: str, reason: str = "Operator cancelled") -> bool:
        """Safely mark an execution as terminated."""
        rec = durable_store.get_execution(execution_id)
        if not rec:
            return False

        durable_store.record_transition(
            execution_id=execution_id,
            state_from=rec.status.value,
            state_to=ExecutionLifecycleState.FAILED.value,
            state_payload=rec.state_payload,
            error=reason,
        )
        return True


recovery_manager = RecoveryManager()
