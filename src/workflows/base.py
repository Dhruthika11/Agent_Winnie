"""
Base Workflow Infrastructure, State Initialization, and Telemetry Utilities.
"""
from __future__ import annotations

import datetime
import time
import uuid
from typing import Any, Optional

from src.workflows.schemas import ExecutionStatus, WorkflowState, WorkflowType


def create_initial_workflow_state(
    objective: str,
    workflow_type: WorkflowType | str,
    user_id: str = "default_user",
    session_id: str = "default_session",
    tenant_id: str = "default",
    max_retries: int = 3,
    workflow_data: Optional[dict[str, Any]] = None,
) -> WorkflowState:
    """Factory creating a well-formed initial WorkflowState instance."""
    wf_type = workflow_type.value if isinstance(workflow_type, WorkflowType) else str(workflow_type)
    wf_id = f"wf_{wf_type}_{uuid.uuid4().hex[:10]}"

    return {
        "workflow_id": wf_id,
        "workflow_type": wf_type,
        "user_id": user_id,
        "session_id": session_id,
        "tenant_id": tenant_id,
        "objective": objective.strip(),
        "plan": [],
        "current_step": "initialize",
        "tool_results": [],
        "evidence": [],
        "errors": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "approval_status": "none",
        "approval_id": None,
        "verification_status": "unverified",
        "final_output": {},
        "execution_status": ExecutionStatus.PENDING.value,
        "step_history": [],
        "metrics": {
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "completed_at": None,
            "duration_ms": 0.0,
            "llm_calls": 0,
            "tool_calls": 0,
            "retry_frequency": 0,
            "cost_usd": 0.0,
            "human_interventions": 0,
        },
        "workflow_data": workflow_data or {},
    }


def record_step(
    state: WorkflowState,
    step_name: str,
    status: str = "completed",
    summary: Optional[str] = None,
    error: Optional[str] = None,
    duration_ms: float = 0.0,
) -> dict[str, Any]:
    """Helper to record step transitions into step_history and update current_step."""
    record = {
        "step_name": step_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "summary": summary,
        "error": error,
    }
    history = list(state.get("step_history", []))
    history.append(record)
    state["step_history"] = history
    state["current_step"] = step_name
    return record


def increment_metric(state: WorkflowState, key: str, amount: int | float = 1):
    """Safely increment a workflow telemetry metric."""
    metrics = state.get("metrics", {})
    metrics[key] = metrics.get(key, 0) + amount
    state["metrics"] = metrics
