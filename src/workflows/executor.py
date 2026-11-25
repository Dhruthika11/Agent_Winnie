"""
Workflow Executor and Orchestrator for Hermes Agent Platform.

Executes workflow graphs, integrates FinOps performance tracking, manages step history,
and records comprehensive telemetry across all 6 specialized workflows.
"""
from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Optional

from src.security.finops import calculate_cost_usd
from src.workflows.base import create_initial_workflow_state
from src.workflows.registry import classify_workflow, get_workflow_graph
from src.workflows.schemas import ExecutionStatus, WorkflowDecision, WorkflowState

logger = logging.getLogger("hermes.workflows.executor")


class WorkflowExecutor:
    """Centralized execution engine for running and observing specialized LangGraph workflows."""

    def __init__(self):
        pass

    def run(
        self,
        user_request: str,
        workflow_type: Optional[str] = None,
        user_id: str = "default_user",
        session_id: str = "default_session",
        tenant_id: str = "default",
        max_retries: int = 3,
        workflow_data: Optional[dict[str, Any]] = None,
    ) -> WorkflowState:
        """
        Execute a workflow to completion:
        1. Classify request if workflow_type is not provided.
        2. Resolve compiled LangGraph graph.
        3. Initialize common state contract.
        4. Invoke graph and track performance, latency, tokens, and FinOps costs.
        """
        start_time = time.perf_counter()

        if workflow_type:
            decision = WorkflowDecision(
                workflow=workflow_type.lower(),
                confidence=1.0,
                reason="Explicit workflow override specified.",
            )
        else:
            decision = classify_workflow(user_request)

        logger.info(
            "workflow_routed",
            extra={
                "request": user_request[:100],
                "selected_workflow": decision.workflow,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
        )

        graph = get_workflow_graph(decision.workflow)

        initial_state = create_initial_workflow_state(
            objective=user_request,
            workflow_type=decision.workflow,
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            max_retries=max_retries,
            workflow_data=workflow_data or {},
        )
        initial_state["execution_status"] = ExecutionStatus.RUNNING.value

        # Execute LangGraph workflow graph
        try:
            final_state = graph.invoke(initial_state)
        except Exception as exc:
            logger.exception("workflow_execution_error", extra={"workflow": decision.workflow})
            initial_state["execution_status"] = ExecutionStatus.FAILED.value
            initial_state["errors"].append(f"Fatal workflow exception: {type(exc).__name__}: {exc}")
            final_state = initial_state

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Update metrics
        metrics = final_state.get("metrics", {})
        metrics["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metrics["duration_ms"] = round(duration_ms, 2)

        # Estimate FinOps cost based on LLM calls
        llm_calls = metrics.get("llm_calls", 1)
        est_prompt_tokens = llm_calls * 450
        est_completion_tokens = llm_calls * 350
        metrics["tokens"] = est_prompt_tokens + est_completion_tokens
        metrics["cost_usd"] = calculate_cost_usd(
            "qwen-2.5-72b-instruct", est_prompt_tokens, est_completion_tokens
        )
        final_state["metrics"] = metrics

        # Record real-time trace in observability engine and baseline optimization tracker
        try:
            from src.workflows.observability import workflow_observability_engine
            workflow_observability_engine.record_trace_from_state(final_state)
            from src.workflows.optimization.telemetry import workflow_optimization_tracker
            workflow_optimization_tracker.record_execution(final_state)
        except Exception:
            pass

        logger.info(
            "workflow_execution_completed",
            extra={
                "workflow": decision.workflow,
                "status": final_state.get("execution_status"),
                "duration_ms": duration_ms,
                "steps_count": len(final_state.get("step_history", [])),
            },
        )
        return final_state


workflow_executor = WorkflowExecutor()
