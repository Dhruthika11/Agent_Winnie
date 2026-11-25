"""
Workflow Execution Telemetry and Baseline Optimization Tracker.
Measures:
- Number of LLM calls per workflow
- Number of tool calls
- Unnecessary retries
- Total latency (ms)
- Cost (USD) per successful task
Provides baseline measurements and detects optimization opportunities.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional
from pydantic import BaseModel, Field

from src.workflows.schemas import ExecutionStatus, WorkflowState

logger = logging.getLogger("hermes.workflows.optimization.telemetry")


class WorkflowBaselineMetrics(BaseModel):
    workflow_type: str
    total_executions: int = 0
    successful_executions: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_llm_calls: float = 0.0
    avg_tool_calls: float = 0.0
    avg_retries: float = 0.0
    avg_cost_usd: float = 0.0
    cost_per_successful_task_usd: float = 0.0


class OptimizationOpportunity(BaseModel):
    workflow_type: str
    metric: str
    current_value: float
    threshold: float
    recommendation: str


class WorkflowOptimizationTracker:
    """Tracks and aggregates operational telemetry to establish optimization baselines."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []

    def record_execution(self, state: WorkflowState) -> None:
        """Record completed workflow state execution metrics."""
        wf_type = state.get("workflow_type", "unknown")
        status = state.get("execution_status", "unknown")
        metrics = state.get("metrics", {})

        entry = {
            "workflow_id": state.get("workflow_id"),
            "workflow_type": wf_type,
            "status": status,
            "duration_ms": metrics.get("duration_ms", 0.0),
            "llm_calls": metrics.get("llm_calls", 0),
            "tool_calls": metrics.get("tool_calls", 0),
            "retries": metrics.get("retries", 0),
            "cost_usd": metrics.get("cost_usd", 0.0),
        }

        with self._lock:
            self._records.append(entry)
            if len(self._records) > 2000:
                self._records = self._records[-2000:]

    def get_baseline_metrics(self, workflow_type: Optional[str] = None) -> dict[str, WorkflowBaselineMetrics]:
        """Compute baseline measurements grouped by workflow type."""
        with self._lock:
            records = list(self._records)

        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            wf = r["workflow_type"]
            if workflow_type and wf != workflow_type:
                continue
            grouped.setdefault(wf, []).append(r)

        baselines: dict[str, WorkflowBaselineMetrics] = {}
        for wf, recs in grouped.items():
            tot = len(recs)
            succ = sum(1 for r in recs if r["status"] == ExecutionStatus.COMPLETED.value)
            rate = round((succ / max(tot, 1)) * 100, 2)

            avg_lat = round(sum(r["duration_ms"] for r in recs) / max(tot, 1), 2)
            avg_llm = round(sum(r["llm_calls"] for r in recs) / max(tot, 1), 2)
            avg_tool = round(sum(r["tool_calls"] for r in recs) / max(tot, 1), 2)
            avg_ret = round(sum(r["retries"] for r in recs) / max(tot, 1), 2)
            avg_cost = round(sum(r["cost_usd"] for r in recs) / max(tot, 1), 4)

            total_cost = sum(r["cost_usd"] for r in recs)
            cost_per_succ = round(total_cost / max(succ, 1), 4)

            baselines[wf] = WorkflowBaselineMetrics(
                workflow_type=wf,
                total_executions=tot,
                successful_executions=succ,
                success_rate=rate,
                avg_latency_ms=avg_lat,
                avg_llm_calls=avg_llm,
                avg_tool_calls=avg_tool,
                avg_retries=avg_ret,
                avg_cost_usd=avg_cost,
                cost_per_successful_task_usd=cost_per_succ,
            )

        return baselines

    def detect_optimization_opportunities(self) -> list[OptimizationOpportunity]:
        """Analyze baseline metrics against reliability and cost thresholds."""
        baselines = self.get_baseline_metrics()
        opps: list[OptimizationOpportunity] = []

        for wf, b in baselines.items():
            # Check excessive retries
            if b.avg_retries > 1.5:
                opps.append(
                    OptimizationOpportunity(
                        workflow_type=wf,
                        metric="avg_retries",
                        current_value=b.avg_retries,
                        threshold=1.5,
                        recommendation="Improve initial prompt context or schema validation to reduce retry loops.",
                    )
                )
            # Check high latency
            if b.avg_latency_ms > 10000.0:
                opps.append(
                    OptimizationOpportunity(
                        workflow_type=wf,
                        metric="avg_latency_ms",
                        current_value=b.avg_latency_ms,
                        threshold=10000.0,
                        recommendation="Introduce parallel batching or smaller model tier for fast-path nodes.",
                    )
                )
            # Check high cost per successful task
            if b.cost_per_successful_task_usd > 0.05:
                opps.append(
                    OptimizationOpportunity(
                        workflow_type=wf,
                        metric="cost_per_successful_task_usd",
                        current_value=b.cost_per_successful_task_usd,
                        threshold=0.05,
                        recommendation="Delegate extraction and classification to fast_cheap model tier (Tier 1).",
                    )
                )

        return opps


workflow_optimization_tracker = WorkflowOptimizationTracker()
