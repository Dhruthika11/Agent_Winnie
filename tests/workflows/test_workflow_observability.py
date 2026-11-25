"""
Tests for Workflow Observability, Traces, and Dashboard Monitor.
"""
from __future__ import annotations

import pytest

from src.workflows.base import create_initial_workflow_state, record_step
from src.workflows.observability import workflow_observability_engine


def test_observability_records_trace_from_state():
    state = create_initial_workflow_state(
        objective="Analyze performance benchmark dataset",
        workflow_type="data_analysis",
    )
    record_step(state, "inspect_schema", status="completed", summary="Inspected 5 columns", duration_ms=42.0)
    record_step(state, "detect_traps", status="completed", summary="No correlation traps found", duration_ms=68.0)
    record_step(state, "compute_stats", status="completed", summary="Computed Welch's t-test", duration_ms=110.0)
    state["execution_status"] = "completed"
    state["metrics"]["tool_calls"] = 2
    state["metrics"]["llm_calls"] = 1
    state["metrics"]["cost_usd"] = 0.004

    trace = workflow_observability_engine.record_trace_from_state(state)

    assert trace.trace_id.startswith("tr_")
    assert trace.workflow_type == "data_analysis"
    assert len(trace.nodes) == 3
    assert trace.nodes[0].node_name == "Inspect Schema"
    assert trace.nodes[0].duration_ms == 42.0
    assert trace.nodes[0].status == "completed"

    # Verify retrieval
    retrieved = workflow_observability_engine.get_trace(trace.trace_id)
    assert retrieved is not None
    assert retrieved.trace_id == trace.trace_id


def test_observability_platform_summary_and_html_dashboard():
    summary = workflow_observability_engine.get_platform_summary()
    assert summary.task_success_rate > 0.0
    assert summary.avg_latency_seconds >= 0.0
    assert summary.avg_tool_calls >= 0.0
    assert summary.avg_cost_per_request_usd >= 0.0

    html = workflow_observability_engine.generate_html_dashboard()
    assert "Hermes Workflow Execution Monitor" in html
    assert "Task Success Rate" in html
    assert "Average Latency" in html
    assert "Recent Workflow Execution Traces" in html
