"""
Tests for Workflow Optimization, Adaptive Model Routing, and Baseline Tracking.
"""
from __future__ import annotations

import pytest

from src.evaluation.benchmark_data import load_benchmark_dataset
from src.workflows.base import create_initial_workflow_state
from src.workflows.optimization.adaptive_router import (
    ModelTier,
    TaskType,
    adaptive_model_router,
    workflow_selection_evaluator,
)
from src.workflows.optimization.telemetry import (
    WorkflowOptimizationTracker,
    workflow_optimization_tracker,
)


def test_adaptive_model_router_assigns_correct_tiers():
    # Tier 1 assignments (fast, lightweight routing and classification)
    t1_sel = adaptive_model_router.route_task(TaskType.WORKFLOW_SELECTION)
    assert t1_sel.model_tier == ModelTier.FAST_CHEAP
    assert "gpt-oss-20b" in t1_sel.model_name or "groq" in t1_sel.model_name
    assert t1_sel.estimated_cost_per_1k_tokens > 0

    # Tier 2 assignments (coding uses specialized syntax model, reasoning uses flagship 120B)
    t2_code = adaptive_model_router.route_task(TaskType.CODE_GENERATION)
    assert t2_code.model_tier == ModelTier.ADVANCED_REASONING
    assert "qwen" in t2_code.model_name

    t2_reason = adaptive_model_router.route_task(TaskType.COMPLEX_REASONING)
    assert t2_reason.model_tier == ModelTier.ADVANCED_REASONING
    assert "120b" in t2_reason.model_name

    # Tier 3 assignments (cross-provider independent verifier)
    t3_sel = adaptive_model_router.route_task(TaskType.GROUNDING_VERIFICATION)
    assert t3_sel.model_tier == ModelTier.INDEPENDENT_VERIFIER
    assert "gemini" in t3_sel.model_name or "verifier" in t3_sel.model_tier.value


def test_workflow_selection_evaluator_against_benchmark_dataset():
    dataset = load_benchmark_dataset()
    # Evaluate across a representative slice of 30 cases (5 from each of the 6 categories)
    slice_cases = []
    seen = {}
    for c in dataset:
        wf = c.expected_workflow
        if seen.get(wf, 0) < 5:
            slice_cases.append(c)
            seen[wf] = seen.get(wf, 0) + 1

    report = workflow_selection_evaluator.evaluate_benchmark(cases=slice_cases)

    assert report.total_samples == 30
    assert report.classification_accuracy >= 85.0
    assert len(report.per_workflow_accuracy) == 6


def test_workflow_optimization_tracker_baselines_and_alerts():
    tracker = WorkflowOptimizationTracker()

    # Simulate completed states
    for i in range(5):
        st = create_initial_workflow_state(
            objective=f"Test {i}",
            workflow_type="coding",
        )
        st["execution_status"] = "completed"
        st["metrics"]["duration_ms"] = 1200.0
        st["metrics"]["llm_calls"] = 2
        st["metrics"]["tool_calls"] = 3
        st["metrics"]["retries"] = 0
        st["metrics"]["cost_usd"] = 0.008
        tracker.record_execution(st)

    baselines = tracker.get_baseline_metrics()
    assert "coding" in baselines
    b = baselines["coding"]
    assert b.total_executions == 5
    assert b.successful_executions == 5
    assert b.success_rate == 100.0
    assert b.avg_llm_calls == 2.0
    assert b.avg_tool_calls == 3.0
    assert b.cost_per_successful_task_usd == 0.008
