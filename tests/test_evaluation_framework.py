"""
Tests for Comprehensive Agent Evaluation Framework.
Verifies:
- Benchmark dataset schema and 120 curated test cases across 6 workflows.
- Evaluation of the 5 core dimensions: Task completion, Reasoning/planning, Tool execution, Grounding, Workflow reliability.
- Evaluation report generation with domain breakdowns and FinOps telemetry.
"""
from __future__ import annotations

import pytest

from src.evaluation.benchmark_data import load_benchmark_dataset
from src.evaluation.evaluators import (
    GroundingEvaluator,
    ReasoningPlanningEvaluator,
    TaskCompletionEvaluator,
    ToolExecutionEvaluator,
    WorkflowReliabilityEvaluator,
)
from src.evaluation.framework import agent_evaluation_framework
from src.workflows.base import create_initial_workflow_state, record_step


def test_benchmark_dataset_contains_120_curated_cases_with_proper_distribution():
    dataset = load_benchmark_dataset()
    assert len(dataset) == 120

    counts = {}
    for case in dataset:
        counts[case.expected_workflow] = counts.get(case.expected_workflow, 0) + 1
        assert case.test_id
        assert case.input
        assert len(case.required_capabilities) > 0
        assert len(case.success_criteria) > 0

    assert counts.get("research") == 25
    assert counts.get("coding") == 25
    assert counts.get("data_analysis") == 20
    assert counts.get("document") == 20
    assert counts.get("diagnostics") == 20
    assert counts.get("autonomous") == 10


def test_five_evaluation_categories_scoring():
    dataset = load_benchmark_dataset()
    case = next(c for c in dataset if c.expected_workflow == "research")

    state = create_initial_workflow_state(
        objective=case.input,
        workflow_type="research",
        workflow_data={
            "grounded_synthesis": "Verified synthesis of research.",
            "evidence_items": [{"url": "https://arxiv.org/1", "source": "Arxiv", "content": "Evidence text"}],
            "raw_search_results": [{"status": "success", "output": "Search result"}],
            "citations": ["https://arxiv.org/1"],
        },
    )
    record_step(state, "query_decomposition", "completed", "Decomposed into subquestions", 25.0)
    record_step(state, "parallel_search", "completed", "Executed searches", 120.0)
    record_step(state, "grounded_synthesis", "completed", "Generated synthesis", 80.0)
    state["verification_status"] = "verified"
    state["execution_status"] = "completed"
    state["final_output"] = {"report_markdown": "# Research Report\nVerified facts."}
    state["metrics"]["tool_calls"] = 2
    state["metrics"]["llm_calls"] = 2

    # 1. Task Completion
    task_eval = TaskCompletionEvaluator()
    score_task = task_eval.evaluate(case, state)
    assert score_task.passed is True
    assert score_task.score >= 0.70

    # 2. Reasoning & Planning
    reason_eval = ReasoningPlanningEvaluator()
    score_reason = reason_eval.evaluate(case, state)
    assert score_reason.passed is True
    assert score_reason.score >= 0.70

    # 3. Tool Execution
    tool_eval = ToolExecutionEvaluator()
    score_tool = tool_eval.evaluate(case, state)
    assert score_tool.passed is True
    assert score_tool.score >= 0.70

    # 4. Grounding
    ground_eval = GroundingEvaluator()
    score_ground = ground_eval.evaluate(case, state)
    assert score_ground.passed is True
    assert score_ground.score >= 0.70

    # 5. Workflow Reliability
    rel_eval = WorkflowReliabilityEvaluator()
    score_rel = rel_eval.evaluate(case, state)
    assert score_rel.passed is True
    assert score_rel.score >= 0.70


def test_framework_benchmark_execution_and_summary_report():
    dataset = load_benchmark_dataset()
    sample_cases = [
        next(c for c in dataset if c.expected_workflow == "coding"),
        next(c for c in dataset if c.expected_workflow == "data_analysis"),
        next(c for c in dataset if c.expected_workflow == "document"),
    ]

    report = agent_evaluation_framework.run_benchmark(cases=sample_cases)

    assert report.total_cases == 3
    assert report.passed_cases == 3
    assert report.pass_rate == 100.0
    assert report.overall_avg_score >= 0.70
    assert len(report.category_averages) == 5
    assert len(report.workflow_breakdown) == 3
    assert report.total_execution_time_ms >= 0.0
