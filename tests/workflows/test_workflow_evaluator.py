"""
Tests for Workflow-Level Evaluation Subsystem.
"""
from __future__ import annotations

from unittest.mock import patch
import pytest

from src.workflows.evaluator import workflow_evaluator
from src.workflows.executor import workflow_executor
from src.workflows.schemas import WorkflowType


def test_evaluator_scores_all_six_workflows_and_measures_operational_telemetry():
    """
    Execute each of the 6 specialized workflows and evaluate quality, accuracy,
    and operational efficiency metrics (LLM calls, tool calls, latency, cost USD).
    """
    scenarios = [
        (
            WorkflowType.RESEARCH.value,
            "Research CardioFix clinical efficacy and patient outcomes",
            {
                "raw_search_results": [
                    {
                        "subquestion_id": "sub_1",
                        "query": "CardioFix",
                        "status": "success",
                        "output": "Study data 34%",
                        "evidence_item": {"url": "https://clinicaltrials.gov/s1", "source": "S1", "content": "Study data 34%"},
                    }
                ]
            },
        ),
        (
            WorkflowType.CODING.value,
            "Write a Python script to compute average dataset values",
            {},
        ),
        (
            WorkflowType.DATA_ANALYSIS.value,
            "Analyze this CSV and tell me whether two groups are statistically different",
            {},
        ),
        (
            WorkflowType.DOCUMENT.value,
            "Create a detailed technical report on Project Orion architecture",
            {},
        ),
        (
            WorkflowType.DIAGNOSTICS.value,
            "Why is my MCP server failing? Collect logs and fix it",
            {},
        ),
        (
            WorkflowType.AUTONOMOUS.value,
            "Research three open-source vector databases, compare their performance and generate a report",
            {},
        ),
    ]

    for wf_type, request_text, custom_data in scenarios:
        with patch("src.tools.execution_engine.tool_execution_engine.execute") as mock_exec:
            mock_exec.return_value.status = "success"
            mock_exec.return_value.output = "Mock tool output data"
            final_state = workflow_executor.run(
                user_request=request_text,
                workflow_type=wf_type,
                workflow_data=custom_data,
            )

        assert final_state["execution_status"] in ("completed", "running")

        # Evaluate workflow
        eval_metrics = workflow_evaluator.evaluate(final_state)

        assert eval_metrics.workflow_type == wf_type
        assert eval_metrics.overall_score >= 0.70
        assert eval_metrics.passed is True

        # Check common operational telemetry
        assert eval_metrics.execution_time_ms >= 0.0
        assert eval_metrics.llm_calls_count >= 1
        assert eval_metrics.cost_usd >= 0.0
        assert len(eval_metrics.domain_scores) >= 3

        # Validate domain-specific scores
        if wf_type == WorkflowType.RESEARCH.value:
            assert "evidence_accuracy" in eval_metrics.domain_scores
            assert "citation_correctness" in eval_metrics.domain_scores
        elif wf_type == WorkflowType.CODING.value:
            assert "test_pass_rate" in eval_metrics.domain_scores
            assert "sandbox_isolation_score" in eval_metrics.domain_scores
        elif wf_type == WorkflowType.DATA_ANALYSIS.value:
            assert "numerical_accuracy" in eval_metrics.domain_scores
            assert "trap_detection_score" in eval_metrics.domain_scores
        elif wf_type == WorkflowType.DOCUMENT.value:
            assert "formatting_correctness" in eval_metrics.domain_scores
            assert "inspection_pass_rate" in eval_metrics.domain_scores
        elif wf_type == WorkflowType.DIAGNOSTICS.value:
            assert "root_cause_accuracy" in eval_metrics.domain_scores
            assert "safety_approval_enforced" in eval_metrics.domain_scores
        elif wf_type == WorkflowType.AUTONOMOUS.value:
            assert "task_completion_rate" in eval_metrics.domain_scores
            assert "plan_efficiency" in eval_metrics.domain_scores
