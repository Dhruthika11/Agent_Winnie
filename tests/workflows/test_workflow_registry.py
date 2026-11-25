"""
Unit Tests for Workflow Registry and Classifier.
"""
from __future__ import annotations

import pytest

from src.workflows.registry import (
    WORKFLOW_REGISTRY,
    classify_workflow,
    get_workflow_graph,
    list_available_workflows,
)
from src.workflows.schemas import WorkflowType


def test_registry_contains_all_six_workflows():
    workflows = list_available_workflows()
    expected = ["research", "coding", "data_analysis", "document", "diagnostics", "autonomous"]
    for exp in expected:
        assert exp in workflows
        assert get_workflow_graph(exp) is not None


@pytest.mark.parametrize(
    "prompt,expected_workflow",
    [
        (
            "Research recent advancements in transformer attention mechanisms and check clinical trials",
            WorkflowType.RESEARCH.value,
        ),
        (
            "Write a Python script to process this dataset and check whether it works",
            WorkflowType.CODING.value,
        ),
        (
            "Analyze this CSV and tell me whether the two groups are statistically different",
            WorkflowType.DATA_ANALYSIS.value,
        ),
        (
            "Create a detailed technical report from these research results with tables and references",
            WorkflowType.DOCUMENT.value,
        ),
        (
            "Why is my MCP server failing? Check logs and troubleshoot the error",
            WorkflowType.DIAGNOSTICS.value,
        ),
        (
            "Research three open-source vector databases, compare their performance and generate a report",
            WorkflowType.AUTONOMOUS.value,
        ),
    ],
)
def test_workflow_classification_routes_accurately(prompt, expected_workflow):
    decision = classify_workflow(prompt)
    assert decision.workflow == expected_workflow
    assert decision.confidence >= 0.90
    assert len(decision.reason) > 10
