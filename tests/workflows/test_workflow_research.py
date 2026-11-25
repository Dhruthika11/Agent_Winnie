"""
Tests for Workflow 1: Research & Evidence.
"""
from __future__ import annotations

from unittest.mock import patch
import pytest

from src.workflows.executor import workflow_executor
from src.workflows.research.graph import research_graph
from src.workflows.base import create_initial_workflow_state


def test_research_workflow_full_execution_with_contradiction_detection():
    """
    Scenario: User asks to research CardioFix clinical efficacy.
    Source A reports 34% plaque reduction.
    Source B reports 71% plaque reduction with high dosage.
    Hermes must explicitly document the contradiction in the final report.
    """
    objective = "Research CardioFix clinical efficacy and patient outcomes"

    initial_state = create_initial_workflow_state(
        objective=objective,
        workflow_type="research",
        workflow_data={
            "raw_search_results": [
                {
                    "subquestion_id": "sub_1",
                    "query": "CardioFix clinical efficacy",
                    "status": "success",
                    "output": "ClinicalTrials.gov registry study reports 34% reduction in arterial plaque.",
                    "evidence_item": {
                        "url": "https://clinicaltrials.gov/study-101",
                        "source": "ClinicalTrials.gov",
                        "content": "Phase 3 clinical trial observed a 34% reduction in arterial plaque across cohorts.",
                    },
                },
                {
                    "subquestion_id": "sub_2",
                    "query": "CardioFix dosage review",
                    "status": "success",
                    "output": "Lancet Review reported 71% reduction in plaque with intensified dosage.",
                    "evidence_item": {
                        "url": "https://thelancet.com/cardiofix-review",
                        "source": "The Lancet",
                        "content": "Multi-center evaluation observed a 71% reduction in arterial plaque with high dosage.",
                    },
                },
            ]
        },
    )

    with patch("src.workflows.research.nodes.tool_execution_engine.execute") as mock_exec:
        final_state = research_graph.invoke(initial_state)

    assert final_state["execution_status"] == "completed"
    assert final_state["verification_status"] == "verified"

    wf_data = final_state["workflow_data"]
    assert len(wf_data["subquestions"]) >= 2
    assert len(wf_data["extracted_evidence"]) == 2

    # Contradiction check: explicit disagreement must be detected and documented
    contradictions = wf_data.get("contradictions", [])
    assert len(contradictions) >= 1
    assert any("34%" in c["claim_a"] or "34%" in c["claim_b"] for c in contradictions)
    assert any("71%" in c["claim_a"] or "71%" in c["claim_b"] for c in contradictions)

    final_out = final_state["final_output"]
    assert final_out["has_contradictions"] is True
    assert final_out["citation_count"] == 2

    report_text = final_out["report_markdown"]
    assert "Research Report: " in report_text
    assert "Conflicting Findings & Disagreements" in report_text
    assert "https://clinicaltrials.gov/study-101" in report_text
    assert "https://thelancet.com/cardiofix-review" in report_text
