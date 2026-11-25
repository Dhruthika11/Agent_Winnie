"""
Tests for Workflow 4: Document Generation.
"""
from __future__ import annotations

import pytest

from src.workflows.base import create_initial_workflow_state
from src.workflows.document.graph import document_graph


def test_document_generation_with_inspection_stage():
    """
    Scenario: User asks: "Create a detailed technical report from these research results."
    Hermes retrieves facts, structures outline, generates sections, checks numerical consistency,
    formats tables and headings, and runs a dedicated document inspection stage before export.
    """
    source_notes = (
        "Project Orion Architecture Notes:\n"
        "Benchmark throughput: 14,250 req/sec at p99 latency 18.4ms.\n"
        "Deployment topology requires 8 vCPUs and 32GB RAM.\n"
        "Uptime SLA commitment is 99.95%.\n"
    )

    initial_state = create_initial_workflow_state(
        objective="Create a detailed technical report on Project Orion architecture and performance",
        workflow_type="document",
        workflow_data={"source_material": source_notes},
    )

    final_state = document_graph.invoke(initial_state)

    assert final_state["execution_status"] == "completed"
    assert final_state["verification_status"] == "verified"

    wf_data = final_state["workflow_data"]

    # 1. Facts extracted
    facts = wf_data["extracted_numerical_facts"]
    assert any("14,250" in f or "18.4" in f for f in facts)

    # 2. Outline & Sections
    sections = wf_data["generated_sections"]
    assert len(sections) >= 5

    # 3. Document inspection results
    inspection = wf_data["inspection_result"]
    assert inspection["is_approved"] is True
    assert len(inspection["formatting_issues"]) == 0
    assert len(inspection["numerical_consistency_issues"]) == 0

    # 4. Final document output
    final_out = final_state["final_output"]
    doc_artifact = final_out["document_artifact"]
    assert doc_artifact["inspection_passed"] is True
    assert doc_artifact["section_count"] >= 5
    assert doc_artifact["word_count"] > 100

    doc_text = final_out["document_markdown"]
    assert "# Technical Report: Create a detailed technical report" in doc_text
    assert "14,250 req/sec" in doc_text
    assert "18.4ms" in doc_text
    assert "| **Throughput** |" in doc_text
