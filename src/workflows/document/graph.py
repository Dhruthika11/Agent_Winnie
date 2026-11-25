"""
LangGraph Definition for Workflow 4: Document Generation.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from src.workflows.schemas import WorkflowState
from src.workflows.document.nodes import (
    retrieve_sources,
    extract_facts_and_numbers,
    create_structure,
    generate_sections,
    verify_claims_and_consistency,
    format_document,
    inspect_document,
    export_document,
)


def route_after_inspection(state: WorkflowState) -> str:
    """Route to export_document if approved or max retries reached, else revise."""
    wf_data = state.get("workflow_data", {})
    inspection_passed = wf_data.get("inspection_passed", True)
    retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if inspection_passed or retries >= max_retries:
        return "export_document"
    # Otherwise increment retry and revise
    state["retry_count"] = retries + 1
    return "generate_sections"


def build_document_graph():
    builder = StateGraph(WorkflowState)

    builder.add_node("retrieve_sources", retrieve_sources)
    builder.add_node("extract_facts_and_numbers", extract_facts_and_numbers)
    builder.add_node("create_structure", create_structure)
    builder.add_node("generate_sections", generate_sections)
    builder.add_node("verify_claims_and_consistency", verify_claims_and_consistency)
    builder.add_node("format_document", format_document)
    builder.add_node("inspect_document", inspect_document)
    builder.add_node("export_document", export_document)

    builder.set_entry_point("retrieve_sources")
    builder.add_edge("retrieve_sources", "extract_facts_and_numbers")
    builder.add_edge("extract_facts_and_numbers", "create_structure")
    builder.add_edge("create_structure", "generate_sections")
    builder.add_edge("generate_sections", "verify_claims_and_consistency")
    builder.add_edge("verify_claims_and_consistency", "format_document")
    builder.add_edge("format_document", "inspect_document")
    builder.add_edge("inspect_document", "export_document")
    builder.add_edge("export_document", END)

    return builder.compile()


document_graph = build_document_graph()
