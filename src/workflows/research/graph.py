"""
LangGraph Definition for Workflow 1: Research & Evidence.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from src.workflows.schemas import WorkflowState
from src.workflows.research.nodes import (
    decompose_query,
    parallel_search,
    extract_evidence,
    cross_source_verify,
    synthesize_report,
)


def build_research_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("decompose_query", decompose_query)
    builder.add_node("parallel_search", parallel_search)
    builder.add_node("extract_evidence", extract_evidence)
    builder.add_node("cross_source_verify", cross_source_verify)
    builder.add_node("synthesize_report", synthesize_report)

    builder.set_entry_point("decompose_query")
    builder.add_edge("decompose_query", "parallel_search")
    builder.add_edge("parallel_search", "extract_evidence")
    builder.add_edge("extract_evidence", "cross_source_verify")
    builder.add_edge("cross_source_verify", "synthesize_report")
    builder.add_edge("synthesize_report", END)

    return builder.compile()


research_graph = build_research_graph()
