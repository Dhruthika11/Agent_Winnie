"""
LangGraph Definition for Workflow 3: Data Analysis.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from src.workflows.schemas import WorkflowState
from src.workflows.data_analysis.nodes import (
    load_dataset,
    inspect_schema,
    detect_anomalies,
    select_methods_and_guard_traps,
    perform_analysis,
    validate_calculations,
    generate_visualizations,
    explain_findings_and_limitations,
)


def build_data_analysis_graph():
    builder = StateGraph(WorkflowState)

    builder.add_node("load_dataset", load_dataset)
    builder.add_node("inspect_schema", inspect_schema)
    builder.add_node("detect_anomalies", detect_anomalies)
    builder.add_node("select_methods_and_guard_traps", select_methods_and_guard_traps)
    builder.add_node("perform_analysis", perform_analysis)
    builder.add_node("validate_calculations", validate_calculations)
    builder.add_node("generate_visualizations", generate_visualizations)
    builder.add_node("explain_findings_and_limitations", explain_findings_and_limitations)

    builder.set_entry_point("load_dataset")
    builder.add_edge("load_dataset", "inspect_schema")
    builder.add_edge("inspect_schema", "detect_anomalies")
    builder.add_edge("detect_anomalies", "select_methods_and_guard_traps")
    builder.add_edge("select_methods_and_guard_traps", "perform_analysis")
    builder.add_edge("perform_analysis", "validate_calculations")
    builder.add_edge("validate_calculations", "generate_visualizations")
    builder.add_edge("generate_visualizations", "explain_findings_and_limitations")
    builder.add_edge("explain_findings_and_limitations", END)

    return builder.compile()


data_analysis_graph = build_data_analysis_graph()
