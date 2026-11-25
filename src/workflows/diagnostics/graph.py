"""
LangGraph Definition for Workflow 5: System Diagnostics & Recovery.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from src.workflows.schemas import WorkflowState
from src.workflows.diagnostics.nodes import (
    health_check,
    collect_logs_traces,
    identify_failing_component,
    classify_root_cause,
    generate_remediation,
    run_diagnostic_tests,
    apply_fix_with_approval,
    verify_recovery,
)


def build_diagnostics_graph():
    builder = StateGraph(WorkflowState)

    builder.add_node("health_check", health_check)
    builder.add_node("collect_logs_traces", collect_logs_traces)
    builder.add_node("identify_failing_component", identify_failing_component)
    builder.add_node("classify_root_cause", classify_root_cause)
    builder.add_node("generate_remediation", generate_remediation)
    builder.add_node("run_diagnostic_tests", run_diagnostic_tests)
    builder.add_node("apply_fix_with_approval", apply_fix_with_approval)
    builder.add_node("verify_recovery", verify_recovery)

    builder.set_entry_point("health_check")
    builder.add_edge("health_check", "collect_logs_traces")
    builder.add_edge("collect_logs_traces", "identify_failing_component")
    builder.add_edge("identify_failing_component", "classify_root_cause")
    builder.add_edge("classify_root_cause", "generate_remediation")
    builder.add_edge("generate_remediation", "run_diagnostic_tests")
    builder.add_edge("run_diagnostic_tests", "apply_fix_with_approval")
    builder.add_edge("apply_fix_with_approval", "verify_recovery")
    builder.add_edge("verify_recovery", END)

    return builder.compile()


diagnostics_graph = build_diagnostics_graph()
