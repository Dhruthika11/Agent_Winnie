"""
LangGraph Definition for Workflow 2: Code Execution & Debugging.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from src.workflows.schemas import WorkflowState
from src.workflows.coding.nodes import (
    understand_task,
    inspect_workspace,
    plan_implementation,
    generate_code,
    execute_sandbox,
    run_tests,
    debug_fix,
    verify_suite,
    report_results,
)


def route_after_testing(state: WorkflowState) -> str:
    """Route to verify_suite if passed or max retries reached, else debug_fix."""
    wf_data = state.get("workflow_data", {})
    tests_passed = wf_data.get("tests_passed", False)
    retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if tests_passed or retries >= max_retries:
        return "verify_suite"
    return "debug_fix"


def build_coding_graph():
    builder = StateGraph(WorkflowState)

    builder.add_node("understand_task", understand_task)
    builder.add_node("inspect_workspace", inspect_workspace)
    builder.add_node("plan_implementation", plan_implementation)
    builder.add_node("generate_code", generate_code)
    builder.add_node("execute_sandbox", execute_sandbox)
    builder.add_node("run_tests", run_tests)
    builder.add_node("debug_fix", debug_fix)
    builder.add_node("verify_suite", verify_suite)
    builder.add_node("report_results", report_results)

    builder.set_entry_point("understand_task")
    builder.add_edge("understand_task", "inspect_workspace")
    builder.add_edge("inspect_workspace", "plan_implementation")
    builder.add_edge("plan_implementation", "generate_code")
    builder.add_edge("generate_code", "execute_sandbox")
    builder.add_edge("execute_sandbox", "run_tests")

    builder.add_conditional_edges(
        "run_tests",
        route_after_testing,
        {
            "verify_suite": "verify_suite",
            "debug_fix": "debug_fix",
        },
    )

    builder.add_edge("debug_fix", "execute_sandbox")
    builder.add_edge("verify_suite", "report_results")
    builder.add_edge("report_results", END)

    return builder.compile()


coding_graph = build_coding_graph()
