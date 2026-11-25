"""
LangGraph Definition for Workflow 6: Autonomous Multi-Step Tasks.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from src.workflows.schemas import WorkflowState
from src.workflows.autonomous.nodes import (
    understand_and_plan,
    execute_dag_stages,
    evaluate_objective_satisfaction,
    assemble_final_artifact,
)


def build_autonomous_graph():
    builder = StateGraph(WorkflowState)

    builder.add_node("understand_and_plan", understand_and_plan)
    builder.add_node("execute_dag_stages", execute_dag_stages)
    builder.add_node("evaluate_objective_satisfaction", evaluate_objective_satisfaction)
    builder.add_node("assemble_final_artifact", assemble_final_artifact)

    builder.set_entry_point("understand_and_plan")
    builder.add_edge("understand_and_plan", "execute_dag_stages")
    builder.add_edge("execute_dag_stages", "evaluate_objective_satisfaction")
    builder.add_edge("evaluate_objective_satisfaction", "assemble_final_artifact")
    builder.add_edge("assemble_final_artifact", END)

    return builder.compile()


autonomous_graph = build_autonomous_graph()
