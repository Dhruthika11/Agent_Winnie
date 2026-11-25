"""
Specialized LangGraph Workflow Subsystem for Hermes Agent Platform.

Exports:
- Shared state contracts and types (WorkflowState, WorkflowType, WorkflowDecision)
- Workflow Registry and Router (WORKFLOW_REGISTRY, classify_workflow, get_workflow_graph)
- Centralized Workflow Executor (WorkflowExecutor, workflow_executor)
- The 6 specialized workflows:
  1. research_graph (Research & Evidence)
  2. coding_graph (Code Execution & Debugging)
  3. data_analysis_graph (Data Analysis)
  4. document_graph (Document Generation)
  5. diagnostics_graph (System Diagnostics & Recovery)
  6. autonomous_graph (Autonomous Multi-Step Tasks)
"""
from src.workflows.autonomous.graph import autonomous_graph, build_autonomous_graph
from src.workflows.base import create_initial_workflow_state, increment_metric, record_step
from src.workflows.coding.graph import build_coding_graph, coding_graph
from src.workflows.data_analysis.graph import build_data_analysis_graph, data_analysis_graph
from src.workflows.diagnostics.graph import build_diagnostics_graph, diagnostics_graph
from src.workflows.document.graph import build_document_graph, document_graph
from src.workflows.executor import WorkflowExecutor, workflow_executor
from src.workflows.registry import (
    WORKFLOW_REGISTRY,
    classify_workflow,
    get_workflow_graph,
    list_available_workflows,
)
from src.workflows.research.graph import build_research_graph, research_graph
from src.workflows.schemas import (
    ExecutionStatus,
    StepRecord,
    WorkflowDecision,
    WorkflowState,
    WorkflowType,
)

__all__ = [
    # Schemas
    "WorkflowState",
    "WorkflowType",
    "ExecutionStatus",
    "WorkflowDecision",
    "StepRecord",
    # Base utilities
    "create_initial_workflow_state",
    "record_step",
    "increment_metric",
    # Registry & Executor
    "WORKFLOW_REGISTRY",
    "classify_workflow",
    "get_workflow_graph",
    "list_available_workflows",
    "WorkflowExecutor",
    "workflow_executor",
    # Specialized Graphs
    "research_graph",
    "coding_graph",
    "data_analysis_graph",
    "document_graph",
    "diagnostics_graph",
    "autonomous_graph",
]
