"""
Pydantic/TypedDict schema for the LangGraph agent state.

Kept as a TypedDict (rather than a plain Pydantic model) because LangGraph's
StateGraph applies reducers (e.g. `add_messages`) keyed on TypedDict fields.
"""
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class ToolCallRecord(TypedDict, total=False):
    id: str
    name: str
    arguments: dict[str, Any]
    risk_level: str  # "low" | "medium" | "high"


class AgentState(TypedDict):
    # Full running transcript. `add_messages` merges new messages instead of
    # overwriting, and handles de-duplication by message id.
    messages: Annotated[list, add_messages]

    # Identity / session
    user_id: str
    session_id: str
    tenant_id: Optional[str]
    user_role: Optional[str]
    execution_id: Optional[str]
    trace_id: Optional[str]
    parent_span_id: Optional[str]

    # Loop control
    iteration: int
    max_iterations: int

    # Memory
    memory_context: str

    # Tool execution / HITL & Evidence Grounding
    pending_tool_calls: list[ToolCallRecord]
    awaiting_approval: bool
    approval_id: Optional[str]
    detected_intent: Optional[str]
    search_decision: Optional[dict[str, Any]]
    evidence_urls: list[str]
    evidence_items: list[dict[str, Any]]
    evidence_verification: Optional[dict[str, Any]]
    denied_tool_calls: list[dict[str, Any]]
    grounding_retries: int

    # Terminal state
    done: bool
    stop_reason: Optional[str]
