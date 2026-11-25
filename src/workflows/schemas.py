"""
Unified Schemas and State Contracts for the Hermes Workflow Engine.

Defines the shared WorkflowState contract, workflow classification decisions,
step lifecycle records, and performance metrics across all 6 specialized workflows:
1. Research & Evidence
2. Code Execution & Debugging
3. Data Analysis
4. Document Generation
5. System Diagnostics & Recovery
6. Autonomous Multi-Step Tasks
"""
from __future__ import annotations

import datetime
import enum
from typing import Any, Optional, TypedDict
from pydantic import BaseModel, Field


class WorkflowType(str, enum.Enum):
    RESEARCH = "research"
    CODING = "coding"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENT = "document"
    DIAGNOSTICS = "diagnostics"
    AUTONOMOUS = "autonomous"


class ExecutionStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowDecision(BaseModel):
    """Routing decision returned by the workflow classification router."""
    workflow: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    composed_workflows: list[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None


class StepRecord(BaseModel):
    """Detailed telemetry record for an individual node/step execution."""
    step_name: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"  # "completed" | "failed" | "skipped" | "awaiting_approval"
    duration_ms: float = 0.0
    output_summary: Optional[str] = None
    error: Optional[str] = None


class WorkflowState(TypedDict, total=False):
    """
    Standardized state contract shared by all Hermes LangGraph workflow graphs.
    Workflows can add task-specific extensions while guaranteeing this contract.
    """
    workflow_id: str
    workflow_type: str
    user_id: str
    session_id: str
    tenant_id: str
    objective: str
    plan: list[dict[str, Any]]
    current_step: str
    tool_results: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    errors: list[str]
    retry_count: int
    max_retries: int
    approval_status: str  # "none" | "required" | "approved" | "rejected"
    approval_id: Optional[str]
    verification_status: str  # "unverified" | "verified" | "failed"
    final_output: dict[str, Any]
    execution_status: str
    step_history: list[dict[str, Any]]
    metrics: dict[str, Any]
    # Task-specific extensions
    workflow_data: dict[str, Any]
