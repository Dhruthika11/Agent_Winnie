"""
Schemas for Workflow 6: Autonomous Multi-Step Tasks.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class TaskNode(BaseModel):
    task_id: str
    name: str
    action_type: str  # "research" | "compute" | "synthesize" | "tool_call"
    description: str
    tool_name: Optional[str] = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)  # IDs of tasks that must finish first
    status: str = "pending"  # "pending" | "running" | "completed" | "failed"
    output: Optional[Any] = None
    error: Optional[str] = None


class TaskDAG(BaseModel):
    tasks: dict[str, TaskNode]
    execution_batches: list[list[str]] = Field(default_factory=list)  # Topological stages


class AutonomousArtifact(BaseModel):
    objective: str
    total_tasks: int
    successful_tasks: int
    objective_satisfied: bool
    summary: str
    combined_outputs: dict[str, Any] = Field(default_factory=dict)
