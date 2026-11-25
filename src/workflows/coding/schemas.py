"""
Schemas for Workflow 2: Code Execution & Debugging.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class CodeTaskSpec(BaseModel):
    task_name: str
    language: str = "python"
    requirements: list[str] = Field(default_factory=list)
    input_format: Optional[str] = None
    expected_output_format: Optional[str] = None
    assertions: list[str] = Field(default_factory=list)


class CodePlan(BaseModel):
    overview: str
    steps: list[str] = Field(default_factory=list)
    files_to_touch: list[str] = Field(default_factory=list)
    safety_checklist: list[str] = Field(default_factory=list)


class DebugAttempt(BaseModel):
    attempt: int
    error_message: str
    root_cause_analysis: str
    fix_applied: str
    timestamp: str


class CodingArtifactReport(BaseModel):
    task_name: str
    language: str
    final_code: str
    execution_verified: bool
    total_debug_attempts: int
    test_results_summary: str
    sandbox_isolation_enforced: bool = True
