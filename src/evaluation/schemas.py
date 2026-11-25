"""
Data models and schemas for the Hermes Comprehensive Agent Evaluation Framework.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class EvaluationCategory(str, Enum):
    TASK_COMPLETION = "task_completion"
    REASONING_PLANNING = "reasoning_planning"
    TOOL_EXECUTION = "tool_execution"
    GROUNDING = "grounding"
    WORKFLOW_RELIABILITY = "workflow_reliability"


class BenchmarkTestCase(BaseModel):
    """Schema for a benchmark test case."""
    test_id: str
    input: str
    expected_workflow: str
    required_capabilities: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CategoryScore(BaseModel):
    category: EvaluationCategory
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    details: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Result of running and scoring a single benchmark test case."""
    test_id: str
    input: str
    expected_workflow: str
    actual_workflow: Optional[str] = None
    passed: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    category_scores: dict[str, CategoryScore] = Field(default_factory=dict)
    execution_time_ms: float = 0.0
    llm_calls_count: int = 0
    tool_calls_count: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


class EvaluationReport(BaseModel):
    """Summary report across an evaluation benchmark run."""
    run_id: str
    total_cases: int
    passed_cases: int
    pass_rate: float
    overall_avg_score: float
    category_averages: dict[str, float] = Field(default_factory=dict)
    workflow_breakdown: dict[str, dict[str, Any]] = Field(default_factory=dict)
    total_execution_time_ms: float = 0.0
    total_cost_usd: float = 0.0
    case_results: list[EvaluationResult] = Field(default_factory=list)
