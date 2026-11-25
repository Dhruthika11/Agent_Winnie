"""
Schemas for Workflow 3: Data Analysis.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    duplicate_count: int


class StatisticalTrapWarning(BaseModel):
    trap_type: str  # "inappropriate_test" | "correlation_not_causation" | "multiple_testing_hazard" | "low_statistical_power"
    severity: str  # "warning" | "critical"
    description: str
    remedy: str


class StatisticalTestResult(BaseModel):
    test_name: str
    group_a_name: str
    group_b_name: str
    group_a_mean: float
    group_b_mean: float
    test_statistic: float
    p_value: float
    effect_size_cohens_d: float
    is_significant_at_alpha_05: bool
    assumptions_satisfied: bool
    traps_detected: list[StatisticalTrapWarning] = Field(default_factory=list)


class DataAnalysisReport(BaseModel):
    title: str
    dataset_summary: DatasetSummary
    test_result: StatisticalTestResult
    visualization_markdown: str
    substantive_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
