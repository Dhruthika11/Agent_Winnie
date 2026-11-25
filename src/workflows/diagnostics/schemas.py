"""
Schemas for Workflow 5: System Diagnostics & Recovery.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class ComponentHealth(BaseModel):
    component_name: str
    status: str  # "healthy" | "degraded" | "unhealthy"
    latency_ms: float
    details: str


class RootCauseDiagnosis(BaseModel):
    failing_component: str
    failure_category: str  # "tool_failure" | "retrieval_failure" | "grounding_failure" | "infrastructure_failure" | "security_failure" | "memory_failure"
    underlying_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class RemediationProposal(BaseModel):
    action_type: str  # "configuration_patch" | "service_restart" | "timeout_increase" | "schema_repair"
    description: str
    target_component: str
    diff_or_command: str
    requires_approval: bool = True
    approval_granted: bool = False


class DiagnosticsReport(BaseModel):
    system_status: str
    failing_component: Optional[str] = None
    diagnosis: Optional[RootCauseDiagnosis] = None
    remediation_proposal: Optional[RemediationProposal] = None
    diagnostic_tests_passed: bool = True
    approval_status: str = "none"
    recovery_verified: bool = False
