"""
Schemas and contracts for the Hermes Security Red-Team Testing Framework.
"""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class AttackCategory(str, Enum):
    PROMPT_INJECTION_DIRECT = "prompt_injection_direct"
    PROMPT_INJECTION_INDIRECT = "prompt_injection_indirect"
    TOOL_ABUSE = "tool_abuse"
    MEMORY_POISONING = "memory_poisoning"
    CROSS_TENANT_ACCESS = "cross_tenant_access"
    SECRET_LEAKAGE = "secret_leakage"
    SANDBOX_ESCAPE = "sandbox_escape"
    APPROVAL_BYPASS = "approval_bypass"


class AttackVector(BaseModel):
    attack_id: str
    category: AttackCategory
    description: str
    payload: Any
    target_boundary: str  # e.g., "rbac", "sandbox_isolation", "hitl_gate", "input_guard", "output_guard", "tenant_isolation"
    expected_enforcement: str


class RedTeamTestResult(BaseModel):
    attack_id: str
    category: AttackCategory
    blocked: bool
    enforcing_boundary: str
    sanitized_output: Optional[str] = None
    defense_details: list[str] = Field(default_factory=list)
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
