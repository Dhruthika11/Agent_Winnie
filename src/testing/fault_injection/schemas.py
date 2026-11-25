"""
Schemas and contracts for the Hermes Fault Injection and Crash Recovery Testing Framework.
"""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class FaultScenarioType(str, Enum):
    LLM_PROVIDER_UNAVAILABLE = "llm_provider_unavailable"
    MCP_SERVER_CRASH = "mcp_server_crash"
    SEARCH_ZERO_RESULTS = "search_zero_results"
    REDIS_UNAVAILABLE = "redis_unavailable"
    POSTGRESQL_DISCONNECT = "postgresql_disconnect"
    AGENT_CRASH_MID_WORKFLOW = "agent_crash_mid_workflow"
    TOOL_MALFORMED_JSON = "tool_malformed_json"
    TOOL_TIMEOUT = "tool_timeout"
    REPEATED_TOOL_FAILURE = "repeated_tool_failure"
    AMBIGUOUS_EXECUTION_OUTCOME = "ambiguous_execution_outcome"


class FaultInjectionConfig(BaseModel):
    scenario_type: FaultScenarioType
    target_component: str
    failure_delay_ms: float = 0.0
    error_message: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class RecoveryVerification(BaseModel):
    scenario_type: FaultScenarioType
    passed: bool
    recovery_strategy_used: str
    state_preserved: bool
    duplicate_actions_prevented: bool
    no_misleading_success: bool
    details: list[str] = Field(default_factory=list)
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
