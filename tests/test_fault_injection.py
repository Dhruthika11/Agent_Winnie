"""
Tests for Fault Injection and Crash Recovery Testing.
Validates the 10 failure injection scenarios:
1. LLM provider unavailable -> fallback provider
2. MCP server crashes -> graceful process error handling
3. Search returns zero results -> secondary backend/reformulation
4. Redis becomes unavailable -> safe in-memory fallback
5. PostgreSQL disconnects -> resilient state preservation
6. Agent crashes mid-workflow -> resume from checkpoint
7. Tool returns malformed JSON -> schema validation and safe handling
8. Tool exceeds timeout -> bounded timeout termination
9. Repeated tool failure -> circuit breaker trip to OPEN
10. Ambiguous execution outcome -> idempotency deduplication prevents duplicate side-effects
"""
from __future__ import annotations

import pytest

from src.testing.fault_injection.fault_injector import fault_injector
from src.testing.fault_injection.schemas import FaultScenarioType


def test_fault_scenario_llm_provider_unavailable():
    verif = fault_injector.test_llm_provider_fallback(
        primary_caller=lambda: "Primary failed",
        fallback_caller=lambda: "Fallback GPT-4o output",
    )
    assert verif.scenario_type == FaultScenarioType.LLM_PROVIDER_UNAVAILABLE
    assert verif.passed is True
    assert verif.state_preserved is True


def test_fault_scenario_mcp_server_crash():
    verif = fault_injector.test_mcp_server_crash()
    assert verif.scenario_type == FaultScenarioType.MCP_SERVER_CRASH
    assert verif.passed is True
    assert verif.no_misleading_success is True


def test_fault_scenario_search_zero_results():
    verif = fault_injector.test_search_zero_results()
    assert verif.scenario_type == FaultScenarioType.SEARCH_ZERO_RESULTS
    assert verif.passed is True


def test_fault_scenario_redis_unavailable():
    verif = fault_injector.test_redis_unavailable()
    assert verif.scenario_type == FaultScenarioType.REDIS_UNAVAILABLE
    assert verif.passed is True
    assert verif.state_preserved is True


def test_fault_scenario_postgresql_disconnect():
    verif = fault_injector.test_postgresql_disconnect()
    assert verif.scenario_type == FaultScenarioType.POSTGRESQL_DISCONNECT
    assert verif.passed is True
    assert verif.state_preserved is True


def test_fault_scenario_agent_crash_mid_workflow():
    verif = fault_injector.test_agent_crash_mid_workflow()
    assert verif.scenario_type == FaultScenarioType.AGENT_CRASH_MID_WORKFLOW
    assert verif.passed is True
    assert verif.duplicate_actions_prevented is True


def test_fault_scenario_tool_malformed_json():
    verif = fault_injector.test_tool_malformed_json()
    assert verif.scenario_type == FaultScenarioType.TOOL_MALFORMED_JSON
    assert verif.passed is True


def test_fault_scenario_tool_timeout():
    verif = fault_injector.test_tool_timeout()
    assert verif.scenario_type == FaultScenarioType.TOOL_TIMEOUT
    assert verif.passed is True
    assert verif.no_misleading_success is True


def test_fault_scenario_repeated_tool_failure_circuit_breaker():
    verif = fault_injector.test_repeated_tool_failure_circuit_breaker()
    assert verif.scenario_type == FaultScenarioType.REPEATED_TOOL_FAILURE
    assert verif.passed is True


def test_fault_scenario_ambiguous_execution_idempotency_recovery():
    verif = fault_injector.test_ambiguous_execution_outcome()
    assert verif.scenario_type == FaultScenarioType.AMBIGUOUS_EXECUTION_OUTCOME
    assert verif.passed is True
    assert verif.duplicate_actions_prevented is True
