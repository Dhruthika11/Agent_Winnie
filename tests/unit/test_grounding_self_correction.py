"""
Unit tests for the Grounding Self-Correction & Verification Re-Search Gate.
Verifies that ungrounded URLs, unverified claims, or missing evidence automatically
trigger active web search tool calls to retrieve verified evidence before answering.
"""
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from src.agent.graph import call_model
from src.agent.state import AgentState


def test_grounding_self_correction_triggers_search_on_unverified_url():
    """Verify that generating an ungrounded profile URL triggers an automated recovery search tool call."""
    raw_content_with_fake_url = (
        "Here is the profile: https://www.linkedin.com/in/ganesh-rapuru. He is an AI researcher."
    )

    state: AgentState = {
        "messages": [{"role": "user", "content": "Find the LinkedIn profile of Ganesh Rapuru"}],
        "user_id": "test_user",
        "session_id": "sess_1",
        "tenant_id": "default",
        "trace_id": None,
        "parent_span_id": None,
        "iteration": 0,
        "max_iterations": 8,
        "memory_context": "",
        "pending_tool_calls": [],
        "awaiting_approval": False,
        "approval_id": None,
        "detected_intent": "SEARCH",
        "search_decision": None,
        "evidence_urls": [],  # No linkedin URLs verified yet
        "evidence_items": [],
        "evidence_verification": None,
        "denied_tool_calls": [],
        "grounding_retries": 0,
        "done": False,
        "stop_reason": None,
    }

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message = MagicMock(content=raw_content_with_fake_url, tool_calls=None)
    mock_resp.usage = None

    with patch("src.agent.graph.agent_router.completion", return_value=mock_resp):
        res = call_model(state, config=None)

    # Must NOT end the turn; must generate an automated recovery search tool call
    assert len(res["pending_tool_calls"]) == 1
    recovery_call = res["pending_tool_calls"][0]
    assert recovery_call["name"] == "web_scraper.search"
    assert "ganesh rapuru" in recovery_call["arguments"]["query"].lower()
    assert res["grounding_retries"] == 1


def test_grounding_self_correction_triggers_on_missing_evidence():
    """Verify that when the search router requires search but no evidence exists, a search tool call is synthesized."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "how many ai startups are in San Francisco"}],
        "user_id": "test_user",
        "session_id": "sess_2",
        "tenant_id": "default",
        "trace_id": None,
        "parent_span_id": None,
        "iteration": 0,
        "max_iterations": 8,
        "memory_context": "",
        "pending_tool_calls": [],
        "awaiting_approval": False,
        "approval_id": None,
        "detected_intent": "SEARCH",
        "search_decision": None,
        "evidence_urls": [],
        "evidence_items": [],
        "evidence_verification": None,
        "denied_tool_calls": [],
        "grounding_retries": 0,
        "done": False,
        "stop_reason": None,
    }

    # Model attempts to answer directly without calling tools
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message = MagicMock(content="There are probably around 1,000 AI startups in SF.", tool_calls=None)
    mock_resp.usage = None

    with patch("src.agent.graph.agent_router.completion", return_value=mock_resp):
        res = call_model(state, config=None)

    # Automated grounding recovery must trigger web search
    assert len(res["pending_tool_calls"]) == 1
    call = res["pending_tool_calls"][0]
    assert call["name"] == "web_scraper.search"
    assert "san francisco" in call["arguments"]["query"].lower() or "ai" in call["arguments"]["query"].lower()
    assert res["grounding_retries"] == 1


def test_grounding_self_correction_bounds_retries():
    """Verify that when max grounding retries are reached, the agent completes without looping infinitely."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Find the LinkedIn profile of unknown persona xyz"}],
        "user_id": "test_user",
        "session_id": "sess_3",
        "tenant_id": "default",
        "trace_id": None,
        "parent_span_id": None,
        "iteration": 2,
        "max_iterations": 8,
        "memory_context": "",
        "pending_tool_calls": [],
        "awaiting_approval": False,
        "approval_id": None,
        "detected_intent": "SEARCH",
        "search_decision": None,
        "evidence_urls": [],
        "evidence_items": [
            {
                "tool_name": "web_scraper.search",
                "validation_status": "verified",
                "content": "No profile found for persona xyz on LinkedIn.",
            }
        ],
        "evidence_verification": None,
        "denied_tool_calls": [],
        "grounding_retries": 2,  # Max retries reached
        "done": False,
        "stop_reason": None,
    }

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message = MagicMock(
        content="I searched for the LinkedIn profile of persona xyz, but no verified profile link was found.",
        tool_calls=None,
    )
    mock_resp.usage = None

    with patch("src.agent.graph.agent_router.completion", return_value=mock_resp):
        res = call_model(state, config=None)

    # Should NOT trigger another retry since grounding_retries == 2
    assert len(res["pending_tool_calls"]) == 0
    # Final reply should be present and clean
    assert "no verified profile link was found" in res["messages"][0]["content"].lower()
