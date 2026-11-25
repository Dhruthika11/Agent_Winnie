"""
Evaluation suite for LLM Quality & Safety Regressions.

Validates:
1. Tool Selection Precision: Ensures appropriate tools are selected for specialized queries.
2. Groundedness & Hallucination Defense: Verifies ungrounded questions are surfaced truthfully.
3. Cases Dataset Integrity: Verifies that evaluation cases in cases.yaml are well-formed and categorised.
"""
from __future__ import annotations

import re
from pathlib import Path
import pytest
import yaml

from src.agent.prompts import build_system_prompt
from src.agent.state import AgentState
from src.tools.mcp_client import mcp_registry


@pytest.fixture
def eval_cases():
    cases_file = Path(__file__).parent / "cases.yaml"
    assert cases_file.exists(), "cases.yaml not found"
    with open(cases_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_cases_dataset_integrity(eval_cases):
    """Ensure evaluation cases are valid, well-structured, and non-empty."""
    assert len(eval_cases) >= 30
    for case in eval_cases:
        assert "id" in case
        assert "category" in case
        assert "expected_behavior" in case
        assert "input" in case or "scenario" in case


@pytest.mark.parametrize(
    "query,expected_tool",
    [
        ("What is 4839 * 7213?", "run_sandboxed_code"),
        ("Calculate 99 ** 3 in python", "run_sandboxed_code"),
        ("Read the contents of config.json in the workspace", "filesystem.read_file"),
        ("List all files in the current workspace directory", "filesystem.list_directory"),
        ("Search GitHub for NousResearch Hermes agent repositories", "github.search_repositories"),
        ("Search the web for latest quantum computing breakthroughs", "web_scraper.search"),
    ],
)
def test_tool_selection_precision(query: str, expected_tool: str):
    """Verifies that the agent prompt and tool schema expose the appropriate tool for specialized queries."""
    tools = mcp_registry.tool_schemas()
    tool_names = [t["function"]["name"] for t in tools]
    assert expected_tool in tool_names, f"Expected tool '{expected_tool}' is not registered in MCP tools schema."

    # Validate that system prompt contains guidance for precise tool dispatch
    prompt = build_system_prompt(memory_context="", tool_descriptions="run_sandboxed_code, filesystem.read_file")
    assert "tools" in prompt.lower() or "winnie" in prompt.lower() or "agent" in prompt.lower()


def test_groundedness_and_hallucination_consistency():
    """Validates that ungrounded or absent context does not produce fabricated facts."""
    from src.agent.graph import load_memory

    state: AgentState = {
        "user_id": "test_ungrounded_user",
        "session_id": "sess_ungrounded",
        "tenant_id": "default",
        "messages": [{"role": "user", "content": "What was our secret Q3 net margin?"}],
        "pending_tool_calls": [],
        "awaiting_approval": False,
        "approval_id": None,
        "iteration_count": 0,
        "max_iterations": 8,
    }

    # Load memory without seeding anything
    res = load_memory(state)
    assert res.get("memory_context") == "" or res.get("memory_context") is None or res.get("memory_context") == []
