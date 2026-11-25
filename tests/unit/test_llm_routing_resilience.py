"""
Tests for LLM Routing Resilience and Token Budgeting in Hermes Agent Platform.

Covers:
1. Normal request reaching the intended model (task-aware routing).
2. Provider fallback when the primary provider fails.
3. Rate limit handling (429 / cooldown / failover).
4. Malformed model responses and tool parsing recovery.
5. Token budget enforcement (FinOps daily spending caps).
6. Correct reporting when all providers fail (no silent failures or infinite loops).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.agent.pipeline_common import TOKEN_BUDGET_CHAR_LIMIT
from src.agent.router import (
    LOGICAL_AGENT_MODEL,
    MultiProviderRouter,
    agent_router,
    classify_task_intent,
)
from src.agent.tool_parser import parse_model_tool_calls, strip_unwanted_tool_xml
from src.config import settings
from src.security.finops import FinOpsManager, finops


# ---------------------------------------------------------------------------
# 1. Does a normal request reach the intended model?
# ---------------------------------------------------------------------------
def test_normal_request_reaches_intended_model():
    """Verify task intent classifier and router route prompts to the intended specialized models."""
    # A. Coding task -> routes to coding primary model (groq/qwen/qwen3.8-27b)
    coding_intent = classify_task_intent("Write a Python script to compute Fibonacci numbers")
    assert coding_intent == "coding"

    # B. Reasoning task -> routes to reasoning primary model (groq/openai/gpt-oss-120b)
    reasoning_intent = classify_task_intent("Compare the trade-offs between Paxos and Raft consensus algorithms")
    assert reasoning_intent == "reasoning"

    # C. Quick chat task -> routes to quick chat primary model (groq/openai/gpt-oss-20b)
    quick_chat_intent = classify_task_intent("Hello! How are you today?")
    assert quick_chat_intent == "quick_chat"

    # D. Multimodal task -> routes to multimodal profile (gemini/gemini-1.5-flash)
    multimodal_intent = classify_task_intent("Analyze this screenshot", has_images=True)
    assert multimodal_intent == "multimodal"

    # Verify router dispatches to candidate model
    router = MultiProviderRouter()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Hello!"))]

    with patch.object(router.router, "completion", return_value=mock_resp) as mock_comp:
        res = router.completion(
            messages=[{"role": "user", "content": "Write a python function to parse json"}]
        )
        assert res == mock_resp
        assert mock_comp.called
        # The dispatched model must match the coding profile primary
        assert router.last_dispatched_model == "groq/qwen/qwen3.8-27b"
        assert router.last_detected_task == "coding"


# ---------------------------------------------------------------------------
# 2. Does provider fallback work when the primary provider fails?
# ---------------------------------------------------------------------------
def test_provider_fallback_when_primary_fails():
    """Verify that when the primary provider throws a 500 or ConnectionError, router fails over to secondary."""
    router = MultiProviderRouter()
    mock_fallback_resp = MagicMock()
    mock_fallback_resp.choices = [MagicMock(message=MagicMock(content="Fallback response"))]

    call_count = 0

    def mock_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        model = kwargs.get("model", "")
        # Primary candidate fails
        if "qwen" in model:
            raise RuntimeError("Primary provider Groq 500 internal server error")
        # Secondary candidate succeeds
        return mock_fallback_resp

    with patch.object(router.router, "completion", side_effect=mock_completion):
        res = router.completion(
            messages=[{"role": "user", "content": "Write python code to sort an array"}]
        )
        assert res == mock_fallback_resp
        assert call_count >= 2
        # Dispatched model was updated to the fallback model
        assert router.last_dispatched_model != "groq/qwen/qwen3.8-27b"
        assert "120b" in router.last_dispatched_model or "groq" in router.last_dispatched_model


# ---------------------------------------------------------------------------
# 3. Does it handle rate limits?
# ---------------------------------------------------------------------------
def test_rate_limit_handling_and_cooldown():
    """Verify 429 RateLimitError adds primary to cooldown and routes to next candidate without infinite retries."""
    router = MultiProviderRouter()
    mock_success_resp = MagicMock()
    mock_success_resp.choices = [MagicMock(message=MagicMock(content="Success after rate limit"))]

    attempted_models = []

    def mock_completion(**kwargs):
        model = kwargs.get("model", "")
        attempted_models.append(model)
        if len(attempted_models) == 1:
            # Simulate 429 rate limit
            raise Exception("RateLimitError: 429 TPM limit exceeded for model")
        return mock_success_resp

    with patch.object(router.router, "completion", side_effect=mock_completion):
        res = router.completion(
            messages=[{"role": "user", "content": "Analyze system design trade-offs"}]
        )
        assert res == mock_success_resp
        # Exactly two attempts: primary failed on 429, secondary succeeded
        assert len(attempted_models) == 2
        assert attempted_models[0] != attempted_models[1]
        # No infinite retry loop occurred
        assert len(attempted_models) < 5


# ---------------------------------------------------------------------------
# 4. Does it handle malformed model responses?
# ---------------------------------------------------------------------------
def test_malformed_model_responses_handling():
    """Verify malformed JSON, unclosed XML, and corrupt tool arguments are gracefully repaired and parsed."""
    # Case A: Corrupted JSON inside <tool_call>
    malformed_json = (
        "Thinking about search...\n"
        "<tool_call>\n"
        '{"name": "web_scraper.search", "arguments": {invalid_json: missing_quotes}}\n'
        "</tool_call>\n"
        "Here are the findings."
    )
    clean_text, calls = parse_model_tool_calls(malformed_json)
    # Parser should strip the malformed XML without raising an unhandled exception
    assert "<tool_call>" not in clean_text
    assert "Here are the findings." in clean_text

    # Case B: Partially unclosed parameter XML
    unclosed_xml = (
        "<function=web_scraper.search>"
        "<parameter=query>quantum algorithms"
        "</function>"
    )
    clean_unclosed = strip_unwanted_tool_xml(unclosed_xml)
    assert "<function=" not in clean_unclosed
    assert "</function>" not in clean_unclosed

    # Case C: Valid tool call with stringified nested JSON arguments
    valid_with_nested_str = (
        '<tool_call>{"name": "web_scraper.search", "arguments": "{\\"query\\": \\"deep learning\\"}"}</tool_call>'
    )
    clean_text_c, calls_c = parse_model_tool_calls(valid_with_nested_str)
    assert len(calls_c) == 1
    assert calls_c[0]["name"] == "web_scraper.search"
    assert calls_c[0]["arguments"] == {"query": "deep learning"}


# ---------------------------------------------------------------------------
# 5. Does it enforce token budgets?
# ---------------------------------------------------------------------------
def test_token_budget_enforcement():
    """Verify that daily token budgets block excessive spend and large contexts are bounded."""
    tracker = FinOpsManager()
    test_user = "user_budget_test_01"

    # Under budget
    tracker.record_usage(user_id=test_user, tokens=1000)
    status, used, limit = tracker.evaluate_budget(user_id=test_user)
    assert status == "ok"
    assert used == 1000

    # Exceed limit
    limit_val = settings.max_daily_tokens_per_user
    tracker.record_usage(user_id=test_user, tokens=limit_val + 500)
    status_blocked, used_blocked, _ = tracker.evaluate_budget(user_id=test_user)
    assert status_blocked == "blocked"
    assert used_blocked > limit_val

    # Router enforces budget block when user_id is passed
    with patch("src.security.finops.finops.evaluate_budget", return_value=("blocked", 60000, 50000)):
        with pytest.raises(PermissionError, match="Token budget exhausted"):
            agent_router.completion(
                user_id="blocked_user",
                messages=[{"role": "user", "content": "Hello"}]
            )

    # Context character limiter boundary verification
    assert TOKEN_BUDGET_CHAR_LIMIT == 2500


# ---------------------------------------------------------------------------
# 6. Does it correctly report when all providers fail?
# ---------------------------------------------------------------------------
def test_correct_reporting_when_all_providers_fail():
    """Verify that when all providers fail, the router raises an explicit error and does not fail silently."""
    router = MultiProviderRouter()

    def mock_all_fail(**kwargs):
        raise ConnectionResetError("Remote server closed connection unexpectedly")

    with patch.object(router.router, "completion", side_effect=mock_all_fail):
        with pytest.raises(RuntimeError) as exc_info:
            router.completion(
                messages=[{"role": "user", "content": "Critical query"}]
            )

        err_msg = str(exc_info.value)
        # Must clearly indicate that all model providers failed
        assert "All model providers in task hierarchy failed" in err_msg
        assert "Remote server closed connection" in err_msg
        # Cause is preserved
        assert isinstance(exc_info.value.__cause__, ConnectionResetError)
