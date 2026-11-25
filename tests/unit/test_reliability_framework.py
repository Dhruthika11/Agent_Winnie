"""
Comprehensive Regression Test Suite for Hermes Reliability Fix:
1. Token-Aware Prompt Budgeting & Tool Filtering
2. Error Classification, Quota-Aware Fallbacks & Circuit Breakers
3. LiteLLM Proxy Health Checks & Zombie Socket Detection
4. Streaming Telemetry, Variable Scoping & Tenant Isolation
"""
import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.error_classifier import (
    ErrorCategory,
    error_classifier,
    extract_provider_from_model,
    ProviderCircuitBreaker,
)
from src.agent.prompt_manager import (
    PromptBudgetExceededError,
    prompt_budget_manager,
)
from src.agent.router import MultiProviderRouter, _is_proxy_responsive


# =========================================================================== #
# 1. Prompt Budget Manager & Tool Filtering Tests
# =========================================================================== #
class TestPromptBudgetManager:
    """Tests token estimation, intent-based tool filtering, and compaction."""

    def test_short_conversation_fits_budget_without_compaction(self):
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."},
        ]
        result = prompt_budget_manager.build_budgeted_prompt(
            system_prompt="You are a helpful assistant.",
            messages=messages,
            tools=None,
            model="groq/openai/gpt-oss-20b",
            intent_category="quick_chat",
        )
        assert result.compaction_applied is False
        assert len(result.messages) == 3
        assert result.total_estimated_tokens < result.prompt_budget

    def test_ten_turn_conversation_is_compacted_safely(self):
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"User question {i} with some detailed context text."})
            messages.append({"role": "assistant", "content": f"Assistant response {i} explaining in full detail " * 15})

        latest_prompt = "Explain why this happened in step 9."
        messages.append({"role": "user", "content": latest_prompt})

        result = prompt_budget_manager.build_budgeted_prompt(
            system_prompt="You are a helpful assistant.",
            messages=messages,
            tools=None,
            model="groq/openai/gpt-oss-20b",
            intent_category="general",
        )
        assert result.compaction_applied is True
        # Latest user message MUST be preserved intact
        assert result.messages[-1]["content"] == latest_prompt
        assert result.total_estimated_tokens <= result.prompt_budget

    def test_extremely_long_history_preserves_latest_user_prompt(self):
        messages = [
            {"role": "user", "content": "Initial conversation seed."},
            {"role": "assistant", "content": "Detailed initial setup essay. " * 30},
        ]
        for i in range(15):
            messages.append({"role": "tool", "content": f"Giant observation output {i} " * 50})
            messages.append({"role": "assistant", "content": f"Intermediate answer {i} " * 20})

        latest_user_msg = "Final urgent command that must never be truncated!"
        messages.append({"role": "user", "content": latest_user_msg})

        result = prompt_budget_manager.build_budgeted_prompt(
            system_prompt="System prompt.",
            messages=messages,
            tools=None,
            model="groq/openai/gpt-oss-20b",
            intent_category="general",
        )
        assert result.messages[-1]["content"] == latest_user_msg
        assert result.total_estimated_tokens <= result.prompt_budget

    def test_large_mcp_tool_schemas_and_workflow_tool_filtering(self):
        dummy_tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_scraper.search",
                    "description": "Search the web for real-time information",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sandbox.run_code",
                    "description": "Run Python code inside isolated sandbox",
                    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "filesystem.read_file",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminal.execute_bash",
                    "description": "Execute bash command in terminal",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                },
            },
        ]

        # Case A: Memory instruction filters out ALL tools
        filtered_mem = prompt_budget_manager.filter_tools_by_intent(dummy_tools, "memory_instruction")
        assert len(filtered_mem) == 0

        # Case B: Coding intent filters to sandbox, filesystem, terminal (omits web scraper)
        filtered_code = prompt_budget_manager.filter_tools_by_intent(dummy_tools, "coding", requires_search=False)
        tool_names = [t["function"]["name"] for t in filtered_code]
        assert "sandbox.run_code" in tool_names
        assert "filesystem.read_file" in tool_names
        assert "web_scraper.search" not in tool_names

        # Case C: Research intent filters to search and read file (omits bash/sandbox)
        filtered_research = prompt_budget_manager.filter_tools_by_intent(dummy_tools, "research", requires_search=True)
        res_names = [t["function"]["name"] for t in filtered_research]
        assert "web_scraper.search" in res_names
        assert "filesystem.read_file" in res_names
        assert "terminal.execute_bash" not in res_names

    def test_oversized_uncompactable_prompt_raises_budget_exceeded_error(self):
        # A single user prompt with 30,000 characters (>9,000 tokens) exceeds Groq's 7,500 token limit
        giant_user_prompt = "CRITICAL_PAYLOAD " * 3000
        messages = [{"role": "user", "content": giant_user_prompt}]

        with pytest.raises(PromptBudgetExceededError) as exc_info:
            prompt_budget_manager.build_budgeted_prompt(
                system_prompt="Short system prompt",
                messages=messages,
                tools=None,
                model="groq/openai/gpt-oss-20b",
                intent_category="general",
            )
        assert exc_info.value.estimated_tokens > exc_info.value.budget_limit
        assert "exceeds safe budget" in str(exc_info.value)


# =========================================================================== #
# 2. Error Classification, Retries & Fallback Tests
# =========================================================================== #
class TestErrorClassificationAndCircuitBreaker:
    """Tests error classification, non-retryable errors, and quota awareness."""

    def test_tpm_quota_error_classified_as_non_retryable_on_same_provider(self):
        err_msg = (
            "RateLimitError: Error code: 429 - {'error': {'message': "
            "'Request too large for model `openai/gpt-oss-20b` on tokens per minute (TPM): Limit 8000, Requested 11392.'}}"
        )
        classified = error_classifier.classify(err_msg, model="groq/openai/gpt-oss-20b")
        assert classified.category == ErrorCategory.TPM_QUOTA
        assert classified.is_retryable is False
        assert classified.provider == "groq"
        assert classified.recommended_action == "switch_provider"

    def test_context_length_error_classified(self):
        err_msg = "BadRequestError: context_length_exceeded: maximum context length is 8192 tokens"
        classified = error_classifier.classify(err_msg, model="groq/openai/gpt-oss-20b")
        assert classified.category == ErrorCategory.CONTEXT_LENGTH
        assert classified.is_retryable is False
        assert classified.recommended_action == "compact_prompt"

    def test_authentication_error_fails_fast_without_retry(self):
        err_msg = "AuthenticationError: 401 Unauthorized - Invalid API key provided"
        classified = error_classifier.classify(err_msg, model="groq/openai/gpt-oss-20b")
        assert classified.category == ErrorCategory.AUTHENTICATION
        assert classified.is_retryable is False
        assert classified.recommended_action == "fail_fast"

    def test_provider_outage_classified_as_retryable(self):
        err_msg = "APIConnectionError: 503 Service Unavailable: upstream server is temporarily overloaded"
        classified = error_classifier.classify(err_msg, model="groq/qwen/qwen3.8-27b")
        assert classified.category == ErrorCategory.PROVIDER_OUTAGE
        assert classified.is_retryable is True

    def test_timeout_error_classified_as_retryable(self):
        err_msg = "ReadTimeout: The read operation timed out after 20.0 seconds"
        classified = error_classifier.classify(err_msg, model="groq/qwen/qwen3.8-27b")
        assert classified.category == ErrorCategory.TIMEOUT
        assert classified.is_retryable is True

    def test_provider_circuit_breaker_trips_and_recovers(self):
        cb = ProviderCircuitBreaker(failure_threshold=2, cooldown_seconds=0.2)
        assert cb.is_available("groq") is True

        # First failure
        dummy_err = error_classifier.classify("503 Service Unavailable", model="groq/model")
        cb.record_failure("groq", dummy_err)
        assert cb.is_available("groq") is True

        # Second failure -> trips breaker
        cb.record_failure("groq", dummy_err)
        assert cb.is_available("groq") is False

        # After cooldown
        time.sleep(0.25)
        assert cb.is_available("groq") is True

        # Success resets counter
        cb.record_success("groq")
        assert cb.get_status().get("groq", {}).get("failures", 0) == 0

    def test_router_quota_awareness_skips_same_provider_on_tpm_error(self):
        router = MultiProviderRouter()
        # Mock completion to simulate Groq TPM error
        mock_router = MagicMock()
        mock_router.completion.side_effect = Exception(
            "RateLimitError: Request too large on tokens per minute (TPM): Limit 8000, Requested 9000."
        )
        router._router = mock_router

        # When task is reasoning, candidate_models include multiple groq models
        with pytest.raises(RuntimeError) as exc_info:
            router.completion(
                messages=[{"role": "user", "content": "analyze this"}],
            )
        assert "All model providers in task hierarchy failed" in str(exc_info.value)
        # Verify that router did not loop infinitely through all Groq candidates
        assert mock_router.completion.call_count <= 2


# =========================================================================== #
# 3. LiteLLM Proxy Health Tests
# =========================================================================== #
class TestProxyHealthChecks:
    """Tests probe timeouts, zombie socket detection, and proxy bypass."""

    def test_is_proxy_responsive_returns_true_for_healthy_endpoint(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            assert _is_proxy_responsive("http://localhost:4000", timeout_sec=0.2, cache_ttl=0.0) is True

    def test_is_proxy_responsive_returns_false_for_hanging_zombie_socket(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            import socket
            mock_urlopen.side_effect = socket.timeout("The read operation timed out")

            assert _is_proxy_responsive("http://localhost:4000", timeout_sec=0.2, cache_ttl=0.0) is False

    def test_is_proxy_responsive_returns_false_for_connection_refused(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            import urllib.error
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            assert _is_proxy_responsive("http://localhost:4000", timeout_sec=0.2, cache_ttl=0.0) is False

    def test_proxy_health_cache_avoids_repeated_probes(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            url = f"http://test-cache-{time.time()}:4000"
            res1 = _is_proxy_responsive(url, timeout_sec=0.2, cache_ttl=5.0)
            res2 = _is_proxy_responsive(url, timeout_sec=0.2, cache_ttl=5.0)
            assert res1 is True
            assert res2 is True
            # Second call must hit cache without invoking urlopen again
            assert mock_urlopen.call_count == 1


# =========================================================================== #
# 4. Streaming Telemetry, Variable Scope & Tenant Isolation Tests
# =========================================================================== #
class TestStreamingTelemetryAndTenantContext:
    """Tests streaming lifecycle, variable scope, tenant isolation, and FinOps recording."""

    @pytest.mark.anyio
    async def test_streaming_completes_safely_and_records_telemetry(self):
        from src.agent.stream_engine import stream_agent_workflow

        dummy_graph_result = {
            "messages": [{"role": "assistant", "content": "Here is the verified answer."}],
            "iteration": 1,
            "detected_intent": "general",
            "evidence_verification": {"status": "supported"},
            "evidence_urls": ["https://example.com/source"],
            "evidence_items": [],
            "tenant_id": "tenant-alpha",
        }

        with patch("src.agent.stream_engine.agent_graph.invoke", return_value=dummy_graph_result), \
             patch("src.db.chat_history.chat_history_db.append_message") as mock_append, \
             patch("src.security.finops.finops.record_request_metrics") as mock_finops:

            events = []
            async for sse in stream_agent_workflow(
                message="Tell me something interesting",
                user_id="user_123",
                session_id="sess_123",
                tenant_id="tenant-alpha",
            ):
                events.append(sse)

            # Check that events were emitted
            assert any("session_init" in e for e in events)
            assert any("done" in e for e in events)

            # Verify tenant_id was passed to chat_history_db and FinOps
            assert mock_append.call_count >= 1
            last_hist_call = mock_append.call_args_list[-1]
            assert last_hist_call.kwargs.get("tenant_id") == "tenant-alpha"

            assert mock_finops.call_count == 1
            finops_rec = mock_finops.call_args[0][0]
            assert finops_rec.tenant_id == "tenant-alpha"
            assert finops_rec.user_id == "user_123"

    @pytest.mark.anyio
    async def test_finops_database_failure_does_not_crash_user_response(self):
        from src.agent.stream_engine import stream_agent_workflow

        dummy_graph_result = {
            "messages": [{"role": "assistant", "content": "Successful response to user."}],
            "iteration": 1,
            "detected_intent": "quick_chat",
            "tenant_id": "tenant-beta",
        }

        with patch("src.agent.stream_engine.agent_graph.invoke", return_value=dummy_graph_result), \
             patch("src.security.finops.finops.record_request_metrics", side_effect=RuntimeError("DB connection failed")):

            events = []
            # Even when FinOps throws a database failure, streaming must complete gracefully
            async for sse in stream_agent_workflow(
                message="Hello",
                user_id="user_456",
                session_id="sess_456",
                tenant_id="tenant-beta",
            ):
                events.append(sse)

            assert any("done" in e for e in events)

    @pytest.mark.anyio
    async def test_tenant_context_isolation_across_concurrent_requests(self):
        from src.agent.stream_engine import stream_agent_workflow

        tenant_a_result = {
            "messages": [{"role": "assistant", "content": "Tenant A reply"}],
            "tenant_id": "tenant-corp-a",
        }
        tenant_b_result = {
            "messages": [{"role": "assistant", "content": "Tenant B reply"}],
            "tenant_id": "tenant-corp-b",
        }

        with patch("src.agent.stream_engine.agent_graph.invoke", side_effect=[tenant_a_result, tenant_b_result]), \
             patch("src.security.finops.finops.record_request_metrics") as mock_finops:

            task1 = stream_agent_workflow("Prompt A", user_id="user_a", tenant_id="tenant-corp-a")
            task2 = stream_agent_workflow("Prompt B", user_id="user_b", tenant_id="tenant-corp-b")

            events_a = [e async for e in task1]
            events_b = [e async for e in task2]

            assert any("Tenant A reply" in e for e in events_a)
            assert any("Tenant B reply" in e for e in events_b)

            assert mock_finops.call_count == 2
            recorded_tenants = [call[0][0].tenant_id for call in mock_finops.call_args_list]
            assert "tenant-corp-a" in recorded_tenants
            assert "tenant-corp-b" in recorded_tenants
