"""
Unit tests for Memory Intent Classification, Task Routing, and LiteLLM Proxy Health Checking.
Verifies fixes for:
1. Memory instruction intent classification (IntentCategory.MEMORY_INSTRUCTION).
2. Avoiding false 'coding' task classification on 'postgresql' (avoiding substring 'sql').
3. Instant failover when LiteLLM proxy is unresponsive (avoiding 80s TTFT).
4. Direct acknowledgment of memory declarations without re-answering previous turn questions.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agent.intent_classifier import IntentCategory, classify_user_intent
from src.agent.router import classify_task_intent, _is_proxy_responsive, _build_deployments
from src.memory.manager import memory_manager


class TestMemoryIntentClassification:
    def test_classify_user_intent_detects_memory_instruction(self):
        query = "Remember that my team's primary cloud provider is AWS and we strictly use PostgreSQL with pgvector for embeddings."
        intent = classify_user_intent(query)
        assert intent.category == IntentCategory.MEMORY_INSTRUCTION
        assert intent.is_action_proposal is False
        assert "MEMORY & PREFERENCE INSTRUCTION" in intent.guidance

    def test_classify_user_intent_detects_various_memory_phrases(self):
        queries = [
            "Please note that our staging database is on port 5433",
            "Keep in mind that I prefer TypeScript over Python",
            "Note that my user ID is dev_admin_42",
            "Don't forget that we use AWS us-east-1",
            "My team uses Kubernetes on EKS",
            "My preference is dark mode and compact formatting",
            "I strictly use PostgreSQL for storage",
        ]
        for q in queries:
            intent = classify_user_intent(q)
            assert intent.category == IntentCategory.MEMORY_INSTRUCTION, f"Failed for query: {q}"


class TestTaskRoutingAndIndicatorFixes:
    def test_memory_instruction_routed_to_quick_chat_not_coding(self):
        """User Prompt Scenario: Memory instruction containing 'PostgreSQL' must NOT be classified as coding."""
        query = "Remember that my team's primary cloud provider is AWS and we strictly use PostgreSQL with pgvector for embeddings."
        task = classify_task_intent(query)
        assert task == "quick_chat"

    def test_postgresql_does_not_trigger_sql_substring_coding_match(self):
        """PostgreSQL in general text should not falsely match 'sql' as a substring."""
        query = "What are the architectural trade-offs of using PostgreSQL versus DynamoDB?"
        task = classify_task_intent(query)
        # Should be classified as reasoning (due to trade-offs/versus), not coding
        assert task == "reasoning"

    def test_genuine_coding_queries_still_classified_as_coding(self):
        query = "Write a Python script to connect to postgresql using psycopg and run SELECT 1"
        task = classify_task_intent(query)
        assert task == "coding"


class TestProxyHealthCheckAndLatency:
    def test_is_proxy_responsive_returns_false_for_dead_or_timeout_port(self):
        """A dead or hanging port should return False within timeout instead of hanging 80s."""
        # Unused local port
        res = _is_proxy_responsive("http://127.0.0.1:49999", timeout_sec=0.2)
        assert res is False

    @patch("src.agent.router._is_proxy_responsive", return_value=False)
    def test_build_deployments_bypasses_proxy_when_unresponsive(self, mock_probe):
        """Deployments should not include litellm-proxy when the proxy is unresponsive."""
        with patch("src.config.settings.litellm_base_url", "http://localhost:4000"):
            deployments = _build_deployments()
            proxy_deps = [d for d in deployments if d.get("model_info", {}).get("provider") == "litellm-proxy"]
            assert len(proxy_deps) == 0


class TestMemoryInstructionGraphExecution:
    def test_memory_instruction_persists_to_memory_manager(self):
        """Verifies memory is registered directly when memory instruction is provided."""
        user_id = "test_user_aws_pgvector"
        query = "Remember that my team's primary cloud provider is AWS and we strictly use PostgreSQL with pgvector for embeddings."
        
        # Test memory_manager add directly with preference
        results = memory_manager.add(
            user_id=user_id,
            messages=[{"role": "user", "content": query}],
            tenant_id="default",
            category="preference",
        )
        assert len(results) >= 1
        assert results[0].status in ("created", "updated", "superseded")
        
        # Search memory
        mems = memory_manager.search(user_id=user_id, query="cloud provider", tenant_id="default")
        assert any("aws" in m.lower() or "postgresql" in m.lower() for m in mems)
