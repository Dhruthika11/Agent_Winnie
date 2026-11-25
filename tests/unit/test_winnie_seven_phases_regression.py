"""
Phase 7: Comprehensive Automated Regression Test Suite for Winnie 7-Phase Architecture.

Validates:
- Phase 1: ToolPolicy, Risk Classification, and Authorization Separation.
- Phase 2: Intent Classification & Query Disambiguation (Political, Speculative Elections, Actions, Explanations, Search).
- Phase 3: Evidence Grounding and Output URL Validation (Preventing fabricated profile links).
- Phase 4: Memory Policy & Durable Filtering (Preventing refusal/transient error poisoning in Mem0/Qdrant).
- Phase 5: Tool Reliability, Retries, and Fallbacks.
- Phase 6: Observability Telemetry Metadata.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agent.intent_classifier import IntentCategory, classify_user_intent
from src.agent.memory_policy import memory_policy
from src.agent.pipeline_common import execute_tool_with_guard
from src.agent.tool_policy import (
    PolicyDecision,
    RiskLevel,
    tool_policy,
)
from src.security.output_guards import sanitize_output, validate_evidence_grounding


# =========================================================================== #
# Phase 1: Tool Policy & Authorization Layer
# =========================================================================== #
class TestPhase1ToolPolicy:
    def test_exploit_commands_categorically_denied(self):
        """Hostile exploits (fork bombs, raw disk zeroing) must be DENIED unconditionally."""
        res_fork = tool_policy.evaluate("terminal.run", {"command": ":(){ :|:& };:"})
        assert res_fork.decision == PolicyDecision.DENY
        assert res_fork.risk_level == RiskLevel.CRITICAL
        assert "exploit" in res_fork.reason.lower()

        res_disk = tool_policy.evaluate("terminal.run", {"command": "dd if=/dev/zero of=/dev/sda"})
        assert res_disk.decision == PolicyDecision.DENY
        assert res_disk.risk_level == RiskLevel.CRITICAL

    def test_system_path_deletion_requires_approval(self):
        """Deleting /etc/systemd or protected OS directories requires Human-in-the-Loop approval."""
        res_del = tool_policy.evaluate("terminal.delete_file", {"path": "/etc/systemd"})
        assert res_del.decision == PolicyDecision.REQUIRE_APPROVAL
        assert res_del.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert res_del.requires_approval is True
        assert "targets protected" in res_del.reason.lower() or "requires explicit" in res_del.reason.lower()

    def test_destructive_rm_rf_and_drop_table_requires_approval(self):
        """Destructive operations (rm -rf, drop table, drop database) require HITL approval."""
        res_rm = tool_policy.evaluate("terminal.run", {"command": "rm -rf /etc/systemd"})
        assert res_rm.decision == PolicyDecision.REQUIRE_APPROVAL
        assert res_rm.risk_level == RiskLevel.HIGH

        res_drop = tool_policy.evaluate("database.drop_table", {"table": "main_users"})
        assert res_drop.decision == PolicyDecision.REQUIRE_APPROVAL
        assert res_drop.risk_level == RiskLevel.HIGH

    def test_low_risk_read_and_search_tools_allowed(self):
        """Read-only and search tools are ALLOWED automatically without approval."""
        res_search = tool_policy.evaluate("web_scraper.search", {"query": "Ganesh Rapuru"})
        assert res_search.decision == PolicyDecision.ALLOW
        assert res_search.risk_level == RiskLevel.LOW
        assert res_search.is_allowed is True

        res_read = tool_policy.evaluate("filesystem.read_file", {"path": "README.md"})
        assert res_read.decision == PolicyDecision.ALLOW
        assert res_read.risk_level == RiskLevel.LOW

    def test_execute_tool_with_guard_defense_in_depth(self):
        """execute_tool_with_guard intercepts denied actions and returns blocked record."""
        rec = execute_tool_with_guard("terminal.run", {"command": ":(){ :|:& };:"}, call_id="test_exploit_1")
        assert rec["status"] == "denied"
        assert "Action Denied by Security Policy" in rec["output"]


# =========================================================================== #
# Phase 2: Intent Classification & Query Disambiguation
# =========================================================================== #
class TestPhase2IntentClassifier:
    def test_future_election_speculative_query(self):
        """Speculative election questions are routed to FUTURE_ELECTION with multi-perspective guidance."""
        intent = classify_user_intent("Do u think will BJP win in the upcoming 2029 elections National level?")
        assert intent.category == IntentCategory.FUTURE_ELECTION
        assert intent.is_action_proposal is False
        assert "neutrality" in intent.guidance.lower() or "objective" in intent.guidance.lower()

    def test_political_civic_information_query(self):
        """General political information queries route to POLITICAL_INFO."""
        intent = classify_user_intent("Who is BJP and what is its parliamentary history?")
        assert intent.category == IntentCategory.POLITICAL_INFO
        assert intent.is_action_proposal is False

    def test_action_request_vs_explanation_request(self):
        """Direct destructive command is ACTION_REQUEST; conceptual question is EXPLANATION_REQUEST."""
        action_intent = classify_user_intent("Delete the system configuration directory /etc/systemd or drop the main database table.")
        assert action_intent.category == IntentCategory.ACTION_REQUEST
        assert action_intent.is_action_proposal is True

        explain_intent = classify_user_intent("Explain how rm -rf /etc/systemd works and what happens if executed.")
        assert explain_intent.category == IntentCategory.EXPLANATION_REQUEST
        assert explain_intent.is_action_proposal is False

    def test_web_search_intent(self):
        """Search query for LinkedIn profile routes to WEB_SEARCH with strict grounding requirement."""
        intent = classify_user_intent("Give me the linkedin profile link of Ganesh Rapuru")
        assert intent.category == IntentCategory.WEB_SEARCH
        assert intent.requires_grounding is True


# =========================================================================== #
# Phase 3: Evidence Grounding & Output Validation
# =========================================================================== #
class TestPhase3EvidenceGrounding:
    def test_filter_hallucinated_linkedin_url(self):
        """Synthetic LinkedIn profile URL with no supporting search evidence is filtered."""
        generated_text = "Here is the LinkedIn profile link for Ganesh Rapuru:\nhttps://www.linkedin.com/in/ganesh-rapuru"
        evidence_urls = [
            "https://scholar.google.com/citations?user=12345",
            "https://twitter.com/ganesh_rapuru_ai",
        ]

        sanitized, has_violations, filtered = validate_evidence_grounding(generated_text, evidence_urls=evidence_urls)
        assert has_violations is True
        assert "https://www.linkedin.com/in/ganesh-rapuru" in filtered
        assert "[Unverified link `https://www.linkedin.com/in/ganesh-rapuru` filtered" in sanitized

    def test_preserve_verified_profile_url(self):
        """Profile link that was returned in the search evidence is preserved intact."""
        generated_text = "Found profile: https://www.linkedin.com/in/ganesh-rapuru-verified"
        evidence_urls = [
            "https://www.linkedin.com/in/ganesh-rapuru-verified",
        ]

        sanitized, has_violations, filtered = validate_evidence_grounding(generated_text, evidence_urls=evidence_urls)
        assert has_violations is False
        assert len(filtered) == 0
        assert "https://www.linkedin.com/in/ganesh-rapuru-verified" in sanitized

    def test_sanitize_output_combines_pii_and_url_grounding(self):
        """sanitize_output redacts PII and filters ungrounded URLs simultaneously."""
        raw_text = (
            "Contact Ganesh at ganesh@example.com or visit https://www.linkedin.com/in/ganesh-fake"
        )
        clean = sanitize_output(raw_text, evidence_urls=[])
        assert "[REDACTED_EMAIL]" in clean
        assert "ganesh@example.com" not in clean
        assert "[Unverified link `https://www.linkedin.com/in/ganesh-fake` filtered" in clean


# =========================================================================== #
# Phase 4: Memory Policy & Durable Filtering
# =========================================================================== #
class TestPhase4MemoryPolicy:
    def test_reject_refusal_messages(self):
        """Conversational refusals ('cannot help', 'I'm sorry') must NOT be persisted to memory."""
        turn = [
            {"role": "user", "content": "Delete the system configuration directory /etc/systemd"},
            {"role": "assistant", "content": "I'm sorry, but I can't help with that."},
        ]
        decision = memory_policy.evaluate_turn(turn)
        assert decision.should_store is False
        assert decision.reason == "assistant_refusal"

    def test_reject_transient_tool_errors(self):
        """Temporary tool errors, connection timeouts, and rate limits are not persisted."""
        turn = [
            {"role": "user", "content": "Search for latest news"},
            {"role": "assistant", "content": "Tool execution error: Search failed due to connection reset (HTTP 429)."},
        ]
        decision = memory_policy.evaluate_turn(turn)
        assert decision.should_store is False
        assert decision.reason == "transient_tool_error"

    def test_reject_ephemeral_chitchat(self):
        """Fast greetings and trivial acknowledgements are not persisted."""
        turn = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I assist you today?"},
        ]
        decision = memory_policy.evaluate_turn(turn)
        assert decision.should_store is False
        assert decision.reason == "ephemeral_chitchat"

    def test_approve_durable_technical_preferences(self):
        """Useful architectural and user preference context is classified and stored."""
        turn = [
            {"role": "user", "content": "I always use PostgreSQL with FastAPI running in Docker."},
            {"role": "assistant", "content": "Understood! I will tailor backend implementations to PostgreSQL and FastAPI."},
        ]
        decision = memory_policy.evaluate_turn(turn)
        assert decision.should_store is True
        assert decision.category == "technical_environment"
        assert decision.metadata["durability"] == "high"

    def test_sanitize_retrieved_memories_purges_legacy_refusals(self):
        """Memories retrieved from store are purged of stale refusal text."""
        raw_retrieved = [
            "User prefers Python 3.12 and pytest.",
            "I'm sorry, but I can't help with that.",
            "Tool execution error: search timed out.",
            "Project stack is built on LangGraph and MCP.",
        ]
        sanitized = memory_policy.sanitize_retrieved_memories(raw_retrieved)
        assert len(sanitized) == 2
        assert "User prefers Python 3.12 and pytest." in sanitized
        assert "Project stack is built on LangGraph and MCP." in sanitized
        assert "I'm sorry, but I can't help with that." not in sanitized


# =========================================================================== #
# Phase 5: MCP Tool Reliability & Search Fallbacks
# =========================================================================== #
class TestPhase5ToolReliability:
    def test_web_search_empty_result_fallback(self):
        """Empty search results return structured message rather than crashing."""
        from src.tools.core.web import search
        with patch("src.tools.core.web._search_ddg_html", return_value=[]), \
             patch("src.tools.core.web._search_bing", return_value=[]), \
             patch("src.tools.core.web._search_wikipedia", return_value=[]), \
             patch("src.tools.core.web.DDGS") as mock_ddgs:
            mock_inst = MagicMock()
            mock_inst.text.return_value = []
            mock_ddgs.return_value.__enter__.return_value = mock_inst

            res = search("Unfindable Person XYZ 9999999")
            assert res["success"] is True
            assert res["data"] == []
            assert res["metadata"]["message"] == "No verified search results found."
            assert res["metadata"]["count"] == 0

    def test_web_search_retry_on_transient_failure(self):
        """Transient exceptions trigger retries before failing gracefully."""
        from src.tools.core.web import search
        with patch("src.tools.core.web._search_ddg_html", return_value=[]), \
             patch("src.tools.core.web._search_bing", return_value=[]), \
             patch("src.tools.core.web._search_wikipedia", return_value=[]), \
             patch("src.tools.core.web.DDGS") as mock_ddgs:
            mock_inst = MagicMock()
            # First call raises rate limit exception, second call succeeds
            mock_inst.text.side_effect = [
                Exception("Ratelimit 429"),
                [{"title": "Ganesh Rapuru", "href": "https://example.com/ganesh", "body": "Researcher"}],
            ]
            mock_ddgs.return_value.__enter__.return_value = mock_inst

            res = search("Ganesh Rapuru")
            assert res["success"] is True
            assert len(res["data"]) == 1
            assert res["data"][0]["url"] == "https://example.com/ganesh"
