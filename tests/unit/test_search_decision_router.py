"""
Unit tests for Search Decision Router in Hermes Agent Platform.

Tests the 4-check decision process and 3-signal triangulation:
1. Check 1: Context sufficiency (no search when code or documents are provided)
2. Check 2: Freshness dependency (recency, versions, current prices/facts)
3. Check 3: Explicit search intent (direct user request for web research)
4. Check 4: Evidence requirement (verifying claims, comparative benchmarks)
5. Search domain classification (web, news, academic, documentation, local, shopping)
6. Clean query formulation
7. Auditable decision logging
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.agent.search_router import (
    SearchDecisionRouter,
    SearchType,
    search_decision_router,
)
from src.security.audit import SecurityAuditLogger


class TestSearchDecisionRouterChecks:
    @pytest.fixture
    def router(self) -> SearchDecisionRouter:
        return SearchDecisionRouter()

    # --- Test Check 1: Context Sufficiency ---

    def test_explain_provided_python_code_requires_no_search(self, router: SearchDecisionRouter):
        prompt = """Explain this Python code:
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```"""
        decision = router.evaluate(prompt)
        assert decision.requires_search is False
        assert decision.search_type == SearchType.NONE
        assert decision.signals.context_sufficient is True
        assert "context" in decision.reason.lower()

    def test_uploaded_attachment_bypasses_web_search(self, router: SearchDecisionRouter):
        prompt = "Summarize this uploaded paper and critique the ablation study."
        decision = router.evaluate(prompt, has_attachments=True)
        assert decision.requires_search is False
        assert decision.signals.context_sufficient is True
        assert "uploaded" in decision.reason.lower()

    # --- Test Stable Conceptual Knowledge ---

    def test_explain_backpropagation_requires_no_search(self, router: SearchDecisionRouter):
        prompt = "Explain backpropagation and how gradients propagate through hidden layers."
        decision = router.evaluate(prompt)
        assert decision.requires_search is False
        assert decision.search_type == SearchType.NONE
        assert "stable" in decision.reason.lower() or "foundational" in decision.reason.lower()

    # --- Test Check 2: Freshness & Time-Varying Data ---

    def test_latest_pytorch_api_requires_documentation_search(self, router: SearchDecisionRouter):
        prompt = "What is the latest PyTorch API for distributed pipeline parallelism?"
        decision = router.evaluate(prompt)
        assert decision.requires_search is True
        assert decision.search_type == SearchType.DOCUMENTATION
        assert decision.signals.freshness_score >= 0.4
        assert any("pytorch" in q.lower() for q in decision.queries)

    def test_net_worth_inquiry_requires_web_search(self, router: SearchDecisionRouter):
        prompt = "what are the net worths of Anirudh Ravichandher and Kavya Maran respectively"
        decision = router.evaluate(prompt)
        assert decision.requires_search is True
        assert decision.signals.freshness_score >= 0.4
        assert any("net worth" in q.lower() for q in decision.queries)

    def test_version_support_maintenance_inquiry_triggers_search(self, router: SearchDecisionRouter):
        prompt = "Is this library still maintained and supported in 2026?"
        decision = router.evaluate(prompt)
        assert decision.requires_search is True
        assert decision.signals.freshness_score >= 0.4

    # --- Test Check 3: Explicit Search Intent ---

    def test_academic_paper_search_intent(self, router: SearchDecisionRouter):
        prompt = "Find latest VLM interpretability papers on arXiv."
        decision = router.evaluate(prompt)
        assert decision.requires_search is True
        assert decision.search_type == SearchType.ACADEMIC
        assert decision.signals.intent_score == 1.0
        assert any("interpretability" in q.lower() for q in decision.queries)

    def test_explicit_web_search_request(self, router: SearchDecisionRouter):
        prompt = "Search the web for the Q3 2024 earnings releases for Spotify and Netflix."
        decision = router.evaluate(prompt)
        assert decision.requires_search is True
        assert decision.signals.intent_score == 1.0

    # --- Test Check 4: Evidence & Comparison Requirements ---

    def test_model_comparison_requires_search(self, router: SearchDecisionRouter):
        prompt = "Compare GPT-6 with Claude's latest model on coding benchmarks."
        decision = router.evaluate(prompt)
        assert decision.requires_search is True
        assert decision.signals.evidence_score >= 0.4 or decision.signals.freshness_score >= 0.4

    def test_local_business_search_classification(self, router: SearchDecisionRouter):
        prompt = "Find Italian restaurants near me open for dinner."
        decision = router.evaluate(prompt)
        assert decision.requires_search is True
        assert decision.search_type == SearchType.LOCAL

    # --- Test Signal Triangulation & Query Formulations ---

    def test_clean_query_extraction_removes_filler(self, router: SearchDecisionRouter):
        prompt = "Please can you search the web for current market cap of Nvidia?"
        decision = router.evaluate(prompt)
        assert decision.requires_search is True
        assert len(decision.queries) >= 1
        assert not decision.queries[0].lower().startswith("please can you search the web for")
        assert "market cap of nvidia" in decision.queries[0].lower()


class TestAuditableSearchDecisionLogging:
    def test_search_decision_logged_to_audit_trail(self, tmp_path: Path):
        log_file = tmp_path / "audit_test.jsonl"
        audit = SecurityAuditLogger(log_path=log_file)

        decision = search_decision_router.evaluate("Find 10 recent papers on agentic VLM interpretability")
        entry = audit.record_search_decision(
            prompt="Find 10 recent papers on agentic VLM interpretability",
            decision=decision.to_dict(),
            session_id="sess_audit_test",
            user_id="researcher_1",
        )

        assert entry["event_type"] == "search_decision"
        assert entry["requires_search"] is True
        assert entry["search_type"] == "academic"
        assert entry["session_id"] == "sess_audit_test"

        # Verify persistent disk logging
        assert log_file.exists()
        lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        assert lines[0]["event_type"] == "search_decision"
        assert lines[0]["requires_search"] is True
