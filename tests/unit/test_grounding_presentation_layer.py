"""
Unit tests for the Two-Layer Grounding Verification Architecture:
1. Internal Verification Layer: Retains raw machine-readable data (claim_id, source_claim,
   quote_match, issues, confidence, evidence_ids, debugging details) for telemetry/auditing.
2. User-Facing Presentation Layer: Deterministically converts results into concise,
   readable warnings with clean markdown links, hiding internal IDs and technical jargon.
"""
from __future__ import annotations

import json
import pytest

from src.security.grounding_presentation import (
    GroundingPresentationFormatter,
    InternalClaimVerification,
    VerificationCategory,
    format_clickable_source_link,
    grounding_presentation_formatter,
)
from src.security.evidence_store import (
    Claim,
    Evidence,
    GroundingVerificationResult,
    verify_response_grounding,
)
from src.security.output_guards import (
    sanitize_output,
    sanitize_user_facing_output,
)


class TestInternalVerificationLayer:
    """Verifies that internal verification layer preserves all raw debugging and telemetry data."""

    def test_internal_verification_model_retains_all_raw_details(self):
        record = InternalClaimVerification(
            claim_id="CLM-98214",
            status="CONTRADICTED",
            claim="Python 3.12 introduced feature X",
            source_claim="Python 3.11",
            quote_match=False,
            issues=["Version mismatch", "Quoted evidence not found"],
            source_url="https://docs.python.org/release/3.11.html",
            source_title="Python 3.11 Release Notes",
            confidence=0.88,
            evidence_ids=["EVID-001", "EVID-002"],
            raw_debugging_details={"latency_ms": 12.4, "eval_engine": "heuristic_regex"},
        )

        audit_dict = record.to_audit_dict()

        # All machine-readable fields preserved
        assert audit_dict["claim_id"] == "CLM-98214"
        assert audit_dict["status"] == "contradicted"
        assert audit_dict["claim"] == "Python 3.12 introduced feature X"
        assert audit_dict["source_claim"] == "Python 3.11"
        assert audit_dict["quote_match"] is False
        assert "Version mismatch" in audit_dict["issues"]
        assert "Quoted evidence not found" in audit_dict["issues"]
        assert audit_dict["source_url"] == "https://docs.python.org/release/3.11.html"
        assert audit_dict["source_title"] == "Python 3.11 Release Notes"
        assert audit_dict["confidence"] == 0.88
        assert audit_dict["evidence_ids"] == ["EVID-001", "EVID-002"]
        assert audit_dict["raw_debugging_details"]["latency_ms"] == 12.4

    def test_verify_response_grounding_preserves_internal_verifications(self):
        evidence_items = [
            {
                "url": "https://docs.python.org/3/whatsnew/3.11.html",
                "source_title": "Python 3.11 Documentation",
                "content": "Python 3.11 was released in October 2022 with improved exception groups.",
            }
        ]
        text = "Python 3.12 introduced feature X, according to Python 3.11 Documentation."
        result = verify_response_grounding(text, evidence_items=evidence_items)

        assert isinstance(result, GroundingVerificationResult)
        assert len(result.internal_verifications) >= 1
        internal_record = result.internal_verifications[0]

        # Machine-readable properties exist in internal layer
        assert "claim_id" in internal_record
        assert "status" in internal_record
        assert "confidence" in internal_record
        assert "evidence_ids" in internal_record


class TestUserFacingPresentationLayer:
    """Verifies that user presentation layer converts raw results into plain language."""

    def test_raw_json_converted_into_readable_text(self):
        """User Prompt Scenario: Raw JSON dict converted to readable warning without curly braces."""
        raw_verifier_dict = {
            "status": "contradicted",
            "claim": "Python 3.12 introduced feature X",
            "source_claim": "Python 3.11",
            "quote_match": False,
            "issues": [
                "Version mismatch",
                "Quoted evidence not found",
            ],
            "source_url": "https://docs.python.org/3/whatsnew/3.11.html",
            "source_title": "Python 3.11 Documentation",
        }

        banner = grounding_presentation_formatter.generate_consolidated_warning_banner([raw_verifier_dict])

        # Must not contain JSON syntax
        assert "{" not in banner
        assert "}" not in banner
        assert '"status":' not in banner
        assert '"claim":' not in banner

        # Must contain readable plain-language warning
        assert "⚠️ **Source verification warning**" in banner
        assert "Python 3.11" in banner
        assert "Python 3.12" in banner
        assert "does not verify the claim" in banner
        assert "Please check the correct Python 3.12 documentation." in banner

    def test_internal_field_names_and_ids_are_hidden(self):
        """Internal IDs such as CLM-..., EVID-..., and 'id' must be hidden from user text."""
        verifications = [
            InternalClaimVerification(
                claim_id="CLM-7890123",
                status="CONTRADICTED",
                claim="Model accuracy is 99%",
                source_claim="82%",
                issues=["CLM-7890123: Numeric conflict with EVID-4451"],
                source_url="https://benchmark.org",
                source_title="Benchmark Report",
                evidence_ids=["EVID-4451"],
            )
        ]

        banner = grounding_presentation_formatter.generate_consolidated_warning_banner(verifications)

        # Internal IDs should be hidden
        assert "CLM-7890123" not in banner
        assert "EVID-4451" not in banner
        assert "claim_id" not in banner
        assert "evidence_ids" not in banner

    def test_repeated_contradiction_warnings_are_consolidated(self):
        """Repeated contradiction warnings must produce ONE consolidated warning header."""
        repeated_records = [
            InternalClaimVerification(
                claim_id="CLM-001",
                status="CONTRADICTED",
                claim="Python 3.12 has feature A",
                source_claim="Python 3.11",
                source_url="https://docs.python.org",
            ),
            InternalClaimVerification(
                claim_id="CLM-002",
                status="CONTRADICTED",
                claim="Python 3.12 has feature B",
                source_claim="Python 3.11",
                source_url="https://docs.python.org",
            ),
        ]

        banner = grounding_presentation_formatter.generate_consolidated_warning_banner(repeated_records)

        # Only one warning header appears
        assert banner.count("⚠️ **Source verification warning**") == 1
        # Bulleted clean items
        assert "• **Conflicting Evidence:**" in banner

    def test_repeated_quoted_span_not_found_summarized_as_citation_mismatch(self):
        """
        Repeated quoted span errors must be summarized as citation mismatches,
        and clearly distinguished from genuine factual contradictions.
        """
        mismatches = [
            InternalClaimVerification(
                claim_id="CLM-003",
                status="QUOTE_MISMATCH",
                claim="The CEO said 'our market share doubled overnight'",
                quote_match=False,
                issues=["Quoted span 'our market share doubled overnight' not found"],
                source_url="https://news.com/article",
            ),
            InternalClaimVerification(
                claim_id="CLM-004",
                status="QUOTE_MISMATCH",
                claim="The report concluded 'growth was completely unprecedented'",
                quote_match=False,
                issues=["Quoted span 'growth was completely unprecedented' not found"],
                source_url="https://news.com/article",
            ),
        ]

        banner = grounding_presentation_formatter.generate_consolidated_warning_banner(mismatches)

        assert "• **Citation Mismatch:**" in banner
        assert "2 cited quotations could not be located verbatim" in banner
        # Must clearly state that claim is not necessarily false
        assert "underlying statements may still be valid" in banner
        assert "exact wording could not be confirmed" in banner

    def test_raw_urls_converted_into_clean_clickable_markdown_links(self):
        """Raw URLs mixed with errors must become clean clickable markdown links."""
        link_with_title = format_clickable_source_link(
            "https://docs.python.org/3.12/whatsnew.html",
            source_title="Official Python 3.12 Documentation",
        )
        assert link_with_title == "[Official Python 3.12 Documentation](https://docs.python.org/3.12/whatsnew.html)"

        link_without_title = format_clickable_source_link("https://www.github.com/astral-sh/uv")
        assert link_without_title == "[github.com](https://www.github.com/astral-sh/uv)"

    def test_predefined_categories_deterministic_formatting(self):
        """Verifies deterministic formatting for all predefined categories."""
        # 1. SUPPORTED -> No warning
        supported = [InternalClaimVerification(claim_id="C1", status="SUPPORTED", claim="Valid claim")]
        assert grounding_presentation_formatter.generate_consolidated_warning_banner(supported) == ""

        # 2. INSUFFICIENT_EVIDENCE
        insufficient = [
            InternalClaimVerification(
                claim_id="C2",
                status="INSUFFICIENT_EVIDENCE",
                claim="Topic X solves problem Y",
                source_url="https://research.org",
            )
        ]
        banner_insufficient = grounding_presentation_formatter.generate_consolidated_warning_banner(insufficient)
        assert "• **Unverified Claim:**" in banner_insufficient
        assert "does not provide sufficient evidence" in banner_insufficient

        # 3. SOURCE_UNAVAILABLE
        unavailable = [
            InternalClaimVerification(
                claim_id="C3",
                status="SOURCE_UNAVAILABLE",
                claim="Topic Z",
                source_url="https://unreachable-source.com",
            )
        ]
        banner_unavailable = grounding_presentation_formatter.generate_consolidated_warning_banner(unavailable)
        assert "• **Source Unavailable:**" in banner_unavailable
        assert "could not be accessed or verified" in banner_unavailable


class TestSanitizeUserFacingOutputIntegration:
    """Verifies end-to-end integration with sanitize_user_facing_output."""

    def test_sanitize_user_facing_output_applies_presentation_banner(self):
        evidence_items = [
            {
                "url": "https://docs.python.org/3.11",
                "source_title": "Python 3.11 Docs",
                "content": "Python 3.11 was released with improved performance.",
            }
        ]
        raw_text = "According to Python 3.11 Docs, Python 3.12 was released with improved performance."
        clean = sanitize_user_facing_output(raw_text, evidence_items=evidence_items)

        # Warning is appended as clean markdown presentation
        assert "⚠️ **Source verification warning**" in clean
        assert "Python 3.11" in clean
        assert "Python 3.12" in clean
        # No raw technical brackets like '[Claim attribution corrected:' in presentation mode
        assert "[Claim attribution corrected:" not in clean

    def test_sanitize_output_backward_compatibility_preserved(self):
        """Ensure standard sanitize_output without user_facing preserves legacy format if needed."""
        evidence_items = [
            {
                "url": "https://website-a.com",
                "source": "Website A",
                "content": "Website A states accuracy is 85%.",
            },
            {
                "url": "https://website-b.com",
                "source": "Website B",
                "content": "Website B states accuracy is 92%.",
            },
        ]
        raw_text = "Model X achieves 92% accuracy, according to Website A."
        clean_legacy = sanitize_output(raw_text, evidence_items=evidence_items, user_facing=False)
        assert "Claim attribution corrected" in clean_legacy


class TestInputOutputGrounding:
    """Verifies Input-to-Output Grounding (Relevance, Instruction Adherence & Topic Drift)."""

    def test_unsolicited_comparative_essay_stripped_for_memory_instruction(self):
        """User Prompt Scenario: Memory instruction generates unsolicited Redis vs Kafka essay."""
        from src.security.output_guards import verify_input_output_grounding

        prompt = "Remember that my team's primary cloud provider is AWS and we strictly use PostgreSQL with pgvector for embeddings."
        unsolicited_response = (
            "Here is the comparison of Redis Streams vs. Apache Kafka for an autonomous agent event queue, "
            "specifically tailored to your AWS and PostgreSQL/pgvector context.\n\n"
            "1. Latency: Redis offers sub-millisecond latency. Kafka is disk-optimized.\n"
            "2. Operational Overhead: Redis on ElastiCache is lighter. MSK has higher operational surface.\n"
            "Summary Recommendation: Choose Redis Streams for real-time responsiveness."
        )

        clean, has_violations, violations = verify_input_output_grounding(
            prompt=prompt,
            response_text=unsolicited_response,
            intent_category="memory_instruction",
        )

        assert has_violations is True
        assert len(violations) >= 1
        assert "unsolicited comparative essay" in violations[0]
        # Replaces stale essay with clean, grounded acknowledgment
        assert "Understood! I have recorded this preference in memory" in clean
        assert "AWS" in clean or "aws" in clean
        assert "Redis Streams" not in clean

    def test_sanitize_output_integrates_input_output_grounding(self):
        prompt = "Remember that my team's primary cloud provider is AWS and we strictly use PostgreSQL with pgvector for embeddings."
        stale_output = "Here is the comparison of Redis Streams vs. Apache Kafka for an autonomous agent event queue. Operational overhead..."
        clean = sanitize_output(stale_output, prompt=prompt, intent_category="memory_instruction")
        assert "Understood! I have recorded this preference in memory" in clean
        assert "Redis Streams" not in clean
