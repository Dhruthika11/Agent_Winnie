"""
Central Evidence Sufficiency and Claim Verifier for Winnie Agent.
Evaluates whether collected evidence is sufficient to answer the user query,
detects contradictions across sources, identifies information gaps, and
guides the final response synthesis.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.tools.validators.base import EvidenceItem, ValidationStatus

logger = logging.getLogger("hermes.agent.evidence_verifier")


@dataclass
class EvidenceSufficiencyReport:
    """Evaluation of evidence sufficiency for an agent turn."""
    is_sufficient: bool
    confidence: float
    total_evidence_count: int
    verified_count: int
    failed_count: int
    contradictions: list[str] = field(default_factory=list)
    information_gaps: list[str] = field(default_factory=list)
    synthesis_guidance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_sufficient": self.is_sufficient,
            "confidence": round(self.confidence, 3),
            "total_evidence_count": self.total_evidence_count,
            "verified_count": self.verified_count,
            "failed_count": self.failed_count,
            "contradictions": self.contradictions,
            "information_gaps": self.information_gaps,
            "synthesis_guidance": self.synthesis_guidance,
        }


class CentralEvidenceVerifier:
    """Evaluates collected evidence against user query requirements."""

    def verify(
        self,
        query: str,
        evidence_items: list[EvidenceItem],
    ) -> EvidenceSufficiencyReport:
        clean_query = (query or "").strip().lower()
        total_count = len(evidence_items)

        if not evidence_items:
            # Query did not use tools or no evidence was collected
            return EvidenceSufficiencyReport(
                is_sufficient=True,
                confidence=1.0,
                total_evidence_count=0,
                verified_count=0,
                failed_count=0,
                synthesis_guidance="No external tool evidence required. Answer directly using verified internal knowledge.",
            )

        verified_items = [
            e for e in evidence_items
            if e.get("validation_status") == ValidationStatus.VERIFIED.value
        ]
        flagged_items = [
            e for e in evidence_items
            if e.get("validation_status") == ValidationStatus.FLAGGED.value
        ]
        failed_items = [
            e for e in evidence_items
            if e.get("validation_status") == ValidationStatus.FAILED.value
        ]

        verified_count = len(verified_items)
        failed_count = len(failed_items)
        gaps: list[str] = []
        contradictions: list[str] = []

        # 1. Total Failure Check (All tools failed / returned 0 results)
        if failed_count == total_count:
            gaps.append("All executed tools returned empty results or execution failures.")
            guidance = (
                "CRITICAL EVIDENCE GAP: Tool execution returned no verified results. "
                "You MUST explicitly inform the user that no verified evidence was found for their request. "
                "Do NOT guess, fabricate, extrapolate, or construct unverified information."
            )
            return EvidenceSufficiencyReport(
                is_sufficient=False,
                confidence=0.1,
                total_evidence_count=total_count,
                verified_count=0,
                failed_count=failed_count,
                information_gaps=gaps,
                synthesis_guidance=guidance,
            )

        # Combine text content from all verified and flagged items
        evidence_corpus = " ".join(
            str(e.get("content", "")) for e in (verified_items + flagged_items)
        ).lower()

        # 2. Targeted Information Gap Detection
        # A. Profile lookup check (e.g. LinkedIn / GitHub)
        if "linkedin" in clean_query:
            has_linkedin = "linkedin.com/in/" in evidence_corpus or "linkedin" in evidence_corpus
            if not has_linkedin:
                gaps.append("No verified LinkedIn profile link found in retrieved search results.")

        if "github" in clean_query and ("repo" in clean_query or "link" in clean_query or "url" in clean_query):
            has_github = "github.com/" in evidence_corpus
            if not has_github:
                gaps.append("No verified GitHub repository link found in search evidence.")

        # B. Calculation check
        if any(w in clean_query for w in ("calculate", "sum", "multiply", "divide", "*", "+", "math")):
            has_math_evidence = any(e.get("type") in ("math", "code_execution") for e in verified_items)
            if not has_math_evidence and failed_count > 0:
                gaps.append("Calculation tool execution failed or was not conclusive.")

        # 3. Contradiction Detection across sources
        # Detect conflicting numbers for the same query entity
        if "rate" in clean_query or "price" in clean_query or "version" in clean_query:
            extracted_numbers = re.findall(r"\b\d+\.?\d*\b", evidence_corpus)
            if len(set(extracted_numbers)) > 5:
                contradictions.append("Multiple differing figures found across search sources.")

        # 4. Synthesize Sufficiency and Confidence
        is_sufficient = (verified_count > 0) and (len(gaps) == 0)
        confidence = 0.95 if is_sufficient else (0.5 if verified_count > 0 else 0.1)

        # 5. Build dynamic synthesis guidance
        guidance_parts = []
        if verified_count > 0:
            guidance_parts.append(
                f"You have {verified_count} verified evidence item(s). Cite findings directly from these verified sources."
            )

        if gaps:
            guidance_parts.append(
                f"INFORMATION GAPS DETECTED: {'; '.join(gaps)}. "
                "State clearly to the user what information was NOT found. "
                "Under NO circumstances should you fabricate or invent missing details or links."
            )

        if contradictions:
            guidance_parts.append(
                f"SOURCE CONTRADICTIONS: {'; '.join(contradictions)}. "
                "Present the differing figures neutrally and note the variance across sources."
            )

        guidance = " ".join(guidance_parts)
        logger.info(
            "evidence_sufficiency_evaluated",
            extra={
                "is_sufficient": is_sufficient,
                "verified": verified_count,
                "failed": failed_count,
                "gaps_count": len(gaps),
            },
        )

        return EvidenceSufficiencyReport(
            is_sufficient=is_sufficient,
            confidence=confidence,
            total_evidence_count=total_count,
            verified_count=verified_count,
            failed_count=failed_count,
            contradictions=contradictions,
            information_gaps=gaps,
            synthesis_guidance=guidance,
        )


central_evidence_verifier = CentralEvidenceVerifier()

