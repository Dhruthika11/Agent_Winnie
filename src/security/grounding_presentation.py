"""
User-Facing Grounding Presentation Layer for Hermes Agent Platform.

Enforces a strict architectural separation:
1. Internal Verification Layer: Retains complete raw diagnostic information
   (claim IDs, source text, contradiction labels, quote-span mismatches, confidence scores,
   evidence IDs, and debugging details) for auditing, logging, telemetry, and evaluation.
2. User-Facing Presentation Layer: Deterministically converts verification results into
   concise, human-readable text and consolidated warnings that explain what is wrong
   and what the user should do, while hiding internal IDs and technical validation jargon.

Predefined Warning Categories:
- SUPPORTED: Claim is supported by retrieved evidence; no warning shown.
- CONTRADICTED: Retrieved source directly conflicts (e.g. version or numerical mismatch).
- INSUFFICIENT_EVIDENCE: Source mentions topic but does not establish or prove the claim.
- QUOTE_MISMATCH: Cited quotation could not be located verbatim in the source text.
  (Treated as a citation/attribution mismatch, not a proven false claim).
- SOURCE_UNAVAILABLE: Referenced source URL could not be reached, checked, or retrieved.
"""
from __future__ import annotations

import logging
import re
import urllib.parse
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("hermes.security.grounding_presentation")


class VerificationCategory(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    QUOTE_MISMATCH = "QUOTE_MISMATCH"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"


class InternalClaimVerification(BaseModel):
    """Machine-readable internal verification record retained for auditing and telemetry."""
    claim_id: str
    status: str = VerificationCategory.SUPPORTED.value
    claim: str
    source_claim: Optional[str] = None
    quote_match: bool = True
    issues: list[str] = Field(default_factory=list)
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    confidence: float = 1.0
    evidence_ids: list[str] = Field(default_factory=list)
    raw_debugging_details: dict[str, Any] = Field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        """Return full machine-readable representation for internal auditing and telemetry."""
        return {
            "claim_id": self.claim_id,
            "status": self.status.lower(),
            "claim": self.claim,
            "source_claim": self.source_claim,
            "quote_match": self.quote_match,
            "issues": list(self.issues),
            "source_url": self.source_url,
            "source_title": self.source_title,
            "confidence": round(self.confidence, 3),
            "evidence_ids": list(self.evidence_ids),
            "raw_debugging_details": self.raw_debugging_details,
        }


def format_clickable_source_link(source_url: Optional[str], source_title: Optional[str] = None) -> str:
    """
    Format clean, user-friendly clickable markdown link from URL and title.
    Never exposes raw unclickable URLs mixed with error messages.
    """
    clean_url = (source_url or "").strip()
    if not clean_url:
        return source_title or "the retrieved source"

    if clean_url.startswith("http://") or clean_url.startswith("https://"):
        if source_title and source_title.strip() and not source_title.lower().startswith("http"):
            return f"[{source_title.strip()}]({clean_url})"
        # Extract clean domain name
        try:
            domain = urllib.parse.urlparse(clean_url).netloc
            if domain:
                clean_domain = re.sub(r"^www\.", "", domain)
                return f"[{clean_domain}]({clean_url})"
        except Exception:
            pass
        return f"[Source Link]({clean_url})"

    return source_title or clean_url


def _extract_counter_entity(claim_text: str, source_claim: str) -> str:
    """Extract contrasting target entity from claim text (e.g. 'Python 3.12' when source is 'Python 3.11')."""
    if not claim_text:
        return ""

    # Check for version pattern (e.g. Python 3.12 or 3.12)
    m_ver = re.findall(r"\b(?:[A-Za-z0-9_+-]+\s+)?v?\d+(?:\.\d+)+\b", claim_text)
    if m_ver:
        for v in m_ver:
            if v.lower() != (source_claim or "").lower():
                return v

    # Check for numbers / percentages
    m_num = re.findall(r"\b\d+(?:\.\d+)?%?\b", claim_text)
    if m_num:
        for n in m_num:
            if n.lower() != (source_claim or "").lower():
                return n

    # Truncate clean snippet
    words = claim_text.split()
    return " ".join(words[:6])


class GroundingPresentationFormatter:
    """Deterministic, rule-based formatter for user-facing grounding notices."""

    @classmethod
    def format_user_presentation(
        cls,
        base_text: str,
        verifications: list[InternalClaimVerification | dict[str, Any] | Any],
    ) -> str:
        """
        Merge model response text with clean, consolidated, user-facing verification notices.
        Never outputs raw JSON or internal field names like claim IDs or EVID tokens.
        """
        clean_text = base_text.strip() if base_text else ""
        banner = cls.generate_consolidated_warning_banner(verifications)
        if not banner:
            return clean_text

        if not clean_text:
            return banner

        return f"{clean_text}\n\n{banner}"

    @classmethod
    def generate_consolidated_warning_banner(
        cls,
        verifications: list[InternalClaimVerification | dict[str, Any] | Any],
    ) -> str:
        """
        Generate a single consolidated, human-readable warning banner from verification records.
        Eliminates duplicate warnings and hides internal IDs.
        """
        if not verifications:
            return ""

        # Normalize items into InternalClaimVerification
        normalized: list[InternalClaimVerification] = []
        for v in verifications:
            if isinstance(v, InternalClaimVerification):
                normalized.append(v)
            elif isinstance(v, dict):
                # Handle raw dict from internal verifier
                status_raw = str(v.get("status") or v.get("verification_status") or "SUPPORTED").upper()
                claim_id = str(v.get("claim_id") or v.get("id") or "CLM-internal")
                claim_text = str(v.get("claim") or v.get("claim_text") or v.get("text") or "")
                source_claim = v.get("source_claim")
                quote_match = bool(v.get("quote_match", True))
                issues = list(v.get("issues") or [])
                s_url = v.get("source_url") or v.get("url")
                s_title = v.get("source_title") or v.get("title")
                confidence = float(v.get("confidence", 1.0))
                ev_ids = list(v.get("evidence_ids") or [])
                raw_debug = dict(v.get("raw_debugging_details") or {})

                normalized.append(
                    InternalClaimVerification(
                        claim_id=claim_id,
                        status=status_raw,
                        claim=claim_text,
                        source_claim=source_claim,
                        quote_match=quote_match,
                        issues=issues,
                        source_url=s_url,
                        source_title=s_title,
                        confidence=confidence,
                        evidence_ids=ev_ids,
                        raw_debugging_details=raw_debug,
                    )
                )
            elif hasattr(v, "verification_status"):
                # Handle Claim model
                status = str(v.verification_status).upper()
                src_claim = getattr(v, "source_claim", None)
                q_match = getattr(v, "quote_match", True)
                issues = getattr(v, "issues", []) or []
                supp_details = getattr(v, "supporting_evidence_details", []) or []
                s_url = supp_details[0].get("source_url") if supp_details else getattr(v, "attributed_source", None)
                s_title = supp_details[0].get("source_title") if supp_details else None

                normalized.append(
                    InternalClaimVerification(
                        claim_id=getattr(v, "claim_id", "internal"),
                        status=status,
                        claim=getattr(v, "text", ""),
                        source_claim=src_claim,
                        quote_match=q_match,
                        issues=issues,
                        source_url=s_url,
                        source_title=s_title,
                        confidence=getattr(v, "confidence", 1.0),
                        evidence_ids=getattr(v, "evidence_ids", []),
                    )
                )

        # Categorize claims into predefined categories
        contradictions: list[InternalClaimVerification] = []
        quote_mismatches: list[InternalClaimVerification] = []
        insufficient: list[InternalClaimVerification] = []
        unavailable: list[InternalClaimVerification] = []

        for item in normalized:
            stat = item.status.upper()
            if stat == VerificationCategory.CONTRADICTED.value:
                contradictions.append(item)
            elif stat == VerificationCategory.QUOTE_MISMATCH.value or (not item.quote_match and stat != VerificationCategory.CONTRADICTED.value):
                quote_mismatches.append(item)
            elif stat == VerificationCategory.INSUFFICIENT_EVIDENCE.value:
                insufficient.append(item)
            elif stat == VerificationCategory.SOURCE_UNAVAILABLE.value:
                unavailable.append(item)

        # If all supported, return empty (Supported: No warning)
        if not contradictions and not quote_mismatches and not insufficient and not unavailable:
            return ""

        # Special case: Single contradiction with clear source_claim (cleanest plain-language format)
        if len(contradictions) == 1 and not quote_mismatches and not insufficient and not unavailable:
            c = contradictions[0]
            source_link = format_clickable_source_link(c.source_url, c.source_title)
            src_text = f"The retrieved source ({source_link})" if c.source_url or c.source_title else "The retrieved source"
            counter = _extract_counter_entity(c.claim, c.source_claim or "")

            lines = ["⚠️ **Source verification warning**\n"]
            if c.source_claim and counter:
                lines.append(f"{src_text} refers to {c.source_claim}, not {counter}. Therefore, it does not verify the claim.\n")
                lines.append(f"Please check the correct {counter} documentation.")
            elif c.source_claim:
                lines.append(f"{src_text} refers to {c.source_claim}, which conflicts with the claim.\n")
                lines.append("Please check the correct documentation.")
            else:
                reason = c.issues[0] if c.issues else "Conflicting factual claims were found in the source."
                clean_reason = re.sub(r"\bCLM-[A-Za-z0-9]+\b|\bEVID-[A-Za-z0-9]+\b|\bid\b", "", reason).strip()
                lines.append(f"{src_text} conflicts with this claim: {clean_reason}\n")
                lines.append("Please verify against the official documentation.")
            return "\n".join(lines).strip()

        # General case: Consolidated Warning Banner with bulleted points
        sections: list[str] = ["⚠️ **Source verification warning**\n"]
        seen_warnings: set[str] = set()

        # 1. Contradictions (Clear explanation of conflicting evidence)
        if contradictions:
            for c in contradictions:
                source_link = format_clickable_source_link(c.source_url, c.source_title)
                src_text = f" ({source_link})" if c.source_url or c.source_title else ""
                counter = _extract_counter_entity(c.claim, c.source_claim or "")
                
                if c.source_claim and counter:
                    warn_key = f"{c.source_claim}::{counter}"
                    if warn_key not in seen_warnings:
                        seen_warnings.add(warn_key)
                        sections.append(
                            f"• **Conflicting Evidence:** The retrieved source{src_text} refers to **{c.source_claim}**, "
                            f"not **{counter}**. Therefore, it does not verify the claim."
                        )
                elif c.source_claim:
                    warn_key = f"{c.source_claim}"
                    if warn_key not in seen_warnings:
                        seen_warnings.add(warn_key)
                        sections.append(
                            f"• **Conflicting Evidence:** The retrieved source{src_text} refers to **{c.source_claim}**, "
                            "which does not support the claim."
                        )
                else:
                    reason = c.issues[0] if c.issues else "Conflicting facts found in retrieved source."
                    clean_reason = re.sub(r"\bCLM-[A-Za-z0-9]+\b|\bEVID-[A-Za-z0-9]+\b|\bid\b", "", reason).strip()
                    warn_key = clean_reason
                    if warn_key not in seen_warnings:
                        seen_warnings.add(warn_key)
                        sections.append(
                            f"• **Conflicting Evidence:** The retrieved source{src_text} conflicts with this claim: {clean_reason}"
                        )

        # 2. Quote Mismatches (Citation/evidence mismatch - NOT a false claim)
        if quote_mismatches:
            if len(quote_mismatches) == 1:
                qm = quote_mismatches[0]
                source_link = format_clickable_source_link(qm.source_url, qm.source_title)
                src_text = f" ({source_link})" if qm.source_url or qm.source_title else ""
                sections.append(
                    f"• **Citation Mismatch:** The cited quotation could not be located verbatim in the referenced source{src_text}. "
                    "The underlying statement may still be valid, but the exact quotation could not be confirmed."
                )
            else:
                sections.append(
                    f"• **Citation Mismatch:** {len(quote_mismatches)} cited quotations could not be located verbatim in the referenced sources. "
                    "The underlying statements may still be valid, but the exact wording could not be confirmed."
                )

        # 3. Insufficient Evidence (Source does not establish the claim)
        if insufficient:
            for ie in insufficient:
                source_link = format_clickable_source_link(ie.source_url, ie.source_title)
                src_text = f" ({source_link})" if ie.source_url or ie.source_title else ""
                warn_key = f"insufficient::{ie.source_url}"
                if warn_key not in seen_warnings:
                    seen_warnings.add(warn_key)
                    sections.append(
                        f"• **Unverified Claim:** The retrieved source{src_text} mentions related concepts "
                        "but does not provide sufficient evidence to confirm this claim."
                    )

        # 4. Source Unavailable (Link could not be checked or reached)
        if unavailable:
            for un in unavailable:
                source_link = format_clickable_source_link(un.source_url, un.source_title)
                src_text = f" ({source_link})" if un.source_url or un.source_title else ""
                warn_key = f"unavailable::{un.source_url}"
                if warn_key not in seen_warnings:
                    seen_warnings.add(warn_key)
                    sections.append(
                        f"• **Source Unavailable:** The referenced source{src_text} could not be accessed or verified during execution."
                    )

        # 5. Consolidated Actionable Recommendation
        sections.append("\n*Recommendation:* Please verify against the official documentation or primary reference sources.")

        return "\n".join(sections).strip()


grounding_presentation_formatter = GroundingPresentationFormatter()
