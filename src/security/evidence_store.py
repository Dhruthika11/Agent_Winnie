"""
Evidence and Provenance Store for Hermes Agent Platform.
Provides explicit storage and claim-to-evidence verification for:
1. Web-derived evidence (URL, title, extracted text, content hash, tool execution ID).
2. Memory-derived evidence (memory ID, preference/fact text, user/tenant context).
3. Tool execution evidence.

Implements the Claim-to-Evidence Grounding Pipeline:
LLM Response -> Extract Atomic Claims -> Retrieve Supporting Evidence ->
Verify Claim-Evidence Relationship -> Classify: SUPPORTED / CONTRADICTED / UNKNOWN -> Final Response
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import re
import threading
import uuid
from typing import Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("hermes.security.evidence")


class Evidence(BaseModel):
    """Explicit evidence record with provenance, content hash, and source metadata."""
    evidence_id: str
    source_url: str = ""
    source_title: str = ""
    extracted_text: str
    content_hash: str
    retrieval_timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    tool_execution_id: str = ""
    provenance_type: str = "web"  # "web" | "memory" | "tool" | "database"
    confidence: float = 1.0
    user_id: str = "default"
    tenant_id: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "extracted_text": self.extracted_text,
            "content_hash": self.content_hash,
            "retrieval_timestamp": self.retrieval_timestamp,
            "tool_execution_id": self.tool_execution_id,
            "provenance_type": self.provenance_type,
            "confidence": self.confidence,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
        }


class Claim(BaseModel):
    """Atomic claim extracted from LLM response linked to supporting evidence."""
    claim_id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    verification_status: str = "UNKNOWN"  # SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE | QUOTE_MISMATCH | SOURCE_UNAVAILABLE
    confidence: float = 0.5
    provenance_type: str = "web"  # "web" | "memory"
    extracted_figures: list[str] = Field(default_factory=list)
    quoted_spans: list[str] = Field(default_factory=list)
    attributed_source: Optional[str] = None
    contradiction_reason: Optional[str] = None
    source_claim: Optional[str] = None
    quote_match: bool = True
    issues: list[str] = Field(default_factory=list)
    supporting_evidence_details: list[dict[str, Any]] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "evidence_ids": self.evidence_ids,
            "verification_status": self.verification_status,
            "confidence": self.confidence,
            "provenance_type": self.provenance_type,
            "extracted_figures": self.extracted_figures,
            "quoted_spans": self.quoted_spans,
            "attributed_source": self.attributed_source,
            "contradiction_reason": self.contradiction_reason,
            "source_claim": self.source_claim,
            "quote_match": self.quote_match,
            "issues": self.issues,
            "supporting_evidence_details": self.supporting_evidence_details,
        }

    def to_internal_dict(self) -> dict[str, Any]:
        """Return the exact machine-readable representation required for internal auditing."""
        return {
            "status": self.verification_status.lower(),
            "claim": self.text,
            "claim_id": self.claim_id,
            "source_claim": self.source_claim,
            "quote_match": self.quote_match,
            "issues": list(self.issues),
            "confidence": round(self.confidence, 3),
            "evidence_ids": list(self.evidence_ids),
        }


class GroundingVerificationResult(BaseModel):
    """Complete evaluation result for response claims verified against evidence store."""
    original_text: str
    sanitized_text: str
    presentation_text: str = ""
    claims: list[Claim] = Field(default_factory=list)
    internal_verifications: list[dict[str, Any]] = Field(default_factory=list)
    supported_claims_count: int = 0
    contradicted_claims_count: int = 0
    unknown_claims_count: int = 0
    has_contradictions: bool = False
    has_unknowns: bool = False
    grounding_score: float = 1.0  # 0.0 to 1.0 ratio of supported claims

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "sanitized_text": self.sanitized_text,
            "claims": [c.to_dict() for c in self.claims],
            "supported_claims_count": self.supported_claims_count,
            "contradicted_claims_count": self.contradicted_claims_count,
            "unknown_claims_count": self.unknown_claims_count,
            "has_contradictions": self.has_contradictions,
            "has_unknowns": self.has_unknowns,
            "grounding_score": self.grounding_score,
        }


class EvidenceStore:
    """Thread-safe centralized store for structured evidence with content hashing and provenance."""

    def __init__(self):
        self._lock = threading.Lock()
        self._evidence_by_id: dict[str, Evidence] = {}
        self._evidence_by_url: dict[str, list[str]] = {}
        self._evidence_by_hash: dict[str, str] = {}
        self._memory_evidence: dict[str, Evidence] = {}

    def compute_content_hash(self, text: str) -> str:
        """Generate SHA-256 content hash of normalized extracted text."""
        normalized = (text or "").strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def record_evidence(
        self,
        extracted_text: str,
        source_url: str = "",
        source_title: str = "",
        tool_execution_id: str = "",
        provenance_type: str = "web",
        confidence: float = 1.0,
        user_id: str = "default",
        tenant_id: str = "default",
        metadata: Optional[dict[str, Any]] = None,
        evidence_id: Optional[str] = None,
    ) -> Evidence:
        """Store a structured evidence item with content hash and index mappings."""
        content_hash = self.compute_content_hash(extracted_text)
        ev_id = evidence_id or f"EVID-{uuid.uuid4().hex[:8].upper()}"

        evidence = Evidence(
            evidence_id=ev_id,
            source_url=source_url,
            source_title=source_title,
            extracted_text=extracted_text,
            content_hash=content_hash,
            tool_execution_id=tool_execution_id,
            provenance_type=provenance_type,
            confidence=confidence,
            user_id=user_id,
            tenant_id=tenant_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._evidence_by_id[ev_id] = evidence
            self._evidence_by_hash[content_hash] = ev_id

            if source_url:
                clean_url = source_url.strip().lower()
                self._evidence_by_url.setdefault(clean_url, [])
                if ev_id not in self._evidence_by_url[clean_url]:
                    self._evidence_by_url[clean_url].append(ev_id)

            if provenance_type == "memory":
                self._memory_evidence[ev_id] = evidence

        logger.debug("evidence_recorded", extra={"evidence_id": ev_id, "type": provenance_type, "url": source_url})
        return evidence

    def record_memory_evidence(
        self,
        memory_id: str,
        memory_text: str,
        user_id: str,
        tenant_id: str = "default",
        category: str = "preference",
        confidence: float = 1.0,
    ) -> Evidence:
        """Record provenance for memory-derived claims separately from web evidence."""
        return self.record_evidence(
            extracted_text=memory_text,
            source_title=f"User Memory ({category})",
            source_url=f"memory://{user_id}/{memory_id}",
            tool_execution_id=memory_id,
            provenance_type="memory",
            confidence=confidence,
            user_id=user_id,
            tenant_id=tenant_id,
            metadata={"memory_id": memory_id, "category": category},
        )

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Fetch evidence by its unique ID."""
        with self._lock:
            return self._evidence_by_id.get(evidence_id)

    def find_evidence_by_url(self, url: str) -> list[Evidence]:
        """Lookup evidence records matching a URL."""
        clean_url = url.strip().lower()
        with self._lock:
            ev_ids = self._evidence_by_url.get(clean_url, [])
            return [self._evidence_by_id[eid] for eid in ev_ids if eid in self._evidence_by_id]

    def list_evidence(
        self,
        provenance_type: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[Evidence]:
        """List stored evidence items with optional provenance or tenant filter."""
        with self._lock:
            items = list(self._evidence_by_id.values())
        if provenance_type:
            items = [e for e in items if e.provenance_type == provenance_type]
        if tenant_id:
            items = [e for e in items if e.tenant_id == tenant_id]
        return items[:limit]

    def clear(self) -> None:
        """Reset the evidence store (for testing or session reset)."""
        with self._lock:
            self._evidence_by_id.clear()
            self._evidence_by_url.clear()
            self._evidence_by_hash.clear()
            self._memory_evidence.clear()


evidence_store = EvidenceStore()


# --------------------------------------------------------------------------- #
# Grounding Pipeline: Extraction & Verification Engine
# --------------------------------------------------------------------------- #
_FIGURE_RE = re.compile(
    r"\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*(?:users|req/s|fps|tps|ms|seconds|minutes|hours|gb|mb|tb|kb|points|marks|dollars|\$|acc|accuracy)?\b",
    re.IGNORECASE,
)
_QUOTE_SPAN_RE = re.compile(r'["\']([^"\']{8,120})["\']')
_ATTRIBUTION_PATTERNS = [
    re.compile(r"according\s+to\s+(?:the\s+)?(https?://[^\s,;]+|[A-Za-z0-9_\-\.\s]+?)(?:,|\.|\sand\b|$)", re.IGNORECASE),
    re.compile(r"as\s+(?:reported|stated|documented)\s+(?:by|in)\s+(?:the\s+)?(https?://[^\s,;]+|[A-Za-z0-9_\-\.\s]+?)(?:,|\.|$)", re.IGNORECASE),
    re.compile(r"\((?:source|via|from):\s*([^)]+)\)", re.IGNORECASE),
    re.compile(r"\[(?:source|via|from):\s*([^\]]+)\]", re.IGNORECASE),
    re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)"),
]


def extract_atomic_claims(text: str) -> list[Claim]:
    """Step 1 of Grounding Pipeline: Decompose text into discrete verifiable atomic claims."""
    if not text:
        return []

    # Segment into sentences
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    claims: list[Claim] = []

    for raw_sent in sentences:
        sent = raw_sent.strip()
        if not sent or len(sent) < 10:
            continue

        figures = [m.group(0).strip() for m in _FIGURE_RE.finditer(sent) if any(c.isdigit() for c in m.group(0))]
        quoted_spans = [m.group(1).strip() for m in _QUOTE_SPAN_RE.finditer(sent)]

        attributed_source = None
        for pattern in _ATTRIBUTION_PATTERNS:
            m = pattern.search(sent)
            if m:
                src = m.group(1).strip().rstrip(".,;:/")
                if src and len(src) < 80:
                    attributed_source = src
                    break

        # Check if this claim is memory-derived (references user preferences or profile)
        is_memory_claim = bool(re.search(r"\b(?:you\s+prefer|as\s+you\s+(?:requested|said|configured)|your\s+preference|your\s+stack)\b", sent, re.IGNORECASE))
        prov_type = "memory" if is_memory_claim else "web"

        claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
        claims.append(
            Claim(
                claim_id=claim_id,
                text=sent,
                evidence_ids=[],
                verification_status="UNKNOWN",
                confidence=0.5,
                provenance_type=prov_type,
                extracted_figures=list(dict.fromkeys(figures)),
                quoted_spans=quoted_spans,
                attributed_source=attributed_source,
            )
        )

    return claims


def verify_claim_against_evidence(
    claim: Claim,
    evidence_candidates: list[Evidence],
) -> Claim:
    """Step 3 & 4 of Grounding Pipeline: Verify claim-evidence relationship and classify.

    Classifies into: SUPPORTED, CONTRADICTED, or UNKNOWN.
    """
    if not evidence_candidates:
        claim.verification_status = "UNKNOWN"
        claim.confidence = 0.40
        return claim

    matching_evidence_ids: list[str] = []
    has_direct_support = False
    has_direct_contradiction = False
    has_quote_mismatch = False
    contradiction_reasons: list[str] = []

    claim_text_lower = claim.text.lower()

    for ev in evidence_candidates:
        ev_text = ev.extracted_text
        ev_text_lower = ev_text.lower()

        # Check if the evidence is relevant to this claim
        # A. Check URL / Source attribution match
        is_source_match = False
        if claim.attributed_source:
            clean_attr = claim.attributed_source.lower()
            if (ev.source_url and clean_attr in ev.source_url.lower()) or (ev.source_title and clean_attr in ev.source_title.lower()):
                is_source_match = True

        # B. Check entity / key terms overlap
        claim_keywords = set(re.findall(r"\b[A-Za-z0-9\-]{4,}\b", claim_text_lower))
        ev_keywords = set(re.findall(r"\b[A-Za-z0-9\-]{4,}\b", ev_text_lower))
        keyword_overlap = len(claim_keywords & ev_keywords)
        is_topical = keyword_overlap >= min(2, len(claim_keywords)) if claim_keywords else False

        if not is_source_match and not is_topical:
            continue

        # 1. Figure verification: Check if claim numbers/percentages are in evidence
        figures_supported = True
        figures_contradicted = False
        if claim.extracted_figures:
            for fig in claim.extracted_figures:
                clean_fig = fig.strip()
                if clean_fig not in ev_text:
                    figures_supported = False
                    # Check if evidence mentions a different figure for the same concept
                    alt_nums = re.findall(r"\b\d+(?:\.\d+)?\b", ev_text)
                    if alt_nums:
                        figures_contradicted = True
                        claim.source_claim = alt_nums[0]
                        issue_msg = f"Claim stated '{clean_fig}', but source '{ev.source_title or ev.source_url}' reports '{alt_nums[0]}'"
                        contradiction_reasons.append(issue_msg)
                        claim.issues.append(issue_msg)

        # 2. Quoted text span verification: Quoted text MUST appear verbatim
        quotes_supported = True
        if claim.quoted_spans:
            for q in claim.quoted_spans:
                if q not in ev_text:
                    quotes_supported = False
                    claim.quote_match = False
                    has_quote_mismatch = True
                    issue_msg = f"Quoted span '{q[:30]}...' not found in source text"
                    contradiction_reasons.append(issue_msg)
                    claim.issues.append(issue_msg)

        # 3. Decision for this evidence chunk
        if figures_supported and quotes_supported and is_topical:
            has_direct_support = True
            matching_evidence_ids.append(ev.evidence_id)
            claim.supporting_evidence_details.append({
                "evidence_id": ev.evidence_id,
                "source_title": ev.source_title,
                "source_url": ev.source_url,
                "provenance_type": ev.provenance_type,
            })
        elif figures_contradicted:
            has_direct_contradiction = True

    claim.evidence_ids = list(dict.fromkeys(matching_evidence_ids))

    if has_direct_contradiction:
        claim.verification_status = "CONTRADICTED"
        claim.confidence = 0.90
        claim.contradiction_reason = "; ".join(contradiction_reasons) if contradiction_reasons else "Contradicted by retrieved evidence"
    elif has_quote_mismatch:
        # Crucial architectural distinction: quote mismatch is not a factual contradiction
        claim.verification_status = "QUOTE_MISMATCH"
        claim.confidence = 0.70
        claim.contradiction_reason = "; ".join(contradiction_reasons)
    elif has_direct_support:
        claim.verification_status = "SUPPORTED"
        claim.confidence = 0.95
    elif not evidence_candidates:
        claim.verification_status = "UNKNOWN"
        claim.confidence = 0.40
    else:
        claim.verification_status = "INSUFFICIENT_EVIDENCE"
        claim.confidence = 0.50
        claim.issues.append("Source mentions related concepts but does not establish or prove the claim.")

    return claim


def verify_response_grounding(
    response_text: str,
    evidence_items: Optional[list[Evidence | dict[str, Any]]] = None,
) -> GroundingVerificationResult:
    """Full Grounding Pipeline:

    LLM Response -> Extract Atomic Claims -> Retrieve Supporting Evidence ->
    Verify Claim-Evidence Relationship -> Classify: SUPPORTED / CONTRADICTED / UNKNOWN ->
    Final Response with structured audit.
    """
    if not response_text:
        return GroundingVerificationResult(
            original_text="",
            sanitized_text="",
            claims=[],
            grounding_score=1.0,
        )

    # Convert dictionary inputs to Evidence models if necessary
    candidates: list[Evidence] = []
    if evidence_items:
        for item in evidence_items:
            if isinstance(item, Evidence):
                candidates.append(item)
            elif isinstance(item, dict):
                # Map from generic evidence_item dictionary
                ev = Evidence(
                    evidence_id=str(item.get("evidence_id") or item.get("id") or f"EVID-{uuid.uuid4().hex[:6]}"),
                    source_url=str(item.get("source_url") or item.get("url") or item.get("source") or ""),
                    source_title=str(item.get("source_title") or item.get("title") or "Source"),
                    extracted_text=str(item.get("extracted_text") or item.get("content") or item.get("raw_output") or ""),
                    content_hash=evidence_store.compute_content_hash(str(item.get("extracted_text") or item.get("content") or "")),
                    tool_execution_id=str(item.get("tool_execution_id") or ""),
                    provenance_type=str(item.get("provenance_type") or "web"),
                )
                candidates.append(ev)

    # 1. Extract atomic claims
    atomic_claims = extract_atomic_claims(response_text)

    # 2 & 3. Retrieve supporting evidence & verify relationships
    verified_claims: list[Claim] = []
    supported_cnt = 0
    contradicted_cnt = 0
    unknown_cnt = 0
    sanitized = response_text

    for claim in atomic_claims:
        # Filter candidate pool based on provenance type (memory vs web)
        pool = [c for c in candidates if c.provenance_type == claim.provenance_type] if candidates else []
        if not pool and candidates:
            pool = candidates

        v_claim = verify_claim_against_evidence(claim, pool)
        verified_claims.append(v_claim)

        if v_claim.verification_status == "SUPPORTED":
            supported_cnt += 1
        elif v_claim.verification_status == "CONTRADICTED":
            contradicted_cnt += 1
            # Qualify contradicted statement with warning note
            note = f" [Warning: Contradicted by evidence - {v_claim.contradiction_reason}]"
            if claim.text in sanitized:
                sanitized = sanitized.replace(claim.text, f"{claim.text}{note}")
        else:
            unknown_cnt += 1

    total_claims = max(len(atomic_claims), 1)
    score = round(supported_cnt / total_claims, 2)

    # User-facing presentation layer: converts raw verifications into clean, consolidated text
    from src.security.grounding_presentation import grounding_presentation_formatter
    presentation = grounding_presentation_formatter.format_user_presentation(response_text, verified_claims)
    internal_records = [c.to_internal_dict() for c in verified_claims]

    return GroundingVerificationResult(
        original_text=response_text,
        sanitized_text=sanitized,
        presentation_text=presentation,
        claims=verified_claims,
        internal_verifications=internal_records,
        supported_claims_count=supported_cnt,
        contradicted_claims_count=contradicted_cnt,
        unknown_claims_count=unknown_cnt,
        has_contradictions=contradicted_cnt > 0,
        has_unknowns=unknown_cnt > 0,
        grounding_score=score,
    )
