"""
Search Decision Router for Hermes Agent Platform.

Implements a dedicated, auditable pre-execution decision engine evaluating three signals:
1. Intent Signal: Did the user explicitly request external research?
2. Freshness Signal: Does the answer depend on time-varying, versioned, or recent data?
3. Evidence Signal: Does the query require grounding/verification against authoritative sources?
4. Context Check: Is the necessary context or document already present in the prompt/conversation?

Outputs a structured SearchDecision and logs decisions to the central security audit log.
"""
from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("hermes.agent.search_router")


class SearchType(str, enum.Enum):
    NONE = "none"
    WEB = "web"
    NEWS = "news"
    ACADEMIC = "academic"
    DOCUMENTATION = "documentation"
    LOCAL = "local"
    SHOPPING = "shopping"


@dataclass
class SearchSignals:
    intent_score: float  # 0.0 to 1.0 (explicit user demand for search)
    freshness_score: float  # 0.0 to 1.0 (recency, version, time-sensitive data)
    evidence_score: float  # 0.0 to 1.0 (external verification/citation necessity)
    context_sufficient: bool  # True if the provided document/context already answers it
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchDecision:
    requires_search: bool
    reason: str
    search_type: SearchType
    confidence: float
    queries: list[str] = field(default_factory=list)
    signals: Optional[SearchSignals] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requires_search": self.requires_search,
            "reason": self.reason,
            "search_type": self.search_type.value,
            "confidence": round(self.confidence, 3),
            "queries": self.queries,
            "signals": {
                "intent_score": self.signals.intent_score,
                "freshness_score": self.signals.freshness_score,
                "evidence_score": self.signals.evidence_score,
                "context_sufficient": self.signals.context_sufficient,
                "details": self.signals.details,
            } if self.signals else None,
        }


class SearchDecisionRouter:
    """
    Evaluates whether a user prompt warrants external web search,
    classifies the appropriate search domain, and extracts optimal query terms.
    """

    # --- Signal 1: Explicit Search Intent Indicators ---
    EXPLICIT_SEARCH_PATTERNS = [
        r"\b(?:search\s+(?:the\s+web|online|google|duckduckgo|internet)|web\s+search)\b",
        r"\b(?:look\s+up|lookup|find\s+(?:me\s+)?(?:recent|latest|online|articles|papers|sources|restaurants?|places|hotels?))\b",
        r"\b(?:browse|scrape|fetch\s+the\s+web|google\s+it)\b",
        r"\b(?:find\s+(?:links?|urls?|documentation|citations?|studies))\b",
    ]

    # --- Signal 2: Freshness / Time-Varying Indicators ---
    FRESHNESS_PATTERNS = [
        r"\b(?:latest|current|recent|newest|today|yesterday|this\s+(?:week|month|year))\b",
        r"\b(?:202[4-9]|203\d)\b",
        r"\b(?:earnings|q[1-4]\s*20\d\d|quarterly\s+results|fiscal\s+year)\b",
        r"\b(?:stock\s+price|market\s+cap|net\s+worths?|valuation|ceo\s+of)\b",
        r"\b(?:version|release|changelog|breaking\s+changes|deprecat(?:ed|ion))\b",
        r"\b(?:weather|live\s+score|schedule|flight\s+status|ticket\s+price)\b",
        r"\b(?:is\s+.*\s+(?:still\s+maintained|alive|deprecated|active|supported))\b",
        r"\b(?:near\s+me|nearby|in\s+(?:my\s+area|this\s+city|town))\b",
        r"\b(?:restaurants?|cafes?|hotels?|hours\s+today|directions\s+to)\b",
        r"\b(?:box\s*office|boxoffice|ticket\s+sales|first\s+weekend\s+collection)\b",
        r"\b(?:startups?|venture\s+capital|funding\s+round|series\s+[a-e])\b",
        r"\b(?:population|census|inflation\s+rate|gdp|unemployment\s+rate)\b",
        r"\b(?:how\s+many\b.*(?:in|across|headquartered))\b",
    ]

    # --- Signal 3: Evidence Grounding / Verification Indicators ---
    EVIDENCE_PATTERNS = [
        r"\b(?:verify|fact[- ]check|cite|citation|authoritative|official\s+(?:doc|policy|source))\b",
        r"\b(?:compare|benchmarks?|vs\b|versus|pros\s+and\s+cons\s+of)\b",
        r"\b(?:efficacy|clinical\s+trials?|statistically|mortality\s+rate|law|statute)\b",
        r"\b(?:who\s+(?:won|holds|leads|founded|created))\b",
        r"\b(?:numbers?|figures?|statistics?|stats?|breakdown)\b",
    ]

    # --- Domain Classification Patterns ---
    ACADEMIC_PATTERNS = [
        r"\b(?:papers?|arxiv|peer[- ]reviewed|literature\s+review|methodology|citations?|interpretability|vlm|llm\s+reasoning)\b",
        r"\b(?:abstract|conference|neurips|icml|iclr|cvpr|acl|emnlp)\b",
    ]

    DOCUMENTATION_PATTERNS = [
        r"\b(?:documentation|api|sdk|reference|syntax|endpoint|parameters?|pytorch|tensorflow|fastapi|langchain|langgraph|pydantic)\b",
        r"\b(?:pip\s+install|npm\s+install|docker-compose|config\.yaml)\b",
    ]

    NEWS_PATTERNS = [
        r"\b(?:news|headline|election|scandal|breaking|announcement|resigned|acquired|merger)\b",
    ]

    LOCAL_PATTERNS = [
        r"\b(?:near\s+me|nearby|in\s+(?:my\s+area|this\s+city)|restaurant|cafe|hotel|store|hours\s+today|directions\s+to)\b",
    ]

    SHOPPING_PATTERNS = [
        r"\b(?:buy|price|cost|deals?|discount|reviews?\s+of|cheapest|where\s+to\s+buy)\b",
    ]

    def evaluate(
        self,
        query: str,
        conversation_context: Optional[str] = None,
        has_attachments: bool = False,
    ) -> SearchDecision:
        """
        Evaluate user prompt against all 4 checks to yield a grounded SearchDecision.
        """
        text = (query or "").strip()
        lower = text.lower()

        # Check 1: Context Sufficiency (Does prompt contain full code/document to analyze?)
        context_sufficient = False
        context_reason = ""

        # Prompt itself has large code blocks or document quotes
        has_code_block = bool(re.search(r"```[\s\S]{30,}```", text))
        has_large_quote = bool(re.search(r'(?:"""[\s\S]{40,}""")|(?:\'\'\'[\s\S]{40,}\'\'\')', text))
        explicit_self_contained = any(
            re.search(p, lower)
            for p in [
                r"\b(?:in|from|explain|summarize|review|fix|refactor|debug)\s+(?:this|the\s+following|the\s+above)\b",
                r"\b(?:here\s+is|using\s+this|based\s+on\s+this|uploaded\s+(?:file|document|paper|pdf|code))\b",
            ]
        )

        if has_attachments:
            context_sufficient = True
            context_reason = "User provided uploaded file attachments; context is available locally."
        elif (has_code_block or has_large_quote) and explicit_self_contained:
            context_sufficient = True
            context_reason = "Prompt contains inline code or document context sufficient for answering without web access."
        elif explicit_self_contained and conversation_context and len(conversation_context.strip()) > 100:
            context_sufficient = True
            context_reason = "Prior conversation context contains the reference material requested."

        # Compute Signal 1: Intent Score
        intent_matches = [p for p in self.EXPLICIT_SEARCH_PATTERNS if re.search(p, lower)]
        intent_score = 1.0 if intent_matches else 0.0

        # Compute Signal 2: Freshness Score
        freshness_matches = [p for p in self.FRESHNESS_PATTERNS if re.search(p, lower)]
        freshness_score = min(1.0, len(freshness_matches) * 0.45)

        # Compute Signal 3: Evidence Score
        evidence_matches = [p for p in self.EVIDENCE_PATTERNS if re.search(p, lower)]
        evidence_score = min(1.0, len(evidence_matches) * 0.4)

        # Detect Stable Conceptual Questions (Explicit no-search heuristic)
        stable_conceptual = any(
            re.search(p, lower)
            for p in [
                r"^explain\s+(?:backpropagation|gradient\s+descent|quicksort|binary\s+search|transformer\s+attention|bayes\s+theorem|tcp/ip)\b",
                r"^what\s+is\s+(?:recursion|polymorphism|eigenvalue|entropy|an\s+api|http\s+status\s+code)\b",
                r"^write\s+a\s+(?:python|javascript|bash|sql)\s+(?:function|script|algorithm)\s+to\b",
            ]
        )

        signals = SearchSignals(
            intent_score=intent_score,
            freshness_score=freshness_score,
            evidence_score=evidence_score,
            context_sufficient=context_sufficient,
            details={
                "intent_matches": len(intent_matches),
                "freshness_matches": len(freshness_matches),
                "evidence_matches": len(evidence_matches),
                "stable_conceptual": stable_conceptual,
            },
        )

        # -------------------------------------------------------------------
        # Triangulate Decision Matrix:
        # 1. If context is already sufficient AND user did NOT explicitly demand external search -> NO SEARCH
        # 2. If explicit search intent -> SEARCH
        # 3. If high freshness or evidence demand AND not purely stable conceptual -> SEARCH
        # 4. Otherwise -> NO SEARCH
        # -------------------------------------------------------------------
        requires_search = False
        decision_reason = ""
        confidence = 0.85

        if context_sufficient and intent_score < 0.8:
            requires_search = False
            decision_reason = f"No search needed: {context_reason}"
            confidence = 0.95
        elif intent_score > 0.5:
            requires_search = True
            decision_reason = "Explicit search intent: user directly requested external research or documentation lookup."
            confidence = 0.98
        elif stable_conceptual and freshness_score < 0.4 and evidence_score < 0.4:
            requires_search = False
            decision_reason = "No search needed: stable foundational/conceptual knowledge well within model training parameters."
            confidence = 0.92
        elif freshness_score >= 0.4:
            requires_search = True
            decision_reason = "Implicit search necessity: query depends on time-varying, versioned, or recent external facts."
            confidence = 0.88
        elif evidence_score >= 0.5:
            requires_search = True
            decision_reason = "Evidence requirement: external verification against authoritative sources materially improves factual accuracy."
            confidence = 0.82
        else:
            requires_search = False
            decision_reason = "No search needed: general synthesis or standard reasoning answerable from foundational model knowledge."
            confidence = 0.80

        # Classify Search Domain if search is required
        search_type = SearchType.NONE
        extracted_queries: list[str] = []

        if requires_search:
            search_type = self._classify_search_type(lower)
            extracted_queries = self._extract_clean_search_queries(text, search_type)

        return SearchDecision(
            requires_search=requires_search,
            reason=decision_reason,
            search_type=search_type,
            confidence=confidence,
            queries=extracted_queries,
            signals=signals,
        )

    def _classify_search_type(self, lower_text: str) -> SearchType:
        """Classify search intent into specialized target domains."""
        if any(re.search(p, lower_text) for p in self.LOCAL_PATTERNS):
            return SearchType.LOCAL
        if any(re.search(p, lower_text) for p in self.ACADEMIC_PATTERNS):
            return SearchType.ACADEMIC
        if any(re.search(p, lower_text) for p in self.DOCUMENTATION_PATTERNS):
            return SearchType.DOCUMENTATION
        if any(re.search(p, lower_text) for p in self.NEWS_PATTERNS):
            return SearchType.NEWS
        if any(re.search(p, lower_text) for p in self.SHOPPING_PATTERNS):
            return SearchType.SHOPPING
        return SearchType.WEB

    def _extract_clean_search_queries(self, raw_query: str, search_type: SearchType) -> list[str]:
        """Formulate clean, high-signal keyword queries from conversational user prompts."""
        # Strip conversational filler prefixes
        cleaned = re.sub(
            r"^(?:please\s+)?(?:can\s+you\s+)?(?:search\s+(?:for|the\s+web\s+for)?|look\s+up|find\s+(?:me\s+)?|what\s+is|tell\s+me\s+about)\s+",
            "",
            raw_query.strip(),
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[?!.,;]+$", "", cleaned).strip()

        queries = [cleaned] if cleaned else [raw_query]

        # Domain-specific keyword enrichment
        if search_type == SearchType.ACADEMIC and not any(k in cleaned.lower() for k in ["paper", "arxiv", "research"]):
            queries.append(f"{cleaned} paper arxiv")
        elif search_type == SearchType.DOCUMENTATION and not any(k in cleaned.lower() for k in ["docs", "documentation", "api"]):
            queries.append(f"{cleaned} official documentation")
        elif search_type == SearchType.LOCAL and "near me" in raw_query.lower():
            queries.append(cleaned)

        return list(dict.fromkeys(q for q in queries if q.strip()))


search_decision_router = SearchDecisionRouter()
