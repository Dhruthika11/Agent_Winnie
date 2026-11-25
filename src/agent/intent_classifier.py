"""
Intent Classification and Query Disambiguation for Winnie Agent.

Separates user requests into explicit categories:
1. POLITICAL_INFO: Neutral, factual political/civic information.
2. FUTURE_ELECTION: Speculative future elections requiring balanced, scenario-based analysis without partisan predictions.
3. ACTION_REQUEST: User wants the agent to perform an execution (file modification, terminal command, script execution).
4. EXPLANATION_REQUEST: User is asking for an explanation, tutorial, or conceptual question (even if it mentions dangerous commands).
5. WEB_SEARCH: Search for real-time external info, profile links, or publications.
6. GENERAL_KNOWLEDGE: Conceptual, technical, or programming knowledge.
7. CONVERSATIONAL: Fast greetings, acknowledgements, chit-chat.
"""
from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("hermes.agent.intent_classifier")


class IntentCategory(str, enum.Enum):
    POLITICAL_INFO = "political_info"
    FUTURE_ELECTION = "future_election"
    ACTION_REQUEST = "action_request"
    EXPLANATION_REQUEST = "explanation_request"
    WEB_SEARCH = "web_search"
    GENERAL_KNOWLEDGE = "general_knowledge"
    CONVERSATIONAL = "conversational"
    MEMORY_INSTRUCTION = "memory_instruction"


@dataclass
class IntentClassification:
    category: IntentCategory
    confidence: float
    is_action_proposal: bool
    requires_grounding: bool
    guidance: str


def classify_user_intent(query: str) -> IntentClassification:
    """Classify the user intent using high-precision heuristic pattern matching."""
    text = (query or "").strip().lower()

    # 1. Fast Conversational check
    greetings = {"hi", "hello", "hey", "good morning", "good evening", "howdy", "thanks", "thank you", "bye"}
    if text in greetings or (len(text) < 20 and any(text.startswith(g) for g in ["hi", "hello", "hey"])):
        return IntentClassification(
            category=IntentCategory.CONVERSATIONAL,
            confidence=0.95,
            is_action_proposal=False,
            requires_grounding=False,
            guidance="Respond warmly and concisely to the greeting or conversational remark.",
        )

    # 2. Memory & User Preference Declaration check
    memory_instruction_patterns = [
        r"^remember\s+(?:that\s+)?",
        r"\bkeep\s+in\s+mind\s+(?:that\s+)?",
        r"\bplease\s+note\s+(?:that\s+)?",
        r"^note\s+that\b",
        r"\bdo\s+not\s+forget\s+(?:that\s+)?",
        r"\bdon't\s+forget\s+(?:that\s+)?",
        r"\bmy\s+team(?:'s)?\s+(?:primary|uses|is|stack)\b",
        r"\bmy\s+(?:preferred|preference)\s+is\b",
        r"\bi\s+(?:strictly\s+)?prefer\b",
        r"\bi\s+(?:strictly\s+)?use\b",
        r"\bupdate\s+my\s+preference\b",
        r"\bsave\s+(?:this\s+)?to\s+memory\b",
        r"\bstore\s+(?:this\s+)?in\s+memory\b",
    ]
    if any(re.search(p, text, re.IGNORECASE) for p in memory_instruction_patterns):
        return IntentClassification(
            category=IntentCategory.MEMORY_INSTRUCTION,
            confidence=0.95,
            is_action_proposal=False,
            requires_grounding=False,
            guidance=(
                "MEMORY & PREFERENCE INSTRUCTION: The user is explicitly providing a permanent preference, constraint, or fact to remember. "
                "Directly acknowledge and confirm what has been saved to memory. "
                "DO NOT re-answer previous queries from conversation history or generate unsolicited technical comparisons "
                "unless the user explicitly asks a new question in this turn."
            ),
        )

    # 2. Future Election / Speculative Political Prediction
    future_election_indicators = [
        r"will\s+.*\s+(?:win|lose|sweep|gain|lose)\s+.*\s*(?:election|vote|poll|seats)?",
        r"who\s+will\s+win\s+.*\s*(?:election|20\d\d)",
        r"upcoming\s+(?:20\d\d|next)?\s*election",
        r"predict\s+.*\s*election",
        r"election\s+prediction",
        r"future\s+election",
    ]
    if any(re.search(p, text, re.IGNORECASE) for p in future_election_indicators) or (
        ("election" in text or "elections" in text or "vote" in text) and any(y in text for y in ["2026", "2027", "2028", "2029", "2030", "upcoming", "future", "next"])
    ):
        return IntentClassification(
            category=IntentCategory.FUTURE_ELECTION,
            confidence=0.92,
            is_action_proposal=False,
            requires_grounding=False,
            guidance=(
                "FUTURE ELECTION INQUIRY: Provide an objective, balanced, multi-perspective socio-economic and "
                "coalition analysis outlining key factors, advantages, and vulnerabilities across parties. "
                "DO NOT make definitive predictions or declare a winner. Maintain strict neutrality."
            ),
        )

    # 3. Political / Civic Information
    political_indicators = [
        "bjp", "congress", "democrat", "republican", "parliament", "senate", "lok sabha",
        "prime minister", "president", "narendra modi", "rahul gandhi", "cabinet",
        "policy", "legislation", "constitution", "governance", "political party"
    ]
    if any(p in text for p in political_indicators):
        # Distinguish between explanation/history vs action
        return IntentClassification(
            category=IntentCategory.POLITICAL_INFO,
            confidence=0.88,
            is_action_proposal=False,
            requires_grounding=False,
            guidance=(
                "POLITICAL & CIVIC TOPIC: Provide objective, factual, and neutral historical context. "
                "Do not exhibit partisan bias or personal political sentiment."
            ),
        )

    # 4. Search & Evidence Grounding Check (Prioritize search over passive explanation)
    search_indicators = [
        "linkedin", "profile", "github repo", "search for", "find the link", "latest news", "website",
        "who is the ceo", "current price", "scrape", "lookup", "url of", "net worth", "earnings",
        "revenue", "release", "stock", "born", "who is", "who are", "what is the latest"
    ]
    if any(s in text for s in search_indicators):
        return IntentClassification(
            category=IntentCategory.WEB_SEARCH,
            confidence=0.88,
            is_action_proposal=True,
            requires_grounding=True,
            guidance=(
                "SEARCH & EVIDENCE GROUNDING: Use web_scraper.search to locate verified sources. "
                "CRITICAL: Never fabricate or guess URLs. Only report URLs explicitly present in the search evidence."
            ),
        )

    # 5. Explanation vs Action Disambiguation
    explanation_indicators = [
        r"^explain\b",
        r"^describe\b",
        r"\bmeaning\s+of\b",
        r"^can\s+you\s+explain\b",
        r"^what\s+does\s+.*\s+(?:do|mean|stand\s+for)\b",
        r"^how\s+does\s+.*\s+work\b",
    ]
    is_explanation = any(re.search(p, text, re.IGNORECASE) for p in explanation_indicators)

    # 6. Action Request (Direct execution commands)
    action_verbs = [
        r"^(?:please\s+)?delete\b",
        r"^(?:please\s+)?remove\b",
        r"^(?:please\s+)?drop\b",
        r"^(?:please\s+)?run\b",
        r"^(?:please\s+)?execute\b",
        r"^(?:please\s+)?create\b",
        r"^(?:please\s+)?write\b",
        r"^(?:please\s+)?update\b",
        r"^(?:please\s+)?modify\b",
        r"^(?:please\s+)?rm\b",
        r"^(?:please\s+)?ls\b",
        r"^(?:please\s+)?cat\b",
        r"^(?:please\s+)?dir\b",
        r"^(?:please\s+)?mkdir\b",
        r"^(?:please\s+)?rmdir\b",
        r"^(?:please\s+)?touch\b",
        r"^(?:please\s+)?cp\b",
        r"^(?:please\s+)?mv\b",
        r"^(?:please\s+)?chmod\b",
        r"^(?:please\s+)?chown\b",
        r"^(?:please\s+)?kill\b",
        r"^(?:please\s+)?echo\b",
        r"\b(?:rm|del|delete|unlink|rmdir|mkdir)\s+(?:-[a-zA-Z]+\s+)*[.\/~]",
    ]
    is_action = any(re.search(p, text, re.IGNORECASE) for p in action_verbs)

    if is_explanation:
        return IntentClassification(
            category=IntentCategory.EXPLANATION_REQUEST,
            confidence=0.90,
            is_action_proposal=False,
            requires_grounding=False,
            guidance=(
                "EXPLANATION REQUEST: The user is asking for conceptual understanding or instructions, NOT asking "
                "you to execute destructive commands. Explain the concept clearly and safely in markdown without proposing tool calls."
            ),
        )

    if is_action:
        return IntentClassification(
            category=IntentCategory.ACTION_REQUEST,
            confidence=0.95,
            is_action_proposal=True,
            requires_grounding=False,
            guidance=(
                "ACTION REQUEST: The user is requesting an actual tool execution. Propose a structured tool call. "
                "The platform ToolPolicy engine will independently evaluate the risk and enforce safety gating."
            ),
        )

    # 7. General Knowledge
    return IntentClassification(
        category=IntentCategory.GENERAL_KNOWLEDGE,
        confidence=0.75,
        is_action_proposal=False,
        requires_grounding=False,
        guidance="Provide a comprehensive, clear, and well-structured technical or conceptual answer using clean Markdown.",
    )

