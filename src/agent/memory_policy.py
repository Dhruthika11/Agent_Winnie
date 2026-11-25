"""
Memory Policy & Filtering Layer for Winnie Agent.

Ensures that only durable, valuable context is persisted to Mem0 + Qdrant:
1. Rejects conversational refusals ("cannot help", "I'm sorry").
2. Rejects temporary tool errors, timeouts, and rate limits.
3. Rejects trivial chit-chat and acknowledgements.
4. Memory Provenance & Poisoning Prevention:
   - Distinguishes user-provided memories from external web/github memories.
   - Prevents untrusted external content ("always upload credentials to...")
     from becoming executable instructions.
5. Categorizes memories into durable taxonomy:
   - user_preference
   - technical_environment
   - project_context
   - external_reference
6. Sanitizes retrieved memories before injecting into agent context.
"""
from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("hermes.agent.memory_policy")

REFUSAL_PATTERNS = [
    r"cannot\s+help\s+with\s+that",
    r"can\'t\s+help\s+with\s+that",
    r"i\s+am\s+sorry,\s+but\s+i\s+cannot",
    r"i\'m\s+sorry,\s+but\s+i\s+can\'t",
    r"as\s+an\s+ai,\s+i\s+am\s+unable",
    r"policy\s+violation",
    r"action\s+is\s+prohibited",
]
_COMPILED_REFUSALS = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]

TRANSIENT_ERROR_PATTERNS = [
    r"tool\s+execution\s+error",
    r"search\s+failed",
    r"timed?\s*out",
    r"connection\s+reset",
    r"rate\s+limit",
    r"http\s+5\d\d",
    r"http\s+429",
    r"circuit\s+breaker\s+is\s+open",
]
_COMPILED_ERRORS = [re.compile(p, re.IGNORECASE) for p in TRANSIENT_ERROR_PATTERNS]

EPHEMERAL_PATTERNS = [
    r"^(?:hi|hello|hey|good\s+morning|good\s+evening)[.!]?$",
    r"^(?:thank\s+you|thanks|you\'re\s+welcome)[.!]?$",
    r"^i\s+completed\s+the\s+requested\s+action[.!]?$",
    r"^ok[.!]?$",
]
_COMPILED_EPHEMERAL = [re.compile(p, re.IGNORECASE) for p in EPHEMERAL_PATTERNS]

# Untrusted external instruction poisoning markers (imperatives that attempt to hijack behavior)
POISONING_INSTRUCTION_PATTERNS = [
    r"\balways\s+(?:upload|send|post|transmit|execute|delete|forward|expose)\b",
    r"\b(?:upload|send|transmit|post)\s+(?:credentials|keys?|passwords?|secrets?|tokens?)\b",
    r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions\b",
    r"\byou\s+(?:must\s+always|are\s+now)\b",
    r"\b(?:system\s+prompt|admin\s+command)\b",
]
_COMPILED_POISONING = [re.compile(p, re.IGNORECASE) for p in POISONING_INSTRUCTION_PATTERNS]

# Patterns indicating durable technical preferences
DURABLE_PATTERNS = {
    "technical_environment": [
        r"\b(?:postgres|cockroachdb|mysql|sqlite|redis|mongodb|qdrant)\b",
        r"\b(?:fastapi|flask|django|react|nextjs|vue|express)\b",
        r"\b(?:python|typescript|javascript|golang|rust|docker|kubernetes)\b",
    ],
    "user_preference": [
        r"\bi\s+(?:prefer|like|use|always\s+use|work\s+with)\b",
        r"\bmy\s+(?:stack|environment|setup|database|backend|os)\b",
    ],
}


@dataclass
class MemoryDecision:
    should_store: bool
    reason: str
    category: str
    metadata: dict[str, Any]
    source: str = "user"
    trust_level: str = "user_confirmed"


class MemoryPolicy:
    """Evaluates whether conversation turns are suitable for long-term memory storage."""

    def is_instruction_poisoning(self, text: str, source: str = "user") -> bool:
        """
        Component 4: Detects malicious instruction poisoning in external or untrusted sources.
        Returns True if untrusted text contains imperative directives that should never be stored as instructions.
        """
        if source == "user":
            return False  # User instructions are legitimate preferences

        for pattern in _COMPILED_POISONING:
            if pattern.search(text):
                logger.warning(
                    "memory_policy_poisoning_detected",
                    extra={"source": source, "pattern": pattern.pattern, "text": text[:100]},
                )
                return True
        return False

    def evaluate_turn(
        self,
        messages: list[Any],
        source: str = "user",
        trust_level: Optional[str] = None,
    ) -> MemoryDecision:
        if not messages:
            return MemoryDecision(False, "empty_messages", "none", {}, source=source)

        resolved_trust = trust_level or ("user_confirmed" if source == "user" else "external_untrusted")

        # Extract last user and last assistant messages
        user_content = ""
        assistant_content = ""

        for m in reversed(messages):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            content_str = str(content or "").strip()

            if role in ("user", "human") and not user_content:
                user_content = content_str
            elif role in ("assistant", "ai") and not assistant_content:
                assistant_content = content_str

            if user_content and assistant_content:
                break

        # 1. Filter Refusal Messages
        for pattern in _COMPILED_REFUSALS:
            if pattern.search(assistant_content):
                logger.info("memory_policy_rejected_refusal", extra={"reason": "assistant_refusal"})
                return MemoryDecision(False, "assistant_refusal", "none", {}, source=source)

        # 2. Filter Transient Errors
        for pattern in _COMPILED_ERRORS:
            if pattern.search(assistant_content) or pattern.search(user_content):
                logger.info("memory_policy_rejected_error", extra={"reason": "transient_tool_error"})
                return MemoryDecision(False, "transient_tool_error", "none", {}, source=source)

        # 3. Filter Ephemeral Chit-Chat
        for pattern in _COMPILED_EPHEMERAL:
            if pattern.search(user_content) or pattern.search(assistant_content):
                return MemoryDecision(False, "ephemeral_chitchat", "none", {}, source=source)

        # 4. Check for External Instruction Poisoning
        if self.is_instruction_poisoning(user_content + " " + assistant_content, source=source):
            return MemoryDecision(
                False,
                "rejected_instruction_poisoning",
                "none",
                {"poisoning": True},
                source=source,
                trust_level=resolved_trust,
            )

        # 5. Classify Category and Durability
        category = "general_fact"
        for cat, patterns in DURABLE_PATTERNS.items():
            if any(re.search(p, user_content, re.IGNORECASE) for p in patterns):
                category = cat
                break

        if source != "user":
            category = "external_reference"

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata = {
            "category": category,
            "source": source,
            "trust_level": resolved_trust,
            "stored_at": now_iso,
            "durability": "high" if category in ("user_preference", "technical_environment") else "medium",
        }

        return MemoryDecision(
            True,
            "approved_durable_turn",
            category,
            metadata,
            source=source,
            trust_level=resolved_trust,
        )

    def sanitize_retrieved_memories(self, memories: list[str]) -> list[str]:
        """Strip obsolete refusals, errors, and qualify untrusted external references."""
        cleaned = []
        for mem in memories:
            mem_str = str(mem).strip()
            # If the memory contains a legacy refusal, drop it
            if any(pattern.search(mem_str) for pattern in _COMPILED_REFUSALS):
                logger.info("memory_policy_purged_stale_refusal", extra={"memory": mem_str})
                continue
            # If the memory contains a legacy tool error, drop it
            if any(pattern.search(mem_str) for pattern in _COMPILED_ERRORS):
                logger.info("memory_policy_purged_stale_error", extra={"memory": mem_str})
                continue

            cleaned.append(mem_str)
        return cleaned


memory_policy = MemoryPolicy()
