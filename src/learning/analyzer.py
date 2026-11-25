"""
Automated Failure Analyzer and Root-Cause Classifier for Hermes Agent Platform.
Provides confidence-aware failure classification across the expanded 7-category taxonomy:
- retrieval_failure (Search routing)
- grounding_failure (Evidence verification)
- reasoning_failure (Reasoning strategy)
- tool_failure (Execution layer)
- memory_failure (Memory retrieval)
- planning_failure (Planning policy)
- security_failure (Authorization policy)
Plus procedural knowledge gaps (knowledge_failure).

Emits evidence trails, alternative causes, and flags low-confidence classifications for human review.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Optional
import uuid

from src.learning.database import failure_db
from src.learning.models import (
    CATEGORY_CORRECTIVE_ACTION_MAP,
    CATEGORY_LAYER_MAP,
    CorrectionType,
    FailureCategory,
    FailureRecord,
    FailureStatus,
)

logger = logging.getLogger("hermes.learning.analyzer")


class FailureAnalyzer:
    """Investigates agent failures to pinpoint root causes, responsible architectural layers,

    confidence scores, and evidence trails.
    """

    def analyze(
        self,
        user_input: str | FailureRecord,
        observed_output: str = "",
        expected_behavior: str = "",
        tool_calls: Optional[list[dict[str, Any]]] = None,
        tool_outputs: Optional[list[dict[str, Any]]] = None,
        model_name: str = "default",
        session_id: str = "unknown",
        user_id: str = "unknown",
        failure_id: Optional[str] = None,
    ) -> FailureRecord:
        """Analyze a failure and classify its category, root cause, confidence, and remediation type."""
        orig_failure_id = failure_id
        if isinstance(user_input, FailureRecord):
            rec = user_input
            orig_failure_id = rec.failure_id
            observed_output = rec.observed_output
            expected_behavior = rec.expected_behavior
            tool_calls = rec.tool_calls
            tool_outputs = rec.tool_outputs
            model_name = rec.model_name
            session_id = rec.session_id
            user_id = rec.user_id
            user_input = rec.user_input

        t_calls = tool_calls or []
        t_outs = tool_outputs or []
        clean_in = (user_input or "").strip()
        clean_obs = (observed_output or "").strip()

        combined_tool_text = " ".join(str(to) for to in t_outs).lower()
        combined_calls_text = " ".join(str(tc) for tc in t_calls).lower()
        full_telemetry_text = f"{clean_in} {clean_obs} {combined_tool_text} {combined_calls_text}".lower()

        # ------------------------------------------------------------------- #
        # Telemetry Feature Detection
        # ------------------------------------------------------------------- #
        # Security / Authorization Signals
        has_security_violation = any(
            w in full_telemetry_text
            for w in [
                "unauthorized", "permission denied", "forbidden", "blocked by policy",
                "rbac", "high-risk action blocked", "hitl rejection", "security gate",
                "privilege", "security_failure", "access denied"
            ]
        )

        # Tool Execution Crash Signals
        has_tool_crash = any(
            "error" in str(to).lower()
            or "exception" in str(to).lower()
            or "traceback" in str(to).lower()
            or "timed out" in str(to).lower()
            or "timeout" in str(to).lower()
            or "null byte" in str(to).lower()
            or "internal server error" in str(to).lower()
            for to in t_outs
        ) or any(
            "internal server error" in clean_obs.lower() or "traceback (most recent" in clean_obs.lower()
            for _ in [1]
        )

        # Search / Retrieval Failure Signals (Tool called, but 0 results / unverified / empty)
        has_retrieval_failure = (
            bool(t_calls)
            and any(
                "search" in tc.get("tool", "") or "scraper" in tc.get("tool", "") or "read_url" in tc.get("tool", "")
                for tc in t_calls
            )
            and (
                all(
                    not to or "no verified" in str(to).lower() or "not found" in str(to).lower()
                    or "empty" in str(to).lower() or "failed" in str(to).lower()
                    for to in t_outs
                )
                or any(
                    phrase in clean_obs.lower()
                    for phrase in [
                        "did not contain any relevant",
                        "no box-office information",
                        "no box‑office information",
                        "unable to locate any verified sources",
                        "no verified sources that provide",
                        "could not find verified",
                    ]
                )
            )
        )

        # Planning Policy / Redundant Loop Signals
        has_planning_loop = (
            len(t_calls) >= 3
            and (
                # Duplicate tool invocations
                len(set(str(tc.get("arguments")) for tc in t_calls)) < len(t_calls)
                or any("loop" in clean_obs.lower() or "repeated" in clean_obs.lower() for _ in [1])
            )
        )

        # Memory Subsystem Signals
        has_memory_signal = any(
            w in clean_in.lower()
            for w in [
                "my preferred", "my preference", "as i told you", "remember that",
                "as mentioned before", "earlier i said", "you forgot that", "my name is",
                "user preference", "stored memory"
            ]
        ) and any(
            w in clean_obs.lower()
            for w in ["i don't remember", "do not recall", "not aware of your", "forgot", "no memory"]
        )

        # Context-Grounding Mismatch & Hallucinated Entity Signals
        has_tools_with_data = bool(t_outs) and len(combined_tool_text) > 80

        clean_in_normalized = re.sub(r"[\-_]", " ", clean_in.lower())

        repo_mentions = re.findall(r"\b[A-Za-z0-9_\-\.]+\/[A-Za-z0-9_\-\.]+\b", clean_obs)
        unrelated_repo = None
        for rm in repo_mentions:
            if rm.lower() not in combined_tool_text and rm.lower() not in clean_in.lower():
                unrelated_repo = rm
                break

        unrelated_claim_match = re.search(
            r"\b(?:don't have|do not have|cannot find|can't find|no|not aware of|unrelated repository)\s+(?:any\s+)?(?:info|information|details)?\s*(?:about|on|regarding)?\s+(?:the\s+)?['\"]?([A-Za-z0-9_\-\.]+)",
            clean_obs,
            re.IGNORECASE,
        )
        detected_entity = None
        if unrelated_claim_match:
            candidate = unrelated_claim_match.group(1).strip("'\".,")
            cand_norm = re.sub(r"[\-_]", " ", candidate.lower())
            if candidate.lower() in ("the", "a", "an", "this", "that", "any", "box", "office", "box-office") or cand_norm in clean_in_normalized:
                m_next = re.search(
                    r"\b(?:don't have|do not have|cannot find|can't find|no|not aware of|unrelated repository)\s+(?:any\s+)?(?:info|information|details)?\s*(?:about|on|regarding)?\s+(?:the|a|an|this|that)\s+['\"]?([A-Za-z0-9_\-\.]+)",
                    clean_obs,
                    re.IGNORECASE,
                )
                if m_next:
                    cand2 = m_next.group(1).strip("'\".,")
                    cand2_norm = re.sub(r"[\-_]", " ", cand2.lower())
                    if cand2_norm not in clean_in_normalized and cand2.lower() not in ("box", "office", "box-office"):
                        detected_entity = cand2
            else:
                detected_entity = candidate

        if not detected_entity:
            for token in re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", clean_obs):
                token_norm = re.sub(r"[\-_]", " ", token.lower())
                if token_norm not in clean_in_normalized and token.lower() not in combined_tool_text:
                    detected_entity = token
                    break

        has_grounding_violation = bool(unrelated_repo or detected_entity)

        # Procedural Knowledge Gap Signals
        has_procedural_gap = any(
            w in clean_in.lower() for w in ["how to", "api", "sdk", "syntax", "procedure", "command to", "configure"]
        ) and any(
            w in clean_obs.lower() for w in ["i don't know", "not documented", "not sure", "unfamiliar", "unable to"]
        )

        # ------------------------------------------------------------------- #
        # Taxonomy Classification with Confidence & Evidence Assignment
        # ------------------------------------------------------------------- #
        category = FailureCategory.REASONING_FAILURE
        correction_type = CorrectionType.PROMPT_CHANGE
        originating_stage = "synthesis"
        root_cause = "General reasoning or synthesis mismatch."
        confidence = 0.75
        evidence: list[str] = []
        alternative_causes: list[str] = []

        # 1. Security Failure
        if has_security_violation:
            category = FailureCategory.SECURITY_FAILURE
            correction_type = CorrectionType.POLICY_FIX
            originating_stage = "tool_execution"
            confidence = 0.95
            evidence = [
                "Unauthorized tool invocation attempted or blocked by security policy",
                "Action violated RBAC or security authorization gate",
            ]
            alternative_causes = [FailureCategory.TOOL_FAILURE.value]
            root_cause = "Security failure: Unauthorized tool invocation or permission boundary violation. Corrective action: Fix authorization policy."

        # 2. Tool Failure
        elif has_tool_crash:
            category = FailureCategory.TOOL_FAILURE
            correction_type = CorrectionType.CODE_FIX
            originating_stage = "tool_execution"
            confidence = 0.92
            evidence = [
                "Tool execution raised runtime exception / error in output telemetry",
                "Execution layer returned unhandled crash, null byte, or timeout",
            ]
            alternative_causes = [FailureCategory.SYSTEM_FAILURE.value]
            root_cause = "Tool failure: Tool crash, timeout, malformed output, or execution layer runtime exception. Corrective action: Fix execution layer."

        # 3. Retrieval Failure (Search attempted but failed or returned zero relevant data)
        elif has_retrieval_failure:
            category = FailureCategory.RETRIEVAL_FAILURE
            correction_type = CorrectionType.ROUTING_CHANGE
            originating_stage = "retrieval"
            confidence = 0.85
            evidence = [
                "Relevant webpage or document not retrieved by tool execution",
                "Search query routing returned zero verified sources for query entities",
            ]
            alternative_causes = [FailureCategory.GROUNDING_FAILURE.value]
            root_cause = "Retrieval failure: Relevant webpage not retrieved. Corrective action: Improve search routing."

        # 4. Grounding Failure (Unsupported claim / Hallucination)
        elif has_grounding_violation:
            entity = unrelated_repo or detected_entity or "unknown"
            category = FailureCategory.GROUNDING_FAILURE
            correction_type = CorrectionType.RESPONSE_VALIDATION
            originating_stage = "synthesis"
            confidence = 0.88
            evidence = [
                f"Unsupported claim: output introduced entity '{entity}' not present in source context",
                "Retrieved documents do not contain the entity referenced in response",
            ]
            alternative_causes = [FailureCategory.RETRIEVAL_FAILURE.value]
            root_cause = f"Grounding failure: Unsupported claim or hallucinated entity '{entity}' in response. Corrective action: Improve evidence verification."

        # 5. Planning Failure
        elif has_planning_loop:
            category = FailureCategory.PLANNING_FAILURE
            correction_type = CorrectionType.PLANNING_POLICY
            originating_stage = "planning"
            confidence = 0.80
            evidence = [
                f"Agent executed {len(t_calls)} tool calls with redundant queries or repetitive loops",
                "Agent chose unnecessary steps without advancing toward the user goal",
            ]
            alternative_causes = [FailureCategory.TOOL_FAILURE.value, FailureCategory.REASONING_FAILURE.value]
            root_cause = "Planning failure: Agent chooses unnecessary steps or repetitive loops. Corrective action: Improve planning policy."

        # 6. Memory Failure
        elif has_memory_signal:
            category = FailureCategory.MEMORY_FAILURE
            correction_type = CorrectionType.MEMORY_UPDATE
            originating_stage = "context_assembly"
            confidence = 0.82
            evidence = [
                "User explicitly referenced stored preferences or prior session context",
                "Context store failed to recall or misattributed user preferences",
            ]
            alternative_causes = [FailureCategory.REASONING_FAILURE.value]
            root_cause = "Memory failure: Incorrect user preference recalled. Corrective action: Improve memory retrieval."

        # 7. Knowledge Gap
        elif has_procedural_gap:
            category = FailureCategory.KNOWLEDGE_FAILURE
            correction_type = CorrectionType.SKILL_UPDATE
            originating_stage = "context_assembly"
            confidence = 0.85
            evidence = [
                "Query requested procedural API or domain instructions",
                "Agent lacked procedural guidance bundle to complete the task",
            ]
            alternative_causes = [FailureCategory.REASONING_FAILURE.value]
            root_cause = "Knowledge failure: Agent lacks procedural instructions or domain knowledge. Corrective action: Update procedural skill."

        # 8. Reasoning Failure (Self-contradiction or logical error)
        elif has_tools_with_data and any(w in clean_obs.lower() for w in ["cannot browse", "no access to external", "i don't have internet"]):
            category = FailureCategory.REASONING_FAILURE
            correction_type = CorrectionType.PROMPT_CHANGE
            originating_stage = "synthesis"
            confidence = 0.84
            evidence = [
                "Agent had access to retrieved tool data but falsely claimed inability to browse",
                "Logical contradiction between retrieved facts and response conclusion",
            ]
            alternative_causes = [FailureCategory.GROUNDING_FAILURE.value]
            root_cause = "Reasoning failure: Incorrect logical conclusion and self-contradictory refusal. Corrective action: Improve reasoning strategy."

        # 9. Low Confidence Ambiguous Case
        else:
            category = FailureCategory.REASONING_FAILURE
            correction_type = CorrectionType.PROMPT_CHANGE
            originating_stage = "synthesis"
            confidence = 0.55  # Below 0.70 threshold -> flags for human triage!
            evidence = [
                "Ambiguous failure symptoms without definitive tool or grounding telemetry",
                "Response diverged from expected behavior without obvious crash or violation",
            ]
            alternative_causes = [
                FailureCategory.RETRIEVAL_FAILURE.value,
                FailureCategory.GROUNDING_FAILURE.value,
                FailureCategory.PLANNING_FAILURE.value,
            ]
            root_cause = "Ambiguous failure: Telemetry does not conclusively isolate the faulty layer. Corrective action: Request human review."

        # Responsible layer and corrective action
        responsible_layer = CATEGORY_LAYER_MAP.get(category, "synthesis_and_reasoning")
        corrective_action = CATEGORY_CORRECTIVE_ACTION_MAP.get(category, "Improve reasoning strategy")
        requires_human_triage = confidence < 0.70

        # Deterministic signature for clustering
        clean_in_sub = re.sub(r"[^\w]", "", clean_in[:40].lower())
        sig_base = f"{category.value}:{originating_stage}:{clean_in_sub}"
        signature = hashlib.sha256(sig_base.encode("utf-8")).hexdigest()[:16]

        failure_id = orig_failure_id or f"FAIL-{uuid.uuid4().hex[:8].upper()}"

        cluster = failure_db.get_or_create_cluster(
            category=category,
            signature=signature,
            rep_failure_id=failure_id,
            description=f"Cluster for {category.value} at {originating_stage}: {clean_in[:60]}",
        )
        is_recurring = cluster.failure_count > 1

        record = FailureRecord(
            failure_id=failure_id,
            session_id=session_id,
            user_id=user_id,
            user_input=clean_in,
            observed_output=clean_obs,
            expected_behavior=expected_behavior or "Generate accurate, grounded, helpful response.",
            category=category,
            correction_type=correction_type,
            root_cause=root_cause,
            originating_stage=originating_stage,
            tool_calls=t_calls,
            tool_outputs=t_outs,
            model_name=model_name,
            is_recurring=is_recurring,
            cluster_id=cluster.cluster_id,
            status=FailureStatus.REQUIRES_HUMAN_TRIAGE if requires_human_triage else FailureStatus.ANALYZED,
            responsible_layer=responsible_layer,
            corrective_action=corrective_action,
            confidence=confidence,
            evidence=evidence,
            alternative_causes=alternative_causes,
            requires_human_triage=requires_human_triage,
        )

        saved = failure_db.record_failure(record)
        logger.info(
            "failure_analyzed_with_confidence",
            extra={
                "failure_id": failure_id,
                "category": category.value,
                "responsible_layer": responsible_layer,
                "confidence": confidence,
                "requires_human_triage": requires_human_triage,
                "is_recurring": is_recurring,
            },
        )
        return saved


failure_analyzer = FailureAnalyzer()
