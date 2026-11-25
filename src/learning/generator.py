"""
Improvement Proposal Generator for Hermes Agent Platform.
Enforces architectural boundaries:
- Knowledge failure -> Procedural skill bundle
- Grounding failure -> Response validation & entity anchor filter
- Reasoning failure -> Intent alignment & synthesis prompt update
- Tool failure -> Code fix in tool execution layer
- Retrieval failure -> Search routing & query transformation update
- Memory failure -> Memory retrieval tuning & relevance threshold
- Planning failure -> Agent planning policy & loop guard
- Security failure -> Strict RBAC & authorization policy enforcement
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from src.learning.database import failure_db
from src.learning.models import (
    CorrectionType,
    FailureCategory,
    FailureRecord,
    ImprovementProposal,
)

logger = logging.getLogger("hermes.learning.generator")


class ProposalGenerator:
    """Generates precise, minimal, targeted improvement proposals based on root cause classification."""

    def generate_proposal(self, failure: FailureRecord) -> ImprovementProposal:
        """Formulate a structured improvement proposal tailored to the diagnosed failure layer."""
        proposal_id = f"PROP-{uuid.uuid4().hex[:6].upper()}"
        target_file = None
        summary = ""
        proposed_diff = ""
        rationale = ""
        correction_type = failure.correction_type
        cat_val = failure.category.value if hasattr(failure.category, "value") else str(failure.category)

        # 1. Knowledge Failure -> Skill Specification
        if cat_val == FailureCategory.KNOWLEDGE_FAILURE.value:
            correction_type = CorrectionType.SKILL_UPDATE
            target_file = f"skills/{failure.user_input[:20].lower().replace(' ', '_')}/SKILL.md"
            summary = f"Add procedural skill bundle for: {failure.user_input[:60]}"
            rationale = (
                "The agent lacked procedural domain knowledge or tool execution guidance. "
                "Per Hermes self-improvement policy, reusable procedural knowledge is formalized as a Skill."
            )
            proposed_diff = (
                f"---\n"
                f"name: {failure.user_input[:20].lower().replace(' ', '-')}\n"
                f"description: Procedural guidance for {failure.user_input[:60]}\n"
                f"---\n\n"
                f"# Purpose & Requirements\n"
                f"When addressing requests like '{failure.user_input}', adhere to the following verified procedure:\n\n"
                f"1. Identify the requested operation and parameters.\n"
                f"2. Execute the appropriate tool rather than stating inability.\n"
                f"3. Synthesize and verify findings against retrieved evidence.\n"
            )

        # 2. Context Grounding / Grounding Failure -> Response Validation / Entity Anchor Filter
        elif cat_val in (FailureCategory.GROUNDING_FAILURE.value, FailureCategory.CONTEXT_GROUNDING_FAILURE.value):
            correction_type = CorrectionType.RESPONSE_VALIDATION
            target_file = "src/security/output_guards.py"
            summary = f"Enforce entity-anchor validation against hallucinated entities in synthesis"
            rationale = (
                "Tool outputs contained relevant data, but the model hallucinated an unsupported entity or claim. "
                "Adding a skill would only patch surface symptoms. The root fix requires evidence verification "
                "and response validation that cross-references output entity tokens against query and tool observation context."
            )
            proposed_diff = (
                f"+ # Grounding Filter for: {failure.user_input[:40]}\n"
                f"+ def validate_entity_grounding(output_text: str, allowed_context: str) -> str:\n"
                f"+     # Filter out ungrounded repositories or hallucinated entity references\n"
                f"+     return filter_unsupported_entities(output_text, allowed_context)\n"
            )

        # 3. Retrieval Failure -> Search Routing & Query Formulation
        elif cat_val in (FailureCategory.RETRIEVAL_FAILURE.value, FailureCategory.TOOL_ORCHESTRATION_FAILURE.value):
            correction_type = CorrectionType.ROUTING_CHANGE
            target_file = "src/agent/search_router.py"
            summary = f"Improve search routing & query reformulation for: {failure.user_input[:40]}"
            rationale = (
                "The retrieval layer failed to return relevant webpages or documents. "
                "Improving search query routing and query transformation prevents empty or irrelevant retrievals."
            )
            proposed_diff = (
                f"+ # Search routing improvement for: {failure.user_input[:30]}\n"
                f"+ FRESHNESS_PATTERNS.append(r'\\b(?:{failure.user_input[:20].lower()})\\b')\n"
                f"+ def rewrite_query_for_retrieval(query: str) -> str:\n"
                f"+     return expand_domain_terms(query)\n"
            )

        # 4. Tool Failure / System Failure -> Execution Layer Code Patch
        elif cat_val in (FailureCategory.TOOL_FAILURE.value, FailureCategory.SYSTEM_FAILURE.value):
            correction_type = CorrectionType.CODE_FIX
            target_file = "src/tools/execution_engine.py"
            summary = f"Fix runtime exception and error handling in tool execution engine"
            rationale = (
                f"Root cause was an execution layer crash or tool error: {failure.root_cause}. "
                "This requires a code-level fix in the centralized tool execution engine."
            )
            proposed_diff = (
                f"+ try:\n"
                f"+     # Safe execution wrapper preventing system crash or unhandled error\n"
                f"+     result = execute_with_circuit_breaker(request)\n"
                f"+ except Exception as exc:\n"
                f"+     logger.warning('Handled tool execution error safely: %s', exc)\n"
            )

        # 5. Memory Failure -> Memory Retrieval Tuning
        elif cat_val == FailureCategory.MEMORY_FAILURE.value:
            correction_type = CorrectionType.MEMORY_UPDATE
            target_file = "src/memory/mem0_client.py"
            summary = f"Tune memory retrieval and relevance threshold for: {failure.user_input[:40]}"
            rationale = (
                "The memory subsystem failed to recall or misattributed user preferences. "
                "Improving memory search filtering and relevance thresholds ensures accurate preference application."
            )
            proposed_diff = (
                f"+ # Memory retrieval tuning\n"
                f"+ MEMORY_RELEVANCE_THRESHOLD = 0.85\n"
                f"+ def retrieve_user_preferences_filtered(user_id: str, query: str) -> list[str]:\n"
                f"+     return memory_client.search(user_id=user_id, query=query, score_threshold=0.85)\n"
            )

        # 6. Planning Failure -> Planning Policy & Loop Guard
        elif cat_val == FailureCategory.PLANNING_FAILURE.value:
            correction_type = CorrectionType.PLANNING_POLICY
            target_file = "src/agent/planner.py"
            summary = f"Update planning policy to prevent redundant steps for: {failure.user_input[:40]}"
            rationale = (
                "The agent planner chose unnecessary or cyclic steps. "
                "Enforcing loop prevention and strict transition guards improves execution efficiency."
            )
            proposed_diff = (
                f"+ # Loop guard: prevent repeating identical tool calls\n"
                f"+ if is_redundant_step(planned_action, history):\n"
                f"+     planned_action = abort_or_fallback(planned_action)\n"
            )

        # 7. Security Failure -> Authorization Policy Patch
        elif cat_val == FailureCategory.SECURITY_FAILURE.value:
            correction_type = CorrectionType.POLICY_FIX
            target_file = "src/security/rbac.py"
            summary = f"Enforce authorization policy and RBAC boundary for: {failure.user_input[:40]}"
            rationale = (
                "An unauthorized tool invocation or policy violation occurred. "
                "Strict authorization policy enforcement and RBAC role checks ensure secure execution."
            )
            proposed_diff = (
                f"+ # Enforce strict RBAC policy check before tool invocation\n"
                f"+ if not rbac.has_permission(user_role, required_permission):\n"
                f"+     raise PermissionDeniedError('Action blocked by RBAC policy')\n"
            )

        # 8. Reasoning Failure -> Prompt & Intent Alignment Guidance
        else:
            correction_type = CorrectionType.PROMPT_CHANGE
            target_file = "src/agent/prompts.py"
            summary = f"Improve reasoning & intent-alignment instructions for: {failure.user_input[:50]}"
            rationale = (
                "The agent had access to information but misaligned on user intent or self-contradicted. "
                "A targeted synthesis prompt constraint ensures alignment with user evidence."
            )
            proposed_diff = (
                f"+ - INTENT ALIGNMENT: When evidence is retrieved for '{failure.user_input[:30]}', "
                f"directly answer the user's core question. Do NOT refuse or introduce unrelated disclaimers.\n"
            )

        proposal = ImprovementProposal(
            proposal_id=proposal_id,
            failure_id=failure.failure_id,
            cluster_id=failure.cluster_id,
            correction_type=correction_type,
            summary=summary,
            proposed_diff=proposed_diff,
            target_file=target_file,
            rationale=rationale,
            validation_status="pending",
            approval_status="pending",
            deployment_stage="staging",
        )

        saved = failure_db.save_proposal(proposal)
        logger.info(
            "improvement_proposal_generated",
            extra={"proposal_id": proposal_id, "type": correction_type.value, "target_file": target_file},
        )
        return saved


proposal_generator = ProposalGenerator()
