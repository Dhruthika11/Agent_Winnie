"""
LangGraph state machine that drives the agent loop.

Nodes:
    load_memory     -> retrieve relevant long-term memory for this user
    call_model      -> invoke the LLM (via LiteLLM) with tools bound
    check_safety    -> classify any requested tool calls, gate high-risk ones
    await_approval  -> terminal-for-this-turn node; waits on a human decision
    execute_tools   -> run approved/low-risk tool calls (MCP or sandbox)
    update_memory   -> persist salient facts from this turn back to Mem0
    enforce_limit   -> stop the loop if max_iterations is exceeded

Edges implement an explicit, bounded loop:
    load_memory -> call_model -> (route) -> {execute_tools, END}
    execute_tools -> (route) -> {await_approval, call_model}
    await_approval -> END   (resumed externally via the approvals API)
"""
from __future__ import annotations

import datetime
import json
import logging
import re
import uuid
from typing import Literal, Optional

import litellm
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from src.agent.pipeline_common import (
    clean_final_reply,
    create_tool_call_record,
    execute_tool_with_guard,
)
from src.agent.prompts import (
    build_recovery_prompt,
    build_synthesis_prompt,
    build_system_prompt,
)
from src.agent.router import agent_router
from src.agent.state import AgentState
from src.agent.tool_parser import parse_model_tool_calls, strip_unwanted_tool_xml
from src.config import settings
from src.memory.manager import memory_manager
from src.security.input_guards import sanitize_input
from src.security.output_guards import sanitize_output
from src.tools import hitl
from src.tools.mcp_client import mcp_registry
from src.tools.sandbox import sandbox
from src.utils.telemetry import get_langfuse_client

logger = logging.getLogger("hermes.agent.graph")


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def load_memory(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    q = config.get("configurable", {}).get("event_queue") if config else None
    if q:
        q.put({"type": "status", "message": "Recalling relevant context from memory..."})

    start_time = datetime.datetime.now(datetime.timezone.utc)
    last_user_msg = _last_user_text(state["messages"])
    tenant_id = state.get("tenant_id") or "default"
    memories = memory_manager.search(user_id=state["user_id"], query=last_user_msg, tenant_id=tenant_id)
    
    # Phase 4: Sanitize retrieved memories using memory policy
    from src.agent.memory_policy import memory_policy
    cleaned_memories = memory_policy.sanitize_retrieved_memories(memories)
    context = "\n".join(f"- {m}" for m in cleaned_memories) if cleaned_memories else ""

    exec_id = state.get("execution_id")
    if exec_id:
        try:
            from src.agent.durable import ExecutionLifecycleState, durable_store
            durable_store.record_transition(
                execution_id=exec_id,
                state_from=ExecutionLifecycleState.REQUEST_RECEIVED.value,
                state_to=ExecutionLifecycleState.AGENT_RUNNING.value,
                state_payload={"step": "load_memory", "user_id": state.get("user_id")},
                session_id=state.get("session_id", "default"),
                user_id=state.get("user_id", "default"),
                tenant_id=state.get("tenant_id") or "default",
            )
        except Exception:
            pass

    lf = get_langfuse_client()
    trace_id = state.get("trace_id")
    parent_span_id = state.get("parent_span_id")
    if lf and trace_id:
        try:
            span = lf.span(
                trace_id=trace_id,
                parent_observation_id=parent_span_id,
                name="load_memory",
                input={"user_id": state["user_id"], "query": last_user_msg},
                start_time=start_time,
            )
            span.end(output=cleaned_memories)
        except Exception:
            pass

    return {"memory_context": context}


def call_model(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    start_time = datetime.datetime.now(datetime.timezone.utc)
    q = config.get("configurable", {}).get("event_queue") if config else None
    if q:
        q.put({"type": "status", "message": f"Winnie is thinking (turn {state.get('iteration', 0) + 1})..."})

    tool_schemas = mcp_registry.tool_schemas()
    tool_descriptions = "\n".join(
        f"- {t['function']['name']}: {t['function'].get('description', '')}"
        for t in tool_schemas
    )

    history = _messages_to_litellm(state["messages"])
    last_user_text_val = _last_user_text(state["messages"])
    clean_last, flagged, reason = sanitize_input(last_user_text_val)
    if flagged:
        logger.warning("input_guard_flagged", extra={"reason": reason})

    # Phase 2: Intent Classification & Auditable Search Decision Routing
    from src.agent.intent_classifier import classify_user_intent, IntentCategory
    from src.agent.search_router import search_decision_router
    from src.security.audit import audit_logger

    user_intent = classify_user_intent(clean_last)

    # If this is an explicit memory instruction, persist it synchronously immediately
    if user_intent.category == IntentCategory.MEMORY_INSTRUCTION:
        tenant_id = state.get("tenant_id") or "default"
        try:
            memory_manager.add(
                user_id=state["user_id"],
                messages=[{"role": "user", "content": clean_last}],
                tenant_id=tenant_id,
                category="preference",
            )
            if q:
                q.put({"type": "status", "message": "Updated persistent user preferences in memory..."})
        except Exception as mem_err:
            logger.warning("failed_to_save_memory_instruction: %s", mem_err)

    search_decision = search_decision_router.evaluate(
        clean_last,
        conversation_context=state.get("memory_context", ""),
        has_attachments=bool(state.get("images")),
    )
    # Record auditable search routing decision
    audit_logger.record_search_decision(
        clean_last,
        search_decision.to_dict(),
        session_id=state.get("session_id", "default"),
        user_id=state.get("user_id", "default"),
    )

    search_guidance = ""
    if search_decision.requires_search:
        search_guidance = (
            f"### Search Decision Policy [Domain: {search_decision.search_type.value.upper()}]\n"
            f"- Search Required: {search_decision.reason}\n"
            f"- Recommended Search Queries: {', '.join(search_decision.queries)}\n"
            f"- Ground all factual claims directly on retrieved sources."
        )
    elif search_decision.signals and search_decision.signals.context_sufficient:
        search_guidance = (
            "### Context Sufficiency Policy\n"
            "- The prompt/context is self-contained. Answer directly without external search."
        )

    # Central Evidence Sufficiency & Guidance Layer
    evidence_items = state.get("evidence_items", [])
    evidence_verification_data = None
    evidence_guidance = ""
    if evidence_items:
        from src.agent.evidence_verifier import central_evidence_verifier
        from src.agent.evidence_store import EvidenceStore
        report = central_evidence_verifier.verify(clean_last, evidence_items)
        evidence_verification_data = report.to_dict()
        evidence_store = EvidenceStore.from_dict_list(evidence_items)
        evidence_context = evidence_store.format_for_context()
        evidence_guidance = (
            f"### Verified Evidence Context\n{evidence_context}\n\n"
            f"### Evidence Sufficiency & Grounding Policy\n{report.synthesis_guidance}"
        )

    # Allow tools during first 3 iterations (intent guidance guides safe behavior, memory instructions need no tools)
    allow_tools = bool(tool_schemas) and (state.get("iteration", 0) <= 3) and (user_intent.category != IntentCategory.MEMORY_INSTRUCTION)
    effective_tools = tool_schemas if allow_tools else None
    effective_tool_choice = "auto" if allow_tools else None

    guidance_parts = [p for p in (user_intent.guidance, search_guidance, evidence_guidance) if p]
    combined_guidance = "\n\n".join(guidance_parts)

    if user_intent.category == IntentCategory.MEMORY_INSTRUCTION:
        system_prompt = (
            "You are Winnie, a helpful AI assistant. The user just gave an explicit instruction to remember a preference or context.\n"
            f"User instruction: '{clean_last}'\n"
            "This has been recorded into persistent memory. Confirm directly, warmly, and concisely that this fact/preference has been saved.\n"
            "DO NOT re-answer previous questions from the chat history or generate unsolicited technical comparisons unless explicitly requested."
        )
        active_history = history
    elif allow_tools:
        system_prompt = build_system_prompt(state["memory_context"], tool_descriptions, intent_guidance=combined_guidance)
        active_history = history
    else:
        system_prompt = build_synthesis_prompt(state["memory_context"], intent_guidance=combined_guidance)
        active_history = _messages_to_litellm_synthesis(state["messages"])

    raw_content = ""
    pending_calls = []
    tool_calls_payload = []
    response = None

    from src.agent.prompt_manager import prompt_budget_manager

    # Token-Aware Prompt Budgeting: dynamically filter tools and compact history within safe context & TPM limits
    budget_res = prompt_budget_manager.build_budgeted_prompt(
        system_prompt=system_prompt,
        messages=state["messages"],
        tools=effective_tools,
        model=agent_router.last_dispatched_model or settings.model_name,
        intent_category=user_intent.category.value if hasattr(user_intent.category, "value") else str(user_intent.category),
        requires_search=search_decision.requires_search,
        allow_tools=allow_tools,
        requested_max_output=2000,
    )
    final_messages = [{"role": "system", "content": budget_res.system_prompt}] + budget_res.messages
    final_tools = budget_res.tools
    final_tool_choice = "auto" if final_tools else None

    if q:
        tag_buffer = ""
        suppress_stream = False
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}

        try:
            stream_iter = agent_router.completion(
                messages=final_messages,
                tools=final_tools,
                tool_choice=final_tool_choice,
                max_tokens=2000,
                stream=True,
                trace_id=state.get("trace_id"),
                parent_span_id=state.get("parent_span_id"),
                user_id=state.get("user_id"),
            )
            for chunk in stream_iter:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    raw_content += delta.content
                    if delta.tool_calls or suppress_stream:
                        pass
                    elif "<" in delta.content or tag_buffer:
                        tag_buffer += delta.content
                        if any(k in tag_buffer for k in ("<tool_call", "<function", "<parameter")):
                            suppress_stream = True
                            tag_buffer = ""
                        elif ">" in tag_buffer or len(tag_buffer) > 30:
                            q.put({"type": "token", "delta": tag_buffer})
                            tag_buffer = ""
                    else:
                        q.put({"type": "token", "delta": delta.content})

                if delta.tool_calls:
                    suppress_stream = True
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                        if tc.id:
                            accumulated_tool_calls[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                clean_name = tc.function.name.split("<|")[0].strip()
                                accumulated_tool_calls[idx]["name"] = clean_name
                            if tc.function.arguments:
                                accumulated_tool_calls[idx]["arguments"] += tc.function.arguments

            if tag_buffer and not suppress_stream:
                q.put({"type": "token", "delta": tag_buffer})
                tag_buffer = ""

            for idx in sorted(accumulated_tool_calls.keys()):
                tc = accumulated_tool_calls[idx]
                tc_id = tc["id"] or f"call_{uuid.uuid4().hex[:8]}"
                tc_name = tc["name"]
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except Exception:
                    args = {}
                from src.agent.tool_policy import tool_policy
                eval_res = tool_policy.evaluate(tc_name, args, user_id=state.get("user_id", "default"))
                pending_calls.append({
                    "id": tc_id,
                    "name": tc_name,
                    "arguments": args,
                    "risk_level": eval_res.risk_level.value,
                    "policy_decision": eval_res.decision.value,
                    "policy_reason": eval_res.reason,
                })
                tool_calls_payload.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc_name,
                        "arguments": tc["arguments"] or "{}",
                    },
                })
        except Exception as exc:
            logger.warning(f"router_completion_failed ({exc}). Attempting recovery completion without tools.")
            try:
                recovery_prompt = build_recovery_prompt()
                recovery_history = _messages_to_litellm_synthesis(state["messages"])
                fb_iter = agent_router.completion(
                    messages=[{"role": "system", "content": recovery_prompt}] + recovery_history,
                    tools=None,
                    max_tokens=3500,
                    stream=True,
                )
                raw_content = ""
                for fb_chunk in fb_iter:
                    if not fb_chunk.choices:
                        continue
                    fb_delta = fb_chunk.choices[0].delta
                    if fb_delta.content:
                        clean_fb = strip_unwanted_tool_xml(fb_delta.content) if "<" in fb_delta.content else fb_delta.content
                        if clean_fb:
                            raw_content += clean_fb
                            q.put({"type": "token", "delta": clean_fb})
            except Exception as fb_exc:
                err_str = f"Inference error: {fb_exc}"
                q.put({"type": "error", "detail": err_str})
                return {
                    "messages": [{"role": "assistant", "content": err_str}],
                    "pending_tool_calls": [],
                    "iteration": state["iteration"] + 1,
                    "done": True,
                    "stop_reason": "provider_error",
                }
    else:
        # Non-streaming mode (e.g. synchronous /chat endpoint)
        try:
            response = agent_router.completion(
                messages=final_messages,
                tools=final_tools,
                tool_choice=final_tool_choice,
                max_tokens=2000,
                trace_id=state.get("trace_id"),
                parent_span_id=state.get("parent_span_id"),
                user_id=state.get("user_id"),
            )
        except Exception as exc:
            logger.warning(f"router_completion_failed ({exc}). Attempting recovery completion without tools.")
            try:
                recovery_budget_res = prompt_budget_manager.build_budgeted_prompt(
                    system_prompt=system_prompt,
                    messages=state["messages"],
                    tools=None,
                    model=agent_router.last_dispatched_model or settings.model_name,
                    intent_category=user_intent.category.value if hasattr(user_intent.category, "value") else str(user_intent.category),
                    requires_search=False,
                    allow_tools=False,
                    requested_max_output=2000,
                )
                response = agent_router.completion(
                    messages=[{"role": "system", "content": recovery_budget_res.system_prompt}] + recovery_budget_res.messages,
                    tools=None,
                    model=agent_router.last_dispatched_model or settings.model_name or LOGICAL_AGENT_MODEL,
                    max_tokens=2000,
                )
            except Exception as fb_exc:
                logger.exception("router_completion_failed_permanent", extra={"error": str(fb_exc)})
                error_str = str(fb_exc)
                return {
                    "messages": [{"role": "assistant", "content": f"Inference error: {error_str}"}],
                    "pending_tool_calls": [],
                    "iteration": state["iteration"] + 1,
                    "done": True,
                    "stop_reason": "provider_error",
                }

        choice = response.choices[0].message
        raw_content = choice.content or ""
        from src.agent.tool_policy import tool_policy
        for tc in getattr(choice, "tool_calls", None) or []:
            args = json.loads(tc.function.arguments or "{}")
            clean_tc_name = tc.function.name.split("<|")[0].strip()
            eval_res = tool_policy.evaluate(clean_tc_name, args, user_id=state.get("user_id", "default"))
            pending_calls.append({
                "id": tc.id,
                "name": clean_tc_name,
                "arguments": args,
                "risk_level": eval_res.risk_level.value,
                "policy_decision": eval_res.decision.value,
                "policy_reason": eval_res.reason,
            })
            tool_calls_payload.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": clean_tc_name,
                    "arguments": tc.function.arguments or "{}",
                },
            })

    # Always check if model emitted XML tool calls in raw_content
    clean_content, xml_calls = parse_model_tool_calls(raw_content)
    if xml_calls and allow_tools:
        from src.agent.tool_policy import tool_policy
        for xc in xml_calls:
            eval_res = tool_policy.evaluate(xc["name"], xc["arguments"], user_id=state.get("user_id", "default"))
            pending_calls.append({
                "id": xc["id"],
                "name": xc["name"],
                "arguments": xc["arguments"],
                "risk_level": eval_res.risk_level.value,
                "policy_decision": eval_res.decision.value,
                "policy_reason": eval_res.reason,
            })
            tool_calls_payload.append({
                "id": xc["id"],
                "type": "function",
                "function": {
                    "name": xc["name"],
                    "arguments": json.dumps(xc["arguments"]),
                },
            })
        raw_content = clean_content

    # Phase 3: Grounding Self-Correction & Verification Re-Search Gate
    grounding_retries = state.get("grounding_retries", 0)
    max_grounding_retries = 2

    if not pending_calls and allow_tools and (grounding_retries < max_grounding_retries):
        from src.security.output_guards import validate_evidence_grounding, verify_claims_against_evidence

        # Check A: Unverified profile or external links in model generation
        _, has_unverified_urls, unverified_urls = validate_evidence_grounding(
            raw_content, evidence_urls=state.get("evidence_urls", [])
        )

        # Check B: Unverified claims, quotes, or misattributions
        _, claim_records, has_claim_violations = verify_claims_against_evidence(
            raw_content,
            evidence_items=state.get("evidence_items", []),
            evidence_urls=state.get("evidence_urls", []),
        )

        # Check C: Search required by policy, but no verified evidence retrieved
        requires_search = search_decision.requires_search if search_decision else False
        evidence_items = state.get("evidence_items", [])
        has_verified_evidence = any(
            e.get("validation_status") == "verified" for e in evidence_items
        )
        needs_initial_search = requires_search and not has_verified_evidence

        if has_unverified_urls or has_claim_violations or needs_initial_search:
            # Formulate targeted recovery search query
            recovery_query = ""
            if unverified_urls:
                recovery_query = f"{clean_last} official site source"
            elif has_claim_violations:
                unsupported_claims = [
                    r for r in claim_records
                    if r.get("status") in ("unsupported", "unverified_quote", "misattributed")
                ]
                if unsupported_claims:
                    target_claim = unsupported_claims[0]
                    attr_src = target_claim.get("attributed_source") or ""
                    figs = " ".join(target_claim.get("extracted_figures", [])[:2])
                    recovery_query = f"{clean_last} {attr_src} {figs}".strip()
            elif needs_initial_search and search_decision and search_decision.queries:
                recovery_query = search_decision.queries[0]

            if not recovery_query:
                recovery_query = clean_last

            # Synthesize automated grounding recovery tool call
            call_id = f"call_grounding_recovery_{uuid.uuid4().hex[:8]}"
            from src.agent.tool_policy import tool_policy
            eval_res = tool_policy.evaluate(
                "web_scraper.search",
                {"query": recovery_query, "max_results": 5},
                user_id=state.get("user_id", "default"),
            )
            recovery_call = {
                "id": call_id,
                "name": "web_scraper.search",
                "arguments": {"query": recovery_query, "max_results": 5},
                "risk_level": eval_res.risk_level.value,
                "policy_decision": eval_res.decision.value,
                "policy_reason": "Automated grounding recovery: searching for verified evidence before answering",
            }
            pending_calls.append(recovery_call)
            tool_calls_payload.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "web_scraper.search",
                    "arguments": json.dumps({"query": recovery_query, "max_results": 5}),
                },
            })
            grounding_retries += 1
            if q:
                q.put({
                    "type": "status",
                    "message": f"Grounding check: Information unverified. Searching to retrieve verified sources: '{recovery_query}'...",
                })

    # Phase 4 & 5: Enforce URL grounding, claim-to-source verification, and input-to-output grounding
    last_user_prompt = _last_user_text(state["messages"])
    current_intent_val = user_intent.category.value if user_intent else None
    assistant_content = sanitize_output(
        strip_unwanted_tool_xml(raw_content),
        evidence_urls=state.get("evidence_urls", []),
        evidence_items=state.get("evidence_items", []),
        prompt=last_user_prompt,
        intent_category=current_intent_val,
    )
    if not pending_calls and not assistant_content:
        # If assistant content was empty after tool execution, force a synthesis pass
        try:
            synthesis_prompt = build_synthesis_prompt(
                state.get("memory_context", ""), intent_guidance=combined_guidance
            )
            synth_resp = agent_router.completion(
                messages=[{"role": "system", "content": synthesis_prompt}] + _messages_to_litellm_synthesis(state["messages"]),
                tools=None,
                max_tokens=2500,
                stream=False,
            )
            if synth_resp and synth_resp.choices and synth_resp.choices[0].message.content:
                assistant_content = sanitize_output(
                    strip_unwanted_tool_xml(synth_resp.choices[0].message.content),
                    evidence_urls=state.get("evidence_urls", []),
                    evidence_items=state.get("evidence_items", []),
                )
        except Exception as exc:
            logger.warning("synthesis_recovery_failed: %s", exc)

        if not assistant_content:
            tool_msgs = [m for m in state.get("messages", []) if (isinstance(m, dict) and m.get("role") == "tool")]
            if tool_msgs:
                assistant_content = clean_final_reply("", [{"tool": m.get("name", "tool")} for m in tool_msgs])

    # If retries were exhausted and content contains unverified link markers, ensure clean presentation
    if not pending_calls and grounding_retries >= max_grounding_retries and assistant_content:
        if "[Unverified link `" in assistant_content:
            assistant_content = re.sub(
                r"\[Unverified link `[^`]+` filtered — not verified in search results\]",
                "[Link unverified in search results]",
                assistant_content,
            )

    assistant_msg = {"role": "assistant", "content": assistant_content}
    if tool_calls_payload:
        assistant_msg["tool_calls"] = tool_calls_payload

    lf = get_langfuse_client()
    trace_id = state.get("trace_id")
    parent_span_id = state.get("parent_span_id")
    if lf and trace_id:
        try:
            usage = None
            if response and hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
            gen = lf.generation(
                trace_id=trace_id,
                parent_observation_id=parent_span_id,
                name="call_model",
                model=getattr(response, "model", settings.model_name) if response else settings.model_name,
                input=history,
                start_time=start_time,
                usage=usage,
                metadata={
                    "intent": user_intent.category.value,
                    "confidence": user_intent.confidence,
                    "is_action_proposal": user_intent.is_action_proposal,
                    "routing_events": agent_router.last_routing_events,
                    "fallback_occurred": len(agent_router.last_routing_events) > 1,
                },
            )
            gen.end(output=assistant_content or str(pending_calls))
        except Exception:
            pass

    res_dict = {
        "messages": [assistant_msg],
        "pending_tool_calls": pending_calls,
        "iteration": state["iteration"] + 1,
        "grounding_retries": grounding_retries,
        "detected_intent": user_intent.category.value,
        "search_decision": search_decision.to_dict(),
    }
    if evidence_verification_data:
        res_dict["evidence_verification"] = evidence_verification_data
    return res_dict


def check_safety(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    from src.agent.tool_policy import PolicyDecision
    
    # Phase 1: Evaluate pending calls using ToolPolicy and route approval-required actions to HITL
    approval_required = [
        c for c in state["pending_tool_calls"]
        if c.get("policy_decision") == PolicyDecision.REQUIRE_APPROVAL.value
        or c.get("risk_level") in ("high", "critical")
    ]
    if not approval_required:
        return {"awaiting_approval": False, "approval_id": None}

    approval_id = str(uuid.uuid4())
    clean_state = dict(state)
    clean_state["messages"] = _messages_to_litellm(state.get("messages", []))
    clean_state["awaiting_approval"] = True
    clean_state["approval_id"] = approval_id

    hitl.request_approval(
        user_id=state["user_id"],
        session_id=state["session_id"],
        tool_calls=approval_required,
        state=clean_state,
        approval_id=approval_id,
        tenant_id=state.get("tenant_id") or "default",
    )

    q = config.get("configurable", {}).get("event_queue") if config else None
    if q:
        q.put({
            "type": "hitl_required",
            "approval_id": approval_id,
            "tool_calls": approval_required,
            "message": "High-risk tool call requires approval before execution.",
        })

    lf = get_langfuse_client()
    trace_id = state.get("trace_id")
    parent_span_id = state.get("parent_span_id")
    if lf and trace_id:
        try:
            span = lf.span(
                trace_id=trace_id,
                parent_observation_id=parent_span_id,
                name="check_safety",
                input={"tool_calls": approval_required},
            )
            span.end(output={"awaiting_approval": True, "approval_id": approval_id})
        except Exception:
            pass

    return {"awaiting_approval": True, "approval_id": approval_id}


def await_approval(state: AgentState) -> dict:
    exec_id = state.get("execution_id")
    if exec_id:
        try:
            from src.agent.durable import ExecutionLifecycleState, durable_store
            durable_store.record_transition(
                execution_id=exec_id,
                state_from=ExecutionLifecycleState.AGENT_RUNNING.value,
                state_to=ExecutionLifecycleState.WAITING_FOR_APPROVAL.value,
                state_payload={
                    "approval_id": state.get("approval_id"),
                    "pending_tool_calls": state.get("pending_tool_calls", []),
                },
                session_id=state.get("session_id", "default"),
                user_id=state.get("user_id", "default"),
                tenant_id=state.get("tenant_id") or "default",
            )
        except Exception:
            pass
    return {
        "done": True,
        "stop_reason": f"awaiting_human_approval:{state['approval_id']}",
    }


def execute_tools(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    results = []
    q = config.get("configurable", {}).get("event_queue") if config else None
    lf = get_langfuse_client()
    trace_id = state.get("trace_id")
    parent_span_id = state.get("parent_span_id")

    exec_id = state.get("execution_id")
    if exec_id:
        try:
            from src.agent.durable import ExecutionLifecycleState, durable_store
            durable_store.record_transition(
                execution_id=exec_id,
                state_from=ExecutionLifecycleState.AGENT_RUNNING.value,
                state_to=ExecutionLifecycleState.WAITING_FOR_TOOL.value,
                state_payload={"pending_tool_calls": state.get("pending_tool_calls", [])},
                session_id=state.get("session_id", "default"),
                user_id=state.get("user_id", "default"),
                tenant_id=state.get("tenant_id") or "default",
            )
        except Exception:
            pass

    from src.security.output_guards import _URL_RE
    extracted_urls = list(state.get("evidence_urls", []))
    accumulated_evidence = list(state.get("evidence_items", []))

    for call in state["pending_tool_calls"]:
        start_time = datetime.datetime.now(datetime.timezone.utc)
        if call.get("risk_level") in ("high", "critical") and not hitl.is_approved(state.get("approval_id")):
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": "Blocked: this action requires human approval and none was granted.",
                }
            )
            continue

        tool_thought = f"Decided to invoke '{call['name']}' with parameters: {json.dumps(call.get('arguments', {}))}."
        if q:
            q.put({
                "type": "tool_start",
                "id": call["id"],
                "tool": call["name"],
                "arguments": call.get("arguments", {}),
                "thought": tool_thought,
            })

        rec = execute_tool_with_guard(
            tool_name=call["name"],
            tool_args=call.get("arguments", {}),
            call_id=call["id"],
            user_id=state.get("user_id", "default"),
            session_id=state.get("session_id", "default"),
            tenant_id=state.get("tenant_id") or "default",
            user_role=state.get("user_role") or "viewer",
            idempotency_key=f"{exec_id or state.get('session_id', 'exec')}:{call['id']}",
            approval_id=state.get("approval_id"),
        )
        output = rec["output"]

        # Central Evidence Store: record normalized, validated evidence item
        ev_item = rec.get("evidence_item")
        if ev_item:
            accumulated_evidence.append(ev_item)

        # Harvest verified evidence URLs from tool outputs
        found_urls = _URL_RE.findall(str(output))
        for u in found_urls:
            if u not in extracted_urls:
                extracted_urls.append(u)

        if q:
            q.put({
                "type": "tool_end",
                "id": rec["id"],
                "tool": rec["tool"],
                "preview": rec["preview"],
                "output": rec["output"],
                "thought": rec["thought"],
                "validation_status": rec.get("validation_status", "verified"),
                "validation_details": rec.get("validation_details", {}),
            })

        if lf and trace_id:
            try:
                tool_span = lf.span(
                    trace_id=trace_id,
                    parent_observation_id=parent_span_id,
                    name=f"tool:{call['name']}",
                    input=call.get("arguments", {}),
                    start_time=start_time,
                )
                tool_span.end(output=str(output)[:2000])
            except Exception:
                pass

        results.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": output,
            }
        )

    if exec_id:
        try:
            from src.agent.durable import ExecutionLifecycleState, durable_store
            durable_store.record_transition(
                execution_id=exec_id,
                state_from=ExecutionLifecycleState.WAITING_FOR_TOOL.value,
                state_to=ExecutionLifecycleState.TOOL_EXECUTED.value,
                state_payload={"executed_count": len(results)},
                session_id=state.get("session_id", "default"),
                user_id=state.get("user_id", "default"),
                tenant_id=state.get("tenant_id") or "default",
            )
        except Exception:
            pass

    return {
        "messages": results,
        "pending_tool_calls": [],
        "awaiting_approval": False,
        "evidence_urls": extracted_urls,
        "evidence_items": accumulated_evidence,
    }


def update_memory(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    exec_id = state.get("execution_id")
    if exec_id and not state.get("awaiting_approval"):
        try:
            from src.agent.durable import ExecutionLifecycleState, durable_store
            durable_store.record_transition(
                execution_id=exec_id,
                state_from=ExecutionLifecycleState.AGENT_RUNNING.value,
                state_to=ExecutionLifecycleState.COMPLETED.value,
                state_payload={"messages_count": len(state.get("messages", []))},
                session_id=state.get("session_id", "default"),
                user_id=state.get("user_id", "default"),
                tenant_id=state.get("tenant_id") or "default",
            )
        except Exception:
            pass

    if state.get("awaiting_approval"):
        return {}

    # Phase 4: Filter out conversational refusals, temporary errors, and ephemeral turns
    from src.agent.memory_policy import memory_policy
    decision = memory_policy.evaluate_turn(state.get("messages", []))
    if not decision.should_store:
        logger.info("memory_persistence_skipped_by_policy", extra={"reason": decision.reason})
        return {}

    q = config.get("configurable", {}).get("event_queue") if config else None
    if q:
        q.put({"type": "status", "message": "Saving context to memory..."})

    start_time = datetime.datetime.now(datetime.timezone.utc)
    tenant_id = state.get("tenant_id") or "default"
    memory_manager.add(user_id=state["user_id"], messages=state["messages"][-4:], tenant_id=tenant_id)
    lf = get_langfuse_client()
    trace_id = state.get("trace_id")
    parent_span_id = state.get("parent_span_id")
    if lf and trace_id:
        try:
            span = lf.span(
                trace_id=trace_id,
                parent_observation_id=parent_span_id,
                name="update_memory",
                input={"user_id": state["user_id"], "category": decision.category},
                start_time=start_time,
            )
            span.end(output="updated")
        except Exception:
            pass
    return {}


def enforce_limit(state: AgentState) -> dict:
    if state["iteration"] >= state["max_iterations"]:
        return {
            "done": True,
            "stop_reason": "max_iterations_reached",
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "I've hit my step limit for this turn. Here is my best "
                        "answer based on what I've gathered so far."
                    ),
                }
            ],
        }
    return {}


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def route_after_model(state: AgentState) -> Literal["check_safety", "end"]:
    return "check_safety" if state["pending_tool_calls"] else "end"


def route_after_safety(state: AgentState) -> Literal["await_approval", "execute_tools"]:
    return "await_approval" if state["awaiting_approval"] else "execute_tools"


def route_after_tools(state: AgentState) -> Literal["enforce_limit", "call_model"]:
    return "enforce_limit" if state["iteration"] >= state["max_iterations"] else "call_model"


def route_after_limit(state: AgentState) -> Literal["update_memory", "call_model"]:
    return "update_memory" if state["done"] else "call_model"


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_memory", load_memory)
    graph.add_node("call_model", call_model)
    graph.add_node("check_safety", check_safety)
    graph.add_node("await_approval", await_approval)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("enforce_limit", enforce_limit)
    graph.add_node("update_memory", update_memory)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "call_model")

    graph.add_conditional_edges(
        "call_model",
        route_after_model,
        {"check_safety": "check_safety", "end": "update_memory"},
    )
    graph.add_conditional_edges(
        "check_safety",
        route_after_safety,
        {"await_approval": "await_approval", "execute_tools": "execute_tools"},
    )
    graph.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {"enforce_limit": "enforce_limit", "call_model": "call_model"},
    )
    graph.add_conditional_edges(
        "enforce_limit",
        route_after_limit,
        {"update_memory": "update_memory", "call_model": "call_model"},
    )

    graph.add_edge("await_approval", END)
    graph.add_edge("update_memory", END)

    from src.agent.checkpointer import get_checkpointer

    return graph.compile(checkpointer=get_checkpointer())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
        if role in ("user", "human"):
            content = msg.get("content") if isinstance(msg, dict) else msg.content
            if isinstance(content, list):
                # Multimodal content block: extract text blocks
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return " ".join(t for t in text_parts if t)
            return str(content or "")
    return ""


def _messages_to_litellm(messages: list) -> list[dict]:
    out = []
    role_map = {"human": "user", "ai": "assistant"}
    # Sliding window and message-level truncation to stay strictly within provider TPM limits (e.g. Groq 8k TPM limit)
    effective_msgs = messages
    if len(messages) > 6:
        effective_msgs = [messages[0]] + list(messages[-5:])
    total_effective = len(effective_msgs)
    for idx, msg in enumerate(effective_msgs):
        is_latest = (idx == total_effective - 1)
        max_chars = 1500 if is_latest else 500
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            role = role_map.get(role, role)
            raw_c = str(msg.get("content") or "")
            item = {"role": role, "content": raw_c[:max_chars]}
            if "tool_calls" in msg and msg["tool_calls"]:
                item["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg and msg["tool_call_id"]:
                item["tool_call_id"] = msg["tool_call_id"]
            out.append(item)
        else:
            msg_type = getattr(msg, "type", "user")
            role = role_map.get(msg_type, msg_type)
            raw_c = str(msg.content or "")
            item = {"role": role, "content": raw_c[:max_chars]}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                formatted_tc = []
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        args = tc.get("args", {})
                        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                        formatted_tc.append(
                            {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": args_str,
                                },
                            }
                        )
                    else:
                        formatted_tc.append(tc)
                item["tool_calls"] = formatted_tc
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            out.append(item)
    return out


def _messages_to_litellm_synthesis(messages: list) -> list[dict]:
    """
    Format message history for final synthesis or recovery completion where tools=None.
    Converts raw tool calls and tool outputs into natural language observations so that
    OpenAI/Groq models do not reject the payload or attempt tool calling when tools are disabled.
    """
    out = []
    role_map = {"human": "user", "ai": "assistant"}
    effective_msgs = messages
    if len(messages) > 6:
        effective_msgs = [messages[0]] + list(messages[-5:])
    total_effective = len(effective_msgs)
    for idx, msg in enumerate(effective_msgs):
        is_latest = (idx == total_effective - 1)
        max_chars = 1500 if is_latest else 500
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            role = role_map.get(role, role)
            if role == "tool":
                tool_name = msg.get("name", "tool")
                content = str(msg.get("content") or "")[:800]
                out.append({
                    "role": "user",
                    "content": f"[Observation from {tool_name}]:\n{content}",
                })
            elif role == "assistant":
                content = str(msg.get("content") or "")[:max_chars]
                if not content and msg.get("tool_calls"):
                    names = [
                        tc.get("function", {}).get("name", "tool") if isinstance(tc, dict) else "tool"
                        for tc in msg.get("tool_calls", [])
                    ]
                    content = f"I retrieved information using: {', '.join(names)}."
                out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": role, "content": str(msg.get("content") or "")[:max_chars]})
        else:
            msg_type = getattr(msg, "type", "user")
            role = role_map.get(msg_type, msg_type)
            if role == "tool":
                tool_name = getattr(msg, "name", "tool")
                content = str(msg.content or "")[:800]
                out.append({
                    "role": "user",
                    "content": f"[Observation from {tool_name}]:\n{content}",
                })
            elif role == "assistant":
                content = str(msg.content or "")[:max_chars]
                if not content and hasattr(msg, "tool_calls") and msg.tool_calls:
                    names = [
                        getattr(tc, "name", "tool") if hasattr(tc, "name") else (tc.get("name") if isinstance(tc, dict) else "tool")
                        for tc in msg.tool_calls
                    ]
                    content = f"I retrieved information using: {', '.join(names)}."
                out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": role, "content": str(msg.content or "")[:max_chars]})
    return out


agent_graph = build_graph()


def resume_approval(record: dict, approve: bool) -> dict:
    state = record.get("state")
    if not state:
        action_str = "approved" if approve else "rejected"
        return {
            "session_id": record.get("session_id", ""),
            "messages": [
                {
                    "role": "assistant",
                    "content": f"The requested action has been {action_str} by the human operator.",
                }
            ],
            "stop_reason": action_str,
            "awaiting_approval": False,
            "approval_id": None,
        }

    state = dict(state)
    state["approval_id"] = record.get("approval_id")
    state["awaiting_approval"] = False

    if not approve:
        rejection_results = [
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": "Action rejected by human operator.",
            }
            for call in state.get("pending_tool_calls", [])
        ]
        state["messages"] = list(state.get("messages", [])) + rejection_results
        state["pending_tool_calls"] = []
        state["stop_reason"] = "rejected"
        model_res = call_model(state)
        state["messages"] = list(state["messages"]) + model_res.get("messages", [])
        update_memory(state)
        return state

    tool_res = execute_tools(state)
    state["messages"] = list(state.get("messages", [])) + tool_res.get("messages", [])
    state["evidence_urls"] = tool_res.get("evidence_urls", [])
    state["evidence_items"] = tool_res.get("evidence_items", [])
    state["pending_tool_calls"] = []
    state["stop_reason"] = "approved"
    model_res = call_model(state)
    state["messages"] = list(state["messages"]) + model_res.get("messages", [])
    update_memory(state)
    return state
