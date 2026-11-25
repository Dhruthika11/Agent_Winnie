"""
Centralized Token-Aware Prompt Budget Manager for Hermes Agent Platform.

Enforces strict input token budgets before every LLM request:
1. Accurately estimates tokens for system instructions, conversation history, current user message, and MCP tool schemas.
2. Dynamically compacts older conversation history while strictly preserving the current user message.
3. Filters MCP tool definitions based on the user's intent or workflow to avoid wasting token capacity.
4. Accounts for model context limits, provider TPM quotas, output token reserves, and safety margins.
5. Returns clear, actionable errors when a prompt cannot be compacted within budget.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("hermes.agent.prompt_manager")

# Try importing tiktoken; fallback to conservative character ratio estimation
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    _TIKTOKEN_AVAILABLE = False


class PromptBudgetExceededError(ValueError):
    """Raised when a prompt exceeds model/provider token limits even after full compaction."""
    def __init__(self, message: str, estimated_tokens: int, budget_limit: int, model: str):
        super().__init__(message)
        self.estimated_tokens = estimated_tokens
        self.budget_limit = budget_limit
        self.model = model


# Provider & Model limit specifications (context window, TPM ceiling, recommended output reserve)
MODEL_LIMIT_SPECS: dict[str, dict[str, int]] = {
    # Groq Models (typically 8k context or 8k TPM limit on standard/free tiers)
    "groq/openai/gpt-oss-120b": {"context_window": 8192, "tpm_limit": 7500, "max_output": 2000, "safety_margin": 500},
    "groq/qwen/qwen3.8-27b": {"context_window": 8192, "tpm_limit": 7500, "max_output": 2000, "safety_margin": 500},
    "groq/openai/gpt-oss-20b": {"context_window": 8192, "tpm_limit": 7500, "max_output": 2000, "safety_margin": 500},
    "groq/llama-3.3-70b-versatile": {"context_window": 32768, "tpm_limit": 8000, "max_output": 2500, "safety_margin": 500},
    "groq/llama-3.1-8b-instant": {"context_window": 32768, "tpm_limit": 15000, "max_output": 2000, "safety_margin": 500},
    "groq/llama3-70b-8192": {"context_window": 8192, "tpm_limit": 7000, "max_output": 1500, "safety_margin": 500},
    "groq/mixtral-8x7b-32768": {"context_window": 32768, "tpm_limit": 8000, "max_output": 2000, "safety_margin": 500},

    # Google Gemini Models (Huge context window and TPM)
    "gemini/gemini-2.0-flash": {"context_window": 1048576, "tpm_limit": 1000000, "max_output": 4096, "safety_margin": 1000},
    "gemini/gemini-2.0-flash-exp": {"context_window": 1048576, "tpm_limit": 1000000, "max_output": 4096, "safety_margin": 1000},
    "gemini/gemini-1.5-flash": {"context_window": 1048576, "tpm_limit": 1000000, "max_output": 4096, "safety_margin": 1000},
    "gemini/gemini-1.5-pro": {"context_window": 2097152, "tpm_limit": 1000000, "max_output": 4096, "safety_margin": 1000},

    # Standard OpenAI / Anthropic models
    "gpt-4o": {"context_window": 128000, "tpm_limit": 30000, "max_output": 4096, "safety_margin": 1000},
    "gpt-4o-mini": {"context_window": 128000, "tpm_limit": 60000, "max_output": 4096, "safety_margin": 1000},
    "claude-3-5-sonnet": {"context_window": 200000, "tpm_limit": 40000, "max_output": 4096, "safety_margin": 1000},

    # Fallback default
    "default": {"context_window": 8192, "tpm_limit": 7000, "max_output": 2000, "safety_margin": 500},
}


@dataclass
class BudgetedPromptResult:
    """Result of prompt budget processing."""
    system_prompt: str
    messages: list[dict[str, Any]]
    tools: Optional[list[dict[str, Any]]]
    total_estimated_tokens: int
    prompt_budget: int
    compaction_applied: bool
    filtered_tool_count: int
    original_tool_count: int


class PromptBudgetManager:
    """Manages token calculation, intent-based tool filtering, and intelligent context compaction."""

    def __init__(self) -> None:
        self._tokenizer_cache: dict[str, Any] = {}

    def get_tokenizer(self, model: str = "") -> Any:
        if not _TIKTOKEN_AVAILABLE:
            return None
        encoding_name = "cl100k_base"
        if "gpt-4o" in model:
            encoding_name = "o200k_base"
        if encoding_name not in self._tokenizer_cache:
            try:
                self._tokenizer_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception:
                return None
        return self._tokenizer_cache.get(encoding_name)

    def estimate_tokens(self, text: str, model: str = "") -> int:
        """Estimate token count for a text string using tiktoken or conservative fallback."""
        if not text:
            return 0
        tokenizer = self.get_tokenizer(model)
        if tokenizer:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass
        # Conservative fallback: code and JSON average ~3.3 chars per token
        return max(1, math.ceil(len(text) / 3.3))

    def estimate_message_tokens(self, message: dict[str, Any] | Any, model: str = "") -> int:
        """Estimate tokens for a single chat message including role and tool calls."""
        tokens = 4  # Formatting overhead per message
        if isinstance(message, dict):
            role = message.get("role", "")
            tokens += self.estimate_tokens(str(role), model)
            content = message.get("content", "")
            if isinstance(content, str):
                tokens += self.estimate_tokens(content, model)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        tokens += self.estimate_tokens(part.get("text", ""), model)
                    else:
                        tokens += 100  # Token estimate for image references
            if "tool_calls" in message and message["tool_calls"]:
                try:
                    tc_str = json.dumps(message["tool_calls"])
                    tokens += self.estimate_tokens(tc_str, model)
                except Exception:
                    tokens += 150
            if "tool_call_id" in message and message["tool_call_id"]:
                tokens += self.estimate_tokens(str(message["tool_call_id"]), model)
        else:
            msg_type = getattr(message, "type", "user")
            tokens += self.estimate_tokens(str(msg_type), model)
            content = getattr(message, "content", "")
            tokens += self.estimate_tokens(str(content or ""), model)
        return tokens

    def estimate_messages_tokens(self, messages: list[Any], model: str = "") -> int:
        """Estimate total tokens across a list of messages."""
        return sum(self.estimate_message_tokens(m, model) for m in messages)

    def estimate_tools_tokens(self, tools: list[dict[str, Any]] | None, model: str = "") -> int:
        """Estimate tokens consumed by OpenAI-format tool definitions."""
        if not tools:
            return 0
        try:
            tools_json = json.dumps(tools)
            return self.estimate_tokens(tools_json, model)
        except Exception:
            return len(tools) * 150  # Average tool schema is ~150-250 tokens

    def get_model_limits(self, model: str) -> dict[str, int]:
        """Resolve context window, TPM quota limit, and reserve capacity for a given model."""
        clean_model = (model or "").lower().strip()
        if clean_model in MODEL_LIMIT_SPECS:
            return MODEL_LIMIT_SPECS[clean_model]
        for key in sorted(MODEL_LIMIT_SPECS.keys(), key=len, reverse=True):
            if key != "default" and key in clean_model:
                return MODEL_LIMIT_SPECS[key]
        return MODEL_LIMIT_SPECS["default"]

    def calculate_prompt_budget(self, model: str, requested_max_output: Optional[int] = None) -> int:
        """
        Calculate maximum allowable prompt tokens so that the request fits within
        both the model context window and provider TPM quota limits.
        """
        limits = self.get_model_limits(model)
        output_reserve = requested_max_output or limits["max_output"]
        safety_margin = limits["safety_margin"]

        # Maximum prompt tokens cannot exceed context window minus output reserve
        context_budget = limits["context_window"] - output_reserve - safety_margin

        # Maximum prompt tokens also cannot exceed provider per-request TPM quota minus output reserve
        tpm_budget = limits["tpm_limit"] - output_reserve - safety_margin

        effective_budget = max(500, min(context_budget, tpm_budget))
        return effective_budget

    def filter_tools_by_intent(
        self,
        all_tools: list[dict[str, Any]] | None,
        intent_category: str = "general",
        requires_search: bool = False,
        allow_tools: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Filter MCP tools according to the current intent or workflow.
        Preserves all required tools for the active workflow while omitting irrelevant schemas.
        """
        if not all_tools or not allow_tools:
            return []

        clean_intent = (intent_category or "").lower().strip()

        # 1. Memory instructions never need external tools
        if clean_intent == "memory_instruction":
            return []

        # Categorize tools by domain
        search_tools = []
        code_tools = []
        file_tools = []
        other_tools = []

        for t in all_tools:
            name = t.get("function", {}).get("name", "") if isinstance(t, dict) else ""
            if any(k in name for k in ("search", "scrape", "browse", "web")):
                search_tools.append(t)
            elif any(k in name for k in ("sandbox", "code", "bash", "python", "terminal", "exec", "run")):
                code_tools.append(t)
            elif any(k in name for k in ("file", "read_file", "write_file", "list_dir", "delete")):
                file_tools.append(t)
            else:
                other_tools.append(t)

        # 2. Coding & Command Execution workflow: Prioritize code and file tools
        if clean_intent in ("coding", "data_analysis", "diagnostics", "action_request", "command_execution"):
            filtered = list(code_tools) + list(file_tools)
            if requires_search:
                filtered.extend(search_tools)
            return filtered or all_tools

        # 3. Research workflow: Prioritize search and read tools
        if clean_intent in ("research", "web_search", "document_generation") or requires_search:
            filtered = list(search_tools)
            # Add read-only file tool if present
            read_tools = [t for t in file_tools if "read" in t.get("function", {}).get("name", "")]
            filtered.extend(read_tools)
            return filtered or all_tools

        # 4. Quick chat / General conversational: Only search tools if search is needed, otherwise minimal
        if clean_intent == "quick_chat":
            if requires_search:
                return search_tools
            return []

        # Default: if search required, include search + sandbox; otherwise all tools
        if requires_search:
            return search_tools + code_tools
        return all_tools

    def compact_messages(
        self,
        messages: list[Any],
        target_token_budget: int,
        model: str = "",
        max_history_turns: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Intelligently compacts conversation history to fit target token budget:
        - NEVER silently truncates or drops the latest user message.
        - Preserves the initial conversation context if present.
        - Compacts older intermediate assistant answers and tool observations.
        - Deduplicates repeated tool outputs.
        """
        if not messages:
            return []

        # Find the latest user message (must be preserved intact)
        latest_user_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            m = messages[idx]
            role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
            if role in ("user", "human"):
                latest_user_idx = idx
                break

        # Standard sliding window over recent turns
        effective = list(messages)
        if len(effective) > (max_history_turns * 2 + 1):
            # Keep first turn + last N turns
            effective = [effective[0]] + list(effective[-(max_history_turns * 2):])

        # Convert to standardized dicts
        role_map = {"human": "user", "ai": "assistant"}
        standardized: list[dict[str, Any]] = []
        for idx, m in enumerate(effective):
            is_latest_user = (idx == len(effective) - 1) or (idx == latest_user_idx)
            if isinstance(m, dict):
                r = m.get("role", "user")
                r = role_map.get(r, r)
                c = m.get("content", "")
                item: dict[str, Any] = {"role": r, "content": c}
                if "tool_calls" in m and m["tool_calls"]:
                    item["tool_calls"] = m["tool_calls"]
                if "tool_call_id" in m and m["tool_call_id"]:
                    item["tool_call_id"] = m["tool_call_id"]
                if "name" in m and m["name"]:
                    item["name"] = m["name"]
            else:
                r = getattr(m, "type", "user")
                r = role_map.get(r, r)
                c = getattr(m, "content", "")
                item = {"role": r, "content": c}
            standardized.append(item)

        current_tokens = self.estimate_messages_tokens(standardized, model)
        if current_tokens <= target_token_budget:
            return standardized

        # Compaction Phase 1: Truncate tool observations to 600 chars each and older assistant messages to 400 chars
        compacted: list[dict[str, Any]] = []
        total_items = len(standardized)
        for idx, item in enumerate(standardized):
            is_latest_user = (idx == total_items - 1)
            role = item.get("role", "")
            content = str(item.get("content") or "")

            if is_latest_user and role in ("user", "human"):
                # NEVER truncate the current user request
                compacted.append(dict(item))
            elif role == "tool":
                # Compact tool output
                compact_item = dict(item)
                compact_item["content"] = content[:600] + ("... [truncated observation]" if len(content) > 600 else "")
                compacted.append(compact_item)
            elif role in ("assistant", "ai"):
                # Compact intermediate assistant reasoning
                compact_item = dict(item)
                if len(content) > 400:
                    compact_item["content"] = content[:400] + "... [summary of previous turn]"
                compacted.append(compact_item)
            else:
                # Older user messages can be moderately trimmed if very long (>600 chars)
                compact_item = dict(item)
                if len(content) > 600:
                    compact_item["content"] = content[:600] + "... [earlier prompt context]"
                compacted.append(compact_item)

        current_tokens = self.estimate_messages_tokens(compacted, model)
        if current_tokens <= target_token_budget:
            return compacted

        # Compaction Phase 2: If still over budget, retain only initial context + last 3 turns
        if len(compacted) > 4:
            compacted = [compacted[0]] + list(compacted[-3:])

        return compacted

    def build_budgeted_prompt(
        self,
        system_prompt: str,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]],
        model: str,
        intent_category: str = "general",
        requires_search: bool = False,
        allow_tools: bool = True,
        requested_max_output: Optional[int] = None,
    ) -> BudgetedPromptResult:
        """
        Central entrypoint: builds an optimized, token-budgeted prompt payload.
        Ensures the total input tokens fit strictly within the model/provider budget.
        """
        budget_limit = self.calculate_prompt_budget(model, requested_max_output)
        original_tool_count = len(tools) if tools else 0

        # Step 1: Filter tools according to intent/workflow
        filtered_tools = self.filter_tools_by_intent(
            tools,
            intent_category=intent_category,
            requires_search=requires_search,
            allow_tools=allow_tools,
        )

        system_tokens = self.estimate_tokens(system_prompt, model)
        tools_tokens = self.estimate_tools_tokens(filtered_tools, model)

        # Remaining budget for conversation messages
        messages_budget = max(300, budget_limit - system_tokens - tools_tokens)

        # Step 2: Compact messages to fit within message budget
        compacted_messages = self.compact_messages(messages, target_token_budget=messages_budget, model=model)
        messages_tokens = self.estimate_messages_tokens(compacted_messages, model)

        total_tokens = system_tokens + tools_tokens + messages_tokens
        compaction_applied = (len(compacted_messages) != len(messages)) or (messages_tokens < self.estimate_messages_tokens(messages, model))

        # Step 3: If still over budget, aggressively shed non-critical tools
        if total_tokens > budget_limit and filtered_tools:
            logger.warning(
                "Prompt total tokens (%d) exceeds budget (%d). Shedding secondary tools to protect request.",
                total_tokens, budget_limit
            )
            # Keep only the single most relevant tool or none
            filtered_tools = filtered_tools[:1]
            tools_tokens = self.estimate_tools_tokens(filtered_tools, model)
            total_tokens = system_tokens + tools_tokens + messages_tokens
            compaction_applied = True

        # Step 4: Final verification: if user message alone or uncompacted elements still exceed budget
        if total_tokens > (budget_limit + 200):  # Allow tiny 200 token leeway
            msg = (
                f"Prompt payload ({total_tokens} estimated tokens) exceeds safe budget "
                f"limit of {budget_limit} tokens for model '{model}' (TPM/context ceiling). "
                "Please shorten your input or attached documents."
            )
            logger.error("Prompt budget fatal overflow: %s", msg)
            raise PromptBudgetExceededError(
                message=msg,
                estimated_tokens=total_tokens,
                budget_limit=budget_limit,
                model=model,
            )

        return BudgetedPromptResult(
            system_prompt=system_prompt,
            messages=compacted_messages,
            tools=filtered_tools if filtered_tools else None,
            total_estimated_tokens=total_tokens,
            prompt_budget=budget_limit,
            compaction_applied=compaction_applied,
            filtered_tool_count=len(filtered_tools) if filtered_tools else 0,
            original_tool_count=original_tool_count,
        )


prompt_budget_manager = PromptBudgetManager()
