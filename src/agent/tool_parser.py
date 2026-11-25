"""
Tool call parsing utilities for LLMs (especially Qwen 2.5, DeepSeek, and open-weight models)
that may emit tool calls as XML or raw JSON blocks inside message content rather than standard
OpenAI delta.tool_calls structures.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger("hermes.agent.tool_parser")


def _normalize_tool_params(fn_name: str, params: dict[str, Any]) -> dict[str, Any]:
    norm = dict(params)
    if fn_name == "github.get_repository":
        if "name" in norm and "repo" not in norm:
            norm["repo"] = norm.pop("name")
        if "url" in norm and "repo" not in norm:
            norm["repo"] = norm.pop("url")
    elif fn_name == "github.list_user_repositories":
        if "user" in norm and "username" not in norm:
            norm["username"] = norm.pop("user")
        elif "name" in norm and "username" not in norm:
            norm["username"] = norm.pop("name")
    elif fn_name == "web_scraper.search":
        if "q" in norm and "query" not in norm:
            norm["query"] = norm.pop("q")
    return norm


def is_raw_tool_call(text: str) -> bool:
    """Check whether a text string contains solely a raw tool call or XML function block."""
    if not text:
        return False
    stripped = text.strip()
    return (
        (stripped.startswith("<tool_call>") and stripped.endswith("</tool_call>"))
        or (stripped.startswith("<function=") and stripped.endswith("</function>"))
    )


def validate_and_normalize_call(
    fn_name: str,
    params: dict[str, Any],
    allowed_tools: set[str] | None = None,
) -> dict[str, Any] | None:
    """
    Validate tool call against allowed tool registry and normalize parameter synonyms.
    Returns standardized tool call dict or None if invalid.
    """
    clean_name = str(fn_name).split("<|")[0].strip()
    if allowed_tools is not None and clean_name not in allowed_tools:
        logger.warning(f"Rejected unknown tool name in model output: '{clean_name}'")
        return None

    if not isinstance(params, dict):
        params = {}

    clean_args = _normalize_tool_params(clean_name, params)
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "name": clean_name,
        "arguments": clean_args,
    }


def parse_model_tool_calls(
    text: str,
    allowed_tools: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Extract XML or JSON-formatted tool calls from model output text.
    Validates function names and normalizes parameters.
    Returns:
        (cleaned_text, list_of_tool_calls)
    where each call is:
        {"id": "...", "name": "...", "arguments": {...}}
    """
    if not text:
        return "", []

    if "<tool_call>" not in text and "<function=" not in text:
        return text, []

    calls: list[dict[str, Any]] = []
    cleaned = text

    # 1. Match <tool_call>...</tool_call> blocks
    for m in re.finditer(r"<tool_call>([\s\S]*?)</tool_call>", text):
        block = m.group(1).strip()
        cleaned = cleaned.replace(m.group(0), "")

        # 1a. JSON inside <tool_call>
        if block.startswith("{") and block.endswith("}"):
            try:
                data = json.loads(block)
                name = data.get("name") or data.get("function", "")
                args = data.get("arguments") or data.get("parameters", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                if name:
                    call = validate_and_normalize_call(name, args if isinstance(args, dict) else {}, allowed_tools)
                    if call:
                        calls.append(call)
                    continue
            except Exception:
                pass

        # 1b. XML <function=...> inside <tool_call>
        fn_match = re.search(r"<function=([a-zA-Z0-9_\-\.]+)>([\s\S]*?)</function>", block)
        if fn_match:
            fn_name = fn_match.group(1)
            params_raw = fn_match.group(2)
            params: dict[str, Any] = {}
            for param_match in re.finditer(r"<parameter=([a-zA-Z0-9_\-]+)>([\s\S]*?)</parameter>", params_raw):
                p_key = param_match.group(1).strip()
                p_val = param_match.group(2).strip()
                try:
                    p_val = json.loads(p_val)
                except Exception:
                    pass
                params[p_key] = p_val

            call = validate_and_normalize_call(fn_name, params, allowed_tools)
            if call:
                calls.append(call)

    # 2. Standalone <function=...> without <tool_call> wrapper
    if not calls and "<function=" in text:
        for fn_match in re.finditer(r"<function=([a-zA-Z0-9_\-\.]+)>([\s\S]*?)</function>", text):
            cleaned = cleaned.replace(fn_match.group(0), "")
            fn_name = fn_match.group(1)
            params_raw = fn_match.group(2)
            params = {}
            for param_match in re.finditer(r"<parameter=([a-zA-Z0-9_\-]+)>([\s\S]*?)</parameter>", params_raw):
                p_key = param_match.group(1).strip()
                p_val = param_match.group(2).strip()
                try:
                    p_val = json.loads(p_val)
                except Exception:
                    pass
                params[p_key] = p_val

            call = validate_and_normalize_call(fn_name, params, allowed_tools)
            if call:
                calls.append(call)

    cleaned = strip_unwanted_tool_xml(cleaned)
    return cleaned, calls


def strip_unwanted_tool_xml(text: str) -> str:
    """Strip any residual tool XML tags or parameter fragments from response text without stripping leading/trailing whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text)
    cleaned = re.sub(r"<function=[^>]+>[\s\S]*?</function>", "", cleaned)
    cleaned = re.sub(r"</?(?:tool_call|function|parameter)[^>]*>", "", cleaned)
    return cleaned
