"""
Web Search and Web Reader Output Validators.
Validates source quality, relevance, freshness, and absence of errors.
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.tools.validators.base import (
    BaseToolValidator,
    EvidenceType,
    ValidationResult,
    ValidationStatus,
)

_URL_PATTERN = re.compile(r"^https?://[^\s\"'<>]+$")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "in", "for", "on", "with", "at", "by", "from", "to", "and", "or",
    "what", "how", "many", "give", "me", "tell", "find", "search", "show", "can", "you", "does", "do",
}


class WebSearchValidator(BaseToolValidator):
    tool_type = EvidenceType.WEB_SEARCH

    def validate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_output: Any,
    ) -> ValidationResult:
        query = str(arguments.get("query", "")).strip()
        issues: list[str] = []
        metrics: dict[str, Any] = {"result_count": 0, "relevance_score": 0.0}

        # Parse raw_output into items list
        items: list[dict[str, Any]] = []
        raw_text = str(raw_output)

        if isinstance(raw_output, dict):
            if not raw_output.get("success", True):
                err = raw_output.get("error", {})
                msg = err.get("message", "Search execution failed")
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    is_valid=False,
                    confidence=0.0,
                    issues=[f"Tool error: {msg}"],
                    sanitized_content=raw_text,
                )
            items = raw_output.get("data", [])
        elif isinstance(raw_output, list):
            items = [item for item in raw_output if isinstance(item, dict)]
        else:
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, list):
                    items = parsed
                elif isinstance(parsed, dict) and "data" in parsed:
                    items = parsed["data"]
            except Exception:
                pass

        # Check for explicit empty result markers
        if not items or "no web search results found" in raw_text.lower() or "no verified search results" in raw_text.lower():
            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                confidence=0.0,
                issues=[f"No search results returned for query '{query}'"],
                metrics={"result_count": 0, "relevance_score": 0.0},
                sanitized_content=raw_text,
            )

        metrics["result_count"] = len(items)

        # 1. Relevance scoring
        query_words = set(re.findall(r"\w+", query.lower())) - _STOPWORDS
        matched_results = 0
        valid_urls: list[str] = []

        for item in items:
            title = str(item.get("title", "")).lower()
            snippet = str(item.get("snippet", "") or item.get("body", "")).lower()
            url = str(item.get("url", "") or item.get("href", "")).strip()

            if url and _URL_PATTERN.match(url):
                valid_urls.append(url)

            if query_words:
                text_combined = f"{title} {snippet}"
                overlap = sum(1 for w in query_words if w in text_combined)
                if overlap > 0:
                    matched_results += 1

        relevance = matched_results / max(len(items), 1)
        metrics["relevance_score"] = round(relevance, 2)
        metrics["valid_urls_count"] = len(valid_urls)

        # 2. Source quality check
        if len(valid_urls) < len(items):
            issues.append(f"{len(items) - len(valid_urls)} search item(s) lacked valid HTTP/HTTPS URLs")

        if relevance < 0.3:
            issues.append("Low relevance: search snippets do not strongly match query terms")
            return ValidationResult(
                status=ValidationStatus.FLAGGED,
                is_valid=True,
                confidence=0.5,
                issues=issues,
                metrics=metrics,
                sanitized_content=raw_text,
            )

        # 3. Freshness check: detect year references
        has_recent_year = any(y in raw_text for y in ("2024", "2025", "2026", "2027"))
        metrics["has_recency_markers"] = has_recent_year

        return ValidationResult(
            status=ValidationStatus.VERIFIED,
            is_valid=True,
            confidence=round(0.7 + (0.3 * relevance), 2),
            issues=issues,
            metrics=metrics,
            sanitized_content=raw_text,
        )


class WebReaderValidator(BaseToolValidator):
    tool_type = EvidenceType.WEB_READER

    def validate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_output: Any,
    ) -> ValidationResult:
        url = str(arguments.get("url", "")).strip()
        raw_text = str(raw_output)
        issues: list[str] = []
        metrics: dict[str, Any] = {"content_length": len(raw_text)}

        # Check for error outputs
        error_indicators = (
            "failed to fetch",
            "404 not found",
            "403 forbidden",
            "access denied",
            "connection refused",
            "request timed out",
            "[webpage returned empty text content]",
        )
        lower_text = raw_text.lower()
        for ind in error_indicators:
            if ind in lower_text and len(raw_text) < 300:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    is_valid=False,
                    confidence=0.0,
                    issues=[f"Webpage fetch failed: {ind}"],
                    metrics=metrics,
                    sanitized_content=raw_text,
                )

        if len(raw_text.strip()) < 50:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                confidence=0.2,
                issues=["Extracted webpage text is excessively short or empty"],
                metrics=metrics,
                sanitized_content=raw_text,
            )

        return ValidationResult(
            status=ValidationStatus.VERIFIED,
            is_valid=True,
            confidence=0.95,
            issues=issues,
            metrics=metrics,
            sanitized_content=raw_text,
        )

