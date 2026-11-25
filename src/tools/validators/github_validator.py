"""
GitHub API Output Validator.
Validates repository schemas, API rate limits, and object integrity.
"""
from __future__ import annotations

import json
from typing import Any

from src.tools.validators.base import (
    BaseToolValidator,
    EvidenceType,
    ValidationResult,
    ValidationStatus,
)


class GitHubValidator(BaseToolValidator):
    tool_type = EvidenceType.GITHUB

    def validate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_output: Any,
    ) -> ValidationResult:
        raw_text = str(raw_output)
        issues: list[str] = []
        metrics: dict[str, Any] = {}

        # 1. Check for API rate limiting or not found
        lower_text = raw_text.lower()
        if "rate limit" in lower_text or "api rate limit exceeded" in lower_text:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                confidence=0.0,
                issues=["GitHub API rate limit exceeded"],
                sanitized_content=raw_text,
            )
        if "not found" in lower_text and len(raw_text) < 200:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                confidence=0.0,
                issues=["GitHub repository or user not found"],
                sanitized_content=raw_text,
            )

        # 2. Check schema for structured repository returns
        parsed = None
        if isinstance(raw_output, (dict, list)):
            parsed = raw_output
        else:
            try:
                parsed = json.loads(raw_text)
            except Exception:
                pass

        if isinstance(parsed, dict):
            if "name" in parsed or "full_name" in parsed:
                metrics["repo_name"] = parsed.get("full_name") or parsed.get("name")
                metrics["stars"] = parsed.get("stargazers_count", 0)
                metrics["has_description"] = bool(parsed.get("description"))
                return ValidationResult(
                    status=ValidationStatus.VERIFIED,
                    is_valid=True,
                    confidence=0.98,
                    issues=[],
                    metrics=metrics,
                    sanitized_content=raw_text,
                )
        elif isinstance(parsed, list):
            metrics["count"] = len(parsed)
            if not parsed:
                return ValidationResult(
                    status=ValidationStatus.FLAGGED,
                    is_valid=True,
                    confidence=0.5,
                    issues=["GitHub query returned 0 items"],
                    metrics=metrics,
                    sanitized_content="[]",
                )
            return ValidationResult(
                status=ValidationStatus.VERIFIED,
                is_valid=True,
                confidence=0.95,
                issues=[],
                metrics=metrics,
                sanitized_content=raw_text,
            )

        return ValidationResult(
            status=ValidationStatus.VERIFIED,
            is_valid=True,
            confidence=0.9,
            issues=issues,
            metrics=metrics,
            sanitized_content=raw_text,
        )

