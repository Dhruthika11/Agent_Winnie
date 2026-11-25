"""
Tool Validator Registry and Output Validation Dispatcher.
Validates outputs close to the tools that produce them and normalizes results into EvidenceItems.
"""
from __future__ import annotations

import fnmatch
import logging
import re
from typing import Any

from src.tools.validators.base import (
    BaseToolValidator,
    EvidenceItem,
    EvidenceType,
    ValidationResult,
    ValidationStatus,
)
from src.tools.validators.code_validator import CodeExecutionValidator
from src.tools.validators.filesystem_validator import FilesystemValidator
from src.tools.validators.github_validator import GitHubValidator
from src.tools.validators.math_validator import MathValidator
from src.tools.validators.terminal_validator import TerminalValidator
from src.tools.validators.web_validator import WebReaderValidator, WebSearchValidator

logger = logging.getLogger("hermes.tools.validators")


class GenericToolValidator(BaseToolValidator):
    tool_type = EvidenceType.GENERIC

    def validate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_output: Any,
    ) -> ValidationResult:
        raw_text = str(raw_output)
        is_error = "error" in raw_text.lower() and len(raw_text) < 300
        return ValidationResult(
            status=ValidationStatus.FAILED if is_error else ValidationStatus.VERIFIED,
            is_valid=not is_error,
            confidence=0.5 if is_error else 0.9,
            issues=["Generic execution error detected"] if is_error else [],
            sanitized_content=raw_text,
        )


_WEB_SEARCH_VALIDATOR = WebSearchValidator()
_WEB_READER_VALIDATOR = WebReaderValidator()
_CODE_VALIDATOR = CodeExecutionValidator()
_MATH_VALIDATOR = MathValidator()
_FILESYSTEM_VALIDATOR = FilesystemValidator()
_TERMINAL_VALIDATOR = TerminalValidator()
_GITHUB_VALIDATOR = GitHubValidator()
_GENERIC_VALIDATOR = GenericToolValidator()


def _is_math_code(code: str) -> bool:
    """Detect if code snippet is purely an arithmetic / mathematical computation."""
    clean = code.strip()
    # e.g., print(4839 * 7213) or 99 ** 3
    math_pattern = r"^(?:print\s*\(\s*)?[-+*/%0-9\s().^**]+(?:\s*\))?$"
    return bool(re.match(math_pattern, clean))


def get_validator_for_tool(tool_name: str, arguments: dict[str, Any]) -> BaseToolValidator:
    """Select the appropriate validator close to the tool producing the output."""
    t_name = tool_name.lower()

    if "web_scraper.search" in t_name or "search" in t_name:
        return _WEB_SEARCH_VALIDATOR
    elif "web_scraper.read_url" in t_name or "read_url" in t_name:
        return _WEB_READER_VALIDATOR
    elif "run_sandboxed_code" in t_name or "sandbox" in t_name:
        code_str = str(arguments.get("code", ""))
        if _is_math_code(code_str):
            return _MATH_VALIDATOR
        return _CODE_VALIDATOR
    elif "filesystem" in t_name:
        return _FILESYSTEM_VALIDATOR
    elif "terminal" in t_name:
        return _TERMINAL_VALIDATOR
    elif "github" in t_name:
        return _GITHUB_VALIDATOR

    return _GENERIC_VALIDATOR


def validate_tool_output(
    tool_name: str,
    arguments: dict[str, Any],
    raw_output: Any,
) -> tuple[ValidationResult, EvidenceItem]:
    """
    Validate a tool's output using its dedicated validator and generate a
    normalized EvidenceItem for the central evidence store.
    """
    clean_args = arguments if isinstance(arguments, dict) else {}
    validator = get_validator_for_tool(tool_name, clean_args)
    
    try:
        val_res = validator.validate(tool_name, clean_args, raw_output)
    except Exception as exc:
        logger.exception("validator_execution_error", extra={"tool": tool_name})
        val_res = ValidationResult(
            status=ValidationStatus.FLAGGED,
            is_valid=False,
            confidence=0.3,
            issues=[f"Validator exception: {exc}"],
            sanitized_content=str(raw_output),
        )

    evidence_item = validator.create_evidence_item(tool_name, clean_args, raw_output, val_res)
    logger.info(
        "tool_output_validated",
        extra={
            "tool": tool_name,
            "status": val_res.status.value,
            "is_valid": val_res.is_valid,
            "confidence": val_res.confidence,
        },
    )
    return val_res, evidence_item


__all__ = [
    "ValidationStatus",
    "EvidenceType",
    "EvidenceItem",
    "ValidationResult",
    "BaseToolValidator",
    "WebSearchValidator",
    "WebReaderValidator",
    "CodeExecutionValidator",
    "MathValidator",
    "FilesystemValidator",
    "TerminalValidator",
    "GitHubValidator",
    "validate_tool_output",
    "get_validator_for_tool",
]

