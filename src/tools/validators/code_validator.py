"""
Code Execution Output Validator.
Validates sandbox execution: exit status, exceptions, syntax errors, timeouts,
and post-execution correctness verification (assertions, expected format, logic checks).
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

from src.tools.validators.base import (
    BaseToolValidator,
    EvidenceType,
    ValidationResult,
    ValidationStatus,
)

_PYTHON_EXCEPTIONS = [
    "SyntaxError",
    "IndentationError",
    "ZeroDivisionError",
    "NameError",
    "TypeError",
    "ValueError",
    "ImportError",
    "ModuleNotFoundError",
    "AttributeError",
    "IndexError",
    "KeyError",
    "FileNotFoundError",
    "PermissionError",
    "RecursionError",
    "MemoryError",
]


def _detect_collection_divisor_mismatch(code_str: str) -> list[str]:
    """
    Deterministic static check: Detects arithmetic divisor mismatches with collection lengths.
    E.g. marks = [80, 90, 70]; average = sum(marks) / 2
    """
    warnings = []
    try:
        tree = ast.parse(code_str)
        list_lengths: dict[str, int] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.List):
                        list_lengths[target.id] = len(node.value.elts)

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                # Check if left is sum(var)
                if (
                    isinstance(node.left, ast.Call)
                    and isinstance(node.left.func, ast.Name)
                    and node.left.func.id == "sum"
                    and node.left.args
                    and isinstance(node.left.args[0], ast.Name)
                ):
                    var_name = node.left.args[0].id
                    if var_name in list_lengths:
                        actual_len = list_lengths[var_name]
                        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                            divisor = node.right.value
                            if divisor != actual_len:
                                warnings.append(
                                    f"Logic Error: Calculation divides sum({var_name}) by {divisor}, "
                                    f"but '{var_name}' contains {actual_len} elements. Expected division by {actual_len}."
                                )
    except Exception:
        pass
    return warnings


class CodeExecutionValidator(BaseToolValidator):
    tool_type = EvidenceType.CODE_EXECUTION

    def validate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_output: Any,
    ) -> ValidationResult:
        code = str(arguments.get("code", ""))
        language = str(arguments.get("language", "python")).lower()
        raw_text = str(raw_output)

        issues: list[str] = []
        metrics: dict[str, Any] = {
            "language": language,
            "output_length": len(raw_text),
            "exit_code": 0,
        }

        # 1. Timeout Check
        if "timed out" in raw_text.lower():
            issues.append("Execution timed out before completion")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                confidence=0.0,
                issues=issues,
                metrics={"timed_out": True},
                sanitized_content=raw_text,
            )

        # 2. Non-zero exit code check
        exit_match = re.search(r"exited\s+non-zero\s*\((\d+)\)", raw_text, re.IGNORECASE)
        if exit_match:
            exit_code = int(exit_match.group(1))
            metrics["exit_code"] = exit_code
            issues.append(f"Process failed with non-zero exit code {exit_code}")

        # 3. Traceback & Python Exception Check (including AssertionError)
        has_traceback = "Traceback (most recent call last)" in raw_text
        has_assertion_err = "AssertionError" in raw_text or "[HERMES_ASSERTION_FAILED" in raw_text
        detected_exceptions = [exc for exc in _PYTHON_EXCEPTIONS if re.search(rf"\b{exc}\b", raw_text)]
        if has_assertion_err:
            detected_exceptions.append("AssertionError")

        if has_traceback or detected_exceptions:
            for exc in detected_exceptions:
                if exc == "AssertionError":
                    issues.append("Post-execution verification failed: AssertionError in result validation")
                else:
                    issues.append(f"Runtime Exception: {exc}")
            if not detected_exceptions and has_traceback:
                issues.append("Uncaught Python Traceback detected in output")

            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                confidence=0.0,
                issues=issues,
                metrics=metrics,
                sanitized_content=raw_text,
            )

        # 4. Post-Execution Correctness Checks
        # A. Empty output check
        if not raw_text.strip() or raw_text.strip() == "[Command completed with no output]":
            issues.append("Code executed successfully but produced no stdout/stderr output")
            return ValidationResult(
                status=ValidationStatus.FLAGGED,
                is_valid=True,
                confidence=0.6,
                issues=issues,
                metrics=metrics,
                sanitized_content="[Executed with no output]",
            )

        # B. Post-Execution check markers from sandbox
        if "[Post-Execution Correctness Check FAILED:" in raw_text:
            m = re.search(r"\[Post-Execution Correctness Check FAILED:\s*(.*?)\]", raw_text)
            fail_msg = m.group(1) if m else "Correctness check failed"
            issues.append(f"Correctness validation failure: {fail_msg}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                confidence=0.0,
                issues=issues,
                metrics=metrics,
                sanitized_content=raw_text,
            )

        # C. Format Check
        expected_format = arguments.get("expected_format")
        if expected_format:
            fmt = str(expected_format).lower().strip()
            trimmed = raw_text.strip()
            if fmt in ("number", "float", "int", "numeric"):
                try:
                    float(trimmed)
                except ValueError:
                    issues.append(f"Output format violation: expected numeric output, received '{trimmed[:100]}'")
            elif fmt == "json":
                try:
                    json.loads(trimmed)
                except Exception:
                    issues.append(f"Output format violation: expected valid JSON, received '{trimmed[:100]}'")
            elif fmt == "list":
                try:
                    val = ast.literal_eval(trimmed)
                    if not isinstance(val, list):
                        issues.append(f"Output format violation: expected list, received {type(val).__name__}")
                except Exception:
                    issues.append(f"Output format violation: expected list syntax, received '{trimmed[:100]}'")

        # D. Expected Output Check
        expected_output = arguments.get("expected_output")
        if expected_output is not None:
            expected_str = str(expected_output).strip()
            if raw_text.strip() != expected_str and expected_str not in raw_text:
                issues.append(f"Output mismatch: received '{raw_text.strip()}', expected '{expected_str}'")

        # E. Static Logic Bug Detection (e.g. sum(marks)/2 when marks has 3 elements)
        logic_warnings = _detect_collection_divisor_mismatch(code)
        if logic_warnings:
            issues.extend(logic_warnings)

        if any("Logic Error:" in iss or "Output format violation:" in iss or "Output mismatch:" in iss for iss in issues):
            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                confidence=0.1,
                issues=issues,
                metrics=metrics,
                sanitized_content=raw_text,
            )

        # 5. Warning or non-zero exit check
        if issues:
            return ValidationResult(
                status=ValidationStatus.FLAGGED,
                is_valid=False,
                confidence=0.3,
                issues=issues,
                metrics=metrics,
                sanitized_content=raw_text,
            )

        return ValidationResult(
            status=ValidationStatus.VERIFIED,
            is_valid=True,
            confidence=0.98,
            issues=[],
            metrics=metrics,
            sanitized_content=raw_text.strip(),
        )
