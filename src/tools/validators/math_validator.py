"""
Math and Calculator Output Validator.
Safely recomputes arithmetic expressions, verifies precision, and checks for NaN/Infinity.
"""
from __future__ import annotations

import ast
import operator
import re
from typing import Any

from src.tools.validators.base import (
    BaseToolValidator,
    EvidenceType,
    ValidationResult,
    ValidationStatus,
)

# Supported safe arithmetic operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node: ast.AST) -> float | int:
    """Safely evaluate an arithmetic AST expression without executing arbitrary code."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("division by zero")
            return _OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    raise ValueError(f"Unsupported AST node in arithmetic: {type(node)}")


class MathValidator(BaseToolValidator):
    tool_type = EvidenceType.MATH

    def validate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_output: Any,
    ) -> ValidationResult:
        raw_text = str(raw_output).strip()
        expr = str(arguments.get("expression") or arguments.get("code") or "")
        issues: list[str] = []
        metrics: dict[str, Any] = {}

        # 1. Check for NaN / Infinity
        if any(marker in raw_text.lower() for marker in ("nan", "inf", "-inf", "infinity")):
            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                confidence=0.0,
                issues=["Calculation resulted in NaN or Infinity"],
                metrics={"is_finite": False},
                sanitized_content=raw_text,
            )

        # 2. Extract potential arithmetic expression if embedded in code (e.g. "print(4839 * 7213)")
        clean_expr = expr
        print_match = re.search(r"print\((.*?)\)", expr)
        if print_match:
            clean_expr = print_match.group(1).strip()

        # Try to parse and recalculate expression
        recalculated = None
        if clean_expr:
            try:
                tree = ast.parse(clean_expr, mode="eval")
                recalculated = _eval_ast(tree)
                metrics["recalculated"] = recalculated
            except ZeroDivisionError:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    is_valid=False,
                    confidence=0.0,
                    issues=["Division by zero detected in expression"],
                    sanitized_content="Error: Division by zero",
                )
            except Exception:
                # Expression is not simple arithmetic (e.g. complex script)
                pass

        # 3. If recalculated, compare with claimed result in output
        if recalculated is not None:
            # Check if output contains the recalculated number
            recalc_str = str(recalculated)
            # Match numeric token in output
            output_numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_text)
            if output_numbers:
                try:
                    out_val = float(output_numbers[0])
                    diff = abs(out_val - float(recalculated))
                    metrics["numerical_difference"] = diff
                    if diff > 1e-4:
                        issues.append(f"Discrepancy: output {out_val} does not match recalculated value {recalculated}")
                        return ValidationResult(
                            status=ValidationStatus.FLAGGED,
                            is_valid=False,
                            confidence=0.4,
                            issues=issues,
                            metrics=metrics,
                            sanitized_content=raw_text,
                        )
                except Exception:
                    pass

        return ValidationResult(
            status=ValidationStatus.VERIFIED,
            is_valid=True,
            confidence=1.0,
            issues=issues,
            metrics=metrics,
            sanitized_content=raw_text,
        )

