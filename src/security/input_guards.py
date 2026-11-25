"""
Pre-execution guardrails applied to user input before it reaches the model.

Two layers:
  1. Fast heuristic checks (always run, zero external dependency) for
     obvious prompt-injection / jailbreak patterns.
  2. Optional Guardrails AI validators, if configured, for deeper checks
     (e.g. topic restriction, toxic language).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("hermes.security.input_guards")

_INJECTION_PATTERNS = [
    r"ignore (all|any|previous|the) (previous |prior )?instructions",
    r"disregard (all|any|previous|the) (previous |prior )?(instructions|rules)",
    r"you are now (in )?dan\b",
    r"pretend (you have no|there are no) (restrictions|rules|guidelines|content policy)",
    r"reveal (your|the) system prompt",
    r"act as if you have no (content policy|filter|restrictions)",
    r"\bjailbreak\b",
    r"do anything now",
    r"override your (safety|content) (settings|policy)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

try:
    import guardrails as gd
    _guard = gd.Guard()  # extend with .use(validator, ...) for hosted validators
except Exception:  # noqa: BLE001
    _guard = None


def _heuristic_flag(text: str) -> tuple[bool, str]:
    for pattern in _COMPILED:
        if pattern.search(text):
            return True, f"matched pattern: {pattern.pattern}"
    return False, ""


def sanitize_input(text: str) -> tuple[str, bool, str]:
    """Returns (possibly-annotated text, flagged, reason). Never blocks
    silently — flags are surfaced to the caller (graph.py logs them and can
    choose to short-circuit) rather than mutating user intent invisibly."""
    if not text:
        return text, False, ""

    flagged, reason = _heuristic_flag(text)

    if _guard is not None:
        try:
            outcome = _guard.validate(text)
            if not outcome.validation_passed:
                flagged = True
                reason = reason or "guardrails_ai_validation_failed"
        except Exception:  # noqa: BLE001
            logger.debug("guardrails_ai_check_skipped", exc_info=True)

    return text, flagged, reason
