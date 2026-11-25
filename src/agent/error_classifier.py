"""
Centralized Error Classification and Provider Circuit Breaker for Hermes Agent Platform.

Categorizes all LLM and provider errors:
1. Distinguishes TPM quota exhaustion, context-length overflow, authentication,
   invalid requests, temporary network errors, provider outages, and timeouts.
2. Identifies whether an error is transient (retryable) or permanent (fail fast).
3. Respects Retry-After metadata when available.
4. Tracks provider health with circuit breakers to prevent cascade failures
   across models sharing the same provider quota.
"""
from __future__ import annotations

import enum
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("hermes.agent.error_classifier")


class ErrorCategory(str, enum.Enum):
    TPM_QUOTA = "tpm_quota"
    CONTEXT_LENGTH = "context_length"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    TEMPORARY_NETWORK = "temporary_network"
    PROVIDER_OUTAGE = "provider_outage"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedError:
    category: ErrorCategory
    is_retryable: bool
    retry_after_seconds: Optional[float]
    provider: str
    model: str
    raw_message: str
    recommended_action: str  # "switch_provider" | "compact_prompt" | "fail_fast" | "wait_and_retry"

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "is_retryable": self.is_retryable,
            "retry_after_seconds": self.retry_after_seconds,
            "provider": self.provider,
            "model": self.model,
            "raw_message": self.raw_message,
            "recommended_action": self.recommended_action,
        }


def extract_provider_from_model(model: str) -> str:
    """Identify the underlying provider for a model string."""
    clean = (model or "").lower().strip()
    if clean.startswith("groq/") or "groq" in clean:
        return "groq"
    if clean.startswith("gemini/") or "gemini" in clean or "google" in clean:
        return "gemini"
    if clean.startswith("openai/") or "gpt-" in clean:
        return "openai"
    if "claude" in clean or "anthropic" in clean:
        return "anthropic"
    if clean.startswith("litellm-proxy") or ":4000" in clean:
        return "litellm-proxy"
    return "unknown"


class ErrorClassifier:
    """Classifies exceptions from LiteLLM and upstream providers into actionable categories."""

    TPM_PATTERNS = [
        r"tokens\s+per\s+minute",
        r"\btpm\b",
        r"limit\s+\d+",
        r"request\s+too\s+large",
        r"rate\s*limit",
        r"quota\s*exceeded",
        r"too\s+many\s+requests",
        r"\b429\b",
    ]

    CONTEXT_LENGTH_PATTERNS = [
        r"context\s*length",
        r"maximum\s+context",
        r"max\s+tokens",
        r"prompt\s+is\s+too\s+long",
        r"token\s+limit\s+exceeded",
    ]

    AUTH_PATTERNS = [
        r"unauthorized",
        r"authentication",
        r"invalid\s*api\s*key",
        r"api\s*key\s*not\s*found",
        r"permission\s*denied",
        r"\b401\b",
        r"\b403\b",
    ]

    INVALID_REQUEST_PATTERNS = [
        r"invalid_request_error",
        r"bad\s*request",
        r"schema\s*error",
        r"unrecognized\s*parameter",
        r"\b400\b",
    ]

    OUTAGE_PATTERNS = [
        r"bad\s*gateway",
        r"service\s*unavailable",
        r"gateway\s*timeout",
        r"internal\s*server\s*error",
        r"\b500\b",
        r"\b502\b",
        r"\b503\b",
        r"\b504\b",
    ]

    NETWORK_PATTERNS = [
        r"connection\s*refused",
        r"connection\s*reset",
        r"network\s*unreachable",
        r"name\s*resolution\s*failed",
        r"failed\s+to\s+connect",
    ]

    TIMEOUT_PATTERNS = [
        r"timed\s*out",
        r"timeout",
        r"readtimeout",
        r"connecttimeout",
    ]

    RETRY_AFTER_REGEX = re.compile(r"retry[\s_-]?after[:\s]+(\d+(?:\.\d+)?)", re.IGNORECASE)

    @classmethod
    def classify(cls, exc: Exception | str, model: str = "") -> ClassifiedError:
        msg = str(exc).lower()
        provider = extract_provider_from_model(model)
        status_code = getattr(exc, "status_code", None)

        # 1. Parse Retry-After if present
        retry_after: Optional[float] = None
        match = cls.RETRY_AFTER_REGEX.search(msg)
        if match:
            try:
                retry_after = float(match.group(1))
            except Exception:
                pass

        # 2. Context length overflow (non-retryable unless compacted; check before generic 400 bad request)
        if any(re.search(p, msg) for p in cls.CONTEXT_LENGTH_PATTERNS):
            return ClassifiedError(
                category=ErrorCategory.CONTEXT_LENGTH,
                is_retryable=False,
                retry_after_seconds=None,
                provider=provider,
                model=model,
                raw_message=str(exc),
                recommended_action="compact_prompt",
            )

        # 3. TPM / Rate limit quota exhaustion
        # (NOT retryable on the same provider with the same payload; must switch provider or compact)
        if status_code == 429 or any(re.search(p, msg) for p in cls.TPM_PATTERNS):
            return ClassifiedError(
                category=ErrorCategory.TPM_QUOTA,
                is_retryable=False,  # An oversized request or exhausted TPM will not succeed by retrying
                retry_after_seconds=retry_after,
                provider=provider,
                model=model,
                raw_message=str(exc),
                recommended_action="switch_provider",
            )

        # 4. Authentication errors (strictly non-retryable)
        if status_code in (401, 403) or any(re.search(p, msg) for p in cls.AUTH_PATTERNS):
            return ClassifiedError(
                category=ErrorCategory.AUTHENTICATION,
                is_retryable=False,
                retry_after_seconds=None,
                provider=provider,
                model=model,
                raw_message=str(exc),
                recommended_action="fail_fast",
            )

        # 5. Invalid Request errors (strictly non-retryable without payload changes)
        if status_code == 400 or any(re.search(p, msg) for p in cls.INVALID_REQUEST_PATTERNS):
            return ClassifiedError(
                category=ErrorCategory.INVALID_REQUEST,
                is_retryable=False,
                retry_after_seconds=None,
                provider=provider,
                model=model,
                raw_message=str(exc),
                recommended_action="fail_fast",
            )

        # 6. Provider Outages (500, 502, 503, 504) -> transient, retryable on fallback
        if status_code in (500, 502, 503, 504) or any(re.search(p, msg) for p in cls.OUTAGE_PATTERNS):
            return ClassifiedError(
                category=ErrorCategory.PROVIDER_OUTAGE,
                is_retryable=True,
                retry_after_seconds=retry_after,
                provider=provider,
                model=model,
                raw_message=str(exc),
                recommended_action="wait_and_retry",
            )

        # 7. Timeouts -> transient, retryable
        if any(re.search(p, msg) for p in cls.TIMEOUT_PATTERNS):
            return ClassifiedError(
                category=ErrorCategory.TIMEOUT,
                is_retryable=True,
                retry_after_seconds=retry_after,
                provider=provider,
                model=model,
                raw_message=str(exc),
                recommended_action="wait_and_retry",
            )

        # 8. Temporary network failures
        if any(re.search(p, msg) for p in cls.NETWORK_PATTERNS):
            return ClassifiedError(
                category=ErrorCategory.TEMPORARY_NETWORK,
                is_retryable=True,
                retry_after_seconds=retry_after,
                provider=provider,
                model=model,
                raw_message=str(exc),
                recommended_action="wait_and_retry",
            )

        # Default fallback
        return ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            is_retryable=False,
            retry_after_seconds=None,
            provider=provider,
            model=model,
            raw_message=str(exc),
            recommended_action="fail_fast",
        )


class ProviderCircuitBreaker:
    """
    Maintains health status and cooldowns for upstream providers.
    Prevents repeated cascading failures across models that share the same provider quota.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failure_counts: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}

    def is_available(self, provider: str) -> bool:
        """Check if provider is healthy and not in cooldown."""
        now = time.time()
        until = self._cooldown_until.get(provider, 0.0)
        if now < until:
            return False
        return True

    def record_success(self, provider: str) -> None:
        """Reset failure count on successful invocation."""
        self._failure_counts[provider] = 0
        self._cooldown_until.pop(provider, None)

    def record_failure(self, provider: str, error: ClassifiedError) -> None:
        """Record a failure and activate circuit breaker if threshold is reached."""
        count = self._failure_counts.get(provider, 0) + 1
        self._failure_counts[provider] = count

        # If TPM quota or provider outage, trip circuit breaker faster
        threshold = 1 if error.category == ErrorCategory.TPM_QUOTA else self.failure_threshold
        if count >= threshold:
            cooldown = error.retry_after_seconds or self.cooldown_seconds
            self._cooldown_until[provider] = time.time() + cooldown
            logger.warning(
                "Circuit breaker tripped for provider '%s' (%s). Cooldown active for %.1fs.",
                provider, error.category.value, cooldown
            )

    def get_status(self) -> dict[str, Any]:
        now = time.time()
        return {
            provider: {
                "available": now >= until,
                "cooldown_remaining_sec": max(0.0, round(until - now, 1)),
                "failures": self._failure_counts.get(provider, 0),
            }
            for provider, until in self._cooldown_until.items()
        }


error_classifier = ErrorClassifier()
provider_circuit_breaker = ProviderCircuitBreaker()
