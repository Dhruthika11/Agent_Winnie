"""
OpenTelemetry instrumentation, exported to an OTLP collector (Langfuse
accepts OTLP directly, or point this at any OTel collector that forwards
to Arize Phoenix / your APM of choice).
"""
from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.config import settings

logger = logging.getLogger("hermes.utils.telemetry")

_configured = False


def setup_telemetry(app=None) -> None:
    global _configured
    if _configured:
        return

    resource = Resource(attributes={SERVICE_NAME: settings.service_name})
    provider = TracerProvider(resource=resource)

    if settings.otel_exporter_otlp_endpoint and "4318" not in settings.otel_exporter_otlp_endpoint:
        try:
            exporter = OTLPSpanExporter(
                endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces"
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except Exception:  # noqa: BLE001
            logger.warning("otlp_exporter_setup_failed", exc_info=True)

    trace.set_tracer_provider(provider)

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except Exception:  # noqa: BLE001
            logger.warning("fastapi_instrumentation_failed", exc_info=True)

    # Langfuse can also be wired as a callback handler for LangChain/LangGraph
    # runs to get rich agent-trace views (inputs, tool calls, token usage) on
    # top of the raw OTel spans above.
    try:
        from langfuse import Langfuse  # noqa: F401
    except Exception:  # noqa: BLE001
        logger.info("langfuse_sdk_not_installed_traces_otel_only")

    _configured = True


_lf_client = None


def get_langfuse_client():
    global _lf_client
    if _lf_client is not None:
        return _lf_client
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse import Langfuse

            _lf_client = Langfuse(
                public_key=settings.langfuse_public_key.strip(),
                secret_key=settings.langfuse_secret_key.strip(),
                host=settings.langfuse_host.strip(),
            )
            return _lf_client
        except Exception:  # noqa: BLE001
            logger.warning("langfuse_client_init_failed", exc_info=True)
            return None
    return None


def get_tracer(name: str = "hermes-agent"):
    return trace.get_tracer(name)


import re

SENSITIVE_KEY_PATTERNS = {
    "password", "secret", "api_key", "apikey", "access_token",
    "private_key", "authorization", "secret_key", "master_key", "token"
}

TOKEN_REGEXES = [
    re.compile(r"gsk_[A-Za-z0-9_-]{15,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9_-]{15,}", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9_-]{15,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9\-\._~+/]+=*", re.IGNORECASE),
    re.compile(r"ey[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}", re.IGNORECASE),
]


def sanitize_payload(val: Any) -> Any:
    """Recursively redact sensitive tokens, passwords, and API keys from telemetry payloads."""
    if val is None:
        return None
    if isinstance(val, str):
        res = val
        for rx in TOKEN_REGEXES:
            res = rx.sub("[REDACTED_CREDENTIAL]", res)
        return res
    if isinstance(val, dict):
        sanitized = {}
        for k, v in val.items():
            k_lower = str(k).lower()
            if any(p in k_lower for p in SENSITIVE_KEY_PATTERNS):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_payload(v)
        return sanitized
    if isinstance(val, list):
        return [sanitize_payload(x) for x in val]
    if isinstance(val, tuple):
        return tuple(sanitize_payload(x) for x in val)
    return val

