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
