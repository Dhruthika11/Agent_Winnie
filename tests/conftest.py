"""Shared pytest fixtures."""
import os

# Prevent OpenBLAS thread allocation memory failures on Windows & disable external telemetry during tests
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("GUARDRAILS_TELEMETRY", "0")

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    # Avoid hitting real OTLP/Langfuse endpoints during tests.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    from src.main import app

    return TestClient(app)


@pytest.fixture()
def mock_litellm_completion(monkeypatch):
    """Patch litellm.completion to return a canned, tool-call-free response."""
    import litellm

    class _Msg:
        content = "This is a mocked response."
        tool_calls = None

    class _Choice:
        message = _Msg()

    class _Response:
        choices = [_Choice()]

    def _fake_completion(*args, **kwargs):
        return _Response()

    monkeypatch.setattr(litellm, "completion", _fake_completion)
    return _fake_completion
