"""
Unit and Integration Tests for Sandbox Isolation Boundaries.

Tests the security restrictions of DockerSandbox and run_sandboxed_code:
1. Filesystem isolation: Attempting to write outside /tmp fails or is blocked by read-only root.
2. Network isolation: Subprocess/container environment has network access disabled.
3. Environment scrubbing: Sensitive host credentials (API keys, DB secrets) are scrubbed from sandbox.
4. Resource caps & Timeout bounds: Infinite loops or slow processes trigger timeout termination.
5. Assertion verification harness: Logic errors and format contract violations caught.
"""
from __future__ import annotations

import os
import subprocess
from unittest.mock import patch, MagicMock
import pytest

from src.tools.sandbox import DockerSandbox
from src.mcp_servers.sandbox_server import run_sandboxed_code, DEFAULT_TIMEOUT_SECONDS


class TestSandboxIsolationRestrictions:
    def test_environment_scrubbing_cleans_host_secrets(self, monkeypatch):
        """Host secrets must never be exposed to sandboxed code."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-supersecretkey123456789")
        monkeypatch.setenv("HERMES_DB_PASSWORD", "db_super_secret_pw")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws_secret_key_abcdef")

        # Python snippet attempting to read sensitive environment variables
        code = (
            "import os\n"
            "leaks = [k for k in os.environ if any(s in k for s in ['KEY', 'SECRET', 'PASSWORD', 'DB'])]\n"
            "print('LEAKED:' + ','.join(leaks) if leaks else 'SAFE_CLEAN_ENV')\n"
        )
        output = run_sandboxed_code(code, language="python")
        assert "SAFE_CLEAN_ENV" in output
        assert "sk-proj-supersecretkey123456789" not in output
        assert "db_super_secret_pw" not in output

    def test_sandbox_docker_spec_includes_strict_isolation_flags(self):
        """Docker sandbox container specs must enforce none network, read-only root, no-new-privileges, and tmpfs."""
        sandbox = DockerSandbox()
        mock_containers = MagicMock()
        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"isolation_test_passed\n"
        mock_containers.run.return_value = mock_container

        sandbox._client = MagicMock(containers=mock_containers)
        output = sandbox._execute_docker_container("print('hello')", language="python")
        assert "isolation_test_passed" in output

        # Verify kwargs passed to Docker daemon
        call_kwargs = mock_containers.run.call_args[1]
        assert call_kwargs["network_mode"] == "none"
        assert call_kwargs["network_disabled"] is True
        assert call_kwargs["read_only"] is True
        assert call_kwargs["cap_drop"] == ["ALL"]
        assert call_kwargs["user"] == "nobody"
        assert "mem_limit" in call_kwargs
        assert "cpu_quota" in call_kwargs
        assert "pids_limit" in call_kwargs
        assert "/tmp" in call_kwargs["tmpfs"]

    def test_execution_timeout_kills_runaway_loop(self):
        """Infinite loop snippets must terminate safely via timeout."""
        infinite_code = "import time\nwhile True:\n    time.sleep(0.1)"
        with patch("src.mcp_servers.sandbox_server.DEFAULT_TIMEOUT_SECONDS", 1):
            output = run_sandboxed_code(infinite_code, language="python")
            assert "timed out" in output.lower()

    def test_unsupported_language_rejected(self):
        """Only explicitly supported runners (Python, Bash) may execute."""
        output = run_sandboxed_code("console.log('hi')", language="javascript")
        assert "Unsupported execution language" in output

    def test_syntax_and_runtime_errors_captured(self):
        """Syntax errors or runtime crashes return stderr safely without hanging."""
        buggy_code = "def faulty_func():\n    return 1 / 0\nfaulty_func()"
        output = run_sandboxed_code(buggy_code, language="python")
        assert "ZeroDivisionError" in output
        assert "[process exited with code" in output or "[stderr]" in output
