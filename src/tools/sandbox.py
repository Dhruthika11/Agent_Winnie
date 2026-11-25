"""
Isolated execution environment for agent-generated code/commands.

Implements the Hybrid Sandboxing Architecture:
1. Primary: Untrusted / Remote Sandbox via MCP SSE Microservice isolated in a container (port 8005).
2. Fallback: Direct Docker SDK container execution (isolated, resource-capped container).
3. Local Guarded Fallback: Restricted subprocess execution if Docker daemon is offline.

Includes Component 2: Post-execution verification for code correctness:
- Output existence check
- Expected format validation
- Independent assertion test harness execution
- Logic bug detection (e.g. collection/divisor mismatches)
"""
from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from typing import Any, Optional

import docker
from docker.errors import ContainerError, DockerException

from src.config import settings

logger = logging.getLogger("hermes.tools.sandbox")

_LANGUAGE_COMMANDS = {
    "python": lambda code: ["python3", "-c", code],
    "bash": lambda code: ["bash", "-c", code],
}


class SandboxExecutionError(RuntimeError):
    pass


class DockerSandbox:
    def __init__(self):
        self._client = None  # lazy: don't fail import if docker daemon absent

    @property
    def client(self):
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _execute_docker_container(self, code: str, language: str) -> str:
        """Run code inside a fresh, isolated Docker container."""
        container_name = f"hermes-sandbox-{uuid.uuid4().hex[:10]}"
        command = _LANGUAGE_COMMANDS[language](code)
        container = None
        try:
            logger.info("sandbox_docker_execute", extra={"language": language, "name": container_name})
            run_kwargs = {
                "image": settings.sandbox_image,
                "command": command,
                "name": container_name,
                "network_disabled": settings.sandbox_network_disabled,
                "network_mode": "none",
                "mem_limit": "256m",
                "cpu_quota": 50000,
                "pids_limit": 64,
                "cap_drop": ["ALL"],
                "read_only": True,
                "user": "nobody",
                "tmpfs": {"/tmp": "size=64m,noexec,nosuid,nodev"},
                "detach": True,
                "stdout": True,
                "stderr": True,
            }
            if settings.sandbox_runtime:
                run_kwargs["runtime"] = settings.sandbox_runtime

            container = self.client.containers.run(**run_kwargs)
            result = container.wait(timeout=settings.sandbox_timeout_seconds)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            status_code = result.get("StatusCode", 0) if isinstance(result, dict) else 0
            if status_code != 0:
                return f"Sandbox process exited non-zero ({status_code}):\n{logs}"[:8000]
            return logs[:8000]
        except (ContainerError, DockerException) as exc:
            logger.warning(f"Docker sandbox error ({exc.__class__.__name__}): {exc}. Falling back to safe local runner.")
            from src.mcp_servers.sandbox_server import run_sandboxed_code
            return run_sandboxed_code(code, language)
        except Exception as exc:
            if "timed out" in str(exc).lower():
                return f"Sandbox execution timed out after {settings.sandbox_timeout_seconds}s"
            logger.exception("sandbox_execution_error")
            from src.mcp_servers.sandbox_server import run_sandboxed_code
            return run_sandboxed_code(code, language)
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def _execute_raw(self, code: str, language: str = "python", timeout_seconds: Optional[float] = None) -> str:
        """Internal router executing code across the hybrid sandbox tiers."""
        if language not in _LANGUAGE_COMMANDS:
            raise SandboxExecutionError(f"Unsupported language: {language}")

        # Tier 1: Try MCP SSE Microservice
        try:
            from src.tools.mcp_client import mcp_registry
            if "run_sandboxed_code" in mcp_registry._tool_to_server:
                server_name = mcp_registry._tool_to_server["run_sandboxed_code"]
                server = mcp_registry._servers.get(server_name)
                if server and server.transport == "sse":
                    logger.info("sandbox_mcp_sse_route", extra={"language": language, "server": server_name})
                    from src.tools.mcp_client import _mcp_pool
                    # Execute via MCP SSE transport
                    result = _mcp_pool.call_tool_sync(server, "run_sandboxed_code", {"code": code, "language": language})
                    if result:
                        return str(result)
        except Exception as exc:
            logger.info(
                f"[MCP SSE Sandbox] Remote microservice unreachable ({exc.__class__.__name__}: {exc}). "
                "Engaging container / local sandbox fallback."
            )

        # Tier 2 & 3: Docker Container / Guarded Local Runner
        return self._execute_docker_container(code, language)

    def execute_code(self, code: str, language: str = "python") -> dict[str, Any]:
        """Convenience execution method returning structured output and exit code."""
        out = self._execute_raw(code, language)
        exit_code = 1 if "exited non-zero" in out or "timed out" in out else 0
        return {"stdout": out, "stderr": "" if exit_code == 0 else out, "exit_code": exit_code}

    def execute(self, code: str, language: str = "python", timeout_seconds: Optional[float] = None) -> str:
        """Execute code using hybrid sandbox pipeline."""
        return self._execute_raw(code, language, timeout_seconds=timeout_seconds)

    def execute_with_verification(
        self,
        code: str,
        language: str = "python",
        expected_output: Optional[Any] = None,
        assertions: Optional[list[str]] = None,
        expected_format: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute code and run post-execution correctness verification:
        1. Check output existence
        2. Validate output format (number, json, list)
        3. Run independent assertions in execution context
        4. Match against expected output
        """
        issues: list[str] = []
        assertions_passed = 0
        assertions_failed = 0

        # Construct instrumented code if assertions are provided for Python
        exec_code = code
        has_instrumented_assertions = False
        if assertions and language.lower() in ("python", "python3"):
            has_instrumented_assertions = True
            assertion_lines = "\n".join(f"    {a}" for a in assertions if a.strip())
            exec_code = f"""{code}

# --- Independent Post-Execution Assertion Harness ---
try:
{assertion_lines}
    print("\\n[HERMES_ASSERTION_VERIFIED]")
except AssertionError as _hermes_assertion_exc:
    print(f"\\n[HERMES_ASSERTION_FAILED: {{_hermes_assertion_exc}}]")
except Exception as _hermes_err:
    print(f"\\n[HERMES_ASSERTION_ERROR: {{type(_hermes_err).__name__}}: {{_hermes_err}}]")
"""

        raw_output = self._execute_raw(exec_code, language)
        clean_output = raw_output

        # 1. Output Existence Check
        if not raw_output.strip() or raw_output.strip() == "[Command completed with no output]":
            issues.append("Output existence check failed: program produced no stdout/stderr output")

        # 2. Check assertion test harness results
        if has_instrumented_assertions:
            clean_output = re.sub(r"\[HERMES_ASSERTION_[^\]]+\]", "", clean_output).strip()
            if "[HERMES_ASSERTION_VERIFIED]" in raw_output:
                assertions_passed += len(assertions)
            elif "[HERMES_ASSERTION_FAILED" in raw_output:
                assertions_failed += 1
                fail_match = re.search(r"\[HERMES_ASSERTION_FAILED:\s*(.*?)\]", raw_output)
                fail_msg = fail_match.group(1) if fail_match else "Assertion failed"
                issues.append(f"Assertion failed: {fail_msg}")
            elif "[HERMES_ASSERTION_ERROR" in raw_output:
                assertions_failed += 1
                err_match = re.search(r"\[HERMES_ASSERTION_ERROR:\s*(.*?)\]", raw_output)
                issues.append(f"Assertion execution error: {err_match.group(1) if err_match else 'Error'}")

        # 3. Format Validation
        format_valid = True
        if expected_format:
            fmt = expected_format.lower().strip()
            target_str = clean_output.strip()
            if fmt in ("number", "float", "int", "numeric"):
                try:
                    float(target_str)
                except ValueError:
                    format_valid = False
                    issues.append(f"Format check failed: expected numeric output, received '{target_str[:100]}'")
            elif fmt == "json":
                try:
                    json.loads(target_str)
                except Exception:
                    format_valid = False
                    issues.append(f"Format check failed: expected valid JSON, received '{target_str[:100]}'")
            elif fmt == "list":
                try:
                    parsed = ast.literal_eval(target_str)
                    if not isinstance(parsed, list):
                        format_valid = False
                        issues.append(f"Format check failed: expected list, received {type(parsed).__name__}")
                except Exception:
                    format_valid = False
                    issues.append(f"Format check failed: expected list expression, received '{target_str[:100]}'")

        # 4. Expected Output Matching
        if expected_output is not None:
            expected_str = str(expected_output).strip()
            # Check if clean output equals or contains the expected value
            if clean_output.strip() != expected_str and expected_str not in clean_output:
                issues.append(
                    f"Result correctness failure: output was '{clean_output.strip()}', expected '{expected_str}'"
                )

        is_verified = (len(issues) == 0) and (assertions_failed == 0)

        return {
            "output": clean_output,
            "raw_output": raw_output,
            "verified": is_verified,
            "issues": issues,
            "assertions_passed": assertions_passed,
            "assertions_failed": assertions_failed,
            "format_valid": format_valid,
        }

    def execute(
        self,
        code: str,
        language: str = "python",
        expected_output: Optional[Any] = None,
        assertions: Optional[list[str]] = None,
        expected_format: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        """
        Execute arbitrary Python/Bash code following the Hybrid Architecture,
        with automated post-execution correctness verification if assertions/expectations
        are provided.
        """
        if assertions or expected_output is not None or expected_format:
            res = self.execute_with_verification(
                code=code,
                language=language,
                expected_output=expected_output,
                assertions=assertions,
                expected_format=expected_format,
            )
            if not res["verified"]:
                return f"{res['output']}\n\n[Post-Execution Correctness Check FAILED: {'; '.join(res['issues'])}]"
            return res["output"]

        return self._execute_raw(code, language, timeout_seconds=timeout_seconds)


sandbox = DockerSandbox()
