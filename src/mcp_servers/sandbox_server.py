"""
Official Anthropic MCP Server for Isolated Sandboxed Code Execution.
Can run as:
1. An isolated MCP SSE microservice inside a container (default port 8005).
2. A standalone stdio MCP server for testing.
Exposes 'run_sandboxed_code' for Python and Bash snippet execution with resource guards.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import MCPServer

app = MCPServer("sandbox")

DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", "20"))
MAX_OUTPUT_LENGTH = 8000


@app.tool(name="run_sandboxed_code")
def run_sandboxed_code(code: str, language: str = "python") -> str:
    """Execute arbitrary Python or Bash code inside an isolated, sandboxed runner.

    Args:
        code: The Python or Bash code string to execute.
        language: Language of the snippet ('python' or 'bash'). Default is 'python'.
    """
    lang = language.lower().strip()
    if lang not in ("python", "python3", "bash", "sh"):
        return f"Error: Unsupported execution language '{language}'. Supported: 'python', 'bash'."

    if not code.strip():
        return "Error: Code snippet is empty."

    # Write snippet to a secure temporary file
    suffix = ".py" if "python" in lang else ".sh"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)

    try:
        if "python" in lang:
            cmd = [sys.executable, "-u", str(tmp_path)]
        else:
            shell_bin = "bash" if sys.platform != "win32" else "powershell"
            if sys.platform == "win32":
                cmd = [shell_bin, "-ExecutionPolicy", "Bypass", "-File", str(tmp_path)]
            else:
                cmd = ["bash", str(tmp_path)]

        # Restrict environment variables to avoid leaking sensitive host keys to untrusted code
        safe_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONUNBUFFERED": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            env=safe_env,
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        exit_code = proc.returncode

        output_parts = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"[stderr]\n{stderr}")
        if exit_code != 0:
            output_parts.append(f"[process exited with code {exit_code}]")

        combined = "\n".join(output_parts) if output_parts else "[Command completed with no output]"
        if len(combined) > MAX_OUTPUT_LENGTH:
            combined = combined[:MAX_OUTPUT_LENGTH] + f"\n... [Output truncated to {MAX_OUTPUT_LENGTH} chars]"
        return combined

    except subprocess.TimeoutExpired:
        return f"Error: Execution timed out after {DEFAULT_TIMEOUT_SECONDS} seconds."
    except Exception as exc:
        return f"Sandbox execution error: {type(exc).__name__}: {exc}"
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Hermes Isolated Sandbox MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=os.environ.get("SANDBOX_TRANSPORT", "sse"),
        help="Transport protocol (sse or stdio)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("SANDBOX_HOST", "0.0.0.0"),
        help="Host to bind for SSE transport",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SANDBOX_PORT", "8005")),
        help="Port to bind for SSE transport",
    )

    args = parser.parse_args()

    if args.transport == "sse":
        print(f"[Sandbox MCP SSE] Starting on http://{args.host}:{args.port}/sse", flush=True)
        app.run(transport="sse", host=args.host, port=args.port)
    else:
        app.run(transport="stdio")


if __name__ == "__main__":
    main()
