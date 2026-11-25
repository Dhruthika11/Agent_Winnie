"""
Official Anthropic Model Context Protocol (MCP) Client with Hybrid Architecture:
1. Primary / Local Tools (Filesystem, Terminal, Web Scraper, GitHub):
   - Persistent stdio Process Pool (warm long-lived subprocesses, zero open network ports, sub-15ms execution).
2. Untrusted / Remote Sandboxes (arbitrary Python/Bash code execution):
   - MCP SSE Microservice transport (official SSE JSON-RPC 2.0 streaming) with robust fallback.
3. Legacy HTTP transport support.
4. Built-in local fallback if any server or subprocess is unreachable.

Each server is configured in infrastructure/mcp_servers/*.json.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import httpx

from src.config import settings
from src.tools.circuit_breaker import tool_circuit_breaker

logger = logging.getLogger("hermes.tools.mcp_client")

CONFIG_DIR = Path(__file__).resolve().parents[2] / "infrastructure" / "mcp_servers"


class MCPServerConfig:
    def __init__(self, raw: dict):
        self.name: str = raw["name"]
        self.transport: str = raw.get("transport", "stdio" if "command" in raw else "http").lower()
        self.command: str = raw.get("command", "python")
        self.args: list[str] = raw.get("args", [])
        self.env: dict[str, str] | None = raw.get("env", None)
        self.base_url: str = raw.get("base_url", "")
        self.risk_level: str = raw.get("risk_level", "low")
        self.tools: list[dict] = raw.get("tools", [])
        self.timeout_seconds: float = float(raw.get("timeout_seconds", 20.0))


class PersistentSession:
    """Manages a single warm, persistent MCP session (stdio subprocess or SSE stream)

    running within the dedicated MCP event loop.
    """

    def __init__(self, server: MCPServerConfig):
        self.server = server
        self.lock = asyncio.Lock()
        self.stack: AsyncExitStack | None = None
        self.session: Any | None = None  # ClientSession
        self.is_connected = False
        self.last_connected_at: float = 0.0
        self.call_count: int = 0
        self.total_latency_ms: float = 0.0

    async def _resolve_command(self) -> str:
        cmd = self.server.command
        if cmd in ("python", "python3"):
            return sys.executable
        if sys.platform == "win32" and cmd in ("npx", "npm"):
            resolved = shutil.which(f"{cmd}.cmd") or shutil.which(cmd)
            if resolved:
                return resolved
        return cmd

    async def connect(self) -> Any:
        """Initialize and maintain persistent connection with auto-reconnect."""
        if self.session is not None and self.is_connected:
            return self.session

        from mcp import ClientSession

        stack = AsyncExitStack()
        t0 = time.perf_counter()

        try:
            if self.server.transport == "stdio":
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                cmd = await self._resolve_command()
                env = os.environ.copy()
                if self.server.env:
                    env.update(self.server.env)

                params = StdioServerParameters(command=cmd, args=self.server.args, env=env)
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                self.stack = stack
                self.session = session
                self.is_connected = True
                self.last_connected_at = time.time()
                latency = (time.perf_counter() - t0) * 1000
                logger.info(
                    f"[MCP STDIO POOL] Warm session established for '{self.server.name}' in {latency:.2f}ms"
                )
                return self.session

            elif self.server.transport == "sse":
                from mcp.client.sse import sse_client

                url = self.server.base_url or getattr(settings, "sandbox_mcp_url", "http://127.0.0.1:8005/sse")
                logger.info(f"[MCP SSE CLIENT] Connecting to persistent SSE stream: {url}")

                read, write = await stack.enter_async_context(
                    sse_client(url, timeout=self.server.timeout_seconds)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                self.stack = stack
                self.session = session
                self.is_connected = True
                self.last_connected_at = time.time()
                latency = (time.perf_counter() - t0) * 1000
                logger.info(
                    f"[MCP SSE CLIENT] Warm session established for '{self.server.name}' at {url} in {latency:.2f}ms"
                )
                return self.session

            else:
                raise ValueError(f"Unsupported transport for persistent pool: {self.server.transport}")

        except Exception as exc:
            await stack.aclose()
            self.session = None
            self.stack = None
            self.is_connected = False
            raise exc

    async def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool call using the warm persistent connection."""
        async with self.lock:
            # Connect or reuse warm session
            session = await self.connect()
            t0 = time.perf_counter()
            try:
                res = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments),
                    timeout=self.server.timeout_seconds,
                )
                duration_ms = (time.perf_counter() - t0) * 1000
                self.call_count += 1
                self.total_latency_ms += duration_ms

                if getattr(res, "isError", False):
                    raise RuntimeError(f"MCP server returned error for tool '{tool_name}': {res.content}")

                if not res.content:
                    return ""

                parsed_items = []
                for item in res.content:
                    if hasattr(item, "text"):
                        text = item.text
                        try:
                            parsed_items.append(json.loads(text))
                        except (json.JSONDecodeError, TypeError):
                            parsed_items.append(text)
                    else:
                        parsed_items.append(str(item))

                result = parsed_items[0] if len(parsed_items) == 1 else parsed_items
                logger.info(
                    f"[MCP POOL] Executed '{tool_name}' on '{self.server.name}' ({self.server.transport}) in {duration_ms:.2f}ms"
                )
                return result

            except BaseException as exc:
                # Broken pipe, connection reset, or timeout -> reset session so next call auto-reconnects
                logger.warning(
                    f"[MCP POOL] Session error for '{self.server.name}' ({exc.__class__.__name__}: {exc}). Resetting persistent session."
                )
                await self.disconnect()
                raise exc

    async def ping(self) -> dict:
        """Measure live round-trip latency to the warm server."""
        async with self.lock:
            start = time.perf_counter()
            session = await self.connect()
            tools_res = await session.list_tools()
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "status": "connected",
                "transport": self.server.transport,
                "latency_ms": latency_ms,
                "protocol_version": "2.0",
                "server_name": self.server.name,
                "tools_count": len(tools_res.tools),
                "calls_served": self.call_count,
            }

    async def disconnect(self) -> None:
        """Gracefully close and release subprocess/SSE resources."""
        stack = self.stack
        self.stack = None
        self.session = None
        self.is_connected = False
        if stack is not None:
            try:
                await stack.aclose()
            except BaseException:
                pass


class PersistentMCPSessionPool:
    """Singleton background pool managing persistent stdio subprocesses and SSE connections
    on a dedicated event loop thread. Provides zero-latency tool calls without thread/loop contention.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True, name="mcp-pool-loop")
        self._thread.start()
        self._sessions: dict[str, PersistentSession] = {}
        self._pool_lock = threading.Lock()

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _get_or_create_session(self, server: MCPServerConfig) -> PersistentSession:
        with self._pool_lock:
            if server.name not in self._sessions:
                self._sessions[server.name] = PersistentSession(server)
            return self._sessions[server.name]

    def call_tool_sync(self, server: MCPServerConfig, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Synchronously execute a tool on the persistent session via the dedicated background loop."""
        session = self._get_or_create_session(server)
        future = asyncio.run_coroutine_threadsafe(
            session.invoke_tool(tool_name, arguments),
            self._loop,
        )
        try:
            return future.result(timeout=server.timeout_seconds + 3.0)
        except Exception:
            future.cancel()
            raise

    async def call_tool_async(self, server: MCPServerConfig, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Asynchronously execute a tool on the persistent session."""
        session = self._get_or_create_session(server)
        future = asyncio.run_coroutine_threadsafe(
            session.invoke_tool(tool_name, arguments),
            self._loop,
        )
        return await asyncio.wrap_future(future)

    def ping_server_sync(self, server: MCPServerConfig) -> dict:
        """Synchronously ping the server via the dedicated loop."""
        session = self._get_or_create_session(server)
        future = asyncio.run_coroutine_threadsafe(session.ping(), self._loop)
        return future.result(timeout=server.timeout_seconds + 2.0)

    def warmup(self, servers: list[MCPServerConfig]) -> None:
        """Pre-warm persistent sessions in parallel."""
        async def _warmup_all():
            tasks = []
            for s in servers:
                if s.transport in ("stdio", "sse"):
                    session = self._get_or_create_session(s)
                    tasks.append(asyncio.create_task(session.connect()))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(_warmup_all(), self._loop)

    def shutdown(self) -> None:
        """Gracefully disconnect all active MCP sessions."""
        async def _close_all():
            with self._pool_lock:
                sessions = list(self._sessions.values())
            for sess in sessions:
                await sess.disconnect()

        future = asyncio.run_coroutine_threadsafe(_close_all(), self._loop)
        try:
            future.result(timeout=5.0)
        except Exception:
            pass


# Global singleton pool
_mcp_pool = PersistentMCPSessionPool()


class MCPRegistry:
    """Loads configured MCP servers and executes tool calls using the Hybrid Architecture:

    - stdio: Persistent stdio Process Pool (<15ms latency, local tools)
    - sse: MCP SSE Microservice transport (isolated containers)
    - http: Standard REST/HTTP transport
    - local fallback: resilient in-process execution on failure
    """

    def __init__(self, config_dir: Path = CONFIG_DIR):
        self._servers: dict[str, MCPServerConfig] = {}
        self._tool_to_server: dict[str, str] = {}
        self._load(config_dir)

    def _load(self, config_dir: Path) -> None:
        if config_dir.exists():
            for path in sorted(config_dir.glob("*.json")):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8-sig"))
                    cfg = MCPServerConfig(raw)
                    self._servers[cfg.name] = cfg
                    for tool in cfg.tools:
                        self._tool_to_server[tool["name"]] = cfg.name
                except Exception:  # noqa: BLE001
                    logger.exception("mcp_config_load_failed", extra={"file": str(path)})
        else:
            logger.warning("mcp_config_dir_missing", extra={"path": str(config_dir)})

        self._load_from_redis()

    def _load_from_redis(self) -> None:
        """Discover dynamically registered servers from distributed Redis storage."""
        if not getattr(settings, "redis_url", ""):
            return
        try:
            import redis
            client = redis.Redis.from_url(settings.redis_url, socket_timeout=1.0, decode_responses=True)
            keys = client.keys("hermes:tools:registry:*")
            for key in keys:
                try:
                    raw_json = client.get(key)
                    if raw_json:
                        raw = json.loads(raw_json)
                        cfg = MCPServerConfig(raw)
                        self._servers[cfg.name] = cfg
                        for tool in cfg.tools:
                            self._tool_to_server[tool["name"]] = cfg.name
                        logger.info(f"[DISTRIBUTED REGISTRY] Discovered dynamic server '{cfg.name}' from Redis")
                except Exception:
                    pass
        except Exception:
            pass

    def register_server(self, raw_or_cfg: dict | MCPServerConfig, sync_redis: bool = True) -> MCPServerConfig:
        """Dynamically register a new tool server at runtime across the platform."""
        if isinstance(raw_or_cfg, dict):
            cfg = MCPServerConfig(raw_or_cfg)
            raw_dict = raw_or_cfg
        else:
            cfg = raw_or_cfg
            raw_dict = {
                "name": cfg.name,
                "transport": cfg.transport,
                "command": cfg.command,
                "args": cfg.args,
                "env": cfg.env,
                "base_url": cfg.base_url,
                "risk_level": cfg.risk_level,
                "tools": cfg.tools,
                "timeout_seconds": cfg.timeout_seconds,
            }

        self._servers[cfg.name] = cfg
        for tool in cfg.tools:
            self._tool_to_server[tool["name"]] = cfg.name

        if sync_redis and getattr(settings, "redis_url", ""):
            try:
                import redis
                client = redis.Redis.from_url(settings.redis_url, socket_timeout=1.0, decode_responses=True)
                client.set(f"hermes:tools:registry:{cfg.name}", json.dumps(raw_dict))
            except Exception as exc:
                logger.warning(f"Failed to persist dynamic server '{cfg.name}' to Redis: {exc}")

        logger.info(f"[DISTRIBUTED REGISTRY] Registered server '{cfg.name}' ({cfg.transport}) with {len(cfg.tools)} tools")
        return cfg

    def deregister_server(self, server_name: str, sync_redis: bool = True) -> bool:
        """Dynamically deregister a tool server at runtime."""
        if server_name not in self._servers:
            return False

        server = self._servers.pop(server_name)
        for t in server.tools:
            self._tool_to_server.pop(t["name"], None)

        if sync_redis and getattr(settings, "redis_url", ""):
            try:
                import redis
                client = redis.Redis.from_url(settings.redis_url, socket_timeout=1.0, decode_responses=True)
                client.delete(f"hermes:tools:registry:{server_name}")
            except Exception:
                pass

        logger.info(f"[DISTRIBUTED REGISTRY] Deregistered server '{server_name}'")
        return True

    def tool_schemas(self) -> list[dict]:
        """Return OpenAI/LiteLLM-style tool schemas for every registered tool,

        plus the built-in sandboxed code execution tool, deduplicated.
        """
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "run_sandboxed_code",
                    "description": (
                        "Execute arbitrary Python or Bash code inside an isolated, "
                        "sandboxed container. Use for calculations, data analysis, "
                        "or code inspection."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "language": {
                                "type": "string",
                                "enum": ["python", "bash"],
                                "default": "python",
                                "description": "Execution language runtime",
                            },
                            "code": {
                                "type": "string",
                                "description": "Source code to execute inside the sandbox",
                            },
                        },
                        "required": ["language", "code"],
                    },
                },
            }
        ]

        registered_names = {"run_sandboxed_code"}
        for server in self._servers.values():
            for tool in server.tools:
                t_name = tool["name"]
                if t_name in registered_names:
                    continue
                registered_names.add(t_name)
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t_name,
                            "description": tool.get("description", ""),
                            "parameters": tool.get(
                                "parameters", {"type": "object", "properties": {}}
                            ),
                        },
                    }
                )
        return schemas

    get_tools_schema = tool_schemas

    def get_servers_info(self) -> list[dict]:
        """Return structured information about all registered MCP servers, their tools, and circuit status."""
        res = []
        for server in self._servers.values():
            circuit_data = tool_circuit_breaker.get_status(server.name)
            res.append({
                "name": server.name,
                "transport": server.transport,
                "command": server.command,
                "args": server.args,
                "risk_level": server.risk_level,
                "timeout_seconds": server.timeout_seconds,
                "base_url": server.base_url,
                "architecture_tier": (
                    "Persistent stdio Process Pool (Local)"
                    if server.transport == "stdio"
                    else "MCP SSE Microservice (Isolated Container)"
                    if server.transport == "sse"
                    else "HTTP Microservice"
                ),
                "circuit_breaker": circuit_data,
                "tools": [
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    }
                    for t in server.tools
                ],
            })
        return res

    def get_circuit_status(self, server_name: str | None = None) -> dict:
        """Return real-time circuit-breaking diagnostics across the distributed registry."""
        return tool_circuit_breaker.get_status(server_name)

    def reset_circuit(self, server_name: str) -> bool:
        """Manually reset a tripped circuit breaker for a server."""
        return tool_circuit_breaker.reset(server_name)

    def trip_circuit(self, server_name: str, reason: str = "Manual override") -> bool:
        """Manually trip a circuit breaker for testing or maintenance."""
        return tool_circuit_breaker.trip(server_name, reason)

    def ping_server(self, server_name: str) -> dict:
        server = self._servers.get(server_name)
        if not server:
            raise ValueError(f"Server '{server_name}' not found")

        if server.transport in ("stdio", "sse"):
            try:
                return _mcp_pool.ping_server_sync(server)
            except Exception as exc:
                return {
                    "status": "degraded",
                    "transport": server.transport,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
        else:
            # Legacy HTTP transport
            start = time.perf_counter()
            with httpx.Client(timeout=server.timeout_seconds) as client:
                resp = client.get(f"{server.base_url}/health")
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                return {"status": "connected", "latency_ms": latency_ms, "code": resp.status_code}

    def warmup(self) -> None:
        """Pre-warm persistent stdio subprocesses and SSE streams."""
        _mcp_pool.warmup(list(self._servers.values()))

    def shutdown(self) -> None:
        """Cleanly terminate all persistent connections and child processes."""
        _mcp_pool.shutdown()

    def _fallback(self, name: str, arguments: dict[str, Any]) -> Any:
        """Local fallback when an MCP server or subprocess is unavailable."""
        from src.tools.builtin_tools import (
            filesystem_list_directory,
            filesystem_read_file,
            filesystem_write_file,
            github_get_repository,
            github_list_issues,
            github_list_user_repositories,
            github_search_repositories,
            terminal_delete_file,
            web_scraper_read_url,
            web_scraper_search,
        )
        if name == "web_scraper.search":
            return web_scraper_search(arguments.get("query", ""), arguments.get("max_results", 5))
        elif name == "web_scraper.read_url":
            return web_scraper_read_url(arguments.get("url", ""))
        elif name == "terminal.delete_file":
            return terminal_delete_file(arguments.get("path", ""))
        elif name == "filesystem.read_file":
            return filesystem_read_file(arguments.get("path", ""))
        elif name == "filesystem.write_file":
            return filesystem_write_file(arguments.get("path", ""), arguments.get("content", ""))
        elif name == "filesystem.list_directory":
            return filesystem_list_directory(arguments.get("path", "."))
        elif name == "github.search_repositories":
            return github_search_repositories(arguments.get("query", ""), arguments.get("limit", 3))
        elif name == "github.get_repository":
            return github_get_repository(arguments.get("owner", ""), arguments.get("repo", ""))
        elif name == "github.list_issues":
            return github_list_issues(
                arguments.get("owner", ""),
                arguments.get("repo", ""),
                arguments.get("state", "open"),
                arguments.get("limit", 5),
            )
        elif name == "github.list_user_repositories":
            return github_list_user_repositories(
                arguments.get("username", ""),
                arguments.get("limit", 10),
            )
        elif name == "run_sandboxed_code":
            # Direct safe fallback runner
            from src.mcp_servers.sandbox_server import run_sandboxed_code
            return run_sandboxed_code(
                code=arguments.get("code", ""),
                language=arguments.get("language", "python"),
            )

        raise ValueError(f"No fallback available for tool '{name}'")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        server_name = self._tool_to_server.get(name)
        if server_name is None:
            # Special check for run_sandboxed_code if not in json config
            if name == "run_sandboxed_code":
                return self._fallback(name, arguments)
            raise ValueError(f"No MCP server registered for tool '{name}'")

        server = self._servers[server_name]
        logger.info("mcp_tool_call", extra={"tool": name, "server": server_name, "transport": server.transport})

        # --- Circuit Breaker Fast-Fail Gate ---
        allowed, reason = tool_circuit_breaker.can_execute(server_name)
        if not allowed:
            logger.warning(
                f"[CIRCUIT OPEN FAST-FAIL] Tool '{name}' on server '{server_name}' aborted immediately: {reason}. "
                "Bypassing network/process execution and dispatching directly to local fallback."
            )
            return self._fallback(name, arguments)

        # 1. Persistent stdio Process Pool or MCP SSE Microservice
        if server.transport in ("stdio", "sse"):
            try:
                res = _mcp_pool.call_tool_sync(server, name, arguments)
                tool_circuit_breaker.record_success(server_name)
                return res
            except Exception as exc:
                tier_label = "Persistent stdio Pool" if server.transport == "stdio" else "MCP SSE Microservice"
                tool_circuit_breaker.record_failure(server_name, f"{exc.__class__.__name__}: {exc}")
                logger.warning(
                    f"[{tier_label} Fallback] Server '{server_name}' failed ({exc.__class__.__name__}: {exc}). "
                    f"Executing tool '{name}' via builtin fallback."
                )
                return self._fallback(name, arguments)

        # 2. Legacy HTTP transport
        try:
            with httpx.Client(timeout=server.timeout_seconds) as client:
                resp = client.post(
                    f"{server.base_url}/call",
                    json={"tool": name, "arguments": arguments},
                )
                resp.raise_for_status()
                res = resp.json().get("result", resp.text)
                tool_circuit_breaker.record_success(server_name)
                return res
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.HTTPError) as exc:
            tool_circuit_breaker.record_failure(server_name, f"{exc.__class__.__name__}: {exc}")
            logger.warning(
                f"[MCP HTTP FALLBACK] Server '{server_name}' at {server.base_url} is unreachable ({exc.__class__.__name__}). "
                f"Executing tool '{name}' locally via builtin fallback."
            )
            return self._fallback(name, arguments)


mcp_registry = MCPRegistry()
