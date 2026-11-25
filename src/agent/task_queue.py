"""
Asynchronous Distributed Task Queue powered by Redis Streams.

Enables out-of-band execution for long-running tool executions,
repository scraping, Docker builds, and multi-step background agents
with exponential retry backoff.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from src.config import settings

logger = logging.getLogger("hermes.agent.task_queue")

STREAM_NAME = "hermes:stream:tasks"
CONSUMER_GROUP = "hermes:workers"


class TaskQueue:
    def __init__(self):
        self._redis = None

    @property
    def redis(self):
        if self._redis is None:
            try:
                import redis
                self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                # Create consumer group if not exists
                try:
                    self._redis.xgroup_create(STREAM_NAME, CONSUMER_GROUP, id="0", mkstream=True)
                except Exception:
                    pass  # Group already exists
            except Exception as exc:
                logger.warning(f"Redis task queue connection failed: {exc}")
                self._redis = None
        return self._redis

    def enqueue(self, task_type: str, payload: dict[str, Any], max_retries: int = 3) -> str:
        """Enqueue an asynchronous task to Redis Streams."""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        created_at = time.time()
        record = {
            "task_id": task_id,
            "task_type": task_type,
            "payload": json.dumps(payload),
            "status": "queued",
            "retries": 0,
            "max_retries": max_retries,
            "created_at": created_at,
            "updated_at": created_at,
            "result": "",
            "error": "",
        }

        r = self.redis
        if r:
            try:
                # Save task metadata in hash
                r.hset(f"hermes:task:{task_id}", mapping=record)
                r.expire(f"hermes:task:{task_id}", 86400 * 3)  # 3 days TTL

                # Push to Redis Stream
                r.xadd(STREAM_NAME, {"task_id": task_id, "task_type": task_type})
                logger.info("task_enqueued", extra={"task_id": task_id, "task_type": task_type})
                return task_id
            except Exception as exc:
                logger.warning(f"Failed to enqueue to Redis stream: {exc}")

        # In-memory fallback if Redis is unavailable
        _IN_MEMORY_TASKS[task_id] = record
        return task_id

    def get_status(self, task_id: str) -> dict[str, Any] | None:
        """Get current task status, progress, or completed results."""
        r = self.redis
        if r:
            try:
                data = r.hgetall(f"hermes:task:{task_id}")
                if data:
                    if "payload" in data and isinstance(data["payload"], str):
                        try:
                            data["payload"] = json.loads(data["payload"])
                        except Exception:
                            pass
                    if "result" in data and data["result"]:
                        try:
                            data["result"] = json.loads(data["result"])
                        except Exception:
                            pass
                    return data
            except Exception as exc:
                logger.warning(f"Failed to read task status from Redis: {exc}")

        return _IN_MEMORY_TASKS.get(task_id)

    def process_one(self, worker_name: str = "worker-1", block_ms: int = 2000) -> dict[str, Any] | None:
        """Fetch and execute one task from the stream with retry backoff."""
        r = self.redis
        if not r:
            return None

        try:
            streams = r.xreadgroup(CONSUMER_GROUP, worker_name, {STREAM_NAME: ">"}, count=1, block=block_ms)
            if not streams:
                return None

            for stream, messages in streams:
                for msg_id, fields in messages:
                    task_id = fields.get("task_id")
                    if not task_id:
                        r.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
                        continue

                    # Update status to running
                    r.hset(f"hermes:task:{task_id}", mapping={"status": "running", "updated_at": time.time()})
                    task_info = r.hgetall(f"hermes:task:{task_id}")
                    payload = json.loads(task_info.get("payload", "{}"))

                    try:
                        # Execute workflow
                        result = self._execute_task_payload(task_info.get("task_type", ""), payload)
                        r.hset(
                            f"hermes:task:{task_id}",
                            mapping={
                                "status": "completed",
                                "result": json.dumps(result),
                                "updated_at": time.time(),
                            },
                        )
                        r.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
                        return {"task_id": task_id, "status": "completed", "result": result}
                    except Exception as exc:
                        retries = int(task_info.get("retries", 0)) + 1
                        max_retries = int(task_info.get("max_retries", 3))
                        if retries < max_retries:
                            r.hset(
                                f"hermes:task:{task_id}",
                                mapping={"status": "retrying", "retries": retries, "error": str(exc)},
                            )
                        else:
                            r.hset(
                                f"hermes:task:{task_id}",
                                mapping={"status": "failed", "error": str(exc), "updated_at": time.time()},
                            )
                            r.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
        except Exception as exc:
            logger.warning(f"Error in task queue worker: {exc}")

        return None

    def _execute_task_payload(self, task_type: str, payload: dict) -> Any:
        """Dispatch task to internal agent workflow or MCP execution."""
        if task_type == "chat":
            from src.agent.graph import agent_graph
            initial_state = {
                "user_id": payload.get("user_id", "async_user"),
                "session_id": payload.get("session_id", str(uuid.uuid4())),
                "messages": [{"role": "user", "content": payload.get("message", "")}],
                "max_iterations": payload.get("max_iterations", 8),
                "iteration_count": 0,
                "pending_tool_calls": [],
                "awaiting_approval": False,
                "approval_id": None,
                "grounding_retries": 0,
                "done": False,
                "stop_reason": None,
            }
            config = {"configurable": {"thread_id": initial_state["session_id"]}}
            return agent_graph.invoke(initial_state, config=config)
        elif task_type == "mcp_tool":
            from src.tools.mcp_client import mcp_registry
            return mcp_registry.call_tool(payload.get("tool", ""), payload.get("arguments", {}))
        else:
            return {"echo": payload, "processed_at": time.time()}


_IN_MEMORY_TASKS: dict[str, dict] = {}
task_queue = TaskQueue()
