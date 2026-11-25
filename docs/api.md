# API Reference

Base URL (local dev): `http://localhost:8080`

## `POST /chat`
Run one turn of the agent loop.

**Request body**
```json
{
  "user_id": "u-123",
  "session_id": "optional-existing-session",
  "message": "What's the weather in Austin?",
  "max_iterations": 8
}
```

**Response**
```json
{
  "session_id": "u-123-session-abc",
  "reply": "It's 78°F and sunny in Austin right now.",
  "stop_reason": null,
  "awaiting_approval": false,
  "approval_id": null
}
```

If a high-risk tool call was requested, `awaiting_approval` is `true` and
`approval_id` is set; the reply may be empty pending human review.

## `GET /health`
Liveness probe. Returns `{"status": "ok"}`.

## `GET /memories/{user_id}`
List all long-term memories Mem0 has stored for a user.

## `DELETE /memories/{user_id}/{memory_id}`
Delete a specific stored memory. Returns `404` if not found or the memory
backend is unavailable.

## `GET /approvals/{approval_id}`
Inspect a pending or resolved HITL approval record.

## `POST /approvals/{approval_id}`
Approve or reject a pending high-risk tool call. (Requires `operator` or `admin` role when auth is enabled).

**Request body**
```json
{ "approve": true, "decided_by": "alice@company.com" }
```

## `POST /api/auth/token`
Issue a signed JWT access token for authenticated API requests.

**Request body**
```json
{
  "user_id": "u-123",
  "role": "operator",
  "tenant_id": "default"
}
```

## `POST /tasks/submit`
Asynchronously enqueue a long-running workflow into the Redis Streams task queue.

**Request body**
```json
{
  "task_type": "chat",
  "payload": {
    "user_id": "u-123",
    "message": "Scrape github repo and summarize codebase"
  }
}
```

**Response**
```json
{
  "task_id": "hermes-task-1730000000000-abc12345",
  "status": "queued"
}
```

## `GET /tasks/{task_id}`
Check the real-time execution status and result of a queued background task.

## `GET /api/circuit-breakers`
Inspect the live circuit breaker status, failure counts, and state (`CLOSED`, `OPEN`, `HALF_OPEN`) for each MCP server.

## `POST /api/circuit-breakers/{tool}/trip`
Manually trip a tool's circuit breaker to force fast-fail and direct fallback.

## `POST /api/circuit-breakers/{tool}/reset`
Manually reset a tripped circuit breaker back to `CLOSED`.
