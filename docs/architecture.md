# Architecture

## Overview
This platform wraps an open-weight instruct model with the engineering
"plumbing" required to run it safely and observably in production: an
explicit agent loop, standardized tool access, sandboxed execution,
human-in-the-loop approval for risky actions, persistent long-term memory,
and full-stack observability.

## Layers

### 1. Inference Layer
- **vLLM / SGLang** serves the model (`infrastructure/vllm/serve.sh`) with
  native Hermes-style tool-call parsing enabled.
- **LiteLLM** (`infrastructure/litellm/config.yaml`) sits in front as a
  router: OpenAI-compatible API, retries, fallback models, cost/usage
  logging to Postgres, and OTel span emission per request.

### 2. Orchestration Layer
- **LangGraph** (`src/agent/graph.py`) implements the loop as an explicit
  state machine rather than an unbounded while-loop:
  `load_memory -> call_model -> check_safety -> execute_tools -> ...`
  with a hard `enforce_limit` node so the agent can never run away.

### 3. Execution Layer
- **MCP client** (`src/tools/mcp_client.py`) loads tool servers from JSON
  configs in `infrastructure/mcp_servers/`, keeping new integrations
  decoupled from agent code.
- **Sandbox** (`src/tools/sandbox.py`) runs any agent-generated code inside
  a network-disabled, resource-capped Docker container.
- **HITL gateway** (`src/tools/hitl.py`) classifies every tool call by risk
  (glob-pattern registry + heuristic argument scan) and parks high-risk
  calls in Redis until a human approves or rejects them via the
  `/approvals` API.

### 4. Memory Layer
- **Mem0** (`src/memory/manager.py`) extracts and stores durable facts per
  user, searched fresh each turn and injected into the system prompt
  instead of replaying full history.

### 5. Observability & Security Layer
- **OpenTelemetry** (`src/utils/telemetry.py`) instruments the FastAPI app
  and exports OTLP spans; **Langfuse** or **Arize Phoenix** can consume
  these for visual trace review.
- **Guardrails** (`src/security/`) sanitize input (prompt-injection
  heuristics + optional Guardrails AI validators) and output (PII redaction).

## Request flow

```
User -> FastAPI /chat -> LangGraph.invoke()
          |
          +-> load_memory (Mem0 search)
          +-> call_model (LiteLLM -> vLLM)
          +-> check_safety (HITL risk classification)
          +-> execute_tools (MCP / sandbox) --loops back to call_model--
          +-> update_memory (Mem0 add)
          -> ChatResponse
```

If `check_safety` flags a high-risk call, the graph halts in
`await_approval` and returns `awaiting_approval: true` with an
`approval_id`. A human resolves it via `POST /approvals/{id}`.

## Known integration points left open
- **Resuming a paused run after approval**: wire a LangGraph checkpointer
  (`SqlitePostgresSaver`) keyed on `thread_id`/`session_id` so
  `POST /approvals/{id}` can resume the exact paused graph state instead of
  requiring a fresh `/chat` call.
- **MCP transport**: `mcp_client.py` uses a generic HTTP+JSON transport for
  simplicity; swap in the official `mcp` SDK's stdio/SSE client if your
  tool servers use the standard MCP transport.
