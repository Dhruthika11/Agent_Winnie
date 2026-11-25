# Runbook

## Local development

```bash
cp .env.example .env        # fill in model/API details
make install
make dev                    # uvicorn with --reload on :8080
```

Note: `src/tools/sandbox.py` and the MCP tool servers require a reachable
Docker daemon / running MCP servers respectively. Without them, tool calls
will surface a clear "unavailable" error rather than crashing the agent.

## Full stack (Docker Compose)

```bash
cp .env.example .env
make docker-up               # vLLM, LiteLLM, Redis, Postgres, Langfuse, agent
make logs                    # tail the agent service
make docker-down
```

## Running tests

```bash
make test-unit     # fast, no external services required
make test-eval     # jailbreak + tool-safety evaluation suite
```

For the full `cases.yaml` scenario suite against a live model, run it
through promptfoo or DeepEval, pointing at the `/chat` endpoint.

## Debugging a stuck / looping agent
1. Check `stop_reason` in the `/chat` response — `max_iterations_reached`
   means `enforce_limit` correctly cut off a runaway loop; raise
   `max_iterations` only if the task genuinely needs more steps.
2. Pull the trace in Langfuse/Arize Phoenix by `session_id` to see each
   tool call, its arguments, latency, and result.
3. Check `GET /approvals/{approval_id}` if `awaiting_approval` is `true`
   and nothing seems to be happening — the run is correctly paused on a
   human decision, not hung.

## Rotating credentials
- Model/API keys: `.env` -> restart `litellm` and `agent` services.
- Mem0 vector store, Postgres, Redis: update `.env`, restart affected
  services. No agent code changes required.

## Common failure signatures
| Symptom | Likely cause | Where to look |
|---|---|---|
| All tool calls report "unavailable" | MCP server unreachable | `infrastructure/mcp_servers/*.json` `base_url` |
| Sandbox errors on every code exec | Docker socket not mounted into agent container | `docker-compose.yml` agent service volumes |
| Every high-risk action stuck pending | No one is polling `/approvals` | Wire an ops Slack/webhook notifier off `hitl.request_approval` |
| Memory search always empty | Mem0/vector store misconfigured | `MEM0_VECTOR_STORE_URL` in `.env` |
