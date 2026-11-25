# Hermes Agent Platform

A production-ready, modular harness for running a Hermes-like open-weight
agent: explicit LangGraph orchestration, MCP-based tool access, sandboxed
execution, human-in-the-loop approval for high-risk actions, Mem0-backed
long-term memory, and full observability via OpenTelemetry + Langfuse.

## Quickstart

```bash
cp .env.example .env
make install
make dev
```

Then:
```bash
curl -X POST localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "demo", "message": "What is 4839 * 7213?"}'
```

For the full containerized stack (vLLM + LiteLLM + Redis + Postgres +
Langfuse), see `docs/runbook.md`.

## Project layout

```
hermes-agent-platform/
├── docker-compose.yml / Dockerfile / Makefile / pyproject.toml
├── src/
│   ├── main.py            # FastAPI service (/chat, /memories, /approvals)
│   ├── config.py          # env-driven settings
│   ├── agent/              # LangGraph state machine, state, prompts
│   ├── tools/               # MCP client, Docker sandbox, HITL gateway
│   ├── memory/               # Mem0 manager
│   ├── security/              # input/output guardrails
│   └── utils/                  # logging, OpenTelemetry
├── infrastructure/
│   ├── vllm/serve.sh
│   ├── litellm/config.yaml
│   └── mcp_servers/*.json
├── tests/
│   ├── unit/
│   └── evaluation/          # jailbreak + tool-safety tests, cases.yaml
└── docs/
    ├── architecture.md
    ├── api.md
    └── runbook.md
```

## Design principles
- **Bounded loops**: every agent turn has a hard iteration ceiling
  (`enforce_limit` node); it can never run away.
- **Untrusted-by-default tool output**: content from MCP tools/web scrapes
  is data, never instructions (see `tests/evaluation/cases.yaml`
  `prompt_injection_in_retrieved_content` cases).
- **Approval before consequence**: risk classification happens in the
  harness, not the model — a jailbroken model still can't self-approve a
  destructive call.
- **Memory as facts, not transcripts**: Mem0 extracts durable
  preferences/facts rather than replaying full chat history into context.

See `docs/architecture.md` for the full layer-by-layer breakdown and known
integration points left open (approval-resume checkpointing, MCP transport
choice).
