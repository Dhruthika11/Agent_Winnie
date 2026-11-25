# Threat Model: Hermes Agent Platform

## 1. Overview & Scope
Hermes is an autonomous agent platform executing tool calls (web search, terminal commands, filesystem reads/writes, code execution in containers, long-term memory).

This threat model defines trust boundaries, attacker capabilities, assets at risk, and security controls across two deployment modes:
- **Local Dev Tool**: Single developer running on a local workstation. Primary risk: prompt injection via untrusted web/repo data leading to local file exfiltration (`.env`, `~/.ssh`) or destructive terminal commands.
- **Multi-User Hosted Service**: Multi-tenant environment serving web clients. Primary risks: container breakout, cross-tenant data leakage, lateral movement, SSRF, credential exfiltration, resource denial-of-service.

---

## 2. Attacker Capabilities & Control Points
An attacker can control or influence the system via:
1. **Untrusted Web Content**: Web pages, search snippets, and scraped articles containing hidden prompt injections (e.g. CSS-hidden text, markdown comments, fake system instructions).
2. **Untrusted Repositories / Files**: Code files, `README.md`, issues, or commit histories containing malicious commands, payloads, or poisoning instructions.
3. **MCP Responses**: Malicious or compromised Model Context Protocol servers returning adversarial tool descriptions or poisoned data.
4. **Adversarial User Input**: Jailbreak prompts, evasion strings (`r\m -rf /`, `$(echo rm)`), and indirect prompt injection vectors.

---

## 3. Assets at Risk
- **Host & Environment Credentials**: API keys (`OPENAI_API_KEY`, `GROQ_API_KEY`, `LITELLM_MASTER_KEY`), SSH keys (`~/.ssh/id_*`), `.env` secrets, database connection strings.
- **Host Integrity**: Host filesystem outside `workspace/`, host processes, system binaries, network interfaces.
- **Tenant Data Isolation**: PostgreSQL checkpointer history, Redis approval tokens, Mem0 long-term memory vectors.
- **User Agency**: Unauthorized payments, unintended git pushes, irreversible file deletions.

---

## 4. Trust Boundaries & Principles
```
[ Untrusted External World ]
   (Web, Repositories, Untrusted MCPs, Raw User Prompts)
                 │
                 ▼  [Boundary 1: Ingestion & Delimitation]
        [ Untrusted Wrapper & Taint Tracking ]
                 │
                 ▼  [Boundary 2: Policy & Authorization Engine]
        [ Tool Policy & HITL Approval Gateway ]
                 │
                 ▼  [Boundary 3: Execution Sandboxes]
   ┌────────────────────────┬────────────────────────┐
   │ Docker/gVisor Container│ Isolated Subprocess    │
   │ (Code, Pytest, Writes) │ (Read-only Allowlist)  │
   └────────────────────────┴────────────────────────┘
                 │
                 ▼  [Boundary 4: Egress & Output Guardrails]
   [ Buffered Streaming Redactor + URL Exfiltration Policy + Audit Log ]
```

1. **Isolation First, String Parsing Second**: Shell parsers (`shlex`) and regexes are secondary defense-in-depth filters. The primary boundary is containerization with dropped capabilities (`ALL`), read-only root, no network by default, and scratch tmpfs.
2. **Data vs. Instruction Separation**: External data from web search, files, and MCP tools is strictly tagged as untrusted data (`is_tainted=True`) and wrapped in delimiters. Untrusted content can never authorize tool execution or alter system policies.
3. **Least Privilege**: Sensitive operations (deletion, external requests, payments) are classified as `IRREVERSIBLE` and require human approval.
4. **Append-Only Auditing**: Every tool invocation, argument payload, policy evaluation, and output is audited with redacted secrets.
