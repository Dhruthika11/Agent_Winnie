"""
Tests for Security Red-Team Attack Vectors.
Validates multi-layered defenses across 8 attack categories:
1. Direct prompt injection
2. Indirect prompt injection (PDFs, Webpages, Tool responses)
3. Tool abuse (unauthorized tools & path traversal)
4. Memory poisoning (adversarial context to alter preferences)
5. Cross-tenant access (tenant isolation bypass)
6. Secret leakage (API keys and credentials)
7. Sandbox escape (OS command execution & host filesystem access)
8. Approval bypass (executing mutating tools without HITL)
"""
from __future__ import annotations

import pytest

from src.security.red_team.attacks import get_all_attack_vectors
from src.security.red_team.harness import red_team_harness
from src.security.red_team.schemas import AttackCategory


def test_red_team_all_attacks_are_blocked_by_concrete_security_boundaries():
    results = red_team_harness.run_all_attacks()
    assert len(results) == 8

    for res in results:
        assert res.blocked is True, f"Attack {res.attack_id} ({res.category}) was NOT blocked!"
        assert res.enforcing_boundary, f"No concrete security boundary recorded for {res.attack_id}"


def test_direct_prompt_injection_blocked_by_input_guards():
    vectors = get_all_attack_vectors()
    vec = next(v for v in vectors if v.category == AttackCategory.PROMPT_INJECTION_DIRECT)
    res = red_team_harness.test_direct_prompt_injection(vec)
    assert res.blocked is True
    assert res.enforcing_boundary == "input_guard"


def test_indirect_prompt_injection_isolated_via_untrusted_wrapping():
    vectors = get_all_attack_vectors()
    vec = next(v for v in vectors if v.category == AttackCategory.PROMPT_INJECTION_INDIRECT)
    res = red_team_harness.test_indirect_prompt_injection(vec)
    assert res.blocked is True
    assert "<untrusted_content" in (res.sanitized_output or "")


def test_tool_abuse_blocked_by_rbac_boundary():
    vectors = get_all_attack_vectors()
    vec = next(v for v in vectors if v.category == AttackCategory.TOOL_ABUSE)
    res = red_team_harness.test_tool_abuse(vec)
    assert res.blocked is True
    assert res.enforcing_boundary == "rbac_engine"


def test_cross_tenant_access_blocked_by_tenant_boundary():
    vectors = get_all_attack_vectors()
    vec = next(v for v in vectors if v.category == AttackCategory.CROSS_TENANT_ACCESS)
    res = red_team_harness.test_cross_tenant_access(vec)
    assert res.blocked is True
    assert res.enforcing_boundary == "tenant_isolation_boundary"


def test_secret_leakage_blocked_by_output_guards():
    vectors = get_all_attack_vectors()
    vec = next(v for v in vectors if v.category == AttackCategory.SECRET_LEAKAGE)
    res = red_team_harness.test_secret_leakage(vec)
    assert res.blocked is True
    assert res.enforcing_boundary == "output_guards_and_secret_redactor"


def test_sandbox_escape_blocked_by_isolation():
    vectors = get_all_attack_vectors()
    vec = next(v for v in vectors if v.category == AttackCategory.SANDBOX_ESCAPE)
    res = red_team_harness.test_sandbox_escape(vec)
    assert res.blocked is True
    assert res.enforcing_boundary == "sandbox_isolation"


def test_approval_bypass_blocked_by_hitl_gate():
    vectors = get_all_attack_vectors()
    vec = next(v for v in vectors if v.category == AttackCategory.APPROVAL_BYPASS)
    res = red_team_harness.test_approval_bypass(vec)
    assert res.blocked is True
    assert res.enforcing_boundary == "hitl_approval_gate"
