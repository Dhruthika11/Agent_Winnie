"""
Comprehensive Regression & Architectural Lifecycle Tests.
Validates:
1. Canonical tool core contracts and standard ToolResult schema.
2. Retryable vs non-retryable error classification.
3. Path traversal jail confinement across filesystem and terminal.
4. Model response normalization and XML tool call parsing.
5. Guaranteed elimination of raw XML in final assistant messages.
6. Unified versioned tool-call persistence schema (schema_version 1).
"""
import pytest

from src.tools.core.contract import error_result, format_for_llm, success_result
from src.tools.core import filesystem, github, terminal, web
from src.agent.tool_parser import (
    is_raw_tool_call,
    parse_model_tool_calls,
    strip_unwanted_tool_xml,
    validate_and_normalize_call,
)
from src.agent.pipeline_common import clean_final_reply, create_tool_call_record


# --------------------------------------------------------------------------- #
# 1. Canonical Tool Contract Tests
# --------------------------------------------------------------------------- #
def test_tool_result_contract_success():
    res = success_result({"count": 42}, "test.tool", custom_field="value")
    assert res["success"] is True
    assert res["data"] == {"count": 42}
    assert res["error"] is None
    assert res["metadata"]["tool_name"] == "test.tool"
    assert res["metadata"]["custom_field"] == "value"

    formatted = format_for_llm(res)
    assert '"count": 42' in formatted


def test_tool_result_contract_error_classification():
    # Retryable error (e.g. rate limit)
    res_retry = error_result("RATE_LIMITED", "Hit rate limit", "test.tool", retryable=True)
    assert res_retry["success"] is False
    assert res_retry["error"]["code"] == "RATE_LIMITED"
    assert res_retry["error"]["retryable"] is True
    assert "(retryable)" in format_for_llm(res_retry)

    # Non-retryable error (e.g. permission denied)
    res_non_retry = error_result("PERMISSION_DENIED", "Cannot access file", "test.tool", retryable=False)
    assert res_non_retry["success"] is False
    assert res_non_retry["error"]["retryable"] is False
    assert "(retryable)" not in format_for_llm(res_non_retry)


# --------------------------------------------------------------------------- #
# 2. Path Traversal & Workspace Confinement
# --------------------------------------------------------------------------- #
def test_filesystem_blocks_path_traversal():
    res = filesystem.read_file("../../secret_passwords.txt")
    assert res["success"] is False
    assert res["error"]["code"] == "PERMISSION_DENIED"


def test_filesystem_safe_read_write(tmp_path):
    # Safe write
    w_res = filesystem.write_file("test_audit_file.txt", "hello audit")
    assert w_res["success"] is True

    # Safe read
    r_res = filesystem.read_file("test_audit_file.txt")
    assert r_res["success"] is True
    assert r_res["data"] == "hello audit"

    # Cleanup
    d_res = terminal.delete_file("test_audit_file.txt")
    assert d_res["success"] is True


# --------------------------------------------------------------------------- #
# 3. GitHub Identifier Parsing & Rate Limit Handling
# --------------------------------------------------------------------------- #
def test_github_identifier_parsing():
    assert github.parse_repo_identifier("", "https://github.com/Srujan0798/Adaptoid-OS?utm_source=chatgpt.com") == ("Srujan0798", "Adaptoid-OS")
    assert github.parse_repo_identifier("octocat", "Hello-World") == ("octocat", "Hello-World")
    assert github.parse_repo_identifier("", "owner/repo") == ("owner", "repo")
    assert github.parse_repo_identifier("owner/repo", "") == ("owner", "repo")


def test_github_missing_arguments_fails_gracefully():
    res = github.get_repository("", "")
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_ARGUMENT"


# --------------------------------------------------------------------------- #
# 4. Model Response Normalizer & XML Tool Parsing
# --------------------------------------------------------------------------- #
def test_xml_tool_call_parsing_and_parameter_normalization():
    model_output = (
        "Let me check that user's repositories for you.\n"
        "<tool_call>\n"
        "<function=github.list_user_repositories>\n"
        "<parameter=username>\nsrujansai\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    clean_text, calls = parse_model_tool_calls(model_output)
    assert len(calls) == 1
    assert calls[0]["name"] == "github.list_user_repositories"
    assert calls[0]["arguments"] == {"username": "srujansai"}
    assert "<tool_call>" not in clean_text
    assert "<function=" not in clean_text
    assert "Let me check that user's repositories for you." in clean_text


def test_is_raw_tool_call_detection():
    assert is_raw_tool_call("<tool_call>\n<function=web_scraper.search>\n<parameter=q>test</parameter>\n</function>\n</tool_call>") is True
    assert is_raw_tool_call("<function=terminal.run>\n<parameter=command>dir</parameter>\n</function>") is True
    assert is_raw_tool_call("Here is the answer: 42") is False


def test_validate_and_normalize_call_filters_unknown_tools():
    allowed = {"web_scraper.search", "github.get_repository"}
    assert validate_and_normalize_call("web_scraper.search", {"q": "ai"}, allowed) is not None
    assert validate_and_normalize_call("unauthorized_malicious_tool", {}, allowed) is None


# --------------------------------------------------------------------------- #
# 5. Clean Final Reply Prevents Raw XML Leakage
# --------------------------------------------------------------------------- #
def test_clean_final_reply_prevents_orphan_parameters():
    # If model outputted raw tool tags and no prose
    raw_xml = "<tool_call>\n<function=github.get_repository>\n<parameter=name>\nAdaptoid-OS\n</parameter>\n</function>\n</tool_call>"
    cleaned = clean_final_reply(raw_xml, [{"tool": "github.get_repository"}])
    assert "<tool_call>" not in cleaned
    assert "Adaptoid-OS" not in cleaned  # Orphan parameter removed
    assert "I completed running the requested tools (`github.get_repository`)" in cleaned


# --------------------------------------------------------------------------- #
# 6. Unified Versioned Persistence Schema
# --------------------------------------------------------------------------- #
def test_create_tool_call_record_versioned_schema():
    rec = create_tool_call_record(
        call_id="call_12345",
        tool_name="github.get_repository",
        arguments={"owner": "Srujan0798", "repo": "Adaptoid-OS"},
        output="Repository details...",
        status="success",
    )
    assert rec["schema_version"] == 1
    assert rec["id"] == "call_12345"
    assert rec["tool"] == "github.get_repository"
    assert rec["arguments"]["owner"] == "Srujan0798"
    assert rec["status"] == "success"
    assert "preview" in rec
    assert "thought" in rec
