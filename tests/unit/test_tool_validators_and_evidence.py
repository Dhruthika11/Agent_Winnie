"""
Unit Test Suite for Tool Output Validators, Central Evidence Store, and Sufficiency Verifier.
"""
import pytest

from src.agent.evidence_store import EvidenceStore
from src.agent.evidence_verifier import central_evidence_verifier
from src.agent.pipeline_common import execute_tool_with_guard
from src.tools.validators import (
    CodeExecutionValidator,
    FilesystemValidator,
    GitHubValidator,
    MathValidator,
    TerminalValidator,
    ValidationStatus,
    WebReaderValidator,
    WebSearchValidator,
    validate_tool_output,
)


class TestIndividualToolValidators:
    def test_web_search_validator_success(self):
        validator = WebSearchValidator()
        args = {"query": "Nvidia Blackwell GPU"}
        raw_output = {
            "success": True,
            "data": [
                {
                    "title": "Nvidia Blackwell Architecture Deep Dive",
                    "url": "https://nvidia.com/blackwell",
                    "snippet": "Nvidia announces the new Blackwell B200 GPU architecture for AI computing in 2024.",
                }
            ],
        }
        res = validator.validate("web_scraper.search", args, raw_output)
        assert res.is_valid is True
        assert res.status == ValidationStatus.VERIFIED
        assert res.metrics["result_count"] == 1
        assert res.metrics["relevance_score"] > 0.5
        assert res.metrics["has_recency_markers"] is True

    def test_web_search_validator_empty_failure(self):
        validator = WebSearchValidator()
        args = {"query": "Unfindable Term 123456789"}
        raw_output = "No verified search results found."
        res = validator.validate("web_scraper.search", args, raw_output)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED
        assert "No search results returned" in res.issues[0]

    def test_web_reader_validator_success_and_failure(self):
        validator = WebReaderValidator()
        # Success case
        res_ok = validator.validate(
            "web_scraper.read_url",
            {"url": "https://example.com"},
            "This is a comprehensive article explaining agent architectures in detail with substantial context.",
        )
        assert res_ok.is_valid is True
        assert res_ok.status == ValidationStatus.VERIFIED

        # 404 Failure case
        res_err = validator.validate(
            "web_scraper.read_url",
            {"url": "https://example.com/404"},
            "Webpage returned 404 Not Found: https://example.com/404",
        )
        assert res_err.is_valid is False
        assert res_err.status == ValidationStatus.FAILED
        assert "404" in res_err.issues[0]

    def test_code_execution_validator_success(self):
        validator = CodeExecutionValidator()
        args = {"language": "python", "code": "print('hello world')"}
        res = validator.validate("run_sandboxed_code", args, "hello world\n")
        assert res.is_valid is True
        assert res.status == ValidationStatus.VERIFIED
        assert res.confidence >= 0.95

    def test_code_execution_validator_runtime_exception(self):
        validator = CodeExecutionValidator()
        args = {"language": "python", "code": "import non_existent_pkg"}
        raw_out = (
            "Traceback (most recent call last):\n"
            "  File '<string>', line 1, in <module>\n"
            "ModuleNotFoundError: No module named 'non_existent_pkg'\n"
        )
        res = validator.validate("run_sandboxed_code", args, raw_out)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED
        assert any("ModuleNotFoundError" in issue for issue in res.issues)

    def test_math_validator_recalculation(self):
        validator = MathValidator()
        args = {"language": "python", "code": "print(4839 * 7213)"}
        expected_product = 4839 * 7213  # 34903707
        res = validator.validate("run_sandboxed_code", args, str(expected_product))
        assert res.is_valid is True
        assert res.status == ValidationStatus.VERIFIED
        assert res.metrics.get("recalculated") == expected_product

    def test_math_validator_discrepancy_and_zero_division(self):
        validator = MathValidator()
        # Discrepancy
        args = {"language": "python", "code": "print(10 + 20)"}
        res_wrong = validator.validate("run_sandboxed_code", args, "999")
        assert res_wrong.is_valid is False
        assert res_wrong.status == ValidationStatus.FLAGGED
        assert "Discrepancy" in res_wrong.issues[0]

        # Division by zero
        args_div0 = {"language": "python", "code": "print(100 / 0)"}
        res_div0 = validator.validate("run_sandboxed_code", args_div0, "ZeroDivisionError")
        assert res_div0.is_valid is False
        assert res_div0.status == ValidationStatus.FAILED

    def test_filesystem_validator(self):
        validator = FilesystemValidator()
        # Read success
        res_read = validator.validate("filesystem.read_file", {"path": "README.md"}, "# Hermes Platform")
        assert res_read.is_valid is True
        assert res_read.status == ValidationStatus.VERIFIED

        # File not found
        res_missing = validator.validate("filesystem.read_file", {"path": "bad.txt"}, "Error: No such file or directory")
        assert res_missing.is_valid is False
        assert res_missing.status == ValidationStatus.FAILED

    def test_terminal_validator(self):
        validator = TerminalValidator()
        res_ok = validator.validate("terminal.run", {"command": "echo 'ok'"}, "ok\n")
        assert res_ok.is_valid is True
        assert res_ok.status == ValidationStatus.VERIFIED

        res_err = validator.validate("terminal.run", {"command": "bad_cmd"}, "Sandbox process exited non-zero (127):\nbash: bad_cmd: command not found")
        assert res_err.is_valid is False
        assert res_err.status == ValidationStatus.FAILED

    def test_github_validator(self):
        validator = GitHubValidator()
        repo_data = {"name": "hermes-agent", "full_name": "nous/hermes-agent", "stargazers_count": 1500}
        res = validator.validate("github.get_repository", {"owner": "nous", "repo": "hermes-agent"}, repo_data)
        assert res.is_valid is True
        assert res.status == ValidationStatus.VERIFIED
        assert res.metrics["stars"] == 1500


class TestCentralEvidenceStoreAndVerifier:
    def test_evidence_store_normalized_format(self):
        store = EvidenceStore()
        _, item = validate_tool_output(
            "web_scraper.search",
            {"query": "LangGraph tutorial"},
            {"success": True, "data": [{"title": "LangGraph", "url": "https://langchain.com", "snippet": "State machine"}]},
        )
        assert item["type"] == "web_search"
        assert item["source"] == "LangGraph tutorial"
        assert item["validation_status"] == "verified"
        assert "is_valid" in item["validation_details"]
        store.add(item)
        assert len(store.get_all()) == 1
        assert len(store.get_verified()) == 1
        assert "https://langchain.com" in store.get_urls()

    def test_central_verifier_identifies_empty_evidence_gaps(self):
        # When all tools fail or return empty
        _, failed_search = validate_tool_output(
            "web_scraper.search",
            {"query": "Ganesh Rapuru LinkedIn"},
            "No verified search results found.",
        )
        report = central_evidence_verifier.verify(
            "Give me the linkedin profile link of Ganesh Rapuru",
            [failed_search],
        )
        assert report.is_sufficient is False
        assert report.verified_count == 0
        assert report.failed_count == 1
        assert "CRITICAL EVIDENCE GAP" in report.synthesis_guidance
        assert "Do NOT guess, fabricate" in report.synthesis_guidance

    def test_central_verifier_detects_missing_profile_gap(self):
        # Search returns general pages, but NO linkedin profile
        _, general_search = validate_tool_output(
            "web_scraper.search",
            {"query": "Ganesh Rapuru profile"},
            {
                "success": True,
                "data": [
                    {"title": "Ganesh Rapuru Papers", "url": "https://scholar.google.com/p", "snippet": "Research papers"},
                ],
            },
        )
        report = central_evidence_verifier.verify(
            "Give me the linkedin profile link of Ganesh Rapuru",
            [general_search],
        )
        assert report.is_sufficient is False
        assert any("LinkedIn" in gap for gap in report.information_gaps)
        assert "INFORMATION GAPS DETECTED" in report.synthesis_guidance

    def test_central_verifier_sufficient_when_evidence_present(self):
        _, verified_search = validate_tool_output(
            "web_scraper.search",
            {"query": "Ganesh Rapuru LinkedIn"},
            {
                "success": True,
                "data": [
                    {"title": "Ganesh Rapuru - LinkedIn", "url": "https://www.linkedin.com/in/ganesh-rapuru-real", "snippet": "Staff AI Engineer at Tech Corp"},
                ],
            },
        )
        report = central_evidence_verifier.verify(
            "Give me the linkedin profile link of Ganesh Rapuru",
            [verified_search],
        )
        assert report.is_sufficient is True
        assert report.verified_count == 1
        assert len(report.information_gaps) == 0

    def test_execute_tool_with_guard_produces_evidence_and_validation(self):
        rec = execute_tool_with_guard(
            "filesystem.read_file",
            {"path": "README.md"},
            call_id="call_test_val_1",
        )
        assert "validation_status" in rec
        assert "validation_details" in rec
        assert rec["evidence_item"] is not None
        assert rec["evidence_item"]["type"] == "filesystem"

