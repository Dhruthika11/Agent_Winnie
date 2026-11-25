"""
Automated Regression Test Generator and Benchmark Harness for Hermes.
Converts confirmed failures into executable benchmark test cases and evaluates fixes.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional
import uuid

from src.learning.database import failure_db
from src.learning.models import FailureCategory, FailureRecord, RegressionTest

logger = logging.getLogger("hermes.learning.regression")


class EvaluationResult(dict):
    """Result object that behaves both as a dict (passed, errors) and unpacks as a tuple (passed, errors)."""
    def __init__(self, passed: bool, errors: list[str]):
        super().__init__(passed=passed, errors=errors)
        self.passed = passed
        self.errors = errors

    def __iter__(self):
        yield self.passed
        yield self.errors


class RegressionHarness:
    """Maintains and executes the fixed benchmark of historical failure regression cases."""

    def create_test_from_failure(self, failure: FailureRecord) -> RegressionTest:
        """Automatically synthesize a formal regression test case from a failure record."""
        test_id = f"REG-{uuid.uuid4().hex[:6].upper()}"

        # 1. Identify key expected properties
        expected_properties: list[str] = []
        user_words = re.findall(r"\b[A-Za-z0-9\-]{4,}\b", failure.user_input)
        if user_words:
            expected_properties.extend(user_words[:3])

        # 2. Extract negative assertions (hallucinated entities, false claims, error phrases)
        negative_assertions: list[str] = [
            "Inference error:",
            "Internal Server Error",
            "[Unverified link `",
            "Traceback (most recent call last):",
        ]

        # If it was a context-grounding failure with a specific hallucinated entity
        if failure.category == FailureCategory.CONTEXT_GROUNDING_FAILURE:
            # Extract the hallucinated entity from root cause or observed output
            match = re.search(
                r"(?:unrelated entity/repository|hallucinated entity|hallucinated repo)\s*['\"]?([A-Za-z0-9_\-\.]+)",
                failure.root_cause,
                re.IGNORECASE,
            )
            if match:
                hallucinated_entity = match.group(1).strip("'\".,")
                if hallucinated_entity and hallucinated_entity not in negative_assertions:
                    negative_assertions.append(hallucinated_entity)

            # Check for candidate entities/repos in observed output not in user input
            obs_tokens = re.findall(r"\b[A-Za-z0-9_\-\.]{5,}\b", failure.observed_output)
            for token in obs_tokens:
                if "-" in token or "_" in token or "/" in token:
                    clean_t = token.strip(".,'")
                    if clean_t.lower() not in failure.user_input.lower() and clean_t not in negative_assertions:
                        negative_assertions.append(clean_t)

        test = RegressionTest(
            test_id=test_id,
            failure_id=failure.failure_id,
            name=f"Regression test for {failure.category.value}: {failure.user_input[:50]}",
            input_prompt=failure.user_input,
            available_evidence=failure.tool_outputs,
            expected_properties=list(dict.fromkeys(expected_properties)),
            negative_assertions=list(dict.fromkeys(negative_assertions)),
            last_run_status="UNTESTED",
        )

        saved = failure_db.save_regression_test(test)
        logger.info("regression_test_generated_from_failure", extra={"test_id": test_id, "failure_id": failure.failure_id})
        return saved

    def evaluate_output(
        self,
        test: RegressionTest,
        output_text: str,
    ) -> tuple[bool, list[str]]:
        """Evaluate whether a candidate response passes the regression test specifications."""
        issues: list[str] = []
        clean_out = (output_text or "").strip()

        if not clean_out:
            return False, ["Output is completely empty"]

        # 1. Negative assertions: None of the forbidden hallucinated terms may appear
        for forbidden in test.negative_assertions:
            if forbidden.lower() in clean_out.lower():
                issues.append(f"Negative assertion violated: output contains forbidden/hallucinated text '{forbidden}'")

        # 2. Expected properties: Check that relevant subject terms are present
        matched_props = 0
        for prop in test.expected_properties:
            if prop.lower() in clean_out.lower():
                matched_props += 1

        if test.expected_properties and (matched_props / len(test.expected_properties)) < 0.3:
            issues.append(
                f"Insufficient coverage of expected properties: matched {matched_props}/{len(test.expected_properties)} core terms"
            )

        is_passed = len(issues) == 0
        status_str = "PASS" if is_passed else "FAIL"
        err_str = "; ".join(issues) if issues else None
        failure_db.update_regression_test_status(test.test_id, status_str, err_str)

        return EvaluationResult(is_passed, issues)

    def run_benchmark(
        self,
        runner_fn: Optional[Callable[[str, list[dict[str, Any]]], str]] = None,
    ) -> dict[str, Any]:
        """Run all registered benchmark regression tests and compute the overall pass rate."""
        all_tests = failure_db.list_regression_tests(limit=200)
        if not all_tests:
            return {
                "total_tests": 0,
                "passed_count": 0,
                "failed_count": 0,
                "pass_rate": 100.0,
                "results": [],
            }

        passed = 0
        detailed = []

        for test in all_tests:
            if runner_fn is not None:
                try:
                    candidate_reply = runner_fn(test.input_prompt, test.available_evidence)
                    is_ok, errs = self.evaluate_output(test, candidate_reply)
                except Exception as exc:
                    is_ok = False
                    errs = [f"Runner error: {exc}"]
                    failure_db.update_regression_test_status(test.test_id, "FAIL", str(exc))
            else:
                is_ok = test.last_run_status == "PASS"
                errs = [test.last_run_error] if test.last_run_error else []

            if is_ok:
                passed += 1

            detailed.append({
                "test_id": test.test_id,
                "name": test.name,
                "status": "PASS" if is_ok else "FAIL",
                "issues": errs,
            })

        pass_rate = round((passed / len(all_tests)) * 100, 2)
        return {
            "total_tests": len(all_tests),
            "passed_count": passed,
            "failed_count": len(all_tests) - passed,
            "pass_rate": pass_rate,
            "results": detailed,
        }

    def run_test(self, test_id: str, candidate_output: Optional[str] = None) -> dict[str, Any]:
        """Run a single regression test by test_id."""
        for t in failure_db.list_regression_tests():
            if t.test_id == test_id:
                if candidate_output is not None:
                    passed, errors = self.evaluate_output(t, candidate_output)
                    return {"test_id": test_id, "passed": passed, "errors": errors}
                return {"test_id": test_id, "status": t.last_run_status, "error": t.last_run_error}
        return {"test_id": test_id, "status": "NOT_FOUND", "error": "Test not found"}

    # Aliases
    create_regression_test = create_test_from_failure
    evaluate_candidate_output = evaluate_output
    run_benchmark_suite = run_benchmark


regression_harness = RegressionHarness()
