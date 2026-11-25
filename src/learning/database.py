"""
PostgreSQL and in-memory persistence layer for the Hermes Self-Improving Failure-Correction Subsystem.
Stores failure records, failure clusters, improvement proposals, regression tests, and versioned deployments.
"""
from __future__ import annotations

import datetime
import json
import logging
import threading
from typing import Any, Optional

from src.config import settings
from src.learning.models import (
    CorrectionType,
    FailureCategory,
    FailureCluster,
    FailureRecord,
    FailureStatus,
    ImprovementProposal,
    ImprovementVersion,
    RegressionTest,
)

logger = logging.getLogger("hermes.learning.db")


class FailureDatabase:
    """PostgreSQL storage manager with thread-safe in-memory fallback for failure intelligence."""

    def __init__(self):
        self._pool = None
        self._is_postgres_ready = False
        self._lock = threading.Lock()

        # In-memory storage structures for fallback and unit tests
        self._mem_failures: dict[str, dict[str, Any]] = {}
        self._mem_clusters: dict[str, dict[str, Any]] = {}
        self._mem_proposals: dict[str, dict[str, Any]] = {}
        self._mem_regression_tests: dict[str, dict[str, Any]] = {}
        self._mem_versions: dict[str, dict[str, Any]] = {}

    def init_db(self) -> None:
        """Initialize PostgreSQL tables for the failure-correction subsystem."""
        try:
            import psycopg
            from psycopg_pool import ConnectionPool

            pg_url = settings.postgres_url
            if not pg_url:
                logger.warning("No POSTGRES_URL configured; using in-memory failure store.")
                return

            self._pool = ConnectionPool(
                conninfo=pg_url,
                min_size=1,
                max_size=10,
                timeout=5.0,
                open=False,
            )
            self._pool.open()

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS hermes_failure_clusters (
                            cluster_id VARCHAR(128) PRIMARY KEY,
                            category VARCHAR(64) NOT NULL,
                            signature VARCHAR(255) UNIQUE NOT NULL,
                            failure_count INT NOT NULL DEFAULT 1,
                            representative_failure_id VARCHAR(128) NOT NULL,
                            description TEXT NOT NULL DEFAULT '',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );

                        CREATE TABLE IF NOT EXISTS hermes_failure_records (
                            failure_id VARCHAR(128) PRIMARY KEY,
                            session_id VARCHAR(128) NOT NULL,
                            user_id VARCHAR(128) NOT NULL,
                            user_input TEXT NOT NULL,
                            observed_output TEXT NOT NULL,
                            expected_behavior TEXT NOT NULL DEFAULT '',
                            category VARCHAR(64) NOT NULL,
                            correction_type VARCHAR(64) NOT NULL,
                            root_cause TEXT NOT NULL DEFAULT '',
                            originating_stage VARCHAR(64) NOT NULL DEFAULT 'synthesis',
                            tool_calls JSONB,
                            tool_outputs JSONB,
                            model_name VARCHAR(128) NOT NULL DEFAULT 'default',
                            is_recurring BOOLEAN NOT NULL DEFAULT FALSE,
                            cluster_id VARCHAR(128) REFERENCES hermes_failure_clusters(cluster_id) ON DELETE SET NULL,
                            status VARCHAR(32) NOT NULL DEFAULT 'new',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_hermes_failures_cat ON hermes_failure_records(category);
                        CREATE INDEX IF NOT EXISTS idx_hermes_failures_status ON hermes_failure_records(status);

                        CREATE TABLE IF NOT EXISTS hermes_improvement_proposals (
                            proposal_id VARCHAR(128) PRIMARY KEY,
                            failure_id VARCHAR(128) NOT NULL REFERENCES hermes_failure_records(failure_id) ON DELETE CASCADE,
                            cluster_id VARCHAR(128) REFERENCES hermes_failure_clusters(cluster_id) ON DELETE SET NULL,
                            correction_type VARCHAR(64) NOT NULL,
                            summary TEXT NOT NULL,
                            proposed_diff TEXT NOT NULL,
                            target_file VARCHAR(255),
                            rationale TEXT NOT NULL,
                            validation_status VARCHAR(32) NOT NULL DEFAULT 'pending',
                            approval_status VARCHAR(32) NOT NULL DEFAULT 'pending',
                            approval_id VARCHAR(128),
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );

                        CREATE TABLE IF NOT EXISTS hermes_regression_tests (
                            test_id VARCHAR(128) PRIMARY KEY,
                            failure_id VARCHAR(128) NOT NULL REFERENCES hermes_failure_records(failure_id) ON DELETE CASCADE,
                            name VARCHAR(255) NOT NULL,
                            input_prompt TEXT NOT NULL,
                            available_evidence JSONB,
                            expected_properties JSONB,
                            negative_assertions JSONB,
                            last_run_status VARCHAR(32) NOT NULL DEFAULT 'UNTESTED',
                            last_run_error TEXT,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );

                        CREATE TABLE IF NOT EXISTS hermes_improvement_versions (
                            version_id VARCHAR(128) PRIMARY KEY,
                            proposal_id VARCHAR(128) NOT NULL REFERENCES hermes_improvement_proposals(proposal_id) ON DELETE CASCADE,
                            deployed_by VARCHAR(128) NOT NULL DEFAULT 'system',
                            diff_applied TEXT NOT NULL,
                            benchmark_pass_rate_before REAL NOT NULL,
                            benchmark_pass_rate_after REAL NOT NULL,
                            deployed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            can_rollback BOOLEAN NOT NULL DEFAULT TRUE
                        );
                    """)
                conn.commit()

            self._is_postgres_ready = True
            logger.info("Hermes Failure & Self-Improvement tables successfully initialized in PostgreSQL.")
        except Exception as exc:
            logger.warning(
                "PostgreSQL failure DB init failed (%s: %s). Using thread-safe in-memory store.",
                exc.__class__.__name__,
                exc,
            )
            self._is_postgres_ready = False

    def close(self) -> None:
        """Safely close PostgreSQL connection pool."""
        if self._pool:
            try:
                self._pool.close()
            except Exception:
                pass
            self._pool = None
            self._is_postgres_ready = False

    # --------------------------------------------------------------------------- #
    # Failure Records
    # --------------------------------------------------------------------------- #
    def record_failure(self, record: FailureRecord) -> FailureRecord:
        """Persist a structured failure record."""
        clean_input = record.user_input.replace("\x00", "")
        clean_obs = record.observed_output.replace("\x00", "")
        clean_exp = record.expected_behavior.replace("\x00", "")
        clean_rc = record.root_cause.replace("\x00", "")

        serialized_tc = json.dumps(record.tool_calls).replace("\x00", "") if record.tool_calls else None
        serialized_to = json.dumps(record.tool_outputs).replace("\x00", "") if record.tool_outputs else None

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO hermes_failure_records (
                                failure_id, session_id, user_id, user_input, observed_output,
                                expected_behavior, category, correction_type, root_cause,
                                originating_stage, tool_calls, tool_outputs, model_name,
                                is_recurring, cluster_id, status, created_at
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s
                            )
                            ON CONFLICT (failure_id) DO UPDATE SET
                                status = EXCLUDED.status,
                                is_recurring = EXCLUDED.is_recurring,
                                cluster_id = EXCLUDED.cluster_id,
                                root_cause = EXCLUDED.root_cause
                            """,
                            (
                                record.failure_id,
                                record.session_id,
                                record.user_id,
                                clean_input,
                                clean_obs,
                                clean_exp,
                                record.category.value,
                                record.correction_type.value,
                                clean_rc,
                                record.originating_stage,
                                serialized_tc,
                                serialized_to,
                                record.model_name,
                                record.is_recurring,
                                record.cluster_id,
                                record.status.value,
                                record.created_at,
                            ),
                        )
                    conn.commit()
            except Exception as exc:
                logger.error("Failed to insert failure record in PostgreSQL: %s", exc)

        with self._lock:
            self._mem_failures[record.failure_id] = record.to_dict()

        return record

    def get_failure(self, failure_id: str) -> Optional[FailureRecord]:
        """Fetch failure record by failure_id."""
        with self._lock:
            rec_dict = self._mem_failures.get(failure_id)
            if rec_dict:
                return FailureRecord(**rec_dict)

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT failure_id, session_id, user_id, user_input, observed_output,
                                   expected_behavior, category, correction_type, root_cause,
                                   originating_stage, tool_calls, tool_outputs, model_name,
                                   is_recurring, cluster_id, status, created_at
                            FROM hermes_failure_records
                            WHERE failure_id = %s
                            """,
                            (failure_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            return FailureRecord(
                                failure_id=row[0],
                                session_id=row[1],
                                user_id=row[2],
                                user_input=row[3],
                                observed_output=row[4],
                                expected_behavior=row[5],
                                category=FailureCategory(row[6]),
                                correction_type=CorrectionType(row[7]),
                                root_cause=row[8],
                                originating_stage=row[9],
                                tool_calls=row[10] or [],
                                tool_outputs=row[11] or [],
                                model_name=row[12],
                                is_recurring=row[13],
                                cluster_id=row[14],
                                status=FailureStatus(row[15]),
                                created_at=row[16],
                            )
            except Exception as exc:
                logger.error("Error fetching failure from PostgreSQL: %s", exc)
        return None

    def list_failures(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> list[FailureRecord]:
        """List failure records with optional filtering."""
        results: list[FailureRecord] = []
        with self._lock:
            for d in self._mem_failures.values():
                if status and d.get("status") != status:
                    continue
                if category and d.get("category") != category:
                    continue
                results.append(FailureRecord(**d))

        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:limit]

    def update_failure_status(
        self,
        failure_id: str,
        status: FailureStatus,
        cluster_id: Optional[str] = None,
        root_cause: Optional[str] = None,
    ) -> bool:
        """Update failure status, cluster, or root cause."""
        updated = False
        with self._lock:
            if failure_id in self._mem_failures:
                self._mem_failures[failure_id]["status"] = status.value
                if cluster_id is not None:
                    self._mem_failures[failure_id]["cluster_id"] = cluster_id
                    self._mem_failures[failure_id]["is_recurring"] = True
                if root_cause is not None:
                    self._mem_failures[failure_id]["root_cause"] = root_cause
                updated = True

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        updates = ["status = %s"]
                        params: list[Any] = [status.value]
                        if cluster_id is not None:
                            updates.append("cluster_id = %s, is_recurring = TRUE")
                            params.append(cluster_id)
                        if root_cause is not None:
                            updates.append("root_cause = %s")
                            params.append(root_cause)
                        params.append(failure_id)
                        query = f"UPDATE hermes_failure_records SET {', '.join(updates)} WHERE failure_id = %s"
                        cur.execute(query, tuple(params))
                        conn.commit()
                        updated = True
            except Exception as exc:
                logger.error("Error updating failure status in PostgreSQL: %s", exc)
        return updated

    # --------------------------------------------------------------------------- #
    # Failure Clusters
    # --------------------------------------------------------------------------- #
    def get_or_create_cluster(
        self,
        category: FailureCategory,
        signature: str,
        rep_failure_id: str,
        description: str = "",
    ) -> FailureCluster:
        """Find existing cluster by signature or create a new cluster."""
        now = datetime.datetime.now(datetime.timezone.utc)
        cluster_id = f"CLUST-{signature[:12]}"

        with self._lock:
            for c_dict in self._mem_clusters.values():
                if c_dict.get("signature") == signature:
                    c_dict["failure_count"] += 1
                    c_dict["updated_at"] = now.isoformat()
                    return FailureCluster(**c_dict)

            new_cluster = FailureCluster(
                cluster_id=cluster_id,
                category=category,
                signature=signature,
                failure_count=1,
                representative_failure_id=rep_failure_id,
                description=description,
                created_at=now,
                updated_at=now,
            )
            self._mem_clusters[cluster_id] = new_cluster.to_dict()

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO hermes_failure_clusters (
                                cluster_id, category, signature, failure_count,
                                representative_failure_id, description, created_at, updated_at
                            ) VALUES (%s, %s, %s, 1, %s, %s, %s, %s)
                            ON CONFLICT (signature) DO UPDATE SET
                                failure_count = hermes_failure_clusters.failure_count + 1,
                                updated_at = EXCLUDED.updated_at
                            RETURNING cluster_id, category, signature, failure_count,
                                      representative_failure_id, description, created_at, updated_at
                            """,
                            (
                                cluster_id,
                                category.value,
                                signature,
                                rep_failure_id,
                                description,
                                now,
                                now,
                            ),
                        )
                        row = cur.fetchone()
                        conn.commit()
                        if row:
                            return FailureCluster(
                                cluster_id=row[0],
                                category=FailureCategory(row[1]),
                                signature=row[2],
                                failure_count=row[3],
                                representative_failure_id=row[4],
                                description=row[5],
                                created_at=row[6],
                                updated_at=row[7],
                            )
            except Exception as exc:
                logger.error("Error creating/updating failure cluster in PostgreSQL: %s", exc)

        return new_cluster

    def list_clusters(self) -> list[FailureCluster]:
        """List all registered failure clusters ordered by recurrence count."""
        clusters = []
        with self._lock:
            for c_dict in self._mem_clusters.values():
                clusters.append(FailureCluster(**c_dict))
        clusters.sort(key=lambda c: c.failure_count, reverse=True)
        return clusters

    # --------------------------------------------------------------------------- #
    # Regression Tests
    # --------------------------------------------------------------------------- #
    def save_regression_test(self, test: RegressionTest) -> RegressionTest:
        """Save an automated benchmark regression test case."""
        serialized_ev = json.dumps(test.available_evidence).replace("\x00", "") if test.available_evidence else None
        serialized_ep = json.dumps(test.expected_properties).replace("\x00", "")
        serialized_na = json.dumps(test.negative_assertions).replace("\x00", "")

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO hermes_regression_tests (
                                test_id, failure_id, name, input_prompt, available_evidence,
                                expected_properties, negative_assertions, last_run_status, last_run_error, created_at
                            ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s)
                            ON CONFLICT (test_id) DO UPDATE SET
                                last_run_status = EXCLUDED.last_run_status,
                                last_run_error = EXCLUDED.last_run_error
                            """,
                            (
                                test.test_id,
                                test.failure_id,
                                test.name,
                                test.input_prompt,
                                serialized_ev,
                                serialized_ep,
                                serialized_na,
                                test.last_run_status,
                                test.last_run_error,
                                test.created_at,
                            ),
                        )
                    conn.commit()
            except Exception as exc:
                logger.error("Error saving regression test to PostgreSQL: %s", exc)

        with self._lock:
            self._mem_regression_tests[test.test_id] = test.to_dict()

        return test

    def list_regression_tests(self, limit: int = 100) -> list[RegressionTest]:
        """List all benchmark regression tests."""
        tests = []
        with self._lock:
            for t_dict in self._mem_regression_tests.values():
                tests.append(RegressionTest(**t_dict))
        return tests[:limit]

    def update_regression_test_status(self, test_id: str, status: str, error: Optional[str] = None) -> bool:
        """Update last run status of a regression test."""
        updated = False
        with self._lock:
            if test_id in self._mem_regression_tests:
                self._mem_regression_tests[test_id]["last_run_status"] = status
                self._mem_regression_tests[test_id]["last_run_error"] = error
                updated = True

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE hermes_regression_tests SET last_run_status = %s, last_run_error = %s WHERE test_id = %s",
                            (status, error, test_id),
                        )
                        conn.commit()
                        updated = True
            except Exception as exc:
                logger.error("Error updating regression test in PostgreSQL: %s", exc)
        return updated

    # --------------------------------------------------------------------------- #
    # Improvement Proposals
    # --------------------------------------------------------------------------- #
    def save_proposal(self, proposal: ImprovementProposal) -> ImprovementProposal:
        """Save a generated improvement proposal."""
        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO hermes_improvement_proposals (
                                proposal_id, failure_id, cluster_id, correction_type, summary,
                                proposed_diff, target_file, rationale, validation_status,
                                approval_status, approval_id, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (proposal_id) DO UPDATE SET
                                validation_status = EXCLUDED.validation_status,
                                approval_status = EXCLUDED.approval_status
                            """,
                            (
                                proposal.proposal_id,
                                proposal.failure_id,
                                proposal.cluster_id,
                                proposal.correction_type.value,
                                proposal.summary,
                                proposal.proposed_diff,
                                proposal.target_file,
                                proposal.rationale,
                                proposal.validation_status,
                                proposal.approval_status,
                                proposal.approval_id,
                                proposal.created_at,
                            ),
                        )
                    conn.commit()
            except Exception as exc:
                logger.error("Error saving proposal in PostgreSQL: %s", exc)

        with self._lock:
            self._mem_proposals[proposal.proposal_id] = proposal.to_dict()

        return proposal

    def get_proposal(self, proposal_id: str) -> Optional[ImprovementProposal]:
        """Fetch proposal by ID."""
        with self._lock:
            p_dict = self._mem_proposals.get(proposal_id)
            if p_dict:
                return ImprovementProposal(**p_dict)
        return None

    def list_proposals(self, status: Optional[str] = None) -> list[ImprovementProposal]:
        """List proposals with optional approval_status filter."""
        proposals = []
        with self._lock:
            for p_dict in self._mem_proposals.values():
                if status and p_dict.get("approval_status") != status:
                    continue
                proposals.append(ImprovementProposal(**p_dict))
        proposals.sort(key=lambda p: p.created_at, reverse=True)
        return proposals

    def update_proposal_status(
        self,
        proposal_id: str,
        approval_status: str,
        validation_status: Optional[str] = None,
    ) -> bool:
        """Update approval and validation status for a proposal."""
        updated = False
        with self._lock:
            if proposal_id in self._mem_proposals:
                self._mem_proposals[proposal_id]["approval_status"] = approval_status
                if validation_status:
                    self._mem_proposals[proposal_id]["validation_status"] = validation_status
                updated = True

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        if validation_status:
                            cur.execute(
                                "UPDATE hermes_improvement_proposals SET approval_status = %s, validation_status = %s WHERE proposal_id = %s",
                                (approval_status, validation_status, proposal_id),
                            )
                        else:
                            cur.execute(
                                "UPDATE hermes_improvement_proposals SET approval_status = %s WHERE proposal_id = %s",
                                (approval_status, proposal_id),
                            )
                        conn.commit()
                        updated = True
            except Exception as exc:
                logger.error("Error updating proposal in PostgreSQL: %s", exc)
        return updated

    # --------------------------------------------------------------------------- #
    # Versions
    # --------------------------------------------------------------------------- #
    def save_version(self, version: ImprovementVersion) -> ImprovementVersion:
        """Save a record of deployed improvement version."""
        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO hermes_improvement_versions (
                                version_id, proposal_id, deployed_by, diff_applied,
                                benchmark_pass_rate_before, benchmark_pass_rate_after,
                                deployed_at, can_rollback
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                version.version_id,
                                version.proposal_id,
                                version.deployed_by,
                                version.diff_applied,
                                version.benchmark_pass_rate_before,
                                version.benchmark_pass_rate_after,
                                version.deployed_at,
                                version.can_rollback,
                            ),
                        )
                    conn.commit()
            except Exception as exc:
                logger.error("Error saving version in PostgreSQL: %s", exc)

        with self._lock:
            self._mem_versions[version.version_id] = version.to_dict()

        return version

    def list_versions(self) -> list[ImprovementVersion]:
        """List deployed improvement versions."""
        versions = []
        with self._lock:
            for v_dict in self._mem_versions.values():
                versions.append(ImprovementVersion(**v_dict))
        versions.sort(key=lambda v: v.deployed_at, reverse=True)
        return versions

    # Aliases
    save_failure = record_failure


failure_db = FailureDatabase()
