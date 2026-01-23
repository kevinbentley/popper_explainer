"""Repository pattern for database operations."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.db.escalation_models import EscalationRunRecord, LawRetestRecord
from src.db.models import (
    AuditLogRecord,
    CaseSetRecord,
    CounterexampleRecord,
    EvaluationRecord,
    LawRecord,
    TheoryRecord,
)

SCHEMA_VERSION = 2


class Repository:
    """Database repository for Popper Explainer persistence."""

    def __init__(self, db_path: str | Path = "popper.db"):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def connect(self, enable_wal: bool = True) -> None:
        """Open database connection and ensure schema exists.

        Args:
            enable_wal: Enable WAL mode for better concurrent access (default True)
        """
        self._conn = sqlite3.connect(self.db_path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        if enable_wal:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
        self._ensure_schema()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Repository":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create schema if it doesn't exist."""
        schema_path = Path(__file__).parent / "schema.sql"
        schema_sql = schema_path.read_text()

        cursor = self.conn.cursor()
        cursor.executescript(schema_sql)

        # Check/set schema version
        cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

        self.conn.commit()

    # --- Law operations ---

    def insert_law(self, law: LawRecord) -> int:
        """Insert a new law record. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO laws (law_id, law_hash, template, law_json)
            VALUES (?, ?, ?, ?)
            """,
            (law.law_id, law.law_hash, law.template, law.law_json),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_law(self, law_id: str) -> LawRecord | None:
        """Get a law by its ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM laws WHERE law_id = ?", (law_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return LawRecord(
            id=row["id"],
            law_id=row["law_id"],
            law_hash=row["law_hash"],
            template=row["template"],
            law_json=row["law_json"],
            created_at=row["created_at"],
        )

    def get_law_by_hash(self, law_hash: str) -> LawRecord | None:
        """Get a law by its content hash (for duplicate detection)."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM laws WHERE law_hash = ?", (law_hash,))
        row = cursor.fetchone()
        if row is None:
            return None
        return LawRecord(
            id=row["id"],
            law_id=row["law_id"],
            law_hash=row["law_hash"],
            template=row["template"],
            law_json=row["law_json"],
            created_at=row["created_at"],
        )

    def list_laws(self, template: str | None = None, limit: int = 100) -> list[LawRecord]:
        """List laws, optionally filtered by template."""
        cursor = self.conn.cursor()
        if template:
            cursor.execute(
                "SELECT * FROM laws WHERE template = ? ORDER BY created_at DESC LIMIT ?",
                (template, limit),
            )
        else:
            cursor.execute("SELECT * FROM laws ORDER BY created_at DESC LIMIT ?", (limit,))
        return [
            LawRecord(
                id=row["id"],
                law_id=row["law_id"],
                law_hash=row["law_hash"],
                template=row["template"],
                law_json=row["law_json"],
                created_at=row["created_at"],
            )
            for row in cursor.fetchall()
        ]

    # --- Case set operations ---

    def insert_case_set(self, case_set: CaseSetRecord) -> int:
        """Insert a new case set. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO case_sets (generator_family, params_hash, seed, cases_json, case_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                case_set.generator_family,
                case_set.params_hash,
                case_set.seed,
                case_set.cases_json,
                case_set.case_count,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_case_set(
        self, generator_family: str, params_hash: str, seed: int
    ) -> CaseSetRecord | None:
        """Get a cached case set by its key."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM case_sets
            WHERE generator_family = ? AND params_hash = ? AND seed = ?
            """,
            (generator_family, params_hash, seed),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return CaseSetRecord(
            id=row["id"],
            generator_family=row["generator_family"],
            params_hash=row["params_hash"],
            seed=row["seed"],
            cases_json=row["cases_json"],
            case_count=row["case_count"],
            created_at=row["created_at"],
        )

    def get_case_set_by_id(self, case_set_id: int) -> CaseSetRecord | None:
        """Get a case set by its database ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM case_sets WHERE id = ?", (case_set_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return CaseSetRecord(
            id=row["id"],
            generator_family=row["generator_family"],
            params_hash=row["params_hash"],
            seed=row["seed"],
            cases_json=row["cases_json"],
            case_count=row["case_count"],
            created_at=row["created_at"],
        )

    # --- Evaluation operations ---

    def insert_evaluation(self, evaluation: EvaluationRecord) -> int:
        """Insert a new evaluation. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO evaluations (
                law_id, law_hash, status, reason_code, case_set_id,
                cases_attempted, cases_used, power_metrics_json, vacuity_json,
                harness_config_hash, sim_hash, runtime_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation.law_id,
                evaluation.law_hash,
                evaluation.status,
                evaluation.reason_code,
                evaluation.case_set_id,
                evaluation.cases_attempted,
                evaluation.cases_used,
                evaluation.power_metrics_json,
                evaluation.vacuity_json,
                evaluation.harness_config_hash,
                evaluation.sim_hash,
                evaluation.runtime_ms,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_evaluation(self, evaluation_id: int) -> EvaluationRecord | None:
        """Get an evaluation by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_evaluation(row)

    def get_latest_evaluation(self, law_id: str) -> EvaluationRecord | None:
        """Get the most recent evaluation for a law."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM evaluations WHERE law_id = ? ORDER BY id DESC LIMIT 1",
            (law_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_evaluation(row)

    def list_evaluations(
        self, status: str | None = None, limit: int = 100
    ) -> list[EvaluationRecord]:
        """List evaluations, optionally filtered by status."""
        cursor = self.conn.cursor()
        if status:
            cursor.execute(
                "SELECT * FROM evaluations WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        return [self._row_to_evaluation(row) for row in cursor.fetchall()]

    def _row_to_evaluation(self, row: sqlite3.Row) -> EvaluationRecord:
        return EvaluationRecord(
            id=row["id"],
            law_id=row["law_id"],
            law_hash=row["law_hash"],
            status=row["status"],
            reason_code=row["reason_code"],
            case_set_id=row["case_set_id"],
            cases_attempted=row["cases_attempted"],
            cases_used=row["cases_used"],
            power_metrics_json=row["power_metrics_json"],
            vacuity_json=row["vacuity_json"],
            harness_config_hash=row["harness_config_hash"],
            sim_hash=row["sim_hash"],
            runtime_ms=row["runtime_ms"],
            created_at=row["created_at"],
        )

    # --- Counterexample operations ---

    def insert_counterexample(self, cx: CounterexampleRecord) -> int:
        """Insert a new counterexample. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO counterexamples (
                evaluation_id, law_id, initial_state, config_json, seed,
                t_max, t_fail, trajectory_excerpt_json, observables_at_fail_json,
                witness_json, minimized
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cx.evaluation_id,
                cx.law_id,
                cx.initial_state,
                cx.config_json,
                cx.seed,
                cx.t_max,
                cx.t_fail,
                cx.trajectory_excerpt_json,
                cx.observables_at_fail_json,
                cx.witness_json,
                cx.minimized,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_counterexamples_for_law(self, law_id: str) -> list[CounterexampleRecord]:
        """Get all counterexamples for a law."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM counterexamples WHERE law_id = ? ORDER BY created_at DESC",
            (law_id,),
        )
        return [
            CounterexampleRecord(
                id=row["id"],
                evaluation_id=row["evaluation_id"],
                law_id=row["law_id"],
                initial_state=row["initial_state"],
                config_json=row["config_json"],
                seed=row["seed"],
                t_max=row["t_max"],
                t_fail=row["t_fail"],
                trajectory_excerpt_json=row["trajectory_excerpt_json"],
                observables_at_fail_json=row["observables_at_fail_json"],
                witness_json=row["witness_json"],
                minimized=bool(row["minimized"]),  # SQLite returns 0/1
                created_at=row["created_at"],
            )
            for row in cursor.fetchall()
        ]

    # --- Theory operations ---

    def insert_theory(self, theory: TheoryRecord) -> int:
        """Insert a new theory. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO theories (theory_id, theory_json, axiom_count, theorem_count)
            VALUES (?, ?, ?, ?)
            """,
            (theory.theory_id, theory.theory_json, theory.axiom_count, theory.theorem_count),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_theory(self, theory_id: str) -> TheoryRecord | None:
        """Get a theory by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM theories WHERE theory_id = ?", (theory_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return TheoryRecord(
            id=row["id"],
            theory_id=row["theory_id"],
            theory_json=row["theory_json"],
            axiom_count=row["axiom_count"],
            theorem_count=row["theorem_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_theory(self, theory: TheoryRecord) -> None:
        """Update an existing theory."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE theories
            SET theory_json = ?, axiom_count = ?, theorem_count = ?, updated_at = CURRENT_TIMESTAMP
            WHERE theory_id = ?
            """,
            (theory.theory_json, theory.axiom_count, theory.theorem_count, theory.theory_id),
        )
        self.conn.commit()

    # --- Audit log operations ---

    def log_audit(
        self,
        operation: str,
        entity_type: str,
        entity_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Record an audit log entry. Returns the new ID."""
        cursor = self.conn.cursor()
        details_json = json.dumps(details) if details else None
        cursor.execute(
            """
            INSERT INTO audit_logs (operation, entity_type, entity_id, details_json)
            VALUES (?, ?, ?, ?)
            """,
            (operation, entity_type, entity_id, details_json),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_audit_logs(
        self,
        operation: str | None = None,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[AuditLogRecord]:
        """Get audit logs with optional filtering."""
        cursor = self.conn.cursor()
        conditions = []
        params: list[Any] = []

        if operation:
            conditions.append("operation = ?")
            params.append(operation)
        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cursor.execute(
            f"SELECT * FROM audit_logs WHERE {where_clause} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        return [
            AuditLogRecord(
                id=row["id"],
                operation=row["operation"],
                entity_type=row["entity_type"],
                entity_id=row["entity_id"],
                details_json=row["details_json"],
                created_at=row["created_at"],
            )
            for row in cursor.fetchall()
        ]

    # --- Summary queries ---

    def get_evaluation_summary(self) -> dict[str, int]:
        """Get counts of evaluations by status."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT status, COUNT(*) as count
            FROM evaluations
            GROUP BY status
            """
        )
        return {row["status"]: row["count"] for row in cursor.fetchall()}

    def get_laws_by_status(self, status: str, limit: int = 50) -> list[str]:
        """Get law IDs that have a specific latest evaluation status."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT e.law_id
            FROM evaluations e
            INNER JOIN (
                SELECT law_id, MAX(created_at) as max_created
                FROM evaluations
                GROUP BY law_id
            ) latest ON e.law_id = latest.law_id AND e.created_at = latest.max_created
            WHERE e.status = ?
            LIMIT ?
            """,
            (status, limit),
        )
        return [row["law_id"] for row in cursor.fetchall()]

    def get_laws_with_status(self, status: str, limit: int = 1000) -> list[tuple[LawRecord, EvaluationRecord]]:
        """Get laws with their latest evaluation for a given status.

        Returns list of (LawRecord, EvaluationRecord) tuples.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT l.*, e.id as eval_id, e.law_hash as eval_law_hash, e.status,
                   e.reason_code, e.case_set_id, e.cases_attempted, e.cases_used,
                   e.power_metrics_json, e.vacuity_json, e.harness_config_hash,
                   e.sim_hash, e.runtime_ms, e.created_at as eval_created_at
            FROM laws l
            INNER JOIN evaluations e ON l.law_id = e.law_id
            INNER JOIN (
                SELECT law_id, MAX(created_at) as max_created
                FROM evaluations
                GROUP BY law_id
            ) latest ON e.law_id = latest.law_id AND e.created_at = latest.max_created
            WHERE e.status = ?
            ORDER BY e.created_at DESC
            LIMIT ?
            """,
            (status, limit),
        )
        results = []
        for row in cursor.fetchall():
            law = LawRecord(
                id=row["id"],
                law_id=row["law_id"],
                law_hash=row["law_hash"],
                template=row["template"],
                law_json=row["law_json"],
                created_at=row["created_at"],
            )
            evaluation = EvaluationRecord(
                id=row["eval_id"],
                law_id=row["law_id"],
                law_hash=row["eval_law_hash"],
                status=row["status"],
                reason_code=row["reason_code"],
                case_set_id=row["case_set_id"],
                cases_attempted=row["cases_attempted"],
                cases_used=row["cases_used"],
                power_metrics_json=row["power_metrics_json"],
                vacuity_json=row["vacuity_json"],
                harness_config_hash=row["harness_config_hash"],
                sim_hash=row["sim_hash"],
                runtime_ms=row["runtime_ms"],
                created_at=row["eval_created_at"],
            )
            results.append((law, evaluation))
        return results

    def law_exists(self, law_hash: str) -> bool:
        """Check if a law with the given content hash already exists."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM laws WHERE law_hash = ? LIMIT 1", (law_hash,))
        return cursor.fetchone() is not None

    def upsert_law(self, law: LawRecord) -> int:
        """Insert or update a law record. Returns the ID."""
        cursor = self.conn.cursor()
        # Try to get existing law
        cursor.execute("SELECT id FROM laws WHERE law_id = ?", (law.law_id,))
        row = cursor.fetchone()
        if row:
            # Update existing
            cursor.execute(
                """
                UPDATE laws SET law_hash = ?, template = ?, law_json = ?
                WHERE law_id = ?
                """,
                (law.law_hash, law.template, law.law_json, law.law_id),
            )
            self.conn.commit()
            return row["id"]
        else:
            # Insert new
            return self.insert_law(law)

    # --- Escalation operations ---

    def insert_escalation_run(self, run: EscalationRunRecord) -> int:
        """Insert a new escalation run. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO escalation_runs (
                level, harness_config_hash, sim_hash, seed,
                laws_tested, stable_count, revoked_count, downgraded_count, runtime_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.level,
                run.harness_config_hash,
                run.sim_hash,
                run.seed,
                run.laws_tested,
                run.stable_count,
                run.revoked_count,
                run.downgraded_count,
                run.runtime_ms,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def update_escalation_run(
        self,
        run_id: int,
        laws_tested: int | None = None,
        stable_count: int | None = None,
        revoked_count: int | None = None,
        downgraded_count: int | None = None,
        runtime_ms: int | None = None,
    ) -> None:
        """Update an escalation run's counts."""
        cursor = self.conn.cursor()
        updates = []
        params: list[Any] = []

        if laws_tested is not None:
            updates.append("laws_tested = ?")
            params.append(laws_tested)
        if stable_count is not None:
            updates.append("stable_count = ?")
            params.append(stable_count)
        if revoked_count is not None:
            updates.append("revoked_count = ?")
            params.append(revoked_count)
        if downgraded_count is not None:
            updates.append("downgraded_count = ?")
            params.append(downgraded_count)
        if runtime_ms is not None:
            updates.append("runtime_ms = ?")
            params.append(runtime_ms)

        if not updates:
            return

        params.append(run_id)
        cursor.execute(
            f"UPDATE escalation_runs SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        self.conn.commit()

    def get_escalation_run(self, run_id: int) -> EscalationRunRecord | None:
        """Get an escalation run by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM escalation_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_escalation_run(row)

    def list_escalation_runs(
        self, level: str | None = None, limit: int = 50
    ) -> list[EscalationRunRecord]:
        """List escalation runs, optionally filtered by level."""
        cursor = self.conn.cursor()
        if level:
            cursor.execute(
                "SELECT * FROM escalation_runs WHERE level = ? ORDER BY created_at DESC LIMIT ?",
                (level, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM escalation_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [self._row_to_escalation_run(row) for row in cursor.fetchall()]

    def _row_to_escalation_run(self, row: sqlite3.Row) -> EscalationRunRecord:
        return EscalationRunRecord(
            id=row["id"],
            level=row["level"],
            harness_config_hash=row["harness_config_hash"],
            sim_hash=row["sim_hash"],
            seed=row["seed"],
            laws_tested=row["laws_tested"],
            stable_count=row["stable_count"],
            revoked_count=row["revoked_count"],
            downgraded_count=row["downgraded_count"],
            runtime_ms=row["runtime_ms"],
            created_at=row["created_at"],
        )

    def insert_law_retest(self, retest: LawRetestRecord) -> int:
        """Insert a new law retest record. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO law_retests (
                escalation_run_id, law_id, old_status, new_status,
                flip_type, evaluation_id, counterexample_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                retest.escalation_run_id,
                retest.law_id,
                retest.old_status,
                retest.new_status,
                retest.flip_type,
                retest.evaluation_id,
                retest.counterexample_id,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_retests_for_run(self, run_id: int) -> list[LawRetestRecord]:
        """Get all law retests for an escalation run."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM law_retests WHERE escalation_run_id = ? ORDER BY id",
            (run_id,),
        )
        return [self._row_to_law_retest(row) for row in cursor.fetchall()]

    def get_retests_by_flip_type(
        self, flip_type: str, limit: int = 100
    ) -> list[tuple[LawRetestRecord, LawRecord]]:
        """Get law retests by flip type with their law records."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT r.*, l.id as law_db_id, l.law_hash, l.template, l.law_json,
                   l.created_at as law_created_at
            FROM law_retests r
            INNER JOIN laws l ON r.law_id = l.law_id
            WHERE r.flip_type = ?
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            (flip_type, limit),
        )
        results = []
        for row in cursor.fetchall():
            retest = self._row_to_law_retest(row)
            law = LawRecord(
                id=row["law_db_id"],
                law_id=row["law_id"],
                law_hash=row["law_hash"],
                template=row["template"],
                law_json=row["law_json"],
                created_at=row["law_created_at"],
            )
            results.append((retest, law))
        return results

    def _row_to_law_retest(self, row: sqlite3.Row) -> LawRetestRecord:
        return LawRetestRecord(
            id=row["id"],
            escalation_run_id=row["escalation_run_id"],
            law_id=row["law_id"],
            old_status=row["old_status"],
            new_status=row["new_status"],
            flip_type=row["flip_type"],
            evaluation_id=row["evaluation_id"],
            counterexample_id=row["counterexample_id"],
            created_at=row["created_at"],
        )

    def get_laws_needing_escalation(
        self, level: str, limit: int = 1000
    ) -> list[tuple[LawRecord, EvaluationRecord]]:
        """Get accepted laws that haven't been tested at the given escalation level.

        Returns laws whose latest evaluation is PASS and that have no retest
        record at the specified escalation level.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT l.*, e.id as eval_id, e.law_hash as eval_law_hash, e.status,
                   e.reason_code, e.case_set_id, e.cases_attempted, e.cases_used,
                   e.power_metrics_json, e.vacuity_json, e.harness_config_hash,
                   e.sim_hash, e.runtime_ms, e.created_at as eval_created_at
            FROM laws l
            INNER JOIN evaluations e ON l.law_id = e.law_id
            INNER JOIN (
                SELECT law_id, MAX(created_at) as max_created
                FROM evaluations
                GROUP BY law_id
            ) latest ON e.law_id = latest.law_id AND e.created_at = latest.max_created
            WHERE e.status = 'PASS'
            AND NOT EXISTS (
                SELECT 1 FROM law_retests r
                INNER JOIN escalation_runs er ON r.escalation_run_id = er.id
                WHERE r.law_id = l.law_id AND er.level = ?
            )
            ORDER BY e.created_at DESC
            LIMIT ?
            """,
            (level, limit),
        )
        results = []
        for row in cursor.fetchall():
            law = LawRecord(
                id=row["id"],
                law_id=row["law_id"],
                law_hash=row["law_hash"],
                template=row["template"],
                law_json=row["law_json"],
                created_at=row["created_at"],
            )
            evaluation = EvaluationRecord(
                id=row["eval_id"],
                law_id=row["law_id"],
                law_hash=row["eval_law_hash"],
                status=row["status"],
                reason_code=row["reason_code"],
                case_set_id=row["case_set_id"],
                cases_attempted=row["cases_attempted"],
                cases_used=row["cases_used"],
                power_metrics_json=row["power_metrics_json"],
                vacuity_json=row["vacuity_json"],
                harness_config_hash=row["harness_config_hash"],
                sim_hash=row["sim_hash"],
                runtime_ms=row["runtime_ms"],
                created_at=row["eval_created_at"],
            )
            results.append((law, evaluation))
        return results
