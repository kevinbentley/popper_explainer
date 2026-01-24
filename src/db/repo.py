"""Repository pattern for database operations."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.db.escalation_models import EscalationRunRecord, LawRetestRecord
from src.db.orchestration_models import (
    ExplanationRecord,
    HeldOutSetRecord,
    OrchestrationIterationRecord,
    OrchestrationRunRecord,
    PhaseTransitionRecord,
    PredictionEvaluationRecord,
    PredictionRecord,
    ReadinessSnapshotRecord,
)
from src.db.models import (
    AuditLogRecord,
    CaseSetRecord,
    ClusterArtifactRecord,
    CounterexampleClassRecord,
    CounterexampleFailureKeyRecord,
    CounterexampleRecord,
    EvaluationRecord,
    FailureClassificationRecord,
    FailureClusterRecord,
    FailureKeyRecord,
    FailureKeySnapshotRecord,
    LawNoveltyRecord,
    LawRecord,
    LawWitnessRecord,
    LLMTranscriptRecord,
    NoveltySnapshotRecord,
    ObservableProposalRecord,
    TheoremGenerationArtifactRecord,
    TheoremRecord,
    TheoremRunRecord,
    TheoryRecord,
)

SCHEMA_VERSION = 6  # PHASE-F: Orchestration engine, predictions, held-out sets


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

    def get_counterexample_for_evaluation(
        self, evaluation_id: int
    ) -> CounterexampleRecord | None:
        """Get counterexample for an evaluation by evaluation ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM counterexamples WHERE evaluation_id = ? LIMIT 1",
            (evaluation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return CounterexampleRecord(
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
            minimized=bool(row["minimized"]),
            created_at=row["created_at"],
        )

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

    # --- Failure classification operations ---

    def insert_failure_classification(
        self, classification: FailureClassificationRecord
    ) -> int:
        """Insert a new failure classification. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO failure_classifications (
                evaluation_id, law_id, failure_class, counterexample_class_id,
                is_known_class, confidence, features_json, reasoning, actionable
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                classification.evaluation_id,
                classification.law_id,
                classification.failure_class,
                classification.counterexample_class_id,
                classification.is_known_class,
                classification.confidence,
                classification.features_json,
                classification.reasoning,
                classification.actionable,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_failure_classification(
        self, evaluation_id: int
    ) -> FailureClassificationRecord | None:
        """Get failure classification for an evaluation."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM failure_classifications WHERE evaluation_id = ?",
            (evaluation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_failure_classification(row)

    def list_failure_classifications(
        self,
        failure_class: str | None = None,
        counterexample_class_id: str | None = None,
        actionable: bool | None = None,
        limit: int = 100,
    ) -> list[FailureClassificationRecord]:
        """List failure classifications with optional filtering."""
        cursor = self.conn.cursor()
        conditions = []
        params: list[Any] = []

        if failure_class:
            conditions.append("failure_class = ?")
            params.append(failure_class)
        if counterexample_class_id:
            conditions.append("counterexample_class_id = ?")
            params.append(counterexample_class_id)
        if actionable is not None:
            conditions.append("actionable = ?")
            params.append(actionable)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cursor.execute(
            f"""
            SELECT * FROM failure_classifications
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        )
        return [self._row_to_failure_classification(row) for row in cursor.fetchall()]

    def get_failure_classification_summary(self) -> dict[str, int]:
        """Get counts of failures by classification type."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT failure_class, COUNT(*) as count
            FROM failure_classifications
            GROUP BY failure_class
            """
        )
        return {row["failure_class"]: row["count"] for row in cursor.fetchall()}

    def get_counterexample_class_counts(self) -> dict[str, int]:
        """Get counts of counterexamples by class."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT counterexample_class_id, COUNT(*) as count
            FROM failure_classifications
            WHERE counterexample_class_id IS NOT NULL
            GROUP BY counterexample_class_id
            ORDER BY count DESC
            """
        )
        return {row["counterexample_class_id"]: row["count"] for row in cursor.fetchall()}

    def _row_to_failure_classification(
        self, row: sqlite3.Row
    ) -> FailureClassificationRecord:
        return FailureClassificationRecord(
            id=row["id"],
            evaluation_id=row["evaluation_id"],
            law_id=row["law_id"],
            failure_class=row["failure_class"],
            counterexample_class_id=row["counterexample_class_id"],
            is_known_class=bool(row["is_known_class"]),
            confidence=row["confidence"],
            features_json=row["features_json"],
            reasoning=row["reasoning"],
            actionable=bool(row["actionable"]),
            created_at=row["created_at"],
        )

    # --- Counterexample class registry operations ---

    def upsert_counterexample_class(
        self, class_record: CounterexampleClassRecord
    ) -> int:
        """Insert or update a counterexample class. Returns the ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, occurrence_count FROM counterexample_classes WHERE class_id = ?",
            (class_record.class_id,),
        )
        row = cursor.fetchone()
        if row:
            # Update existing - increment count, update last_seen
            new_count = row["occurrence_count"] + 1
            cursor.execute(
                """
                UPDATE counterexample_classes
                SET occurrence_count = ?, last_seen_at = CURRENT_TIMESTAMP,
                    description = COALESCE(?, description),
                    example_state = COALESCE(?, example_state)
                WHERE class_id = ?
                """,
                (
                    new_count,
                    class_record.description,
                    class_record.example_state,
                    class_record.class_id,
                ),
            )
            self.conn.commit()
            return row["id"]
        else:
            # Insert new
            cursor.execute(
                """
                INSERT INTO counterexample_classes (
                    class_id, description, example_state, occurrence_count
                )
                VALUES (?, ?, ?, 1)
                """,
                (
                    class_record.class_id,
                    class_record.description,
                    class_record.example_state,
                ),
            )
            self.conn.commit()
            return cursor.lastrowid  # type: ignore

    def get_counterexample_class(
        self, class_id: str
    ) -> CounterexampleClassRecord | None:
        """Get a counterexample class by ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM counterexample_classes WHERE class_id = ?",
            (class_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_counterexample_class(row)

    def list_counterexample_classes(
        self, limit: int = 100
    ) -> list[CounterexampleClassRecord]:
        """List all counterexample classes ordered by occurrence count."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM counterexample_classes
            ORDER BY occurrence_count DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._row_to_counterexample_class(row) for row in cursor.fetchall()]

    def get_known_class_ids(self) -> set[str]:
        """Get the set of all known counterexample class IDs."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT class_id FROM counterexample_classes")
        return {row["class_id"] for row in cursor.fetchall()}

    def _row_to_counterexample_class(
        self, row: sqlite3.Row
    ) -> CounterexampleClassRecord:
        return CounterexampleClassRecord(
            id=row["id"],
            class_id=row["class_id"],
            description=row["description"],
            example_state=row["example_state"],
            occurrence_count=row["occurrence_count"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
        )

    # --- Novelty tracking operations ---

    def insert_law_novelty(self, novelty: LawNoveltyRecord) -> int:
        """Insert a law novelty record. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO law_novelty (
                law_id, syntactic_fingerprint, semantic_signature_hash,
                is_syntactically_novel, is_semantically_novel,
                is_novel, is_fully_novel, behavior_summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                novelty.law_id,
                novelty.syntactic_fingerprint,
                novelty.semantic_signature_hash,
                novelty.is_syntactically_novel,
                novelty.is_semantically_novel,
                novelty.is_novel,
                novelty.is_fully_novel,
                novelty.behavior_summary_json,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_law_novelty(self, law_id: str) -> LawNoveltyRecord | None:
        """Get novelty record for a law."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM law_novelty WHERE law_id = ?",
            (law_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_law_novelty(row)

    def get_syntactic_fingerprints(self) -> set[str]:
        """Get all known syntactic fingerprints."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT syntactic_fingerprint FROM law_novelty")
        return {row["syntactic_fingerprint"] for row in cursor.fetchall()}

    def get_semantic_signatures(self) -> set[str]:
        """Get all known semantic signature hashes."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DISTINCT semantic_signature_hash FROM law_novelty "
            "WHERE semantic_signature_hash IS NOT NULL"
        )
        return {row["semantic_signature_hash"] for row in cursor.fetchall()}

    def get_novelty_summary(self) -> dict[str, Any]:
        """Get summary of novelty across all laws."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_laws,
                SUM(CASE WHEN is_syntactically_novel THEN 1 ELSE 0 END) as syntactically_novel,
                SUM(CASE WHEN is_semantically_novel THEN 1 ELSE 0 END) as semantically_novel,
                SUM(CASE WHEN is_novel THEN 1 ELSE 0 END) as novel_either,
                SUM(CASE WHEN is_fully_novel THEN 1 ELSE 0 END) as novel_both
            FROM law_novelty
            """
        )
        row = cursor.fetchone()
        return {
            "total_laws": row["total_laws"] or 0,
            "syntactically_novel": row["syntactically_novel"] or 0,
            "semantically_novel": row["semantically_novel"] or 0,
            "novel_either": row["novel_either"] or 0,
            "novel_both": row["novel_both"] or 0,
        }

    def list_duplicate_laws(self, by: str = "syntactic", limit: int = 100) -> list[tuple[str, list[str]]]:
        """List laws grouped by duplicate fingerprint/signature.

        Args:
            by: "syntactic" or "semantic"
            limit: Maximum groups to return

        Returns:
            List of (fingerprint, [law_ids]) tuples for duplicates
        """
        cursor = self.conn.cursor()
        if by == "syntactic":
            cursor.execute(
                """
                SELECT syntactic_fingerprint, GROUP_CONCAT(law_id) as law_ids
                FROM law_novelty
                GROUP BY syntactic_fingerprint
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
                LIMIT ?
                """,
                (limit,),
            )
        else:
            cursor.execute(
                """
                SELECT semantic_signature_hash, GROUP_CONCAT(law_id) as law_ids
                FROM law_novelty
                WHERE semantic_signature_hash IS NOT NULL
                GROUP BY semantic_signature_hash
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
                LIMIT ?
                """,
                (limit,),
            )

        return [
            (row[0], row[1].split(",")) for row in cursor.fetchall()
        ]

    def _row_to_law_novelty(self, row: sqlite3.Row) -> LawNoveltyRecord:
        return LawNoveltyRecord(
            id=row["id"],
            law_id=row["law_id"],
            syntactic_fingerprint=row["syntactic_fingerprint"],
            semantic_signature_hash=row["semantic_signature_hash"],
            is_syntactically_novel=bool(row["is_syntactically_novel"]),
            is_semantically_novel=bool(row["is_semantically_novel"]),
            is_novel=bool(row["is_novel"]),
            is_fully_novel=bool(row["is_fully_novel"]),
            behavior_summary_json=row["behavior_summary_json"],
            created_at=row["created_at"],
        )

    # --- Novelty snapshot operations ---

    def insert_novelty_snapshot(self, snapshot: NoveltySnapshotRecord) -> int:
        """Insert a novelty snapshot. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO novelty_snapshots (
                window_size, total_laws_in_window,
                syntactically_novel_count, semantically_novel_count,
                fully_novel_count, syntactic_novelty_rate,
                semantic_novelty_rate, combined_novelty_rate,
                is_saturated, total_laws_seen,
                unique_syntactic_fingerprints, unique_semantic_signatures
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.window_size,
                snapshot.total_laws_in_window,
                snapshot.syntactically_novel_count,
                snapshot.semantically_novel_count,
                snapshot.fully_novel_count,
                snapshot.syntactic_novelty_rate,
                snapshot.semantic_novelty_rate,
                snapshot.combined_novelty_rate,
                snapshot.is_saturated,
                snapshot.total_laws_seen,
                snapshot.unique_syntactic_fingerprints,
                snapshot.unique_semantic_signatures,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_latest_novelty_snapshot(self) -> NoveltySnapshotRecord | None:
        """Get the most recent novelty snapshot."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM novelty_snapshots ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_novelty_snapshot(row)

    def list_novelty_snapshots(self, limit: int = 100) -> list[NoveltySnapshotRecord]:
        """List novelty snapshots in reverse chronological order."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM novelty_snapshots ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_novelty_snapshot(row) for row in cursor.fetchall()]

    def get_saturation_history(self) -> list[tuple[str, bool, float]]:
        """Get history of saturation status.

        Returns:
            List of (timestamp, is_saturated, combined_novelty_rate) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT created_at, is_saturated, combined_novelty_rate
            FROM novelty_snapshots
            ORDER BY created_at
            """
        )
        return [
            (row["created_at"], bool(row["is_saturated"]), row["combined_novelty_rate"])
            for row in cursor.fetchall()
        ]

    def _row_to_novelty_snapshot(self, row: sqlite3.Row) -> NoveltySnapshotRecord:
        return NoveltySnapshotRecord(
            id=row["id"],
            window_size=row["window_size"],
            total_laws_in_window=row["total_laws_in_window"],
            syntactically_novel_count=row["syntactically_novel_count"],
            semantically_novel_count=row["semantically_novel_count"],
            fully_novel_count=row["fully_novel_count"],
            syntactic_novelty_rate=row["syntactic_novelty_rate"],
            semantic_novelty_rate=row["semantic_novelty_rate"],
            combined_novelty_rate=row["combined_novelty_rate"],
            is_saturated=bool(row["is_saturated"]),
            total_laws_seen=row["total_laws_seen"],
            unique_syntactic_fingerprints=row["unique_syntactic_fingerprints"],
            unique_semantic_signatures=row["unique_semantic_signatures"],
            created_at=row["created_at"],
        )

    # --- Failure key operations ---

    def upsert_failure_key(self, key: FailureKeyRecord) -> int:
        """Insert or update a failure key. Returns the ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, occurrence_count FROM failure_keys WHERE key_hash = ?",
            (key.key_hash,),
        )
        row = cursor.fetchone()
        if row:
            # Update existing - increment count, update last_seen
            new_count = row["occurrence_count"] + 1
            cursor.execute(
                """
                UPDATE failure_keys
                SET occurrence_count = ?, last_seen_at = CURRENT_TIMESTAMP
                WHERE key_hash = ?
                """,
                (new_count, key.key_hash),
            )
            self.conn.commit()
            return row["id"]
        else:
            # Insert new
            cursor.execute(
                """
                INSERT INTO failure_keys (
                    key_hash, canonical_initial, canonical_fail_state,
                    t_fail_relative, observable_signature_json,
                    trajectory_signature, occurrence_count
                )
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    key.key_hash,
                    key.canonical_initial,
                    key.canonical_fail_state,
                    key.t_fail_relative,
                    key.observable_signature_json,
                    key.trajectory_signature,
                ),
            )
            self.conn.commit()
            return cursor.lastrowid  # type: ignore

    def get_failure_key(self, key_hash: str) -> FailureKeyRecord | None:
        """Get a failure key by its hash."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM failure_keys WHERE key_hash = ?",
            (key_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_failure_key(row)

    def get_failure_key_by_id(self, key_id: int) -> FailureKeyRecord | None:
        """Get a failure key by its database ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM failure_keys WHERE id = ?",
            (key_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_failure_key(row)

    def list_failure_keys(
        self,
        canonical_initial: str | None = None,
        limit: int = 100,
    ) -> list[FailureKeyRecord]:
        """List failure keys, optionally filtered by canonical initial state."""
        cursor = self.conn.cursor()
        if canonical_initial:
            cursor.execute(
                """
                SELECT * FROM failure_keys
                WHERE canonical_initial = ?
                ORDER BY occurrence_count DESC
                LIMIT ?
                """,
                (canonical_initial, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM failure_keys
                ORDER BY occurrence_count DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [self._row_to_failure_key(row) for row in cursor.fetchall()]

    def get_top_failure_patterns(self, limit: int = 10) -> list[FailureKeyRecord]:
        """Get the most common failure patterns."""
        return self.list_failure_keys(limit=limit)

    def get_failure_key_counts(self) -> dict[str, int]:
        """Get occurrence counts for all failure keys."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT key_hash, occurrence_count FROM failure_keys")
        return {row["key_hash"]: row["occurrence_count"] for row in cursor.fetchall()}

    def _row_to_failure_key(self, row: sqlite3.Row) -> FailureKeyRecord:
        return FailureKeyRecord(
            id=row["id"],
            key_hash=row["key_hash"],
            canonical_initial=row["canonical_initial"],
            canonical_fail_state=row["canonical_fail_state"],
            t_fail_relative=row["t_fail_relative"],
            observable_signature_json=row["observable_signature_json"],
            trajectory_signature=row["trajectory_signature"],
            occurrence_count=row["occurrence_count"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
        )

    # --- Counterexample failure key link operations ---

    def insert_counterexample_failure_key(
        self,
        link: CounterexampleFailureKeyRecord,
    ) -> int:
        """Insert a counterexample-failure key link. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO counterexample_failure_keys (
                counterexample_id, failure_key_id, law_id, is_novel
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                link.counterexample_id,
                link.failure_key_id,
                link.law_id,
                link.is_novel,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_failure_keys_for_counterexample(
        self,
        counterexample_id: int,
    ) -> list[FailureKeyRecord]:
        """Get all failure keys for a counterexample."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT fk.* FROM failure_keys fk
            INNER JOIN counterexample_failure_keys cfk ON fk.id = cfk.failure_key_id
            WHERE cfk.counterexample_id = ?
            """,
            (counterexample_id,),
        )
        return [self._row_to_failure_key(row) for row in cursor.fetchall()]

    def get_counterexamples_for_failure_key(
        self,
        failure_key_id: int,
    ) -> list[tuple[CounterexampleFailureKeyRecord, str]]:
        """Get all counterexample links for a failure key.

        Returns:
            List of (link_record, law_id) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM counterexample_failure_keys
            WHERE failure_key_id = ?
            ORDER BY created_at DESC
            """,
            (failure_key_id,),
        )
        return [
            (
                CounterexampleFailureKeyRecord(
                    id=row["id"],
                    counterexample_id=row["counterexample_id"],
                    failure_key_id=row["failure_key_id"],
                    law_id=row["law_id"],
                    is_novel=bool(row["is_novel"]),
                    created_at=row["created_at"],
                ),
                row["law_id"],
            )
            for row in cursor.fetchall()
        ]

    def get_failure_key_summary(self) -> dict[str, Any]:
        """Get summary of failure keys."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_keys,
                SUM(occurrence_count) as total_occurrences,
                AVG(occurrence_count) as avg_occurrences,
                MAX(occurrence_count) as max_occurrences
            FROM failure_keys
            """
        )
        row = cursor.fetchone()
        return {
            "total_keys": row["total_keys"] or 0,
            "total_occurrences": row["total_occurrences"] or 0,
            "avg_occurrences": row["avg_occurrences"] or 0,
            "max_occurrences": row["max_occurrences"] or 0,
        }

    # --- Failure key snapshot operations ---

    def insert_failure_key_snapshot(
        self,
        snapshot: FailureKeySnapshotRecord,
    ) -> int:
        """Insert a failure key snapshot. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO failure_key_snapshots (
                window_size, total_falsifications, unique_failure_keys,
                new_cex_rate, repetition_rate, total_keys_seen
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.window_size,
                snapshot.total_falsifications,
                snapshot.unique_failure_keys,
                snapshot.new_cex_rate,
                snapshot.repetition_rate,
                snapshot.total_keys_seen,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_latest_failure_key_snapshot(
        self,
    ) -> FailureKeySnapshotRecord | None:
        """Get the most recent failure key snapshot."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM failure_key_snapshots ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_failure_key_snapshot(row)

    def list_failure_key_snapshots(
        self,
        limit: int = 100,
    ) -> list[FailureKeySnapshotRecord]:
        """List failure key snapshots in reverse chronological order."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM failure_key_snapshots ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_failure_key_snapshot(row) for row in cursor.fetchall()]

    def get_new_cex_rate_history(self) -> list[tuple[str, float]]:
        """Get history of new counterexample rates.

        Returns:
            List of (timestamp, new_cex_rate) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT created_at, new_cex_rate
            FROM failure_key_snapshots
            ORDER BY created_at
            """
        )
        return [(row["created_at"], row["new_cex_rate"]) for row in cursor.fetchall()]

    def _row_to_failure_key_snapshot(
        self,
        row: sqlite3.Row,
    ) -> FailureKeySnapshotRecord:
        return FailureKeySnapshotRecord(
            id=row["id"],
            window_size=row["window_size"],
            total_falsifications=row["total_falsifications"],
            unique_failure_keys=row["unique_failure_keys"],
            new_cex_rate=row["new_cex_rate"],
            repetition_rate=row["repetition_rate"],
            total_keys_seen=row["total_keys_seen"],
            created_at=row["created_at"],
        )

    # --- Theorem generation artifact operations (PHASE-D) ---

    def insert_theorem_generation_artifact(
        self, artifact: TheoremGenerationArtifactRecord
    ) -> int:
        """Insert a new theorem generation artifact. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO theorem_generation_artifacts (
                artifact_hash, snapshot_hash, prompt_template_version,
                model_name, model_params_json, raw_response, parsed_response_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.artifact_hash,
                artifact.snapshot_hash,
                artifact.prompt_template_version,
                artifact.model_name,
                artifact.model_params_json,
                artifact.raw_response,
                artifact.parsed_response_json,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_artifact_by_hash(
        self, artifact_hash: str
    ) -> TheoremGenerationArtifactRecord | None:
        """Get a theorem generation artifact by its hash."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM theorem_generation_artifacts WHERE artifact_hash = ?",
            (artifact_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_artifact(row)

    def get_artifact_by_id(
        self, artifact_id: int
    ) -> TheoremGenerationArtifactRecord | None:
        """Get a theorem generation artifact by its database ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM theorem_generation_artifacts WHERE id = ?",
            (artifact_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_artifact(row)

    def get_artifacts_by_snapshot_hash(
        self, snapshot_hash: str, limit: int = 100
    ) -> list[TheoremGenerationArtifactRecord]:
        """Get all artifacts for a given snapshot hash."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM theorem_generation_artifacts
            WHERE snapshot_hash = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (snapshot_hash, limit),
        )
        return [self._row_to_artifact(row) for row in cursor.fetchall()]

    def _row_to_artifact(
        self, row: sqlite3.Row
    ) -> TheoremGenerationArtifactRecord:
        return TheoremGenerationArtifactRecord(
            id=row["id"],
            artifact_hash=row["artifact_hash"],
            snapshot_hash=row["snapshot_hash"],
            prompt_template_version=row["prompt_template_version"],
            model_name=row["model_name"],
            model_params_json=row["model_params_json"],
            raw_response=row["raw_response"],
            parsed_response_json=row["parsed_response_json"],
            created_at=row["created_at"],
        )

    # --- Theorem run operations ---

    def insert_theorem_run(self, run: TheoremRunRecord) -> int:
        """Insert a new theorem run. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO theorem_runs (
                run_id, status, config_json, prompt_hash,
                pass_laws_count, fail_laws_count,
                theorems_generated, clusters_found, observable_proposals,
                artifact_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.status,
                run.config_json,
                run.prompt_hash,
                run.pass_laws_count,
                run.fail_laws_count,
                run.theorems_generated,
                run.clusters_found,
                run.observable_proposals,
                run.artifact_id,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def update_theorem_run(
        self,
        run_id: str,
        status: str | None = None,
        theorems_generated: int | None = None,
        clusters_found: int | None = None,
        observable_proposals: int | None = None,
        prompt_hash: str | None = None,
        artifact_id: int | None = None,
        completed_at: str | None = None,
    ) -> None:
        """Update a theorem run."""
        cursor = self.conn.cursor()
        updates = []
        params: list[Any] = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if theorems_generated is not None:
            updates.append("theorems_generated = ?")
            params.append(theorems_generated)
        if clusters_found is not None:
            updates.append("clusters_found = ?")
            params.append(clusters_found)
        if observable_proposals is not None:
            updates.append("observable_proposals = ?")
            params.append(observable_proposals)
        if prompt_hash is not None:
            updates.append("prompt_hash = ?")
            params.append(prompt_hash)
        if artifact_id is not None:
            updates.append("artifact_id = ?")
            params.append(artifact_id)
        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at)

        if not updates:
            return

        params.append(run_id)
        cursor.execute(
            f"UPDATE theorem_runs SET {', '.join(updates)} WHERE run_id = ?",
            params,
        )
        self.conn.commit()

    def get_theorem_run(self, run_id: str) -> TheoremRunRecord | None:
        """Get a theorem run by run_id."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM theorem_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_theorem_run(row)

    def get_theorem_run_by_db_id(self, db_id: int) -> TheoremRunRecord | None:
        """Get a theorem run by database ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM theorem_runs WHERE id = ?", (db_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_theorem_run(row)

    def list_theorem_runs(
        self, status: str | None = None, limit: int = 50
    ) -> list[TheoremRunRecord]:
        """List theorem runs, optionally filtered by status."""
        cursor = self.conn.cursor()
        if status:
            cursor.execute(
                "SELECT * FROM theorem_runs WHERE status = ? ORDER BY started_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM theorem_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
        return [self._row_to_theorem_run(row) for row in cursor.fetchall()]

    def _row_to_theorem_run(self, row: sqlite3.Row) -> TheoremRunRecord:
        keys = row.keys()
        return TheoremRunRecord(
            id=row["id"],
            run_id=row["run_id"],
            status=row["status"],
            config_json=row["config_json"],
            prompt_hash=row["prompt_hash"],
            pass_laws_count=row["pass_laws_count"],
            fail_laws_count=row["fail_laws_count"],
            theorems_generated=row["theorems_generated"],
            clusters_found=row["clusters_found"],
            observable_proposals=row["observable_proposals"],
            artifact_id=row["artifact_id"] if "artifact_id" in keys else None,
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    # --- Theorem operations ---

    def insert_theorem(self, theorem: TheoremRecord) -> int:
        """Insert a new theorem. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO theorems (
                theorem_run_id, theorem_id, name, status, claim,
                support_json, failure_modes_json, missing_structure_json,
                typed_missing_structure_json, failure_signature_text,
                failure_signature_hash, role_coded_signature, bucket_tags_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                theorem.theorem_run_id,
                theorem.theorem_id,
                theorem.name,
                theorem.status,
                theorem.claim,
                theorem.support_json,
                theorem.failure_modes_json,
                theorem.missing_structure_json,
                theorem.typed_missing_structure_json,
                theorem.failure_signature_text,
                theorem.failure_signature_hash,
                theorem.role_coded_signature,
                theorem.bucket_tags_json,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_theorem(self, theorem_id: str) -> TheoremRecord | None:
        """Get a theorem by theorem_id."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM theorems WHERE theorem_id = ?", (theorem_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_theorem(row)

    def list_theorems(
        self,
        run_id: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[TheoremRecord]:
        """List theorems with optional filtering."""
        cursor = self.conn.cursor()
        conditions = []
        params: list[Any] = []

        if run_id is not None:
            conditions.append("theorem_run_id = ?")
            params.append(run_id)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cursor.execute(
            f"""
            SELECT * FROM theorems
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        )
        return [self._row_to_theorem(row) for row in cursor.fetchall()]

    def get_theorems_by_run(self, run_id: int) -> list[TheoremRecord]:
        """Get all theorems for a theorem run."""
        return self.list_theorems(run_id=run_id, limit=1000)

    def _row_to_theorem(self, row: sqlite3.Row) -> TheoremRecord:
        keys = row.keys()
        return TheoremRecord(
            id=row["id"],
            theorem_run_id=row["theorem_run_id"],
            theorem_id=row["theorem_id"],
            name=row["name"],
            status=row["status"],
            claim=row["claim"],
            support_json=row["support_json"],
            failure_modes_json=row["failure_modes_json"],
            missing_structure_json=row["missing_structure_json"],
            typed_missing_structure_json=row["typed_missing_structure_json"] if "typed_missing_structure_json" in keys else None,
            failure_signature_text=row["failure_signature_text"],
            failure_signature_hash=row["failure_signature_hash"],
            role_coded_signature=row["role_coded_signature"] if "role_coded_signature" in keys else None,
            bucket_tags_json=row["bucket_tags_json"] if "bucket_tags_json" in keys else None,
            created_at=row["created_at"],
        )

    # --- Failure cluster operations ---

    def insert_failure_cluster(self, cluster: FailureClusterRecord) -> int:
        """Insert a new failure cluster. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO failure_clusters (
                theorem_run_id, cluster_id, bucket, bucket_tags_json, semantic_cluster_idx,
                theorem_ids_json, cluster_size, centroid_signature, avg_similarity,
                top_keywords_json, recommended_action, distance_threshold
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cluster.theorem_run_id,
                cluster.cluster_id,
                cluster.bucket,
                cluster.bucket_tags_json,
                cluster.semantic_cluster_idx,
                cluster.theorem_ids_json,
                cluster.cluster_size,
                cluster.centroid_signature,
                cluster.avg_similarity,
                cluster.top_keywords_json,
                cluster.recommended_action,
                cluster.distance_threshold,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_failure_cluster(self, cluster_id: str) -> FailureClusterRecord | None:
        """Get a failure cluster by cluster_id."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM failure_clusters WHERE cluster_id = ?",
            (cluster_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_failure_cluster(row)

    def list_failure_clusters(
        self,
        run_id: int | None = None,
        bucket: str | None = None,
        limit: int = 100,
    ) -> list[FailureClusterRecord]:
        """List failure clusters with optional filtering."""
        cursor = self.conn.cursor()
        conditions = []
        params: list[Any] = []

        if run_id is not None:
            conditions.append("theorem_run_id = ?")
            params.append(run_id)
        if bucket:
            conditions.append("bucket = ?")
            params.append(bucket)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cursor.execute(
            f"""
            SELECT * FROM failure_clusters
            WHERE {where_clause}
            ORDER BY cluster_size DESC
            LIMIT ?
            """,
            params,
        )
        return [self._row_to_failure_cluster(row) for row in cursor.fetchall()]

    def get_clusters_by_run(self, run_id: int) -> list[FailureClusterRecord]:
        """Get all failure clusters for a theorem run."""
        return self.list_failure_clusters(run_id=run_id, limit=1000)

    def _row_to_failure_cluster(self, row: sqlite3.Row) -> FailureClusterRecord:
        keys = row.keys()
        return FailureClusterRecord(
            id=row["id"],
            theorem_run_id=row["theorem_run_id"],
            cluster_id=row["cluster_id"],
            bucket=row["bucket"],
            semantic_cluster_idx=row["semantic_cluster_idx"],
            theorem_ids_json=row["theorem_ids_json"],
            cluster_size=row["cluster_size"],
            centroid_signature=row["centroid_signature"],
            avg_similarity=row["avg_similarity"],
            bucket_tags_json=row["bucket_tags_json"] if "bucket_tags_json" in keys else None,
            top_keywords_json=row["top_keywords_json"] if "top_keywords_json" in keys else None,
            recommended_action=row["recommended_action"] if "recommended_action" in keys else None,
            distance_threshold=row["distance_threshold"] if "distance_threshold" in keys else None,
            created_at=row["created_at"],
        )

    # --- Observable proposal operations ---

    def insert_observable_proposal(self, proposal: ObservableProposalRecord) -> int:
        """Insert a new observable proposal. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO observable_proposals (
                theorem_run_id, cluster_id, proposal_id,
                observable_name, observable_expr, rationale, priority, action_type, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposal.theorem_run_id,
                proposal.cluster_id,
                proposal.proposal_id,
                proposal.observable_name,
                proposal.observable_expr,
                proposal.rationale,
                proposal.priority,
                proposal.action_type,
                proposal.status,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_observable_proposal(
        self, proposal_id: str
    ) -> ObservableProposalRecord | None:
        """Get an observable proposal by proposal_id."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM observable_proposals WHERE proposal_id = ?",
            (proposal_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_observable_proposal(row)

    def list_observable_proposals(
        self,
        run_id: int | None = None,
        cluster_id: str | None = None,
        priority: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[ObservableProposalRecord]:
        """List observable proposals with optional filtering."""
        cursor = self.conn.cursor()
        conditions = []
        params: list[Any] = []

        if run_id is not None:
            conditions.append("theorem_run_id = ?")
            params.append(run_id)
        if cluster_id:
            conditions.append("cluster_id = ?")
            params.append(cluster_id)
        if priority:
            conditions.append("priority = ?")
            params.append(priority)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cursor.execute(
            f"""
            SELECT * FROM observable_proposals
            WHERE {where_clause}
            ORDER BY
                CASE priority
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'low' THEN 3
                END,
                created_at DESC
            LIMIT ?
            """,
            params,
        )
        return [self._row_to_observable_proposal(row) for row in cursor.fetchall()]

    def get_proposals_by_run(self, run_id: int) -> list[ObservableProposalRecord]:
        """Get all observable proposals for a theorem run."""
        return self.list_observable_proposals(run_id=run_id, limit=1000)

    def update_observable_proposal_status(
        self, proposal_id: str, status: str
    ) -> None:
        """Update the status of an observable proposal."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE observable_proposals SET status = ? WHERE proposal_id = ?",
            (status, proposal_id),
        )
        self.conn.commit()

    def _row_to_observable_proposal(
        self, row: sqlite3.Row
    ) -> ObservableProposalRecord:
        keys = row.keys()
        return ObservableProposalRecord(
            id=row["id"],
            theorem_run_id=row["theorem_run_id"],
            cluster_id=row["cluster_id"],
            proposal_id=row["proposal_id"],
            observable_name=row["observable_name"],
            observable_expr=row["observable_expr"],
            rationale=row["rationale"],
            priority=row["priority"],
            action_type=row["action_type"] if "action_type" in keys else None,
            status=row["status"],
            created_at=row["created_at"],
        )

    # --- Cluster artifact operations (PHASE-E) ---

    def insert_cluster_artifact(self, artifact: ClusterArtifactRecord) -> int:
        """Insert a new cluster artifact. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO cluster_artifacts (
                artifact_hash, theorem_run_id, snapshot_hash, signature_version,
                method, params_json, assignments_json, cluster_summaries_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.artifact_hash,
                artifact.theorem_run_id,
                artifact.snapshot_hash,
                artifact.signature_version,
                artifact.method,
                artifact.params_json,
                artifact.assignments_json,
                artifact.cluster_summaries_json,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_cluster_artifact_by_hash(
        self, artifact_hash: str
    ) -> ClusterArtifactRecord | None:
        """Get a cluster artifact by its hash."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM cluster_artifacts WHERE artifact_hash = ?",
            (artifact_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_cluster_artifact(row)

    def get_cluster_artifact_by_id(
        self, artifact_id: int
    ) -> ClusterArtifactRecord | None:
        """Get a cluster artifact by its database ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM cluster_artifacts WHERE id = ?",
            (artifact_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_cluster_artifact(row)

    def get_cluster_artifacts_by_run(
        self, theorem_run_id: int, limit: int = 100
    ) -> list[ClusterArtifactRecord]:
        """Get all cluster artifacts for a theorem run."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM cluster_artifacts
            WHERE theorem_run_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (theorem_run_id, limit),
        )
        return [self._row_to_cluster_artifact(row) for row in cursor.fetchall()]

    def list_cluster_artifacts(
        self, limit: int = 100
    ) -> list[ClusterArtifactRecord]:
        """List cluster artifacts in reverse chronological order."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM cluster_artifacts ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_cluster_artifact(row) for row in cursor.fetchall()]

    def _row_to_cluster_artifact(
        self, row: sqlite3.Row
    ) -> ClusterArtifactRecord:
        return ClusterArtifactRecord(
            id=row["id"],
            artifact_hash=row["artifact_hash"],
            theorem_run_id=row["theorem_run_id"],
            snapshot_hash=row["snapshot_hash"],
            signature_version=row["signature_version"],
            method=row["method"],
            params_json=row["params_json"],
            assignments_json=row["assignments_json"],
            cluster_summaries_json=row["cluster_summaries_json"],
            created_at=row["created_at"],
        )

    # --- Law witness operations (PHASE-E) ---

    def insert_law_witness(self, witness: LawWitnessRecord) -> int:
        """Insert a new law witness. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO law_witnesses (
                law_id, evaluation_id, t_fail, formatted_witness,
                state_at_t, state_at_t1, observables_at_t_json,
                observables_at_t1_json, neighborhood_hash, is_primary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                witness.law_id,
                witness.evaluation_id,
                witness.t_fail,
                witness.formatted_witness,
                witness.state_at_t,
                witness.state_at_t1,
                witness.observables_at_t_json,
                witness.observables_at_t1_json,
                witness.neighborhood_hash,
                witness.is_primary,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_witnesses_for_law(
        self, law_id: str, limit: int = 20
    ) -> list[LawWitnessRecord]:
        """Get witnesses for a law, primary witness first."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM law_witnesses
            WHERE law_id = ?
            ORDER BY is_primary DESC, created_at ASC
            LIMIT ?
            """,
            (law_id, limit),
        )
        return [self._row_to_law_witness(row) for row in cursor.fetchall()]

    def get_witnesses_for_evaluation(
        self, evaluation_id: int, limit: int = 20
    ) -> list[LawWitnessRecord]:
        """Get witnesses for a specific evaluation."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM law_witnesses
            WHERE evaluation_id = ?
            ORDER BY is_primary DESC, created_at ASC
            LIMIT ?
            """,
            (evaluation_id, limit),
        )
        return [self._row_to_law_witness(row) for row in cursor.fetchall()]

    def get_primary_witness_for_law(
        self, law_id: str
    ) -> LawWitnessRecord | None:
        """Get the primary witness for a law."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM law_witnesses
            WHERE law_id = ? AND is_primary = TRUE
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (law_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_law_witness(row)

    def get_witness_diversity(self, law_id: str) -> int:
        """Get the count of unique neighborhood hashes for a law's witnesses."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(DISTINCT neighborhood_hash) as diversity
            FROM law_witnesses
            WHERE law_id = ?
            """,
            (law_id,),
        )
        row = cursor.fetchone()
        return row["diversity"] if row else 0

    def _row_to_law_witness(self, row: sqlite3.Row) -> LawWitnessRecord:
        return LawWitnessRecord(
            id=row["id"],
            law_id=row["law_id"],
            evaluation_id=row["evaluation_id"],
            t_fail=row["t_fail"],
            formatted_witness=row["formatted_witness"],
            state_at_t=row["state_at_t"],
            state_at_t1=row["state_at_t1"],
            observables_at_t_json=row["observables_at_t_json"],
            observables_at_t1_json=row["observables_at_t1_json"],
            neighborhood_hash=row["neighborhood_hash"],
            is_primary=bool(row["is_primary"]),
            created_at=row["created_at"],
        )

    # =========================================================================
    # PHASE-F: Orchestration engine operations
    # =========================================================================

    # --- Orchestration run operations ---

    def insert_orchestration_run(self, run: OrchestrationRunRecord) -> int:
        """Insert a new orchestration run. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO orchestration_runs (
                run_id, status, current_phase, config_json,
                universe_id, sim_hash, harness_hash,
                discovery_model_id, tester_model_id, total_iterations
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.status,
                run.current_phase,
                run.config_json,
                run.universe_id,
                run.sim_hash,
                run.harness_hash,
                run.discovery_model_id,
                run.tester_model_id,
                run.total_iterations,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_orchestration_run(self, run_id: str) -> OrchestrationRunRecord | None:
        """Get an orchestration run by run_id."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM orchestration_runs WHERE run_id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_orchestration_run(row)

    def update_orchestration_run(
        self,
        run_id: str,
        status: str | None = None,
        current_phase: str | None = None,
        total_iterations: int | None = None,
        completed_at: str | None = None,
    ) -> None:
        """Update an orchestration run."""
        updates = []
        params: list[Any] = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if current_phase is not None:
            updates.append("current_phase = ?")
            params.append(current_phase)
        if total_iterations is not None:
            updates.append("total_iterations = ?")
            params.append(total_iterations)
        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at)

        if not updates:
            return

        params.append(run_id)
        cursor = self.conn.cursor()
        cursor.execute(
            f"UPDATE orchestration_runs SET {', '.join(updates)} WHERE run_id = ?",
            params,
        )
        self.conn.commit()

    def list_orchestration_runs(
        self, status: str | None = None, limit: int = 100
    ) -> list[OrchestrationRunRecord]:
        """List orchestration runs."""
        cursor = self.conn.cursor()
        if status:
            cursor.execute(
                """
                SELECT * FROM orchestration_runs
                WHERE status = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (status, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM orchestration_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
        return [self._row_to_orchestration_run(row) for row in cursor.fetchall()]

    def _row_to_orchestration_run(
        self, row: sqlite3.Row
    ) -> OrchestrationRunRecord:
        return OrchestrationRunRecord(
            id=row["id"],
            run_id=row["run_id"],
            status=row["status"],
            current_phase=row["current_phase"],
            config_json=row["config_json"],
            universe_id=row["universe_id"],
            sim_hash=row["sim_hash"],
            harness_hash=row["harness_hash"],
            discovery_model_id=row["discovery_model_id"],
            tester_model_id=row["tester_model_id"],
            total_iterations=row["total_iterations"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    # --- Orchestration iteration operations ---

    def insert_orchestration_iteration(
        self, iteration: OrchestrationIterationRecord
    ) -> int:
        """Insert a new orchestration iteration. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO orchestration_iterations (
                run_id, iteration_index, phase, status,
                prompt_hash, control_block_json, readiness_metrics_json, summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                iteration.run_id,
                iteration.iteration_index,
                iteration.phase,
                iteration.status,
                iteration.prompt_hash,
                iteration.control_block_json,
                iteration.readiness_metrics_json,
                iteration.summary_json,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_orchestration_iteration(
        self, run_id: str, iteration_index: int
    ) -> OrchestrationIterationRecord | None:
        """Get an orchestration iteration by run_id and index."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM orchestration_iterations
            WHERE run_id = ? AND iteration_index = ?
            """,
            (run_id, iteration_index),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_orchestration_iteration(row)

    def get_latest_iteration(
        self, run_id: str
    ) -> OrchestrationIterationRecord | None:
        """Get the most recent iteration for a run."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM orchestration_iterations
            WHERE run_id = ?
            ORDER BY iteration_index DESC
            LIMIT 1
            """,
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_orchestration_iteration(row)

    def update_orchestration_iteration(
        self,
        iteration_id: int,
        status: str | None = None,
        control_block_json: str | None = None,
        readiness_metrics_json: str | None = None,
        summary_json: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        """Update an orchestration iteration."""
        updates = []
        params: list[Any] = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if control_block_json is not None:
            updates.append("control_block_json = ?")
            params.append(control_block_json)
        if readiness_metrics_json is not None:
            updates.append("readiness_metrics_json = ?")
            params.append(readiness_metrics_json)
        if summary_json is not None:
            updates.append("summary_json = ?")
            params.append(summary_json)
        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at)

        if not updates:
            return

        params.append(iteration_id)
        cursor = self.conn.cursor()
        cursor.execute(
            f"UPDATE orchestration_iterations SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        self.conn.commit()

    def list_iterations_for_run(
        self, run_id: str, limit: int = 100
    ) -> list[OrchestrationIterationRecord]:
        """List iterations for a run in order."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM orchestration_iterations
            WHERE run_id = ?
            ORDER BY iteration_index ASC
            LIMIT ?
            """,
            (run_id, limit),
        )
        return [self._row_to_orchestration_iteration(row) for row in cursor.fetchall()]

    def _row_to_orchestration_iteration(
        self, row: sqlite3.Row
    ) -> OrchestrationIterationRecord:
        return OrchestrationIterationRecord(
            id=row["id"],
            run_id=row["run_id"],
            iteration_index=row["iteration_index"],
            phase=row["phase"],
            status=row["status"],
            prompt_hash=row["prompt_hash"],
            control_block_json=row["control_block_json"],
            readiness_metrics_json=row["readiness_metrics_json"],
            summary_json=row["summary_json"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    # --- Phase transition operations ---

    def insert_phase_transition(self, transition: PhaseTransitionRecord) -> int:
        """Insert a phase transition record. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO phase_transitions (
                run_id, iteration_id, from_phase, to_phase,
                trigger, readiness_score, evidence_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transition.run_id,
                transition.iteration_id,
                transition.from_phase,
                transition.to_phase,
                transition.trigger,
                transition.readiness_score,
                transition.evidence_json,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def list_phase_transitions(
        self, run_id: str, limit: int = 100
    ) -> list[PhaseTransitionRecord]:
        """List phase transitions for a run."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM phase_transitions
            WHERE run_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (run_id, limit),
        )
        return [self._row_to_phase_transition(row) for row in cursor.fetchall()]

    def _row_to_phase_transition(
        self, row: sqlite3.Row
    ) -> PhaseTransitionRecord:
        return PhaseTransitionRecord(
            id=row["id"],
            run_id=row["run_id"],
            iteration_id=row["iteration_id"],
            from_phase=row["from_phase"],
            to_phase=row["to_phase"],
            trigger=row["trigger"],
            readiness_score=row["readiness_score"],
            evidence_json=row["evidence_json"],
            created_at=row["created_at"],
        )

    # --- Readiness snapshot operations ---

    def insert_readiness_snapshot(
        self, snapshot: ReadinessSnapshotRecord
    ) -> int:
        """Insert a readiness snapshot. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO readiness_snapshots (
                run_id, iteration_id, phase,
                s_pass, s_stability, s_novel_cex, s_harness_health, s_redundancy,
                s_coverage, s_prediction_accuracy, s_adversarial_accuracy,
                s_held_out_accuracy, combined_score, weights_json, source_counts_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.run_id,
                snapshot.iteration_id,
                snapshot.phase,
                snapshot.s_pass,
                snapshot.s_stability,
                snapshot.s_novel_cex,
                snapshot.s_harness_health,
                snapshot.s_redundancy,
                snapshot.s_coverage,
                snapshot.s_prediction_accuracy,
                snapshot.s_adversarial_accuracy,
                snapshot.s_held_out_accuracy,
                snapshot.combined_score,
                snapshot.weights_json,
                snapshot.source_counts_json,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_latest_readiness_snapshot(
        self, run_id: str
    ) -> ReadinessSnapshotRecord | None:
        """Get the most recent readiness snapshot for a run."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM readiness_snapshots
            WHERE run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_readiness_snapshot(row)

    def _row_to_readiness_snapshot(
        self, row: sqlite3.Row
    ) -> ReadinessSnapshotRecord:
        return ReadinessSnapshotRecord(
            id=row["id"],
            run_id=row["run_id"],
            iteration_id=row["iteration_id"],
            phase=row["phase"],
            s_pass=row["s_pass"],
            s_stability=row["s_stability"],
            s_novel_cex=row["s_novel_cex"],
            s_harness_health=row["s_harness_health"],
            s_redundancy=row["s_redundancy"],
            s_coverage=row["s_coverage"],
            s_prediction_accuracy=row["s_prediction_accuracy"],
            s_adversarial_accuracy=row["s_adversarial_accuracy"],
            s_held_out_accuracy=row["s_held_out_accuracy"],
            combined_score=row["combined_score"],
            weights_json=row["weights_json"],
            source_counts_json=row["source_counts_json"],
            created_at=row["created_at"],
        )

    # --- Explanation operations ---

    def insert_explanation(self, explanation: ExplanationRecord) -> int:
        """Insert a new explanation. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO explanations (
                run_id, iteration_id, explanation_id, hypothesis_text,
                mechanism_json, supporting_theorem_ids_json,
                open_questions_json, criticisms_json, confidence, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                explanation.run_id,
                explanation.iteration_id,
                explanation.explanation_id,
                explanation.hypothesis_text,
                explanation.mechanism_json,
                explanation.supporting_theorem_ids_json,
                explanation.open_questions_json,
                explanation.criticisms_json,
                explanation.confidence,
                explanation.status,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_explanation(self, explanation_id: str) -> ExplanationRecord | None:
        """Get an explanation by ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM explanations WHERE explanation_id = ?",
            (explanation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_explanation(row)

    def _row_to_explanation(self, row: sqlite3.Row) -> ExplanationRecord:
        return ExplanationRecord(
            id=row["id"],
            run_id=row["run_id"],
            iteration_id=row["iteration_id"],
            explanation_id=row["explanation_id"],
            hypothesis_text=row["hypothesis_text"],
            mechanism_json=row["mechanism_json"],
            supporting_theorem_ids_json=row["supporting_theorem_ids_json"],
            open_questions_json=row["open_questions_json"],
            criticisms_json=row["criticisms_json"],
            confidence=row["confidence"],
            status=row["status"],
            created_at=row["created_at"],
        )

    # --- Prediction operations ---

    def insert_prediction(self, prediction: PredictionRecord) -> int:
        """Insert a new prediction. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                run_id, iteration_id, prediction_id, explanation_id,
                initial_state, horizon, predicted_state,
                predicted_observables_json, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction.run_id,
                prediction.iteration_id,
                prediction.prediction_id,
                prediction.explanation_id,
                prediction.initial_state,
                prediction.horizon,
                prediction.predicted_state,
                prediction.predicted_observables_json,
                prediction.confidence,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_prediction(self, prediction_id: str) -> PredictionRecord | None:
        """Get a prediction by ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM predictions WHERE prediction_id = ?",
            (prediction_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_prediction(row)

    def list_predictions_for_run(
        self, run_id: str, limit: int = 100
    ) -> list[PredictionRecord]:
        """List predictions for a run."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM predictions
            WHERE run_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (run_id, limit),
        )
        return [self._row_to_prediction(row) for row in cursor.fetchall()]

    def _row_to_prediction(self, row: sqlite3.Row) -> PredictionRecord:
        return PredictionRecord(
            id=row["id"],
            run_id=row["run_id"],
            iteration_id=row["iteration_id"],
            prediction_id=row["prediction_id"],
            explanation_id=row["explanation_id"],
            initial_state=row["initial_state"],
            horizon=row["horizon"],
            predicted_state=row["predicted_state"],
            predicted_observables_json=row["predicted_observables_json"],
            confidence=row["confidence"],
            created_at=row["created_at"],
        )

    # --- Prediction evaluation operations ---

    def insert_prediction_evaluation(
        self, evaluation: PredictionEvaluationRecord
    ) -> int:
        """Insert a prediction evaluation. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_evaluations (
                prediction_id, run_id, actual_state, is_exact_match,
                hamming_distance, cell_accuracy, observable_errors_json, evaluation_set
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation.prediction_id,
                evaluation.run_id,
                evaluation.actual_state,
                1 if evaluation.is_exact_match else 0,
                evaluation.hamming_distance,
                evaluation.cell_accuracy,
                evaluation.observable_errors_json,
                evaluation.evaluation_set,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_prediction_evaluations(
        self, run_id: str, evaluation_set: str | None = None
    ) -> list[PredictionEvaluationRecord]:
        """Get prediction evaluations for a run."""
        cursor = self.conn.cursor()
        if evaluation_set:
            cursor.execute(
                """
                SELECT * FROM prediction_evaluations
                WHERE run_id = ? AND evaluation_set = ?
                ORDER BY created_at DESC
                """,
                (run_id, evaluation_set),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM prediction_evaluations
                WHERE run_id = ?
                ORDER BY created_at DESC
                """,
                (run_id,),
            )
        return [self._row_to_prediction_evaluation(row) for row in cursor.fetchall()]

    def _row_to_prediction_evaluation(
        self, row: sqlite3.Row
    ) -> PredictionEvaluationRecord:
        return PredictionEvaluationRecord(
            id=row["id"],
            prediction_id=row["prediction_id"],
            run_id=row["run_id"],
            actual_state=row["actual_state"],
            is_exact_match=bool(row["is_exact_match"]),
            hamming_distance=row["hamming_distance"],
            cell_accuracy=row["cell_accuracy"],
            observable_errors_json=row["observable_errors_json"],
            evaluation_set=row["evaluation_set"],
            created_at=row["created_at"],
        )

    # --- Held-out set operations ---

    def insert_held_out_set(self, held_out_set: HeldOutSetRecord) -> int:
        """Insert a held-out test set. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO held_out_sets (
                run_id, set_type, generation_seed, cases_json, case_count, locked
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                held_out_set.run_id,
                held_out_set.set_type,
                held_out_set.generation_seed,
                held_out_set.cases_json,
                held_out_set.case_count,
                1 if held_out_set.locked else 0,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_held_out_set(
        self, run_id: str, set_type: str
    ) -> HeldOutSetRecord | None:
        """Get a held-out set by run_id and type."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM held_out_sets
            WHERE run_id = ? AND set_type = ?
            """,
            (run_id, set_type),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_held_out_set(row)

    def list_held_out_sets(self, run_id: str) -> list[HeldOutSetRecord]:
        """List all held-out sets for a run."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM held_out_sets WHERE run_id = ?",
            (run_id,),
        )
        return [self._row_to_held_out_set(row) for row in cursor.fetchall()]

    def _row_to_held_out_set(self, row: sqlite3.Row) -> HeldOutSetRecord:
        return HeldOutSetRecord(
            id=row["id"],
            run_id=row["run_id"],
            set_type=row["set_type"],
            generation_seed=row["generation_seed"],
            cases_json=row["cases_json"],
            case_count=row["case_count"],
            locked=bool(row["locked"]),
            created_at=row["created_at"],
        )

    # =========================================================================
    # Web Viewer: LLM Transcript operations
    # =========================================================================

    def insert_llm_transcript(self, record: LLMTranscriptRecord) -> int:
        """Insert a new LLM transcript record. Returns the new ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO llm_transcripts (
                run_id, iteration_id, phase, component, model_name,
                system_instruction, prompt, raw_response, prompt_hash,
                prompt_tokens, output_tokens, thinking_tokens, total_tokens,
                duration_ms, success, error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                record.iteration_id,
                record.phase,
                record.component,
                record.model_name,
                record.system_instruction,
                record.prompt,
                record.raw_response,
                record.prompt_hash,
                record.prompt_tokens,
                record.output_tokens,
                record.thinking_tokens,
                record.total_tokens,
                record.duration_ms,
                1 if record.success else 0,
                record.error_message,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_llm_transcript(self, transcript_id: int) -> LLMTranscriptRecord | None:
        """Get an LLM transcript by ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM llm_transcripts WHERE id = ?",
            (transcript_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_llm_transcript(row)

    def list_llm_transcripts(
        self,
        run_id: str | None = None,
        component: str | None = None,
        phase: str | None = None,
        search_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LLMTranscriptRecord]:
        """List LLM transcripts with optional filtering.

        Args:
            run_id: Filter by orchestration run ID
            component: Filter by component name
            phase: Filter by phase name
            search_query: Search in prompt and response text
            limit: Maximum records to return
            offset: Records to skip for pagination

        Returns:
            List of LLMTranscriptRecord objects
        """
        cursor = self.conn.cursor()
        conditions = []
        params: list[Any] = []

        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if component:
            conditions.append("component = ?")
            params.append(component)
        if phase:
            conditions.append("phase = ?")
            params.append(phase)
        if search_query:
            conditions.append("(prompt LIKE ? OR raw_response LIKE ?)")
            params.append(f"%{search_query}%")
            params.append(f"%{search_query}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        cursor.execute(
            f"""
            SELECT * FROM llm_transcripts
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params,
        )
        return [self._row_to_llm_transcript(row) for row in cursor.fetchall()]

    def get_llm_transcript_count(
        self,
        run_id: str | None = None,
        component: str | None = None,
    ) -> int:
        """Get count of LLM transcripts with optional filtering."""
        cursor = self.conn.cursor()
        conditions = []
        params: list[Any] = []

        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if component:
            conditions.append("component = ?")
            params.append(component)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor.execute(
            f"SELECT COUNT(*) as count FROM llm_transcripts WHERE {where_clause}",
            params,
        )
        row = cursor.fetchone()
        return row["count"] if row else 0

    def get_llm_transcript_stats(self, run_id: str | None = None) -> dict[str, Any]:
        """Get statistics about LLM transcripts.

        Returns dict with:
            - total_calls: Total number of LLM calls
            - by_component: Dict of counts by component
            - total_tokens: Total tokens used
            - total_duration_ms: Total duration in milliseconds
            - success_rate: Ratio of successful calls
        """
        cursor = self.conn.cursor()

        where_clause = "WHERE run_id = ?" if run_id else ""
        params = [run_id] if run_id else []

        # Get totals
        cursor.execute(
            f"""
            SELECT
                COUNT(*) as total_calls,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(duration_ms), 0) as total_duration_ms,
                COALESCE(AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END), 0) as success_rate
            FROM llm_transcripts
            {where_clause}
            """,
            params,
        )
        row = cursor.fetchone()

        # Get counts by component
        cursor.execute(
            f"""
            SELECT component, COUNT(*) as count
            FROM llm_transcripts
            {where_clause}
            GROUP BY component
            """,
            params,
        )
        by_component = {r["component"]: r["count"] for r in cursor.fetchall()}

        return {
            "total_calls": row["total_calls"] if row else 0,
            "total_tokens": row["total_tokens"] if row else 0,
            "total_duration_ms": row["total_duration_ms"] if row else 0,
            "success_rate": row["success_rate"] if row else 0,
            "by_component": by_component,
        }

    def _row_to_llm_transcript(self, row: sqlite3.Row) -> LLMTranscriptRecord:
        return LLMTranscriptRecord(
            id=row["id"],
            run_id=row["run_id"],
            iteration_id=row["iteration_id"],
            phase=row["phase"],
            component=row["component"],
            model_name=row["model_name"],
            system_instruction=row["system_instruction"],
            prompt=row["prompt"],
            raw_response=row["raw_response"],
            prompt_hash=row["prompt_hash"],
            prompt_tokens=row["prompt_tokens"],
            output_tokens=row["output_tokens"],
            thinking_tokens=row["thinking_tokens"] or 0,
            total_tokens=row["total_tokens"],
            duration_ms=row["duration_ms"],
            success=bool(row["success"]),
            error_message=row["error_message"],
            created_at=row["created_at"],
        )

    # --- Additional web viewer query methods ---

    def list_explanations_for_run(
        self, run_id: str, limit: int = 100
    ) -> list[ExplanationRecord]:
        """List explanations for a run."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM explanations
            WHERE run_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (run_id, limit),
        )
        return [self._row_to_explanation(row) for row in cursor.fetchall()]

    def list_readiness_snapshots_for_run(
        self, run_id: str, limit: int = 100
    ) -> list[ReadinessSnapshotRecord]:
        """List readiness snapshots for a run in chronological order."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM readiness_snapshots
            WHERE run_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (run_id, limit),
        )
        return [self._row_to_readiness_snapshot(row) for row in cursor.fetchall()]
