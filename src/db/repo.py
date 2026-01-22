"""Repository pattern for database operations."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.db.models import (
    AuditLogRecord,
    CaseSetRecord,
    CounterexampleRecord,
    EvaluationRecord,
    LawRecord,
    TheoryRecord,
)

SCHEMA_VERSION = 1


class Repository:
    """Database repository for Popper Explainer persistence."""

    def __init__(self, db_path: str | Path = "popper.db"):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Open database connection and ensure schema exists."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
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
