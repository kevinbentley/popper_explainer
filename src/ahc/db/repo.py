"""Repository for AHC-DS database operations."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.ahc.db.models import (
    SessionRecord,
    SessionStatus,
    JournalEntry,
    JournalEntryType,
    ToolCallRecord,
    PredictionRecord,
    TheoremRecord,
    TheoremStatus,
    LawEvaluationRecord,
    TrajectorySampleRecord,
    TransitionRuleRecord,
    ConversationTurnRecord,
    MetaKnowledgeRecord,
)

logger = logging.getLogger(__name__)


class AHCRepository:
    """Database repository for AHC-DS.

    Handles all persistence operations for the agentic discovery system.
    Uses a separate SQLite database (high_context.db) to avoid interfering
    with the existing popper.db.
    """

    def __init__(self, db_path: str | Path = "high_context.db"):
        """Initialize the repository.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Connect to the database and initialize schema."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            schema = f.read()
        self._conn.executescript(schema)
        self._conn.commit()

        # Run migrations for existing databases
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Run schema migrations for existing databases.

        This handles adding new columns/tables to databases created
        before those features were added.
        """
        # Migration 1: Add token_count column to conversation_turns
        cursor = self._conn.execute("PRAGMA table_info(conversation_turns)")
        columns = [row[1] for row in cursor.fetchall()]
        if "token_count" not in columns:
            logger.info("Migration: Adding token_count column to conversation_turns")
            self._conn.execute(
                "ALTER TABLE conversation_turns ADD COLUMN token_count INTEGER DEFAULT 0"
            )
            self._conn.commit()

        # Migration 2: meta_knowledge table is created by schema.sql (CREATE IF NOT EXISTS)
        # No additional action needed

    @property
    def conn(self) -> sqlite3.Connection:
        """Get the database connection, raising if not connected."""
        if self._conn is None:
            raise RuntimeError("Not connected to database. Call connect() first.")
        return self._conn

    # =========================================================================
    # Session operations
    # =========================================================================

    def insert_session(self, session: SessionRecord) -> int:
        """Insert a new session.

        Returns:
            The database ID of the inserted session.
        """
        cursor = self.conn.execute(
            """
            INSERT INTO sessions (
                session_id, status, config_json, model_id, seed,
                total_predictions, correct_predictions, accuracy,
                transition_rules_complete
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                session.status.value,
                session.config_json,
                session.model_id,
                session.seed,
                session.total_predictions,
                session.correct_predictions,
                session.accuracy,
                1 if session.transition_rules_complete else 0,
            )
        )
        self.conn.commit()
        session.id = cursor.lastrowid
        return cursor.lastrowid

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Get a session by its session_id."""
        cursor = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def get_session_by_db_id(self, db_id: int) -> SessionRecord | None:
        """Get a session by its database ID."""
        cursor = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (db_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def update_session(self, session: SessionRecord) -> None:
        """Update an existing session."""
        self.conn.execute(
            """
            UPDATE sessions SET
                status = ?,
                updated_at = datetime('now'),
                config_json = ?,
                total_predictions = ?,
                correct_predictions = ?,
                accuracy = ?,
                transition_rules_complete = ?,
                terminated_at = ?,
                termination_reason = ?
            WHERE id = ?
            """,
            (
                session.status.value,
                session.config_json,
                session.total_predictions,
                session.correct_predictions,
                session.accuracy,
                1 if session.transition_rules_complete else 0,
                session.terminated_at.isoformat() if session.terminated_at else None,
                session.termination_reason,
                session.id,
            )
        )
        self.conn.commit()

    def _row_to_session(self, row: sqlite3.Row) -> SessionRecord:
        """Convert a database row to a SessionRecord."""
        return SessionRecord(
            id=row["id"],
            session_id=row["session_id"],
            status=SessionStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            config_json=row["config_json"],
            model_id=row["model_id"],
            seed=row["seed"],
            total_predictions=row["total_predictions"],
            correct_predictions=row["correct_predictions"],
            accuracy=row["accuracy"],
            transition_rules_complete=bool(row["transition_rules_complete"]),
            terminated_at=datetime.fromisoformat(row["terminated_at"]) if row["terminated_at"] else None,
            termination_reason=row["termination_reason"],
        )

    # =========================================================================
    # Journal entry operations
    # =========================================================================

    def insert_journal_entry(self, entry: JournalEntry) -> int:
        """Insert a journal entry."""
        cursor = self.conn.execute(
            """
            INSERT INTO journal_entries (
                session_id, turn_number, entry_type, content, metadata_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                entry.session_id,
                entry.turn_number,
                entry.entry_type.value,
                entry.content,
                entry.metadata_json,
            )
        )
        self.conn.commit()
        entry.id = cursor.lastrowid
        return cursor.lastrowid

    def get_journal_entries(
        self,
        session_id: int,
        entry_type: JournalEntryType | None = None,
        limit: int | None = None,
    ) -> list[JournalEntry]:
        """Get journal entries for a session."""
        query = "SELECT * FROM journal_entries WHERE session_id = ?"
        params: list[Any] = [session_id]

        if entry_type:
            query += " AND entry_type = ?"
            params.append(entry_type.value)

        query += " ORDER BY turn_number, id"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.conn.execute(query, params)
        return [self._row_to_journal_entry(row) for row in cursor.fetchall()]

    def _row_to_journal_entry(self, row: sqlite3.Row) -> JournalEntry:
        """Convert a database row to a JournalEntry."""
        return JournalEntry(
            id=row["id"],
            session_id=row["session_id"],
            turn_number=row["turn_number"],
            entry_type=JournalEntryType(row["entry_type"]),
            content=row["content"],
            metadata_json=row["metadata_json"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    # =========================================================================
    # Tool call operations
    # =========================================================================

    def insert_tool_call(self, call: ToolCallRecord) -> int:
        """Insert a tool call record."""
        cursor = self.conn.execute(
            """
            INSERT INTO tool_calls (
                session_id, turn_number, tool_name, arguments_json,
                result_json, error, completed_at, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                call.session_id,
                call.turn_number,
                call.tool_name,
                call.arguments_json,
                call.result_json,
                call.error,
                call.completed_at.isoformat() if call.completed_at else None,
                call.duration_ms,
            )
        )
        self.conn.commit()
        call.id = cursor.lastrowid
        return cursor.lastrowid

    def get_tool_calls(
        self,
        session_id: int,
        tool_name: str | None = None,
        limit: int | None = None,
    ) -> list[ToolCallRecord]:
        """Get tool calls for a session."""
        query = "SELECT * FROM tool_calls WHERE session_id = ?"
        params: list[Any] = [session_id]

        if tool_name:
            query += " AND tool_name = ?"
            params.append(tool_name)

        query += " ORDER BY turn_number, id"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.conn.execute(query, params)
        return [self._row_to_tool_call(row) for row in cursor.fetchall()]

    def _row_to_tool_call(self, row: sqlite3.Row) -> ToolCallRecord:
        """Convert a database row to a ToolCallRecord."""
        return ToolCallRecord(
            id=row["id"],
            session_id=row["session_id"],
            turn_number=row["turn_number"],
            tool_name=row["tool_name"],
            arguments_json=row["arguments_json"],
            result_json=row["result_json"],
            error=row["error"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            duration_ms=row["duration_ms"],
        )

    # =========================================================================
    # Prediction operations
    # =========================================================================

    def insert_prediction(self, prediction: PredictionRecord) -> int:
        """Insert a prediction record."""
        cursor = self.conn.execute(
            """
            INSERT INTO predictions (
                session_id, turn_number, state_t0, predicted_state_t1,
                actual_state_t1, is_correct, prediction_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction.session_id,
                prediction.turn_number,
                prediction.state_t0,
                prediction.predicted_state_t1,
                prediction.actual_state_t1,
                1 if prediction.is_correct else 0,
                prediction.prediction_method,
            )
        )
        self.conn.commit()
        prediction.id = cursor.lastrowid
        return cursor.lastrowid

    def get_accuracy_stats(self, session_id: int) -> dict[str, Any]:
        """Get prediction accuracy statistics for a session."""
        cursor = self.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(is_correct) as correct
            FROM predictions
            WHERE session_id = ?
            """,
            (session_id,)
        )
        row = cursor.fetchone()
        total = row["total"] or 0
        correct = row["correct"] or 0
        return {
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    # =========================================================================
    # Theorem operations
    # =========================================================================

    def insert_theorem(self, theorem: TheoremRecord) -> int:
        """Insert a theorem."""
        cursor = self.conn.execute(
            """
            INSERT INTO theorems (
                session_id, name, description, law_ids_json, status, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                theorem.session_id,
                theorem.name,
                theorem.description,
                theorem.law_ids_json,
                theorem.status.value,
                theorem.evidence_json,
            )
        )
        self.conn.commit()
        theorem.id = cursor.lastrowid
        return cursor.lastrowid

    def get_theorem(self, session_id: int, name: str) -> TheoremRecord | None:
        """Get a theorem by name."""
        cursor = self.conn.execute(
            "SELECT * FROM theorems WHERE session_id = ? AND name = ?",
            (session_id, name)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_theorem(row)

    def get_theorems(
        self,
        session_id: int,
        status: TheoremStatus | None = None,
    ) -> list[TheoremRecord]:
        """Get theorems for a session."""
        query = "SELECT * FROM theorems WHERE session_id = ?"
        params: list[Any] = [session_id]

        if status:
            query += " AND status = ?"
            params.append(status.value)

        query += " ORDER BY created_at"

        cursor = self.conn.execute(query, params)
        return [self._row_to_theorem(row) for row in cursor.fetchall()]

    def update_theorem(self, theorem: TheoremRecord) -> None:
        """Update a theorem."""
        self.conn.execute(
            """
            UPDATE theorems SET
                description = ?,
                law_ids_json = ?,
                status = ?,
                evidence_json = ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (
                theorem.description,
                theorem.law_ids_json,
                theorem.status.value,
                theorem.evidence_json,
                theorem.id,
            )
        )
        self.conn.commit()

    def _row_to_theorem(self, row: sqlite3.Row) -> TheoremRecord:
        """Convert a database row to a TheoremRecord."""
        return TheoremRecord(
            id=row["id"],
            session_id=row["session_id"],
            name=row["name"],
            description=row["description"],
            law_ids_json=row["law_ids_json"],
            status=TheoremStatus(row["status"]),
            evidence_json=row["evidence_json"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    # =========================================================================
    # Law evaluation operations
    # =========================================================================

    def insert_law_evaluation(self, evaluation: LawEvaluationRecord) -> int:
        """Insert a law evaluation result."""
        cursor = self.conn.execute(
            """
            INSERT INTO law_evaluations (
                session_id, turn_number, law_json, law_id, law_hash,
                status, reason_code, counterexample_json, power_metrics_json, runtime_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation.session_id,
                evaluation.turn_number,
                evaluation.law_json,
                evaluation.law_id,
                evaluation.law_hash,
                evaluation.status,
                evaluation.reason_code,
                evaluation.counterexample_json,
                evaluation.power_metrics_json,
                evaluation.runtime_ms,
            )
        )
        self.conn.commit()
        evaluation.id = cursor.lastrowid
        return cursor.lastrowid

    def get_law_evaluations(
        self,
        session_id: int,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[LawEvaluationRecord]:
        """Get law evaluations for a session."""
        query = "SELECT * FROM law_evaluations WHERE session_id = ?"
        params: list[Any] = [session_id]

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY turn_number, id"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.conn.execute(query, params)
        return [self._row_to_law_evaluation(row) for row in cursor.fetchall()]

    def get_recent_failed_evaluations(
        self,
        session_id: int,
        limit: int = 10,
    ) -> list[LawEvaluationRecord]:
        """Get recent FAIL evaluations, most-recent-first.

        Used to build the counterexample gallery for context injection.

        Args:
            session_id: Session ID
            limit: Maximum number of results

        Returns:
            List of FAIL evaluations ordered most-recent-first
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM law_evaluations
            WHERE session_id = ? AND status = 'FAIL'
            ORDER BY turn_number DESC, id DESC
            LIMIT ?
            """,
            (session_id, limit)
        )
        return [self._row_to_law_evaluation(row) for row in cursor.fetchall()]

    def _row_to_law_evaluation(self, row: sqlite3.Row) -> LawEvaluationRecord:
        """Convert a database row to a LawEvaluationRecord."""
        return LawEvaluationRecord(
            id=row["id"],
            session_id=row["session_id"],
            turn_number=row["turn_number"],
            law_json=row["law_json"],
            law_id=row["law_id"],
            law_hash=row["law_hash"],
            status=row["status"],
            reason_code=row["reason_code"],
            counterexample_json=row["counterexample_json"],
            power_metrics_json=row["power_metrics_json"],
            runtime_ms=row["runtime_ms"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    # =========================================================================
    # Trajectory sample operations
    # =========================================================================

    def insert_trajectory_samples(self, samples: TrajectorySampleRecord) -> int:
        """Insert trajectory samples."""
        cursor = self.conn.execute(
            """
            INSERT INTO trajectory_samples (
                session_id, pattern, count_requested, samples_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                samples.session_id,
                samples.pattern,
                samples.count_requested,
                samples.samples_json,
            )
        )
        self.conn.commit()
        samples.id = cursor.lastrowid
        return cursor.lastrowid

    def get_trajectory_samples(
        self,
        session_id: int,
        pattern: str | None = None,
    ) -> list[TrajectorySampleRecord]:
        """Get trajectory samples for a session."""
        query = "SELECT * FROM trajectory_samples WHERE session_id = ?"
        params: list[Any] = [session_id]

        if pattern:
            query += " AND pattern = ?"
            params.append(pattern)

        cursor = self.conn.execute(query, params)
        return [self._row_to_trajectory_samples(row) for row in cursor.fetchall()]

    def _row_to_trajectory_samples(self, row: sqlite3.Row) -> TrajectorySampleRecord:
        """Convert a database row to a TrajectorySampleRecord."""
        return TrajectorySampleRecord(
            id=row["id"],
            session_id=row["session_id"],
            pattern=row["pattern"],
            count_requested=row["count_requested"],
            samples_json=row["samples_json"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    # =========================================================================
    # Transition rule operations
    # =========================================================================

    def insert_transition_rule(self, rule: TransitionRuleRecord) -> int:
        """Insert or update a transition rule."""
        cursor = self.conn.execute(
            """
            INSERT INTO transition_rules (
                session_id, symbol, neighbor_config, coordinate_class,
                result_symbol, confidence, evidence_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, symbol, neighbor_config, coordinate_class)
            DO UPDATE SET
                result_symbol = excluded.result_symbol,
                confidence = excluded.confidence,
                evidence_count = evidence_count + excluded.evidence_count,
                updated_at = datetime('now')
            """,
            (
                rule.session_id,
                rule.symbol,
                rule.neighbor_config,
                rule.coordinate_class,
                rule.result_symbol,
                rule.confidence,
                rule.evidence_count,
            )
        )
        self.conn.commit()
        rule.id = cursor.lastrowid
        return cursor.lastrowid

    def get_transition_rules(self, session_id: int) -> list[TransitionRuleRecord]:
        """Get all transition rules for a session."""
        cursor = self.conn.execute(
            "SELECT * FROM transition_rules WHERE session_id = ? ORDER BY symbol, neighbor_config",
            (session_id,)
        )
        return [self._row_to_transition_rule(row) for row in cursor.fetchall()]

    def get_transition_completeness(self, session_id: int) -> dict[str, Any]:
        """Check completeness of transition rules.

        Returns statistics on how many rules have been discovered
        for each symbol.
        """
        cursor = self.conn.execute(
            """
            SELECT symbol, COUNT(*) as rule_count
            FROM transition_rules
            WHERE session_id = ?
            GROUP BY symbol
            """,
            (session_id,)
        )
        symbol_counts = {row["symbol"]: row["rule_count"] for row in cursor.fetchall()}

        # Expected symbols: >, <, ., X
        expected_symbols = {">", "<", ".", "X"}

        # For completeness, we need rules covering all symbols
        # The exact number depends on neighbor configurations
        covered_symbols = set(symbol_counts.keys())

        return {
            "symbol_counts": symbol_counts,
            "covered_symbols": list(covered_symbols),
            "missing_symbols": list(expected_symbols - covered_symbols),
            "is_complete": covered_symbols >= expected_symbols,
        }

    def _row_to_transition_rule(self, row: sqlite3.Row) -> TransitionRuleRecord:
        """Convert a database row to a TransitionRuleRecord."""
        return TransitionRuleRecord(
            id=row["id"],
            session_id=row["session_id"],
            symbol=row["symbol"],
            neighbor_config=row["neighbor_config"],
            coordinate_class=row["coordinate_class"],
            result_symbol=row["result_symbol"],
            confidence=row["confidence"],
            evidence_count=row["evidence_count"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    # =========================================================================
    # Conversation turn operations
    # =========================================================================

    def insert_conversation_turn(self, turn: ConversationTurnRecord) -> int:
        """Insert a conversation turn."""
        cursor = self.conn.execute(
            """
            INSERT INTO conversation_turns (
                session_id, turn_number, role, content, tool_calls_json, token_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                turn.session_id,
                turn.turn_number,
                turn.role,
                turn.content,
                turn.tool_calls_json,
                turn.token_count,
            )
        )
        self.conn.commit()
        turn.id = cursor.lastrowid
        return cursor.lastrowid

    def get_conversation_history(
        self,
        session_id: int,
        from_turn: int = 0,
    ) -> list[ConversationTurnRecord]:
        """Get conversation history for a session."""
        cursor = self.conn.execute(
            """
            SELECT * FROM conversation_turns
            WHERE session_id = ? AND turn_number >= ?
            ORDER BY turn_number, role
            """,
            (session_id, from_turn)
        )
        return [self._row_to_conversation_turn(row) for row in cursor.fetchall()]

    def get_last_turn_number(self, session_id: int) -> int:
        """Get the last turn number for a session."""
        cursor = self.conn.execute(
            "SELECT MAX(turn_number) as max_turn FROM conversation_turns WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        return row["max_turn"] or 0

    def _row_to_conversation_turn(self, row: sqlite3.Row) -> ConversationTurnRecord:
        """Convert a database row to a ConversationTurnRecord."""
        return ConversationTurnRecord(
            id=row["id"],
            session_id=row["session_id"],
            turn_number=row["turn_number"],
            role=row["role"],
            content=row["content"],
            tool_calls_json=row["tool_calls_json"],
            token_count=row["token_count"] if "token_count" in row.keys() else 0,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    # =========================================================================
    # Meta Knowledge (Context Management)
    # =========================================================================

    def insert_meta_knowledge(self, record: MetaKnowledgeRecord) -> int:
        """Insert a meta knowledge compaction snapshot."""
        cursor = self.conn.execute(
            """
            INSERT INTO meta_knowledge (
                session_id, version, theorems_snapshot_json, negative_knowledge,
                last_compacted_turn, turns_compacted, token_count_before,
                token_count_after, compaction_prompt_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.version,
                record.theorems_snapshot_json,
                record.negative_knowledge,
                record.last_compacted_turn,
                record.turns_compacted,
                record.token_count_before,
                record.token_count_after,
                record.compaction_prompt_hash,
            )
        )
        self.conn.commit()
        record.id = cursor.lastrowid
        return cursor.lastrowid

    def get_latest_meta_knowledge(
        self,
        session_id: int,
    ) -> MetaKnowledgeRecord | None:
        """Get the most recent meta knowledge snapshot for a session."""
        cursor = self.conn.execute(
            """
            SELECT * FROM meta_knowledge
            WHERE session_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (session_id,)
        )
        row = cursor.fetchone()
        return self._row_to_meta_knowledge(row) if row else None

    def get_next_meta_knowledge_version(self, session_id: int) -> int:
        """Get the next version number for meta knowledge."""
        cursor = self.conn.execute(
            "SELECT MAX(version) as max_version FROM meta_knowledge WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        max_version = row["max_version"] if row["max_version"] is not None else 0
        return max_version + 1

    def _row_to_meta_knowledge(self, row: sqlite3.Row) -> MetaKnowledgeRecord:
        """Convert a database row to a MetaKnowledgeRecord."""
        return MetaKnowledgeRecord(
            id=row["id"],
            session_id=row["session_id"],
            version=row["version"],
            theorems_snapshot_json=row["theorems_snapshot_json"],
            negative_knowledge=row["negative_knowledge"],
            last_compacted_turn=row["last_compacted_turn"],
            turns_compacted=row["turns_compacted"],
            token_count_before=row["token_count_before"],
            token_count_after=row["token_count_after"],
            compaction_prompt_hash=row["compaction_prompt_hash"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    # =========================================================================
    # Context Management Queries
    # =========================================================================

    def get_recent_turns(
        self,
        session_id: int,
        limit: int = 15,
    ) -> list[ConversationTurnRecord]:
        """Get the most recent N turns for the rolling window (Tier 3).

        Args:
            session_id: Session ID
            limit: Maximum number of turns to return

        Returns:
            List of recent turns, ordered oldest to newest
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM conversation_turns
            WHERE session_id = ?
            ORDER BY turn_number DESC, id DESC
            LIMIT ?
            """,
            (session_id, limit)
        )
        rows = cursor.fetchall()
        # Reverse to get chronological order (oldest first)
        return [self._row_to_conversation_turn(row) for row in reversed(rows)]

    def get_total_token_count(self, session_id: int) -> int:
        """Get total token count for all turns in a session."""
        cursor = self.conn.execute(
            "SELECT SUM(token_count) as total FROM conversation_turns WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        return row["total"] or 0

    def get_token_count_since_turn(
        self,
        session_id: int,
        from_turn: int,
    ) -> int:
        """Get token count for turns since a given turn number."""
        cursor = self.conn.execute(
            """
            SELECT SUM(token_count) as total FROM conversation_turns
            WHERE session_id = ? AND turn_number > ?
            """,
            (session_id, from_turn)
        )
        row = cursor.fetchone()
        return row["total"] or 0

    def get_turns_since_compaction(
        self,
        session_id: int,
    ) -> list[ConversationTurnRecord]:
        """Get all turns since the last compaction.

        If no compaction has occurred, returns all turns except system prompt.
        """
        # Get last compacted turn
        meta = self.get_latest_meta_knowledge(session_id)
        from_turn = meta.last_compacted_turn if meta else 0

        cursor = self.conn.execute(
            """
            SELECT * FROM conversation_turns
            WHERE session_id = ? AND turn_number > ?
            ORDER BY turn_number, id
            """,
            (session_id, from_turn)
        )
        return [self._row_to_conversation_turn(row) for row in cursor.fetchall()]

    def get_turn_count_since_compaction(self, session_id: int) -> int:
        """Get count of turns since last compaction."""
        meta = self.get_latest_meta_knowledge(session_id)
        from_turn = meta.last_compacted_turn if meta else 0

        cursor = self.conn.execute(
            """
            SELECT COUNT(*) as count FROM conversation_turns
            WHERE session_id = ? AND turn_number > ?
            """,
            (session_id, from_turn)
        )
        row = cursor.fetchone()
        return row["count"] or 0
