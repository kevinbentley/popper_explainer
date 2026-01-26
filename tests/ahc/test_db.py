"""Tests for AHC database layer."""

import json
import pytest
import tempfile
from pathlib import Path

from src.ahc.db import (
    AHCRepository,
    SessionRecord,
    SessionStatus,
    JournalEntry,
    JournalEntryType,
    ToolCallRecord,
    PredictionRecord,
    TheoremRecord,
    TheoremStatus,
    TransitionRuleRecord,
)


@pytest.fixture
def repo():
    """Create a temporary repository for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    repo = AHCRepository(db_path)
    repo.connect()
    yield repo
    repo.close()

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


class TestSessionOperations:
    """Tests for session CRUD operations."""

    def test_create_session(self, repo):
        """Test creating a new session."""
        session = SessionRecord(
            session_id="test-001",
            status=SessionStatus.RUNNING,
            model_id="gemini-2.5-flash",
            seed=42,
        )
        session_id = repo.insert_session(session)

        assert session_id > 0
        assert session.id == session_id

    def test_get_session(self, repo):
        """Test retrieving a session."""
        session = SessionRecord(
            session_id="test-002",
            status=SessionStatus.RUNNING,
            config_json='{"test": true}',
        )
        repo.insert_session(session)

        retrieved = repo.get_session("test-002")
        assert retrieved is not None
        assert retrieved.session_id == "test-002"
        assert retrieved.status == SessionStatus.RUNNING
        assert retrieved.config_json == '{"test": true}'

    def test_update_session(self, repo):
        """Test updating a session."""
        session = SessionRecord(
            session_id="test-003",
            status=SessionStatus.RUNNING,
        )
        repo.insert_session(session)

        session.status = SessionStatus.COMPLETED
        session.accuracy = 0.95
        session.termination_reason = "Test completed"
        repo.update_session(session)

        retrieved = repo.get_session("test-003")
        assert retrieved.status == SessionStatus.COMPLETED
        assert retrieved.accuracy == 0.95
        assert retrieved.termination_reason == "Test completed"

    def test_get_nonexistent_session(self, repo):
        """Test getting a session that doesn't exist."""
        result = repo.get_session("nonexistent")
        assert result is None


class TestJournalOperations:
    """Tests for journal entry operations."""

    def test_insert_journal_entry(self, repo):
        """Test inserting a journal entry."""
        session = SessionRecord(session_id="journal-test")
        repo.insert_session(session)

        entry = JournalEntry(
            session_id=session.id,
            turn_number=1,
            entry_type=JournalEntryType.OBSERVATION,
            content="Observed pattern >.<",
        )
        entry_id = repo.insert_journal_entry(entry)

        assert entry_id > 0

    def test_get_journal_entries(self, repo):
        """Test retrieving journal entries."""
        session = SessionRecord(session_id="journal-test-2")
        repo.insert_session(session)

        # Insert multiple entries
        for i, etype in enumerate([
            JournalEntryType.THOUGHT,
            JournalEntryType.OBSERVATION,
            JournalEntryType.HYPOTHESIS,
        ]):
            entry = JournalEntry(
                session_id=session.id,
                turn_number=i,
                entry_type=etype,
                content=f"Entry {i}",
            )
            repo.insert_journal_entry(entry)

        # Get all entries
        entries = repo.get_journal_entries(session.id)
        assert len(entries) == 3

        # Get by type
        observations = repo.get_journal_entries(
            session.id,
            entry_type=JournalEntryType.OBSERVATION
        )
        assert len(observations) == 1


class TestPredictionOperations:
    """Tests for prediction operations."""

    def test_insert_prediction(self, repo):
        """Test inserting a prediction."""
        session = SessionRecord(session_id="pred-test")
        repo.insert_session(session)

        prediction = PredictionRecord(
            session_id=session.id,
            turn_number=1,
            state_t0=">..",
            predicted_state_t1=".>.",
            actual_state_t1=".>.",
            is_correct=True,
            prediction_method="manual",
        )
        pred_id = repo.insert_prediction(prediction)
        assert pred_id > 0

    def test_get_accuracy_stats(self, repo):
        """Test getting accuracy statistics."""
        session = SessionRecord(session_id="accuracy-test")
        repo.insert_session(session)

        # Insert predictions
        for i, is_correct in enumerate([True, True, True, False, True]):
            prediction = PredictionRecord(
                session_id=session.id,
                turn_number=i,
                state_t0=f"state{i}",
                predicted_state_t1="predicted",
                actual_state_t1="actual" if not is_correct else "predicted",
                is_correct=is_correct,
            )
            repo.insert_prediction(prediction)

        stats = repo.get_accuracy_stats(session.id)
        assert stats["total_predictions"] == 5
        assert stats["correct_predictions"] == 4
        assert stats["accuracy"] == 0.8


class TestTheoremOperations:
    """Tests for theorem operations."""

    def test_insert_theorem(self, repo):
        """Test inserting a theorem."""
        session = SessionRecord(session_id="theorem-test")
        repo.insert_session(session)

        theorem = TheoremRecord(
            session_id=session.id,
            name="momentum_conservation",
            description="Momentum is conserved",
            law_ids_json='["law-1", "law-2"]',
            status=TheoremStatus.PROPOSED,
        )
        theorem_id = repo.insert_theorem(theorem)
        assert theorem_id > 0

    def test_get_theorem(self, repo):
        """Test retrieving a theorem by name."""
        session = SessionRecord(session_id="theorem-test-2")
        repo.insert_session(session)

        theorem = TheoremRecord(
            session_id=session.id,
            name="test_theorem",
            description="A test theorem",
            law_ids_json='[]',
        )
        repo.insert_theorem(theorem)

        retrieved = repo.get_theorem(session.id, "test_theorem")
        assert retrieved is not None
        assert retrieved.name == "test_theorem"
        assert retrieved.description == "A test theorem"

    def test_update_theorem(self, repo):
        """Test updating a theorem."""
        session = SessionRecord(session_id="theorem-test-3")
        repo.insert_session(session)

        theorem = TheoremRecord(
            session_id=session.id,
            name="updatable",
            description="Original",
            law_ids_json='[]',
            status=TheoremStatus.PROPOSED,
        )
        repo.insert_theorem(theorem)

        theorem.status = TheoremStatus.VALIDATED
        theorem.description = "Updated description"
        repo.update_theorem(theorem)

        retrieved = repo.get_theorem(session.id, "updatable")
        assert retrieved.status == TheoremStatus.VALIDATED
        assert retrieved.description == "Updated description"


class TestTransitionRuleOperations:
    """Tests for transition rule operations."""

    def test_insert_transition_rule(self, repo):
        """Test inserting a transition rule."""
        session = SessionRecord(session_id="rule-test")
        repo.insert_session(session)

        rule = TransitionRuleRecord(
            session_id=session.id,
            symbol=">",
            neighbor_config="..>",
            coordinate_class="any",
            result_symbol=">",
            confidence=1.0,
            evidence_count=10,
        )
        rule_id = repo.insert_transition_rule(rule)
        assert rule_id > 0

    def test_get_transition_completeness(self, repo):
        """Test checking transition rule completeness."""
        session = SessionRecord(session_id="completeness-test")
        repo.insert_session(session)

        # Insert rules for some symbols
        for symbol in [">", "<"]:
            rule = TransitionRuleRecord(
                session_id=session.id,
                symbol=symbol,
                neighbor_config="...",
                coordinate_class="any",
                result_symbol=".",
            )
            repo.insert_transition_rule(rule)

        completeness = repo.get_transition_completeness(session.id)
        assert ">" in completeness["covered_symbols"]
        assert "<" in completeness["covered_symbols"]
        assert "." in completeness["missing_symbols"]
        assert "X" in completeness["missing_symbols"]
        assert not completeness["is_complete"]

    def test_transition_rule_upsert(self, repo):
        """Test that transition rules support upsert."""
        session = SessionRecord(session_id="upsert-test")
        repo.insert_session(session)

        # Insert first rule
        rule1 = TransitionRuleRecord(
            session_id=session.id,
            symbol=">",
            neighbor_config="...",
            coordinate_class="any",
            result_symbol=">",
            evidence_count=5,
        )
        repo.insert_transition_rule(rule1)

        # Insert same rule (should update)
        rule2 = TransitionRuleRecord(
            session_id=session.id,
            symbol=">",
            neighbor_config="...",
            coordinate_class="any",
            result_symbol=">",
            evidence_count=5,
        )
        repo.insert_transition_rule(rule2)

        # Check that evidence was accumulated
        rules = repo.get_transition_rules(session.id)
        assert len(rules) == 1
        assert rules[0].evidence_count == 10
