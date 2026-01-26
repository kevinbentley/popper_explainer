"""Tests for reflection persistence (DB round-trip)."""

import json
import os
import tempfile
import pytest

from src.db.models import (
    ReflectionSessionRecord,
    SevereTestCommandRecord,
    StandardModelRecord,
)
from src.db.repo import Repository
from src.reflection.models import (
    AuditorResult,
    ArchiveDecision,
    ConflictEntry,
    DerivedObservable,
    HiddenVariable,
    ReflectionResult,
    SevereTestCommand,
    StandardModel,
    TheoristResult,
)
from src.reflection.persistence import (
    build_standard_model_summary,
    load_latest_standard_model,
    save_reflection_result,
)


@pytest.fixture
def repo():
    """Create a temporary database repository."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    r = Repository(db_path=path)
    r.connect()
    yield r
    r.close()
    os.unlink(path)


@pytest.fixture
def run_id(repo):
    """Create a test orchestration run and return its ID."""
    from src.db.orchestration_models import OrchestrationRunRecord
    run = OrchestrationRunRecord(
        run_id="test-run-1",
        status="running",
        current_phase="discovery",
        config_json="{}",
    )
    repo.insert_orchestration_run(run)
    return "test-run-1"


class TestStandardModelPersistence:
    def test_insert_and_retrieve(self, repo, run_id):
        """Insert a standard model and retrieve it."""
        record = StandardModelRecord(
            run_id=run_id,
            iteration_id=5,
            version=1,
            fixed_laws_json=json.dumps(["L1", "L2"]),
            archived_laws_json=json.dumps(["L_old"]),
            derived_observables_json=json.dumps([{
                "name": "net_flow",
                "expression": "count(A) - count(B)",
                "rationale": "test",
                "source_laws": ["L1"],
            }]),
            causal_narrative="The universe conserves stuff.",
            hidden_variables_json=json.dumps([]),
            k_decomposition="Established: conservation.",
            confidence=0.7,
        )
        sm_id = repo.insert_standard_model(record)
        assert sm_id > 0

        retrieved = repo.get_latest_standard_model(run_id)
        assert retrieved is not None
        assert retrieved.version == 1
        assert json.loads(retrieved.fixed_laws_json) == ["L1", "L2"]
        assert retrieved.confidence == 0.7

    def test_versioning(self, repo, run_id):
        """Multiple versions are tracked and latest is retrieved."""
        for v in range(1, 4):
            record = StandardModelRecord(
                run_id=run_id,
                version=v,
                fixed_laws_json=json.dumps([f"L{v}"]),
                archived_laws_json=json.dumps([]),
            )
            repo.insert_standard_model(record)

        latest = repo.get_latest_standard_model(run_id)
        assert latest is not None
        assert latest.version == 3
        assert json.loads(latest.fixed_laws_json) == ["L3"]

    def test_get_by_version(self, repo, run_id):
        """Retrieve a specific version."""
        for v in [1, 2]:
            repo.insert_standard_model(StandardModelRecord(
                run_id=run_id,
                version=v,
                fixed_laws_json=json.dumps([f"L{v}"]),
                archived_laws_json=json.dumps([]),
            ))

        v1 = repo.get_standard_model_by_version(run_id, 1)
        assert v1 is not None
        assert json.loads(v1.fixed_laws_json) == ["L1"]

    def test_next_version(self, repo, run_id):
        """Next version is computed correctly."""
        assert repo.get_next_standard_model_version(run_id) == 1

        repo.insert_standard_model(StandardModelRecord(
            run_id=run_id, version=1,
            fixed_laws_json="[]", archived_laws_json="[]",
        ))
        assert repo.get_next_standard_model_version(run_id) == 2


class TestReflectionSessionPersistence:
    def test_insert_and_retrieve(self, repo, run_id):
        """Insert and retrieve a reflection session."""
        record = ReflectionSessionRecord(
            run_id=run_id,
            iteration_index=5,
            trigger_reason="periodic",
            auditor_result_json=json.dumps({"summary": "ok"}),
            theorist_result_json=json.dumps({"confidence": 0.5}),
            conflicts_found=2,
            laws_archived=1,
            hidden_variables_postulated=3,
            standard_model_version=1,
            runtime_ms=1500,
        )
        session_id = repo.insert_reflection_session(record)
        assert session_id > 0

        retrieved = repo.get_reflection_session(run_id, 5)
        assert retrieved is not None
        assert retrieved.trigger_reason == "periodic"
        assert retrieved.conflicts_found == 2
        assert retrieved.runtime_ms == 1500

    def test_list_sessions(self, repo, run_id):
        """List sessions for a run."""
        for i in range(3):
            repo.insert_reflection_session(ReflectionSessionRecord(
                run_id=run_id,
                iteration_index=i * 3,
                trigger_reason="periodic",
            ))

        sessions = repo.list_reflection_sessions(run_id)
        assert len(sessions) == 3

    def test_no_session_returns_none(self, repo, run_id):
        """Non-existent session returns None."""
        result = repo.get_reflection_session(run_id, 999)
        assert result is None


class TestSevereTestCommandPersistence:
    def test_insert_and_list_unconsumed(self, repo, run_id):
        """Insert commands and list unconsumed ones."""
        # Create a session first
        session_id = repo.insert_reflection_session(ReflectionSessionRecord(
            run_id=run_id, iteration_index=3, trigger_reason="periodic",
        ))

        # Insert commands
        for priority in ["high", "medium", "low"]:
            repo.insert_severe_test_command(SevereTestCommandRecord(
                run_id=run_id,
                reflection_session_id=session_id,
                command_type="initial_condition",
                description=f"{priority} priority test",
                priority=priority,
            ))

        commands = repo.list_unconsumed_severe_test_commands(run_id)
        assert len(commands) == 3
        # Should be ordered by priority: high first
        assert commands[0].priority == "high"
        assert commands[1].priority == "medium"
        assert commands[2].priority == "low"

    def test_mark_consumed(self, repo, run_id):
        """Mark a command as consumed."""
        session_id = repo.insert_reflection_session(ReflectionSessionRecord(
            run_id=run_id, iteration_index=3, trigger_reason="periodic",
        ))

        cmd_id = repo.insert_severe_test_command(SevereTestCommandRecord(
            run_id=run_id,
            reflection_session_id=session_id,
            command_type="initial_condition",
            description="test",
        ))

        repo.mark_severe_test_consumed(cmd_id)

        # Should not appear in unconsumed list
        commands = repo.list_unconsumed_severe_test_commands(run_id)
        assert len(commands) == 0


class TestSaveReflectionResult:
    def test_full_save(self, repo, run_id):
        """Save a complete reflection result and verify DB state."""
        result = ReflectionResult(
            auditor_result=AuditorResult(
                conflicts=[ConflictEntry(
                    law_id="L1", description="test conflict", severity="high",
                )],
                archives=[ArchiveDecision(law_id="L_old", reason="redundant")],
                summary="Found issues.",
            ),
            theorist_result=TheoristResult(
                derived_observables=[DerivedObservable(
                    name="net", expression="A-B", rationale="test",
                )],
                hidden_variables=[HiddenVariable(
                    name="hv1", description="test hv",
                    evidence="anomaly", testable_prediction="filter K",
                )],
                causal_narrative="Test narrative.",
                confidence=0.7,
            ),
            standard_model=StandardModel(
                fixed_laws=["L1", "L2"],
                archived_laws=["L_old"],
                derived_observables=[DerivedObservable(
                    name="net", expression="A-B", rationale="test",
                )],
                causal_narrative="Test narrative.",
                confidence=0.7,
                version=1,
            ),
            severe_test_commands=[
                SevereTestCommand(
                    command_type="initial_condition",
                    description="test with all A",
                    priority="high",
                ),
                SevereTestCommand(
                    command_type="topology_test",
                    description="test boundary",
                    priority="medium",
                ),
            ],
            research_log_addendum="Reflection complete.",
            runtime_ms=2000,
        )

        session_id, sm_id = save_reflection_result(
            repo=repo,
            run_id=run_id,
            iteration_index=5,
            result=result,
        )

        assert session_id > 0
        assert sm_id > 0

        # Verify standard model
        sm = repo.get_latest_standard_model(run_id)
        assert sm is not None
        assert sm.version == 1
        assert json.loads(sm.fixed_laws_json) == ["L1", "L2"]

        # Verify session
        session = repo.get_reflection_session(run_id, 5)
        assert session is not None
        assert session.conflicts_found == 1
        assert session.laws_archived == 1
        assert session.hidden_variables_postulated == 1
        assert session.runtime_ms == 2000

        # Verify severe test commands
        commands = repo.list_unconsumed_severe_test_commands(run_id)
        assert len(commands) == 2
        assert commands[0].priority == "high"


class TestLoadLatestStandardModel:
    def test_load_round_trip(self, repo, run_id):
        """Save and load a standard model via domain objects."""
        result = ReflectionResult(
            auditor_result=AuditorResult(),
            theorist_result=TheoristResult(confidence=0.6),
            standard_model=StandardModel(
                fixed_laws=["L1"],
                archived_laws=["L_old"],
                derived_observables=[DerivedObservable(
                    name="d1", expression="A+B", rationale="r",
                    source_laws=["L1"],
                )],
                hidden_variables=[HiddenVariable(
                    name="h1", description="d", evidence="e",
                    testable_prediction="p",
                )],
                causal_narrative="narrative",
                k_decomposition="k",
                confidence=0.8,
                version=1,
            ),
            severe_test_commands=[],
        )

        save_reflection_result(repo, run_id, 5, result)

        loaded = load_latest_standard_model(repo, run_id)
        assert loaded is not None
        assert loaded.fixed_laws == ["L1"]
        assert loaded.archived_laws == ["L_old"]
        assert len(loaded.derived_observables) == 1
        assert loaded.derived_observables[0].name == "d1"
        assert len(loaded.hidden_variables) == 1
        assert loaded.hidden_variables[0].name == "h1"
        assert loaded.causal_narrative == "narrative"
        assert loaded.confidence == 0.8

    def test_no_model_returns_none(self, repo, run_id):
        """No model returns None."""
        result = load_latest_standard_model(repo, run_id)
        assert result is None


class TestBuildStandardModelSummary:
    def test_summary_structure(self):
        """Summary contains expected fields."""
        model = StandardModel(
            fixed_laws=["L1", "L2", "L3"],
            archived_laws=["L_old"],
            derived_observables=[DerivedObservable(
                name="net", expression="A-B", rationale="r",
            )],
            hidden_variables=[HiddenVariable(
                name="hv", description="d", evidence="e",
                testable_prediction="p",
            )],
            causal_narrative="A long narrative " * 100,
            confidence=0.75,
            version=2,
        )

        summary = build_standard_model_summary(model)

        assert summary["version"] == 2
        assert summary["fixed_law_count"] == 3
        assert summary["archived_law_count"] == 1
        assert summary["archived_law_ids"] == ["L_old"]
        assert len(summary["derived_observables"]) == 1
        assert len(summary["hidden_variables"]) == 1
        assert summary["confidence"] == 0.75
        # Narrative should be truncated
        assert len(summary["causal_narrative_excerpt"]) <= 500
