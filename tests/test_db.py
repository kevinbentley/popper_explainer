"""Tests for the persistence layer (Phase 0)."""

import json
import tempfile
from pathlib import Path

import pytest

from src.db.models import (
    CaseSetRecord,
    CounterexampleRecord,
    EvaluationRecord,
    LawRecord,
)
from src.db.repo import Repository


@pytest.fixture
def repo():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    repo = Repository(db_path)
    repo.connect()
    yield repo
    repo.close()
    db_path.unlink()


class TestLawOperations:
    def test_insert_and_get_law(self, repo):
        law = LawRecord(
            law_id="test_law_1",
            law_hash="abc123",
            template="invariant",
            law_json='{"claim": "R_total(t) == R_total(0)"}',
        )
        law_id = repo.insert_law(law)
        assert law_id is not None

        retrieved = repo.get_law("test_law_1")
        assert retrieved is not None
        assert retrieved.law_id == "test_law_1"
        assert retrieved.law_hash == "abc123"
        assert retrieved.template == "invariant"

    def test_get_law_by_hash(self, repo):
        law = LawRecord(
            law_id="hash_test_law",
            law_hash="unique_hash_xyz",
            template="monotone",
            law_json="{}",
        )
        repo.insert_law(law)

        retrieved = repo.get_law_by_hash("unique_hash_xyz")
        assert retrieved is not None
        assert retrieved.law_id == "hash_test_law"

    def test_list_laws_by_template(self, repo):
        repo.insert_law(LawRecord("inv1", "h1", "invariant", "{}"))
        repo.insert_law(LawRecord("inv2", "h2", "invariant", "{}"))
        repo.insert_law(LawRecord("mon1", "h3", "monotone", "{}"))

        invariants = repo.list_laws(template="invariant")
        assert len(invariants) == 2

        monotones = repo.list_laws(template="monotone")
        assert len(monotones) == 1

    def test_duplicate_law_id_raises(self, repo):
        law = LawRecord("dup_test", "h1", "invariant", "{}")
        repo.insert_law(law)

        with pytest.raises(Exception):  # sqlite3.IntegrityError
            repo.insert_law(LawRecord("dup_test", "h2", "invariant", "{}"))


class TestCaseSetOperations:
    def test_insert_and_get_case_set(self, repo):
        case_set = CaseSetRecord(
            generator_family="random_density_sweep",
            params_hash="params_abc",
            seed=42,
            cases_json='[{"initial_state": "..><.."}]',
            case_count=1,
        )
        cs_id = repo.insert_case_set(case_set)
        assert cs_id is not None

        retrieved = repo.get_case_set("random_density_sweep", "params_abc", 42)
        assert retrieved is not None
        assert retrieved.case_count == 1
        assert retrieved.seed == 42

    def test_case_set_uniqueness(self, repo):
        cs1 = CaseSetRecord("gen1", "params1", 100, "[]", 0)
        repo.insert_case_set(cs1)

        # Same key should fail
        with pytest.raises(Exception):
            repo.insert_case_set(CaseSetRecord("gen1", "params1", 100, "[]", 0))

        # Different seed is OK
        cs2 = CaseSetRecord("gen1", "params1", 101, "[]", 0)
        repo.insert_case_set(cs2)


class TestEvaluationOperations:
    def test_insert_and_get_evaluation(self, repo):
        # First insert a law
        repo.insert_law(LawRecord("eval_law", "h1", "invariant", "{}"))

        eval_record = EvaluationRecord(
            law_id="eval_law",
            law_hash="h1",
            status="PASS",
            harness_config_hash="config_123",
            sim_hash="sim_456",
            cases_attempted=100,
            cases_used=95,
            power_metrics_json='{"coverage": 0.8}',
        )
        eval_id = repo.insert_evaluation(eval_record)
        assert eval_id is not None

        retrieved = repo.get_evaluation(eval_id)
        assert retrieved is not None
        assert retrieved.status == "PASS"
        assert retrieved.cases_used == 95

    def test_get_latest_evaluation(self, repo):
        repo.insert_law(LawRecord("latest_law", "h1", "invariant", "{}"))

        # Insert two evaluations
        repo.insert_evaluation(EvaluationRecord(
            law_id="latest_law",
            law_hash="h1",
            status="UNKNOWN",
            harness_config_hash="config1",
            sim_hash="sim1",
            cases_attempted=10,
            cases_used=5,
            reason_code="inconclusive_low_power"
        ))
        repo.insert_evaluation(EvaluationRecord(
            law_id="latest_law",
            law_hash="h1",
            status="PASS",
            harness_config_hash="config2",
            sim_hash="sim1",
            cases_attempted=100,
            cases_used=95
        ))

        latest = repo.get_latest_evaluation("latest_law")
        assert latest is not None
        assert latest.status == "PASS"

    def test_list_evaluations_by_status(self, repo):
        repo.insert_law(LawRecord("status_law", "h1", "invariant", "{}"))

        repo.insert_evaluation(EvaluationRecord(
            "status_law", "h1", "PASS", "c1", "s1", 100, 90
        ))
        repo.insert_evaluation(EvaluationRecord(
            "status_law", "h1", "FAIL", "c2", "s1", 50, 50
        ))

        passes = repo.list_evaluations(status="PASS")
        assert len(passes) == 1
        assert passes[0].status == "PASS"


class TestCounterexampleOperations:
    def test_insert_and_get_counterexample(self, repo):
        repo.insert_law(LawRecord("cx_law", "h1", "invariant", "{}"))
        eval_id = repo.insert_evaluation(EvaluationRecord(
            "cx_law", "h1", "FAIL", "c1", "s1", 50, 50
        ))

        cx = CounterexampleRecord(
            evaluation_id=eval_id,
            law_id="cx_law",
            initial_state="..><..",
            config_json='{"grid_length": 6}',
            t_max=10,
            t_fail=3,
            minimized=True,
        )
        cx_id = repo.insert_counterexample(cx)
        assert cx_id is not None

        counterexamples = repo.get_counterexamples_for_law("cx_law")
        assert len(counterexamples) == 1
        assert counterexamples[0].t_fail == 3
        assert counterexamples[0].minimized is True


class TestAuditLog:
    def test_audit_logging(self, repo):
        repo.log_audit("evaluate", "law", "test_law_1", {"result": "PASS"})
        repo.log_audit("propose", "law", "test_law_2", {"source": "llm"})

        logs = repo.get_audit_logs(operation="evaluate")
        assert len(logs) == 1
        assert logs[0].entity_id == "test_law_1"

        all_logs = repo.get_audit_logs()
        assert len(all_logs) == 2


class TestSummaryQueries:
    def test_evaluation_summary(self, repo):
        repo.insert_law(LawRecord("sum_law1", "h1", "invariant", "{}"))
        repo.insert_law(LawRecord("sum_law2", "h2", "invariant", "{}"))

        repo.insert_evaluation(EvaluationRecord(
            "sum_law1", "h1", "PASS", "c1", "s1", 100, 90
        ))
        repo.insert_evaluation(EvaluationRecord(
            "sum_law2", "h2", "FAIL", "c1", "s1", 50, 50
        ))
        repo.insert_evaluation(EvaluationRecord(
            "sum_law1", "h1", "PASS", "c2", "s1", 100, 95
        ))

        summary = repo.get_evaluation_summary()
        assert summary.get("PASS", 0) == 2
        assert summary.get("FAIL", 0) == 1

    def test_get_laws_by_status(self, repo):
        repo.insert_law(LawRecord("status1", "h1", "invariant", "{}"))
        repo.insert_law(LawRecord("status2", "h2", "invariant", "{}"))

        repo.insert_evaluation(EvaluationRecord(
            "status1", "h1", "PASS", "c1", "s1", 100, 90
        ))
        repo.insert_evaluation(EvaluationRecord(
            "status2", "h2", "FAIL", "c1", "s1", 50, 50
        ))

        passing = repo.get_laws_by_status("PASS")
        assert "status1" in passing
        assert "status2" not in passing


class TestContextManager:
    def test_context_manager_usage(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        with Repository(db_path) as repo:
            repo.insert_law(LawRecord("ctx_law", "h1", "invariant", "{}"))
            law = repo.get_law("ctx_law")
            assert law is not None

        db_path.unlink()
