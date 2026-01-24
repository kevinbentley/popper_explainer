"""Tests for the persistence layer (Phase 0)."""

import json
import tempfile
from pathlib import Path

import pytest

from src.db.models import (
    CaseSetRecord,
    CounterexampleClassRecord,
    CounterexampleRecord,
    EvaluationRecord,
    FailureClassificationRecord,
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


class TestFailureClassificationOperations:
    """Tests for failure classification persistence."""

    def test_insert_and_get_failure_classification(self, repo):
        """Insert and retrieve a failure classification."""
        repo.insert_law(LawRecord("fc_law", "h1", "invariant", "{}"))
        eval_id = repo.insert_evaluation(EvaluationRecord(
            "fc_law", "h1", "FAIL", "c1", "s1", 50, 50
        ))

        fc = FailureClassificationRecord(
            evaluation_id=eval_id,
            law_id="fc_law",
            failure_class="type_b_novel_counterexample",
            counterexample_class_id="converging_pair",
            is_known_class=False,
            confidence=0.95,
            features_json='{"grid_length": 3}',
            reasoning="Contains converging pair pattern",
            actionable=False,
        )
        fc_id = repo.insert_failure_classification(fc)
        assert fc_id is not None

        retrieved = repo.get_failure_classification(eval_id)
        assert retrieved is not None
        assert retrieved.failure_class == "type_b_novel_counterexample"
        assert retrieved.counterexample_class_id == "converging_pair"
        assert retrieved.is_known_class is False
        assert retrieved.confidence == 0.95

    def test_list_failure_classifications_by_class(self, repo):
        """List failure classifications filtered by failure class."""
        repo.insert_law(LawRecord("fc_law1", "h1", "invariant", "{}"))
        repo.insert_law(LawRecord("fc_law2", "h2", "invariant", "{}"))

        eval_id1 = repo.insert_evaluation(EvaluationRecord(
            "fc_law1", "h1", "FAIL", "c1", "s1", 50, 50
        ))
        eval_id2 = repo.insert_evaluation(EvaluationRecord(
            "fc_law2", "h2", "UNKNOWN", "c1", "s1", 10, 5
        ))

        repo.insert_failure_classification(FailureClassificationRecord(
            evaluation_id=eval_id1,
            law_id="fc_law1",
            failure_class="type_b_novel_counterexample",
            is_known_class=False,
            actionable=False,
        ))
        repo.insert_failure_classification(FailureClassificationRecord(
            evaluation_id=eval_id2,
            law_id="fc_law2",
            failure_class="type_c_process_issue",
            is_known_class=False,
            actionable=True,
        ))

        type_b = repo.list_failure_classifications(
            failure_class="type_b_novel_counterexample"
        )
        assert len(type_b) == 1
        assert type_b[0].law_id == "fc_law1"

        actionable = repo.list_failure_classifications(actionable=True)
        assert len(actionable) == 1
        assert actionable[0].law_id == "fc_law2"

    def test_failure_classification_summary(self, repo):
        """Get summary counts by failure class."""
        repo.insert_law(LawRecord("sum_fc1", "h1", "invariant", "{}"))
        repo.insert_law(LawRecord("sum_fc2", "h2", "invariant", "{}"))
        repo.insert_law(LawRecord("sum_fc3", "h3", "invariant", "{}"))

        eval_id1 = repo.insert_evaluation(EvaluationRecord(
            "sum_fc1", "h1", "FAIL", "c1", "s1", 50, 50
        ))
        eval_id2 = repo.insert_evaluation(EvaluationRecord(
            "sum_fc2", "h2", "FAIL", "c1", "s1", 50, 50
        ))
        eval_id3 = repo.insert_evaluation(EvaluationRecord(
            "sum_fc3", "h3", "UNKNOWN", "c1", "s1", 10, 5
        ))

        repo.insert_failure_classification(FailureClassificationRecord(
            eval_id1, "sum_fc1", "type_a_known_counterexample", False, False
        ))
        repo.insert_failure_classification(FailureClassificationRecord(
            eval_id2, "sum_fc2", "type_a_known_counterexample", False, False
        ))
        repo.insert_failure_classification(FailureClassificationRecord(
            eval_id3, "sum_fc3", "type_c_process_issue", False, True
        ))

        summary = repo.get_failure_classification_summary()
        assert summary.get("type_a_known_counterexample", 0) == 2
        assert summary.get("type_c_process_issue", 0) == 1

    def test_counterexample_class_counts(self, repo):
        """Get counts by counterexample class."""
        repo.insert_law(LawRecord("cc_law1", "h1", "invariant", "{}"))
        repo.insert_law(LawRecord("cc_law2", "h2", "invariant", "{}"))
        repo.insert_law(LawRecord("cc_law3", "h3", "invariant", "{}"))

        eval_id1 = repo.insert_evaluation(EvaluationRecord(
            "cc_law1", "h1", "FAIL", "c1", "s1", 50, 50
        ))
        eval_id2 = repo.insert_evaluation(EvaluationRecord(
            "cc_law2", "h2", "FAIL", "c1", "s1", 50, 50
        ))
        eval_id3 = repo.insert_evaluation(EvaluationRecord(
            "cc_law3", "h3", "FAIL", "c1", "s1", 50, 50
        ))

        repo.insert_failure_classification(FailureClassificationRecord(
            eval_id1, "cc_law1", "type_a_known_counterexample", False, False,
            counterexample_class_id="converging_pair"
        ))
        repo.insert_failure_classification(FailureClassificationRecord(
            eval_id2, "cc_law2", "type_a_known_counterexample", False, False,
            counterexample_class_id="converging_pair"
        ))
        repo.insert_failure_classification(FailureClassificationRecord(
            eval_id3, "cc_law3", "type_b_novel_counterexample", False, False,
            counterexample_class_id="x_emission"
        ))

        counts = repo.get_counterexample_class_counts()
        assert counts.get("converging_pair", 0) == 2
        assert counts.get("x_emission", 0) == 1


class TestCounterexampleClassRegistry:
    """Tests for counterexample class registry persistence."""

    def test_upsert_counterexample_class(self, repo):
        """Insert and update counterexample classes."""
        class_record = CounterexampleClassRecord(
            class_id="converging_pair",
            description="Two particles converging: >.<",
            example_state=">.<",
        )
        class_id = repo.upsert_counterexample_class(class_record)
        assert class_id is not None

        retrieved = repo.get_counterexample_class("converging_pair")
        assert retrieved is not None
        assert retrieved.occurrence_count == 1
        assert retrieved.description == "Two particles converging: >.<"

        # Upsert again - should increment count
        repo.upsert_counterexample_class(CounterexampleClassRecord(
            class_id="converging_pair"
        ))
        retrieved = repo.get_counterexample_class("converging_pair")
        assert retrieved.occurrence_count == 2

    def test_list_counterexample_classes(self, repo):
        """List classes ordered by occurrence count."""
        repo.upsert_counterexample_class(CounterexampleClassRecord(
            class_id="x_emission", description="X particle emission"
        ))
        repo.upsert_counterexample_class(CounterexampleClassRecord(
            class_id="converging_pair", description="Converging pair"
        ))
        # Increment converging_pair twice
        repo.upsert_counterexample_class(CounterexampleClassRecord(
            class_id="converging_pair"
        ))
        repo.upsert_counterexample_class(CounterexampleClassRecord(
            class_id="converging_pair"
        ))

        classes = repo.list_counterexample_classes()
        assert len(classes) == 2
        # Should be ordered by count descending
        assert classes[0].class_id == "converging_pair"
        assert classes[0].occurrence_count == 3
        assert classes[1].class_id == "x_emission"
        assert classes[1].occurrence_count == 1

    def test_get_known_class_ids(self, repo):
        """Get set of known class IDs."""
        repo.upsert_counterexample_class(CounterexampleClassRecord(
            class_id="converging_pair"
        ))
        repo.upsert_counterexample_class(CounterexampleClassRecord(
            class_id="x_emission"
        ))
        repo.upsert_counterexample_class(CounterexampleClassRecord(
            class_id="high_density"
        ))

        known = repo.get_known_class_ids()
        assert "converging_pair" in known
        assert "x_emission" in known
        assert "high_density" in known
        assert len(known) == 3
