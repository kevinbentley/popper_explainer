"""Tests for theorem domain models."""

import pytest

from src.theorem.models import (
    FailureBucket,
    FailureCluster,
    LawSnapshot,
    LawSupport,
    ObservableProposal,
    Theorem,
    TheoremBatch,
    TheoremStatus,
)


class TestTheoremStatus:
    def test_enum_values(self):
        assert TheoremStatus.ESTABLISHED.value == "Established"
        assert TheoremStatus.CONDITIONAL.value == "Conditional"
        assert TheoremStatus.CONJECTURAL.value == "Conjectural"

    def test_str_enum(self):
        # TheoremStatus is a str enum - .value gives the string value
        assert TheoremStatus.ESTABLISHED.value == "Established"


class TestFailureBucket:
    def test_enum_values(self):
        # PHASE-C bucket taxonomy
        assert FailureBucket.LOCAL_PATTERN.value == "LOCAL_PATTERN"
        assert FailureBucket.EVENTUALITY.value == "EVENTUALITY"
        assert FailureBucket.MONOTONICITY.value == "MONOTONICITY"
        assert FailureBucket.COLLISION_TRIGGERS.value == "COLLISION_TRIGGERS"
        assert FailureBucket.DEFINITION_GAP.value == "DEFINITION_GAP"
        assert FailureBucket.SYMMETRY.value == "SYMMETRY"
        assert FailureBucket.OTHER.value == "OTHER"

    def test_legacy_bucket_mapping(self):
        # Legacy names should be handled by _missing_
        assert FailureBucket("TEMPORAL_EVENTUAL") == FailureBucket.EVENTUALITY
        assert FailureBucket("SYMMETRY_MISAPPLIED") == FailureBucket.SYMMETRY


class TestLawSupport:
    def test_creation(self):
        support = LawSupport(law_id="law_001", role="confirms")
        assert support.law_id == "law_001"
        assert support.role == "confirms"

    def test_to_dict(self):
        support = LawSupport(law_id="law_001", role="constrains")
        d = support.to_dict()
        assert d == {"law_id": "law_001", "role": "constrains"}

    def test_from_dict(self):
        data = {"law_id": "law_002", "role": "refutes_alternative"}
        support = LawSupport.from_dict(data)
        assert support.law_id == "law_002"
        assert support.role == "refutes_alternative"

    def test_from_dict_default_role(self):
        data = {"law_id": "law_003"}
        support = LawSupport.from_dict(data)
        assert support.role == "confirms"


class TestTheorem:
    def test_creation(self):
        theorem = Theorem(
            theorem_id="thm_001",
            name="Test Theorem",
            status=TheoremStatus.ESTABLISHED,
            claim="Test claim",
            support=[LawSupport("law_001", "confirms")],
        )
        assert theorem.theorem_id == "thm_001"
        assert theorem.name == "Test Theorem"
        assert theorem.status == TheoremStatus.ESTABLISHED
        assert len(theorem.support) == 1

    def test_default_lists(self):
        theorem = Theorem(
            theorem_id="thm_001",
            name="Test",
            status=TheoremStatus.CONDITIONAL,
            claim="Test",
            support=[],
        )
        assert theorem.failure_modes == []
        assert theorem.missing_structure == []

    def test_to_dict(self):
        theorem = Theorem(
            theorem_id="thm_001",
            name="Test",
            status=TheoremStatus.CONJECTURAL,
            claim="Test claim",
            support=[LawSupport("law_001", "confirms")],
            failure_modes=["mode1"],
            missing_structure=["struct1"],
        )
        d = theorem.to_dict()
        assert d["theorem_id"] == "thm_001"
        assert d["status"] == "Conjectural"
        assert d["support"] == [{"law_id": "law_001", "role": "confirms"}]
        assert d["failure_modes"] == ["mode1"]

    def test_from_dict(self):
        data = {
            "theorem_id": "thm_002",
            "name": "Restored Theorem",
            "status": "Established",
            "claim": "Restored claim",
            "support": [{"law_id": "law_001", "role": "confirms"}],
            "failure_modes": ["fm1", "fm2"],
            "missing_structure": ["ms1"],
        }
        theorem = Theorem.from_dict(data)
        assert theorem.theorem_id == "thm_002"
        assert theorem.status == TheoremStatus.ESTABLISHED
        assert len(theorem.support) == 1
        assert len(theorem.failure_modes) == 2


class TestTheoremBatch:
    def test_creation(self):
        theorems = [
            Theorem(
                theorem_id="thm_001",
                name="Test",
                status=TheoremStatus.ESTABLISHED,
                claim="Test",
                support=[],
            )
        ]
        batch = TheoremBatch(
            theorems=theorems,
            rejections=[],
            prompt_hash="abc123",
            runtime_ms=100,
        )
        assert batch.accepted_count == 1
        assert batch.rejected_count == 0

    def test_rejections(self):
        batch = TheoremBatch(
            theorems=[],
            rejections=[
                ({"name": "bad"}, "Missing required field"),
                ({"name": "worse"}, "Invalid status"),
            ],
            prompt_hash="def456",
            runtime_ms=50,
        )
        assert batch.accepted_count == 0
        assert batch.rejected_count == 2


class TestLawSnapshot:
    def test_creation(self):
        snapshot = LawSnapshot(
            law_id="law_001",
            template="invariant",
            claim="count >= 0",
            status="PASS",
        )
        assert snapshot.law_id == "law_001"
        assert snapshot.status == "PASS"
        assert snapshot.counterexample is None

    def test_with_counterexample(self):
        snapshot = LawSnapshot(
            law_id="law_002",
            template="implication",
            claim="if X then Y",
            status="FAIL",
            counterexample={"initial_state": "><", "t_fail": 3},
        )
        assert snapshot.status == "FAIL"
        assert snapshot.counterexample["t_fail"] == 3

    def test_to_dict(self):
        snapshot = LawSnapshot(
            law_id="law_001",
            template="invariant",
            claim="test",
            status="PASS",
            power_metrics={"coverage": 0.95},
        )
        d = snapshot.to_dict()
        assert d["law_id"] == "law_001"
        assert d["power_metrics"] == {"coverage": 0.95}


class TestFailureCluster:
    def test_creation(self):
        # PHASE-C: Uses bucket_tags (multi-label) instead of single bucket
        cluster = FailureCluster(
            cluster_id="fc_001",
            bucket_tags=["LOCAL_PATTERN"],
            semantic_cluster_idx=0,
            theorem_ids=["thm_001", "thm_002"],
            centroid_signature="local pattern adjacent",
            avg_similarity=0.75,
        )
        assert cluster.cluster_id == "fc_001"
        assert cluster.bucket == FailureBucket.LOCAL_PATTERN  # Legacy property
        assert cluster.bucket_tags == ["LOCAL_PATTERN"]
        assert len(cluster.theorem_ids) == 2

    def test_multi_label_bucket_tags(self):
        cluster = FailureCluster(
            cluster_id="fc_001",
            bucket_tags=["LOCAL_PATTERN", "COLLISION_TRIGGERS"],
            semantic_cluster_idx=0,
            theorem_ids=["thm_001"],
            centroid_signature="test",
            avg_similarity=1.0,
        )
        # Multiple tags
        assert "LOCAL_PATTERN" in cluster.bucket_tags
        assert "COLLISION_TRIGGERS" in cluster.bucket_tags
        # Legacy property returns first
        assert cluster.bucket == FailureBucket.LOCAL_PATTERN

    def test_to_dict(self):
        cluster = FailureCluster(
            cluster_id="fc_002",
            bucket_tags=["MONOTONICITY"],
            semantic_cluster_idx=1,
            theorem_ids=["thm_003"],
            centroid_signature="monotone increase",
            avg_similarity=1.0,
            top_keywords=[("increase", 0.8), ("monotone", 0.6)],
            recommended_action="OBSERVABLE",
        )
        d = cluster.to_dict()
        assert d["bucket"] == "MONOTONICITY"  # Legacy field
        assert d["bucket_tags"] == ["MONOTONICITY"]
        assert d["theorem_ids"] == ["thm_003"]
        assert d["top_keywords"] == [("increase", 0.8), ("monotone", 0.6)]
        assert d["recommended_action"] == "OBSERVABLE"


class TestObservableProposal:
    def test_creation(self):
        proposal = ObservableProposal(
            proposal_id="prop_001",
            cluster_id="fc_001",
            observable_name="adjacent_count",
            observable_expr="count_adjacent(state)",
            rationale="Addresses local pattern failures",
            priority="high",
        )
        assert proposal.proposal_id == "prop_001"
        assert proposal.priority == "high"

    def test_to_dict(self):
        proposal = ObservableProposal(
            proposal_id="prop_002",
            cluster_id="fc_002",
            observable_name="delta_count",
            observable_expr="count(t) - count(t-1)",
            rationale="Track changes",
            priority="medium",
        )
        d = proposal.to_dict()
        assert d["observable_name"] == "delta_count"
        assert d["priority"] == "medium"
