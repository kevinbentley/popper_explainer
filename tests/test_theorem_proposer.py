"""Tests for observable proposer."""

import pytest

from src.theorem.models import FailureBucket, FailureCluster
from src.theorem.observable_proposer import (
    BUCKET_OBSERVABLE_RULES,
    ObservableProposer,
)


def make_cluster(
    cluster_id: str,
    bucket: FailureBucket,
    theorem_ids: list[str] | None = None,
) -> FailureCluster:
    """Helper to create test clusters."""
    return FailureCluster(
        cluster_id=cluster_id,
        bucket_tags=[bucket.value],  # Convert bucket enum to bucket_tags list
        semantic_cluster_idx=0,
        theorem_ids=theorem_ids or ["thm_001"],
        centroid_signature="test signature",
        avg_similarity=0.8,
    )


class TestObservableProposer:
    @pytest.fixture
    def proposer(self):
        return ObservableProposer()

    def test_propose_from_local_pattern_cluster(self, proposer):
        cluster = make_cluster("fc_001", FailureBucket.LOCAL_PATTERN)
        proposals = proposer.propose_from_cluster(cluster)

        assert len(proposals) > 0
        # Should have high-priority proposals
        assert any(p.priority == "high" for p in proposals)
        # All proposals should reference the cluster
        assert all(p.cluster_id == "fc_001" for p in proposals)

    def test_propose_from_temporal_cluster(self, proposer):
        cluster = make_cluster("fc_002", FailureBucket.EVENTUALITY)
        proposals = proposer.propose_from_cluster(cluster)

        assert len(proposals) > 0
        # Check for temporal-specific observables
        names = {p.observable_name for p in proposals}
        assert "time_since_last_change" in names or "collision_forecast" in names

    def test_propose_from_monotonicity_cluster(self, proposer):
        cluster = make_cluster("fc_003", FailureBucket.MONOTONICITY)
        proposals = proposer.propose_from_cluster(cluster)

        assert len(proposals) > 0
        names = {p.observable_name for p in proposals}
        assert "delta_observable" in names

    def test_propose_from_other_bucket_empty(self, proposer):
        cluster = make_cluster("fc_004", FailureBucket.OTHER)
        proposals = proposer.propose_from_cluster(cluster)

        # OTHER bucket has no default rules
        assert proposals == []

    def test_propose_from_all_clusters(self, proposer):
        clusters = [
            make_cluster("fc_001", FailureBucket.LOCAL_PATTERN),
            make_cluster("fc_002", FailureBucket.EVENTUALITY),
            make_cluster("fc_003", FailureBucket.MONOTONICITY),
        ]
        proposals = proposer.propose_from_all_clusters(clusters)

        # Should have proposals from multiple buckets
        assert len(proposals) > 3

    def test_dedupe_proposals(self, proposer):
        # Two clusters with same bucket should dedupe observable names
        clusters = [
            make_cluster("fc_001", FailureBucket.LOCAL_PATTERN),
            make_cluster("fc_002", FailureBucket.LOCAL_PATTERN),
        ]
        proposals = proposer.propose_from_all_clusters(clusters, dedupe=True)

        # No duplicate observable names
        names = [p.observable_name for p in proposals]
        assert len(names) == len(set(names))

    def test_no_dedupe_allows_duplicates(self, proposer):
        clusters = [
            make_cluster("fc_001", FailureBucket.LOCAL_PATTERN),
            make_cluster("fc_002", FailureBucket.LOCAL_PATTERN),
        ]
        proposals = proposer.propose_from_all_clusters(clusters, dedupe=False)

        # Should have duplicates (same observable from both clusters)
        names = [p.observable_name for p in proposals]
        assert len(names) > len(set(names))

    def test_proposals_sorted_by_priority(self, proposer):
        clusters = [
            make_cluster("fc_001", FailureBucket.LOCAL_PATTERN),
            make_cluster("fc_002", FailureBucket.EVENTUALITY),
        ]
        proposals = proposer.propose_from_all_clusters(clusters)

        # High priority should come first
        priorities = [p.priority for p in proposals]
        high_indices = [i for i, p in enumerate(priorities) if p == "high"]
        medium_indices = [i for i, p in enumerate(priorities) if p == "medium"]
        low_indices = [i for i, p in enumerate(priorities) if p == "low"]

        if high_indices and medium_indices:
            assert max(high_indices) < min(medium_indices)
        if medium_indices and low_indices:
            assert max(medium_indices) < min(low_indices)

    def test_get_high_priority_proposals(self, proposer):
        clusters = [
            make_cluster("fc_001", FailureBucket.LOCAL_PATTERN),
            make_cluster("fc_002", FailureBucket.EVENTUALITY),
        ]
        proposals = proposer.get_high_priority_proposals(clusters, max_count=3)

        assert len(proposals) <= 3
        assert all(p.priority == "high" for p in proposals)

    def test_proposal_has_rationale(self, proposer):
        cluster = make_cluster(
            "fc_001",
            FailureBucket.LOCAL_PATTERN,
            theorem_ids=["thm_001", "thm_002"],
        )
        proposals = proposer.propose_from_cluster(cluster)

        for p in proposals:
            assert p.rationale
            assert "LOCAL_PATTERN" in p.rationale
            assert "2 theorem" in p.rationale

    def test_proposal_ids_unique(self, proposer):
        clusters = [
            make_cluster("fc_001", FailureBucket.LOCAL_PATTERN),
            make_cluster("fc_002", FailureBucket.EVENTUALITY),
        ]
        proposals = proposer.propose_from_all_clusters(clusters, dedupe=False)

        # All proposal IDs should be unique
        ids = [p.proposal_id for p in proposals]
        assert len(ids) == len(set(ids))


class TestBucketObservableRulesCompleteness:
    def test_all_observable_action_buckets_have_rules(self):
        """Ensure buckets that have OBSERVABLE action have observable rules.

        Note: DEFINITION_GAP (SCHEMA_FIX action) and OTHER have no observable rules.
        EVENTUALITY (GATING action) may have some but is optional.
        """
        # Buckets that should have observable rules
        observable_buckets = ["LOCAL_PATTERN", "COLLISION_TRIGGERS", "MONOTONICITY", "SYMMETRY"]
        for bucket_name in observable_buckets:
            assert bucket_name in BUCKET_OBSERVABLE_RULES, f"{bucket_name} missing from rules"
            assert len(BUCKET_OBSERVABLE_RULES[bucket_name]) > 0, f"{bucket_name} has empty rules"

    def test_all_templates_have_required_fields(self):
        """Ensure all observable templates have required fields."""
        for bucket, templates in BUCKET_OBSERVABLE_RULES.items():
            for template in templates:
                assert template.name
                assert template.expr
                assert template.description
                assert template.priority in ("high", "medium", "low")
