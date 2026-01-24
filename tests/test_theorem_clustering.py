"""Tests for theorem clustering.

PHASE-C updates:
- New bucket taxonomy (DEFINITION_GAP, COLLISION_TRIGGERS, EVENTUALITY, SYMMETRY)
- Multi-label bucket assignment
- TF-IDF clustering with action mapping
"""

import pytest

from src.theorem.clustering import (
    BUCKET_KEYWORDS,
    ClusterActionMapper,
    KeywordBucketAssigner,
    MultiLabelBucketAssigner,
    SemanticClusterer,
    TfidfClusterer,
    TwoPassClusterer,
    tag_buckets,
)
from src.theorem.models import FailureBucket, LawSupport, Theorem, TheoremStatus


def make_theorem(
    theorem_id: str,
    failure_modes: list[str],
    missing_structure: list[str] | None = None,
) -> Theorem:
    """Helper to create test theorems."""
    return Theorem(
        theorem_id=theorem_id,
        name=f"Theorem {theorem_id}",
        status=TheoremStatus.CONDITIONAL,
        claim="Test claim",
        support=[
            LawSupport("law_001", "confirms"),
            LawSupport("law_002", "confirms"),
        ],
        failure_modes=failure_modes,
        missing_structure=missing_structure or [],
    )


class TestTagBuckets:
    """Tests for multi-label bucket tagging."""

    def test_single_bucket_match(self):
        tags = tag_buckets("local adjacent cells pattern")
        assert "LOCAL_PATTERN" in tags

    def test_multiple_bucket_matches(self):
        tags = tag_buckets("adjacent cells triggering collision")
        assert "LOCAL_PATTERN" in tags
        assert "COLLISION_TRIGGERS" in tags

    def test_no_match_returns_other(self):
        tags = tag_buckets("completely unrelated xyz abc")
        assert tags == {"OTHER"}

    def test_empty_signature(self):
        tags = tag_buckets("")
        assert tags == {"OTHER"}


class TestMultiLabelBucketAssigner:
    """Tests for multi-label bucket assignment."""

    @pytest.fixture
    def assigner(self):
        return MultiLabelBucketAssigner()

    def test_tag_signature(self, assigner):
        tags = assigner.tag_signature("local adjacent cells")
        assert "LOCAL_PATTERN" in tags

    def test_assign_all_updates_theorem_bucket_tags(self, assigner):
        theorems = [
            make_theorem("t1", ["local adjacent cells"]),
            make_theorem("t2", ["eventually converge timeout"]),
        ]

        bucketed = assigner.assign_all(theorems)

        # Check theorem bucket_tags were updated
        assert "LOCAL_PATTERN" in theorems[0].bucket_tags
        assert "EVENTUALITY" in theorems[1].bucket_tags


class TestKeywordBucketAssigner:
    """Tests for legacy single-bucket assignment."""

    @pytest.fixture
    def assigner(self):
        return KeywordBucketAssigner()

    def test_local_pattern_keywords(self, assigner):
        bucket = assigner.assign_bucket("adjacent cells and local neighbors")
        assert bucket == FailureBucket.LOCAL_PATTERN

    def test_eventuality_keywords(self, assigner):
        bucket = assigner.assign_bucket("eventually converge to attractor")
        assert bucket == FailureBucket.EVENTUALITY

    def test_monotonicity_keywords(self, assigner):
        bucket = assigner.assign_bucket("monotonic trend strictly decreasing")
        assert bucket == FailureBucket.MONOTONICITY

    def test_collision_trigger_keywords(self, assigner):
        bucket = assigner.assign_bucket("incoming collision trigger impact")
        assert bucket == FailureBucket.COLLISION_TRIGGERS

    def test_definition_gap_keywords(self, assigner):
        bucket = assigner.assign_bucket("missing definition unclear context")
        assert bucket == FailureBucket.DEFINITION_GAP

    def test_symmetry_keywords(self, assigner):
        bucket = assigner.assign_bucket("symmetric mirror reflection transform")
        assert bucket == FailureBucket.SYMMETRY

    def test_no_match_returns_other(self, assigner):
        bucket = assigner.assign_bucket("completely unrelated text xyz")
        assert bucket == FailureBucket.OTHER

    def test_empty_signature(self, assigner):
        bucket = assigner.assign_bucket("")
        assert bucket == FailureBucket.OTHER

    def test_assign_all(self, assigner):
        theorems = [
            make_theorem("t1", ["local adjacent cells"]),
            make_theorem("t2", ["eventually converge"]),
            make_theorem("t3", ["random unrelated xyz"]),
        ]

        bucketed = assigner.assign_all(theorems)

        assert len(bucketed[FailureBucket.LOCAL_PATTERN]) == 1
        assert len(bucketed[FailureBucket.EVENTUALITY]) == 1
        assert len(bucketed[FailureBucket.OTHER]) == 1


class TestClusterActionMapper:
    """Tests for cluster action mapping."""

    @pytest.fixture
    def mapper(self):
        return ClusterActionMapper()

    def test_definition_gap_alone_returns_schema_fix(self, mapper):
        action = mapper.get_action({"DEFINITION_GAP"})
        assert action == "SCHEMA_FIX"

    def test_definition_gap_with_local_pattern_returns_observable(self, mapper):
        action = mapper.get_action({"DEFINITION_GAP", "LOCAL_PATTERN"})
        assert action == "OBSERVABLE"

    def test_local_pattern_returns_observable(self, mapper):
        action = mapper.get_action({"LOCAL_PATTERN"})
        assert action == "OBSERVABLE"

    def test_collision_triggers_returns_observable(self, mapper):
        action = mapper.get_action({"COLLISION_TRIGGERS"})
        assert action == "OBSERVABLE"

    def test_eventuality_returns_gating(self, mapper):
        action = mapper.get_action({"EVENTUALITY"})
        assert action == "GATING"

    def test_other_returns_observable(self, mapper):
        action = mapper.get_action({"OTHER"})
        assert action == "OBSERVABLE"


class TestTfidfClusterer:
    """Tests for TF-IDF based clustering."""

    @pytest.fixture
    def clusterer(self):
        return TfidfClusterer(distance_threshold=0.6)

    def test_single_item_cluster(self, clusterer):
        items = [(make_theorem("t1", ["test failure"]), "test failure", {"LOCAL_PATTERN"})]
        clusters = clusterer.cluster_bucket("LOCAL_PATTERN", items)

        assert len(clusters) == 1
        assert clusters[0].theorem_ids == ["t1"]
        assert clusters[0].avg_similarity == 1.0
        assert clusters[0].recommended_action == "OBSERVABLE"

    def test_empty_items(self, clusterer):
        clusters = clusterer.cluster_bucket("LOCAL_PATTERN", [])
        assert clusters == []

    def test_clusters_have_top_keywords(self, clusterer):
        items = [
            (make_theorem("t1", ["local adjacent cells"]), "local adjacent cells pattern", {"LOCAL_PATTERN"}),
            (make_theorem("t2", ["adjacent local pattern"]), "adjacent local pattern cells", {"LOCAL_PATTERN"}),
        ]
        clusters = clusterer.cluster_bucket("LOCAL_PATTERN", items)

        # Should have top keywords extracted
        for cluster in clusters:
            assert isinstance(cluster.top_keywords, list)

    def test_clusters_have_action(self, clusterer):
        items = [
            (make_theorem("t1", ["test"]), "test", {"EVENTUALITY"}),
        ]
        clusters = clusterer.cluster_bucket("EVENTUALITY", items)

        assert clusters[0].recommended_action == "GATING"


class TestSemanticClusterer:
    """Tests for legacy Jaccard-based clusterer."""

    @pytest.fixture
    def clusterer(self):
        return SemanticClusterer(similarity_threshold=0.3)

    def test_single_item_cluster(self, clusterer):
        items = [(make_theorem("t1", ["test failure"]), "test failure")]
        clusters = clusterer.cluster_bucket(FailureBucket.LOCAL_PATTERN, items)

        assert len(clusters) == 1
        assert clusters[0].theorem_ids == ["t1"]
        assert clusters[0].avg_similarity == 1.0

    def test_empty_items(self, clusterer):
        clusters = clusterer.cluster_bucket(FailureBucket.LOCAL_PATTERN, [])
        assert clusters == []

    def test_similar_items_merged(self, clusterer):
        # TF-IDF clustering uses cosine distance, which needs very similar text
        items = [
            (make_theorem("t1", ["local adjacent cells"]), "local adjacent cells pattern"),
            (make_theorem("t2", ["local adjacent cells pattern"]), "local adjacent cells pattern"),
        ]
        clusters = clusterer.cluster_bucket(FailureBucket.LOCAL_PATTERN, items)

        # Should be merged into one cluster (identical text)
        assert len(clusters) == 1
        assert set(clusters[0].theorem_ids) == {"t1", "t2"}

    def test_dissimilar_items_separate(self, clusterer):
        items = [
            (make_theorem("t1", ["local adjacent cells"]), "local adjacent cells"),
            (make_theorem("t2", ["completely different xyz"]), "completely different xyz"),
        ]
        clusters = clusterer.cluster_bucket(FailureBucket.LOCAL_PATTERN, items)

        # Should remain separate (no common terms)
        assert len(clusters) == 2

    def test_cluster_bucket_assignment(self, clusterer):
        items = [
            (make_theorem("t1", ["test"]), "test failure mode"),
        ]
        clusters = clusterer.cluster_bucket(FailureBucket.MONOTONICITY, items)

        assert clusters[0].bucket == FailureBucket.MONOTONICITY


class TestTwoPassClusterer:
    """Tests for two-pass clustering pipeline."""

    @pytest.fixture
    def clusterer(self):
        return TwoPassClusterer(distance_threshold=0.6)

    def test_full_pipeline(self, clusterer):
        theorems = [
            make_theorem("t1", ["local adjacent cells"]),
            make_theorem("t2", ["adjacent local pattern"]),
            make_theorem("t3", ["eventually converge timeout"]),
            make_theorem("t4", ["random unrelated xyz"]),
        ]

        clusters = clusterer.cluster(theorems)

        # Should have clusters from multiple buckets
        assert len(clusters) >= 2

        # Check we have both local pattern and eventuality buckets
        all_tags = set()
        for c in clusters:
            all_tags.update(c.bucket_tags)
        assert "LOCAL_PATTERN" in all_tags
        assert "EVENTUALITY" in all_tags

    def test_cluster_with_stats(self, clusterer):
        theorems = [
            make_theorem("t1", ["local adjacent"]),
            make_theorem("t2", ["eventually converge"]),
        ]

        clusters, stats = clusterer.cluster_with_stats(theorems)

        assert stats["total_theorems"] == 2
        assert "bucket_counts" in stats
        assert "clusters_per_bucket" in stats
        assert "action_distribution" in stats
        assert stats["total_clusters"] == len(clusters)

    def test_empty_input(self, clusterer):
        clusters = clusterer.cluster([])
        assert clusters == []

    def test_all_same_bucket(self, clusterer):
        theorems = [
            make_theorem("t1", ["local adjacent"]),
            make_theorem("t2", ["local neighbor"]),
            make_theorem("t3", ["adjacent cells"]),
        ]

        clusters = clusterer.cluster(theorems)

        # All should have LOCAL_PATTERN in bucket_tags
        for c in clusters:
            assert "LOCAL_PATTERN" in c.bucket_tags


class TestBucketKeywordsCompleteness:
    """Tests for bucket keyword coverage."""

    def test_all_new_buckets_have_keywords(self):
        """Ensure all PHASE-C buckets have keywords defined."""
        expected_buckets = {
            "DEFINITION_GAP",
            "COLLISION_TRIGGERS",
            "LOCAL_PATTERN",
            "EVENTUALITY",
            "MONOTONICITY",
            "SYMMETRY",
        }
        for bucket in expected_buckets:
            assert bucket in BUCKET_KEYWORDS
            assert len(BUCKET_KEYWORDS[bucket]) > 0


class TestNormalizeTextPreservesUnderscores:
    """Tests for signature normalization preserving underscores."""

    def test_preserve_law_id_underscores(self):
        from src.theorem.signature import normalize_text
        result = normalize_text("law_001 and law_002")
        assert "law_001" in result
        assert "law_002" in result

    def test_still_removes_other_punctuation(self):
        from src.theorem.signature import normalize_text
        result = normalize_text("test! text? with, punctuation.")
        assert "!" not in result
        assert "?" not in result
        assert "," not in result
