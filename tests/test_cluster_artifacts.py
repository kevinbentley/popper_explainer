"""Tests for PHASE-E deterministic clustering and artifacts."""

import pytest

from src.cluster.models import (
    ClusterArtifact,
    ClusterDiff,
    ClusteringParams,
    ClusterSummary,
    compute_artifact_hash,
    diff_cluster_artifacts,
)
from src.cluster.clusterer import (
    generate_deterministic_cluster_id,
    compute_cluster_content_hash,
    compute_snapshot_hash,
)
from src.theorem.signature import ROLE_CODED_SIGNATURE_VERSION, SignatureVersionMismatchError


class TestSignatureVersioning:
    def test_version_constant_exists(self):
        """Verify signature version constant is defined."""
        assert ROLE_CODED_SIGNATURE_VERSION == "1.0.0"

    def test_version_mismatch_error(self):
        """Test SignatureVersionMismatchError."""
        mismatched = [("thm_001", "0.9.0"), ("thm_002", "0.8.0")]
        error = SignatureVersionMismatchError(
            expected_version="1.0.0",
            mismatched_theorems=mismatched,
        )
        assert "1.0.0" in str(error)
        assert "thm_001" in str(error)


class TestClusteringParams:
    def test_default_params(self):
        """Test default clustering parameters."""
        params = ClusteringParams()
        assert params.distance_threshold == 0.6
        assert params.min_df == 1
        assert params.ngram_range == (1, 2)
        assert params.random_seed == 42

    def test_to_dict(self):
        """Test serialization to dict."""
        params = ClusteringParams(distance_threshold=0.5, min_df=2)
        d = params.to_dict()
        assert d["distance_threshold"] == 0.5
        assert d["min_df"] == 2

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {"distance_threshold": 0.7, "ngram_range": [2, 3]}
        params = ClusteringParams.from_dict(d)
        assert params.distance_threshold == 0.7
        assert params.ngram_range == (2, 3)

    def test_content_hash_deterministic(self):
        """Same params should produce same hash."""
        params1 = ClusteringParams()
        params2 = ClusteringParams()
        assert params1.content_hash() == params2.content_hash()

    def test_content_hash_different(self):
        """Different params should produce different hash."""
        params1 = ClusteringParams(distance_threshold=0.5)
        params2 = ClusteringParams(distance_threshold=0.6)
        assert params1.content_hash() != params2.content_hash()


class TestClusterSummary:
    def test_to_dict(self):
        summary = ClusterSummary(
            cluster_id="fc_LOCAL_PATTERN_0_abc12345",
            bucket_tags=["LOCAL_PATTERN"],
            theorem_ids=["thm_001", "thm_002"],
            size=2,
            avg_similarity=0.85,
            top_keywords=[("collision", 0.8), ("pattern", 0.6)],
            recommended_action="OBSERVABLE",
        )
        d = summary.to_dict()
        assert d["cluster_id"] == "fc_LOCAL_PATTERN_0_abc12345"
        assert d["size"] == 2

    def test_from_dict(self):
        d = {
            "cluster_id": "fc_test_0_abc",
            "bucket_tags": ["TEST"],
            "theorem_ids": ["thm_001"],
            "size": 1,
            "avg_similarity": 0.9,
            "top_keywords": [["word", 0.5]],
            "recommended_action": "OBSERVABLE",
        }
        summary = ClusterSummary.from_dict(d)
        assert summary.cluster_id == "fc_test_0_abc"
        assert summary.top_keywords == [("word", 0.5)]


class TestDeterministicClusterIds:
    def test_generate_deterministic_cluster_id(self):
        """Cluster IDs should be deterministic based on content."""
        theorem_ids = ["thm_001", "thm_002"]
        id1 = generate_deterministic_cluster_id("LOCAL_PATTERN", 0, theorem_ids)
        id2 = generate_deterministic_cluster_id("LOCAL_PATTERN", 0, theorem_ids)
        assert id1 == id2
        assert id1.startswith("fc_LOCAL_PATTERN_0_")

    def test_different_content_different_id(self):
        """Different theorem sets should produce different IDs."""
        id1 = generate_deterministic_cluster_id("LOCAL_PATTERN", 0, ["thm_001"])
        id2 = generate_deterministic_cluster_id("LOCAL_PATTERN", 0, ["thm_002"])
        assert id1 != id2

    def test_order_independent(self):
        """Order of theorem IDs shouldn't affect the hash."""
        # The content hash sorts theorem_ids, so order shouldn't matter
        hash1 = compute_cluster_content_hash(["thm_001", "thm_002"])
        hash2 = compute_cluster_content_hash(["thm_002", "thm_001"])
        assert hash1 == hash2


class TestClusterArtifact:
    def test_to_dict(self):
        params = ClusteringParams()
        artifact = ClusterArtifact(
            artifact_hash="abc123",
            theorem_run_id=1,
            snapshot_hash="snap123",
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments={"thm_001": "fc_001"},
            cluster_summaries=[],
        )
        d = artifact.to_dict()
        assert d["artifact_hash"] == "abc123"
        assert d["signature_version"] == "1.0.0"

    def test_from_dict(self):
        d = {
            "artifact_hash": "xyz789",
            "theorem_run_id": 2,
            "snapshot_hash": "snap456",
            "signature_version": "1.0.0",
            "method": "bucket+tfidf",
            "params": {"distance_threshold": 0.5},
            "assignments": {"thm_001": "fc_001"},
            "cluster_summaries": [],
            "created_at": "2024-01-01T00:00:00",
        }
        artifact = ClusterArtifact.from_dict(d)
        assert artifact.artifact_hash == "xyz789"
        assert artifact.params.distance_threshold == 0.5

    def test_properties(self):
        params = ClusteringParams()
        summary = ClusterSummary(
            cluster_id="fc_001",
            bucket_tags=["TEST"],
            theorem_ids=["thm_001", "thm_002"],
            size=2,
            avg_similarity=0.8,
            top_keywords=[],
            recommended_action="OBSERVABLE",
        )
        artifact = ClusterArtifact(
            artifact_hash="abc",
            theorem_run_id=1,
            snapshot_hash="snap",
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments={"thm_001": "fc_001", "thm_002": "fc_001"},
            cluster_summaries=[summary],
        )
        assert artifact.cluster_count == 1
        assert artifact.theorem_count == 2


class TestComputeArtifactHash:
    def test_deterministic(self):
        """Same inputs should produce same hash."""
        params = ClusteringParams()
        assignments = {"thm_001": "fc_001", "thm_002": "fc_002"}
        hash1 = compute_artifact_hash(
            snapshot_hash="snap123",
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments=assignments,
        )
        hash2 = compute_artifact_hash(
            snapshot_hash="snap123",
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments=assignments,
        )
        assert hash1 == hash2

    def test_different_inputs_different_hash(self):
        """Different inputs should produce different hash."""
        params = ClusteringParams()
        hash1 = compute_artifact_hash(
            snapshot_hash="snap123",
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments={"thm_001": "fc_001"},
        )
        hash2 = compute_artifact_hash(
            snapshot_hash="snap456",  # Different snapshot
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments={"thm_001": "fc_001"},
        )
        assert hash1 != hash2


class TestClusterDiff:
    def test_diff_identical_artifacts(self):
        """Diffing identical artifacts should show no changes."""
        params = ClusteringParams()
        summary = ClusterSummary(
            cluster_id="fc_001",
            bucket_tags=["TEST"],
            theorem_ids=["thm_001"],
            size=1,
            avg_similarity=0.9,
            top_keywords=[],
            recommended_action="OBSERVABLE",
        )
        artifact1 = ClusterArtifact(
            artifact_hash="hash1",
            theorem_run_id=1,
            snapshot_hash="snap",
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments={"thm_001": "fc_001"},
            cluster_summaries=[summary],
        )
        artifact2 = ClusterArtifact(
            artifact_hash="hash1",
            theorem_run_id=1,
            snapshot_hash="snap",
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments={"thm_001": "fc_001"},
            cluster_summaries=[summary],
        )
        diff = diff_cluster_artifacts(artifact1, artifact2)
        assert not diff.has_changes
        assert diff.added_theorems == []
        assert diff.removed_theorems == []
        assert diff.reassigned_theorems == []

    def test_diff_added_theorem(self):
        """Diffing should detect added theorems."""
        params = ClusteringParams()
        artifact1 = ClusterArtifact(
            artifact_hash="hash1",
            theorem_run_id=1,
            snapshot_hash="snap",
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments={"thm_001": "fc_001"},
            cluster_summaries=[],
        )
        artifact2 = ClusterArtifact(
            artifact_hash="hash2",
            theorem_run_id=1,
            snapshot_hash="snap",
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments={"thm_001": "fc_001", "thm_002": "fc_002"},
            cluster_summaries=[],
        )
        diff = diff_cluster_artifacts(artifact1, artifact2)
        assert diff.has_changes
        assert "thm_002" in diff.added_theorems

    def test_diff_reassigned_theorem(self):
        """Diffing should detect reassigned theorems."""
        params = ClusteringParams()
        artifact1 = ClusterArtifact(
            artifact_hash="hash1",
            theorem_run_id=1,
            snapshot_hash="snap",
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments={"thm_001": "fc_001"},
            cluster_summaries=[],
        )
        artifact2 = ClusterArtifact(
            artifact_hash="hash2",
            theorem_run_id=1,
            snapshot_hash="snap",
            signature_version="1.0.0",
            method="bucket+tfidf",
            params=params,
            assignments={"thm_001": "fc_002"},  # Reassigned
            cluster_summaries=[],
        )
        diff = diff_cluster_artifacts(artifact1, artifact2)
        assert diff.has_changes
        assert len(diff.reassigned_theorems) == 1
        thm_id, old_cluster, new_cluster = diff.reassigned_theorems[0]
        assert thm_id == "thm_001"
        assert old_cluster == "fc_001"
        assert new_cluster == "fc_002"
