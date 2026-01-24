"""Deterministic clusterer for theorem failure analysis (PHASE-E).

This module provides the high-level clustering API that wraps the
TwoPassClusterer with deterministic ID generation and artifact tracking.
"""

import hashlib
import json

from src.cluster.models import (
    ClusterArtifact,
    ClusterDiff,
    ClusteringParams,
    ClusterSummary,
    compute_artifact_hash,
    diff_cluster_artifacts,
)
from src.theorem.clustering import TwoPassClusterer, TfidfClusterer
from src.theorem.models import FailureCluster, Theorem
from src.theorem.signature import (
    ROLE_CODED_SIGNATURE_VERSION,
    SignatureVersionMismatchError,
    build_role_coded_signature,
    hash_signature,
)


def generate_deterministic_cluster_id(
    bucket: str,
    cluster_idx: int,
    content_hash: str,
) -> str:
    """Generate a deterministic cluster ID from bucket, index, and content.

    Format: fc_{bucket}_{idx}_{content_hash[:8]}

    Args:
        bucket: The primary bucket name
        cluster_idx: Index within the bucket
        content_hash: Hash of the cluster contents (theorem IDs)

    Returns:
        Deterministic cluster ID string
    """
    return f"fc_{bucket}_{cluster_idx}_{content_hash[:8]}"


def compute_cluster_content_hash(theorem_ids: list[str]) -> str:
    """Compute a hash of the cluster contents (sorted theorem IDs)."""
    content = json.dumps(sorted(theorem_ids), sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def compute_snapshot_hash(theorems: list[Theorem]) -> str:
    """Compute a hash of the theorem snapshot used for clustering.

    This captures the essential content of the input theorems so we can
    detect when inputs have changed.
    """
    # Sort by theorem_id for determinism
    sorted_theorems = sorted(theorems, key=lambda t: t.theorem_id)
    content = []
    for t in sorted_theorems:
        content.append({
            "theorem_id": t.theorem_id,
            "failure_modes": sorted(t.failure_modes),
            "missing_structure": sorted(t.missing_structure),
            "support": sorted(s.to_dict()["law_id"] for s in t.support),
        })
    content_str = json.dumps(content, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()


def validate_signature_versions(theorems: list[Theorem]) -> None:
    """Check that all theorems have compatible signature versions.

    Args:
        theorems: List of theorems to validate

    Raises:
        SignatureVersionMismatchError: If any theorems have incompatible versions
    """
    mismatched = []
    for t in theorems:
        if t.signature_version and t.signature_version != ROLE_CODED_SIGNATURE_VERSION:
            mismatched.append((t.theorem_id, t.signature_version))

    if mismatched:
        raise SignatureVersionMismatchError(
            expected_version=ROLE_CODED_SIGNATURE_VERSION,
            mismatched_theorems=mismatched,
        )


class DeterministicClusterer:
    """Deterministic clusterer with artifact tracking.

    This wraps TwoPassClusterer to provide:
    - Deterministic cluster IDs (content-based, not UUID)
    - Artifact generation for reproducibility
    - Signature version validation
    """

    def __init__(
        self,
        params: ClusteringParams | None = None,
        method: str = "bucket+tfidf",
    ):
        self.params = params or ClusteringParams()
        self.method = method
        self._tfidf_clusterer = TfidfClusterer(
            distance_threshold=self.params.distance_threshold,
            min_df=self.params.min_df,
            ngram_range=self.params.ngram_range,
        )
        self._two_pass = TwoPassClusterer(
            tfidf_clusterer=self._tfidf_clusterer,
            distance_threshold=self.params.distance_threshold,
        )

    def cluster(
        self,
        theorems: list[Theorem],
        theorem_run_id: int | None = None,
        validate_versions: bool = True,
    ) -> tuple[list[FailureCluster], ClusterArtifact]:
        """Cluster theorems and return clusters with artifact.

        Args:
            theorems: Theorems to cluster
            theorem_run_id: Optional run ID for artifact tracking
            validate_versions: Whether to validate signature versions

        Returns:
            Tuple of (clusters, artifact)

        Raises:
            SignatureVersionMismatchError: If version validation fails
        """
        if validate_versions:
            validate_signature_versions(theorems)

        # Stamp signature version on theorems that don't have it
        for t in theorems:
            if t.signature_version is None:
                t.signature_version = ROLE_CODED_SIGNATURE_VERSION

        # Compute snapshot hash before clustering
        snapshot_hash = compute_snapshot_hash(theorems)

        # Run the two-pass clustering
        clusters_raw = self._two_pass.cluster(theorems)

        # Replace UUIDs with deterministic IDs
        clusters = self._make_deterministic(clusters_raw)

        # Build assignments map
        assignments: dict[str, str] = {}
        for cluster in clusters:
            for theorem_id in cluster.theorem_ids:
                assignments[theorem_id] = cluster.cluster_id

        # Build cluster summaries
        summaries = [
            ClusterSummary(
                cluster_id=c.cluster_id,
                bucket_tags=c.bucket_tags,
                theorem_ids=c.theorem_ids,
                size=len(c.theorem_ids),
                avg_similarity=c.avg_similarity,
                top_keywords=c.top_keywords,
                recommended_action=c.recommended_action,
            )
            for c in clusters
        ]

        # Compute artifact hash
        artifact_hash = compute_artifact_hash(
            snapshot_hash=snapshot_hash,
            signature_version=ROLE_CODED_SIGNATURE_VERSION,
            method=self.method,
            params=self.params,
            assignments=assignments,
        )

        artifact = ClusterArtifact(
            artifact_hash=artifact_hash,
            theorem_run_id=theorem_run_id,
            snapshot_hash=snapshot_hash,
            signature_version=ROLE_CODED_SIGNATURE_VERSION,
            method=self.method,
            params=self.params,
            assignments=assignments,
            cluster_summaries=summaries,
        )

        return clusters, artifact

    def _make_deterministic(
        self,
        clusters: list[FailureCluster],
    ) -> list[FailureCluster]:
        """Replace UUID-based cluster IDs with deterministic ones."""
        result = []
        # Group by primary bucket for consistent indexing
        bucket_indices: dict[str, int] = {}

        for cluster in clusters:
            # Get primary bucket
            primary_bucket = cluster.bucket_tags[0] if cluster.bucket_tags else "OTHER"

            # Get next index for this bucket
            idx = bucket_indices.get(primary_bucket, 0)
            bucket_indices[primary_bucket] = idx + 1

            # Compute content hash
            content_hash = compute_cluster_content_hash(cluster.theorem_ids)

            # Generate deterministic ID
            new_id = generate_deterministic_cluster_id(
                bucket=primary_bucket,
                cluster_idx=idx,
                content_hash=content_hash,
            )

            # Create new cluster with deterministic ID
            result.append(
                FailureCluster(
                    cluster_id=new_id,
                    bucket_tags=cluster.bucket_tags,
                    semantic_cluster_idx=cluster.semantic_cluster_idx,
                    theorem_ids=cluster.theorem_ids,
                    centroid_signature=cluster.centroid_signature,
                    avg_similarity=cluster.avg_similarity,
                    top_keywords=cluster.top_keywords,
                    recommended_action=cluster.recommended_action,
                    distance_threshold=cluster.distance_threshold,
                )
            )

        return result


def cluster_theorems(
    theorems: list[Theorem],
    params: ClusteringParams | None = None,
    theorem_run_id: int | None = None,
) -> ClusterArtifact:
    """Convenience function to cluster theorems and return artifact.

    This is the main entry point for deterministic clustering.

    Args:
        theorems: Theorems to cluster
        params: Optional clustering parameters
        theorem_run_id: Optional run ID for tracking

    Returns:
        ClusterArtifact with all clustering results
    """
    clusterer = DeterministicClusterer(params=params)
    _, artifact = clusterer.cluster(theorems, theorem_run_id=theorem_run_id)
    return artifact


# Re-export for convenience
__all__ = [
    "DeterministicClusterer",
    "cluster_theorems",
    "diff_cluster_artifacts",
    "generate_deterministic_cluster_id",
    "compute_cluster_content_hash",
    "compute_snapshot_hash",
    "validate_signature_versions",
]
