"""Clustering module for theorem failure analysis (PHASE-E)."""

from src.cluster.models import (
    ClusterArtifact,
    ClusterDiff,
    ClusteringParams,
    ClusterSummary,
    compute_artifact_hash,
    diff_cluster_artifacts,
)
from src.cluster.clusterer import (
    DeterministicClusterer,
    cluster_theorems,
    generate_deterministic_cluster_id,
    compute_cluster_content_hash,
    compute_snapshot_hash,
    validate_signature_versions,
)

__all__ = [
    "ClusterArtifact",
    "ClusterDiff",
    "ClusteringParams",
    "ClusterSummary",
    "compute_artifact_hash",
    "diff_cluster_artifacts",
    "DeterministicClusterer",
    "cluster_theorems",
    "generate_deterministic_cluster_id",
    "compute_cluster_content_hash",
    "compute_snapshot_hash",
    "validate_signature_versions",
]
