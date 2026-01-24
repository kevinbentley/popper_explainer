"""Cluster models for deterministic clustering artifacts (PHASE-E).

This module defines the data structures needed for reproducible clustering
with artifact tracking and deterministic cluster IDs.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ClusteringParams:
    """Parameters controlling the clustering algorithm.

    These parameters are stored with each cluster artifact to ensure
    reproducibility. Same params + same inputs = same clusters.
    """

    distance_threshold: float = 0.6
    min_df: int = 1
    ngram_range: tuple[int, int] = (1, 2)
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance_threshold": self.distance_threshold,
            "min_df": self.min_df,
            "ngram_range": list(self.ngram_range),
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClusteringParams":
        return cls(
            distance_threshold=data.get("distance_threshold", 0.6),
            min_df=data.get("min_df", 1),
            ngram_range=tuple(data.get("ngram_range", [1, 2])),
            random_seed=data.get("random_seed", 42),
        )

    def content_hash(self) -> str:
        """Compute a deterministic hash of parameters for deduplication."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ClusterSummary:
    """Summary of a single cluster for artifact storage."""

    cluster_id: str
    bucket_tags: list[str]
    theorem_ids: list[str]
    size: int
    avg_similarity: float
    top_keywords: list[tuple[str, float]]
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "bucket_tags": self.bucket_tags,
            "theorem_ids": self.theorem_ids,
            "size": self.size,
            "avg_similarity": self.avg_similarity,
            "top_keywords": self.top_keywords,
            "recommended_action": self.recommended_action,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClusterSummary":
        return cls(
            cluster_id=data["cluster_id"],
            bucket_tags=data["bucket_tags"],
            theorem_ids=data["theorem_ids"],
            size=data["size"],
            avg_similarity=data["avg_similarity"],
            top_keywords=[tuple(kw) for kw in data.get("top_keywords", [])],
            recommended_action=data["recommended_action"],
        )


@dataclass
class ClusterArtifact:
    """Complete clustering artifact for reproducibility tracking.

    This captures everything needed to reproduce a clustering run:
    - Input snapshot (theorem_run_id, snapshot_hash)
    - Method and parameters
    - Output assignments and summaries
    - Deterministic artifact_hash for deduplication

    Same inputs + params = same artifact_hash.
    """

    artifact_hash: str
    theorem_run_id: int | None
    snapshot_hash: str
    signature_version: str
    method: str  # e.g., "bucket+tfidf"
    params: ClusteringParams
    assignments: dict[str, str]  # theorem_id -> cluster_id
    cluster_summaries: list[ClusterSummary]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_hash": self.artifact_hash,
            "theorem_run_id": self.theorem_run_id,
            "snapshot_hash": self.snapshot_hash,
            "signature_version": self.signature_version,
            "method": self.method,
            "params": self.params.to_dict(),
            "assignments": self.assignments,
            "cluster_summaries": [cs.to_dict() for cs in self.cluster_summaries],
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClusterArtifact":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            artifact_hash=data["artifact_hash"],
            theorem_run_id=data.get("theorem_run_id"),
            snapshot_hash=data["snapshot_hash"],
            signature_version=data["signature_version"],
            method=data["method"],
            params=ClusteringParams.from_dict(data["params"]),
            assignments=data["assignments"],
            cluster_summaries=[
                ClusterSummary.from_dict(cs) for cs in data.get("cluster_summaries", [])
            ],
            created_at=created_at,
        )

    @property
    def cluster_count(self) -> int:
        """Number of clusters in this artifact."""
        return len(self.cluster_summaries)

    @property
    def theorem_count(self) -> int:
        """Number of theorems clustered."""
        return len(self.assignments)


@dataclass
class ClusterDiff:
    """Diff between two cluster artifacts showing changes."""

    old_artifact_hash: str
    new_artifact_hash: str
    added_theorems: list[str]
    removed_theorems: list[str]
    reassigned_theorems: list[tuple[str, str, str]]  # (theorem_id, old_cluster, new_cluster)
    new_clusters: list[str]
    removed_clusters: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(
            self.added_theorems
            or self.removed_theorems
            or self.reassigned_theorems
            or self.new_clusters
            or self.removed_clusters
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_artifact_hash": self.old_artifact_hash,
            "new_artifact_hash": self.new_artifact_hash,
            "added_theorems": self.added_theorems,
            "removed_theorems": self.removed_theorems,
            "reassigned_theorems": [
                {"theorem_id": t, "old_cluster": o, "new_cluster": n}
                for t, o, n in self.reassigned_theorems
            ],
            "new_clusters": self.new_clusters,
            "removed_clusters": self.removed_clusters,
        }


def compute_artifact_hash(
    snapshot_hash: str,
    signature_version: str,
    method: str,
    params: ClusteringParams,
    assignments: dict[str, str],
) -> str:
    """Compute deterministic artifact hash from inputs and assignments.

    Same inputs with same clustering results = same hash.
    """
    content = {
        "snapshot_hash": snapshot_hash,
        "signature_version": signature_version,
        "method": method,
        "params": params.to_dict(),
        # Sort assignments for determinism
        "assignments": dict(sorted(assignments.items())),
    }
    content_str = json.dumps(content, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()


def diff_cluster_artifacts(old: ClusterArtifact, new: ClusterArtifact) -> ClusterDiff:
    """Compute the diff between two cluster artifacts.

    This is useful for understanding how clustering changed between runs.
    """
    old_theorems = set(old.assignments.keys())
    new_theorems = set(new.assignments.keys())

    added = list(new_theorems - old_theorems)
    removed = list(old_theorems - new_theorems)

    # Find reassignments among theorems present in both
    common = old_theorems & new_theorems
    reassigned = []
    for theorem_id in common:
        old_cluster = old.assignments[theorem_id]
        new_cluster = new.assignments[theorem_id]
        if old_cluster != new_cluster:
            reassigned.append((theorem_id, old_cluster, new_cluster))

    old_clusters = {cs.cluster_id for cs in old.cluster_summaries}
    new_clusters = {cs.cluster_id for cs in new.cluster_summaries}

    return ClusterDiff(
        old_artifact_hash=old.artifact_hash,
        new_artifact_hash=new.artifact_hash,
        added_theorems=sorted(added),
        removed_theorems=sorted(removed),
        reassigned_theorems=sorted(reassigned),
        new_clusters=sorted(new_clusters - old_clusters),
        removed_clusters=sorted(old_clusters - new_clusters),
    )
