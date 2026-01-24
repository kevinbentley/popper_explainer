"""Two-pass failure clustering for theorems.

Pass A: Multi-label keyword-based bucket tagging
Pass B: TF-IDF + Agglomerative clustering within buckets

PHASE-C updates:
- Multi-label bucket assignment (theorems can belong to multiple buckets)
- TF-IDF + cosine distance clustering (replaces Jaccard)
- Top keyword extraction per cluster
- Action mapping (SCHEMA_FIX, OBSERVABLE, GATING)
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from src.theorem.models import FailureBucket, FailureCluster, Theorem
from src.theorem.signature import (
    build_failure_signature,
    extract_key_terms,
)


# PHASE-C bucket keywords (refined taxonomy)
BUCKET_KEYWORDS: dict[str, list[str]] = {
    "DEFINITION_GAP": [
        "definition", "precise", "clear definitions", "constitutes",
        "context", "ambiguous", "undefined", "meaning", "what is",
        "missing definition", "unclear",
    ],
    "COLLISION_TRIGGERS": [
        "incoming collision", "trigger", "prevent", "guarantee",
        "insufficient", "bound", "collision", "collide", "impact",
        "approaching", "converging pair",
    ],
    "LOCAL_PATTERN": [
        "configuration", "arrangement", "adjacent", "pairs", "spread",
        "gap", "neighbors", "nearby", "local", "cell", "position",
        "bracket", "pattern", "alternating", "spacing", "proximity",
    ],
    "EVENTUALITY": [
        "eventually", "long term", "asymptotic", "cease", "resolve",
        "attractor", "converge", "steady", "horizon", "timeout",
        "never", "always", "future", "until", "bounded time",
    ],
    "MONOTONICITY": [
        "monotonic", "non increasing", "non decreasing", "never increase",
        "strictly", "always increase", "always decrease", "trend",
        "growth", "decay", "monotone",
    ],
    "SYMMETRY": [
        "symmetric", "mirror", "swap", "reflection", "translation",
        "shift", "commute", "commutation", "time reversal", "invariance",
        "transform",
    ],
}


def tag_buckets(signature_text: str) -> set[str]:
    """Assign multiple bucket tags to a signature.

    Args:
        signature_text: Normalized signature text

    Returns:
        Set of bucket tag strings (may contain multiple)
    """
    signature_lower = signature_text.lower()
    tags: set[str] = set()

    for bucket, keywords in BUCKET_KEYWORDS.items():
        for keyword in keywords:
            if keyword in signature_lower:
                tags.add(bucket)
                break  # One match per bucket is enough

    return tags if tags else {"OTHER"}


class MultiLabelBucketAssigner:
    """Assigns multiple bucket tags based on keyword matching."""

    def __init__(
        self,
        keywords: dict[str, list[str]] | None = None,
    ):
        self.keywords = keywords or BUCKET_KEYWORDS

    def tag_signature(self, signature: str) -> set[str]:
        """Assign bucket tags to a single signature."""
        return tag_buckets(signature)

    def assign_all(
        self,
        theorems: list[Theorem],
    ) -> dict[str, list[tuple[Theorem, str, set[str]]]]:
        """Assign buckets to all theorems.

        Returns:
            Dict mapping primary bucket to list of (theorem, signature, all_tags) tuples
        """
        result: dict[str, list[tuple[Theorem, str, set[str]]]] = defaultdict(list)

        for theorem in theorems:
            signature = build_failure_signature(theorem)
            tags = self.tag_signature(signature)

            # Update theorem's bucket_tags
            theorem.bucket_tags = sorted(tags)

            # Add to primary bucket (first alphabetically, or first match)
            primary = sorted(tags)[0] if tags else "OTHER"
            result[primary].append((theorem, signature, tags))

        return result


class ClusterActionMapper:
    """Maps cluster bucket tags and keywords to recommended actions.

    Action types:
    - SCHEMA_FIX: Definition/prompt issues that need schema changes
    - OBSERVABLE: Need new observables to track patterns
    - GATING: Policy/template gating for eventual behaviors
    """

    def get_action(self, bucket_tags: set[str], keywords: list[str] | None = None) -> str:
        """Determine recommended action based on bucket tags and keywords.

        Args:
            bucket_tags: Set of bucket tag strings
            keywords: Optional list of top keywords from cluster

        Returns:
            Action string: SCHEMA_FIX, OBSERVABLE, or GATING
        """
        # DEFINITION_GAP alone (not combined with pattern buckets) → SCHEMA_FIX
        pattern_buckets = {"LOCAL_PATTERN", "COLLISION_TRIGGERS"}
        if "DEFINITION_GAP" in bucket_tags and not bucket_tags & pattern_buckets:
            return "SCHEMA_FIX"

        # LOCAL_PATTERN or COLLISION_TRIGGERS → OBSERVABLE
        if bucket_tags & pattern_buckets:
            return "OBSERVABLE"

        # EVENTUALITY → GATING
        if "EVENTUALITY" in bucket_tags:
            return "GATING"

        # Default: OBSERVABLE (propose new measurements)
        return "OBSERVABLE"


@dataclass
class ClusterCandidate:
    """Candidate for clustering with precomputed data."""

    theorem: Theorem
    signature: str
    bucket_tags: set[str]


class TfidfClusterer:
    """TF-IDF based clustering within buckets using agglomerative clustering."""

    def __init__(
        self,
        distance_threshold: float = 0.6,
        min_df: int = 1,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        """Initialize TF-IDF clusterer.

        Args:
            distance_threshold: Cosine distance threshold for merging (0-1)
            min_df: Minimum document frequency for TF-IDF
            ngram_range: N-gram range for TF-IDF vectorizer
        """
        self.distance_threshold = distance_threshold
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.action_mapper = ClusterActionMapper()

    def cluster_bucket(
        self,
        bucket_name: str,
        items: list[tuple[Theorem, str, set[str]]],
    ) -> list[FailureCluster]:
        """Cluster items within a bucket using TF-IDF + Agglomerative.

        Args:
            bucket_name: The primary bucket name for these items
            items: List of (theorem, signature, bucket_tags) tuples

        Returns:
            List of FailureCluster objects
        """
        if not items:
            return []

        # Single item case
        if len(items) == 1:
            theorem, signature, tags = items[0]
            bucket_tags = sorted(tags)
            action = self.action_mapper.get_action(tags)
            return [
                FailureCluster(
                    cluster_id=f"fc_{bucket_name}_0_{uuid.uuid4().hex[:8]}",
                    bucket_tags=bucket_tags,
                    semantic_cluster_idx=0,
                    theorem_ids=[theorem.theorem_id],
                    centroid_signature=signature,
                    avg_similarity=1.0,
                    top_keywords=self._extract_simple_keywords(signature),
                    recommended_action=action,
                    distance_threshold=self.distance_threshold,
                )
            ]

        # Extract signatures
        signatures = [item[1] for item in items]

        # TF-IDF vectorization
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                stop_words=None,  # Keep all words for domain-specific terms
            )
            X = vectorizer.fit_transform(signatures)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            # Empty vocabulary - treat all as one cluster
            all_tags: set[str] = set()
            for _, _, tags in items:
                all_tags.update(tags)
            bucket_tags = sorted(all_tags)
            action = self.action_mapper.get_action(all_tags)
            return [
                FailureCluster(
                    cluster_id=f"fc_{bucket_name}_0_{uuid.uuid4().hex[:8]}",
                    bucket_tags=bucket_tags,
                    semantic_cluster_idx=0,
                    theorem_ids=[item[0].theorem_id for item in items],
                    centroid_signature=" ".join(signatures),
                    avg_similarity=1.0,
                    top_keywords=[],
                    recommended_action=action,
                    distance_threshold=self.distance_threshold,
                )
            ]

        # Compute cosine distance matrix
        D = cosine_distances(X)

        # Agglomerative clustering with precomputed distances
        if len(items) > 1:
            model = AgglomerativeClustering(
                metric="precomputed",
                linkage="average",
                distance_threshold=self.distance_threshold,
                n_clusters=None,
            )
            labels = model.fit_predict(D)
        else:
            labels = np.array([0])

        # Group by cluster label
        label_to_items: dict[int, list[tuple[Theorem, str, set[str], int]]] = defaultdict(list)
        for idx, (item, label) in enumerate(zip(items, labels)):
            theorem, sig, tags = item
            label_to_items[label].append((theorem, sig, tags, idx))

        # Build FailureCluster objects
        result = []
        for cluster_idx, (label, cluster_items) in enumerate(sorted(label_to_items.items())):
            theorem_ids = [item[0].theorem_id for item in cluster_items]
            signatures_in_cluster = [item[1] for item in cluster_items]

            # Merge all bucket tags from cluster members
            all_tags: set[str] = set()
            for _, _, tags, _ in cluster_items:
                all_tags.update(tags)
            bucket_tags = sorted(all_tags)

            # Compute centroid signature
            centroid = " ".join(signatures_in_cluster)

            # Compute average internal similarity
            indices = [item[3] for item in cluster_items]
            avg_sim = self._compute_avg_similarity(D, indices)

            # Extract top keywords from TF-IDF weights
            top_keywords = self._extract_top_keywords(
                X, indices, feature_names, top_n=8
            )

            # Determine action
            action = self.action_mapper.get_action(all_tags, [kw for kw, _ in top_keywords])

            cluster = FailureCluster(
                cluster_id=f"fc_{bucket_name}_{cluster_idx}_{uuid.uuid4().hex[:8]}",
                bucket_tags=bucket_tags,
                semantic_cluster_idx=cluster_idx,
                theorem_ids=theorem_ids,
                centroid_signature=centroid,
                avg_similarity=avg_sim,
                top_keywords=top_keywords,
                recommended_action=action,
                distance_threshold=self.distance_threshold,
            )
            result.append(cluster)

        return result

    def _compute_avg_similarity(
        self,
        D: np.ndarray,
        indices: list[int],
    ) -> float:
        """Compute average pairwise similarity within a cluster."""
        if len(indices) <= 1:
            return 1.0

        total_dist = 0.0
        count = 0
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i + 1:]:
                total_dist += D[idx1, idx2]
                count += 1

        avg_dist = total_dist / count if count > 0 else 0.0
        return 1.0 - avg_dist  # Convert distance to similarity

    def _extract_top_keywords(
        self,
        X: Any,  # sparse matrix
        indices: list[int],
        feature_names: np.ndarray,
        top_n: int = 8,
    ) -> list[tuple[str, float]]:
        """Extract top keywords from cluster based on TF-IDF weights."""
        if len(indices) == 0:
            return []

        # Compute centroid of cluster (mean of TF-IDF vectors)
        cluster_vectors = X[indices].toarray()
        centroid = cluster_vectors.mean(axis=0)

        # Get top keywords by weight
        top_indices = np.argsort(centroid)[-top_n:][::-1]
        keywords = [
            (feature_names[i], float(centroid[i]))
            for i in top_indices
            if centroid[i] > 0
        ]
        return keywords

    def _extract_simple_keywords(
        self,
        signature: str,
        top_n: int = 8,
    ) -> list[tuple[str, float]]:
        """Extract keywords from a single signature."""
        terms = extract_key_terms(signature)
        # Just return terms with equal weight
        return [(t, 1.0) for t in sorted(terms)[:top_n]]


class TwoPassClusterer:
    """Two-pass clustering: multi-label bucket tagging then TF-IDF clustering."""

    def __init__(
        self,
        bucket_assigner: MultiLabelBucketAssigner | None = None,
        tfidf_clusterer: TfidfClusterer | None = None,
        distance_threshold: float = 0.6,
    ):
        self.bucket_assigner = bucket_assigner or MultiLabelBucketAssigner()
        self.tfidf_clusterer = tfidf_clusterer or TfidfClusterer(
            distance_threshold=distance_threshold
        )

    def cluster(self, theorems: list[Theorem]) -> list[FailureCluster]:
        """Cluster theorems using two-pass algorithm.

        Args:
            theorems: List of theorems to cluster

        Returns:
            List of all FailureCluster objects across all buckets
        """
        # Pass A: Assign buckets (multi-label)
        bucketed = self.bucket_assigner.assign_all(theorems)

        # Pass B: TF-IDF clustering within each bucket
        all_clusters: list[FailureCluster] = []
        for bucket_name, items in bucketed.items():
            clusters = self.tfidf_clusterer.cluster_bucket(bucket_name, items)
            all_clusters.extend(clusters)

        return all_clusters

    def cluster_with_stats(
        self,
        theorems: list[Theorem],
    ) -> tuple[list[FailureCluster], dict[str, Any]]:
        """Cluster theorems and return clustering statistics.

        Returns:
            Tuple of (clusters, stats_dict)
        """
        # Pass A: Assign buckets (multi-label)
        bucketed = self.bucket_assigner.assign_all(theorems)

        # Collect stats
        stats: dict[str, Any] = {
            "total_theorems": len(theorems),
            "bucket_counts": {
                bucket: len(items) for bucket, items in bucketed.items()
            },
            "clusters_per_bucket": {},
            "action_distribution": defaultdict(int),
        }

        # Pass B: TF-IDF clustering within each bucket
        all_clusters: list[FailureCluster] = []
        for bucket_name, items in bucketed.items():
            clusters = self.tfidf_clusterer.cluster_bucket(bucket_name, items)
            all_clusters.extend(clusters)
            stats["clusters_per_bucket"][bucket_name] = len(clusters)

            # Track action distribution
            for cluster in clusters:
                stats["action_distribution"][cluster.recommended_action] += 1

        stats["total_clusters"] = len(all_clusters)
        stats["action_distribution"] = dict(stats["action_distribution"])

        return all_clusters, stats


# Backwards compatibility: legacy KeywordBucketAssigner
class KeywordBucketAssigner:
    """Legacy bucket assigner for backwards compatibility."""

    def __init__(self, keywords: dict[FailureBucket, set[str]] | None = None):
        self._multi_assigner = MultiLabelBucketAssigner()

    def assign_bucket(self, signature: str) -> FailureBucket:
        """Assign a single bucket (primary) to a signature."""
        tags = self._multi_assigner.tag_signature(signature)
        primary = sorted(tags)[0] if tags else "OTHER"
        try:
            return FailureBucket(primary)
        except ValueError:
            return FailureBucket.OTHER

    def assign_all(
        self,
        theorems: list[Theorem],
    ) -> dict[FailureBucket, list[tuple[Theorem, str]]]:
        """Legacy interface: returns single bucket assignment."""
        result: dict[FailureBucket, list[tuple[Theorem, str]]] = defaultdict(list)
        bucketed = self._multi_assigner.assign_all(theorems)

        for bucket_name, items in bucketed.items():
            try:
                bucket = FailureBucket(bucket_name)
            except ValueError:
                bucket = FailureBucket.OTHER
            for theorem, sig, _ in items:
                result[bucket].append((theorem, sig))

        return result


# Backwards compatibility: legacy SemanticClusterer
class SemanticClusterer:
    """Legacy Jaccard-based clusterer for backwards compatibility."""

    def __init__(self, similarity_threshold: float = 0.3):
        # Use TF-IDF clusterer with approximate equivalent threshold
        # Jaccard 0.3 roughly corresponds to cosine distance 0.6
        self._tfidf = TfidfClusterer(distance_threshold=1.0 - similarity_threshold)

    def cluster_bucket(
        self,
        bucket: FailureBucket,
        items: list[tuple[Theorem, str]],
    ) -> list[FailureCluster]:
        """Legacy interface: accepts (theorem, signature) tuples."""
        # Convert to new format with empty tags
        new_items = [
            (theorem, sig, {bucket.value})
            for theorem, sig in items
        ]
        return self._tfidf.cluster_bucket(bucket.value, new_items)
