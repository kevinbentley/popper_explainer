"""Redundancy detection for law proposals."""

import hashlib
from dataclasses import dataclass
from typing import Any

from src.claims.fingerprint import compute_semantic_fingerprint
from src.claims.schema import CandidateLaw


@dataclass
class RedundancyMatch:
    """Information about a redundancy match.

    Attributes:
        matched_law_id: ID of the matching law
        similarity: Similarity score (0-1)
        match_type: Type of match (exact, normalized, semantic, fingerprint)
        details: Additional match details
    """

    matched_law_id: str
    similarity: float
    match_type: str
    details: str | None = None


class RedundancyDetector:
    """Detects redundant law proposals.

    Uses multiple strategies (in order):
    1. Exact hash matching (identical content)
    2. Semantic fingerprint matching (equivalent meaning, different names)
    3. Normalized form matching (syntactic variations)
    4. Semantic similarity heuristics (optional, for fuzzy matching)
    """

    def __init__(self, similarity_threshold: float = 0.0, use_semantic: bool = False):
        """Initialize redundancy detector.

        Args:
            similarity_threshold: Minimum similarity to consider redundant (if use_semantic=True)
            use_semantic: Whether to use fuzzy semantic matching (default: False for thoroughness)
        """
        self.similarity_threshold = similarity_threshold
        self.use_semantic = use_semantic
        self._known_hashes: dict[str, str] = {}  # hash -> law_id
        self._known_fingerprints: dict[str, str] = {}  # semantic fingerprint -> law_id
        self._known_normalized: dict[str, str] = {}  # normalized -> law_id
        self._known_laws: dict[str, CandidateLaw] = {}  # law_id -> law

    def add_known_law(self, law: CandidateLaw) -> None:
        """Add a law to the known set.

        Args:
            law: Law to add
        """
        # Store by exact hash
        exact_hash = self._compute_exact_hash(law)
        self._known_hashes[exact_hash] = law.law_id

        # Store by semantic fingerprint
        fingerprint = compute_semantic_fingerprint(law)
        self._known_fingerprints[fingerprint] = law.law_id

        # Store by normalized form
        normalized = self._normalize_law(law)
        normalized_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        self._known_normalized[normalized_hash] = law.law_id

        # Store the law itself
        self._known_laws[law.law_id] = law

    def check(self, law: CandidateLaw) -> RedundancyMatch | None:
        """Check if a law is redundant.

        Args:
            law: Law to check

        Returns:
            RedundancyMatch if redundant, None otherwise
        """
        # Check exact hash
        exact_hash = self._compute_exact_hash(law)
        if exact_hash in self._known_hashes:
            return RedundancyMatch(
                matched_law_id=self._known_hashes[exact_hash],
                similarity=1.0,
                match_type="exact",
                details="Exact content hash match",
            )

        # Check semantic fingerprint (catches "momentum_conserved" vs "net_momentum_conserved")
        fingerprint = compute_semantic_fingerprint(law)
        if fingerprint in self._known_fingerprints:
            return RedundancyMatch(
                matched_law_id=self._known_fingerprints[fingerprint],
                similarity=1.0,
                match_type="fingerprint",
                details=f"Semantic fingerprint match: {fingerprint[:12]}...",
            )

        # Check normalized form
        normalized = self._normalize_law(law)
        normalized_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        if normalized_hash in self._known_normalized:
            return RedundancyMatch(
                matched_law_id=self._known_normalized[normalized_hash],
                similarity=1.0,
                match_type="normalized",
                details="Normalized form match",
            )

        # Check semantic similarity (only if enabled)
        if self.use_semantic and self.similarity_threshold > 0:
            for known_id, known_law in self._known_laws.items():
                similarity = self._compute_similarity(law, known_law)
                if similarity >= self.similarity_threshold:
                    return RedundancyMatch(
                        matched_law_id=known_id,
                        similarity=similarity,
                        match_type="semantic",
                        details=f"High similarity score: {similarity:.2f}",
                    )

        return None

    def filter_batch(
        self, laws: list[CandidateLaw]
    ) -> tuple[list[CandidateLaw], list[tuple[CandidateLaw, RedundancyMatch]]]:
        """Filter a batch of laws for redundancy.

        Args:
            laws: Laws to filter

        Returns:
            Tuple of (non-redundant laws, redundant laws with matches)
        """
        non_redundant = []
        redundant = []

        # Track within-batch duplicates by both exact hash and fingerprint
        batch_hashes: dict[str, CandidateLaw] = {}
        batch_fingerprints: dict[str, CandidateLaw] = {}

        for law in laws:
            # Check against known laws
            match = self.check(law)
            if match:
                redundant.append((law, match))
                continue

            # Check against earlier items in this batch (exact hash)
            exact_hash = self._compute_exact_hash(law)
            if exact_hash in batch_hashes:
                match = RedundancyMatch(
                    matched_law_id=batch_hashes[exact_hash].law_id,
                    similarity=1.0,
                    match_type="batch_duplicate",
                    details="Exact duplicate within same batch",
                )
                redundant.append((law, match))
                continue

            # Check against earlier items in this batch (semantic fingerprint)
            fingerprint = compute_semantic_fingerprint(law)
            if fingerprint in batch_fingerprints:
                match = RedundancyMatch(
                    matched_law_id=batch_fingerprints[fingerprint].law_id,
                    similarity=1.0,
                    match_type="batch_fingerprint",
                    details="Semantic duplicate within same batch",
                )
                redundant.append((law, match))
                continue

            # Not redundant
            batch_hashes[exact_hash] = law
            batch_fingerprints[fingerprint] = law
            non_redundant.append(law)

        return non_redundant, redundant

    def _compute_exact_hash(self, law: CandidateLaw) -> str:
        """Compute exact content hash for a law."""
        return law.content_hash()

    def _normalize_law(self, law: CandidateLaw) -> str:
        """Normalize a law to canonical form.

        Normalization includes:
        - Sorting preconditions
        - Sorting observables by name
        - Removing whitespace from expressions
        - Converting to lowercase
        """
        parts = []

        # Template
        parts.append(law.template.value)

        # Preconditions (sorted)
        preconds = sorted(
            f"{p.lhs}{p.op.value}{p.rhs}".lower().replace(" ", "")
            for p in law.preconditions
        )
        parts.append("|".join(preconds))

        # Observables (sorted by name)
        observables = sorted(
            f"{o.name}={o.expr}".lower().replace(" ", "")
            for o in law.observables
        )
        parts.append("|".join(observables))

        # Claim (normalized)
        claim = law.claim.lower().replace(" ", "")
        parts.append(claim)

        # Transform (if symmetry)
        if law.transform:
            parts.append(law.transform.lower())

        return "::".join(parts)

    def _compute_similarity(self, law1: CandidateLaw, law2: CandidateLaw) -> float:
        """Compute semantic similarity between two laws.

        Returns a score from 0 to 1.
        """
        score = 0.0
        weights_total = 0.0

        # Template match (weight: 0.3)
        weights_total += 0.3
        if law1.template == law2.template:
            score += 0.3

        # Observable names overlap (weight: 0.2)
        weights_total += 0.2
        obs1 = {o.name for o in law1.observables}
        obs2 = {o.name for o in law2.observables}
        if obs1 and obs2:
            overlap = len(obs1 & obs2) / len(obs1 | obs2)
            score += 0.2 * overlap
        elif not obs1 and not obs2:
            score += 0.2  # Both empty

        # Claim token similarity (weight: 0.3)
        weights_total += 0.3
        claim_sim = self._token_similarity(law1.claim, law2.claim)
        score += 0.3 * claim_sim

        # Precondition overlap (weight: 0.2)
        weights_total += 0.2
        pre1 = {f"{p.lhs}{p.op.value}" for p in law1.preconditions}
        pre2 = {f"{p.lhs}{p.op.value}" for p in law2.preconditions}
        if pre1 and pre2:
            overlap = len(pre1 & pre2) / len(pre1 | pre2)
            score += 0.2 * overlap
        elif not pre1 and not pre2:
            score += 0.2

        return score / weights_total if weights_total > 0 else 0.0

    def _token_similarity(self, text1: str, text2: str) -> float:
        """Compute token-level similarity between two texts."""
        # Simple tokenization
        tokens1 = set(text1.lower().replace("(", " ").replace(")", " ").split())
        tokens2 = set(text2.lower().replace("(", " ").replace(")", " ").split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def clear(self) -> None:
        """Clear all known laws."""
        self._known_hashes = {}
        self._known_normalized = {}
        self._known_laws = {}

    @property
    def known_count(self) -> int:
        """Number of known laws."""
        return len(self._known_laws)
