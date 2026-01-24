"""Counterexample canonicalization and failure key tracking.

Canonicalizes counterexamples into unique failure keys for:
- Detecting repeated failure patterns across different laws
- Computing new_cex_rate = unique_failure_keys / total_falsifications
- Identifying when falsification is finding the same bugs repeatedly

Canonicalization includes:
- Shift-invariant rotation: lexicographically smallest rotation of state
- Trajectory snippet canonicalization
- Observable vector at failure
"""

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.harness.verdict import Counterexample


def canonical_rotation(state: str) -> str:
    """Find the lexicographically smallest rotation of a state string.

    This makes the representation shift-invariant - the same pattern
    at different positions will have the same canonical form.

    Examples:
        ">.<.." -> "..>.<"  (rotate to get smallest)
        "..><." -> "..><."  (already smallest)
        "X...." -> "....X"  (rotate X to end)

    Args:
        state: The state string to canonicalize

    Returns:
        The lexicographically smallest rotation
    """
    if not state:
        return state

    n = len(state)
    if n == 1:
        return state

    # Generate all rotations and find the minimum
    # For efficiency, use the "minimum rotation" algorithm
    rotations = [state[i:] + state[:i] for i in range(n)]
    return min(rotations)


def canonical_trajectory_snippet(
    trajectory: list[str],
    center_t: int,
    window: int = 2,
) -> list[str]:
    """Canonicalize a trajectory snippet around a time point.

    Extracts a window around center_t and canonicalizes each state.

    Args:
        trajectory: Full trajectory
        center_t: Time step to center on (usually t_fail)
        window: Number of steps before and after to include

    Returns:
        List of canonicalized states
    """
    if not trajectory:
        return []

    start = max(0, center_t - window)
    end = min(len(trajectory), center_t + window + 1)

    return [canonical_rotation(trajectory[i]) for i in range(start, end)]


def observable_vector(
    observables: dict[str, int | float] | None,
    sort_keys: bool = True,
) -> tuple[tuple[str, Any], ...]:
    """Create a hashable observable vector.

    Args:
        observables: Dictionary of observable name -> value
        sort_keys: Whether to sort by key for determinism

    Returns:
        Tuple of (name, value) pairs
    """
    if not observables:
        return ()

    items = list(observables.items())
    if sort_keys:
        items.sort(key=lambda x: x[0])

    return tuple(items)


@dataclass
class FailureKey:
    """A canonical key identifying a unique failure pattern.

    Two counterexamples with the same FailureKey represent the same
    underlying failure pattern, even if they differ in:
    - Absolute position (rotation)
    - Specific timing
    - Non-essential observable values

    Attributes:
        canonical_initial: Shift-invariant initial state
        canonical_fail_state: Shift-invariant state at failure
        t_fail_relative: Relative timing (for pattern matching)
        observable_signature: Key observables at failure
        trajectory_signature: Canonical trajectory snippet hash
        key_hash: Compact hash for efficient comparison
    """

    canonical_initial: str
    canonical_fail_state: str | None
    t_fail_relative: int  # 0 = immediate, 1 = one step, etc.
    observable_signature: tuple[tuple[str, Any], ...]
    trajectory_signature: str | None
    key_hash: str

    def __hash__(self) -> int:
        return hash(self.key_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FailureKey):
            return False
        return self.key_hash == other.key_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_initial": self.canonical_initial,
            "canonical_fail_state": self.canonical_fail_state,
            "t_fail_relative": self.t_fail_relative,
            "observable_signature": list(self.observable_signature),
            "trajectory_signature": self.trajectory_signature,
            "key_hash": self.key_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureKey":
        return cls(
            canonical_initial=data["canonical_initial"],
            canonical_fail_state=data.get("canonical_fail_state"),
            t_fail_relative=data["t_fail_relative"],
            observable_signature=tuple(
                tuple(x) for x in data.get("observable_signature", [])
            ),
            trajectory_signature=data.get("trajectory_signature"),
            key_hash=data["key_hash"],
        )


def compute_failure_key(
    counterexample: Counterexample,
    include_trajectory: bool = True,
    include_observables: bool = True,
    observable_filter: set[str] | None = None,
) -> FailureKey:
    """Compute a canonical failure key for a counterexample.

    Args:
        counterexample: The counterexample to canonicalize
        include_trajectory: Include trajectory snippet in key
        include_observables: Include observables in key
        observable_filter: If set, only include these observable names

    Returns:
        FailureKey uniquely identifying this failure pattern
    """
    ce = counterexample

    # Canonical initial state
    canonical_initial = canonical_rotation(ce.initial_state)

    # Canonical fail state (from trajectory if available)
    canonical_fail_state = None
    if ce.trajectory_excerpt and len(ce.trajectory_excerpt) > 0:
        # Find the state at t_fail within the excerpt
        # The excerpt is typically centered around t_fail
        fail_idx = min(ce.t_fail, len(ce.trajectory_excerpt) - 1)
        if fail_idx >= 0:
            canonical_fail_state = canonical_rotation(ce.trajectory_excerpt[fail_idx])

    # Relative timing (binned for pattern matching)
    # 0 = immediate (t=0), 1 = early (t=1-2), 2 = mid (t=3-5), 3 = late (t>5)
    if ce.t_fail == 0:
        t_fail_relative = 0
    elif ce.t_fail <= 2:
        t_fail_relative = 1
    elif ce.t_fail <= 5:
        t_fail_relative = 2
    else:
        t_fail_relative = 3

    # Observable signature
    observable_signature: tuple[tuple[str, Any], ...] = ()
    if include_observables and ce.observables_at_fail:
        obs = ce.observables_at_fail
        if observable_filter:
            obs = {k: v for k, v in obs.items() if k in observable_filter}
        observable_signature = observable_vector(obs)

    # Trajectory signature (hash of canonical snippet)
    trajectory_signature = None
    if include_trajectory and ce.trajectory_excerpt:
        canonical_traj = [canonical_rotation(s) for s in ce.trajectory_excerpt]
        traj_str = "|".join(canonical_traj)
        trajectory_signature = hashlib.sha256(traj_str.encode()).hexdigest()[:12]

    # Compute overall key hash
    key_parts = [
        f"init:{canonical_initial}",
        f"fail:{canonical_fail_state or 'none'}",
        f"t:{t_fail_relative}",
    ]
    if observable_signature:
        key_parts.append(f"obs:{observable_signature}")
    if trajectory_signature:
        key_parts.append(f"traj:{trajectory_signature}")

    key_str = "|".join(key_parts)
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:24]

    return FailureKey(
        canonical_initial=canonical_initial,
        canonical_fail_state=canonical_fail_state,
        t_fail_relative=t_fail_relative,
        observable_signature=observable_signature,
        trajectory_signature=trajectory_signature,
        key_hash=key_hash,
    )


@dataclass
class FailureKeyStats:
    """Statistics for failure key tracking over a window."""

    window_size: int
    total_falsifications: int
    unique_failure_keys: int

    @property
    def new_cex_rate(self) -> float:
        """Rate of unique failure keys (novelty of falsifications)."""
        if self.total_falsifications == 0:
            return 1.0
        return self.unique_failure_keys / self.total_falsifications

    @property
    def repetition_rate(self) -> float:
        """Rate of repeated failure patterns."""
        return 1.0 - self.new_cex_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "total_falsifications": self.total_falsifications,
            "unique_failure_keys": self.unique_failure_keys,
            "new_cex_rate": self.new_cex_rate,
            "repetition_rate": self.repetition_rate,
        }


@dataclass
class FailureKeyResult:
    """Result of failure key computation and tracking."""

    failure_key: FailureKey
    is_novel: bool  # First time seeing this key
    occurrence_count: int  # How many times we've seen this key
    similar_keys: list[str] = field(default_factory=list)  # Related keys

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_key": self.failure_key.to_dict(),
            "is_novel": self.is_novel,
            "occurrence_count": self.occurrence_count,
            "similar_keys": self.similar_keys,
        }


class FailureKeyTracker:
    """Tracks failure keys for counterexample novelty detection.

    Maintains a registry of seen failure patterns and computes
    new_cex_rate over a sliding window.
    """

    def __init__(
        self,
        window_size: int = 50,
        include_trajectory: bool = True,
        include_observables: bool = True,
    ):
        """Initialize failure key tracker.

        Args:
            window_size: Size of sliding window for rate computation
            include_trajectory: Include trajectory in key computation
            include_observables: Include observables in key computation
        """
        self.window_size = window_size
        self.include_trajectory = include_trajectory
        self.include_observables = include_observables

        # Registry of all seen keys (key_hash -> count)
        self._seen_keys: dict[str, int] = {}

        # Registry of key details (key_hash -> FailureKey)
        self._key_details: dict[str, FailureKey] = {}

        # Sliding window of recent results
        self._window: deque[FailureKeyResult] = deque(maxlen=window_size)

        # Total counts
        self._total_falsifications = 0

    def track(
        self,
        counterexample: Counterexample,
        observable_filter: set[str] | None = None,
    ) -> FailureKeyResult:
        """Track a new counterexample and compute its failure key.

        Args:
            counterexample: The counterexample to track
            observable_filter: Optional filter for which observables to include

        Returns:
            FailureKeyResult with novelty information
        """
        self._total_falsifications += 1

        # Compute failure key
        key = compute_failure_key(
            counterexample,
            include_trajectory=self.include_trajectory,
            include_observables=self.include_observables,
            observable_filter=observable_filter,
        )

        # Check if novel
        is_novel = key.key_hash not in self._seen_keys

        # Update registry
        if is_novel:
            self._seen_keys[key.key_hash] = 0
            self._key_details[key.key_hash] = key

        self._seen_keys[key.key_hash] += 1
        occurrence_count = self._seen_keys[key.key_hash]

        # Find similar keys (same canonical initial state)
        similar_keys = [
            kh for kh, k in self._key_details.items()
            if k.canonical_initial == key.canonical_initial and kh != key.key_hash
        ]

        result = FailureKeyResult(
            failure_key=key,
            is_novel=is_novel,
            occurrence_count=occurrence_count,
            similar_keys=similar_keys[:5],  # Limit to top 5
        )

        # Add to sliding window
        self._window.append(result)

        return result

    def get_window_stats(self) -> FailureKeyStats:
        """Get statistics for the current window."""
        if not self._window:
            return FailureKeyStats(
                window_size=self.window_size,
                total_falsifications=0,
                unique_failure_keys=0,
            )

        # Count unique keys in window
        window_keys = {r.failure_key.key_hash for r in self._window}

        return FailureKeyStats(
            window_size=self.window_size,
            total_falsifications=len(self._window),
            unique_failure_keys=len(window_keys),
        )

    def get_top_failure_patterns(self, limit: int = 10) -> list[tuple[FailureKey, int]]:
        """Get the most common failure patterns.

        Returns:
            List of (FailureKey, count) tuples sorted by count descending
        """
        sorted_keys = sorted(
            self._seen_keys.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

        return [
            (self._key_details[key_hash], count)
            for key_hash, count in sorted_keys
        ]

    def get_failure_pattern_by_initial_state(
        self,
        initial_state: str,
    ) -> list[tuple[FailureKey, int]]:
        """Get all failure patterns for a given initial state.

        Uses canonical rotation for matching.

        Args:
            initial_state: The initial state to look up

        Returns:
            List of (FailureKey, count) tuples
        """
        canonical = canonical_rotation(initial_state)
        results = []

        for key_hash, key in self._key_details.items():
            if key.canonical_initial == canonical:
                count = self._seen_keys[key_hash]
                results.append((key, count))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of failure key tracking."""
        window_stats = self.get_window_stats()
        top_patterns = self.get_top_failure_patterns(5)

        return {
            "total_falsifications": self._total_falsifications,
            "unique_failure_keys_total": len(self._seen_keys),
            "window_stats": window_stats.to_dict(),
            "top_patterns": [
                {
                    "canonical_initial": key.canonical_initial,
                    "t_fail_relative": key.t_fail_relative,
                    "count": count,
                }
                for key, count in top_patterns
            ],
        }

    def reset(self) -> None:
        """Reset all tracking state."""
        self._seen_keys.clear()
        self._key_details.clear()
        self._window.clear()
        self._total_falsifications = 0

    def seed_known_keys(self, keys: dict[str, int]) -> None:
        """Seed with known failure keys from a previous run.

        Args:
            keys: Dictionary of key_hash -> count
        """
        self._seen_keys.update(keys)


# Grouping similar failures by structural features


def group_failures_by_structure(
    failure_keys: list[FailureKey],
) -> dict[str, list[FailureKey]]:
    """Group failure keys by structural similarity.

    Groups by:
    - Same canonical initial state
    - Same relative timing class
    - Same observable pattern

    Args:
        failure_keys: List of failure keys to group

    Returns:
        Dictionary mapping group key -> list of FailureKeys
    """
    groups: dict[str, list[FailureKey]] = {}

    for key in failure_keys:
        # Create group key based on structure
        group_key = f"{key.canonical_initial}|t{key.t_fail_relative}"
        if key.observable_signature:
            # Include observable names (not values) in group
            obs_names = sorted(name for name, _ in key.observable_signature)
            group_key += f"|{','.join(obs_names)}"

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(key)

    return groups


# Module-level tracker instance
_failure_key_tracker = FailureKeyTracker()


def track_failure(
    counterexample: Counterexample,
    observable_filter: set[str] | None = None,
) -> FailureKeyResult:
    """Track a counterexample using module-level tracker."""
    return _failure_key_tracker.track(counterexample, observable_filter)


def get_failure_key_stats() -> FailureKeyStats:
    """Get current stats from module-level tracker."""
    return _failure_key_tracker.get_window_stats()


def get_failure_key_tracker() -> FailureKeyTracker:
    """Get the module-level failure key tracker."""
    return _failure_key_tracker


def reset_failure_key_tracker() -> None:
    """Reset the module-level failure key tracker."""
    _failure_key_tracker.reset()
