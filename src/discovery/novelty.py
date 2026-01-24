"""Novelty tracking for candidate law discovery.

Tracks two dimensions of novelty:
1. Syntactic: Normalized AST fingerprint (detects structurally equivalent laws)
2. Semantic: Behavior over probe trajectories (detects functionally equivalent laws)

Used for:
- Detecting saturation (when discovery stops producing novel laws)
- Computing novelty_rate = unique_signatures / total_candidates
- Alerting when last K laws are >X% not novel
"""

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.claims.ast_evaluator import ASTClaimEvaluator, ASTEvaluationError
from src.claims.fingerprint import canonicalize_ast, compute_semantic_fingerprint
from src.claims.schema import CandidateLaw, Template
from src.universe.simulator import run, step
from src.universe.types import Trajectory


# Fixed probe suite for semantic fingerprinting
# Designed to cover various edge cases and behaviors
PROBE_INITIAL_STATES = [
    # Single particle cases
    ">.....",  # Single right-mover
    ".....<",  # Single left-mover
    "...X..",  # Single collision

    # Two particle cases
    ">.<...",  # Converging pair
    ">..<..",  # Non-converging pair
    "><....",  # Adjacent opposite
    ".>...<",  # Will collide later

    # Collision chains
    ">.<.>.<",  # Multiple converging pairs
    "X..X..",  # Multiple collisions

    # High density
    "><><><",  # Alternating full
    ">>>...",  # Multiple same direction
    "..<<<.",  # Multiple same direction

    # Edge cases
    "",       # Empty
    ">",      # Minimal L=1
    "><",     # Minimal L=2
    "X",      # Single X at L=1

    # Mixed
    ">X<.>.",  # Complex mixed state
    "..>.X.<.",  # Sparse with collision
]

# Time horizon for probe trajectories
PROBE_TIME_HORIZON = 15


@dataclass
class ProbeResult:
    """Result of evaluating a law component on a probe trajectory."""
    initial_state: str
    trajectory_length: int
    # For each time step: value or None if error
    values: list[int | float | bool | None]
    error: str | None = None


@dataclass
class SemanticSignature:
    """Semantic signature from evaluating on probe suite."""
    # Hash of the behavior vector
    signature_hash: str
    # Raw behavior data for debugging
    behavior_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature_hash": self.signature_hash,
            "behavior_summary": self.behavior_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticSignature":
        return cls(
            signature_hash=data["signature_hash"],
            behavior_summary=data.get("behavior_summary", {}),
        )


@dataclass
class NoveltyResult:
    """Result of novelty check for a candidate law."""
    # Syntactic fingerprint (from AST canonicalization)
    syntactic_fingerprint: str
    is_syntactically_novel: bool

    # Semantic signature (from probe suite evaluation)
    semantic_signature: SemanticSignature | None
    is_semantically_novel: bool

    # Combined
    is_novel: bool  # Novel by at least one measure
    is_fully_novel: bool  # Novel by both measures

    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "syntactic_fingerprint": self.syntactic_fingerprint,
            "is_syntactically_novel": self.is_syntactically_novel,
            "semantic_signature": (
                self.semantic_signature.to_dict()
                if self.semantic_signature else None
            ),
            "is_semantically_novel": self.is_semantically_novel,
            "is_novel": self.is_novel,
            "is_fully_novel": self.is_fully_novel,
            "reason": self.reason,
        }


class ProbeSuite:
    """Fixed suite of probe trajectories for semantic fingerprinting."""

    def __init__(
        self,
        initial_states: list[str] | None = None,
        time_horizon: int = PROBE_TIME_HORIZON,
    ):
        self.initial_states = initial_states or PROBE_INITIAL_STATES
        self.time_horizon = time_horizon
        self._trajectories: dict[str, Trajectory] = {}
        self._generate_trajectories()

    def _generate_trajectories(self) -> None:
        """Pre-generate all probe trajectories."""
        for state in self.initial_states:
            if state:  # Skip empty state
                try:
                    traj = run(state, self.time_horizon)
                    self._trajectories[state] = traj
                except Exception:
                    # Invalid state - skip
                    pass
            else:
                # Empty state has trivial trajectory
                self._trajectories[""] = [""]

    def get_trajectory(self, initial_state: str) -> Trajectory | None:
        """Get pre-computed trajectory for an initial state."""
        return self._trajectories.get(initial_state)

    def get_all_trajectories(self) -> dict[str, Trajectory]:
        """Get all probe trajectories."""
        return dict(self._trajectories)


class SemanticEvaluator:
    """Evaluates law semantics over probe trajectories."""

    def __init__(self, probe_suite: ProbeSuite | None = None):
        self.probe_suite = probe_suite or ProbeSuite()

    def compute_signature(self, law: CandidateLaw) -> SemanticSignature:
        """Compute semantic signature for a law.

        Evaluates the law's claim components over all probe trajectories
        and creates a hash of the resulting behavior vector.
        """
        behavior_vector: list[Any] = []
        behavior_summary: dict[str, Any] = {
            "template": law.template.value,
            "probe_count": len(self.probe_suite.get_all_trajectories()),
            "results": {},
        }

        # Try to create an evaluator
        try:
            evaluator = ASTClaimEvaluator(law)
        except (ASTEvaluationError, Exception) as e:
            # Law can't be evaluated - use error as signature
            # Catch all exceptions since various parsing/validation errors can occur
            error_hash = hashlib.sha256(f"error:{e}".encode()).hexdigest()[:24]
            behavior_summary["error"] = str(e)
            return SemanticSignature(
                signature_hash=f"err_{error_hash}",
                behavior_summary=behavior_summary,
            )

        # Evaluate on each probe trajectory
        for initial_state, trajectory in self.probe_suite.get_all_trajectories().items():
            try:
                result = self._evaluate_on_trajectory(law, evaluator, trajectory)
                behavior_vector.append(result)
                behavior_summary["results"][initial_state] = result
            except Exception as e:
                # Record error as part of signature
                behavior_vector.append(f"error:{type(e).__name__}")
                behavior_summary["results"][initial_state] = f"error:{e}"

        # Hash the behavior vector
        vector_str = json.dumps(behavior_vector, sort_keys=True)
        signature_hash = hashlib.sha256(vector_str.encode()).hexdigest()[:24]

        return SemanticSignature(
            signature_hash=signature_hash,
            behavior_summary=behavior_summary,
        )

    def _evaluate_on_trajectory(
        self,
        law: CandidateLaw,
        evaluator: ASTClaimEvaluator,
        trajectory: Trajectory,
    ) -> dict[str, Any]:
        """Evaluate law on a single trajectory.

        Returns a dictionary capturing the semantic behavior.
        """
        result: dict[str, Any] = {
            "trajectory_length": len(trajectory),
        }

        if not law.claim_ast:
            result["no_ast"] = True
            return result

        ast = law.claim_ast

        # Extract components based on template
        if law.template in (Template.IMPLICATION_STEP, Template.IMPLICATION_STATE):
            # For implications, evaluate antecedent and consequent separately
            if ast.get("op") == "=>":
                antecedent = ast.get("lhs", {})
                consequent = ast.get("rhs", {})

                result["antecedent_values"] = self._evaluate_component(
                    evaluator, antecedent, trajectory
                )
                result["consequent_values"] = self._evaluate_component(
                    evaluator, consequent, trajectory
                )
            else:
                # No implication structure - evaluate whole thing
                result["claim_values"] = self._evaluate_component(
                    evaluator, ast, trajectory
                )
        else:
            # For other templates, evaluate the whole claim
            result["claim_values"] = self._evaluate_component(
                evaluator, ast, trajectory
            )

        return result

    def _evaluate_component(
        self,
        evaluator: ASTClaimEvaluator,
        ast: dict[str, Any],
        trajectory: Trajectory,
    ) -> list[Any]:
        """Evaluate an AST component at each time step."""
        values = []
        for t in range(len(trajectory)):
            try:
                val = evaluator.evaluate_at_time(ast, trajectory, t)
                values.append(val)
            except Exception:
                values.append(None)
        return values


@dataclass
class NoveltyStats:
    """Statistics for novelty tracking over a window."""
    window_size: int
    total_laws: int
    syntactically_novel: int
    semantically_novel: int
    fully_novel: int

    @property
    def syntactic_novelty_rate(self) -> float:
        """Rate of syntactically novel laws in window."""
        return self.syntactically_novel / self.total_laws if self.total_laws > 0 else 1.0

    @property
    def semantic_novelty_rate(self) -> float:
        """Rate of semantically novel laws in window."""
        return self.semantically_novel / self.total_laws if self.total_laws > 0 else 1.0

    @property
    def combined_novelty_rate(self) -> float:
        """Rate of laws novel by either measure."""
        novel_either = self.syntactically_novel + self.semantically_novel - self.fully_novel
        return novel_either / self.total_laws if self.total_laws > 0 else 1.0

    @property
    def full_novelty_rate(self) -> float:
        """Rate of laws novel by both measures."""
        return self.fully_novel / self.total_laws if self.total_laws > 0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "total_laws": self.total_laws,
            "syntactically_novel": self.syntactically_novel,
            "semantically_novel": self.semantically_novel,
            "fully_novel": self.fully_novel,
            "syntactic_novelty_rate": self.syntactic_novelty_rate,
            "semantic_novelty_rate": self.semantic_novelty_rate,
            "combined_novelty_rate": self.combined_novelty_rate,
            "full_novelty_rate": self.full_novelty_rate,
        }


class NoveltyTracker:
    """Tracks novelty of candidate laws over time.

    Maintains sliding windows of novelty results and detects saturation.
    """

    def __init__(
        self,
        window_size: int = 50,
        saturation_threshold: float = 0.2,
        probe_suite: ProbeSuite | None = None,
    ):
        """Initialize novelty tracker.

        Args:
            window_size: Size of sliding window for novelty stats
            saturation_threshold: Novelty rate below which we're "saturated"
            probe_suite: Probe suite for semantic evaluation
        """
        self.window_size = window_size
        self.saturation_threshold = saturation_threshold
        self.probe_suite = probe_suite or ProbeSuite()
        self.semantic_evaluator = SemanticEvaluator(self.probe_suite)

        # Sets of all seen fingerprints (ever)
        self._seen_syntactic: set[str] = set()
        self._seen_semantic: set[str] = set()

        # Sliding window of recent novelty results
        self._window: deque[NoveltyResult] = deque(maxlen=window_size)

        # Total counts
        self._total_laws = 0

    def check_novelty(self, law: CandidateLaw) -> NoveltyResult:
        """Check if a law is novel.

        Args:
            law: The candidate law to check

        Returns:
            NoveltyResult with syntactic and semantic novelty info
        """
        self._total_laws += 1

        # Syntactic fingerprint (uses existing canonicalization)
        syntactic_fp = compute_semantic_fingerprint(law)
        is_syntactically_novel = syntactic_fp not in self._seen_syntactic

        # Semantic signature
        semantic_sig = self.semantic_evaluator.compute_signature(law)
        is_semantically_novel = semantic_sig.signature_hash not in self._seen_semantic

        # Register fingerprints
        self._seen_syntactic.add(syntactic_fp)
        self._seen_semantic.add(semantic_sig.signature_hash)

        # Build result
        is_novel = is_syntactically_novel or is_semantically_novel
        is_fully_novel = is_syntactically_novel and is_semantically_novel

        # Build reason string
        reasons = []
        if is_syntactically_novel:
            reasons.append("syntactically novel")
        else:
            reasons.append("syntactic duplicate")
        if is_semantically_novel:
            reasons.append("semantically novel")
        else:
            reasons.append("semantic duplicate")

        result = NoveltyResult(
            syntactic_fingerprint=syntactic_fp,
            is_syntactically_novel=is_syntactically_novel,
            semantic_signature=semantic_sig,
            is_semantically_novel=is_semantically_novel,
            is_novel=is_novel,
            is_fully_novel=is_fully_novel,
            reason="; ".join(reasons),
        )

        # Add to sliding window
        self._window.append(result)

        return result

    def get_window_stats(self) -> NoveltyStats:
        """Get novelty statistics for the current window."""
        if not self._window:
            return NoveltyStats(
                window_size=self.window_size,
                total_laws=0,
                syntactically_novel=0,
                semantically_novel=0,
                fully_novel=0,
            )

        return NoveltyStats(
            window_size=self.window_size,
            total_laws=len(self._window),
            syntactically_novel=sum(1 for r in self._window if r.is_syntactically_novel),
            semantically_novel=sum(1 for r in self._window if r.is_semantically_novel),
            fully_novel=sum(1 for r in self._window if r.is_fully_novel),
        )

    def is_saturated(self) -> bool:
        """Check if discovery is saturated (low novelty rate).

        Returns True if the novelty rate in the window is below threshold.
        """
        stats = self.get_window_stats()

        # Need at least half the window to make a decision
        if stats.total_laws < self.window_size // 2:
            return False

        return stats.combined_novelty_rate < self.saturation_threshold

    def get_saturation_report(self) -> dict[str, Any]:
        """Get a detailed saturation report."""
        stats = self.get_window_stats()
        return {
            "is_saturated": self.is_saturated(),
            "saturation_threshold": self.saturation_threshold,
            "total_laws_seen": self._total_laws,
            "unique_syntactic_fingerprints": len(self._seen_syntactic),
            "unique_semantic_signatures": len(self._seen_semantic),
            "window_stats": stats.to_dict(),
        }

    def reset(self) -> None:
        """Reset all tracking state."""
        self._seen_syntactic.clear()
        self._seen_semantic.clear()
        self._window.clear()
        self._total_laws = 0

    def seed_known_fingerprints(
        self,
        syntactic_fps: set[str] | None = None,
        semantic_hashes: set[str] | None = None,
    ) -> None:
        """Seed the tracker with known fingerprints from a previous run.

        Args:
            syntactic_fps: Known syntactic fingerprints
            semantic_hashes: Known semantic signature hashes
        """
        if syntactic_fps:
            self._seen_syntactic.update(syntactic_fps)
        if semantic_hashes:
            self._seen_semantic.update(semantic_hashes)


# Module-level instances for convenience
_probe_suite = ProbeSuite()
_novelty_tracker = NoveltyTracker(probe_suite=_probe_suite)


def check_law_novelty(law: CandidateLaw) -> NoveltyResult:
    """Check novelty of a law using module-level tracker."""
    return _novelty_tracker.check_novelty(law)


def get_novelty_stats() -> NoveltyStats:
    """Get current novelty stats from module-level tracker."""
    return _novelty_tracker.get_window_stats()


def is_discovery_saturated() -> bool:
    """Check if discovery is saturated using module-level tracker."""
    return _novelty_tracker.is_saturated()


def get_novelty_tracker() -> NoveltyTracker:
    """Get the module-level novelty tracker."""
    return _novelty_tracker


def reset_novelty_tracker() -> None:
    """Reset the module-level novelty tracker."""
    _novelty_tracker.reset()
