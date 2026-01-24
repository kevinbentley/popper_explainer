"""Readiness metrics computation.

Computes objective readiness metrics from database state.
These metrics are used by the orchestrator to make transition decisions,
independent of (but informed by) LLM recommendations.

Key principle: The LLM can ADVISE readiness, but the orchestrator COMPUTES
the actual readiness score from harness data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.db.repo import Repository
    from src.orchestration.phases import Phase


# Default weights for readiness score computation
DEFAULT_DISCOVERY_WEIGHTS = {
    "s_pass": 0.25,  # Fraction of core laws passing
    "s_stability": 0.20,  # Promoted law set stability
    "s_novel_cex": 0.20,  # Novel counterexample rate (lower is better for advance)
    "s_harness_health": 0.15,  # No harness errors
    "s_redundancy": 0.20,  # New law redundancy rate (higher indicates saturation)
}

DEFAULT_THEOREM_WEIGHTS = {
    "s_pass": 0.30,  # Laws with PASS status
    "s_stability": 0.25,  # Theorem set stability
    "s_coverage": 0.25,  # Template/behavior coverage
    "s_harness_health": 0.20,  # No errors
}

DEFAULT_EXPLANATION_WEIGHTS = {
    "s_prediction_accuracy": 0.40,  # One-step prediction accuracy
    "s_mechanism_completeness": 0.30,  # Mechanism defined
    "s_theorem_coverage": 0.30,  # Theorems addressed
}

DEFAULT_PREDICTION_WEIGHTS = {
    "s_held_out_accuracy": 0.40,  # Held-out set accuracy
    "s_adversarial_accuracy": 0.35,  # Adversarial set accuracy
    "s_consistency": 0.25,  # Prediction consistency
}


@dataclass
class ReadinessMetrics:
    """Objective readiness metrics computed from harness data.

    These metrics are phase-specific and used to determine
    whether a phase transition should occur.
    """

    # Discovery phase metrics
    s_pass: float = 0.0  # Fraction of core law suite passing
    s_stability: float = 0.0  # Promoted-law stability over K iterations
    s_novel_cex: float = 1.0  # Novel counterexample rate (0 = saturated)
    s_harness_health: float = 1.0  # 1.0 if no errors, decreases with issues
    s_redundancy: float = 0.0  # Fraction of new laws semantically redundant

    # Theorem phase metrics
    s_coverage: float = 0.0  # Template/behavior coverage

    # Explanation phase metrics
    s_mechanism_completeness: float = 0.0  # Is mechanism well-defined
    s_theorem_coverage: float = 0.0  # Fraction of theorems addressed

    # Prediction phase metrics
    s_prediction_accuracy: float | None = None  # One-step prediction accuracy
    s_adversarial_accuracy: float | None = None  # Adversarial set accuracy
    s_held_out_accuracy: float | None = None  # Held-out set accuracy
    s_consistency: float | None = None  # Prediction consistency

    # Computed combined score
    combined_score: float = 0.0

    # Weights used for combination
    weights: dict[str, float] = field(default_factory=dict)

    # Source data for auditability
    source_counts: dict[str, int] = field(default_factory=dict)

    def compute_discovery_readiness(
        self, weights: dict[str, float] | None = None
    ) -> float:
        """Compute weighted readiness for discovery phase.

        Higher score means more ready to advance to theorem phase.
        """
        w = weights or DEFAULT_DISCOVERY_WEIGHTS
        self.weights = w

        # Invert novel_cex (lower is better for advancement)
        # If novel_cex is high (lots of new counterexamples), we're still learning
        # If novel_cex is low (few new counterexamples), we're saturated
        s_novel_cex_inverted = 1.0 - self.s_novel_cex

        self.combined_score = (
            w.get("s_pass", 0.25) * self.s_pass
            + w.get("s_stability", 0.20) * self.s_stability
            + w.get("s_novel_cex", 0.20) * s_novel_cex_inverted
            + w.get("s_harness_health", 0.15) * self.s_harness_health
            + w.get("s_redundancy", 0.20) * self.s_redundancy
        )

        # Scale to 0-100
        return self.combined_score * 100

    def compute_theorem_readiness(
        self, weights: dict[str, float] | None = None
    ) -> float:
        """Compute weighted readiness for theorem phase."""
        w = weights or DEFAULT_THEOREM_WEIGHTS
        self.weights = w

        self.combined_score = (
            w.get("s_pass", 0.30) * self.s_pass
            + w.get("s_stability", 0.25) * self.s_stability
            + w.get("s_coverage", 0.25) * self.s_coverage
            + w.get("s_harness_health", 0.20) * self.s_harness_health
        )

        return self.combined_score * 100

    def compute_explanation_readiness(
        self, weights: dict[str, float] | None = None
    ) -> float:
        """Compute weighted readiness for explanation phase."""
        w = weights or DEFAULT_EXPLANATION_WEIGHTS
        self.weights = w

        pred_acc = self.s_prediction_accuracy or 0.0

        self.combined_score = (
            w.get("s_prediction_accuracy", 0.40) * pred_acc
            + w.get("s_mechanism_completeness", 0.30) * self.s_mechanism_completeness
            + w.get("s_theorem_coverage", 0.30) * self.s_theorem_coverage
        )

        return self.combined_score * 100

    def compute_prediction_readiness(
        self, weights: dict[str, float] | None = None
    ) -> float:
        """Compute weighted readiness for prediction phase."""
        w = weights or DEFAULT_PREDICTION_WEIGHTS
        self.weights = w

        held_out = self.s_held_out_accuracy or 0.0
        adversarial = self.s_adversarial_accuracy or 0.0
        consistency = self.s_consistency or 0.0

        self.combined_score = (
            w.get("s_held_out_accuracy", 0.40) * held_out
            + w.get("s_adversarial_accuracy", 0.35) * adversarial
            + w.get("s_consistency", 0.25) * consistency
        )

        return self.combined_score * 100

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "s_pass": self.s_pass,
            "s_stability": self.s_stability,
            "s_novel_cex": self.s_novel_cex,
            "s_harness_health": self.s_harness_health,
            "s_redundancy": self.s_redundancy,
            "s_coverage": self.s_coverage,
            "s_mechanism_completeness": self.s_mechanism_completeness,
            "s_theorem_coverage": self.s_theorem_coverage,
            "s_prediction_accuracy": self.s_prediction_accuracy,
            "s_adversarial_accuracy": self.s_adversarial_accuracy,
            "s_held_out_accuracy": self.s_held_out_accuracy,
            "s_consistency": self.s_consistency,
            "combined_score": self.combined_score,
            "weights": self.weights,
            "source_counts": self.source_counts,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReadinessMetrics:
        """Deserialize from dictionary."""
        return cls(
            s_pass=data.get("s_pass", 0.0),
            s_stability=data.get("s_stability", 0.0),
            s_novel_cex=data.get("s_novel_cex", 1.0),
            s_harness_health=data.get("s_harness_health", 1.0),
            s_redundancy=data.get("s_redundancy", 0.0),
            s_coverage=data.get("s_coverage", 0.0),
            s_mechanism_completeness=data.get("s_mechanism_completeness", 0.0),
            s_theorem_coverage=data.get("s_theorem_coverage", 0.0),
            s_prediction_accuracy=data.get("s_prediction_accuracy"),
            s_adversarial_accuracy=data.get("s_adversarial_accuracy"),
            s_held_out_accuracy=data.get("s_held_out_accuracy"),
            s_consistency=data.get("s_consistency"),
            combined_score=data.get("combined_score", 0.0),
            weights=data.get("weights", {}),
            source_counts=data.get("source_counts", {}),
        )


class ReadinessComputer:
    """Computes readiness metrics from database state.

    This class queries the database to compute objective metrics
    that determine phase transition readiness.
    """

    def __init__(self, repo: Repository, window_size: int = 50):
        """Initialize readiness computer.

        Args:
            repo: Database repository for queries
            window_size: Window size for sliding metrics (e.g., novelty rate)
        """
        self.repo = repo
        self.window_size = window_size

    def compute_for_phase(
        self,
        run_id: str,
        phase: Phase,
        weights: dict[str, float] | None = None,
    ) -> ReadinessMetrics:
        """Compute readiness metrics for a specific phase.

        Args:
            run_id: Current orchestration run ID
            phase: Phase to compute readiness for
            weights: Optional custom weights for the score

        Returns:
            ReadinessMetrics with computed values
        """
        from src.orchestration.phases import Phase

        if phase == Phase.DISCOVERY:
            return self.compute_discovery_readiness(run_id, weights)
        elif phase == Phase.THEOREM:
            return self.compute_theorem_readiness(run_id, weights)
        elif phase == Phase.EXPLANATION:
            return self.compute_explanation_readiness(run_id, weights)
        elif phase == Phase.PREDICTION:
            return self.compute_prediction_readiness(run_id, weights)
        else:
            # FINALIZE phase - return high readiness
            return ReadinessMetrics(combined_score=1.0)

    def compute_discovery_readiness(
        self,
        run_id: str,
        weights: dict[str, float] | None = None,
    ) -> ReadinessMetrics:
        """Compute readiness metrics for discovery phase.

        Metrics:
        - s_pass: Fraction of laws with PASS status
        - s_stability: Whether promoted law set is stable
        - s_novel_cex: Rate of novel counterexamples (from novelty tracker)
        - s_harness_health: Absence of harness errors
        - s_redundancy: Rate of semantically redundant laws
        """
        metrics = ReadinessMetrics()

        # Get law counts by status
        pass_count = self._count_laws_by_status("PASS")
        fail_count = self._count_laws_by_status("FAIL")
        unknown_count = self._count_laws_by_status("UNKNOWN")
        total = pass_count + fail_count + unknown_count

        metrics.source_counts = {
            "pass": pass_count,
            "fail": fail_count,
            "unknown": unknown_count,
            "total": total,
        }

        # S_PASS: fraction of laws passing
        metrics.s_pass = pass_count / total if total > 0 else 0.0

        # S_STABILITY: check if promoted set is stable
        # For now, use a simple heuristic based on recent evaluations
        metrics.s_stability = self._compute_stability()

        # S_NOVEL_CEX: get from novelty snapshots
        metrics.s_novel_cex = self._compute_novel_cex_rate()

        # S_HARNESS_HEALTH: check for Type-D errors
        metrics.s_harness_health = self._compute_harness_health()

        # S_REDUNDANCY: check law novelty records
        metrics.s_redundancy = self._compute_redundancy_rate()

        # Compute combined score
        metrics.compute_discovery_readiness(weights)

        return metrics

    def compute_theorem_readiness(
        self,
        run_id: str,
        weights: dict[str, float] | None = None,
    ) -> ReadinessMetrics:
        """Compute readiness metrics for theorem phase."""
        metrics = ReadinessMetrics()

        # Get theorem counts
        theorem_count = self._count_theorems()
        established_count = self._count_theorems_by_status("Established")

        metrics.source_counts = {
            "theorems": theorem_count,
            "established": established_count,
        }

        # S_PASS: fraction of established theorems
        metrics.s_pass = established_count / theorem_count if theorem_count > 0 else 0.0

        # S_STABILITY: theorem set stability
        metrics.s_stability = self._compute_theorem_stability()

        # S_COVERAGE: template coverage
        metrics.s_coverage = self._compute_template_coverage()

        # S_HARNESS_HEALTH
        metrics.s_harness_health = self._compute_harness_health()

        # Compute combined score
        metrics.compute_theorem_readiness(weights)

        return metrics

    def compute_explanation_readiness(
        self,
        run_id: str,
        weights: dict[str, float] | None = None,
    ) -> ReadinessMetrics:
        """Compute readiness metrics for explanation phase."""
        metrics = ReadinessMetrics()

        # These will be populated when explanation phase is implemented
        metrics.s_prediction_accuracy = 0.0
        metrics.s_mechanism_completeness = 0.0
        metrics.s_theorem_coverage = 0.0

        metrics.compute_explanation_readiness(weights)

        return metrics

    def compute_prediction_readiness(
        self,
        run_id: str,
        weights: dict[str, float] | None = None,
    ) -> ReadinessMetrics:
        """Compute readiness metrics for prediction phase."""
        metrics = ReadinessMetrics()

        # These will be populated when prediction phase is implemented
        metrics.s_held_out_accuracy = 0.0
        metrics.s_adversarial_accuracy = 0.0
        metrics.s_consistency = 0.0

        metrics.compute_prediction_readiness(weights)

        return metrics

    # =========================================================================
    # Private helper methods for metric computation
    # =========================================================================

    def _count_laws_by_status(self, status: str) -> int:
        """Count laws with a specific evaluation status."""
        conn = self.repo._conn
        if not conn:
            return 0

        cursor = conn.execute(
            """
            SELECT COUNT(DISTINCT l.law_id)
            FROM laws l
            JOIN evaluations e ON l.law_id = e.law_id
            WHERE e.status = ?
            AND e.id = (
                SELECT MAX(e2.id) FROM evaluations e2 WHERE e2.law_id = l.law_id
            )
            """,
            (status,),
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def _compute_stability(self) -> float:
        """Compute law set stability based on recent evaluations.

        Returns 1.0 if no status changes in recent window, 0.0 if all changed.
        """
        conn = self.repo._conn
        if not conn:
            return 0.0

        # Get laws with multiple evaluations and check for status changes
        cursor = conn.execute(
            """
            SELECT COUNT(*) as changed
            FROM (
                SELECT law_id, COUNT(DISTINCT status) as status_count
                FROM evaluations
                GROUP BY law_id
                HAVING status_count > 1
            )
            """
        )
        row = cursor.fetchone()
        changed_count = row[0] if row else 0

        # Get total laws with evaluations
        cursor = conn.execute(
            "SELECT COUNT(DISTINCT law_id) FROM evaluations"
        )
        row = cursor.fetchone()
        total = row[0] if row else 0

        if total == 0:
            return 0.0

        # Stability = 1 - (changed / total)
        return 1.0 - (changed_count / total)

    def _compute_novel_cex_rate(self) -> float:
        """Get novel counterexample rate from most recent snapshot."""
        conn = self.repo._conn
        if not conn:
            return 1.0  # Assume high novelty if no data

        # Get most recent novelty snapshot
        cursor = conn.execute(
            """
            SELECT combined_novelty_rate
            FROM novelty_snapshots
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        return row[0] if row else 1.0

    def _compute_harness_health(self) -> float:
        """Compute harness health based on error classifications.

        Returns 1.0 if no Type-D errors, lower if errors present.
        """
        conn = self.repo._conn
        if not conn:
            return 1.0

        # Count Type-D (harness error) classifications in recent window
        cursor = conn.execute(
            """
            SELECT COUNT(*) as error_count
            FROM failure_classifications
            WHERE failure_class = 'type_d_harness_error'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (self.window_size,),
        )
        row = cursor.fetchone()
        error_count = row[0] if row else 0

        # Get total classifications in window
        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT id FROM failure_classifications
                ORDER BY created_at DESC
                LIMIT ?
            )
            """,
            (self.window_size,),
        )
        row = cursor.fetchone()
        total = row[0] if row else 0

        if total == 0:
            return 1.0

        return 1.0 - (error_count / total)

    def _compute_redundancy_rate(self) -> float:
        """Compute rate of semantically redundant laws."""
        conn = self.repo._conn
        if not conn:
            return 0.0

        # Get novelty stats from most recent snapshot
        cursor = conn.execute(
            """
            SELECT total_laws_in_window, syntactically_novel_count
            FROM novelty_snapshots
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if not row or row[0] == 0:
            return 0.0

        total, novel = row[0], row[1]
        # Redundancy = 1 - novelty
        return 1.0 - (novel / total)

    def _count_theorems(self) -> int:
        """Count total theorems."""
        conn = self.repo._conn
        if not conn:
            return 0

        cursor = conn.execute("SELECT COUNT(*) FROM theorems")
        row = cursor.fetchone()
        return row[0] if row else 0

    def _count_theorems_by_status(self, status: str) -> int:
        """Count theorems with a specific status."""
        conn = self.repo._conn
        if not conn:
            return 0

        cursor = conn.execute(
            "SELECT COUNT(*) FROM theorems WHERE status = ?",
            (status,),
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def _compute_theorem_stability(self) -> float:
        """Compute theorem set stability.

        For now, returns 1.0 if we have theorems, 0.0 otherwise.
        Will be enhanced when theorem tracking is more sophisticated.
        """
        return 1.0 if self._count_theorems() > 0 else 0.0

    def _compute_template_coverage(self) -> float:
        """Compute coverage of law templates.

        Returns fraction of templates that have at least one PASS law.
        """
        conn = self.repo._conn
        if not conn:
            return 0.0

        # Get templates with PASS laws
        cursor = conn.execute(
            """
            SELECT COUNT(DISTINCT l.template)
            FROM laws l
            JOIN evaluations e ON l.law_id = e.law_id
            WHERE e.status = 'PASS'
            """
        )
        row = cursor.fetchone()
        covered = row[0] if row else 0

        # Known templates (from schema)
        known_templates = 7  # invariant, monotone, implication_step, etc.

        return covered / known_templates if known_templates > 0 else 0.0
