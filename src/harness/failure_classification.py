"""Failure classification for convergence detection.

Classifies failed laws into four types:
- Type A: Falsified by a known counterexample class (expected failure)
- Type B: Falsified by a new counterexample class (learning opportunity)
- Type C: Invalid law form or ambiguous observable (process/harness issue)
- Type D: Harness error (show-stopper)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.harness.verdict import (
    Counterexample,
    FailureType,
    LawVerdict,
    ReasonCode,
)


class FailureClass(str, Enum):
    """High-level failure classification for convergence analysis."""

    # Type A: Expected failure - counterexample matches known class
    TYPE_A_KNOWN_COUNTEREXAMPLE = "type_a_known_counterexample"

    # Type B: Learning failure - counterexample is novel
    TYPE_B_NOVEL_COUNTEREXAMPLE = "type_b_novel_counterexample"

    # Type C: Process issue - invalid law form or ambiguous definition
    TYPE_C_PROCESS_ISSUE = "type_c_process_issue"

    # Type D: Harness error - infrastructure failure
    TYPE_D_HARNESS_ERROR = "type_d_harness_error"


class CounterexampleClassId(str, Enum):
    """Known counterexample class identifiers.

    Each represents a structural pattern in counterexamples that
    characterizes a failure mode in the particle universe.
    """

    # Collision-related classes
    CONVERGING_PAIR = "converging_pair"  # >.<  pattern causing collision
    X_EMISSION = "x_emission"  # X emitting particles
    X_SELF_COLLISION = "x_self_collision"  # X in L=1 or L=2 wrapping
    MULTI_COLLISION = "multi_collision"  # Multiple simultaneous collisions
    COLLISION_CHAIN = "collision_chain"  # Collision products colliding again

    # Conservation/particle count related
    PARTICLE_CREATION = "particle_creation"  # Unexpected particle count increase
    PARTICLE_DESTRUCTION = "particle_destruction"  # Unexpected particle loss
    DIRECTION_CHANGE = "direction_change"  # Particle direction unexpectedly changed

    # Boundary/wrapping related
    BOUNDARY_WRAP = "boundary_wrap"  # Failure involving periodic boundary
    EDGE_CASE = "edge_case"  # L=1, L=2, or single particle edge cases

    # Density related
    HIGH_DENSITY = "high_density"  # Dense state (>50% occupied)
    SPARSE_STATE = "sparse_state"  # Very sparse state

    # Temporal patterns
    LATE_VIOLATION = "late_violation"  # Violation at t > 10
    IMMEDIATE_VIOLATION = "immediate_violation"  # Violation at t=0 or t=1

    # Observable-specific
    OBSERVABLE_BOUNDARY = "observable_boundary"  # Failure at observable min/max
    POSITION_CONFLICT = "position_conflict"  # leftmost/rightmost issues

    # Unknown - doesn't match any known class
    UNKNOWN = "unknown"


@dataclass
class CounterexampleClassification:
    """Classification of a counterexample.

    Attributes:
        class_id: The identified counterexample class
        confidence: Classification confidence (0.0 to 1.0)
        features: Extracted structural features
        reasoning: Explanation of classification
    """

    class_id: CounterexampleClassId
    confidence: float
    features: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id.value,
            "confidence": self.confidence,
            "features": self.features,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CounterexampleClassification":
        return cls(
            class_id=CounterexampleClassId(data["class_id"]),
            confidence=data["confidence"],
            features=data.get("features", {}),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class FailureClassificationResult:
    """Complete failure classification for a verdict.

    Attributes:
        failure_class: High-level Type A/B/C/D classification
        counterexample_class: Counterexample class if applicable
        is_known_class: Whether this matches a previously seen class
        reason: Explanation of classification
        actionable: Whether this failure suggests harness/process fixes
    """

    failure_class: FailureClass
    counterexample_class: CounterexampleClassification | None = None
    is_known_class: bool = False
    reason: str = ""
    actionable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_class": self.failure_class.value,
            "counterexample_class": (
                self.counterexample_class.to_dict()
                if self.counterexample_class
                else None
            ),
            "is_known_class": self.is_known_class,
            "reason": self.reason,
            "actionable": self.actionable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureClassificationResult":
        ce_class = None
        if data.get("counterexample_class"):
            ce_class = CounterexampleClassification.from_dict(
                data["counterexample_class"]
            )
        return cls(
            failure_class=FailureClass(data["failure_class"]),
            counterexample_class=ce_class,
            is_known_class=data.get("is_known_class", False),
            reason=data.get("reason", ""),
            actionable=data.get("actionable", False),
        )


# Reason codes that indicate Type C (process/harness issues)
TYPE_C_REASON_CODES = {
    ReasonCode.AMBIGUOUS_CLAIM,
    ReasonCode.MISSING_OBSERVABLE,
    ReasonCode.MISSING_TRANSFORM,
    ReasonCode.MISSING_GENERATOR,
}


class FailureClassifier:
    """Classifies law failures into Type A/B/C/D.

    Maintains a registry of known counterexample classes and classifies
    new failures based on structural similarity.
    """

    def __init__(self):
        # Known counterexample classes seen so far (class_id -> count)
        self._known_classes: dict[CounterexampleClassId, int] = {}

    def classify(self, verdict: LawVerdict) -> FailureClassificationResult:
        """Classify a failed law verdict.

        Args:
            verdict: The law verdict to classify

        Returns:
            FailureClassificationResult with Type A/B/C/D classification
        """
        # Type D: Harness errors (infrastructure failures)
        if verdict.failure_type in (
            FailureType.INVALID_INITIAL_STATE,
            FailureType.EVALUATION_ERROR,
            FailureType.TIMEOUT,
        ):
            return FailureClassificationResult(
                failure_class=FailureClass.TYPE_D_HARNESS_ERROR,
                reason=f"Infrastructure failure: {verdict.failure_type.value}",
                actionable=True,  # Harness errors need fixing
            )

        # Type C: Process issues (invalid law form, missing capabilities)
        if verdict.status == "UNKNOWN" and verdict.reason_code in TYPE_C_REASON_CODES:
            return FailureClassificationResult(
                failure_class=FailureClass.TYPE_C_PROCESS_ISSUE,
                reason=f"Process issue: {verdict.reason_code.value}",
                actionable=True,  # May need capability or law form fixes
            )

        # Type A or B: Genuine law falsification
        if verdict.status == "FAIL" and verdict.counterexample:
            ce_class = self._classify_counterexample(verdict.counterexample)

            # Check if this class is known
            is_known = ce_class.class_id in self._known_classes

            # Register this class
            self._known_classes[ce_class.class_id] = (
                self._known_classes.get(ce_class.class_id, 0) + 1
            )

            if is_known and ce_class.class_id != CounterexampleClassId.UNKNOWN:
                # Type A: Known counterexample class
                return FailureClassificationResult(
                    failure_class=FailureClass.TYPE_A_KNOWN_COUNTEREXAMPLE,
                    counterexample_class=ce_class,
                    is_known_class=True,
                    reason=f"Matches known class: {ce_class.class_id.value}",
                    actionable=False,
                )
            else:
                # Type B: Novel counterexample class
                return FailureClassificationResult(
                    failure_class=FailureClass.TYPE_B_NOVEL_COUNTEREXAMPLE,
                    counterexample_class=ce_class,
                    is_known_class=False,
                    reason=f"Novel counterexample: {ce_class.class_id.value}",
                    actionable=False,
                )

        # Fallback for UNKNOWN without Type C reason codes
        if verdict.status == "UNKNOWN":
            return FailureClassificationResult(
                failure_class=FailureClass.TYPE_C_PROCESS_ISSUE,
                reason=f"Inconclusive: {verdict.reason_code.value if verdict.reason_code else 'unknown'}",
                actionable=False,
            )

        # PASS should not be classified as failure
        return FailureClassificationResult(
            failure_class=FailureClass.TYPE_A_KNOWN_COUNTEREXAMPLE,
            reason="Not a failure (PASS verdict)",
            actionable=False,
        )

    def _classify_counterexample(
        self, ce: Counterexample
    ) -> CounterexampleClassification:
        """Classify a counterexample into a structural class.

        Extracts features from the counterexample and matches against
        known structural patterns.
        """
        features = self._extract_features(ce)

        # Classification rules (order matters - more specific first)
        class_id, confidence, reasoning = self._match_class(features, ce)

        return CounterexampleClassification(
            class_id=class_id,
            confidence=confidence,
            features=features,
            reasoning=reasoning,
        )

    def _extract_features(self, ce: Counterexample) -> dict[str, Any]:
        """Extract structural features from a counterexample."""
        initial = ce.initial_state
        grid_length = len(initial) if initial else 0

        features = {
            "grid_length": grid_length,
            "t_fail": ce.t_fail,
            "t_max": ce.t_max,
            "initial_state": initial,
        }

        if initial:
            # Particle counts
            features["count_right"] = initial.count(">")
            features["count_left"] = initial.count("<")
            features["count_x"] = initial.count("X")
            features["count_empty"] = initial.count(".")
            features["particle_count"] = (
                features["count_right"] + features["count_left"] + features["count_x"]
            )

            # Density
            if grid_length > 0:
                features["density"] = features["particle_count"] / grid_length
            else:
                features["density"] = 0.0

            # Pattern detection
            features["has_converging_pair"] = ">.<" in initial or "<.>" in initial
            features["has_adjacent_opposite"] = "><" in initial or "<>" in initial
            features["has_x"] = "X" in initial

            # Boundary features
            features["particle_at_left_edge"] = initial[0] in "><X" if initial else False
            features["particle_at_right_edge"] = (
                initial[-1] in "><X" if initial else False
            )

        # Trajectory features
        if ce.trajectory_excerpt:
            excerpt = ce.trajectory_excerpt
            features["trajectory_length"] = len(excerpt)
            features["collision_count_in_excerpt"] = sum(
                s.count("X") for s in excerpt
            )

        return features

    def _match_class(
        self, features: dict[str, Any], ce: Counterexample
    ) -> tuple[CounterexampleClassId, float, str]:
        """Match extracted features to a counterexample class."""
        grid_length = features.get("grid_length", 0)
        t_fail = features.get("t_fail", 0)

        # Edge cases: very small grids
        if grid_length <= 2:
            if features.get("has_x"):
                return (
                    CounterexampleClassId.X_SELF_COLLISION,
                    0.9,
                    f"X in small grid (L={grid_length}) causes self-collision",
                )
            return (
                CounterexampleClassId.EDGE_CASE,
                0.8,
                f"Small grid edge case (L={grid_length})",
            )

        # Converging pair pattern
        if features.get("has_converging_pair"):
            return (
                CounterexampleClassId.CONVERGING_PAIR,
                0.95,
                "Contains converging pair pattern (>.<)",
            )

        # X emission patterns
        if features.get("has_x") and t_fail <= 2:
            return (
                CounterexampleClassId.X_EMISSION,
                0.85,
                "X emission causing early violation",
            )

        # Multi-collision
        if features.get("collision_count_in_excerpt", 0) > 1:
            return (
                CounterexampleClassId.MULTI_COLLISION,
                0.8,
                "Multiple collisions in trajectory",
            )

        # High density
        if features.get("density", 0) > 0.5:
            return (
                CounterexampleClassId.HIGH_DENSITY,
                0.7,
                f"High particle density ({features.get('density', 0):.2f})",
            )

        # Sparse state
        if features.get("density", 0) < 0.1 and grid_length > 5:
            return (
                CounterexampleClassId.SPARSE_STATE,
                0.7,
                f"Sparse state ({features.get('density', 0):.2f})",
            )

        # Late violation
        if t_fail > 10:
            return (
                CounterexampleClassId.LATE_VIOLATION,
                0.6,
                f"Late violation at t={t_fail}",
            )

        # Immediate violation
        if t_fail <= 1:
            return (
                CounterexampleClassId.IMMEDIATE_VIOLATION,
                0.7,
                f"Immediate violation at t={t_fail}",
            )

        # Boundary effects
        if features.get("particle_at_left_edge") or features.get(
            "particle_at_right_edge"
        ):
            return (
                CounterexampleClassId.BOUNDARY_WRAP,
                0.6,
                "Particle at boundary edge",
            )

        # Default: unknown class
        return (
            CounterexampleClassId.UNKNOWN,
            0.5,
            "No matching known pattern",
        )

    def register_known_class(self, class_id: CounterexampleClassId) -> None:
        """Pre-register a known counterexample class.

        Use this to seed the classifier with known failure patterns
        from previous runs or domain knowledge.
        """
        if class_id not in self._known_classes:
            self._known_classes[class_id] = 0

    def get_known_classes(self) -> dict[CounterexampleClassId, int]:
        """Get the current known class registry with counts."""
        return dict(self._known_classes)

    def reset(self) -> None:
        """Reset the known class registry."""
        self._known_classes.clear()


# Module-level classifier instance
_classifier = FailureClassifier()


def classify_failure(verdict: LawVerdict) -> FailureClassificationResult:
    """Classify a law verdict failure.

    Convenience function using module-level classifier.
    """
    return _classifier.classify(verdict)


def get_classifier() -> FailureClassifier:
    """Get the module-level classifier instance."""
    return _classifier
