"""Transition policy with hysteresis.

Manages phase transitions based on objective readiness metrics and
LLM recommendations, using hysteresis to prevent oscillation.

Key principle: Transitions require SUSTAINED high readiness over
multiple consecutive rounds, not just a single high reading.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.orchestration.control_block import ControlBlock, PhaseRecommendation, StopReason
from src.orchestration.phases import Phase, OrchestratorConfig
from src.orchestration.readiness import ReadinessMetrics


@dataclass
class TransitionRule:
    """A single transition rule with hysteresis requirements.

    Attributes:
        from_phase: Source phase
        to_phase: Target phase
        min_readiness: Minimum readiness score (0-100) to trigger
        consecutive_rounds: Number of consecutive rounds above threshold required
        harness_health_required: Whether s_harness_health must be 1.0
        max_novel_cex_rate: Maximum novel counterexample rate to allow advance
        can_retreat: Whether retreat is allowed under this rule
        retreat_requires_requests: Whether retreat requires specific requests
    """

    from_phase: Phase
    to_phase: Phase
    min_readiness: float = 85.0
    consecutive_rounds: int = 3
    harness_health_required: bool = True
    max_novel_cex_rate: float = 0.1
    can_retreat: bool = False
    retreat_requires_requests: bool = True


@dataclass
class TransitionDecision:
    """Result of transition policy evaluation.

    Attributes:
        should_transition: Whether to execute the transition
        target_phase: Target phase if transitioning
        trigger: What triggered the transition
        reason: Human-readable explanation
        evidence: Supporting metrics/data
    """

    should_transition: bool
    target_phase: Phase | None = None
    trigger: str = ""  # 'readiness_threshold', 'llm_recommendation', 'plateau_escape', 'manual', 'max_iterations'
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


class TransitionPolicy:
    """Manages phase transitions with hysteresis.

    Hysteresis prevents oscillation by requiring:
    1. Readiness above threshold for N consecutive rounds (not just once)
    2. Additional conditions (harness health, novelty rate)
    3. LLM recommendation alignment (for tie-breaking)
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        rules: list[TransitionRule] | None = None,
    ):
        """Initialize transition policy.

        Args:
            config: Orchestrator configuration
            rules: Custom transition rules (defaults to standard rules)
        """
        self.config = config or OrchestratorConfig()
        self.rules = rules or self._default_rules()

        # Track readiness history per phase for hysteresis
        self._readiness_history: dict[Phase, list[float]] = defaultdict(list)

        # Track phase iteration counts
        self._phase_iterations: dict[Phase, int] = defaultdict(int)

    def _default_rules(self) -> list[TransitionRule]:
        """Create default transition rules."""
        return [
            # Forward transitions (ADVANCE)
            TransitionRule(
                from_phase=Phase.DISCOVERY,
                to_phase=Phase.THEOREM,
                min_readiness=self.config.discovery_to_theorem_threshold,
                consecutive_rounds=self.config.consecutive_rounds_for_advance,
                harness_health_required=True,
                max_novel_cex_rate=0.1,
            ),
            TransitionRule(
                from_phase=Phase.THEOREM,
                to_phase=Phase.EXPLANATION,
                min_readiness=self.config.theorem_to_explanation_threshold,
                consecutive_rounds=self.config.consecutive_rounds_for_advance,
                harness_health_required=True,
            ),
            TransitionRule(
                from_phase=Phase.EXPLANATION,
                to_phase=Phase.PREDICTION,
                min_readiness=self.config.explanation_to_prediction_threshold,
                consecutive_rounds=2,  # Faster for explanation
            ),
            TransitionRule(
                from_phase=Phase.PREDICTION,
                to_phase=Phase.FINALIZE,
                min_readiness=self.config.prediction_to_finalize_threshold,
                consecutive_rounds=2,
            ),
            # Backward transitions (RETREAT)
            TransitionRule(
                from_phase=Phase.THEOREM,
                to_phase=Phase.DISCOVERY,
                min_readiness=0.0,  # No min - triggered by needs_refinement
                consecutive_rounds=1,
                can_retreat=True,
                retreat_requires_requests=True,
            ),
            TransitionRule(
                from_phase=Phase.EXPLANATION,
                to_phase=Phase.THEOREM,
                min_readiness=0.0,
                consecutive_rounds=1,
                can_retreat=True,
                retreat_requires_requests=True,
            ),
            TransitionRule(
                from_phase=Phase.PREDICTION,
                to_phase=Phase.EXPLANATION,
                min_readiness=0.0,
                consecutive_rounds=1,
                can_retreat=True,
                retreat_requires_requests=True,
            ),
        ]

    def record_readiness(self, phase: Phase, score: float) -> None:
        """Record a readiness score for hysteresis tracking.

        Args:
            phase: Current phase
            score: Readiness score (0-100)
        """
        history = self._readiness_history[phase]
        history.append(score)

        # Keep only recent history (2x max consecutive rounds)
        max_history = self.config.consecutive_rounds_for_advance * 2
        if len(history) > max_history:
            self._readiness_history[phase] = history[-max_history:]

    def record_iteration(self, phase: Phase) -> None:
        """Record an iteration for a phase.

        Args:
            phase: Current phase
        """
        self._phase_iterations[phase] += 1

    def get_phase_iterations(self, phase: Phase) -> int:
        """Get number of iterations in a phase.

        Args:
            phase: Phase to check

        Returns:
            Number of iterations
        """
        return self._phase_iterations.get(phase, 0)

    def should_transition(
        self,
        current_phase: Phase,
        readiness: ReadinessMetrics,
        control_block: ControlBlock,
    ) -> TransitionDecision:
        """Determine if a transition should occur.

        Decision process:
        1. Check for max iterations (forced transition)
        2. Check for RETREAT recommendation with requests
        3. Check for ADVANCE with sustained readiness
        4. Default to STAY

        Args:
            current_phase: Current phase
            readiness: Objective readiness metrics
            control_block: LLM control block output

        Returns:
            TransitionDecision with decision and reasoning
        """
        # Record this readiness score
        score = readiness.combined_score * 100  # Convert to 0-100
        self.record_readiness(current_phase, score)

        # Check max iterations
        max_iter = self.config.max_phase_iterations.get(current_phase, 100)
        phase_iters = self.get_phase_iterations(current_phase)
        if phase_iters >= max_iter:
            next_phase = current_phase.next_phase()
            if next_phase:
                return TransitionDecision(
                    should_transition=True,
                    target_phase=next_phase,
                    trigger="max_iterations",
                    reason=f"Reached max {max_iter} iterations for {current_phase.value}",
                    evidence={
                        "phase_iterations": phase_iters,
                        "max_allowed": max_iter,
                    },
                )

        # Check for RETREAT
        if control_block.phase_recommendation == PhaseRecommendation.RETREAT:
            retreat_decision = self._check_retreat(
                current_phase, readiness, control_block
            )
            # Return the decision even if not transitioning (to preserve reason)
            if retreat_decision.should_transition or retreat_decision.reason:
                return retreat_decision

        # Check for ADVANCE
        if control_block.phase_recommendation == PhaseRecommendation.ADVANCE:
            advance_decision = self._check_advance(
                current_phase, readiness, control_block
            )
            # Return the decision even if not transitioning (to preserve reason)
            if advance_decision.should_transition or advance_decision.reason:
                return advance_decision

        # Even without LLM recommendation, check if objective metrics warrant advance
        objective_decision = self._check_objective_advance(current_phase, readiness)
        if objective_decision.should_transition:
            return objective_decision

        # Default: stay in current phase
        return TransitionDecision(
            should_transition=False,
            target_phase=None,
            trigger="",
            reason="Continuing in current phase",
            evidence={
                "readiness_score": score,
                "phase_iterations": phase_iters,
                "llm_recommendation": control_block.phase_recommendation.value,
            },
        )

    def _check_retreat(
        self,
        current_phase: Phase,
        readiness: ReadinessMetrics,
        control_block: ControlBlock,
    ) -> TransitionDecision:
        """Check if retreat conditions are met.

        Args:
            current_phase: Current phase
            readiness: Readiness metrics
            control_block: Control block with requests

        Returns:
            TransitionDecision
        """
        # Find applicable retreat rule
        rule = self._find_rule(current_phase, is_retreat=True)
        if not rule:
            return TransitionDecision(
                should_transition=False,
                reason=f"No retreat rule from {current_phase.value}",
            )

        # Check if requests are provided (required for retreat)
        if rule.retreat_requires_requests and not control_block.requests:
            return TransitionDecision(
                should_transition=False,
                reason="RETREAT requires specific requests but none provided",
                evidence={"llm_recommendation": "retreat", "requests_count": 0},
            )

        # Retreat is allowed
        return TransitionDecision(
            should_transition=True,
            target_phase=rule.to_phase,
            trigger="llm_recommendation",
            reason=f"LLM recommends retreat with {len(control_block.requests)} requests",
            evidence={
                "requests": [r.to_dict() for r in control_block.requests],
                "stop_reason": control_block.stop_reason.value,
            },
        )

    def _check_advance(
        self,
        current_phase: Phase,
        readiness: ReadinessMetrics,
        control_block: ControlBlock,
    ) -> TransitionDecision:
        """Check if advance conditions are met.

        Args:
            current_phase: Current phase
            readiness: Readiness metrics
            control_block: Control block

        Returns:
            TransitionDecision
        """
        # Find applicable advance rule
        rule = self._find_rule(current_phase, is_retreat=False)
        if not rule:
            return TransitionDecision(
                should_transition=False,
                reason=f"No advance rule from {current_phase.value}",
            )

        score = readiness.combined_score * 100

        # Check for saturation override - high redundancy is a strong signal
        # that discovery has exhausted its search space
        effective_threshold = rule.min_readiness
        is_saturation_override = False

        if control_block.stop_reason in (StopReason.HIGH_REDUNDANCY, StopReason.SATURATION):
            # Allow transition at lower threshold when saturated
            # Requires at least 60% readiness and high redundancy signal
            saturation_threshold = max(60.0, rule.min_readiness - 25.0)
            if score >= saturation_threshold and readiness.s_redundancy >= 0.5:
                effective_threshold = saturation_threshold
                is_saturation_override = True

        # Check readiness threshold
        if score < effective_threshold:
            return TransitionDecision(
                should_transition=False,
                reason=f"Readiness {score:.1f} below threshold {effective_threshold}",
                evidence={
                    "readiness": score,
                    "threshold": effective_threshold,
                    "base_threshold": rule.min_readiness,
                    "saturation_override": is_saturation_override,
                },
            )

        # Check hysteresis (consecutive rounds above threshold)
        # Use reduced rounds for saturation override (discovery exhausted faster)
        required_rounds = 2 if is_saturation_override else rule.consecutive_rounds
        if not self._check_hysteresis(current_phase, effective_threshold, required_rounds):
            history = self._readiness_history.get(current_phase, [])
            above_threshold = sum(1 for s in history[-required_rounds:] if s >= effective_threshold)
            return TransitionDecision(
                should_transition=False,
                reason=f"Only {above_threshold}/{required_rounds} consecutive rounds above threshold",
                evidence={
                    "readiness_history": history[-required_rounds:],
                    "required_rounds": required_rounds,
                    "saturation_override": is_saturation_override,
                },
            )

        # Check harness health
        if rule.harness_health_required and readiness.s_harness_health < 1.0:
            return TransitionDecision(
                should_transition=False,
                reason=f"Harness health {readiness.s_harness_health:.2f} < 1.0",
                evidence={"harness_health": readiness.s_harness_health},
            )

        # Check novel counterexample rate (for discovery)
        # Skip this check during saturation override - when proposal redundancy is high,
        # we can't make progress even if we're still finding counterexamples
        if not is_saturation_override and readiness.s_novel_cex > rule.max_novel_cex_rate:
            return TransitionDecision(
                should_transition=False,
                reason=f"Novel CEX rate {readiness.s_novel_cex:.2f} > {rule.max_novel_cex_rate}",
                evidence={"novel_cex_rate": readiness.s_novel_cex},
            )

        # All conditions met - advance
        trigger = "saturation_override" if is_saturation_override else "readiness_threshold"
        reason = (
            f"Saturation detected (redundancy={readiness.s_redundancy:.0%}), "
            f"readiness {score:.1f} above reduced threshold {effective_threshold}"
            if is_saturation_override
            else f"Readiness {score:.1f} sustained above {effective_threshold} for {required_rounds} rounds"
        )
        return TransitionDecision(
            should_transition=True,
            target_phase=rule.to_phase,
            trigger=trigger,
            reason=reason,
            evidence={
                "readiness": score,
                "threshold": effective_threshold,
                "base_threshold": rule.min_readiness,
                "consecutive_rounds": required_rounds,
                "llm_alignment": True,
                "saturation_override": is_saturation_override,
                "redundancy": readiness.s_redundancy,
            },
        )

    def _check_objective_advance(
        self,
        current_phase: Phase,
        readiness: ReadinessMetrics,
    ) -> TransitionDecision:
        """Check if objective metrics alone warrant advance.

        This handles cases where the LLM doesn't recommend ADVANCE
        but objective metrics are strong enough.

        Args:
            current_phase: Current phase
            readiness: Readiness metrics

        Returns:
            TransitionDecision
        """
        rule = self._find_rule(current_phase, is_retreat=False)
        if not rule:
            return TransitionDecision(should_transition=False)

        score = readiness.combined_score * 100

        # Require HIGHER threshold for objective-only advance (no LLM alignment)
        objective_threshold = min(rule.min_readiness + 10, 95.0)

        if score < objective_threshold:
            return TransitionDecision(should_transition=False)

        # Need more consecutive rounds for objective-only
        objective_rounds = rule.consecutive_rounds + 2

        if not self._check_hysteresis(current_phase, objective_threshold, objective_rounds):
            return TransitionDecision(should_transition=False)

        # Check other conditions
        if rule.harness_health_required and readiness.s_harness_health < 1.0:
            return TransitionDecision(should_transition=False)

        if readiness.s_novel_cex > rule.max_novel_cex_rate:
            return TransitionDecision(should_transition=False)

        return TransitionDecision(
            should_transition=True,
            target_phase=rule.to_phase,
            trigger="readiness_threshold",
            reason=f"Objective readiness {score:.1f} sustained above {objective_threshold} (no LLM alignment)",
            evidence={
                "readiness": score,
                "threshold": objective_threshold,
                "consecutive_rounds": objective_rounds,
                "llm_alignment": False,
            },
        )

    def _find_rule(
        self,
        from_phase: Phase,
        is_retreat: bool,
    ) -> TransitionRule | None:
        """Find applicable transition rule.

        Args:
            from_phase: Source phase
            is_retreat: Whether looking for retreat rule

        Returns:
            Matching rule or None
        """
        for rule in self.rules:
            if rule.from_phase != from_phase:
                continue

            if is_retreat and rule.can_retreat:
                return rule
            elif not is_retreat and not rule.can_retreat:
                return rule

        return None

    def _check_hysteresis(
        self,
        phase: Phase,
        threshold: float,
        required_rounds: int,
    ) -> bool:
        """Check if readiness has been above threshold for N rounds.

        Args:
            phase: Phase to check
            threshold: Minimum readiness threshold
            required_rounds: Number of consecutive rounds required

        Returns:
            True if hysteresis condition is met
        """
        history = self._readiness_history.get(phase, [])

        if len(history) < required_rounds:
            return False

        recent = history[-required_rounds:]
        return all(score >= threshold for score in recent)

    def reset_phase_history(self, phase: Phase) -> None:
        """Reset history for a phase (call after transition).

        Args:
            phase: Phase to reset
        """
        self._readiness_history[phase] = []
        # Don't reset iteration count - that's cumulative
