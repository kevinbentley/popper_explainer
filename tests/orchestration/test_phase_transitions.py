"""Tests for phase transition logic.

Tests the two-way gate between Discovery and Theorem phases,
including hysteresis and refinement request handling.
"""

import json
import pytest
import tempfile
from pathlib import Path

from src.db.repo import Repository
from src.orchestration.control_block import (
    ControlBlock,
    EvidenceReference,
    PhaseRecommendation,
    PhaseRequest,
    StopReason,
)
from src.orchestration.phases import OrchestratorConfig, Phase
from src.orchestration.readiness import ReadinessMetrics
from src.orchestration.transitions import TransitionPolicy, TransitionDecision


class TestTransitionPolicy:
    """Tests for TransitionPolicy hysteresis and rules."""

    def test_advance_requires_consecutive_rounds(self):
        """Verify that advancing requires N consecutive rounds above threshold."""
        policy = TransitionPolicy()
        config = policy.config

        # Record some readiness scores, not all above threshold
        policy.record_readiness(Phase.DISCOVERY, 90.0)
        policy.record_readiness(Phase.DISCOVERY, 80.0)  # Below threshold
        policy.record_readiness(Phase.DISCOVERY, 90.0)

        # Create metrics that produce combined score > 85% so threshold check passes
        # Formula: 0.25*s_pass + 0.20*s_stability + 0.20*(1-s_novel_cex) + 0.15*s_harness_health + 0.20*s_redundancy
        # With these values: 0.25*0.9 + 0.20*0.9 + 0.20*0.95 + 0.15*1.0 + 0.20*0.8 = 0.905 (90.5%)
        readiness = ReadinessMetrics(
            s_pass=0.9,
            s_stability=0.9,
            s_novel_cex=0.05,
            s_harness_health=1.0,
            s_redundancy=0.8,
        )
        readiness.compute_discovery_readiness()

        control_block = ControlBlock(
            readiness_score_suggestion=90,
            readiness_justification="Test",
            phase_recommendation=PhaseRecommendation.ADVANCE,
            stop_reason=StopReason.SATURATION,
        )

        # Should NOT transition - not enough consecutive rounds above threshold
        decision = policy.should_transition(Phase.DISCOVERY, readiness, control_block)
        assert not decision.should_transition
        assert "consecutive" in decision.reason.lower()

    def test_advance_with_sustained_readiness(self):
        """Verify advance works with sustained high readiness."""
        policy = TransitionPolicy()
        config = policy.config

        # Record enough consecutive high scores
        for _ in range(config.consecutive_rounds_for_advance):
            policy.record_readiness(Phase.DISCOVERY, 90.0)

        # Create high readiness metrics
        readiness = ReadinessMetrics(
            s_pass=0.9,
            s_stability=0.9,
            s_novel_cex=0.05,
            s_harness_health=1.0,
            s_redundancy=0.8,
        )
        readiness.compute_discovery_readiness()

        control_block = ControlBlock(
            readiness_score_suggestion=90,
            readiness_justification="Test",
            phase_recommendation=PhaseRecommendation.ADVANCE,
            stop_reason=StopReason.SATURATION,
        )

        # Should transition
        decision = policy.should_transition(Phase.DISCOVERY, readiness, control_block)
        assert decision.should_transition
        assert decision.target_phase == Phase.THEOREM
        assert decision.trigger == "readiness_threshold"

    def test_retreat_requires_requests(self):
        """Verify retreat requires refinement requests."""
        policy = TransitionPolicy()

        readiness = ReadinessMetrics()
        readiness.compute_theorem_readiness()

        # Control block recommending retreat WITHOUT requests
        control_block = ControlBlock(
            readiness_score_suggestion=40,
            readiness_justification="Gaps found",
            phase_recommendation=PhaseRecommendation.RETREAT,
            stop_reason=StopReason.GAPS_IDENTIFIED,
            requests=[],  # No requests!
        )

        decision = policy.should_transition(Phase.THEOREM, readiness, control_block)
        assert not decision.should_transition
        assert "requests" in decision.reason.lower()

    def test_retreat_with_requests(self):
        """Verify retreat works when refinement requests are provided."""
        policy = TransitionPolicy()

        readiness = ReadinessMetrics()
        readiness.compute_theorem_readiness()

        # Control block recommending retreat WITH requests
        control_block = ControlBlock(
            readiness_score_suggestion=40,
            readiness_justification="Gaps found",
            phase_recommendation=PhaseRecommendation.RETREAT,
            stop_reason=StopReason.GAPS_IDENTIFIED,
            requests=[
                PhaseRequest(
                    request_type="explore_template",
                    target_id="thm_001",
                    description="Need laws for collision handling",
                    priority="high",
                )
            ],
        )

        decision = policy.should_transition(Phase.THEOREM, readiness, control_block)
        assert decision.should_transition
        assert decision.target_phase == Phase.DISCOVERY
        assert decision.trigger == "llm_recommendation"

    def test_max_iterations_forces_advance(self):
        """Verify max iterations forces phase advancement."""
        config = OrchestratorConfig(
            max_phase_iterations={Phase.DISCOVERY: 5},
        )
        policy = TransitionPolicy(config=config)

        # Simulate hitting max iterations
        for _ in range(6):
            policy.record_iteration(Phase.DISCOVERY)

        readiness = ReadinessMetrics()
        readiness.compute_discovery_readiness()

        control_block = ControlBlock(
            readiness_score_suggestion=50,
            readiness_justification="Still learning",
            phase_recommendation=PhaseRecommendation.STAY,
            stop_reason=StopReason.CONTINUING,
        )

        decision = policy.should_transition(Phase.DISCOVERY, readiness, control_block)
        assert decision.should_transition
        assert decision.target_phase == Phase.THEOREM
        assert decision.trigger == "max_iterations"

    def test_harness_health_blocks_advance(self):
        """Verify poor harness health blocks advancement."""
        policy = TransitionPolicy()

        # Record high readiness
        for _ in range(5):
            policy.record_readiness(Phase.DISCOVERY, 95.0)

        # Create metrics that produce combined score > 85% despite poor health
        # Formula: 0.25*s_pass + 0.20*s_stability + 0.20*(1-s_novel_cex) + 0.15*s_harness_health + 0.20*s_redundancy
        # With these values: 0.25*1.0 + 0.20*1.0 + 0.20*1.0 + 0.15*0.5 + 0.20*0.8 = 0.885 (88.5%)
        # The combined score passes threshold, but harness_health check should fail
        readiness = ReadinessMetrics(
            s_pass=1.0,
            s_stability=1.0,
            s_novel_cex=0.0,
            s_harness_health=0.5,  # Poor health!
            s_redundancy=0.8,
        )
        readiness.compute_discovery_readiness()

        control_block = ControlBlock(
            readiness_score_suggestion=90,
            readiness_justification="Ready but with issues",
            phase_recommendation=PhaseRecommendation.ADVANCE,
            stop_reason=StopReason.SATURATION,
        )

        decision = policy.should_transition(Phase.DISCOVERY, readiness, control_block)
        assert not decision.should_transition
        assert "health" in decision.reason.lower()


class TestRefinementTargets:
    """Tests for refinement target generation from theorems."""

    def test_missing_observable_creates_target(self):
        """Verify missing observables create refinement targets."""
        from src.orchestration.handlers.theorem_handler import TheoremPhaseHandler
        from src.theorem.models import (
            Theorem,
            TheoremStatus,
            LawSupport,
            TypedMissingStructure,
            MissingStructureType,
        )

        # Create theorem with missing observable
        theorem = Theorem(
            theorem_id="thm_001",
            name="Collision Theorem",
            status=TheoremStatus.CONDITIONAL,
            claim="Collisions cause X to appear",
            support=[LawSupport(law_id="law_001", role="confirms")],
            typed_missing_structure=[
                TypedMissingStructure(
                    type=MissingStructureType.DEFINITION_MISSING,
                    target="IncomingCollisions observable",
                )
            ],
        )

        # Use handler's private method to generate targets
        handler = TheoremPhaseHandler.__new__(TheoremPhaseHandler)
        targets = handler._identify_refinement_targets([theorem])

        assert len(targets) >= 1
        assert any(t.target_type == "missing_observable" for t in targets)

    def test_conjectural_theorem_creates_falsification_target(self):
        """Verify conjectural theorems create falsification targets."""
        from src.orchestration.handlers.theorem_handler import TheoremPhaseHandler
        from src.theorem.models import Theorem, TheoremStatus, LawSupport

        theorem = Theorem(
            theorem_id="thm_002",
            name="Speculative Theorem",
            status=TheoremStatus.CONJECTURAL,
            claim="Maybe particles always bounce",
            support=[LawSupport(law_id="law_001", role="confirms")],
        )

        handler = TheoremPhaseHandler.__new__(TheoremPhaseHandler)
        targets = handler._identify_refinement_targets([theorem])

        assert len(targets) >= 1
        assert any(t.target_type == "falsify_conjecture" for t in targets)


class TestReadinessMetrics:
    """Tests for readiness metric computation."""

    def test_discovery_readiness_high_when_saturated(self):
        """Verify high readiness when discovery is saturated."""
        metrics = ReadinessMetrics(
            s_pass=0.8,
            s_stability=0.9,
            s_novel_cex=0.05,  # Low = saturated
            s_harness_health=1.0,
            s_redundancy=0.7,
        )

        score = metrics.compute_discovery_readiness()

        # Score should be high because novel_cex is inverted (low = good)
        assert score > 70

    def test_discovery_readiness_low_when_learning(self):
        """Verify low readiness when still learning."""
        metrics = ReadinessMetrics(
            s_pass=0.3,
            s_stability=0.5,
            s_novel_cex=0.8,  # High = still learning
            s_harness_health=1.0,
            s_redundancy=0.2,
        )

        score = metrics.compute_discovery_readiness()

        # Score should be lower
        assert score < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
