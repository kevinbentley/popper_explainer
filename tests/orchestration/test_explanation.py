"""Tests for the explanation synthesis subsystem.

Tests cover:
- Explanation models
- Mechanism-based predictor
- Explanation generator
- ExplanationPhaseHandler behavior
"""

import pytest

from src.orchestration.explanation.models import (
    Criticism,
    Explanation,
    ExplanationStatus,
    Mechanism,
    MechanismRule,
    MechanismType,
    OpenQuestion,
)
from src.orchestration.explanation.predictor import (
    MechanismBasedPredictor,
    create_predictor,
)
from src.universe.types import Config


class TestExplanationModels:
    """Tests for explanation domain models."""

    def test_mechanism_rule_serialization(self):
        """Test MechanismRule to_dict and from_dict."""
        rule = MechanismRule(
            rule_id="test_rule",
            rule_type=MechanismType.MOVEMENT,
            condition="Cell contains >",
            effect="Particle moves right",
            priority=0,
            supporting_laws=["law_001", "law_002"],
        )

        data = rule.to_dict()
        restored = MechanismRule.from_dict(data)

        assert restored.rule_id == rule.rule_id
        assert restored.rule_type == rule.rule_type
        assert restored.condition == rule.condition
        assert restored.effect == rule.effect

    def test_mechanism_get_rules_by_type(self):
        """Test filtering rules by type."""
        rules = [
            MechanismRule("r1", MechanismType.MOVEMENT, "cond1", "eff1"),
            MechanismRule("r2", MechanismType.INTERACTION, "cond2", "eff2"),
            MechanismRule("r3", MechanismType.MOVEMENT, "cond3", "eff3"),
        ]
        mechanism = Mechanism(rules=rules, description="Test")

        movement_rules = mechanism.get_rules_by_type(MechanismType.MOVEMENT)
        assert len(movement_rules) == 2
        assert all(r.rule_type == MechanismType.MOVEMENT for r in movement_rules)

    def test_explanation_fingerprint(self):
        """Test explanation fingerprint generation."""
        mechanism = Mechanism(
            rules=[MechanismRule("r1", MechanismType.MOVEMENT, "cond", "eff")],
            description="Test",
        )
        explanation = Explanation(
            explanation_id="exp_001",
            hypothesis_text="Test hypothesis",
            mechanism=mechanism,
        )

        fingerprint = explanation.fingerprint
        assert len(fingerprint) == 16
        assert fingerprint.isalnum()

    def test_explanation_has_critical_issues(self):
        """Test critical issue detection."""
        explanation = Explanation(
            explanation_id="exp_001",
            hypothesis_text="Test",
            mechanism=Mechanism(rules=[], description=""),
            criticisms=[
                Criticism("Minor issue", "minor", "llm"),
                Criticism("Critical issue", "critical", "prediction"),
            ],
        )

        assert explanation.has_critical_issues() is True

    def test_explanation_has_high_priority_questions(self):
        """Test high-priority question detection."""
        explanation = Explanation(
            explanation_id="exp_001",
            hypothesis_text="Test",
            mechanism=Mechanism(rules=[], description=""),
            open_questions=[
                OpenQuestion("Low priority?", "mechanism", "low"),
                OpenQuestion("High priority?", "mechanism", "high"),
            ],
        )

        assert explanation.has_high_priority_questions() is True


class TestMechanismBasedPredictor:
    """Tests for the mechanism-based predictor."""

    def test_predict_empty_state(self):
        """Test prediction on empty state."""
        explanation = _create_test_explanation()
        predictor = MechanismBasedPredictor(explanation)

        result = predictor.predict("........", 1)
        assert result == "........"

    def test_predict_single_right_particle(self):
        """Test single right-moving particle."""
        explanation = _create_test_explanation()
        predictor = MechanismBasedPredictor(explanation)

        result = predictor.predict(">.......", 1)
        assert result == ".>......"

    def test_predict_single_left_particle(self):
        """Test single left-moving particle."""
        explanation = _create_test_explanation()
        predictor = MechanismBasedPredictor(explanation)

        result = predictor.predict(".......<", 1)
        assert result == "......<."

    def test_predict_collision_formation(self):
        """Test collision formation."""
        explanation = _create_test_explanation()
        predictor = MechanismBasedPredictor(explanation)

        # >< - particles swap positions (no collision since they pass through)
        result = predictor.predict("><......", 1)
        assert result == "<>......"

        # >..< - particles approach, will collide at position 2
        result = predictor.predict(">...<...", 2)
        assert result == "..X.....", f"Expected collision, got {result}"

    def test_predict_collision_resolution(self):
        """Test collision resolution."""
        explanation = _create_test_explanation()
        predictor = MechanismBasedPredictor(explanation)

        # X resolves: > exits right (pos 1), < exits left (wraps to pos 7)
        result = predictor.predict("X.......", 1)
        assert result == ".>.....<"

    def test_predict_boundary_wrap_right(self):
        """Test boundary wrapping for right particle."""
        explanation = _create_test_explanation()
        predictor = MechanismBasedPredictor(explanation)

        result = predictor.predict(".......>", 1)
        assert result == ">......."

    def test_predict_boundary_wrap_left(self):
        """Test boundary wrapping for left particle."""
        explanation = _create_test_explanation()
        predictor = MechanismBasedPredictor(explanation)

        result = predictor.predict("<.......", 1)
        assert result == ".......<"

    def test_predict_multi_step(self):
        """Test multi-step prediction."""
        explanation = _create_test_explanation()
        predictor = MechanismBasedPredictor(explanation)

        # Right particle moves 3 steps
        result = predictor.predict(">.......", 3)
        assert result == "...>....", f"Expected '...>....', got '{result}'"

    def test_predict_matches_simulator(self):
        """Test that predictor matches the actual simulator."""
        from src.universe.simulator import run as sim_run

        explanation = _create_test_explanation()
        predictor = MechanismBasedPredictor(explanation)

        test_states = [
            "........",
            ">.......",
            ".......<",
            "><......",
            "X.......",
            ".>.<....",
            ">...<...",
        ]

        for initial_state in test_states:
            n = len(initial_state)
            for horizon in [1, 2, 3]:
                expected = sim_run(initial_state, horizon, Config(grid_length=n))[-1]
                predicted = predictor.predict(initial_state, horizon)
                assert predicted == expected, (
                    f"Mismatch for {initial_state} h={horizon}: "
                    f"expected {expected}, got {predicted}"
                )

    def test_predictor_function_factory(self):
        """Test create_predictor factory function."""
        explanation = _create_test_explanation()
        predictor_fn = create_predictor(explanation, mode="mechanism")

        result = predictor_fn(">.......", 1)
        assert result == ".>......"


class TestExplanationPhaseHandler:
    """Tests for the explanation phase handler."""

    def _make_context(self, run_id: str = "test_run", iteration_index: int = 0):
        """Create a minimal PhaseContext for testing."""
        from src.orchestration.phases import PhaseContext, Phase, OrchestratorConfig

        return PhaseContext(
            run_id=run_id,
            iteration_index=iteration_index,
            repo=None,  # type: ignore
            config=OrchestratorConfig(),
            current_phase=Phase.EXPLANATION,
        )

    def test_handler_without_theorems(self):
        """Test handler behavior with no theorems."""
        from src.orchestration.handlers.explanation_handler import (
            ExplanationPhaseHandler,
            ExplanationPhaseConfig,
        )
        from src.orchestration.control_block import PhaseRecommendation

        config = ExplanationPhaseConfig(min_theorems=3)
        handler = ExplanationPhaseHandler(repo=None, config=config)
        context = self._make_context()

        result = handler.execute(context)

        # Should retreat due to insufficient theorems
        assert result.control_block.phase_recommendation == PhaseRecommendation.RETREAT
        assert "Insufficient" in result.summary.get("error", "") or "theorems" in result.control_block.readiness_justification.lower()

    def test_handler_creates_predictor(self):
        """Test that handler creates a usable predictor."""
        from src.orchestration.handlers.explanation_handler import (
            ExplanationPhaseHandler,
            ExplanationPhaseConfig,
        )
        from unittest.mock import MagicMock

        config = ExplanationPhaseConfig(min_theorems=0)  # Allow 0 theorems
        handler = ExplanationPhaseHandler(repo=None, config=config)
        context = self._make_context()

        # Execute to create explanation
        handler.execute(context)

        # Should now have a predictor
        predictor_fn = handler.get_predictor()
        assert predictor_fn is not None

        # Test the predictor works
        result = predictor_fn(">.......", 1)
        assert result == ".>......"

    def test_handler_provides_explanation(self):
        """Test that handler provides current explanation."""
        from src.orchestration.handlers.explanation_handler import (
            ExplanationPhaseHandler,
            ExplanationPhaseConfig,
        )

        config = ExplanationPhaseConfig(min_theorems=0)
        handler = ExplanationPhaseHandler(repo=None, config=config)
        context = self._make_context()

        handler.execute(context)

        explanation = handler.get_current_explanation()
        assert explanation is not None
        assert explanation.mechanism is not None
        assert len(explanation.mechanism.rules) > 0


def _create_test_explanation() -> Explanation:
    """Create a standard test explanation."""
    rules = [
        MechanismRule(
            rule_id="movement_right",
            rule_type=MechanismType.MOVEMENT,
            condition="Cell contains >",
            effect="Particle moves right",
            priority=0,
        ),
        MechanismRule(
            rule_id="movement_left",
            rule_type=MechanismType.MOVEMENT,
            condition="Cell contains <",
            effect="Particle moves left",
            priority=0,
        ),
        MechanismRule(
            rule_id="collision",
            rule_type=MechanismType.INTERACTION,
            condition="> and < meet",
            effect="Form collision X",
            priority=1,
        ),
        MechanismRule(
            rule_id="resolution",
            rule_type=MechanismType.TRANSFORMATION,
            condition="Cell contains X",
            effect="X resolves",
            priority=2,
        ),
    ]

    mechanism = Mechanism(
        rules=rules,
        description="Test mechanism",
        assumptions=["Periodic boundaries"],
        limitations=[],
    )

    return Explanation(
        explanation_id="test_exp",
        hypothesis_text="Test explanation",
        mechanism=mechanism,
        confidence=0.8,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
