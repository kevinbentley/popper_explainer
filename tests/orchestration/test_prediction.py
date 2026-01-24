"""Tests for the prediction verification subsystem.

Tests cover:
- Test set generation (held-out, adversarial, regression)
- Prediction verification and scoring
- PredictionPhaseHandler behavior
"""

import pytest

from src.orchestration.prediction.test_sets import (
    HeldOutSetManager,
    PredictionTestCase,
    PredictionTestSet,
)
from src.orchestration.prediction.adversarial import (
    AdversarialPredictionGenerator,
    AdversarialSearchConfig,
)
from src.orchestration.prediction.verifier import (
    AccuracyThresholds,
    PredictionVerifier,
    VerificationReport,
)
from src.universe.types import Config


class TestHeldOutSetManager:
    """Tests for held-out test set management."""

    def test_create_held_out_set(self):
        """Verify held-out set creation."""
        manager = HeldOutSetManager()

        test_set = manager.create_held_out_set(
            run_id="test_run",
            seed=42,
            count=10,
            horizons=[1],
            grid_lengths=[8],
        )

        assert test_set.set_type == "held_out"
        assert test_set.case_count == 10
        assert test_set.locked is True
        assert test_set.generation_seed == 42

        # All cases should have expected states computed
        for case in test_set.cases:
            assert case.expected_state is not None
            assert len(case.expected_state) == 8

    def test_create_regression_set(self):
        """Verify regression set creation with edge cases."""
        manager = HeldOutSetManager()

        test_set = manager.create_regression_set(run_id="test_run")

        assert test_set.set_type == "regression"
        assert test_set.locked is True
        assert test_set.case_count > 0

        # Regression set should include collision cases
        has_collision_case = any(
            "X" in case.initial_state or "><" in case.initial_state
            for case in test_set.cases
        )
        assert has_collision_case

    def test_held_out_set_deterministic(self):
        """Verify same seed produces same set."""
        manager = HeldOutSetManager()

        set1 = manager.create_held_out_set(
            run_id="run1", seed=123, count=5, horizons=[1], grid_lengths=[10]
        )
        set2 = manager.create_held_out_set(
            run_id="run2", seed=123, count=5, horizons=[1], grid_lengths=[10]
        )

        # Same seed should produce same initial states
        for case1, case2 in zip(set1.cases, set2.cases):
            assert case1.initial_state == case2.initial_state
            assert case1.expected_state == case2.expected_state


class TestAdversarialGenerator:
    """Tests for adversarial test generation."""

    def test_generate_without_predictor(self):
        """Verify generation works without predictor (edge cases only)."""
        generator = AdversarialPredictionGenerator()

        test_set = generator.generate_adversarial_set(
            run_id="test_run",
            seed=42,
            count=20,
            horizons=[1],
            grid_lengths=[8],
        )

        assert test_set.set_type == "adversarial"
        assert test_set.case_count >= 15  # Allow some flexibility
        assert test_set.locked is True

    def test_generate_with_perfect_predictor(self):
        """Verify adversarial generation with a perfect predictor."""
        from src.universe.simulator import run as sim_run

        def perfect_predictor(state: str, horizon: int) -> str:
            trajectory = sim_run(state, horizon, Config(grid_length=len(state)))
            return trajectory[-1]

        generator = AdversarialPredictionGenerator()
        generator.set_predictor(perfect_predictor)

        test_set = generator.generate_adversarial_set(
            run_id="test_run",
            seed=42,
            count=10,
            horizons=[1],
            grid_lengths=[8],
        )

        # Should still generate cases even if predictor is perfect
        assert test_set.case_count >= 5  # Allow flexibility

    def test_generate_with_bad_predictor(self):
        """Verify adversarial generation finds failures with bad predictor."""

        def bad_predictor(state: str, horizon: int) -> str:
            # Always predicts empty state
            return "." * len(state)

        generator = AdversarialPredictionGenerator()
        generator.set_predictor(bad_predictor)

        test_set = generator.generate_adversarial_set(
            run_id="test_run",
            seed=42,
            count=20,
            horizons=[1],
            grid_lengths=[8],
        )

        # Bad predictor should generate cases - allow flexibility
        assert test_set.case_count >= 15


class TestPredictionVerifier:
    """Tests for prediction verification and scoring."""

    def test_verify_exact_match(self):
        """Verify exact match detection."""
        verifier = PredictionVerifier()

        case = PredictionTestCase(
            initial_state=">..<....",
            horizon=1,
            config=Config(grid_length=8),
            expected_state=".>.<....",
        )

        result = verifier.verify_prediction(case, ".>.<....")

        assert result.is_exact_match is True
        assert result.hamming_distance == 0
        assert result.cell_accuracy == 1.0

    def test_verify_partial_match(self):
        """Verify partial match scoring."""
        verifier = PredictionVerifier()

        case = PredictionTestCase(
            initial_state=">..<....",
            horizon=1,
            config=Config(grid_length=8),
            expected_state=".>.<....",  # 8 chars: .>.<....
        )

        # Prediction: ".>......" differs from expected ".>.<...." in position 3 (< vs .)
        result = verifier.verify_prediction(case, ".>......")

        assert result.is_exact_match is False
        assert result.hamming_distance == 1  # Only position 3 differs (< vs .)
        assert result.cell_accuracy == 7/8  # 7 of 8 cells correct

    def test_verify_complete_mismatch(self):
        """Verify complete mismatch scoring."""
        verifier = PredictionVerifier()

        case = PredictionTestCase(
            initial_state=">>>>>>>>",
            horizon=1,
            config=Config(grid_length=8),
            expected_state=">>>>>>>>",  # All rights stay (wrap around)
        )

        # Completely wrong prediction
        result = verifier.verify_prediction(case, "<<<<<<<<")

        assert result.is_exact_match is False
        assert result.hamming_distance == 8
        assert result.cell_accuracy == 0.0

    def test_verify_test_set(self):
        """Verify full test set verification."""
        from src.universe.simulator import run as sim_run

        verifier = PredictionVerifier()
        manager = HeldOutSetManager()

        test_set = manager.create_held_out_set(
            run_id="test", seed=42, count=10, horizons=[1], grid_lengths=[8]
        )

        def perfect_predictor(state: str, horizon: int) -> str:
            trajectory = sim_run(state, horizon, Config(grid_length=len(state)))
            return trajectory[-1]

        report = verifier.verify_test_set(test_set, perfect_predictor)

        assert report.total_cases == 10
        assert report.exact_matches == 10
        assert report.exact_match_rate == 1.0
        assert report.mean_cell_accuracy == 1.0


class TestAccuracyThresholds:
    """Tests for accuracy threshold checking."""

    def test_held_out_passes(self):
        """Verify held-out threshold checking."""
        thresholds = AccuracyThresholds(
            held_out_exact_match=0.90,
            held_out_cell_accuracy=0.95,
        )

        # Passing report
        report = VerificationReport(
            set_id="test",
            set_type="held_out",
            total_cases=100,
            exact_matches=95,
            exact_match_rate=0.95,
            mean_hamming_distance=0.1,
            mean_cell_accuracy=0.98,
        )

        passes, reason = thresholds.check_report(report)
        assert passes is True

    def test_held_out_fails_exact_match(self):
        """Verify held-out fails when exact match is low."""
        thresholds = AccuracyThresholds(
            held_out_exact_match=0.90,
            held_out_cell_accuracy=0.95,
        )

        report = VerificationReport(
            set_id="test",
            set_type="held_out",
            total_cases=100,
            exact_matches=80,
            exact_match_rate=0.80,
            mean_hamming_distance=0.5,
            mean_cell_accuracy=0.96,  # Meets cell accuracy threshold
        )

        passes, reason = thresholds.check_report(report)
        assert passes is False
        assert "80" in reason  # Contains the exact match percentage

    def test_adversarial_thresholds(self):
        """Verify adversarial thresholds are lower than held-out."""
        thresholds = AccuracyThresholds()

        # Adversarial should have lower bar
        assert thresholds.adversarial_exact_match < thresholds.held_out_exact_match


class TestPredictionPhaseHandler:
    """Tests for the prediction phase handler."""

    def _make_context(self, run_id: str = "test_run", iteration_index: int = 0):
        """Create a minimal PhaseContext for testing."""
        from src.orchestration.phases import PhaseContext, Phase, OrchestratorConfig

        return PhaseContext(
            run_id=run_id,
            iteration_index=iteration_index,
            repo=None,  # type: ignore
            config=OrchestratorConfig(),
            current_phase=Phase.PREDICTION,
        )

    def test_handler_without_predictor(self):
        """Verify handler returns error without predictor."""
        from src.orchestration.handlers.prediction_handler import PredictionPhaseHandler

        handler = PredictionPhaseHandler(repo=None)
        context = self._make_context()

        result = handler.execute(context)

        assert "No predictor" in result.summary.get("error", "")

    def test_handler_with_perfect_predictor(self):
        """Verify handler advances with perfect predictor."""
        from src.orchestration.handlers.prediction_handler import (
            PredictionPhaseHandler,
            PredictionPhaseConfig,
        )
        from src.orchestration.control_block import PhaseRecommendation
        from src.universe.simulator import run as sim_run

        config = PredictionPhaseConfig(
            held_out_count=10,
            adversarial_count=5,
            horizons=[1],
            grid_lengths=[8],
        )
        handler = PredictionPhaseHandler(repo=None, config=config)

        def perfect_predictor(state: str, horizon: int) -> str:
            trajectory = sim_run(state, horizon, Config(grid_length=len(state)))
            return trajectory[-1]

        handler.set_predictor(perfect_predictor)
        context = self._make_context()

        result = handler.execute(context)

        # Perfect predictor should pass all thresholds
        assert result.control_block.phase_recommendation == PhaseRecommendation.ADVANCE
        assert result.summary["held_out_exact_match"] == 1.0
        assert result.summary["regression_exact_match"] == 1.0

    def test_handler_with_bad_predictor(self):
        """Verify handler stays with bad predictor."""
        from src.orchestration.handlers.prediction_handler import (
            PredictionPhaseHandler,
            PredictionPhaseConfig,
        )
        from src.orchestration.control_block import PhaseRecommendation

        config = PredictionPhaseConfig(
            held_out_count=10,
            adversarial_count=5,
            horizons=[1],
            grid_lengths=[8],
        )
        handler = PredictionPhaseHandler(repo=None, config=config)

        def bad_predictor(state: str, horizon: int) -> str:
            return "." * len(state)  # Always predicts empty

        handler.set_predictor(bad_predictor)
        context = self._make_context()

        result = handler.execute(context)

        # Bad predictor should not pass
        assert result.control_block.phase_recommendation in (
            PhaseRecommendation.STAY,
            PhaseRecommendation.RETREAT,
        )
        assert result.summary["held_out_exact_match"] < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
