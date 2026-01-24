"""Integration tests for the full orchestration loop.

Tests the complete flow through all phases:
DISCOVERY -> THEOREM -> EXPLANATION -> PREDICTION -> FINALIZE
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.db.repo import Repository
from src.orchestration.engine import OrchestratorEngine
from src.orchestration.phases import Phase, OrchestratorConfig
from src.orchestration.control_block import ControlBlock, PhaseRecommendation, StopReason
from src.orchestration.readiness import ReadinessMetrics, ReadinessComputer


class MockDiscoveryHandler:
    """Mock discovery handler that advances after a few iterations."""

    def __init__(self, advance_after: int = 3):
        self.advance_after = advance_after
        self.iteration_count = 0

    @property
    def phase(self) -> Phase:
        return Phase.DISCOVERY

    def run_iteration(self, context) -> ControlBlock:
        self.iteration_count += 1

        if self.iteration_count >= self.advance_after:
            return ControlBlock(
                readiness_score_suggestion=95,
                readiness_justification="Discovery saturated",
                phase_recommendation=PhaseRecommendation.ADVANCE,
                stop_reason=StopReason.SATURATION,
            )

        return ControlBlock(
            readiness_score_suggestion=60,
            readiness_justification="Still discovering",
            phase_recommendation=PhaseRecommendation.STAY,
            stop_reason=StopReason.CONTINUING,
        )

    def can_handle_requests(self, requests) -> bool:
        return False


class MockTheoremHandler:
    """Mock theorem handler that advances after a few iterations."""

    def __init__(self, advance_after: int = 2):
        self.advance_after = advance_after
        self.iteration_count = 0

    @property
    def phase(self) -> Phase:
        return Phase.THEOREM

    def run_iteration(self, context) -> ControlBlock:
        self.iteration_count += 1

        if self.iteration_count >= self.advance_after:
            return ControlBlock(
                readiness_score_suggestion=90,
                readiness_justification="Theorems synthesized",
                phase_recommendation=PhaseRecommendation.ADVANCE,
                stop_reason=StopReason.SATURATION,
            )

        return ControlBlock(
            readiness_score_suggestion=70,
            readiness_justification="Synthesizing theorems",
            phase_recommendation=PhaseRecommendation.STAY,
            stop_reason=StopReason.CONTINUING,
        )

    def can_handle_requests(self, requests) -> bool:
        return False


class MockExplanationHandler:
    """Mock explanation handler that provides a predictor."""

    def __init__(self, advance_after: int = 2):
        self.advance_after = advance_after
        self.iteration_count = 0

    @property
    def phase(self) -> Phase:
        return Phase.EXPLANATION

    def run_iteration(self, context) -> ControlBlock:
        self.iteration_count += 1

        if self.iteration_count >= self.advance_after:
            return ControlBlock(
                readiness_score_suggestion=85,
                readiness_justification="Explanation complete",
                phase_recommendation=PhaseRecommendation.ADVANCE,
                stop_reason=StopReason.SATURATION,
            )

        return ControlBlock(
            readiness_score_suggestion=70,
            readiness_justification="Building explanation",
            phase_recommendation=PhaseRecommendation.STAY,
            stop_reason=StopReason.CONTINUING,
        )

    def can_handle_requests(self, requests) -> bool:
        return False

    def get_predictor(self):
        """Return a perfect predictor."""
        from src.universe.simulator import run as sim_run
        from src.universe.types import Config

        def predictor(state: str, horizon: int) -> str:
            return sim_run(state, horizon, Config(grid_length=len(state)))[-1]

        return predictor


class MockPredictionHandler:
    """Mock prediction handler that advances after verification."""

    def __init__(self, advance_after: int = 2):
        self.advance_after = advance_after
        self.iteration_count = 0
        self._predictor = None

    @property
    def phase(self) -> Phase:
        return Phase.PREDICTION

    def set_predictor(self, predictor):
        self._predictor = predictor

    def run_iteration(self, context) -> ControlBlock:
        self.iteration_count += 1

        if self.iteration_count >= self.advance_after:
            return ControlBlock(
                readiness_score_suggestion=98,
                readiness_justification="Predictions verified",
                phase_recommendation=PhaseRecommendation.ADVANCE,
                stop_reason=StopReason.SATURATION,
            )

        return ControlBlock(
            readiness_score_suggestion=80,
            readiness_justification="Verifying predictions",
            phase_recommendation=PhaseRecommendation.STAY,
            stop_reason=StopReason.CONTINUING,
        )

    def can_handle_requests(self, requests) -> bool:
        return False


class MockFinalizeHandler:
    """Mock finalize handler that completes the run."""

    @property
    def phase(self) -> Phase:
        return Phase.FINALIZE

    def run_iteration(self, context) -> ControlBlock:
        return ControlBlock(
            readiness_score_suggestion=100,
            readiness_justification="Run complete",
            phase_recommendation=PhaseRecommendation.STAY,
            stop_reason=StopReason.SATURATION,
        )

    def can_handle_requests(self, requests) -> bool:
        return False


class TestOrchestratorIntegration:
    """Integration tests for the orchestration engine."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except OSError:
            pass

    @pytest.fixture
    def repo(self, temp_db):
        """Create a repository with the temporary database."""
        repo = Repository(temp_db)
        repo.connect()
        yield repo
        repo.close()

    def test_full_loop_with_mock_handlers(self, repo):
        """Test complete orchestration loop with mock handlers."""
        # Configure for fast transitions
        config = OrchestratorConfig(
            consecutive_rounds_for_advance=1,  # Fast transitions
            max_total_iterations=50,
        )

        # Create engine
        engine = OrchestratorEngine(repo=repo, config=config)

        # Create mock handlers
        discovery = MockDiscoveryHandler(advance_after=2)
        theorem = MockTheoremHandler(advance_after=2)
        explanation = MockExplanationHandler(advance_after=2)
        prediction = MockPredictionHandler(advance_after=2)
        finalize = MockFinalizeHandler()

        # Register handlers
        engine.register_handler(discovery)
        engine.register_handler(theorem)
        engine.register_handler(explanation)
        engine.register_handler(prediction)
        engine.register_handler(finalize)

        # Wire predictor callback
        def wire_predictor():
            predictor = explanation.get_predictor()
            if predictor:
                prediction.set_predictor(predictor)

        engine._wire_predictor = wire_predictor

        # Mock the readiness computer to return high readiness
        def mock_compute_for_phase(run_id, phase):
            metrics = ReadinessMetrics()
            metrics.s_pass = 0.98
            metrics.s_stability = 0.95
            metrics.s_novel_cex = 0.05  # Low = saturated, not finding new counterexamples
            metrics.s_harness_health = 1.0
            metrics.s_coverage = 0.95
            metrics.combined_score = 0.96  # Above 95% threshold for prediction->finalize
            return metrics

        with patch.object(engine.readiness_computer, 'compute_for_phase', mock_compute_for_phase):
            # Run
            result = engine.run(max_iterations=30)

        # Verify
        assert result.status == "completed"
        assert result.final_phase == Phase.FINALIZE
        assert result.total_iterations > 0

        # Check that all phases were visited
        assert result.phase_iterations.get(Phase.DISCOVERY, 0) > 0
        assert result.phase_iterations.get(Phase.THEOREM, 0) > 0

    def test_max_iterations_stops_run(self, repo):
        """Test that max iterations limit is respected."""
        config = OrchestratorConfig(
            max_total_iterations=5,
            consecutive_rounds_for_advance=10,  # High so we don't advance
        )

        engine = OrchestratorEngine(repo=repo, config=config)

        # Only register discovery - won't advance
        discovery = MockDiscoveryHandler(advance_after=100)  # Never advances
        engine.register_handler(discovery)

        result = engine.run()

        assert result.total_iterations == 5
        assert result.final_phase == Phase.DISCOVERY

    def test_resume_continues_run(self, repo):
        """Test that runs can be resumed after interruption."""
        config = OrchestratorConfig(
            consecutive_rounds_for_advance=1,
            max_total_iterations=10,
        )

        engine = OrchestratorEngine(repo=repo, config=config)

        discovery = MockDiscoveryHandler(advance_after=100)
        engine.register_handler(discovery)

        # Start a run with limit of 3
        result1 = engine.run(max_iterations=3)
        run_id = result1.run_id

        assert result1.total_iterations == 3

        # Manually update run status back to "running" to simulate interruption
        # (In real usage, an aborted run would have status "aborted" or "running")
        repo.update_orchestration_run(run_id, status="running")

        # Create new engine to resume
        engine2 = OrchestratorEngine(repo=repo, config=config)
        engine2.register_handler(MockDiscoveryHandler(advance_after=100))

        # Resume the run for 3 more iterations
        result2 = engine2.run(run_id=run_id, resume=True, max_iterations=6)

        # Should have continued from iteration 3
        assert result2.total_iterations == 6
        assert result2.run_id == run_id

    def test_phase_transition_persisted(self, repo):
        """Test that phase transitions are persisted to database."""
        config = OrchestratorConfig(
            consecutive_rounds_for_advance=1,
            max_total_iterations=20,
        )

        engine = OrchestratorEngine(repo=repo, config=config)

        # Handlers that advance quickly
        engine.register_handler(MockDiscoveryHandler(advance_after=2))
        engine.register_handler(MockTheoremHandler(advance_after=2))

        # Mock the readiness computer to return high readiness
        def mock_compute_for_phase(run_id, phase):
            metrics = ReadinessMetrics()
            metrics.s_pass = 0.95
            metrics.s_stability = 0.90
            metrics.s_novel_cex = 0.05  # Low = saturated, not finding new counterexamples
            metrics.s_harness_health = 1.0
            metrics.s_coverage = 0.90
            metrics.combined_score = 0.92
            return metrics

        with patch.object(engine.readiness_computer, 'compute_for_phase', mock_compute_for_phase):
            result = engine.run(max_iterations=10)

        # Check transitions were recorded
        transitions = repo.list_phase_transitions(result.run_id)
        assert len(transitions) >= 1

        # First transition should be discovery -> theorem
        assert transitions[0].from_phase == "discovery"
        assert transitions[0].to_phase == "theorem"

    def test_error_handling(self, repo):
        """Test that errors are handled gracefully."""

        class FailingHandler:
            @property
            def phase(self) -> Phase:
                return Phase.DISCOVERY

            def run_iteration(self, context):
                raise RuntimeError("Handler failed!")

            def can_handle_requests(self, requests) -> bool:
                return False

        config = OrchestratorConfig(max_total_iterations=10)
        engine = OrchestratorEngine(repo=repo, config=config)
        engine.register_handler(FailingHandler())

        result = engine.run()

        assert result.status == "error"
        assert "Handler failed!" in result.error


class TestOrchestratorWithRealHandlers:
    """Integration tests using real handlers (where possible)."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except OSError:
            pass

    @pytest.fixture
    def repo(self, temp_db):
        """Create a repository with the temporary database."""
        repo = Repository(temp_db)
        repo.connect()
        yield repo
        repo.close()

    def test_explanation_to_prediction_wiring(self, repo):
        """Test that explanation handler's predictor is wired to prediction handler."""
        from src.orchestration.handlers.explanation_handler import (
            ExplanationPhaseHandler,
            ExplanationPhaseConfig,
        )
        from src.orchestration.handlers.prediction_handler import (
            PredictionPhaseHandler,
            PredictionPhaseConfig,
        )

        # Create handlers
        explanation_handler = ExplanationPhaseHandler(
            repo=None,  # type: ignore
            config=ExplanationPhaseConfig(min_theorems=0),  # Allow 0 theorems
        )
        prediction_handler = PredictionPhaseHandler(
            repo=repo,
            config=PredictionPhaseConfig(held_out_count=10, adversarial_count=5),
        )

        # Mock explanation execution to create a predictor
        from src.orchestration.phases import PhaseContext, OrchestratorConfig

        context = PhaseContext(
            run_id="test_run",
            iteration_index=0,
            repo=None,  # type: ignore
            config=OrchestratorConfig(),
            current_phase=Phase.EXPLANATION,
        )

        # Execute explanation handler to create predictor
        explanation_handler.execute(context)

        # Get predictor and wire to prediction handler
        predictor = explanation_handler.get_predictor()
        assert predictor is not None

        prediction_handler.set_predictor(predictor)

        # Verify predictor works
        result = predictor(">.......", 1)
        assert result == ".>......"

    def test_finalize_handler_generates_report(self, repo):
        """Test that finalize handler generates a proper report."""
        from src.orchestration.handlers.finalize_handler import FinalizePhaseHandler
        from src.orchestration.phases import PhaseContext, OrchestratorConfig

        handler = FinalizePhaseHandler(repo=repo)

        context = PhaseContext(
            run_id="test_run",
            iteration_index=100,
            repo=repo,
            config=OrchestratorConfig(),
            current_phase=Phase.FINALIZE,
        )

        control_block = handler.run_iteration(context)

        # Verify report was generated
        report = handler.get_final_report()
        assert report is not None
        assert report.run_id == "test_run"
        assert report.total_iterations == 100

        # Verify control block
        assert control_block.stop_reason == StopReason.SATURATION
        assert "report" in control_block.phase_outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
