"""Tests for the finalize phase handler.

Tests cover:
- Report generation
- FinalizePhaseHandler behavior
"""

import pytest

from src.orchestration.handlers.finalize_handler import (
    FinalizePhaseHandler,
    FinalizePhaseConfig,
    FinalReport,
)
from src.orchestration.control_block import PhaseRecommendation, StopReason
from src.orchestration.phases import Phase, PhaseContext, OrchestratorConfig


class TestFinalReport:
    """Tests for FinalReport dataclass."""

    def test_report_to_dict(self):
        """Test FinalReport serialization."""
        report = FinalReport(
            run_id="test_run",
            generated_at="2025-01-01T00:00:00",
            total_iterations=100,
            phase_iterations={"discovery": 50, "theorem": 30, "explanation": 20},
            final_phase="finalize",
            laws_proposed=200,
            laws_passed=150,
            laws_failed=30,
            laws_unknown=20,
            theorems_synthesized=25,
            theorems_established=20,
            explanations_generated=5,
            final_explanation_confidence=0.9,
            held_out_accuracy=0.95,
            adversarial_accuracy=0.88,
            regression_accuracy=0.99,
            final_readiness={"s_pass": 0.95, "combined_score": 92.0},
            counterexample_count=30,
            run_duration_seconds=3600.0,
        )

        data = report.to_dict()

        assert data["run_id"] == "test_run"
        assert data["total_iterations"] == 100
        assert data["discovery"]["laws_passed"] == 150
        assert data["theorems"]["established"] == 20
        assert data["predictions"]["held_out_accuracy"] == 0.95
        assert data["diagnostics"]["counterexample_count"] == 30

    def test_report_to_json(self):
        """Test FinalReport JSON serialization."""
        report = FinalReport(
            run_id="test_run",
            generated_at="2025-01-01T00:00:00",
            total_iterations=50,
            phase_iterations={},
            final_phase="finalize",
            laws_proposed=100,
            laws_passed=80,
            laws_failed=15,
            laws_unknown=5,
            theorems_synthesized=10,
            theorems_established=8,
            explanations_generated=2,
            final_explanation_confidence=0.85,
            held_out_accuracy=0.92,
            adversarial_accuracy=0.85,
            regression_accuracy=0.98,
            final_readiness={},
            counterexample_count=15,
            run_duration_seconds=None,
        )

        json_str = report.to_json()

        assert "test_run" in json_str
        assert '"laws_passed": 80' in json_str


class TestFinalizePhaseHandler:
    """Tests for the finalize phase handler."""

    def _make_context(self, run_id: str = "test_run", iteration_index: int = 0):
        """Create a minimal PhaseContext for testing."""
        return PhaseContext(
            run_id=run_id,
            iteration_index=iteration_index,
            repo=None,  # type: ignore
            config=OrchestratorConfig(),
            current_phase=Phase.FINALIZE,
        )

    def test_handler_phase(self):
        """Test handler reports correct phase."""
        handler = FinalizePhaseHandler(repo=None)
        assert handler.phase == Phase.FINALIZE

    def test_handler_runs_without_repo(self):
        """Test handler can run without database."""
        handler = FinalizePhaseHandler(repo=None)
        context = self._make_context()

        control_block = handler.run_iteration(context)

        assert control_block.stop_reason == StopReason.SATURATION
        assert control_block.readiness_score_suggestion == 100

    def test_handler_generates_report(self):
        """Test handler generates a report."""
        handler = FinalizePhaseHandler(repo=None)
        context = self._make_context()

        handler.run_iteration(context)

        report = handler.get_final_report()
        assert report is not None
        assert report.run_id == "test_run"

    def test_handler_report_in_outputs(self):
        """Test report is included in phase_outputs."""
        handler = FinalizePhaseHandler(repo=None)
        context = self._make_context(iteration_index=100)

        control_block = handler.run_iteration(context)

        assert "report" in control_block.phase_outputs
        assert control_block.phase_outputs["report"]["total_iterations"] == 100

    def test_handler_cannot_handle_requests(self):
        """Test finalize handler doesn't handle any requests."""
        handler = FinalizePhaseHandler(repo=None)

        assert handler.can_handle_requests([]) is False
        assert handler.can_handle_requests([object()]) is False  # type: ignore

    def test_handler_is_terminal(self):
        """Test handler indicates terminal phase behavior."""
        handler = FinalizePhaseHandler(repo=None)
        context = self._make_context()

        control_block = handler.run_iteration(context)

        # Should stay in finalize (terminal)
        assert control_block.phase_recommendation == PhaseRecommendation.STAY
        # But indicate completion
        assert control_block.stop_reason == StopReason.SATURATION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
