"""Finalize phase handler for the orchestration engine.

Handles the FINALIZE phase where a comprehensive report is generated
summarizing the discovery run including:
- Validated laws and theorems
- Mechanistic explanations
- Prediction accuracy statistics
- Counterexample gallery
- Run metrics and diagnostics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.orchestration.control_block import (
    ControlBlock,
    EvidenceReference,
    PhaseRecommendation,
    StopReason,
)
from src.orchestration.phases import Phase, PhaseContext
from src.orchestration.readiness import ReadinessMetrics

if TYPE_CHECKING:
    from src.db.repo import Repository

logger = logging.getLogger(__name__)


@dataclass
class FinalReport:
    """Final report from a discovery run."""

    run_id: str
    generated_at: str
    total_iterations: int
    phase_iterations: dict[str, int]
    final_phase: str

    # Discovery outcomes
    laws_proposed: int
    laws_passed: int
    laws_failed: int
    laws_unknown: int

    # Theorems
    theorems_synthesized: int
    theorems_established: int

    # Explanations
    explanations_generated: int
    final_explanation_confidence: float

    # Predictions
    held_out_accuracy: float
    adversarial_accuracy: float
    regression_accuracy: float

    # Diagnostics
    final_readiness: dict[str, float]
    counterexample_count: int
    run_duration_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "total_iterations": self.total_iterations,
            "phase_iterations": self.phase_iterations,
            "final_phase": self.final_phase,
            "discovery": {
                "laws_proposed": self.laws_proposed,
                "laws_passed": self.laws_passed,
                "laws_failed": self.laws_failed,
                "laws_unknown": self.laws_unknown,
            },
            "theorems": {
                "synthesized": self.theorems_synthesized,
                "established": self.theorems_established,
            },
            "explanations": {
                "generated": self.explanations_generated,
                "final_confidence": self.final_explanation_confidence,
            },
            "predictions": {
                "held_out_accuracy": self.held_out_accuracy,
                "adversarial_accuracy": self.adversarial_accuracy,
                "regression_accuracy": self.regression_accuracy,
            },
            "diagnostics": {
                "final_readiness": self.final_readiness,
                "counterexample_count": self.counterexample_count,
                "run_duration_seconds": self.run_duration_seconds,
            },
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class FinalizePhaseConfig:
    """Configuration for the finalize phase.

    Attributes:
        include_counterexamples: Include counterexample gallery in report
        max_counterexamples: Max counterexamples to include
        include_law_details: Include detailed law information
    """

    include_counterexamples: bool = True
    max_counterexamples: int = 20
    include_law_details: bool = True


class FinalizePhaseHandler:
    """Handler for the FINALIZE phase.

    This phase generates a comprehensive report of the discovery run
    and marks the run as complete. It is terminal - once entered,
    the run is considered finished.
    """

    def __init__(
        self,
        repo: Repository | None,
        config: FinalizePhaseConfig | None = None,
    ):
        """Initialize the finalize handler.

        Args:
            repo: Database repository
            config: Phase configuration
        """
        self.repo = repo
        self.config = config or FinalizePhaseConfig()
        self._final_report: FinalReport | None = None

    @property
    def phase(self) -> Phase:
        return Phase.FINALIZE

    def run_iteration(self, context: PhaseContext) -> ControlBlock:
        """Execute the finalize phase.

        This generates the final report and returns a control block
        indicating completion.

        Args:
            context: Phase execution context

        Returns:
            ControlBlock indicating completion
        """
        run_id = context.run_id

        logger.info(f"Finalizing run {run_id}")

        # Generate report
        report = self._generate_report(context)
        self._final_report = report

        # Persist report
        if self.repo:
            self._persist_report(run_id, report)

        # Build control block
        evidence: list[EvidenceReference] = []
        evidence.append(EvidenceReference(
            artifact_type="final_report",
            artifact_id=f"report_{run_id}",
            role="supports",
            note=f"Final report generated at {report.generated_at}",
        ))

        justification = (
            f"Run complete. "
            f"{report.laws_passed} laws validated, "
            f"{report.theorems_established} theorems established. "
            f"Held-out accuracy: {report.held_out_accuracy:.2%}."
        )

        return ControlBlock(
            readiness_score_suggestion=100,
            readiness_justification=justification,
            phase_recommendation=PhaseRecommendation.STAY,  # Terminal phase
            stop_reason=StopReason.SATURATION,
            evidence=evidence,
            phase_outputs={"report": report.to_dict()},
        )

    def can_handle_requests(self, requests: list) -> bool:
        """Finalize doesn't handle any requests - it's terminal."""
        return False

    def get_final_report(self) -> FinalReport | None:
        """Get the generated final report."""
        return self._final_report

    def _generate_report(self, context: PhaseContext) -> FinalReport:
        """Generate the final report from database state.

        Args:
            context: Phase context

        Returns:
            FinalReport with run summary
        """
        run_id = context.run_id

        # Get run metadata
        run_record = None
        if self.repo:
            run_record = self.repo.get_orchestration_run(run_id)

        # Count iterations by phase
        phase_iterations: dict[str, int] = {}
        if self.repo:
            iterations = self.repo.list_iterations_for_run(run_id, limit=1000)
            for it in iterations:
                phase_iterations[it.phase] = phase_iterations.get(it.phase, 0) + 1

        # Get law statistics
        laws_proposed = 0
        laws_passed = 0
        laws_failed = 0
        laws_unknown = 0
        if self.repo:
            # Count by evaluation status
            pass_evals = self.repo.list_evaluations(status="PASS", limit=10000)
            fail_evals = self.repo.list_evaluations(status="FAIL", limit=10000)
            unknown_evals = self.repo.list_evaluations(status="UNKNOWN", limit=10000)
            laws_passed = len(pass_evals)
            laws_failed = len(fail_evals)
            laws_unknown = len(unknown_evals)
            laws_proposed = laws_passed + laws_failed + laws_unknown

        # Get theorem statistics
        theorems_synthesized = 0
        theorems_established = 0
        if self.repo:
            theorem_records = self.repo.list_theorems(limit=1000)
            theorems_synthesized = len(theorem_records)
            theorems_established = sum(
                1 for t in theorem_records
                if t.status == "established"
            )

        # Get explanation statistics
        explanations_generated = 0
        final_explanation_confidence = 0.0
        if self.repo:
            try:
                explanation_records = self.repo.list_explanations(run_id, limit=100)
                explanations_generated = len(explanation_records)
                if explanation_records:
                    final_explanation_confidence = explanation_records[-1].confidence
            except Exception:
                pass  # Explanation table might not exist

        # Get prediction statistics from latest readiness
        held_out_accuracy = 0.0
        adversarial_accuracy = 0.0
        regression_accuracy = 0.0
        if self.repo:
            try:
                latest_snap = self.repo.get_latest_readiness_snapshot(run_id)
                if latest_snap:
                    held_out_accuracy = latest_snap.s_held_out_accuracy or 0.0
                    adversarial_accuracy = latest_snap.s_adversarial_accuracy or 0.0
            except Exception:
                pass

        # Get counterexample count
        counterexample_count = 0
        if self.repo:
            try:
                cex_records = self.repo.list_counterexamples(limit=10000)
                counterexample_count = len(cex_records)
            except Exception:
                pass

        # Build final readiness dict
        final_readiness: dict[str, float] = {}
        if self.repo:
            try:
                latest_snap = self.repo.get_latest_readiness_snapshot(run_id)
                if latest_snap:
                    final_readiness = {
                        "s_pass": latest_snap.s_pass,
                        "s_stability": latest_snap.s_stability,
                        "s_novel_cex": latest_snap.s_novel_cex,
                        "s_harness_health": latest_snap.s_harness_health,
                        "combined_score": latest_snap.combined_score,
                    }
            except Exception:
                pass

        # Calculate run duration
        run_duration_seconds = None
        if run_record and run_record.started_at:
            try:
                start = datetime.fromisoformat(run_record.started_at.replace('Z', '+00:00'))
                end = datetime.utcnow()
                run_duration_seconds = (end - start).total_seconds()
            except Exception:
                pass

        return FinalReport(
            run_id=run_id,
            generated_at=datetime.utcnow().isoformat(),
            total_iterations=context.iteration_index,
            phase_iterations=phase_iterations,
            final_phase=context.current_phase.value,
            laws_proposed=laws_proposed,
            laws_passed=laws_passed,
            laws_failed=laws_failed,
            laws_unknown=laws_unknown,
            theorems_synthesized=theorems_synthesized,
            theorems_established=theorems_established,
            explanations_generated=explanations_generated,
            final_explanation_confidence=final_explanation_confidence,
            held_out_accuracy=held_out_accuracy,
            adversarial_accuracy=adversarial_accuracy,
            regression_accuracy=regression_accuracy,
            final_readiness=final_readiness,
            counterexample_count=counterexample_count,
            run_duration_seconds=run_duration_seconds,
        )

    def _persist_report(self, run_id: str, report: FinalReport) -> None:
        """Persist final report to database.

        Args:
            run_id: Run ID
            report: Final report
        """
        if not self.repo:
            return

        # Store report JSON in run record
        try:
            self.repo.update_orchestration_run(
                run_id,
                final_report_json=report.to_json(),
            )
        except Exception as e:
            logger.error(f"Failed to persist final report: {e}")
