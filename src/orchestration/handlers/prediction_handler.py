"""Prediction phase handler for the orchestration engine.

Handles the PREDICTION phase where explanations are verified against
held-out and adversarial test sets to ensure predictions are accurate
and not overfit to the training feedback loop.

The prediction phase:
1. Generates or loads held-out test sets
2. Generates adversarial test sets targeting weak areas
3. Runs predictions and verifies against ground truth
4. Computes accuracy metrics for phase transition decisions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from src.orchestration.control_block import (
    ControlBlock,
    EvidenceReference,
    PhaseRecommendation,
    PhaseRequest,
    StopReason,
)
from src.orchestration.phases import IterationResult, Phase, PhaseContext, PhaseHandler
from src.orchestration.prediction.adversarial import AdversarialPredictionGenerator
from src.orchestration.prediction.test_sets import HeldOutSetManager, PredictionTestSet
from src.orchestration.prediction.verifier import (
    AccuracyThresholds,
    PredictionVerifier,
    VerificationReport,
)
from src.orchestration.readiness import ReadinessMetrics

if TYPE_CHECKING:
    from src.db.repo import Repository

logger = logging.getLogger(__name__)


@dataclass
class PredictionPhaseConfig:
    """Configuration for the prediction phase.

    Attributes:
        held_out_count: Number of held-out test cases
        adversarial_count: Number of adversarial test cases
        horizons: Prediction horizons to test
        grid_lengths: Grid sizes to test
        thresholds: Accuracy thresholds for advancement
        regenerate_adversarial: Whether to regenerate adversarial set each iteration
    """

    held_out_count: int = 100
    adversarial_count: int = 50
    horizons: list[int] | None = None
    grid_lengths: list[int] | None = None
    thresholds: AccuracyThresholds | None = None
    regenerate_adversarial: bool = True

    def __post_init__(self):
        self.horizons = self.horizons or [1, 2, 5]
        self.grid_lengths = self.grid_lengths or [8, 16, 32]
        self.thresholds = self.thresholds or AccuracyThresholds()


class PredictionPhaseHandler(PhaseHandler):
    """Handler for the PREDICTION phase.

    In this phase, explanations are tested against held-out and
    adversarial test sets. The phase advances to FINALIZE when
    prediction accuracy meets thresholds.

    The handler expects a predictor function to be set via
    `set_predictor()` before execution.
    """

    def __init__(
        self,
        repo: Repository,
        config: PredictionPhaseConfig | None = None,
    ):
        """Initialize the prediction handler.

        Args:
            repo: Database repository
            config: Phase configuration
        """
        self.repo = repo
        self.config = config or PredictionPhaseConfig()

        self._held_out_manager = HeldOutSetManager(repo=repo)
        self._adversarial_generator = AdversarialPredictionGenerator()
        self._verifier = PredictionVerifier(repo=repo)

        self._predictor_fn: Callable[[str, int], str] | None = None
        self._held_out_set: PredictionTestSet | None = None
        self._regression_set: PredictionTestSet | None = None

    @property
    def phase(self) -> Phase:
        return Phase.PREDICTION

    def set_predictor(self, predictor_fn: Callable[[str, int], str]) -> None:
        """Set the predictor function to test.

        The predictor should take (initial_state, horizon) and return
        the predicted state at t=horizon.

        Args:
            predictor_fn: Prediction function to test
        """
        self._predictor_fn = predictor_fn
        self._adversarial_generator.set_predictor(predictor_fn)

    def execute(self, context: PhaseContext) -> IterationResult:
        """Execute one iteration of the prediction phase.

        Steps:
        1. Initialize test sets if needed
        2. Generate fresh adversarial set (if configured)
        3. Run predictions on all test sets
        4. Compute accuracy metrics
        5. Generate control block with recommendation

        Args:
            context: Phase execution context

        Returns:
            IterationResult with metrics and control block
        """
        run_id = context.run_id
        iteration_index = context.iteration_index

        if self._predictor_fn is None:
            # No predictor set - return error state
            return self._error_result(
                run_id,
                iteration_index,
                "No predictor function set. Call set_predictor() first.",
            )

        logger.info(f"Prediction phase iteration {iteration_index}")

        # Step 1: Initialize test sets if needed
        if self._held_out_set is None:
            self._held_out_set = self._held_out_manager.create_held_out_set(
                run_id=run_id,
                seed=42,  # Fixed seed for reproducibility
                count=self.config.held_out_count,
                horizons=self.config.horizons,
                grid_lengths=self.config.grid_lengths,
            )
            logger.info(f"Created held-out set with {self._held_out_set.case_count} cases")

        if self._regression_set is None:
            self._regression_set = self._held_out_manager.create_regression_set(
                run_id=run_id,
            )
            logger.info(f"Created regression set with {self._regression_set.case_count} cases")

        # Step 2: Generate adversarial set
        adversarial_seed = iteration_index * 1000  # Different seed each iteration
        adversarial_set = self._adversarial_generator.generate_adversarial_set(
            run_id=run_id,
            seed=adversarial_seed,
            count=self.config.adversarial_count,
            horizons=self.config.horizons,
            grid_lengths=self.config.grid_lengths,
        )
        logger.info(f"Generated adversarial set with {adversarial_set.case_count} cases")

        # Step 3: Run verification on all sets
        held_out_report = self._verifier.verify_test_set(
            self._held_out_set, self._predictor_fn, run_id
        )
        adversarial_report = self._verifier.verify_test_set(
            adversarial_set, self._predictor_fn, run_id
        )
        regression_report = self._verifier.verify_test_set(
            self._regression_set, self._predictor_fn, run_id
        )

        logger.info(
            f"Verification results - "
            f"Held-out: {held_out_report.exact_match_rate:.2%}, "
            f"Adversarial: {adversarial_report.exact_match_rate:.2%}, "
            f"Regression: {regression_report.exact_match_rate:.2%}"
        )

        # Step 4: Record weak patterns for future adversarial generation
        for result in adversarial_report.worst_cases:
            if result.hamming_distance > 0:
                self._adversarial_generator.record_weak_pattern(result.initial_state)

        # Step 5: Compute readiness metrics
        readiness = self._compute_readiness(
            held_out_report, adversarial_report, regression_report
        )

        # Step 6: Generate control block
        control_block = self._generate_control_block(
            held_out_report, adversarial_report, regression_report, readiness
        )

        # Build iteration result
        result = IterationResult(
            control_block=control_block,
            readiness_metrics=readiness,
        )
        # Store summary in control_block for access
        result.summary = {
            "held_out_exact_match": held_out_report.exact_match_rate,
            "held_out_cell_accuracy": held_out_report.mean_cell_accuracy,
            "adversarial_exact_match": adversarial_report.exact_match_rate,
            "adversarial_cell_accuracy": adversarial_report.mean_cell_accuracy,
            "regression_exact_match": regression_report.exact_match_rate,
        }
        return result

    def _compute_readiness(
        self,
        held_out_report: VerificationReport,
        adversarial_report: VerificationReport,
        regression_report: VerificationReport,
    ) -> ReadinessMetrics:
        """Compute readiness metrics for the prediction phase."""
        metrics = ReadinessMetrics()

        # Map verification results to readiness metrics
        metrics.s_held_out_accuracy = held_out_report.exact_match_rate
        metrics.s_adversarial_accuracy = adversarial_report.exact_match_rate

        # Use cell accuracy as a consistency measure
        avg_cell_accuracy = (
            held_out_report.mean_cell_accuracy
            + adversarial_report.mean_cell_accuracy
            + regression_report.mean_cell_accuracy
        ) / 3
        metrics.s_consistency = avg_cell_accuracy

        # Compute combined prediction readiness
        metrics.compute_prediction_readiness()

        return metrics

    def _generate_control_block(
        self,
        held_out_report: VerificationReport,
        adversarial_report: VerificationReport,
        regression_report: VerificationReport,
        readiness: ReadinessMetrics,
    ) -> ControlBlock:
        """Generate control block based on verification results."""
        thresholds = self.config.thresholds or AccuracyThresholds()

        # Check each report against thresholds
        held_out_passes, held_out_reason = thresholds.check_report(held_out_report)
        adversarial_passes, adversarial_reason = thresholds.check_report(adversarial_report)
        regression_passes, regression_reason = thresholds.check_report(regression_report)

        all_pass = held_out_passes and adversarial_passes and regression_passes

        # Build evidence references
        evidence: list[EvidenceReference] = []
        evidence.append(EvidenceReference(
            artifact_type="verification_report",
            artifact_id=held_out_report.set_id,
            role="supports",
            note=f"Held-out: {held_out_report.exact_match_rate:.2%} exact match",
        ))
        evidence.append(EvidenceReference(
            artifact_type="verification_report",
            artifact_id=adversarial_report.set_id,
            role="supports",
            note=f"Adversarial: {adversarial_report.exact_match_rate:.2%} exact match",
        ))
        evidence.append(EvidenceReference(
            artifact_type="verification_report",
            artifact_id=regression_report.set_id,
            role="supports",
            note=f"Regression: {regression_report.exact_match_rate:.2%} exact match",
        ))

        # Determine recommendation
        if all_pass:
            recommendation = PhaseRecommendation.ADVANCE
            stop_reason = StopReason.SATURATION
            justification = "All prediction accuracy thresholds met"
        else:
            recommendation = PhaseRecommendation.STAY
            stop_reason = StopReason.CONTINUING

            # Build justification from failures
            failures = []
            if not held_out_passes:
                failures.append(held_out_reason)
            if not adversarial_passes:
                failures.append(adversarial_reason)
            if not regression_passes:
                failures.append(regression_reason)
            justification = "; ".join(failures)

        # Check for severe regression (may need to retreat)
        requests: list[PhaseRequest] = []
        if regression_report.exact_match_rate < 0.8:
            # Significant regression - may need to revisit explanation
            recommendation = PhaseRecommendation.RETREAT
            stop_reason = StopReason.GAPS_IDENTIFIED
            requests.append(PhaseRequest(
                request_type="refine_explanation",
                target_id=None,
                description=f"Regression failures indicate explanation may be incorrect",
                priority="high",
            ))

        # Add requests for worst-case analysis
        if not all_pass and adversarial_report.worst_cases:
            for worst in adversarial_report.worst_cases[:3]:
                requests.append(PhaseRequest(
                    request_type="analyze_failure",
                    target_id=worst.case_id,
                    description=f"Prediction failed on {worst.initial_state} (h={worst.hamming_distance})",
                    priority="medium",
                ))

        return ControlBlock(
            readiness_score_suggestion=int(readiness.combined_score * 100),
            readiness_justification=justification,
            phase_recommendation=recommendation,
            stop_reason=stop_reason,
            evidence=evidence,
            requests=requests,
        )

    def _error_result(
        self,
        run_id: str,
        iteration_index: int,
        error_message: str,
    ) -> IterationResult:
        """Create an error iteration result."""
        readiness = ReadinessMetrics()
        readiness.compute_prediction_readiness()

        control_block = ControlBlock(
            readiness_score_suggestion=0,
            readiness_justification=error_message,
            phase_recommendation=PhaseRecommendation.STAY,
            stop_reason=StopReason.CONTINUING,
        )

        result = IterationResult(
            control_block=control_block,
            readiness_metrics=readiness,
        )
        result.summary = {"error": error_message}
        return result

    def run_iteration(self, context: PhaseContext) -> ControlBlock:
        """Execute one iteration and return control block.

        This implements the PhaseHandler protocol by wrapping execute().

        Args:
            context: Phase execution context

        Returns:
            ControlBlock with phase outputs
        """
        result = self.execute(context)
        return result.control_block

    def can_handle_requests(self, requests: list) -> bool:
        """Check if this handler can process the given requests.

        Prediction phase can handle:
        - analyze_failure: Analyze prediction failures
        - regenerate_tests: Regenerate test sets

        Args:
            requests: List of PhaseRequests from other phases

        Returns:
            True if any request is handleable
        """
        handleable_types = {"analyze_failure", "regenerate_tests"}
        return any(r.request_type in handleable_types for r in requests)

    def reset(self) -> None:
        """Reset handler state for a new run."""
        self._held_out_set = None
        self._regression_set = None
        self._adversarial_generator = AdversarialPredictionGenerator()
        if self._predictor_fn:
            self._adversarial_generator.set_predictor(self._predictor_fn)
