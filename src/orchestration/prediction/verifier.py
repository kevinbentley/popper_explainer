"""Prediction verification and scoring.

Verifies predictions against ground truth computed by the simulator,
computing accuracy metrics for phase transition decisions.

Scoring metrics:
- Exact match: Binary match/mismatch
- Hamming distance: Number of differing cells
- Cell accuracy: Fraction of cells correct
- Observable errors: Per-observable prediction errors
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.orchestration.prediction.test_sets import PredictionTestCase, PredictionTestSet
from src.universe.types import State

if TYPE_CHECKING:
    from src.db.repo import Repository


@dataclass
class PredictionResult:
    """Result of a single prediction evaluation.

    Attributes:
        case_id: Test case identifier
        initial_state: Input state
        horizon: Prediction horizon
        predicted_state: What was predicted
        expected_state: Ground truth from simulator
        is_exact_match: Whether prediction exactly matches
        hamming_distance: Number of differing cells
        cell_accuracy: Fraction of cells correct (0-1)
        observable_errors: Per-observable error details
    """

    case_id: str
    initial_state: State
    horizon: int
    predicted_state: State
    expected_state: State
    is_exact_match: bool
    hamming_distance: int
    cell_accuracy: float
    observable_errors: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "case_id": self.case_id,
            "initial_state": self.initial_state,
            "horizon": self.horizon,
            "predicted_state": self.predicted_state,
            "expected_state": self.expected_state,
            "is_exact_match": self.is_exact_match,
            "hamming_distance": self.hamming_distance,
            "cell_accuracy": self.cell_accuracy,
            "observable_errors": self.observable_errors,
        }


@dataclass
class VerificationReport:
    """Aggregate report for a test set verification.

    Attributes:
        set_id: Test set identifier
        set_type: Type of test set ('held_out', 'adversarial', 'regression')
        total_cases: Number of cases evaluated
        exact_matches: Number of exact matches
        exact_match_rate: Fraction with exact match
        mean_hamming_distance: Average Hamming distance
        mean_cell_accuracy: Average cell accuracy
        results: Individual prediction results
        worst_cases: Cases with highest error
    """

    set_id: str
    set_type: str
    total_cases: int
    exact_matches: int
    exact_match_rate: float
    mean_hamming_distance: float
    mean_cell_accuracy: float
    results: list[PredictionResult] = field(default_factory=list)
    worst_cases: list[PredictionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "set_id": self.set_id,
            "set_type": self.set_type,
            "total_cases": self.total_cases,
            "exact_matches": self.exact_matches,
            "exact_match_rate": self.exact_match_rate,
            "mean_hamming_distance": self.mean_hamming_distance,
            "mean_cell_accuracy": self.mean_cell_accuracy,
            "worst_cases": [r.to_dict() for r in self.worst_cases],
        }


class PredictionVerifier:
    """Verifies predictions against ground truth.

    The verifier takes predictions and compares them to the expected
    states computed by the simulator, generating accuracy metrics
    for phase transition decisions.
    """

    def __init__(self, repo: Repository | None = None):
        """Initialize the verifier.

        Args:
            repo: Database repository for persistence
        """
        self.repo = repo

    def verify_prediction(
        self,
        case: PredictionTestCase,
        predicted_state: State,
    ) -> PredictionResult:
        """Verify a single prediction.

        Args:
            case: Test case with expected state
            predicted_state: The prediction to verify

        Returns:
            PredictionResult with metrics
        """
        expected = case.expected_state
        if expected is None:
            raise ValueError(f"Test case {case.case_id} has no expected state")

        # Compute metrics
        is_exact = predicted_state == expected
        hamming = self._hamming_distance(predicted_state, expected)
        cell_acc = self._cell_accuracy(predicted_state, expected)
        observable_errors = self._compute_observable_errors(predicted_state, expected)

        return PredictionResult(
            case_id=case.case_id or "",
            initial_state=case.initial_state,
            horizon=case.horizon,
            predicted_state=predicted_state,
            expected_state=expected,
            is_exact_match=is_exact,
            hamming_distance=hamming,
            cell_accuracy=cell_acc,
            observable_errors=observable_errors,
        )

    def verify_test_set(
        self,
        test_set: PredictionTestSet,
        predictor_fn,
        run_id: str | None = None,
    ) -> VerificationReport:
        """Verify predictions for an entire test set.

        Args:
            test_set: Test set to verify
            predictor_fn: Function (state, horizon) -> predicted_state
            run_id: Orchestration run ID for persistence

        Returns:
            VerificationReport with aggregate metrics
        """
        results: list[PredictionResult] = []

        for case in test_set.cases:
            try:
                # Get prediction
                predicted = predictor_fn(case.initial_state, case.horizon)
                result = self.verify_prediction(case, predicted)
                results.append(result)

                # Persist individual result
                if self.repo and run_id:
                    self._persist_result(run_id, test_set.set_type, result)

            except Exception as e:
                # Create a failure result
                result = PredictionResult(
                    case_id=case.case_id or "",
                    initial_state=case.initial_state,
                    horizon=case.horizon,
                    predicted_state="<error>",
                    expected_state=case.expected_state or "",
                    is_exact_match=False,
                    hamming_distance=len(case.initial_state),  # Worst case
                    cell_accuracy=0.0,
                    observable_errors={"error": str(e)},
                )
                results.append(result)

        # Compute aggregates
        total = len(results)
        exact_matches = sum(1 for r in results if r.is_exact_match)
        exact_match_rate = exact_matches / total if total > 0 else 0.0
        mean_hamming = sum(r.hamming_distance for r in results) / total if total > 0 else 0.0
        mean_cell_acc = sum(r.cell_accuracy for r in results) / total if total > 0 else 0.0

        # Find worst cases (by Hamming distance)
        sorted_results = sorted(results, key=lambda r: r.hamming_distance, reverse=True)
        worst_cases = sorted_results[:5]

        report = VerificationReport(
            set_id=test_set.set_id,
            set_type=test_set.set_type,
            total_cases=total,
            exact_matches=exact_matches,
            exact_match_rate=exact_match_rate,
            mean_hamming_distance=mean_hamming,
            mean_cell_accuracy=mean_cell_acc,
            results=results,
            worst_cases=worst_cases,
        )

        return report

    def _hamming_distance(self, predicted: State, expected: State) -> int:
        """Compute Hamming distance between two states.

        If lengths differ, the difference in length is added to the count.
        """
        min_len = min(len(predicted), len(expected))
        max_len = max(len(predicted), len(expected))

        mismatches = sum(1 for i in range(min_len) if predicted[i] != expected[i])
        length_diff = max_len - min_len

        return mismatches + length_diff

    def _cell_accuracy(self, predicted: State, expected: State) -> float:
        """Compute cell-level accuracy.

        Returns fraction of cells that match exactly.
        """
        if len(expected) == 0:
            return 1.0 if len(predicted) == 0 else 0.0

        min_len = min(len(predicted), len(expected))
        max_len = max(len(predicted), len(expected))

        matches = sum(1 for i in range(min_len) if predicted[i] == expected[i])

        return matches / max_len

    def _compute_observable_errors(
        self,
        predicted: State,
        expected: State,
    ) -> dict[str, Any]:
        """Compute errors for derived observables.

        Observables:
        - particle_count: Total particles (> + < + 2*X)
        - right_count: Number of >
        - left_count: Number of <
        - collision_count: Number of X
        - collision_positions: Where collisions occur
        """
        from src.universe.types import Symbol

        def count_observables(state: State) -> dict[str, Any]:
            rights = state.count(Symbol.RIGHT.value)
            lefts = state.count(Symbol.LEFT.value)
            collisions = state.count(Symbol.COLLISION.value)
            collision_positions = [i for i, c in enumerate(state) if c == Symbol.COLLISION.value]

            return {
                "particle_count": rights + lefts + 2 * collisions,
                "right_count": rights,
                "left_count": lefts,
                "collision_count": collisions,
                "collision_positions": collision_positions,
            }

        pred_obs = count_observables(predicted)
        exp_obs = count_observables(expected)

        errors = {}
        for key in pred_obs:
            if pred_obs[key] != exp_obs[key]:
                errors[key] = {
                    "predicted": pred_obs[key],
                    "expected": exp_obs[key],
                    "difference": (
                        pred_obs[key] - exp_obs[key]
                        if isinstance(pred_obs[key], (int, float))
                        else None
                    ),
                }

        return errors

    def _persist_result(
        self,
        run_id: str,
        evaluation_set: str,
        result: PredictionResult,
    ) -> None:
        """Persist a prediction evaluation result."""
        if not self.repo:
            return

        from src.db.orchestration_models import PredictionEvaluationRecord

        record = PredictionEvaluationRecord(
            prediction_id=result.case_id,
            run_id=run_id,
            actual_state=result.expected_state,
            is_exact_match=result.is_exact_match,
            evaluation_set=evaluation_set,
            hamming_distance=result.hamming_distance,
            cell_accuracy=result.cell_accuracy,
            observable_errors_json=json.dumps(result.observable_errors),
        )
        self.repo.insert_prediction_evaluation(record)


class AccuracyThresholds:
    """Thresholds for prediction accuracy decisions.

    These thresholds determine when prediction quality is sufficient
    to advance to the FINALIZE phase.
    """

    def __init__(
        self,
        held_out_exact_match: float = 0.90,
        held_out_cell_accuracy: float = 0.95,
        adversarial_exact_match: float = 0.70,
        adversarial_cell_accuracy: float = 0.85,
        regression_exact_match: float = 1.0,
    ):
        """Initialize thresholds.

        Args:
            held_out_exact_match: Required exact match rate on held-out set
            held_out_cell_accuracy: Required cell accuracy on held-out set
            adversarial_exact_match: Required exact match rate on adversarial set
            adversarial_cell_accuracy: Required cell accuracy on adversarial set
            regression_exact_match: Required exact match rate on regression set
        """
        self.held_out_exact_match = held_out_exact_match
        self.held_out_cell_accuracy = held_out_cell_accuracy
        self.adversarial_exact_match = adversarial_exact_match
        self.adversarial_cell_accuracy = adversarial_cell_accuracy
        self.regression_exact_match = regression_exact_match

    def check_report(self, report: VerificationReport) -> tuple[bool, str]:
        """Check if a report meets thresholds.

        Args:
            report: Verification report to check

        Returns:
            Tuple of (passes, reason)
        """
        if report.set_type == "held_out":
            if report.exact_match_rate < self.held_out_exact_match:
                return False, f"Held-out exact match {report.exact_match_rate:.2%} < {self.held_out_exact_match:.0%}"
            if report.mean_cell_accuracy < self.held_out_cell_accuracy:
                return False, f"Held-out cell accuracy {report.mean_cell_accuracy:.2%} < {self.held_out_cell_accuracy:.0%}"

        elif report.set_type == "adversarial":
            if report.exact_match_rate < self.adversarial_exact_match:
                return False, f"Adversarial exact match {report.exact_match_rate:.2%} < {self.adversarial_exact_match:.0%}"
            if report.mean_cell_accuracy < self.adversarial_cell_accuracy:
                return False, f"Adversarial cell accuracy {report.mean_cell_accuracy:.2%} < {self.adversarial_cell_accuracy:.0%}"

        elif report.set_type == "regression":
            if report.exact_match_rate < self.regression_exact_match:
                return False, f"Regression exact match {report.exact_match_rate:.2%} < {self.regression_exact_match:.0%}"

        return True, "Meets thresholds"
