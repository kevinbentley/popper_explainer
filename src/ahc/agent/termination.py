"""Termination checking for AHC-DS agent."""

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from src.ahc.db.models import PredictionRecord
from src.ahc.db.repo import AHCRepository
from src.universe.simulator import step

logger = logging.getLogger(__name__)


@dataclass
class TerminationStatus:
    """Status of termination conditions.

    Attributes:
        terminated: Whether termination conditions are met
        accuracy: Current prediction accuracy
        accuracy_target: Target accuracy (default: 1.0 = 100%)
        predictions_count: Number of predictions made
        predictions_target: Target number of predictions (default: 5000)
        transition_complete: Whether transition rules are complete
        reason: Explanation of termination status
    """
    terminated: bool = False
    accuracy: float = 0.0
    accuracy_target: float = 1.0
    predictions_count: int = 0
    predictions_target: int = 5000
    transition_complete: bool = False
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class TerminationChecker:
    """Checks termination conditions for the AHC agent.

    Termination occurs when BOTH are satisfied:
    1. Predictive Accuracy: 100% accuracy on 5,000 randomized states
    2. Symbolic Completeness: Complete local transition function

    The agent can request a final validation run to test termination.
    """

    def __init__(
        self,
        repo: AHCRepository,
        session_id: int,
        accuracy_target: float = 1.0,
        predictions_target: int = 5000,
    ):
        """Initialize the termination checker.

        Args:
            repo: Database repository
            session_id: Session ID to check
            accuracy_target: Target accuracy (default: 1.0)
            predictions_target: Number of predictions required
        """
        self._repo = repo
        self._session_id = session_id
        self._accuracy_target = accuracy_target
        self._predictions_target = predictions_target

    def check(self) -> TerminationStatus:
        """Check current progress toward termination.

        This is a quick check that uses existing predictions
        from the database.

        Returns:
            TerminationStatus with current progress
        """
        # Get accuracy stats
        accuracy_stats = self._repo.get_accuracy_stats(self._session_id)

        # Get transition rule completeness
        transition_completeness = self._repo.get_transition_completeness(self._session_id)

        # Build status
        status = TerminationStatus(
            accuracy=accuracy_stats["accuracy"],
            accuracy_target=self._accuracy_target,
            predictions_count=accuracy_stats["total_predictions"],
            predictions_target=self._predictions_target,
            transition_complete=transition_completeness.get("is_complete", False),
            details={
                "accuracy_stats": accuracy_stats,
                "transition_completeness": transition_completeness,
            },
        )

        # Check termination conditions
        accuracy_met = (
            status.predictions_count >= self._predictions_target
            and status.accuracy >= self._accuracy_target
        )

        if accuracy_met and status.transition_complete:
            status.terminated = True
            status.reason = "Both accuracy and transition completeness targets met"
        elif accuracy_met:
            status.reason = f"Accuracy target met ({status.accuracy:.2%}), but transition rules incomplete"
        elif status.transition_complete:
            status.reason = f"Transition rules complete, but need more predictions ({status.predictions_count}/{self._predictions_target})"
        else:
            status.reason = f"Progress: {status.accuracy:.2%} accuracy on {status.predictions_count} predictions"

        return status

    def run_final_validation(
        self,
        agent_predict_fn,
        turn_number: int,
        seed: int = 42,
    ) -> TerminationStatus:
        """Run final validation with 5000 random states.

        This is the definitive termination check. It generates
        5000 random states, asks the agent to predict the next
        state for each, and checks accuracy.

        Args:
            agent_predict_fn: Function that takes a state and returns predicted next state
            turn_number: Current turn number for logging
            seed: Random seed for reproducibility

        Returns:
            TerminationStatus with validation results
        """
        logger.info(f"Running final validation with {self._predictions_target} states")

        random.seed(seed)
        correct = 0
        incorrect_examples = []

        for i in range(self._predictions_target):
            # Generate random state
            length = random.randint(4, 20)
            state = self._generate_random_state(length)

            # Get actual next state
            actual_next = step(state)

            # Get agent's prediction
            try:
                predicted_next = agent_predict_fn(state)
            except Exception as e:
                logger.warning(f"Prediction failed for state '{state}': {e}")
                predicted_next = None

            # Check correctness
            is_correct = predicted_next == actual_next

            if is_correct:
                correct += 1
            else:
                if len(incorrect_examples) < 10:  # Keep first 10 examples
                    incorrect_examples.append({
                        "state": state,
                        "predicted": predicted_next,
                        "actual": actual_next,
                    })

            # Log prediction to database
            prediction = PredictionRecord(
                session_id=self._session_id,
                turn_number=turn_number,
                state_t0=state,
                predicted_state_t1=predicted_next or "",
                actual_state_t1=actual_next,
                is_correct=is_correct,
                prediction_method="final_validation",
            )
            self._repo.insert_prediction(prediction)

            # Progress logging
            if (i + 1) % 500 == 0:
                logger.info(f"Validation progress: {i + 1}/{self._predictions_target}, accuracy: {correct / (i + 1):.2%}")

        # Compute final accuracy
        accuracy = correct / self._predictions_target

        # Get transition completeness
        transition_completeness = self._repo.get_transition_completeness(self._session_id)

        # Build final status
        status = TerminationStatus(
            accuracy=accuracy,
            accuracy_target=self._accuracy_target,
            predictions_count=self._predictions_target,
            predictions_target=self._predictions_target,
            transition_complete=transition_completeness.get("is_complete", False),
            details={
                "correct": correct,
                "total": self._predictions_target,
                "incorrect_examples": incorrect_examples,
                "transition_completeness": transition_completeness,
            },
        )

        # Check if terminated
        if accuracy >= self._accuracy_target and status.transition_complete:
            status.terminated = True
            status.reason = f"TERMINATED: {accuracy:.2%} accuracy on {self._predictions_target} states with complete transition rules"
        else:
            status.reason = f"NOT TERMINATED: {accuracy:.2%} accuracy ({correct}/{self._predictions_target} correct)"

        logger.info(f"Final validation complete: {status.reason}")

        return status

    def _generate_random_state(self, length: int) -> str:
        """Generate a random valid state.

        Args:
            length: State length

        Returns:
            Random state string
        """
        symbols = ['.', '>', '<']
        weights = [0.5, 0.25, 0.25]  # Bias toward empty cells

        state = random.choices(symbols, weights=weights, k=length)
        return "".join(state)


def create_mock_predictor(rules: dict[str, str] | None = None):
    """Create a mock predictor function for testing.

    Args:
        rules: Optional mapping from state to predicted next state

    Returns:
        Predictor function
    """
    rules = rules or {}

    def predict(state: str) -> str:
        if state in rules:
            return rules[state]
        # Default: use actual simulator (cheating for testing)
        return step(state)

    return predict
