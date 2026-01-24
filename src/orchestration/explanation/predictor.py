"""State predictor based on mechanistic explanations.

The predictor uses the mechanism rules from an explanation
to predict future states of the universe.

Two modes:
1. Rule-based: Apply mechanism rules directly (deterministic)
2. LLM-based: Have LLM predict based on explanation (for complex cases)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from src.orchestration.explanation.models import (
    Explanation,
    Mechanism,
    MechanismType,
)
from src.universe.types import Config, State, Symbol

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MechanismBasedPredictor:
    """Predictor that applies mechanism rules to predict states.

    This predictor interprets the mechanism rules from an explanation
    and applies them to predict future states. It's deterministic
    and directly implements the understood rules.
    """

    def __init__(self, explanation: Explanation | None = None):
        """Initialize the predictor.

        Args:
            explanation: Explanation containing mechanism rules
        """
        self._explanation = explanation
        self._mechanism = explanation.mechanism if explanation else None

    def set_explanation(self, explanation: Explanation) -> None:
        """Set the explanation to use for predictions.

        Args:
            explanation: Explanation with mechanism rules
        """
        self._explanation = explanation
        self._mechanism = explanation.mechanism

    def predict(self, initial_state: State, horizon: int) -> State:
        """Predict the state at t=horizon given initial state.

        Args:
            initial_state: State at t=0
            horizon: Number of steps to predict

        Returns:
            Predicted state at t=horizon
        """
        if self._mechanism is None:
            raise ValueError("No explanation set. Call set_explanation() first.")

        current_state = initial_state
        for _ in range(horizon):
            current_state = self._step(current_state)

        return current_state

    def _step(self, state: State) -> State:
        """Apply mechanism rules for one time step.

        This implements the core prediction logic based on
        the mechanism rules in the explanation.
        """
        n = len(state)
        if n == 0:
            return state

        # Track what arrives at each cell
        rights_arriving = [0] * n
        lefts_arriving = [0] * n

        # Apply movement and transformation rules
        for i, cell in enumerate(state):
            if cell == Symbol.RIGHT.value:
                # Movement rule: > moves right
                dest = (i + 1) % n
                rights_arriving[dest] += 1

            elif cell == Symbol.LEFT.value:
                # Movement rule: < moves left
                dest = (i - 1) % n
                lefts_arriving[dest] += 1

            elif cell == Symbol.COLLISION.value:
                # Transformation rule: X resolves
                right_dest = (i + 1) % n
                left_dest = (i - 1) % n
                rights_arriving[right_dest] += 1
                lefts_arriving[left_dest] += 1

        # Build new state based on arrivals (interaction rules)
        new_state = []
        for i in range(n):
            r = rights_arriving[i]
            l = lefts_arriving[i]

            if r > 0 and l > 0:
                # Interaction rule: collision forms
                new_state.append(Symbol.COLLISION.value)
            elif r > 0:
                new_state.append(Symbol.RIGHT.value)
            elif l > 0:
                new_state.append(Symbol.LEFT.value)
            else:
                new_state.append(Symbol.EMPTY.value)

        return "".join(new_state)

    def get_predictor_function(self) -> Callable[[State, int], State]:
        """Return a predictor function for use with the prediction verifier.

        Returns:
            Function (state, horizon) -> predicted_state
        """
        def predictor_fn(state: State, horizon: int) -> State:
            return self.predict(state, horizon)

        return predictor_fn


class LLMBasedPredictor:
    """Predictor that uses LLM to predict states based on explanation.

    This predictor asks the LLM to predict the next state given
    the current state and the mechanistic explanation.
    """

    def __init__(
        self,
        client: Any,  # LLM client
        explanation: Explanation | None = None,
    ):
        """Initialize the LLM predictor.

        Args:
            client: LLM client for predictions
            explanation: Explanation to use for context
        """
        self.client = client
        self._explanation = explanation

    def set_explanation(self, explanation: Explanation) -> None:
        """Set the explanation to use for predictions."""
        self._explanation = explanation

    def predict(self, initial_state: State, horizon: int) -> State:
        """Predict the state at t=horizon.

        Args:
            initial_state: State at t=0
            horizon: Number of steps to predict

        Returns:
            Predicted state at t=horizon
        """
        if self._explanation is None:
            raise ValueError("No explanation set.")

        # Build prediction prompt
        prompt = self._build_prompt(initial_state, horizon)

        try:
            response = self.client.generate(prompt, temperature=0.0)
            predicted_state = self._parse_response(response, len(initial_state))
            return predicted_state
        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            # Fall back to mechanism-based prediction
            fallback = MechanismBasedPredictor(self._explanation)
            return fallback.predict(initial_state, horizon)

    def _build_prompt(self, initial_state: State, horizon: int) -> str:
        """Build the prediction prompt."""
        return f'''Given the following mechanistic explanation of a 1D particle universe:

{self._explanation.hypothesis_text}

Rules:
{self._format_rules()}

Predict the state after {horizon} time step(s).

Initial state at t=0: {initial_state}

Symbols:
- . = empty cell
- > = right-moving particle
- < = left-moving particle
- X = collision

Output ONLY the predicted state string (same length as input, no explanation):'''

    def _format_rules(self) -> str:
        """Format mechanism rules for the prompt."""
        lines = []
        for rule in self._explanation.mechanism.rules:
            lines.append(f"- {rule.condition} → {rule.effect}")
        return "\n".join(lines)

    def _parse_response(self, response: str, expected_length: int) -> State:
        """Parse LLM response to extract state."""
        # Extract just the state string
        response = response.strip()

        # Find a valid state string in the response
        for line in response.split("\n"):
            line = line.strip()
            if len(line) == expected_length and all(
                c in ".><X" for c in line
            ):
                return line

        # If no valid state found, take the first line-like thing
        for line in response.split():
            if all(c in ".><X" for c in line):
                # Pad or truncate to expected length
                if len(line) < expected_length:
                    line = line + "." * (expected_length - len(line))
                elif len(line) > expected_length:
                    line = line[:expected_length]
                return line

        # Ultimate fallback - return empty state
        return "." * expected_length

    def get_predictor_function(self) -> Callable[[State, int], State]:
        """Return a predictor function."""
        def predictor_fn(state: State, horizon: int) -> State:
            return self.predict(state, horizon)

        return predictor_fn


class HybridPredictor:
    """Hybrid predictor that combines rule-based and LLM approaches.

    Uses rule-based prediction for standard cases and falls back
    to LLM for edge cases or when confidence is low.
    """

    def __init__(
        self,
        explanation: Explanation | None = None,
        llm_client: Any = None,
        use_llm_threshold: float = 0.3,
    ):
        """Initialize hybrid predictor.

        Args:
            explanation: Explanation to use
            llm_client: Optional LLM client for fallback
            use_llm_threshold: Density threshold to trigger LLM
        """
        self._explanation = explanation
        self._mechanism_predictor = MechanismBasedPredictor(explanation)
        self._llm_predictor = (
            LLMBasedPredictor(llm_client, explanation) if llm_client else None
        )
        self._use_llm_threshold = use_llm_threshold

    def set_explanation(self, explanation: Explanation) -> None:
        """Set the explanation."""
        self._explanation = explanation
        self._mechanism_predictor.set_explanation(explanation)
        if self._llm_predictor:
            self._llm_predictor.set_explanation(explanation)

    def predict(self, initial_state: State, horizon: int) -> State:
        """Predict with hybrid approach."""
        # Calculate state complexity
        density = self._calculate_density(initial_state)

        # Use LLM for high-density/complex states if available
        if (
            density > self._use_llm_threshold
            and self._llm_predictor
            and self._explanation
            and self._explanation.confidence < 0.8
        ):
            return self._llm_predictor.predict(initial_state, horizon)

        # Otherwise use rule-based prediction
        return self._mechanism_predictor.predict(initial_state, horizon)

    def _calculate_density(self, state: State) -> float:
        """Calculate particle density of a state."""
        if len(state) == 0:
            return 0.0
        non_empty = sum(1 for c in state if c != Symbol.EMPTY.value)
        return non_empty / len(state)

    def get_predictor_function(self) -> Callable[[State, int], State]:
        """Return a predictor function."""
        def predictor_fn(state: State, horizon: int) -> State:
            return self.predict(state, horizon)

        return predictor_fn


def create_predictor(
    explanation: Explanation,
    llm_client: Any = None,
    mode: str = "mechanism",
) -> Callable[[State, int], State]:
    """Factory function to create a predictor.

    Args:
        explanation: Explanation to use for prediction
        llm_client: Optional LLM client
        mode: 'mechanism', 'llm', or 'hybrid'

    Returns:
        Predictor function (state, horizon) -> predicted_state
    """
    if mode == "llm" and llm_client:
        predictor = LLMBasedPredictor(llm_client, explanation)
    elif mode == "hybrid" and llm_client:
        predictor = HybridPredictor(explanation, llm_client)
    else:
        predictor = MechanismBasedPredictor(explanation)

    return predictor.get_predictor_function()
