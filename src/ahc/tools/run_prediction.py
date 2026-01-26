"""Tool for running predictions against the simulator."""

import logging
from typing import Any

from src.ahc.tools.base import BaseTool, ToolParameter, ToolResult
from src.universe.simulator import step, run
from src.universe.observables import (
    count_symbol,
    grid_length,
    neighbor_config,
    count_pattern,
    index_parity,
    incoming_collisions,
    particle_count,
    momentum,
    free_movers,
    right_component,
    left_component,
)

logger = logging.getLogger(__name__)


class RunPredictionTool(BaseTool):
    """Tool for running the simulator and making predictions.

    Provides direct access to the simulator's step() function,
    allowing the agent to observe state transitions and test
    its understanding of the physics.
    """

    @property
    def name(self) -> str:
        return "run_prediction"

    @property
    def description(self) -> str:
        return """Run the physics simulator on a state and observe the result.

Use this tool to:
1. Observe state transitions (what happens after one step)
2. Test your predictions (compare your prediction to actual result)
3. Build understanding of the physics rules

You can optionally provide your predicted next state to check if you're correct.

The state string uses these symbols:
- '.' = empty cell
- '>' = right-moving particle
- '<' = left-moving particle
- 'X' = collision (two particles in same cell)

Returns the actual next state plus rich observables about both states."""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="state",
                type="string",
                description="The current state string (e.g., '>..<' or '..>.<..')",
                required=True,
            ),
            ToolParameter(
                name="steps",
                type="integer",
                description="Number of time steps to simulate (default: 1)",
                required=False,
                default=1,
            ),
            ToolParameter(
                name="predicted_next_state",
                type="string",
                description="Your predicted next state (optional). If provided, we'll tell you if you were correct.",
                required=False,
            ),
        ]

    def execute(
        self,
        state: str,
        steps: int = 1,
        predicted_next_state: str | None = None,
        **kwargs,
    ) -> ToolResult:
        """Execute the simulation.

        Args:
            state: Initial state string
            steps: Number of steps to simulate
            predicted_next_state: Optional prediction to verify

        Returns:
            ToolResult with simulation results
        """
        try:
            # Coerce types (Gemini may return floats)
            steps = int(steps)

            # Validate state
            valid_chars = {'.', '>', '<', 'X'}
            if not state or not all(c in valid_chars for c in state):
                return ToolResult.fail(
                    f"Invalid state: must contain only '.', '>', '<', 'X'. Got: '{state}'"
                )

            # Run simulation
            if steps == 1:
                next_state = step(state)
                trajectory = [state, next_state]
            else:
                trajectory = run(state, steps)
                next_state = trajectory[-1]

            # Compute observables for initial state
            initial_obs = self._compute_observables(state)

            # Compute observables for final state
            final_obs = self._compute_observables(next_state)

            # Build result
            result_data = {
                "initial_state": state,
                "final_state": next_state,
                "steps": steps,
                "trajectory": trajectory if steps <= 10 else [trajectory[0], "...", trajectory[-1]],
                "initial_observables": initial_obs,
                "final_observables": final_obs,
            }

            # Check prediction if provided
            if predicted_next_state is not None:
                is_correct = predicted_next_state == next_state
                result_data["prediction"] = {
                    "predicted": predicted_next_state,
                    "actual": next_state,
                    "correct": is_correct,
                }
                if not is_correct:
                    # Provide helpful diff info
                    diff_positions = []
                    for i, (p, a) in enumerate(zip(predicted_next_state, next_state)):
                        if p != a:
                            diff_positions.append({
                                "position": i,
                                "predicted": p,
                                "actual": a,
                                "neighbor_at_t0": neighbor_config(state, i) if i < len(state) else "N/A",
                            })
                    result_data["prediction"]["differences"] = diff_positions

            # Add local transition data for each cell
            local_transitions = []
            for i in range(len(state)):
                local_transitions.append({
                    "position": i,
                    "symbol_at_t": state[i],
                    "symbol_at_t1": next_state[i] if i < len(next_state) else "?",
                    "neighbor_config_at_t": neighbor_config(state, i),
                    "parity": "even" if index_parity(i) == 0 else "odd",
                })
            result_data["local_transitions"] = local_transitions

            return ToolResult.ok(data=result_data)

        except Exception as e:
            logger.exception(f"Simulation failed for state: {state}")
            return ToolResult.fail(f"Simulation failed: {str(e)}")

    def _compute_observables(self, state: str) -> dict[str, Any]:
        """Compute rich observables for a state.

        Returns a dictionary of observable names to values.
        """
        observables = {
            "length": grid_length(state),
            "count_empty": count_symbol(state, '.'),
            "count_right": count_symbol(state, '>'),
            "count_left": count_symbol(state, '<'),
            "count_collision": count_symbol(state, 'X'),
            "particle_count": particle_count(state),
            "momentum": momentum(state),
            "free_movers": free_movers(state),
            "right_component": right_component(state),
            "left_component": left_component(state),
            "incoming_collisions": incoming_collisions(state),
        }

        # Add some pattern counts that might be useful
        interesting_patterns = [">.<", "<.>", "><", "<>", ".X.", "X.X"]
        for pattern in interesting_patterns:
            try:
                observables[f"pattern_{pattern}"] = count_pattern(state, pattern)
            except ValueError:
                pass  # Skip invalid patterns

        return observables
