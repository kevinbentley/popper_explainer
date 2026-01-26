"""Tool for simulating trajectories in the abstract symbol space.

This tool is the LLM-facing trajectory simulator. It:
- Accepts states in the abstract alphabet (W, A, B, K)
- Returns trajectories in the abstract alphabet
- Uses opaque error messages that do not leak universe physics

The tool enforces the universe contract (K is invalid at t=0) but
describes the rejection as "undefined behavior" to avoid revealing
why K is special.
"""

import logging
from typing import Any

from src.ahc.tools.base import BaseTool, ToolParameter, ToolResult
from src.proposer.scrambler import SymbolScrambler, get_default_scrambler
from src.universe.simulator import run
from src.universe.types import ABSTRACT_SYMBOLS
from src.universe.validation import is_valid_initial_state

logger = logging.getLogger(__name__)

# Maximum trajectory length to prevent abuse
_MAX_TIME_STEPS = 500


class SimulateTrajectoryTool(BaseTool):
    """Simulate a trajectory from an initial state in abstract symbols.

    Takes an initial configuration string using abstract symbols (W, A, B, K)
    and a number of time steps T, then returns the state at each step
    t_0, t_1, ..., t_T.
    """

    def __init__(self, scrambler: SymbolScrambler | None = None):
        self._scrambler = scrambler or get_default_scrambler()

    @property
    def name(self) -> str:
        return "simulate_trajectory"

    @property
    def description(self) -> str:
        return (
            "Simulate the evolution of a configuration over T time steps.\n\n"
            "Input: A string of symbols (W, A, B, K) representing the "
            "initial configuration, and a number of time steps T.\n"
            "Output: The configuration at each step t_0, t_1, ..., t_T.\n\n"
            "Some configurations may be invalid at t=0. If so, the tool "
            "will reject the input."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="state",
                type="string",
                description=(
                    "The initial configuration string using symbols W, A, B, K "
                    "(e.g., 'WWABWWW')"
                ),
                required=True,
            ),
            ToolParameter(
                name="T",
                type="integer",
                description="Number of time steps to simulate (must be >= 0)",
                required=True,
            ),
        ]

    def execute(self, state: str, T: int = 1, **kwargs) -> ToolResult:
        """Execute the trajectory simulation.

        Args:
            state: Initial state in abstract symbols (W, A, B, K)
            T: Number of time steps

        Returns:
            ToolResult with trajectory in abstract symbols
        """
        try:
            T = int(T)
        except (TypeError, ValueError):
            return ToolResult.fail("T must be an integer.")

        # --- Validate alphabet ---
        if not state:
            return ToolResult.fail("State string must not be empty.")

        invalid_chars = [c for c in state if c not in ABSTRACT_SYMBOLS]
        if invalid_chars:
            unique_invalid = sorted(set(invalid_chars))
            return ToolResult.fail(
                f"Invalid symbol(s): {unique_invalid}. "
                f"Valid symbols are: {sorted(ABSTRACT_SYMBOLS)}"
            )

        # --- Validate time steps ---
        if T < 0:
            return ToolResult.fail("T must be non-negative.")
        if T > _MAX_TIME_STEPS:
            return ToolResult.fail(
                f"T exceeds maximum allowed ({_MAX_TIME_STEPS})."
            )

        # --- Translate to physical symbols for simulation ---
        physical_state = self._scrambler.to_physical(state)

        # --- Enforce initial-state constraint (opaque error) ---
        if not is_valid_initial_state(physical_state):
            return ToolResult.fail(
                "Undefined behavior: the input configuration is not "
                "valid at t=0."
            )

        # --- Run simulation ---
        try:
            trajectory = run(physical_state, T)
        except Exception as e:
            logger.exception("Simulation failed for state: %s", state)
            return ToolResult.fail(f"Simulation error: {e}")

        # --- Translate trajectory back to abstract symbols ---
        abstract_trajectory = [
            self._scrambler.to_abstract(s) for s in trajectory
        ]

        return ToolResult.ok(
            data={
                "initial_state": abstract_trajectory[0],
                "T": T,
                "trajectory": abstract_trajectory,
            }
        )
