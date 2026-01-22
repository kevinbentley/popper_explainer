"""Core simulator for the Kinetic Grid Universe.

This implements the deterministic evolution rules defined in UNIVERSE.md:
- Particles (> <) move at speed 1
- Collisions (X) form when > and < meet
- Collisions resolve in one step to <.>
- Periodic boundary conditions
"""

import hashlib
from typing import Final

from src.universe.types import Config, State, Symbol, Trajectory
from src.universe.validation import validate_state

# Simulator version for reproducibility tracking
_SIMULATOR_VERSION: Final[str] = "kinetic_grid_v1.0.0"


def version_hash() -> str:
    """Return a hash identifying this simulator version.

    Used for reproducibility: evaluations record this hash so results
    can be traced to a specific simulator implementation.
    """
    # Hash the version string and this file's core logic
    content = f"{_SIMULATOR_VERSION}:{__file__}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def step(state: State) -> State:
    """Advance the universe by one time step.

    Evolution rules (from UNIVERSE.md):
    1. Movement Phase: Each particle attempts to move simultaneously
       - > moves one cell to the right
       - < moves one cell to the left
       - Movement wraps at boundaries (periodic)

    2. Collision Formation: If > and < attempt to occupy the same cell,
       they form an X at that cell

    3. Collision Resolution: Every X resolves in the next step:
       - The > exits to the right
       - The < exits to the left
       - The original cell becomes empty

    Args:
        state: Current state string

    Returns:
        Next state string after one time step

    Raises:
        InvalidStateError: If state is invalid
    """
    validate_state(state)
    n = len(state)

    if n == 0:
        return state

    # Track what arrives at each cell
    # Each cell can receive: rights moving in, lefts moving in
    rights_arriving = [0] * n  # Count of > arriving at each cell
    lefts_arriving = [0] * n   # Count of < arriving at each cell

    # Process each cell
    for i, cell in enumerate(state):
        if cell == Symbol.RIGHT.value:
            # > moves right (wrapping)
            dest = (i + 1) % n
            rights_arriving[dest] += 1

        elif cell == Symbol.LEFT.value:
            # < moves left (wrapping)
            dest = (i - 1) % n
            lefts_arriving[dest] += 1

        elif cell == Symbol.COLLISION.value:
            # X resolves: > exits right, < exits left
            right_dest = (i + 1) % n
            left_dest = (i - 1) % n
            rights_arriving[right_dest] += 1
            lefts_arriving[left_dest] += 1

        # Empty cells contribute nothing

    # Build new state from arrivals
    new_state = []
    for i in range(n):
        r = rights_arriving[i]
        l = lefts_arriving[i]

        if r > 0 and l > 0:
            # Collision: one > and one < meet
            # Note: By conservation, we should never have r>1 or l>1
            # but the rules handle it gracefully
            new_state.append(Symbol.COLLISION.value)
        elif r > 0:
            new_state.append(Symbol.RIGHT.value)
        elif l > 0:
            new_state.append(Symbol.LEFT.value)
        else:
            new_state.append(Symbol.EMPTY.value)

    return "".join(new_state)


def run(state: State, t: int, config: Config | None = None) -> Trajectory:
    """Run the simulation for T time steps.

    Args:
        state: Initial state string
        t: Number of time steps to simulate
        config: Optional configuration (used for validation)

    Returns:
        Trajectory as a list of states, where trajectory[0] is the initial
        state and trajectory[T] is the final state (length T+1)

    Raises:
        InvalidStateError: If initial state is invalid
        ValueError: If T is negative
    """
    if t < 0:
        raise ValueError(f"Time steps T must be non-negative, got {t}")

    validate_state(state, config)

    trajectory: Trajectory = [state]
    current = state

    for _ in range(t):
        current = step(current)
        trajectory.append(current)

    return trajectory
