"""State transforms for the Kinetic Grid Universe.

These transforms are used for:
- Symmetry testing (metamorphic tests)
- Generating related test cases
- Verifying commutation properties
"""

from typing import Callable

from src.universe.types import State, Symbol
from src.universe.validation import validate_state


def mirror_swap(state: State) -> State:
    """Apply full mirror symmetry: reverse grid AND swap directions.

    This is a TRUE symmetry of the universe (commutes with evolution):
        evolve(mirror_swap(S), t) == mirror_swap(evolve(S, t))

    Transform:
        - Reverse the string spatially
        - Swap > ↔ <
        - X stays X (contains one of each)

    Args:
        state: Input state string

    Returns:
        Transformed state string
    """
    validate_state(state)

    # Swap directions
    swapped = []
    for c in state:
        if c == Symbol.RIGHT.value:
            swapped.append(Symbol.LEFT.value)
        elif c == Symbol.LEFT.value:
            swapped.append(Symbol.RIGHT.value)
        else:
            swapped.append(c)

    # Reverse spatially
    return "".join(reversed(swapped))


def mirror_only(state: State) -> State:
    """Apply spatial mirror only (reverse grid, keep directions).

    This is NOT a symmetry of the universe - it does not commute with evolution.
    Useful for testing that false symmetries are correctly rejected.

    Args:
        state: Input state string

    Returns:
        Transformed state string
    """
    validate_state(state)
    return state[::-1]


def swap_only(state: State) -> State:
    """Swap particle directions only (no spatial mirror).

    This is NOT a symmetry of the universe - it does not commute with evolution.
    Useful for testing that false symmetries are correctly rejected.

    Args:
        state: Input state string

    Returns:
        Transformed state string
    """
    validate_state(state)

    result = []
    for c in state:
        if c == Symbol.RIGHT.value:
            result.append(Symbol.LEFT.value)
        elif c == Symbol.LEFT.value:
            result.append(Symbol.RIGHT.value)
        else:
            result.append(c)

    return "".join(result)


def shift_k(state: State, k: int) -> State:
    """Shift the state by k positions (cyclic rotation).

    This is a TRUE symmetry of the universe due to periodic boundaries:
        evolve(shift_k(S, k), t) == shift_k(evolve(S, t), k)

    Args:
        state: Input state string
        k: Number of positions to shift (positive = right, negative = left)

    Returns:
        Transformed state string
    """
    validate_state(state)

    if len(state) == 0:
        return state

    # Normalize k to be within bounds
    n = len(state)
    k = k % n

    if k == 0:
        return state

    # Shift right by k means taking last k chars and moving to front
    return state[-k:] + state[:-k]


# Type alias for transform functions
TransformFunc = Callable[[State], State]

# Registry of available transforms for capability checking
TRANSFORMS: dict[str, TransformFunc] = {
    "mirror_swap": mirror_swap,
    "mirror_only": mirror_only,
    "swap_only": swap_only,
    "shift_k": shift_k,  # Note: shift_k takes extra arg, handle specially
}


def get_transform(name: str) -> TransformFunc | None:
    """Get a transform function by name.

    Args:
        name: Transform name (e.g., "mirror_swap")

    Returns:
        Transform function or None if not found
    """
    return TRANSFORMS.get(name)


def list_transforms() -> list[str]:
    """List all available transform names."""
    return list(TRANSFORMS.keys())
