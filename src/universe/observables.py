"""Observable computations for the Kinetic Grid Universe.

Observables are functions that extract numeric values from states.
These are the building blocks for law claims.
"""

from src.universe.types import State, Symbol, VALID_SYMBOLS
from src.universe.validation import validate_state


def count_symbol(state: State, symbol: str) -> int:
    """Count occurrences of a symbol in the state.

    This is the primary primitive observable.

    Args:
        state: State string
        symbol: Symbol to count (one of . > < X)

    Returns:
        Count of the symbol

    Raises:
        ValueError: If symbol is not valid
    """
    if symbol not in VALID_SYMBOLS:
        raise ValueError(f"Invalid symbol '{symbol}'. Valid symbols: {sorted(VALID_SYMBOLS)}")

    return state.count(symbol)


def grid_length(state: State) -> int:
    """Return the length of the grid.

    Args:
        state: State string

    Returns:
        Length of the state string
    """
    return len(state)


# Derived observable helpers (these can be composed in expression language)

def count_right(state: State) -> int:
    """Count right-moving particles (>)."""
    return count_symbol(state, Symbol.RIGHT.value)


def count_left(state: State) -> int:
    """Count left-moving particles (<)."""
    return count_symbol(state, Symbol.LEFT.value)


def count_collision(state: State) -> int:
    """Count collision cells (X)."""
    return count_symbol(state, Symbol.COLLISION.value)


def count_empty(state: State) -> int:
    """Count empty cells (.)."""
    return count_symbol(state, Symbol.EMPTY.value)


def right_component(state: State) -> int:
    """Count right-moving components: count('>') + count('X').

    This is a conserved quantity.
    """
    return count_right(state) + count_collision(state)


def left_component(state: State) -> int:
    """Count left-moving components: count('<') + count('X').

    This is a conserved quantity.
    """
    return count_left(state) + count_collision(state)


def particle_count(state: State) -> int:
    """Total particle count: count('>') + count('<') + 2*count('X').

    This is a conserved quantity.
    """
    return count_right(state) + count_left(state) + 2 * count_collision(state)


def momentum(state: State) -> int:
    """Net momentum: right_component - left_component.

    Simplifies to count('>') - count('<').
    This is a conserved quantity.
    """
    return count_right(state) - count_left(state)


def occupied_cells(state: State) -> int:
    """Count occupied cells: count('>') + count('<') + count('X').

    This is NOT conserved (changes during collisions).
    """
    return count_right(state) + count_left(state) + count_collision(state)


# Registry of primitive observables for capability checking
PRIMITIVE_OBSERVABLES: dict[str, str] = {
    "count": "count(symbol) - count occurrences of a symbol",
    "grid_length": "grid_length - length of the state",
}


def evaluate_observable(name: str, state: State, **kwargs) -> int:
    """Evaluate a named observable on a state.

    This is a dispatch function for the expression evaluator.

    Args:
        name: Observable name
        state: State to evaluate on
        **kwargs: Additional arguments (e.g., symbol for count)

    Returns:
        Observable value

    Raises:
        ValueError: If observable is unknown
    """
    if name == "count":
        symbol = kwargs.get("symbol")
        if symbol is None:
            raise ValueError("count() requires a 'symbol' argument")
        return count_symbol(state, symbol)

    elif name == "grid_length":
        return grid_length(state)

    else:
        raise ValueError(f"Unknown observable: {name}")
