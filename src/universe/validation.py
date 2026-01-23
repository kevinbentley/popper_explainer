"""State validation for the Kinetic Grid Universe.

Per UNIVERSE.md:
- Valid symbols: . > < X
- X represents a collision (simultaneous occupation by > and <)
- X is ephemeral and only exists as the result of collision
- Initial states (t=0) MUST NOT contain X

This module enforces the universe contract for state validity.
"""

from src.universe.types import VALID_SYMBOLS, Config, State


class InvalidStateError(Exception):
    """Raised when a state is invalid."""

    pass


class InvalidInitialStateError(InvalidStateError):
    """Raised when an initial state (t=0) violates the universe contract.

    Per UNIVERSE.md section 3.2: X is ephemeral and can only arise from
    collision between > and < particles. It cannot exist at t=0.
    """

    pass


def is_valid_state(state: State, config: Config | None = None) -> bool:
    """Check if a state is valid.

    A valid state:
    - Is a non-empty string
    - Contains only valid symbols (. > < X)
    - Has length matching config.grid_length (if config provided)

    Args:
        state: The state string to validate
        config: Optional configuration to check length against

    Returns:
        True if valid, False otherwise
    """
    if not state:
        return False

    if not all(c in VALID_SYMBOLS for c in state):
        return False

    if config is not None and len(state) != config.grid_length:
        return False

    return True


def validate_state(state: State, config: Config | None = None) -> None:
    """Validate a state, raising InvalidStateError if invalid.

    Args:
        state: The state string to validate
        config: Optional configuration to check length against

    Raises:
        InvalidStateError: If the state is invalid
    """
    if not state:
        raise InvalidStateError("State cannot be empty")

    invalid_chars = [c for c in state if c not in VALID_SYMBOLS]
    if invalid_chars:
        raise InvalidStateError(
            f"Invalid characters in state: {invalid_chars}. "
            f"Valid symbols are: {sorted(VALID_SYMBOLS)}"
        )

    if config is not None and len(state) != config.grid_length:
        raise InvalidStateError(
            f"State length {len(state)} does not match config grid_length {config.grid_length}"
        )


def is_valid_initial_state(state: State, config: Config | None = None) -> bool:
    """Check if a state is valid as an initial state (t=0).

    Per UNIVERSE.md:
    - X represents a collision outcome, not an initial configuration
    - X can only arise from > and < particles colliding
    - Initial states MUST have count('X') == 0

    Args:
        state: The state string to validate
        config: Optional configuration to check length against

    Returns:
        True if valid as initial state, False otherwise
    """
    if not is_valid_state(state, config):
        return False

    # Universe contract: X cannot exist at t=0
    if state.count('X') > 0:
        return False

    return True


def validate_initial_state(state: State, config: Config | None = None) -> None:
    """Validate an initial state (t=0), raising error if invalid.

    Per UNIVERSE.md section 3.2:
    - X is "ephemeral (exists for exactly one time step)"
    - X only forms when "> and < attempt to occupy the same cell"
    - Therefore X cannot exist in an initial configuration

    Args:
        state: The state string to validate
        config: Optional configuration to check length against

    Raises:
        InvalidStateError: If the state has invalid characters or length
        InvalidInitialStateError: If the state contains X (collision cells)
    """
    # First validate basic state properties
    validate_state(state, config)

    # Universe contract: initial states cannot contain X
    x_count = state.count('X')
    if x_count > 0:
        raise InvalidInitialStateError(
            f"Initial state contains {x_count} collision cell(s) 'X'. "
            f"Per universe contract, X can only arise from collision between "
            f"> and < particles and cannot exist at t=0. "
            f"State: '{state}'"
        )


def get_state_validity_report(state: State, config: Config | None = None) -> dict:
    """Get detailed validity report for a state.

    Useful for debugging and logging invalid states.

    Args:
        state: The state string to analyze
        config: Optional configuration

    Returns:
        Dict with validity info and symbol counts
    """
    report = {
        "state": state,
        "length": len(state) if state else 0,
        "is_valid_state": False,
        "is_valid_initial_state": False,
        "symbol_counts": {},
        "issues": [],
    }

    if not state:
        report["issues"].append("State is empty")
        return report

    # Count symbols
    report["symbol_counts"] = {
        ">": state.count(">"),
        "<": state.count("<"),
        "X": state.count("X"),
        ".": state.count("."),
    }

    # Check for invalid characters
    invalid_chars = [c for c in state if c not in VALID_SYMBOLS]
    if invalid_chars:
        report["issues"].append(f"Invalid characters: {invalid_chars}")
    else:
        report["is_valid_state"] = True

    # Check length if config provided
    if config is not None and len(state) != config.grid_length:
        report["issues"].append(
            f"Length {len(state)} != config.grid_length {config.grid_length}"
        )
        report["is_valid_state"] = False

    # Check initial state validity
    if report["is_valid_state"]:
        if report["symbol_counts"]["X"] > 0:
            report["issues"].append(
                f"Contains {report['symbol_counts']['X']} X cells (invalid for t=0)"
            )
        else:
            report["is_valid_initial_state"] = True

    return report
