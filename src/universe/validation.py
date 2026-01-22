"""State validation for the Kinetic Grid Universe."""

from src.universe.types import VALID_SYMBOLS, Config, State


class InvalidStateError(Exception):
    """Raised when a state is invalid."""

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
