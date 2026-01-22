"""Type definitions for the Kinetic Grid Universe."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Symbol(str, Enum):
    """Valid cell symbols in the universe.

    . = empty cell
    > = right-moving particle
    < = left-moving particle
    X = collision (one > and one < in same cell)
    """

    EMPTY = "."
    RIGHT = ">"
    LEFT = "<"
    COLLISION = "X"


# Type aliases
State = str  # A string of length N containing only valid symbols
Trajectory = list[State]  # States from t=0 to t=T

# Valid symbol characters
VALID_SYMBOLS: frozenset[str] = frozenset(s.value for s in Symbol)


@dataclass(frozen=True)
class Config:
    """Universe configuration.

    Attributes:
        grid_length: Number of cells in the universe (N)
        boundary: Boundary condition type (only 'periodic' is supported)
    """

    grid_length: int
    boundary: Literal["periodic"] = "periodic"

    def __post_init__(self) -> None:
        if self.grid_length < 1:
            raise ValueError(f"grid_length must be >= 1, got {self.grid_length}")
        if self.boundary != "periodic":
            raise ValueError(f"Only 'periodic' boundary supported, got {self.boundary}")
