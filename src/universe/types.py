"""Type definitions for the Kinetic Grid Universe."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Symbol(str, Enum):
    """Valid cell symbols in the universe (physical representation).

    . = empty cell
    > = right-moving particle
    < = left-moving particle
    X = collision (one > and one < in same cell)
    """

    EMPTY = "."
    RIGHT = ">"
    LEFT = "<"
    COLLISION = "X"


class AbstractSymbol(str, Enum):
    """Abstract symbols used in LLM prompts to prevent physics inference.

    The LLM sees these neutral symbols instead of physical ones:
    _ = Background (maps to .)
    A = Chiral-1 (maps to >)
    B = Chiral-2 (maps to <)
    K = Kinetic (maps to X)
    """

    BACKGROUND = "_"
    CHIRAL_1 = "A"
    CHIRAL_2 = "B"
    KINETIC = "K"


# Type aliases
State = str  # A string of length N containing only valid symbols
Trajectory = list[State]  # States from t=0 to t=T

# Valid symbol characters (physical only - for simulator)
PHYSICAL_SYMBOLS: frozenset[str] = frozenset(s.value for s in Symbol)

# Abstract symbols (for LLM interface)
ABSTRACT_SYMBOLS: frozenset[str] = frozenset(s.value for s in AbstractSymbol)

# All valid symbols (both physical and abstract - for expression parsing)
VALID_SYMBOLS: frozenset[str] = PHYSICAL_SYMBOLS | ABSTRACT_SYMBOLS


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
