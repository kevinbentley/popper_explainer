"""AST nodes for observable expressions.

The expression language supports:
- count('<symbol>') - count occurrences of a symbol
- grid_length - length of the state
- leftmost('<symbol>') - position of first occurrence (-1 if none)
- rightmost('<symbol>') - position of last occurrence (-1 if none)
- max_gap('<symbol>') - longest contiguous run of symbol
- adjacent_pairs('<s1>', '<s2>') - count of s1 followed by s2
- gap_pairs('<s1>', '<s2>', gap) - count of s1 followed by s2 with gap cells between
- transition_indicator - a count related to future state transitions
- spread('<symbol>') - rightmost - leftmost position
- Integer literals
- Binary operators: +, -, *, %
- Parentheses for grouping
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union


class Operator(Enum):
    """Binary operators for expressions."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    MOD = "%"


@dataclass(frozen=True)
class Literal:
    """An integer literal."""

    value: int

    def __repr__(self) -> str:
        return f"Literal({self.value})"


@dataclass(frozen=True)
class Count:
    """count('<symbol>') - count occurrences of a symbol in the state."""

    symbol: str

    def __repr__(self) -> str:
        return f"Count('{self.symbol}')"


@dataclass(frozen=True)
class GridLength:
    """grid_length - the length of the state."""

    def __repr__(self) -> str:
        return "GridLength()"


@dataclass(frozen=True)
class TransitionIndicator:
    """transition_indicator - a count related to future state transitions.

    This observable provides information about cells that will change
    in the next timestep. Its exact semantics must be discovered
    through experimentation.
    """

    def __repr__(self) -> str:
        return "TransitionIndicator()"


# Backwards compatibility alias
IncomingCollisions = TransitionIndicator


@dataclass(frozen=True)
class Leftmost:
    """leftmost('<symbol>') - position of first occurrence (-1 if none)."""

    symbol: str

    def __repr__(self) -> str:
        return f"Leftmost('{self.symbol}')"


@dataclass(frozen=True)
class Rightmost:
    """rightmost('<symbol>') - position of last occurrence (-1 if none)."""

    symbol: str

    def __repr__(self) -> str:
        return f"Rightmost('{self.symbol}')"


@dataclass(frozen=True)
class MaxGap:
    """max_gap('<symbol>') - longest contiguous run of symbol."""

    symbol: str

    def __repr__(self) -> str:
        return f"MaxGap('{self.symbol}')"


@dataclass(frozen=True)
class AdjacentPairs:
    """adjacent_pairs('<s1>', '<s2>') - count of s1 immediately followed by s2."""

    symbol1: str
    symbol2: str

    def __repr__(self) -> str:
        return f"AdjacentPairs('{self.symbol1}', '{self.symbol2}')"


@dataclass(frozen=True)
class GapPairs:
    """gap_pairs('<s1>', '<s2>', gap) - count of s1 followed by s2 with gap cells between.

    Essential for detecting converging particles:
    - gap_pairs('>', '<', 1) counts '>.<' patterns (will collide in 1 step)
    - gap_pairs('>', '<', 0) is equivalent to adjacent_pairs('>', '<')
    """

    symbol1: str
    symbol2: str
    gap: int

    def __repr__(self) -> str:
        return f"GapPairs('{self.symbol1}', '{self.symbol2}', {self.gap})"


@dataclass(frozen=True)
class Spread:
    """spread('<symbol>') - rightmost - leftmost position (0 if <2 occurrences)."""

    symbol: str

    def __repr__(self) -> str:
        return f"Spread('{self.symbol}')"


@dataclass(frozen=True)
class CountAtParity:
    """count_at_parity('<symbol>', parity) - count at even (0) or odd (1) indices.

    This enables discovering grid-phase dependent patterns.
    Example: count_at_parity('>', 0) counts right-movers at even indices.
    """

    symbol: str
    parity: int  # 0 for even, 1 for odd

    def __repr__(self) -> str:
        return f"CountAtParity('{self.symbol}', {self.parity})"


@dataclass(frozen=True)
class CountEven:
    """count_even('<symbol>') - count at even indices (convenience wrapper)."""

    symbol: str

    def __repr__(self) -> str:
        return f"CountEven('{self.symbol}')"


@dataclass(frozen=True)
class CountOdd:
    """count_odd('<symbol>') - count at odd indices (convenience wrapper)."""

    symbol: str

    def __repr__(self) -> str:
        return f"CountOdd('{self.symbol}')"


@dataclass(frozen=True)
class CountPattern:
    """count_pattern('<3-char-pattern>') - count cells with that neighborhood.

    The pattern is a 3-character string representing [left, center, right].
    Example: count_pattern('>.<') counts positions where the neighborhood is '>.<'.

    This enables the AI to discover correlations between local patterns
    and global events (e.g., ">.<" patterns predict future collisions).
    """

    pattern: str  # Exactly 3 characters

    def __repr__(self) -> str:
        return f"CountPattern('{self.pattern}')"


@dataclass(frozen=True)
class BinOp:
    """A binary operation: left op right."""

    left: "Expr"
    op: Operator
    right: "Expr"

    def __repr__(self) -> str:
        return f"BinOp({self.left}, {self.op.value}, {self.right})"


# Union type for all expression nodes
Expr = Union[Literal, Count, GridLength, TransitionIndicator, Leftmost, Rightmost, MaxGap, AdjacentPairs, GapPairs, Spread, CountAtParity, CountEven, CountOdd, CountPattern, BinOp]


def expr_to_string(expr: Expr) -> str:
    """Convert an expression AST back to string form."""
    if isinstance(expr, Literal):
        return str(expr.value)
    elif isinstance(expr, Count):
        return f"count('{expr.symbol}')"
    elif isinstance(expr, GridLength):
        return "grid_length"
    elif isinstance(expr, TransitionIndicator):
        return "transition_indicator"
    elif isinstance(expr, Leftmost):
        return f"leftmost('{expr.symbol}')"
    elif isinstance(expr, Rightmost):
        return f"rightmost('{expr.symbol}')"
    elif isinstance(expr, MaxGap):
        return f"max_gap('{expr.symbol}')"
    elif isinstance(expr, AdjacentPairs):
        return f"adjacent_pairs('{expr.symbol1}', '{expr.symbol2}')"
    elif isinstance(expr, GapPairs):
        return f"gap_pairs('{expr.symbol1}', '{expr.symbol2}', {expr.gap})"
    elif isinstance(expr, Spread):
        return f"spread('{expr.symbol}')"
    elif isinstance(expr, CountAtParity):
        return f"count_at_parity('{expr.symbol}', {expr.parity})"
    elif isinstance(expr, CountEven):
        return f"count_even('{expr.symbol}')"
    elif isinstance(expr, CountOdd):
        return f"count_odd('{expr.symbol}')"
    elif isinstance(expr, CountPattern):
        return f"count_pattern('{expr.pattern}')"
    elif isinstance(expr, BinOp):
        left_str = expr_to_string(expr.left)
        right_str = expr_to_string(expr.right)
        # Add parens if needed for precedence (MUL and MOD have same precedence)
        high_precedence = (Operator.MUL, Operator.MOD)
        low_precedence = (Operator.ADD, Operator.SUB)
        if isinstance(expr.left, BinOp) and expr.op in high_precedence:
            if expr.left.op in low_precedence:
                left_str = f"({left_str})"
        if isinstance(expr.right, BinOp) and expr.op in high_precedence:
            if expr.right.op in low_precedence:
                right_str = f"({right_str})"
        return f"{left_str} {expr.op.value} {right_str}"
    else:
        raise ValueError(f"Unknown expression type: {type(expr)}")
