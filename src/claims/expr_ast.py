"""AST nodes for observable expressions.

The expression language supports:
- count('<symbol>') - count occurrences of a symbol
- grid_length - length of the state
- leftmost('<symbol>') - position of first occurrence (-1 if none)
- rightmost('<symbol>') - position of last occurrence (-1 if none)
- max_gap('<symbol>') - longest contiguous run of symbol
- adjacent_pairs('<s1>', '<s2>') - count of s1 followed by s2
- gap_pairs('<s1>', '<s2>', gap) - count of s1 followed by s2 with gap cells between
- incoming_collisions - count of cells that will have collision at t+1
- spread('<symbol>') - rightmost - leftmost position
- Integer literals
- Binary operators: +, -, *
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
class IncomingCollisions:
    """incoming_collisions - count of cells that will have collision at t+1.

    This is THE canonical collision predictor. A cell j has an incoming
    collision if state[(j-1)%L]='>' AND state[(j+1)%L]='<'.
    Properly handles wrap-around with periodic boundaries.
    """

    def __repr__(self) -> str:
        return "IncomingCollisions()"


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
class BinOp:
    """A binary operation: left op right."""

    left: "Expr"
    op: Operator
    right: "Expr"

    def __repr__(self) -> str:
        return f"BinOp({self.left}, {self.op.value}, {self.right})"


# Union type for all expression nodes
Expr = Union[Literal, Count, GridLength, IncomingCollisions, Leftmost, Rightmost, MaxGap, AdjacentPairs, GapPairs, Spread, BinOp]


def expr_to_string(expr: Expr) -> str:
    """Convert an expression AST back to string form."""
    if isinstance(expr, Literal):
        return str(expr.value)
    elif isinstance(expr, Count):
        return f"count('{expr.symbol}')"
    elif isinstance(expr, GridLength):
        return "grid_length"
    elif isinstance(expr, IncomingCollisions):
        return "incoming_collisions"
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
    elif isinstance(expr, BinOp):
        left_str = expr_to_string(expr.left)
        right_str = expr_to_string(expr.right)
        # Add parens if needed for precedence
        if isinstance(expr.left, BinOp) and expr.op == Operator.MUL:
            if expr.left.op in (Operator.ADD, Operator.SUB):
                left_str = f"({left_str})"
        if isinstance(expr.right, BinOp) and expr.op == Operator.MUL:
            if expr.right.op in (Operator.ADD, Operator.SUB):
                right_str = f"({right_str})"
        return f"{left_str} {expr.op.value} {right_str}"
    else:
        raise ValueError(f"Unknown expression type: {type(expr)}")
