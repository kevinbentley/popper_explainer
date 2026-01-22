"""AST nodes for observable expressions.

The expression language is intentionally minimal:
- count('<symbol>') - count occurrences of a symbol
- grid_length - length of the state
- Integer literals
- Binary operators: +, -, *
- Parentheses for grouping

This is a CLOSED set - no other constructs are allowed.
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
class BinOp:
    """A binary operation: left op right."""

    left: "Expr"
    op: Operator
    right: "Expr"

    def __repr__(self) -> str:
        return f"BinOp({self.left}, {self.op.value}, {self.right})"


# Union type for all expression nodes
Expr = Union[Literal, Count, GridLength, BinOp]


def expr_to_string(expr: Expr) -> str:
    """Convert an expression AST back to string form."""
    if isinstance(expr, Literal):
        return str(expr.value)
    elif isinstance(expr, Count):
        return f"count('{expr.symbol}')"
    elif isinstance(expr, GridLength):
        return "grid_length"
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
