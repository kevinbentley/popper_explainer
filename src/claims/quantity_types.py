"""Quantity type system for observables.

Each derived quantity has a semantic type that describes what it measures:
- CELL_COUNT: Counts cells (occupied, empty) - X counts as 1
- PARTICLE_COUNT: Counts particles - X counts as 2 (one > and one <)
- COMPONENT_COUNT: Counts directional components (R+X or L+X)
- MOMENTUM_LIKE: Signed difference (can be negative)
- POSITION: Positions in the grid (leftmost, rightmost)
- LENGTH: Lengths/distances (grid_length, spread, max_gap)
- RATIO: Dimensionless ratio
- UNKNOWN: Cannot determine type

This enables semantic linting to catch misnamed laws.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.claims.expr_ast import (
    AdjacentPairs,
    BinOp,
    Count,
    Expr,
    GridLength,
    Leftmost,
    Literal,
    MaxGap,
    Operator,
    Rightmost,
    Spread,
)


class QuantityType(str, Enum):
    """Semantic type of a quantity."""

    CELL_COUNT = "cell_count"  # Counts cells (X=1)
    PARTICLE_COUNT = "particle_count"  # Counts particles (X=2)
    COMPONENT_COUNT = "component_count"  # Counts R+X or L+X
    MOMENTUM_LIKE = "momentum_like"  # Signed difference
    POSITION = "position"  # Grid position
    LENGTH = "length"  # Length/distance
    RATIO = "ratio"  # Dimensionless
    PAIR_COUNT = "pair_count"  # Counts adjacent pairs
    UNKNOWN = "unknown"


@dataclass
class TypedQuantity:
    """A quantity with its inferred type."""

    quantity_type: QuantityType
    description: str
    confidence: float = 1.0  # 0-1, how confident we are in the type


def infer_quantity_type(expr: Expr) -> TypedQuantity:
    """Infer the semantic type of an expression.

    Uses pattern matching to recognize common quantity patterns.

    Args:
        expr: The expression AST

    Returns:
        TypedQuantity with inferred type and description
    """
    # Normalize the expression to a canonical form for pattern matching
    pattern = _extract_pattern(expr)

    # Check against known patterns
    return _match_pattern(pattern, expr)


def _extract_pattern(expr: Expr) -> dict[str, Any]:
    """Extract a pattern representation from an expression for matching."""
    if isinstance(expr, Literal):
        return {"type": "literal", "value": expr.value}

    elif isinstance(expr, Count):
        return {"type": "count", "symbol": expr.symbol}

    elif isinstance(expr, GridLength):
        return {"type": "grid_length"}

    elif isinstance(expr, Leftmost):
        return {"type": "leftmost", "symbol": expr.symbol}

    elif isinstance(expr, Rightmost):
        return {"type": "rightmost", "symbol": expr.symbol}

    elif isinstance(expr, MaxGap):
        return {"type": "max_gap", "symbol": expr.symbol}

    elif isinstance(expr, Spread):
        return {"type": "spread", "symbol": expr.symbol}

    elif isinstance(expr, AdjacentPairs):
        return {"type": "adjacent_pairs", "sym1": expr.symbol1, "sym2": expr.symbol2}

    elif isinstance(expr, BinOp):
        left = _extract_pattern(expr.left)
        right = _extract_pattern(expr.right)
        return {
            "type": "binop",
            "op": expr.op,
            "left": left,
            "right": right,
        }

    return {"type": "unknown"}


def _match_pattern(pattern: dict[str, Any], expr: Expr) -> TypedQuantity:
    """Match a pattern against known quantity types."""

    # Simple counts
    if pattern["type"] == "count":
        symbol = pattern["symbol"]
        if symbol == ".":
            return TypedQuantity(
                QuantityType.CELL_COUNT,
                "empty cell count",
            )
        elif symbol == "X":
            return TypedQuantity(
                QuantityType.CELL_COUNT,
                "collision cell count",
            )
        elif symbol in (">", "<"):
            # Single mover count is ambiguous - could be component or cell
            return TypedQuantity(
                QuantityType.COMPONENT_COUNT,
                f"{'right' if symbol == '>' else 'left'} mover count",
                confidence=0.7,  # Could also be cell count
            )

    # Grid length
    if pattern["type"] == "grid_length":
        return TypedQuantity(QuantityType.LENGTH, "grid length")

    # Position observables
    if pattern["type"] == "leftmost":
        return TypedQuantity(QuantityType.POSITION, f"leftmost {pattern['symbol']} position")

    if pattern["type"] == "rightmost":
        return TypedQuantity(QuantityType.POSITION, f"rightmost {pattern['symbol']} position")

    # Length observables
    if pattern["type"] == "max_gap":
        return TypedQuantity(QuantityType.LENGTH, f"max gap of {pattern['symbol']}")

    if pattern["type"] == "spread":
        return TypedQuantity(QuantityType.LENGTH, f"spread of {pattern['symbol']}")

    # Pair counts
    if pattern["type"] == "adjacent_pairs":
        return TypedQuantity(
            QuantityType.PAIR_COUNT,
            f"adjacent pairs {pattern['sym1']}{pattern['sym2']}",
        )

    # Literals
    if pattern["type"] == "literal":
        return TypedQuantity(QuantityType.UNKNOWN, f"literal {pattern['value']}")

    # Binary operations - the interesting cases
    if pattern["type"] == "binop":
        return _infer_binop_type(pattern, expr)

    return TypedQuantity(QuantityType.UNKNOWN, "unknown quantity")


def _infer_binop_type(pattern: dict[str, Any], expr: Expr) -> TypedQuantity:
    """Infer type for binary operations."""
    op = pattern["op"]
    left = pattern["left"]
    right = pattern["right"]

    # Pattern: count('>') - count('<') → MOMENTUM_LIKE
    if op == Operator.SUB:
        if _is_count_of(left, ">") and _is_count_of(right, "<"):
            return TypedQuantity(QuantityType.MOMENTUM_LIKE, "net momentum (R - L)")
        if _is_count_of(left, "<") and _is_count_of(right, ">"):
            return TypedQuantity(QuantityType.MOMENTUM_LIKE, "net momentum (L - R)")

    # Pattern: count('>') + count('X') → COMPONENT_COUNT (right component)
    if op == Operator.ADD:
        if _is_count_of(left, ">") and _is_count_of(right, "X"):
            return TypedQuantity(QuantityType.COMPONENT_COUNT, "right component (R + X)")
        if _is_count_of(left, "X") and _is_count_of(right, ">"):
            return TypedQuantity(QuantityType.COMPONENT_COUNT, "right component (X + R)")
        if _is_count_of(left, "<") and _is_count_of(right, "X"):
            return TypedQuantity(QuantityType.COMPONENT_COUNT, "left component (L + X)")
        if _is_count_of(left, "X") and _is_count_of(right, "<"):
            return TypedQuantity(QuantityType.COMPONENT_COUNT, "left component (X + L)")

    # Pattern: count('>') + count('<') + ... → need to check for particle or cell count
    # This requires deeper analysis
    counts = _collect_count_terms(expr)
    if counts:
        return _analyze_count_sum(counts)

    # Pattern: 2 * count('X') or count('X') * 2 → part of particle count
    if op == Operator.MUL:
        if _is_literal(left, 2) and _is_count_of(right, "X"):
            return TypedQuantity(
                QuantityType.PARTICLE_COUNT,
                "collision particle contribution (2*X)",
                confidence=0.8,
            )
        if _is_count_of(left, "X") and _is_literal(right, 2):
            return TypedQuantity(
                QuantityType.PARTICLE_COUNT,
                "collision particle contribution (X*2)",
                confidence=0.8,
            )

    # Default: propagate from operands
    left_type = _match_pattern(left, expr)
    right_type = _match_pattern(right, expr)

    # Addition of same types preserves type
    if op == Operator.ADD and left_type.quantity_type == right_type.quantity_type:
        return TypedQuantity(
            left_type.quantity_type,
            f"sum of {left_type.description} and {right_type.description}",
            confidence=min(left_type.confidence, right_type.confidence),
        )

    return TypedQuantity(QuantityType.UNKNOWN, "complex expression")


def _is_count_of(pattern: dict[str, Any], symbol: str) -> bool:
    """Check if pattern is count(symbol)."""
    return pattern.get("type") == "count" and pattern.get("symbol") == symbol


def _is_literal(pattern: dict[str, Any], value: int) -> bool:
    """Check if pattern is a specific literal."""
    return pattern.get("type") == "literal" and pattern.get("value") == value


def _collect_count_terms(expr: Expr) -> dict[str, int]:
    """Collect count terms with their coefficients.

    Returns dict mapping symbol to coefficient.
    E.g., count('>') + count('<') + 2*count('X') → {'>': 1, '<': 1, 'X': 2}
    """
    if isinstance(expr, Count):
        return {expr.symbol: 1}

    if isinstance(expr, BinOp):
        if expr.op == Operator.ADD:
            left_counts = _collect_count_terms(expr.left)
            right_counts = _collect_count_terms(expr.right)
            if left_counts is not None and right_counts is not None:
                # Merge counts
                result = dict(left_counts)
                for sym, coef in right_counts.items():
                    result[sym] = result.get(sym, 0) + coef
                return result

        if expr.op == Operator.MUL:
            # Check for coefficient * count(sym)
            if isinstance(expr.left, Literal) and isinstance(expr.right, Count):
                return {expr.right.symbol: expr.left.value}
            if isinstance(expr.right, Literal) and isinstance(expr.left, Count):
                return {expr.left.symbol: expr.right.value}

    return None


def _analyze_count_sum(counts: dict[str, int]) -> TypedQuantity:
    """Analyze a sum of counts to determine its type."""
    # Check for particle count: R + L + 2*X
    if counts.get(">", 0) == 1 and counts.get("<", 0) == 1 and counts.get("X", 0) == 2:
        return TypedQuantity(
            QuantityType.PARTICLE_COUNT,
            "total particle count (R + L + 2*X)",
        )

    # Check for occupied cells: R + L + X (or any subset without .)
    movers = set(counts.keys()) - {"."}
    if movers and "." not in counts:
        # All mover types, X coefficient is 1
        if counts.get("X", 0) == 1 or "X" not in counts:
            if ">" in counts and "<" in counts:
                return TypedQuantity(
                    QuantityType.CELL_COUNT,
                    "occupied cell count",
                )

    # Check for component count: R + X or L + X
    if len(counts) == 2:
        if ">" in counts and "X" in counts and counts[">"] == 1 and counts["X"] == 1:
            return TypedQuantity(QuantityType.COMPONENT_COUNT, "right component (R + X)")
        if "<" in counts and "X" in counts and counts["<"] == 1 and counts["X"] == 1:
            return TypedQuantity(QuantityType.COMPONENT_COUNT, "left component (L + X)")

    return TypedQuantity(QuantityType.UNKNOWN, f"count sum: {counts}")


# Semantic keywords for law name validation
KEYWORD_TYPE_MAP = {
    "particle": {QuantityType.PARTICLE_COUNT},
    "momentum": {QuantityType.MOMENTUM_LIKE},
    "component": {QuantityType.COMPONENT_COUNT},
    "cell": {QuantityType.CELL_COUNT},
    "position": {QuantityType.POSITION},
    "length": {QuantityType.LENGTH},
    "spread": {QuantityType.LENGTH},
    "gap": {QuantityType.LENGTH},
}


@dataclass
class LintWarning:
    """A warning from the semantic linter."""

    law_id: str
    message: str
    severity: str = "warning"  # "warning" or "error"
    suggested_fix: str | None = None


def lint_law_name(
    law_id: str,
    observable_types: list[TypedQuantity],
) -> list[LintWarning]:
    """Check if a law's name matches its observable types.

    Args:
        law_id: The law identifier (used as the "name")
        observable_types: Types of observables used in the law

    Returns:
        List of warnings about semantic mismatches
    """
    warnings = []
    law_id_lower = law_id.lower()

    for keyword, expected_types in KEYWORD_TYPE_MAP.items():
        if keyword in law_id_lower:
            # Check if any observable matches the expected type
            actual_types = {ot.quantity_type for ot in observable_types}

            # Filter out UNKNOWN types for comparison
            actual_types_known = actual_types - {QuantityType.UNKNOWN}

            if actual_types_known and not (actual_types_known & expected_types):
                actual_str = ", ".join(t.value for t in actual_types_known)
                expected_str = ", ".join(t.value for t in expected_types)

                # Generate suggested fix
                suggested_name = None
                if QuantityType.CELL_COUNT in actual_types_known and keyword == "particle":
                    suggested_name = law_id.replace("particle", "cell")
                elif QuantityType.PARTICLE_COUNT in actual_types_known and keyword == "cell":
                    suggested_name = law_id.replace("cell", "particle")
                elif QuantityType.COMPONENT_COUNT in actual_types_known and keyword == "particle":
                    suggested_name = law_id.replace("particle", "component")

                warnings.append(LintWarning(
                    law_id=law_id,
                    message=f"Law name contains '{keyword}' but uses {actual_str} "
                            f"(expected {expected_str})",
                    severity="warning",
                    suggested_fix=suggested_name,
                ))

    return warnings


def lint_observable_expression(
    observable_name: str,
    expr: Expr,
) -> list[LintWarning]:
    """Check if an observable's name matches its expression type.

    Args:
        observable_name: The name given to the observable
        expr: The expression AST

    Returns:
        List of warnings
    """
    warnings = []
    typed = infer_quantity_type(expr)
    name_lower = observable_name.lower()

    for keyword, expected_types in KEYWORD_TYPE_MAP.items():
        if keyword in name_lower:
            if typed.quantity_type not in expected_types and typed.quantity_type != QuantityType.UNKNOWN:
                warnings.append(LintWarning(
                    law_id=observable_name,
                    message=f"Observable '{observable_name}' implies {keyword} but "
                            f"expression has type {typed.quantity_type.value}: {typed.description}",
                    severity="warning",
                ))

    return warnings
