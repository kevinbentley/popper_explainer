"""Evaluator for observable expressions.

Takes an expression AST and a state, returns an integer value.
"""

from src.claims.expr_ast import (
    AdjacentPairs,
    BinOp,
    Count,
    CountAtParity,
    CountEven,
    CountOdd,
    CountPattern,
    Expr,
    GapPairs,
    GridLength,
    TransitionIndicator,
    Leftmost,
    Literal,
    MaxGap,
    Operator,
    Rightmost,
    Spread,
)
from src.universe.observables import (
    adjacent_pairs,
    count_at_parity,
    count_even,
    count_odd,
    count_pattern,
    count_symbol,
    gap_pairs,
    grid_length,
    incoming_collisions as transition_indicator_impl,  # Internal implementation
    leftmost,
    max_gap,
    rightmost,
    spread,
)
from src.universe.types import State


class EvaluationError(Exception):
    """Raised when expression evaluation fails."""

    pass


class ExpressionEvaluator:
    """Evaluates expression ASTs against universe states."""

    def evaluate(self, expr: Expr, state: State) -> int:
        """Evaluate an expression against a state.

        Args:
            expr: The expression AST
            state: The universe state string

        Returns:
            The computed integer value

        Raises:
            EvaluationError: If evaluation fails
        """
        if isinstance(expr, Literal):
            return expr.value

        elif isinstance(expr, Count):
            try:
                return count_symbol(state, expr.symbol)
            except ValueError as e:
                raise EvaluationError(f"Invalid count: {e}") from e

        elif isinstance(expr, GridLength):
            return grid_length(state)

        elif isinstance(expr, TransitionIndicator):
            return transition_indicator_impl(state)

        elif isinstance(expr, Leftmost):
            try:
                return leftmost(state, expr.symbol)
            except ValueError as e:
                raise EvaluationError(f"Invalid leftmost: {e}") from e

        elif isinstance(expr, Rightmost):
            try:
                return rightmost(state, expr.symbol)
            except ValueError as e:
                raise EvaluationError(f"Invalid rightmost: {e}") from e

        elif isinstance(expr, MaxGap):
            try:
                return max_gap(state, expr.symbol)
            except ValueError as e:
                raise EvaluationError(f"Invalid max_gap: {e}") from e

        elif isinstance(expr, AdjacentPairs):
            try:
                return adjacent_pairs(state, expr.symbol1, expr.symbol2)
            except ValueError as e:
                raise EvaluationError(f"Invalid adjacent_pairs: {e}") from e

        elif isinstance(expr, GapPairs):
            try:
                return gap_pairs(state, expr.symbol1, expr.symbol2, expr.gap)
            except ValueError as e:
                raise EvaluationError(f"Invalid gap_pairs: {e}") from e

        elif isinstance(expr, Spread):
            try:
                return spread(state, expr.symbol)
            except ValueError as e:
                raise EvaluationError(f"Invalid spread: {e}") from e

        elif isinstance(expr, CountAtParity):
            try:
                return count_at_parity(state, expr.symbol, expr.parity)
            except ValueError as e:
                raise EvaluationError(f"Invalid count_at_parity: {e}") from e

        elif isinstance(expr, CountEven):
            try:
                return count_even(state, expr.symbol)
            except ValueError as e:
                raise EvaluationError(f"Invalid count_even: {e}") from e

        elif isinstance(expr, CountOdd):
            try:
                return count_odd(state, expr.symbol)
            except ValueError as e:
                raise EvaluationError(f"Invalid count_odd: {e}") from e

        elif isinstance(expr, CountPattern):
            try:
                return count_pattern(state, expr.pattern)
            except ValueError as e:
                raise EvaluationError(f"Invalid count_pattern: {e}") from e

        elif isinstance(expr, BinOp):
            left_val = self.evaluate(expr.left, state)
            right_val = self.evaluate(expr.right, state)

            if expr.op == Operator.ADD:
                return left_val + right_val
            elif expr.op == Operator.SUB:
                return left_val - right_val
            elif expr.op == Operator.MUL:
                return left_val * right_val
            elif expr.op == Operator.MOD:
                if right_val == 0:
                    raise EvaluationError("Modulo by zero")
                return left_val % right_val
            else:
                raise EvaluationError(f"Unknown operator: {expr.op}")

        else:
            raise EvaluationError(f"Unknown expression type: {type(expr)}")


# Module-level evaluator instance for convenience
_evaluator = ExpressionEvaluator()


def evaluate_expression(expr: Expr, state: State) -> int:
    """Evaluate an expression against a state.

    Convenience function using module-level evaluator.
    """
    return _evaluator.evaluate(expr, state)
