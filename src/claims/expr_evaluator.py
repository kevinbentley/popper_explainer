"""Evaluator for observable expressions.

Takes an expression AST and a state, returns an integer value.
"""

from src.claims.expr_ast import BinOp, Count, Expr, GridLength, Literal, Operator
from src.universe.observables import count_symbol, grid_length
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

        elif isinstance(expr, BinOp):
            left_val = self.evaluate(expr.left, state)
            right_val = self.evaluate(expr.right, state)

            if expr.op == Operator.ADD:
                return left_val + right_val
            elif expr.op == Operator.SUB:
                return left_val - right_val
            elif expr.op == Operator.MUL:
                return left_val * right_val
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
