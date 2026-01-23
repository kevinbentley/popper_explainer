"""Evaluator for structured claim ASTs.

This module evaluates structured claim ASTs against trajectories,
providing unambiguous semantics for all claim types.
"""

from typing import Any, Callable

from src.claims.ast_schema import ARITHMETIC_OPS, COMPARISON_OPS, LOGICAL_OPS, validate_claim_ast
from src.claims.expr_ast import Expr
from src.claims.expr_evaluator import EvaluationError, evaluate_expression
from src.claims.expr_parser import ExpressionParser
from src.claims.schema import CandidateLaw, Observable, Template
from src.universe.types import State, Trajectory


class ASTEvaluationError(Exception):
    """Error during AST evaluation."""
    pass


class ASTClaimEvaluator:
    """Evaluates structured claim ASTs against trajectories."""

    def __init__(self, law: CandidateLaw):
        """Initialize evaluator with a law.

        Args:
            law: The candidate law containing observables and claim_ast
        """
        self.law = law
        self._parser = ExpressionParser()

        # Compile observable expressions
        self._observable_exprs: dict[str, Expr] = {}
        for obs in law.observables:
            self._observable_exprs[obs.name] = self._parser.parse(obs.expr)

        # Validate the claim AST if present
        if law.claim_ast:
            obs_names = set(self._observable_exprs.keys())
            is_valid, errors = validate_claim_ast(law.claim_ast, obs_names)
            if not is_valid:
                error_msgs = [f"{e.path}: {e.message}" for e in errors]
                raise ASTEvaluationError(f"Invalid claim AST:\n" + "\n".join(error_msgs))

    def evaluate_at_time(
        self,
        ast: dict[str, Any],
        trajectory: Trajectory,
        t: int,
    ) -> Any:
        """Evaluate an AST node at a specific time step.

        Args:
            ast: The AST node to evaluate
            trajectory: The trajectory to evaluate against
            t: Current time step (for variable 't')

        Returns:
            The evaluated value (int, float, or bool)
        """
        if "const" in ast:
            return ast["const"]

        elif "var" in ast:
            if ast["var"] == "t":
                return t
            raise ASTEvaluationError(f"Unknown variable: {ast['var']}")

        elif "t_plus_1" in ast:
            return t + 1

        elif "obs" in ast:
            obs_name = ast["obs"]
            if obs_name not in self._observable_exprs:
                raise ASTEvaluationError(f"Unknown observable: {obs_name}")

            # Evaluate the time index
            time_idx = self.evaluate_at_time(ast["t"], trajectory, t)

            # Check bounds
            if time_idx < 0 or time_idx >= len(trajectory):
                raise ASTEvaluationError(
                    f"Time index {time_idx} out of bounds [0, {len(trajectory)-1}]"
                )

            # Evaluate the observable at that time
            state = trajectory[time_idx]
            try:
                return evaluate_expression(self._observable_exprs[obs_name], state)
            except EvaluationError as e:
                raise ASTEvaluationError(f"Observable '{obs_name}' evaluation failed: {e}") from e

        elif "op" in ast:
            op = ast["op"]

            if op == "not":
                arg = self.evaluate_at_time(ast["arg"], trajectory, t)
                return not arg

            lhs = self.evaluate_at_time(ast["lhs"], trajectory, t)
            rhs = self.evaluate_at_time(ast["rhs"], trajectory, t)

            # Arithmetic
            if op == "+":
                return lhs + rhs
            elif op == "-":
                return lhs - rhs
            elif op == "*":
                return lhs * rhs
            elif op == "/":
                return lhs / rhs if rhs != 0 else float('inf')

            # Comparison
            elif op == "==":
                return lhs == rhs
            elif op == "!=":
                return lhs != rhs
            elif op == "<":
                return lhs < rhs
            elif op == "<=":
                return lhs <= rhs
            elif op == ">":
                return lhs > rhs
            elif op == ">=":
                return lhs >= rhs

            # Logical
            elif op == "=>":
                return (not lhs) or rhs  # Implication
            elif op == "and":
                return lhs and rhs
            elif op == "or":
                return lhs or rhs

            raise ASTEvaluationError(f"Unknown operator: {op}")

        raise ASTEvaluationError(f"Unknown AST node type: {ast}")

    def check_invariant(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None]:
        """Check invariant: claim holds for all t.

        Returns:
            Tuple of (passed, t_fail, details)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        for t in range(len(trajectory)):
            try:
                result = self.evaluate_at_time(ast, trajectory, t)
                if not result:
                    return False, t, {"t": t}
            except ASTEvaluationError as e:
                return False, t, {"error": str(e)}
            except Exception as e:
                # Catch any other exceptions (e.g., EvaluationError from expr_evaluator)
                return False, t, {"error": f"Evaluation error: {e}"}

        return True, None, None

    def check_implication_step(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None]:
        """Check implication_step: for all t, if antecedent(t) then consequent(t+1).

        The claim_ast should have op "=>" with lhs as antecedent and rhs as consequent.

        Returns:
            Tuple of (passed, t_fail, details)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        if ast.get("op") != "=>":
            raise ASTEvaluationError("implication_step requires '=>' operator at root")

        antecedent_ast = ast["lhs"]
        consequent_ast = ast["rhs"]

        for t in range(len(trajectory) - 1):
            try:
                # Check antecedent at time t
                ante_result = self.evaluate_at_time(antecedent_ast, trajectory, t)

                if ante_result:
                    # Check consequent (which should reference t+1)
                    cons_result = self.evaluate_at_time(consequent_ast, trajectory, t)

                    if not cons_result:
                        return False, t, {
                            "t": t,
                            "antecedent": True,
                            "consequent": False,
                        }
            except ASTEvaluationError as e:
                return False, t, {"error": str(e)}
            except Exception as e:
                return False, t, {"error": f"Evaluation error: {e}"}

        return True, None, None

    def check_implication_state(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None]:
        """Check implication_state: for all t, if antecedent(t) then consequent(t).

        Returns:
            Tuple of (passed, t_fail, details)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        if ast.get("op") != "=>":
            raise ASTEvaluationError("implication_state requires '=>' operator at root")

        antecedent_ast = ast["lhs"]
        consequent_ast = ast["rhs"]

        for t in range(len(trajectory)):
            try:
                ante_result = self.evaluate_at_time(antecedent_ast, trajectory, t)

                if ante_result:
                    cons_result = self.evaluate_at_time(consequent_ast, trajectory, t)

                    if not cons_result:
                        return False, t, {
                            "t": t,
                            "antecedent": True,
                            "consequent": False,
                        }
            except ASTEvaluationError as e:
                return False, t, {"error": str(e)}
            except Exception as e:
                return False, t, {"error": f"Evaluation error: {e}"}

        return True, None, None

    def check_monotone(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None]:
        """Check monotone: f(t+1) op f(t) for all t.

        The claim_ast should be a comparison between f at t+1 and f at t.

        Returns:
            Tuple of (passed, t_fail, details)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        for t in range(len(trajectory) - 1):
            try:
                result = self.evaluate_at_time(ast, trajectory, t)
                if not result:
                    return False, t, {"t": t}
            except ASTEvaluationError as e:
                return False, t, {"error": str(e)}
            except Exception as e:
                return False, t, {"error": f"Evaluation error: {e}"}

        return True, None, None

    def check_bound(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None]:
        """Check bound: f(t) op k for all t.

        Returns:
            Tuple of (passed, t_fail, details)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        for t in range(len(trajectory)):
            try:
                result = self.evaluate_at_time(ast, trajectory, t)
                if not result:
                    return False, t, {"t": t}
            except ASTEvaluationError as e:
                return False, t, {"error": str(e)}
            except Exception as e:
                return False, t, {"error": f"Evaluation error: {e}"}

        return True, None, None

    def check_eventually(self, trajectory: Trajectory, horizon: int) -> tuple[bool, int | None, dict | None]:
        """Check eventually: if antecedent(t0) then exists t in [t0..t0+H] where consequent(t).

        Returns:
            Tuple of (passed, t_fail, details)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        if ast.get("op") != "=>":
            raise ASTEvaluationError("eventually requires '=>' operator at root")

        antecedent_ast = ast["lhs"]
        consequent_ast = ast["rhs"]

        for t0 in range(len(trajectory)):
            try:
                ante_result = self.evaluate_at_time(antecedent_ast, trajectory, t0)

                if ante_result:
                    # Search for consequent in [t0..t0+H]
                    found = False
                    end = min(t0 + horizon + 1, len(trajectory))

                    for t in range(t0, end):
                        cons_result = self.evaluate_at_time(consequent_ast, trajectory, t)
                        if cons_result:
                            found = True
                            break

                    if not found:
                        return False, t0, {
                            "t0": t0,
                            "horizon": horizon,
                            "search_end": end - 1,
                        }
            except ASTEvaluationError as e:
                return False, t0, {"error": str(e)}
            except Exception as e:
                return False, t0, {"error": f"Evaluation error: {e}"}

        return True, None, None

    def check(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None]:
        """Check the claim against a trajectory based on template type.

        Returns:
            Tuple of (passed, t_fail, details)
        """
        template = self.law.template

        if template == Template.INVARIANT:
            return self.check_invariant(trajectory)
        elif template == Template.MONOTONE:
            return self.check_monotone(trajectory)
        elif template == Template.BOUND:
            return self.check_bound(trajectory)
        elif template == Template.IMPLICATION_STEP:
            return self.check_implication_step(trajectory)
        elif template == Template.IMPLICATION_STATE:
            return self.check_implication_state(trajectory)
        elif template == Template.EVENTUALLY:
            horizon = self.law.quantifiers.H or 10
            return self.check_eventually(trajectory, horizon)
        elif template == Template.SYMMETRY_COMMUTATION:
            # Symmetry is handled specially by the harness
            raise ASTEvaluationError("Use SymmetryCommutationChecker for symmetry_commutation")
        else:
            raise ASTEvaluationError(f"Unknown template: {template}")


def create_ast_checker(law: CandidateLaw) -> ASTClaimEvaluator | None:
    """Create an AST checker for a law if it has claim_ast.

    Args:
        law: The candidate law

    Returns:
        ASTClaimEvaluator if claim_ast is present, None otherwise
    """
    if law.claim_ast is None:
        return None

    return ASTClaimEvaluator(law)
