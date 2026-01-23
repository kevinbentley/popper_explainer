"""Evaluator for structured claim ASTs.

This module evaluates structured claim ASTs against trajectories,
providing unambiguous semantics for all claim types.

TIME INDEXING DISCIPLINE (critical for avoiding false FAILs):
=============================================================
For a trajectory of length T+1 (states at t=0,1,...,T):

- invariant: evaluates at t ∈ [start_t, T - max_offset]
  where start_t = max(0, -min_offset) accounts for t-1 references
  and max_offset accounts for t+1 references

- monotone: evaluates at t ∈ [start_t, T - max(1, max_offset)]
  Always stops at least 1 early since monotone inherently uses t+1

- implication_step: evaluates at t ∈ [start_t, T - max(1, max_offset)]
  Always stops at least 1 early since consequent typically references t+1

- implication_state: evaluates at t ∈ [start_t, T - max_offset]
  Both antecedent and consequent evaluated at same t

- eventually: for each trigger at t0, searches t ∈ [t0, min(t0+H, traj_limit)]
  where traj_limit = T - cons_max to avoid out-of-bounds in consequent

- bound: evaluates at t ∈ [start_t, T - max_offset]

The functions get_min_time_offset() and get_max_time_offset() scan ASTs
to determine the required bounds automatically.
"""

from dataclasses import dataclass
from typing import Any, Callable

from src.claims.ast_schema import ARITHMETIC_OPS, COMPARISON_OPS, LOGICAL_OPS, validate_claim_ast
from src.claims.expr_ast import Expr
from src.claims.expr_evaluator import EvaluationError, evaluate_expression
from src.claims.expr_parser import ExpressionParser
from src.claims.schema import CandidateLaw, Observable, Template
from src.claims.vacuity import VacuityReport
from src.universe.types import State, Trajectory


class ASTEvaluationError(Exception):
    """Error during AST evaluation."""
    pass


def get_min_time_offset(ast: dict) -> int:
    """Determine the minimum time offset referenced in an AST.

    Scans the AST for time expressions like (t - k) and returns the most
    negative offset. This allows evaluation loops to start at the appropriate
    time step to avoid out-of-bounds errors.

    Examples:
        - Obs(t) → offset 0
        - Obs(t-1) → offset -1
        - Obs(t) == Obs(t-2) → offset -2
        - Obs(t+1) → offset 1 (but min returns 0)

    Args:
        ast: The AST node to scan

    Returns:
        The minimum time offset (0 or negative)
    """
    if not isinstance(ast, dict):
        return 0

    min_offset = 0

    if "obs" in ast:
        # Check the time expression
        t_ast = ast.get("t", {})
        offset = _compute_time_offset(t_ast)
        min_offset = min(min_offset, offset)

    elif "op" in ast:
        op = ast["op"]
        if op == "not":
            min_offset = min(min_offset, get_min_time_offset(ast.get("arg", {})))
        else:
            min_offset = min(min_offset, get_min_time_offset(ast.get("lhs", {})))
            min_offset = min(min_offset, get_min_time_offset(ast.get("rhs", {})))

    return min_offset


def get_max_time_offset(ast: dict) -> int:
    """Determine the maximum time offset referenced in an AST.

    Scans the AST for time expressions like (t + k) or t_plus_1 and returns
    the largest positive offset. This allows evaluation loops to stop early
    to avoid out-of-bounds errors.

    Examples:
        - Obs(t) → offset 0
        - Obs(t+1) → offset 1
        - Obs(t) == Obs(t+2) → offset 2
        - Obs(t-1) → offset 0 (but max returns 0)

    Args:
        ast: The AST node to scan

    Returns:
        The maximum time offset (0 or positive)
    """
    if not isinstance(ast, dict):
        return 0

    max_offset = 0

    if "obs" in ast:
        # Check the time expression
        t_ast = ast.get("t", {})
        offset = _compute_time_offset(t_ast)
        max_offset = max(max_offset, offset)

    elif "op" in ast:
        op = ast["op"]
        if op == "not":
            max_offset = max(max_offset, get_max_time_offset(ast.get("arg", {})))
        else:
            max_offset = max(max_offset, get_max_time_offset(ast.get("lhs", {})))
            max_offset = max(max_offset, get_max_time_offset(ast.get("rhs", {})))

    return max_offset


def _compute_time_offset(t_ast: dict) -> int:
    """Compute the offset from t in a time expression.

    Args:
        t_ast: AST node representing a time expression

    Returns:
        The offset: 0 for t, -1 for t-1, +1 for t+1, etc.
    """
    if not isinstance(t_ast, dict):
        return 0

    if "var" in t_ast and t_ast["var"] == "t":
        return 0

    if "t_plus_1" in t_ast:
        return 1

    if "const" in t_ast:
        # Constant time like t=0 doesn't affect the loop start
        return 0

    if "op" in t_ast:
        op = t_ast["op"]
        lhs = t_ast.get("lhs", {})
        rhs = t_ast.get("rhs", {})

        # Check for pattern: t - k or t + k
        if op == "-" and lhs.get("var") == "t" and "const" in rhs:
            return -rhs["const"]
        if op == "+" and lhs.get("var") == "t" and "const" in rhs:
            return rhs["const"]

    return 0


@dataclass
class ASTCheckResult:
    """Result of checking a claim AST against a trajectory.

    Attributes:
        passed: Whether the claim held
        t_fail: Time step where claim failed (if applicable)
        details: Additional details about the result
        vacuity: Vacuity report with antecedent tracking
    """
    passed: bool
    t_fail: int | None = None
    details: dict | None = None
    vacuity: VacuityReport | None = None


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

    def check_invariant(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None, VacuityReport]:
        """Check invariant: claim holds for all t.

        If the AST references past times (e.g., t-1), evaluation starts at
        the appropriate time step to avoid out-of-bounds errors.
        If the AST references future times (e.g., t+1), evaluation stops early.

        Returns:
            Tuple of (passed, t_fail, details, vacuity)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        # Determine start time based on past references (e.g., t-1 means start at t=1)
        min_offset = get_min_time_offset(ast)
        start_t = max(0, -min_offset)

        # Determine end time based on future references (e.g., t+1 means stop 1 early)
        max_offset = get_max_time_offset(ast)
        end_t = len(trajectory) - max_offset

        total_checks = max(0, end_t - start_t)
        vacuity = VacuityReport(total_checks=total_checks)

        for t in range(start_t, end_t):
            try:
                result = self.evaluate_at_time(ast, trajectory, t)
                if not result:
                    return False, t, {"t": t}, vacuity
            except ASTEvaluationError as e:
                return False, t, {"error": str(e)}, vacuity
            except Exception as e:
                # Catch any other exceptions (e.g., EvaluationError from expr_evaluator)
                return False, t, {"error": f"Evaluation error: {e}"}, vacuity

        return True, None, None, vacuity

    def check_implication_step(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None, VacuityReport]:
        """Check implication_step: for all t, if antecedent(t) then consequent(t+1).

        The claim_ast should have op "=>" with lhs as antecedent and rhs as consequent.
        If no "=>" at root, treats the entire AST as the consequent with trivially
        true antecedent (effectively an invariant over step pairs).

        Returns:
            Tuple of (passed, t_fail, details, vacuity)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        # Handle ASTs without explicit implication - treat as "true => claim"
        if ast.get("op") == "=>":
            antecedent_ast = ast["lhs"]
            consequent_ast = ast["rhs"]
        else:
            # No explicit antecedent - always check the consequent
            antecedent_ast = {"const": 1}  # Always true
            consequent_ast = ast

        # Determine time bounds based on references
        # Antecedent is evaluated at t, consequent typically references t+1
        min_offset = min(get_min_time_offset(antecedent_ast), get_min_time_offset(consequent_ast))
        max_offset = max(get_max_time_offset(antecedent_ast), get_max_time_offset(consequent_ast))
        start_t = max(0, -min_offset)
        # Must stop early: step checks naturally use t+1, plus any additional offset
        end_t = len(trajectory) - max(1, max_offset)

        total_checks = max(0, end_t - start_t)
        antecedent_hits = 0

        for t in range(start_t, end_t):
            try:
                # Check antecedent at time t
                ante_result = self.evaluate_at_time(antecedent_ast, trajectory, t)

                if ante_result:
                    antecedent_hits += 1
                    # Check consequent (which should reference t+1)
                    cons_result = self.evaluate_at_time(consequent_ast, trajectory, t)

                    if not cons_result:
                        vacuity = VacuityReport(
                            antecedent_true_count=antecedent_hits,
                            consequent_evaluated_count=antecedent_hits,
                            total_checks=total_checks,
                            is_vacuous=False,
                        )
                        return False, t, {
                            "t": t,
                            "antecedent": True,
                            "consequent": False,
                        }, vacuity
            except ASTEvaluationError as e:
                vacuity = VacuityReport(antecedent_true_count=antecedent_hits, total_checks=total_checks)
                return False, t, {"error": str(e)}, vacuity
            except Exception as e:
                vacuity = VacuityReport(antecedent_true_count=antecedent_hits, total_checks=total_checks)
                return False, t, {"error": f"Evaluation error: {e}"}, vacuity

        vacuity = VacuityReport(
            antecedent_true_count=antecedent_hits,
            consequent_evaluated_count=antecedent_hits,
            total_checks=total_checks,
            is_vacuous=antecedent_hits == 0,
        )
        return True, None, None, vacuity

    def check_implication_state(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None, VacuityReport]:
        """Check implication_state: for all t, if antecedent(t) then consequent(t).

        If the AST references past times (e.g., t-1), evaluation starts at
        the appropriate time step to avoid out-of-bounds errors.
        If the AST references future times (e.g., t+1), evaluation stops early.

        If no "=>" at root, treats the entire AST as the consequent with trivially
        true antecedent (effectively an invariant).

        Returns:
            Tuple of (passed, t_fail, details, vacuity)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        # Handle ASTs without explicit implication - treat as "true => claim"
        if ast.get("op") == "=>":
            antecedent_ast = ast["lhs"]
            consequent_ast = ast["rhs"]
        else:
            antecedent_ast = {"const": 1}  # Always true
            consequent_ast = ast

        # Determine time bounds based on references
        min_offset = min(get_min_time_offset(antecedent_ast), get_min_time_offset(consequent_ast))
        max_offset = max(get_max_time_offset(antecedent_ast), get_max_time_offset(consequent_ast))
        start_t = max(0, -min_offset)
        end_t = len(trajectory) - max_offset

        total_checks = max(0, end_t - start_t)
        antecedent_hits = 0

        for t in range(start_t, end_t):
            try:
                ante_result = self.evaluate_at_time(antecedent_ast, trajectory, t)

                if ante_result:
                    antecedent_hits += 1
                    cons_result = self.evaluate_at_time(consequent_ast, trajectory, t)

                    if not cons_result:
                        vacuity = VacuityReport(
                            antecedent_true_count=antecedent_hits,
                            consequent_evaluated_count=antecedent_hits,
                            total_checks=total_checks,
                            is_vacuous=False,
                        )
                        return False, t, {
                            "t": t,
                            "antecedent": True,
                            "consequent": False,
                        }, vacuity
            except ASTEvaluationError as e:
                vacuity = VacuityReport(antecedent_true_count=antecedent_hits, total_checks=total_checks)
                return False, t, {"error": str(e)}, vacuity
            except Exception as e:
                vacuity = VacuityReport(antecedent_true_count=antecedent_hits, total_checks=total_checks)
                return False, t, {"error": f"Evaluation error: {e}"}, vacuity

        vacuity = VacuityReport(
            antecedent_true_count=antecedent_hits,
            consequent_evaluated_count=antecedent_hits,
            total_checks=total_checks,
            is_vacuous=antecedent_hits == 0,
        )
        return True, None, None, vacuity

    def check_monotone(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None, VacuityReport]:
        """Check monotone: f(t+1) op f(t) for all t.

        The claim_ast should be a comparison between f at t+1 and f at t.

        Returns:
            Tuple of (passed, t_fail, details, vacuity)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        # Determine time bounds based on references
        min_offset = get_min_time_offset(ast)
        max_offset = get_max_time_offset(ast)
        start_t = max(0, -min_offset)
        # Monotone templates inherently reference t+1, account for additional offsets
        end_t = len(trajectory) - max(1, max_offset)

        total_checks = max(0, end_t - start_t)
        vacuity = VacuityReport(total_checks=total_checks)

        for t in range(start_t, end_t):
            try:
                result = self.evaluate_at_time(ast, trajectory, t)
                if not result:
                    return False, t, {"t": t}, vacuity
            except ASTEvaluationError as e:
                return False, t, {"error": str(e)}, vacuity
            except Exception as e:
                return False, t, {"error": f"Evaluation error: {e}"}, vacuity

        return True, None, None, vacuity

    def check_bound(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None, VacuityReport]:
        """Check bound: f(t) op k for all t.

        If the AST references past times (e.g., t-1), evaluation starts at
        the appropriate time step to avoid out-of-bounds errors.
        If the AST references future times (e.g., t+1), evaluation stops early.

        Returns:
            Tuple of (passed, t_fail, details, vacuity)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        # Determine time bounds based on references
        min_offset = get_min_time_offset(ast)
        max_offset = get_max_time_offset(ast)
        start_t = max(0, -min_offset)
        end_t = len(trajectory) - max_offset

        total_checks = max(0, end_t - start_t)
        vacuity = VacuityReport(total_checks=total_checks)

        for t in range(start_t, end_t):
            try:
                result = self.evaluate_at_time(ast, trajectory, t)
                if not result:
                    return False, t, {"t": t}, vacuity
            except ASTEvaluationError as e:
                return False, t, {"error": str(e)}, vacuity
            except Exception as e:
                return False, t, {"error": f"Evaluation error: {e}"}, vacuity

        return True, None, None, vacuity

    def check_eventually(self, trajectory: Trajectory, horizon: int) -> tuple[bool, int | None, dict | None, VacuityReport]:
        """Check eventually: if antecedent(t0) then exists t in [t0..t0+H] where consequent(t).

        If no "=>" at root, treats the entire AST as the consequent with trivially
        true antecedent (meaning "eventually claim becomes true").

        Returns:
            Tuple of (passed, t_fail, details, vacuity)
        """
        ast = self.law.claim_ast
        if ast is None:
            raise ASTEvaluationError("No claim_ast provided")

        # Handle ASTs without explicit implication - treat as "true => claim"
        if ast.get("op") == "=>":
            antecedent_ast = ast["lhs"]
            consequent_ast = ast["rhs"]
        else:
            antecedent_ast = {"const": 1}  # Always true
            consequent_ast = ast

        # Determine time bounds based on references
        ante_min = get_min_time_offset(antecedent_ast)
        cons_max = get_max_time_offset(consequent_ast)
        start_t0 = max(0, -ante_min)
        # The trajectory limit for the search window
        traj_limit = len(trajectory) - cons_max

        total_checks = max(0, traj_limit - start_t0)
        antecedent_hits = 0
        consequent_checks = 0

        for t0 in range(start_t0, traj_limit):
            try:
                ante_result = self.evaluate_at_time(antecedent_ast, trajectory, t0)

                if ante_result:
                    antecedent_hits += 1
                    # Search for consequent in [t0..t0+H], bounded by trajectory limit
                    found = False
                    end = min(t0 + horizon + 1, traj_limit)

                    for t in range(t0, end):
                        consequent_checks += 1
                        cons_result = self.evaluate_at_time(consequent_ast, trajectory, t)
                        if cons_result:
                            found = True
                            break

                    if not found:
                        vacuity = VacuityReport(
                            antecedent_true_count=antecedent_hits,
                            consequent_evaluated_count=consequent_checks,
                            total_checks=total_checks,
                            is_vacuous=False,
                        )
                        return False, t0, {
                            "t0": t0,
                            "horizon": horizon,
                            "search_end": end - 1,
                        }, vacuity
            except ASTEvaluationError as e:
                vacuity = VacuityReport(antecedent_true_count=antecedent_hits, total_checks=total_checks)
                return False, t0, {"error": str(e)}, vacuity
            except Exception as e:
                vacuity = VacuityReport(antecedent_true_count=antecedent_hits, total_checks=total_checks)
                return False, t0, {"error": f"Evaluation error: {e}"}, vacuity

        vacuity = VacuityReport(
            antecedent_true_count=antecedent_hits,
            consequent_evaluated_count=consequent_checks,
            total_checks=total_checks,
            is_vacuous=antecedent_hits == 0,
        )
        return True, None, None, vacuity

    def check(self, trajectory: Trajectory) -> tuple[bool, int | None, dict | None, VacuityReport]:
        """Check the claim against a trajectory based on template type.

        Returns:
            Tuple of (passed, t_fail, details, vacuity)
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
