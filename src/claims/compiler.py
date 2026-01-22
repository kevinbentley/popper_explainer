"""Claim compiler: converts CandidateLaw into executable TrajectoryChecker.

This is the bridge between the schema (what laws look like) and the
templates (how laws are checked).
"""

from typing import Callable

from src.claims.expr_ast import Expr
from src.claims.expr_evaluator import evaluate_expression
from src.claims.expr_parser import ExpressionParser, ParseError
from src.claims.schema import (
    CandidateLaw,
    ComparisonOp,
    MonotoneDirection,
    Observable,
    Precondition,
    Template,
)
from src.claims.templates import (
    BoundChecker,
    EventuallyChecker,
    ImplicationStateChecker,
    ImplicationStepChecker,
    InvariantChecker,
    MonotoneChecker,
    SymmetryCommutationChecker,
    TemplateChecker,
)
from src.universe.transforms import list_transforms
from src.universe.types import State


class CompilationError(Exception):
    """Raised when law compilation fails."""

    pass


class ClaimCompiler:
    """Compiles CandidateLaw objects into executable TrajectoryCheckers."""

    def __init__(self):
        self._parser = ExpressionParser()
        self._compiled_observables: dict[str, Expr] = {}

    def compile(self, law: CandidateLaw) -> TemplateChecker:
        """Compile a candidate law into a trajectory checker.

        Args:
            law: The candidate law to compile

        Returns:
            A TemplateChecker that can evaluate the law against trajectories

        Raises:
            CompilationError: If compilation fails
        """
        # First, compile all observable expressions
        self._compiled_observables = {}
        for obs in law.observables:
            try:
                expr = self._parser.parse(obs.expr)
                self._compiled_observables[obs.name] = expr
            except ParseError as e:
                raise CompilationError(
                    f"Failed to parse observable '{obs.name}': {obs.expr}\n{e}"
                ) from e

        # Dispatch to template-specific compiler
        if law.template == Template.INVARIANT:
            return self._compile_invariant(law)
        elif law.template == Template.MONOTONE:
            return self._compile_monotone(law)
        elif law.template == Template.BOUND:
            return self._compile_bound(law)
        elif law.template == Template.IMPLICATION_STEP:
            return self._compile_implication_step(law)
        elif law.template == Template.IMPLICATION_STATE:
            return self._compile_implication_state(law)
        elif law.template == Template.EVENTUALLY:
            return self._compile_eventually(law)
        elif law.template == Template.SYMMETRY_COMMUTATION:
            return self._compile_symmetry(law)
        else:
            raise CompilationError(f"Unknown template: {law.template}")

    def _compile_invariant(self, law: CandidateLaw) -> InvariantChecker:
        """Compile an invariant law."""
        if len(law.observables) != 1:
            raise CompilationError(
                f"Invariant template requires exactly 1 observable, got {len(law.observables)}"
            )

        obs = law.observables[0]
        expr = self._compiled_observables[obs.name]
        return InvariantChecker(expr)

    def _compile_monotone(self, law: CandidateLaw) -> MonotoneChecker:
        """Compile a monotone law."""
        if len(law.observables) != 1:
            raise CompilationError(
                f"Monotone template requires exactly 1 observable, got {len(law.observables)}"
            )

        if law.direction is None:
            raise CompilationError("Monotone template requires 'direction' field")

        obs = law.observables[0]
        expr = self._compiled_observables[obs.name]
        return MonotoneChecker(expr, law.direction)

    def _compile_bound(self, law: CandidateLaw) -> BoundChecker:
        """Compile a bound law."""
        if len(law.observables) != 1:
            raise CompilationError(
                f"Bound template requires exactly 1 observable, got {len(law.observables)}"
            )

        if law.bound_value is None:
            raise CompilationError("Bound template requires 'bound_value' field")

        if law.bound_op is None:
            raise CompilationError("Bound template requires 'bound_op' field")

        obs = law.observables[0]
        expr = self._compiled_observables[obs.name]
        return BoundChecker(expr, law.bound_op, law.bound_value)

    def _compile_implication_step(self, law: CandidateLaw) -> ImplicationStepChecker:
        """Compile an implication_step law.

        The claim format should indicate which observable is the antecedent
        and which is the consequent. For simplicity, we expect exactly 2
        observables and parse the claim to determine the structure.
        """
        if len(law.observables) < 1:
            raise CompilationError("Implication_step requires at least 1 observable")

        # Parse the claim to extract antecedent and consequent conditions
        # Expected format: "P(t) -> Q(t+1)" or similar
        # For now, we use a simplified approach based on observable order
        antecedent, consequent = self._parse_implication_claim(law)
        return ImplicationStepChecker(antecedent, consequent)

    def _compile_implication_state(self, law: CandidateLaw) -> ImplicationStateChecker:
        """Compile an implication_state law."""
        if len(law.observables) < 1:
            raise CompilationError("Implication_state requires at least 1 observable")

        antecedent, consequent = self._parse_implication_claim(law)
        return ImplicationStateChecker(antecedent, consequent)

    def _compile_eventually(self, law: CandidateLaw) -> EventuallyChecker:
        """Compile an eventually law."""
        if len(law.observables) < 1:
            raise CompilationError("Eventually requires at least 1 observable")

        if law.quantifiers.H is None:
            raise CompilationError("Eventually template requires 'H' in quantifiers")

        antecedent, consequent = self._parse_implication_claim(law)
        return EventuallyChecker(antecedent, consequent, law.quantifiers.H)

    def _compile_symmetry(self, law: CandidateLaw) -> SymmetryCommutationChecker:
        """Compile a symmetry_commutation law."""
        if law.transform is None:
            raise CompilationError("Symmetry_commutation requires 'transform' field")

        if law.transform not in list_transforms():
            raise CompilationError(
                f"Unknown transform '{law.transform}'. "
                f"Available: {list_transforms()}"
            )

        return SymmetryCommutationChecker(law.transform, law.quantifiers.T)

    def _parse_implication_claim(
        self, law: CandidateLaw
    ) -> tuple[Callable[[State], bool], Callable[[State], bool]]:
        """Parse an implication claim into antecedent and consequent predicates.

        This is a simplified parser that handles common patterns:
        - "obs_name(t) > 0 -> consequent"
        - "obs_name(t) == 0 -> consequent"

        For complex claims, extend this parser.
        """
        claim = law.claim.lower()

        # Try to find the implication arrow
        if "->" not in claim:
            raise CompilationError(
                f"Implication claim must contain '->': {law.claim}"
            )

        parts = claim.split("->")
        if len(parts) != 2:
            raise CompilationError(f"Invalid implication format: {law.claim}")

        ante_str = parts[0].strip()
        cons_str = parts[1].strip()

        antecedent = self._parse_predicate(ante_str, law)
        consequent = self._parse_predicate(cons_str, law)

        return antecedent, consequent

    def _parse_predicate(self, pred_str: str, law: CandidateLaw) -> Callable[[State], bool]:
        """Parse a predicate string into a callable.

        Handles patterns like:
        - "obs_name(t) > 0"
        - "obs_name(t) == obs_name(0)"
        - "obs_name(t+1) <= obs_name(t)"
        """
        # Remove time references for simpler parsing
        pred_str = pred_str.replace("(t)", "").replace("(t+1)", "").replace("(0)", "")
        pred_str = pred_str.strip()

        # Find comparison operator
        ops = ["<=", ">=", "==", "!=", "<", ">"]
        op = None
        op_pos = -1
        for o in ops:
            pos = pred_str.find(o)
            if pos != -1:
                if op is None or pos < op_pos:
                    op = o
                    op_pos = pos

        if op is None:
            # No comparison - treat as "expression > 0" (truthy)
            expr = self._resolve_expression(pred_str, law)
            return lambda state, e=expr: evaluate_expression(e, state) > 0

        lhs_str = pred_str[:op_pos].strip()
        rhs_str = pred_str[op_pos + len(op):].strip()

        lhs_expr = self._resolve_expression(lhs_str, law)

        # RHS could be a number or an observable
        try:
            rhs_value = int(rhs_str)
            rhs_expr = None
        except ValueError:
            rhs_expr = self._resolve_expression(rhs_str, law)
            rhs_value = None

        def predicate(state: State) -> bool:
            lhs_val = evaluate_expression(lhs_expr, state)
            if rhs_expr is not None:
                rhs_val = evaluate_expression(rhs_expr, state)
            else:
                rhs_val = rhs_value

            if op == "==":
                return lhs_val == rhs_val
            elif op == "!=":
                return lhs_val != rhs_val
            elif op == "<":
                return lhs_val < rhs_val
            elif op == "<=":
                return lhs_val <= rhs_val
            elif op == ">":
                return lhs_val > rhs_val
            elif op == ">=":
                return lhs_val >= rhs_val
            return False

        return predicate

    def _resolve_expression(self, expr_str: str, law: CandidateLaw) -> Expr:
        """Resolve an expression string, checking observable names first."""
        expr_str = expr_str.strip()

        # Check if it's a defined observable name
        if expr_str in self._compiled_observables:
            return self._compiled_observables[expr_str]

        # Try to parse as an expression
        try:
            return self._parser.parse(expr_str)
        except ParseError as e:
            raise CompilationError(
                f"Cannot resolve expression '{expr_str}': {e}"
            ) from e


def compile_precondition(precondition: Precondition, law: CandidateLaw) -> Callable[[State], bool]:
    """Compile a precondition into a callable predicate.

    Args:
        precondition: The precondition to compile
        law: The law containing observable definitions

    Returns:
        A function that takes a state and returns True if precondition is met
    """
    parser = ExpressionParser()

    # Compile LHS
    if precondition.lhs == "grid_length":
        def get_lhs(state: State) -> int:
            return len(state)
    else:
        # Look for observable in law
        obs_expr = None
        for obs in law.observables:
            if obs.name == precondition.lhs:
                obs_expr = parser.parse(obs.expr)
                break

        if obs_expr is None:
            # Try parsing as expression directly
            try:
                obs_expr = parser.parse(precondition.lhs)
            except ParseError:
                raise CompilationError(
                    f"Unknown observable in precondition: {precondition.lhs}"
                )

        def get_lhs(state: State, expr=obs_expr) -> int:
            return evaluate_expression(expr, state)

    # Compile RHS
    if isinstance(precondition.rhs, int):
        rhs_value = precondition.rhs

        def get_rhs(state: State) -> int:
            return rhs_value
    else:
        # RHS is an observable name
        rhs_expr = None
        for obs in law.observables:
            if obs.name == precondition.rhs:
                rhs_expr = parser.parse(obs.expr)
                break

        if rhs_expr is None:
            try:
                rhs_expr = parser.parse(precondition.rhs)
            except ParseError:
                raise CompilationError(
                    f"Unknown observable in precondition RHS: {precondition.rhs}"
                )

        def get_rhs(state: State, expr=rhs_expr) -> int:
            return evaluate_expression(expr, state)

    # Build the comparison function
    op = precondition.op

    def check_precondition(state: State) -> bool:
        lhs = get_lhs(state)
        rhs = get_rhs(state)

        if op == ComparisonOp.EQ:
            return lhs == rhs
        elif op == ComparisonOp.NE:
            return lhs != rhs
        elif op == ComparisonOp.LT:
            return lhs < rhs
        elif op == ComparisonOp.LE:
            return lhs <= rhs
        elif op == ComparisonOp.GT:
            return lhs > rhs
        elif op == ComparisonOp.GE:
            return lhs >= rhs
        return False

    return check_precondition
