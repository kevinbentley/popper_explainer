"""Claim compiler: converts CandidateLaw into executable TrajectoryChecker.

This is the bridge between the schema (what laws look like) and the
templates (how laws are checked).
"""

from typing import Callable

from src.claims.ast_evaluator import ASTClaimEvaluator, create_ast_checker
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
    LocalTransitionChecker,
    MonotoneChecker,
    SymmetryCommutationChecker,
    TemplateChecker,
)
from src.universe.transforms import list_transforms
from src.universe.types import State


class CompilationError(Exception):
    """Raised when law compilation fails."""

    pass


class ASTCheckerAdapter(TemplateChecker):
    """Adapts ASTClaimEvaluator to the TemplateChecker interface.

    This allows laws with claim_ast to use the same interface as
    string-based compiled laws.
    """

    def __init__(self, ast_evaluator: ASTClaimEvaluator):
        self._evaluator = ast_evaluator

    def check(self, trajectory):
        """Check the law against a trajectory.

        Adapts the ASTClaimEvaluator.check() return format to CheckResult.
        """
        from src.claims.templates import CheckResult, Violation
        from src.claims.vacuity import VacuityReport

        passed, t_fail, details, vacuity = self._evaluator.check(trajectory)

        if passed:
            return CheckResult(passed=True, vacuity=vacuity or VacuityReport())

        # Build violation from details
        violation = Violation(
            t=t_fail if t_fail is not None else 0,
            state=trajectory[t_fail] if t_fail is not None and t_fail < len(trajectory) else trajectory[0],
            details=details or {},
            message=f"AST claim violated at t={t_fail}: {details}" if details else f"AST claim violated at t={t_fail}",
        )
        return CheckResult(passed=False, violation=violation, vacuity=vacuity or VacuityReport())


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
        # Try AST-based compilation first if claim_ast is present
        # This handles structured claims from the LLM more robustly
        if law.claim_ast is not None:
            ast_evaluator = create_ast_checker(law)
            if ast_evaluator is not None:
                return ASTCheckerAdapter(ast_evaluator)

        # Fall back to string-based compilation
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
        elif law.template == Template.LOCAL_TRANSITION:
            return self._compile_local_transition(law)
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
        """Compile a symmetry_commutation law.

        Handles parameterized transforms like shift_k:
        - "shift_k" with default k=1
        - "shift_1", "shift_2", etc. with explicit k value
        - The k value is extracted and passed to the checker
        """
        if law.transform is None:
            raise CompilationError("Symmetry_commutation requires 'transform' field")

        transform_name = law.transform
        shift_k_value: int | None = None

        # Handle "shift_N" patterns (e.g., "shift_1", "shift_2")
        import re
        shift_match = re.match(r'^shift_(\d+)$', transform_name)
        if shift_match:
            shift_k_value = int(shift_match.group(1))
            transform_name = "shift_k"  # Normalize to base transform name

        # Validate the base transform name
        if transform_name not in list_transforms():
            raise CompilationError(
                f"Unknown transform '{law.transform}'. "
                f"Available: {list_transforms()} (shift_k accepts shift_N format, e.g., shift_1)"
            )

        return SymmetryCommutationChecker(transform_name, law.quantifiers.T, shift_k_value)

    def _compile_local_transition(self, law: CandidateLaw) -> LocalTransitionChecker:
        """Compile a local_transition law.

        Local transition expresses per-cell behavior:
        ∀t,i: state[i] == trigger_symbol at t → state[i] result_op result_symbol at t+1

        With optional neighbor_pattern for context-dependent rules:
        ∀t,i: state[i] == trigger_symbol AND neighbor_config(i) == pattern at t
              → state[i] result_op result_symbol at t+1

        With optional required_parity for index-parity-dependent rules:
        ∀t,i: state[i] == trigger_symbol AND i % 2 == required_parity at t
              → state[i] result_op result_symbol at t+1

        Example: "Each X cell resolves in one step"
        - trigger_symbol='X', result_op='!=', result_symbol='X'

        Example: "Empty cells with >.< neighborhood become X"
        - trigger_symbol='.', neighbor_pattern='>.<', result_op='==', result_symbol='X'

        Example: "Right-movers at even indices become left-movers"
        - trigger_symbol='>', required_parity=0, result_op='==', result_symbol='<'
        """
        # Try to infer missing fields from claim text
        trigger, result_op, result_symbol, neighbor_pattern = self._infer_local_transition_fields(law)

        if trigger is None:
            raise CompilationError("Local_transition requires 'trigger_symbol' field")

        if result_op is None:
            raise CompilationError("Local_transition requires 'result_op' field")

        if result_symbol is None:
            raise CompilationError("Local_transition requires 'result_symbol' field")

        # Validate that result_op is == or !=
        if result_op not in (ComparisonOp.EQ, ComparisonOp.NE):
            raise CompilationError(
                f"Local_transition result_op must be '==' or '!=', got '{result_op.value}'"
            )

        # Validate neighbor_pattern if provided
        if neighbor_pattern is not None:
            if len(neighbor_pattern) != 3:
                raise CompilationError(
                    f"neighbor_pattern must be exactly 3 characters, got '{neighbor_pattern}'"
                )
            # Accept both physical (.><X) and abstract (WABK) symbols
            valid_symbols = set('.><XWABK')
            if not all(c in valid_symbols for c in neighbor_pattern):
                raise CompilationError(
                    f"neighbor_pattern must contain only valid symbols, got '{neighbor_pattern}'"
                )

        # Extract and validate required_parity if provided
        required_parity = law.required_parity
        if required_parity is not None:
            if required_parity not in (0, 1):
                raise CompilationError(
                    f"required_parity must be 0 (even) or 1 (odd), got {required_parity}"
                )

        return LocalTransitionChecker(
            trigger_symbol=trigger,
            result_op=result_op,
            result_symbol=result_symbol,
            neighbor_pattern=neighbor_pattern,
            required_parity=required_parity,
        )

    def _infer_local_transition_fields(
        self, law: CandidateLaw
    ) -> tuple[str | None, ComparisonOp | None, str | None, str | None]:
        """Infer trigger_symbol, result_op, result_symbol, neighbor_pattern from claim text.

        Returns tuple of (trigger_symbol, result_op, result_symbol, neighbor_pattern).
        Uses law's explicit fields if set, otherwise parses claim text.
        """
        import re

        trigger = law.trigger_symbol
        result_op = law.result_op
        result_symbol = law.result_symbol
        neighbor_pattern = law.neighbor_pattern

        # If all required fields are already set, return them
        if trigger is not None and result_op is not None and result_symbol is not None:
            return trigger, result_op, result_symbol, neighbor_pattern

        claim = law.claim.lower()
        # Fix escaped characters that may appear in LLM output
        claim = claim.replace("\\!", "!")

        # Symbol mapping for common terms
        symbol_map = {
            "x": "X",
            "collision": "X",
            "empty": ".",
            "right-mover": ">",
            "right mover": ">",
            "rightmover": ">",
            "left-mover": "<",
            "left mover": "<",
            "leftmover": "<",
        }

        # Try to extract neighbor_pattern from claim text if not set
        if neighbor_pattern is None:
            # Look for patterns like "neighbor_config(i)=='>.<'" or "neighborhood '>.<'"
            # or "with pattern '>.<'" or just quoted 3-char patterns in context
            pattern_matches = [
                r"neighbor[_\s]*(?:config|hood)\s*(?:\(i\))?\s*==\s*['\"]([<>.X]{3})['\"]",
                r"neighborhood\s+['\"]([<>.X]{3})['\"]",
                r"pattern\s+['\"]([<>.X]{3})['\"]",
                r"with\s+['\"]([<>.X]{3})['\"]",
            ]
            for p in pattern_matches:
                match = re.search(p, claim, re.IGNORECASE)
                if match:
                    neighbor_pattern = match.group(1)
                    break

        # Pattern: "X cell(s) becomes/resolves/transitions to non-X"
        # -> trigger='X', result_op='!=', result_symbol='X'
        patterns = [
            # "if state[i]=='X' at t, then state[i]!='X' at t+1" or "state[i]=='>' ... state[i]!='.'"
            # Most common LLM format - look for quoted symbols with operators
            (r"state\[i\]\s*==\s*['\"]([<>.X])['\"].*?state\[i\]\s*(!=|==)\s*['\"]([<>.X])['\"]", None),

            # "Each '>' cell becomes '.' at its position"
            (r"each\s+['\"]([<>.X])['\"]?\s*cell\s+becomes?\s+['\"]([<>.X])['\"]", "=="),

            # "Each X cell becomes non-X" or "X cells resolve"
            (r"(?:each\s+)?(\w+)\s+cell[s]?\s+(?:becomes?|resolves?|transitions?\s+to)\s+non-(\w+)", "!="),

            # "X resolves in one step" or "X disappears"
            (r"(\w+)\s+(?:resolves?|disappears?|changes?)\s+in\s+(?:one|1|the\s+next)\s+step", "!="),

            # "X at position i becomes Y"
            (r"(\w+)\s+at\s+(?:position\s+)?i\s+becomes?\s+(\w+)", "=="),

            # "Empty cells stay empty"
            (r"(\w+)\s+cells?\s+(?:stay|remain)[s]?\s+(\w+)", "=="),
        ]

        for pattern, default_op in patterns:
            match = re.search(pattern, claim, re.IGNORECASE)
            if match:
                groups = match.groups()

                if len(groups) == 2:
                    # Patterns with just trigger and result
                    trigger_str, result_str = groups
                    inferred_trigger = symbol_map.get(trigger_str.lower(), trigger_str)
                    inferred_result = symbol_map.get(result_str.lower(), result_str)

                    if trigger is None:
                        trigger = inferred_trigger
                    if result_symbol is None:
                        result_symbol = inferred_result
                    if result_op is None and default_op:
                        result_op = ComparisonOp.NE if default_op == "!=" else ComparisonOp.EQ

                elif len(groups) == 3:
                    # Pattern with explicit operator
                    trigger_str, op_str, result_str = groups
                    inferred_trigger = symbol_map.get(trigger_str.lower(), trigger_str)
                    inferred_result = symbol_map.get(result_str.lower(), result_str)

                    if trigger is None:
                        trigger = inferred_trigger
                    if result_symbol is None:
                        result_symbol = inferred_result
                    if result_op is None:
                        result_op = ComparisonOp.NE if "!" in op_str else ComparisonOp.EQ

                break

        # Special case: "X resolves" without explicit result means X becomes non-X
        if trigger is not None and result_symbol is None:
            # Common inference: if trigger is X and claim mentions "resolves" or "disappears"
            if any(word in claim for word in ["resolve", "disappear", "change"]):
                result_symbol = trigger
                if result_op is None:
                    result_op = ComparisonOp.NE

        return trigger, result_op, result_symbol, neighbor_pattern

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
