"""Template checkers for evaluating laws against trajectories.

Each template defines a specific pattern of claim:
- invariant: f(t) == f(0) for all t
- monotone: f(t+1) <= f(t) for all t (or >=)
- implication_step: P(t) → Q(t+1) for all t
- implication_state: P(t) → Q(t) for all t
- eventually: P(t0) → ∃t∈[t0..t0+H]: Q(t)
- symmetry_commutation: evolve(T(S), t) == T(evolve(S, t))
- bound: f(t) <= k for all t (or >=)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from src.claims.expr_ast import Expr
from src.claims.expr_evaluator import evaluate_expression
from src.claims.schema import ComparisonOp, MonotoneDirection, Template
from src.universe.simulator import run
from src.universe.transforms import get_transform
from src.universe.types import State, Trajectory


@dataclass
class Violation:
    """A violation of a law - a counterexample witness."""

    t: int  # Time step where violation occurred
    state: State  # State at time t
    details: dict[str, Any] = field(default_factory=dict)  # Observable values, etc.
    message: str = ""  # Human-readable description


from src.claims.vacuity import VacuityReport


@dataclass
class CheckResult:
    """Result of checking a law against a trajectory."""

    passed: bool
    violation: Violation | None = None
    vacuity: VacuityReport = field(default_factory=VacuityReport)


class TemplateChecker(ABC):
    """Abstract base class for template checkers."""

    @abstractmethod
    def check(self, trajectory: Trajectory) -> CheckResult:
        """Check the law against a trajectory.

        Args:
            trajectory: List of states from t=0 to t=T

        Returns:
            CheckResult with pass/fail and any violation
        """
        pass


class InvariantChecker(TemplateChecker):
    """Checker for invariant template: f(t) == f(0) for all t."""

    def __init__(self, observable_expr: Expr | None = None, *, observable_fn: Callable | None = None):
        if observable_fn is not None:
            self._eval = observable_fn
            self._is_temporal = getattr(observable_fn, 'is_temporal', False)
        elif observable_expr is not None:
            self.observable_expr = observable_expr
            self._eval = lambda state: evaluate_expression(self.observable_expr, state)
            self._is_temporal = False
        else:
            raise ValueError("Must provide either observable_expr or observable_fn")

    def _eval_at(self, trajectory: Trajectory, t: int) -> float | int:
        """Evaluate the observable at time t, passing next_state for temporal probes."""
        if self._is_temporal:
            next_state = trajectory[t + 1] if t + 1 < len(trajectory) else None
            return self._eval(trajectory[t], next_state=next_state)
        return self._eval(trajectory[t])

    def check(self, trajectory: Trajectory) -> CheckResult:
        if not trajectory:
            return CheckResult(passed=True)

        end = len(trajectory) - 1 if self._is_temporal else len(trajectory)
        if end <= 0:
            return CheckResult(passed=True)

        initial_value = self._eval_at(trajectory, 0)

        for t in range(end):
            value = self._eval_at(trajectory, t)
            if value != initial_value:
                return CheckResult(
                    passed=False,
                    violation=Violation(
                        t=t,
                        state=trajectory[t],
                        details={"expected": initial_value, "actual": value},
                        message=f"Invariant violated at t={t}: expected {initial_value}, got {value}",
                    ),
                )

        return CheckResult(passed=True)


class MonotoneChecker(TemplateChecker):
    """Checker for monotone template: f(t+1) <= f(t) or f(t+1) >= f(t)."""

    def __init__(self, observable_expr: Expr | None = None, direction: MonotoneDirection = MonotoneDirection.NON_INCREASING, *, observable_fn: Callable | None = None):
        self.direction = direction
        if observable_fn is not None:
            self._eval = observable_fn
            self._is_temporal = getattr(observable_fn, 'is_temporal', False)
        elif observable_expr is not None:
            self.observable_expr = observable_expr
            self._eval = lambda state: evaluate_expression(self.observable_expr, state)
            self._is_temporal = False
        else:
            raise ValueError("Must provide either observable_expr or observable_fn")

    def _eval_at(self, trajectory: Trajectory, t: int) -> float | int:
        """Evaluate the observable at time t, passing next_state for temporal probes."""
        if self._is_temporal:
            next_state = trajectory[t + 1] if t + 1 < len(trajectory) else None
            return self._eval(trajectory[t], next_state=next_state)
        return self._eval(trajectory[t])

    def check(self, trajectory: Trajectory) -> CheckResult:
        if len(trajectory) < 2:
            return CheckResult(passed=True)

        end = len(trajectory) - 1 if self._is_temporal else len(trajectory)

        prev_value = self._eval_at(trajectory, 0)

        for t in range(1, end):
            curr_value = self._eval_at(trajectory, t)

            violated = False
            if self.direction == MonotoneDirection.NON_INCREASING:
                violated = curr_value > prev_value
            else:  # NON_DECREASING
                violated = curr_value < prev_value

            if violated:
                return CheckResult(
                    passed=False,
                    violation=Violation(
                        t=t,
                        state=trajectory[t],
                        details={
                            "prev_value": prev_value,
                            "curr_value": curr_value,
                            "direction": self.direction.value,
                        },
                        message=f"Monotone ({self.direction.value}) violated at t={t}: "
                        f"f(t-1)={prev_value}, f(t)={curr_value}",
                    ),
                )

            prev_value = curr_value

        return CheckResult(passed=True)


class BoundChecker(TemplateChecker):
    """Checker for bound template: f(t) op k for all t."""

    def __init__(self, observable_expr: Expr | None = None, op: ComparisonOp = ComparisonOp.LE, bound: int = 0, *, observable_fn: Callable | None = None):
        self.op = op
        self.bound = bound
        if observable_fn is not None:
            self._eval = observable_fn
            self._is_temporal = getattr(observable_fn, 'is_temporal', False)
        elif observable_expr is not None:
            self.observable_expr = observable_expr
            self._eval = lambda state: evaluate_expression(self.observable_expr, state)
            self._is_temporal = False
        else:
            raise ValueError("Must provide either observable_expr or observable_fn")

    def _compare(self, value: float | int) -> bool:
        """Check if value satisfies the bound."""
        if self.op == ComparisonOp.LE:
            return value <= self.bound
        elif self.op == ComparisonOp.LT:
            return value < self.bound
        elif self.op == ComparisonOp.GE:
            return value >= self.bound
        elif self.op == ComparisonOp.GT:
            return value > self.bound
        elif self.op == ComparisonOp.EQ:
            return value == self.bound
        elif self.op == ComparisonOp.NE:
            return value != self.bound
        else:
            raise ValueError(f"Unknown comparison operator: {self.op}")

    def _eval_at(self, trajectory: Trajectory, t: int) -> float | int:
        """Evaluate the observable at time t, passing next_state for temporal probes."""
        if self._is_temporal:
            next_state = trajectory[t + 1] if t + 1 < len(trajectory) else None
            return self._eval(trajectory[t], next_state=next_state)
        return self._eval(trajectory[t])

    def check(self, trajectory: Trajectory) -> CheckResult:
        end = len(trajectory) - 1 if self._is_temporal else len(trajectory)

        for t in range(end):
            value = self._eval_at(trajectory, t)
            if not self._compare(value):
                return CheckResult(
                    passed=False,
                    violation=Violation(
                        t=t,
                        state=trajectory[t],
                        details={"value": value, "bound": self.bound, "op": self.op.value},
                        message=f"Bound violated at t={t}: {value} {self.op.value} {self.bound} is false",
                    ),
                )

        return CheckResult(passed=True)


class ImplicationStepChecker(TemplateChecker):
    """Checker for implication_step template: P(t) → Q(t+1)."""

    def __init__(self, antecedent: Callable[[State], bool], consequent: Callable[[State], bool]):
        self.antecedent = antecedent
        self.consequent = consequent

    def check(self, trajectory: Trajectory) -> CheckResult:
        if len(trajectory) < 2:
            return CheckResult(passed=True)

        vacuity = VacuityReport()
        vacuity.total_checks = len(trajectory) - 1  # Step checks

        for t in range(len(trajectory) - 1):
            if self.antecedent(trajectory[t]):
                vacuity.antecedent_true_count += 1
                vacuity.consequent_evaluated_count += 1

                if not self.consequent(trajectory[t + 1]):
                    return CheckResult(
                        passed=False,
                        violation=Violation(
                            t=t,
                            state=trajectory[t],
                            details={"t_plus_1_state": trajectory[t + 1]},
                            message=f"Implication violated: P(t={t}) true but Q(t={t + 1}) false",
                        ),
                        vacuity=vacuity,
                    )

        vacuity.is_vacuous = vacuity.antecedent_true_count == 0
        return CheckResult(passed=True, vacuity=vacuity)


class ImplicationStateChecker(TemplateChecker):
    """Checker for implication_state template: P(t) → Q(t)."""

    def __init__(self, antecedent: Callable[[State], bool], consequent: Callable[[State], bool]):
        self.antecedent = antecedent
        self.consequent = consequent

    def check(self, trajectory: Trajectory) -> CheckResult:
        vacuity = VacuityReport()
        vacuity.total_checks = len(trajectory)  # State checks

        for t, state in enumerate(trajectory):
            if self.antecedent(state):
                vacuity.antecedent_true_count += 1
                vacuity.consequent_evaluated_count += 1

                if not self.consequent(state):
                    return CheckResult(
                        passed=False,
                        violation=Violation(
                            t=t,
                            state=state,
                            message=f"Implication violated: P(t={t}) true but Q(t={t}) false",
                        ),
                        vacuity=vacuity,
                    )

        vacuity.is_vacuous = vacuity.antecedent_true_count == 0
        return CheckResult(passed=True, vacuity=vacuity)


class EventuallyChecker(TemplateChecker):
    """Checker for eventually template: P(t0) → ∃t∈[t0..t0+H]: Q(t)."""

    def __init__(
        self, antecedent: Callable[[State], bool], consequent: Callable[[State], bool], horizon: int
    ):
        self.antecedent = antecedent
        self.consequent = consequent
        self.horizon = horizon

    def check(self, trajectory: Trajectory) -> CheckResult:
        vacuity = VacuityReport()
        vacuity.total_checks = len(trajectory)  # Each timestep is a potential trigger point

        for t0, state in enumerate(trajectory):
            if self.antecedent(state):
                vacuity.antecedent_true_count += 1

                # Check if Q holds within the horizon
                found = False
                end = min(t0 + self.horizon + 1, len(trajectory))
                for t in range(t0, end):
                    vacuity.consequent_evaluated_count += 1
                    if self.consequent(trajectory[t]):
                        found = True
                        break

                if not found:
                    return CheckResult(
                        passed=False,
                        violation=Violation(
                            t=t0,
                            state=state,
                            details={"horizon": self.horizon, "search_end": end - 1},
                            message=f"Eventually violated: P(t={t0}) true but Q never true in [{t0}..{end - 1}]",
                        ),
                        vacuity=vacuity,
                    )

        vacuity.is_vacuous = vacuity.antecedent_true_count == 0
        return CheckResult(passed=True, vacuity=vacuity)


class SymmetryCommutationChecker(TemplateChecker):
    """Checker for symmetry_commutation: evolve(T(S), t) == T(evolve(S, t)).

    A transformation is a UNIVERSAL SYMMETRY if and only if:
    1. State-level commutation: evolve(T(S)) == T(evolve(S)) for all t
    2. Local probe invariance: Position-dependent observables are preserved

    Translation (shift_k) passes condition 1 (state commutation) because the
    physics are homogeneous, but FAILS condition 2 because cell_at(i) changes
    under translation - the grid indices are fixed parts of the state structure.

    Only mirror_swap is a TRUE universal symmetry - it satisfies both conditions
    because it maps position i to N-1-i with direction swap, preserving the
    physical equivalence of all observables.
    """

    # Transforms that pass state commutation but fail local probe invariance
    # These are "partial symmetries" - physics are homogeneous but positions are fixed
    PARTIAL_SYMMETRIES = {"shift_k"}

    def __init__(self, transform_name: str, time_horizon: int, shift_k_value: int | None = None):
        """Initialize the symmetry checker.

        Args:
            transform_name: Name of the transform to test
            time_horizon: Number of timesteps to evolve
            shift_k_value: For shift_k transform, the k value to use.
                           If None and transform is shift_k, defaults to 1.
                           Can be overridden per-check via check() metadata.
        """
        self.transform_name = transform_name
        self.time_horizon = time_horizon
        self._raw_transform = get_transform(transform_name)
        if self._raw_transform is None:
            raise ValueError(f"Unknown transform: {transform_name}")

        # Store default k value for shift_k
        self._default_k = shift_k_value if shift_k_value is not None else 1

        # Handle transforms that need extra arguments (e.g., shift_k needs k)
        if transform_name == "shift_k":
            # Create a wrapper that uses the default k value
            self._transform = lambda state, k=self._default_k: self._raw_transform(state, k)
        else:
            self._transform = self._raw_transform

    def _get_transform_for_k(self, k: int):
        """Get a transform function for a specific k value."""
        if self.transform_name == "shift_k":
            return lambda state: self._raw_transform(state, k)
        return self._transform

    def check(self, trajectory: Trajectory, k: int | None = None) -> CheckResult:
        """Check symmetry commutation for the initial state.

        Args:
            trajectory: List of states (only trajectory[0] is used)
            k: For shift_k, the shift amount. If None, uses default.

        Note: For symmetry tests, we only check the initial state.
        The trajectory is provided for consistency, but we recompute
        both paths from trajectory[0].
        """
        if not trajectory:
            return CheckResult(passed=True)

        initial_state = trajectory[0]

        # Get the transform function (possibly with custom k for shift_k)
        if self.transform_name == "shift_k" and k is not None:
            transform_fn = self._get_transform_for_k(k)
        else:
            # For non-shift_k or when k not specified, use default
            if self.transform_name == "shift_k":
                k = self._default_k
                transform_fn = self._get_transform_for_k(k)
            else:
                transform_fn = self._transform

        # Path 1: transform then evolve
        try:
            transformed = transform_fn(initial_state)
            path1_trajectory = run(transformed, self.time_horizon)
        except Exception as e:
            return CheckResult(
                passed=False,
                violation=Violation(
                    t=0,
                    state=initial_state,
                    message=f"Transform-then-evolve failed: {e}",
                ),
            )

        # Path 2: evolve then transform
        try:
            path2_trajectory = run(initial_state, self.time_horizon)
            path2_transformed = [transform_fn(s) for s in path2_trajectory]
        except Exception as e:
            return CheckResult(
                passed=False,
                violation=Violation(
                    t=0,
                    state=initial_state,
                    message=f"Evolve-then-transform failed: {e}",
                ),
            )

        # Compare at each time step
        for t in range(len(path1_trajectory)):
            if path1_trajectory[t] != path2_transformed[t]:
                return CheckResult(
                    passed=False,
                    violation=Violation(
                        t=t,
                        state=initial_state,
                        details={
                            "transform_then_evolve": path1_trajectory[t],
                            "evolve_then_transform": path2_transformed[t],
                            "k": k if self.transform_name == "shift_k" else None,
                        },
                        message=f"Symmetry broken at t={t}: "
                        f"evolve(T(S))={path1_trajectory[t]} != T(evolve(S))={path2_transformed[t]}"
                        f"{f' (k={k})' if self.transform_name == 'shift_k' else ''}",
                    ),
                )

        # State-level commutation passed. Now check LOCAL PROBE INVARIANCE.
        # For a UNIVERSAL symmetry, position-dependent observables must be preserved.
        # Transforms like shift_k pass state commutation (physics are homogeneous)
        # but FAIL local probe invariance (grid indices are fixed structure).
        if self.transform_name in self.PARTIAL_SYMMETRIES:
            # Check if cell_at(i) is preserved for at least one position
            # For shift_k: cell_at(S, i) != cell_at(shift_k(S), i) for non-uniform states
            probe_violation = self._check_local_probe_invariance(
                initial_state, transform_fn, k
            )
            if probe_violation is not None:
                return CheckResult(passed=False, violation=probe_violation)

        return CheckResult(passed=True)

    def _check_local_probe_invariance(
        self, state: State, transform_fn, k: int | None
    ) -> Violation | None:
        """Check if local probes (cell_at) are invariant under transformation.

        For a transformation to be a UNIVERSAL symmetry (not just state-commuting),
        position-dependent observables must be preserved. This means:
        - For mirror_swap: cell_at(S, i) maps consistently to cell_at(MS(S), N-1-i)
        - For shift_k: cell_at(S, i) != cell_at(shift_k(S), i), so it FAILS

        Returns:
            Violation if local probe invariance fails, None if it passes
        """
        n = len(state)
        if n == 0:
            return None

        transformed = transform_fn(state)

        # Check multiple positions for local probe violation
        # For shift_k, this should fail unless state is uniform
        violation_positions = []

        for i in range(n):
            if state[i] != transformed[i]:
                violation_positions.append(i)

        if violation_positions:
            # Found positions where cell_at differs - this is NOT a universal symmetry
            pos = violation_positions[0]  # Report first violation
            return Violation(
                t=0,
                state=state,
                details={
                    "violation_type": "local_probe_invariance",
                    "position": pos,
                    "original_cell": state[pos],
                    "transformed_cell": transformed[pos],
                    "transform": self.transform_name,
                    "k": k,
                    "all_violations": violation_positions[:5],  # First 5 violations
                },
                message=f"Local probe invariance violated: cell_at({pos}) = '{state[pos]}' "
                f"in original but '{transformed[pos]}' after {self.transform_name}"
                f"{f'(k={k})' if k else ''}. "
                f"Translation preserves global counts but NOT position-dependent observables. "
                f"Grid indices are fixed structure, not symmetric under translation.",
            )


class LocalTransitionChecker(TemplateChecker):
    """Checker for local_transition template: ∀t,i: state[i]==P at t → state[i] satisfies Q at t+1.

    This template expresses per-cell (micro-level) behavior, as opposed to
    aggregate (macro-level) behavior expressed by count-based templates.

    Example: "Each X cell resolves in one step"
    - trigger_symbol='X', result_op='!=', result_symbol='X'
    - For each position i where state_t[i]=='X', state_{t+1}[i] != 'X'

    With neighbor_pattern: "Empty cells with >.< neighborhood become X"
    - trigger_symbol='.', neighbor_pattern='>.<', result_op='==', result_symbol='X'
    - For each position i where state_t[i]=='.' AND neighbor_config(i)=='>.<',
      state_{t+1}[i] == 'X'

    With required_parity: "Right-movers at even indices become left-movers"
    - trigger_symbol='>', required_parity=0, result_op='==', result_symbol='<'
    - For each even position i where state_t[i]=='>', state_{t+1}[i] == '<'
    """

    def __init__(
        self,
        trigger_symbol: str,
        result_op: ComparisonOp,
        result_symbol: str,
        neighbor_pattern: str | None = None,
        required_parity: int | None = None,
    ):
        self.trigger_symbol = trigger_symbol
        self.result_op = result_op
        self.result_symbol = result_symbol
        self.neighbor_pattern = neighbor_pattern
        self.required_parity = required_parity

        # Validate neighbor_pattern if provided
        if neighbor_pattern is not None:
            if len(neighbor_pattern) != 3:
                raise ValueError(
                    f"neighbor_pattern must be exactly 3 characters, got '{neighbor_pattern}'"
                )
            valid_symbols = set('.><XWABK')  # Accept both physical and abstract symbols
            if not all(c in valid_symbols for c in neighbor_pattern):
                raise ValueError(
                    f"neighbor_pattern must contain only valid symbols, got '{neighbor_pattern}'"
                )

        # Validate required_parity if provided
        if required_parity is not None:
            if required_parity not in (0, 1):
                raise ValueError(
                    f"required_parity must be 0 (even) or 1 (odd), got {required_parity}"
                )

    def _check_result(self, cell_value: str) -> bool:
        """Check if a cell value satisfies the result condition."""
        if self.result_op == ComparisonOp.EQ:
            return cell_value == self.result_symbol
        elif self.result_op == ComparisonOp.NE:
            return cell_value != self.result_symbol
        else:
            raise ValueError(f"Local transition only supports == or != operators, got {self.result_op}")

    def _get_neighbor_config(self, state: State, index: int) -> str:
        """Return the 3-cell neighborhood window centered on index i.

        Uses wrapping (periodic boundary) for edges.
        """
        n = len(state)
        left = state[(index - 1) % n]
        center = state[index % n]
        right = state[(index + 1) % n]
        return f"{left}{center}{right}"

    def _matches_trigger(self, state: State, index: int) -> bool:
        """Check if position i matches the trigger conditions.

        Returns True if:
        1. state[index] == trigger_symbol, AND
        2. (neighbor_pattern is None OR neighbor_config(index) == neighbor_pattern), AND
        3. (required_parity is None OR index % 2 == required_parity)
        """
        if state[index] != self.trigger_symbol:
            return False

        if self.neighbor_pattern is not None:
            actual_pattern = self._get_neighbor_config(state, index)
            if actual_pattern != self.neighbor_pattern:
                return False

        if self.required_parity is not None:
            if index % 2 != self.required_parity:
                return False

        return True

    def check(self, trajectory: Trajectory) -> CheckResult:
        if len(trajectory) < 2:
            return CheckResult(passed=True)

        vacuity = VacuityReport()
        vacuity.total_checks = len(trajectory) - 1  # Step checks

        for t in range(len(trajectory) - 1):
            state_t = trajectory[t]
            state_t1 = trajectory[t + 1]

            # Check each position
            for i in range(len(state_t)):
                if self._matches_trigger(state_t, i):
                    vacuity.antecedent_true_count += 1
                    vacuity.consequent_evaluated_count += 1

                    # Check the same position in the next state
                    if i < len(state_t1):
                        next_cell = state_t1[i]
                    else:
                        # Grid shrunk? Treat as violation or skip
                        # For now, treat as violation since grid length should be invariant
                        return CheckResult(
                            passed=False,
                            violation=Violation(
                                t=t,
                                state=state_t,
                                details={
                                    "position": i,
                                    "trigger_symbol": self.trigger_symbol,
                                    "neighbor_pattern": self.neighbor_pattern,
                                    "error": "grid_shrunk",
                                },
                                message=f"Grid shrunk: position {i} doesn't exist at t={t+1}",
                            ),
                            vacuity=vacuity,
                        )

                    if not self._check_result(next_cell):
                        # Build context description for message
                        context_parts = []
                        if self.neighbor_pattern:
                            context_parts.append(f"neighborhood {self.neighbor_pattern}")
                        if self.required_parity is not None:
                            parity_name = "even" if self.required_parity == 0 else "odd"
                            context_parts.append(f"{parity_name} index")
                        context_str = f" with {' and '.join(context_parts)}" if context_parts else ""

                        return CheckResult(
                            passed=False,
                            violation=Violation(
                                t=t,
                                state=state_t,
                                details={
                                    "position": i,
                                    "index_parity": i % 2,
                                    "trigger_symbol": self.trigger_symbol,
                                    "neighbor_pattern": self.neighbor_pattern,
                                    "required_parity": self.required_parity,
                                    "actual_neighbor": self._get_neighbor_config(state_t, i),
                                    "expected_op": self.result_op.value,
                                    "expected_symbol": self.result_symbol,
                                    "actual_symbol": next_cell,
                                    "state_t": state_t,
                                    "state_t1": state_t1,
                                },
                                message=f"Local transition violated at t={t}, position {i}: "
                                f"cell was '{self.trigger_symbol}'{context_str}, "
                                f"expected {self.result_op.value} '{self.result_symbol}' at t+1 but got '{next_cell}'",
                            ),
                            vacuity=vacuity,
                        )

        vacuity.is_vacuous = vacuity.antecedent_true_count == 0
        return CheckResult(passed=True, vacuity=vacuity)


# Type alias for the unified checker interface
TrajectoryChecker = TemplateChecker
