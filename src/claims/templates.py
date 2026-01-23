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

    def __init__(self, observable_expr: Expr):
        self.observable_expr = observable_expr

    def check(self, trajectory: Trajectory) -> CheckResult:
        if not trajectory:
            return CheckResult(passed=True)

        initial_value = evaluate_expression(self.observable_expr, trajectory[0])

        for t, state in enumerate(trajectory):
            value = evaluate_expression(self.observable_expr, state)
            if value != initial_value:
                return CheckResult(
                    passed=False,
                    violation=Violation(
                        t=t,
                        state=state,
                        details={"expected": initial_value, "actual": value},
                        message=f"Invariant violated at t={t}: expected {initial_value}, got {value}",
                    ),
                )

        return CheckResult(passed=True)


class MonotoneChecker(TemplateChecker):
    """Checker for monotone template: f(t+1) <= f(t) or f(t+1) >= f(t)."""

    def __init__(self, observable_expr: Expr, direction: MonotoneDirection):
        self.observable_expr = observable_expr
        self.direction = direction

    def check(self, trajectory: Trajectory) -> CheckResult:
        if len(trajectory) < 2:
            return CheckResult(passed=True)

        prev_value = evaluate_expression(self.observable_expr, trajectory[0])

        for t in range(1, len(trajectory)):
            curr_value = evaluate_expression(self.observable_expr, trajectory[t])

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

    def __init__(self, observable_expr: Expr, op: ComparisonOp, bound: int):
        self.observable_expr = observable_expr
        self.op = op
        self.bound = bound

    def _compare(self, value: int) -> bool:
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

    def check(self, trajectory: Trajectory) -> CheckResult:
        for t, state in enumerate(trajectory):
            value = evaluate_expression(self.observable_expr, state)
            if not self._compare(value):
                return CheckResult(
                    passed=False,
                    violation=Violation(
                        t=t,
                        state=state,
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
    """Checker for symmetry_commutation: evolve(T(S), t) == T(evolve(S, t))."""

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

        return CheckResult(passed=True)


class LocalTransitionChecker(TemplateChecker):
    """Checker for local_transition template: ∀t,i: state[i]==P at t → state[i] satisfies Q at t+1.

    This template expresses per-cell (micro-level) behavior, as opposed to
    aggregate (macro-level) behavior expressed by count-based templates.

    Example: "Each X cell resolves in one step"
    - trigger_symbol='X', result_op='!=', result_symbol='X'
    - For each position i where state_t[i]=='X', state_{t+1}[i] != 'X'
    """

    def __init__(
        self,
        trigger_symbol: str,
        result_op: ComparisonOp,
        result_symbol: str,
    ):
        self.trigger_symbol = trigger_symbol
        self.result_op = result_op
        self.result_symbol = result_symbol

    def _check_result(self, cell_value: str) -> bool:
        """Check if a cell value satisfies the result condition."""
        if self.result_op == ComparisonOp.EQ:
            return cell_value == self.result_symbol
        elif self.result_op == ComparisonOp.NE:
            return cell_value != self.result_symbol
        else:
            raise ValueError(f"Local transition only supports == or != operators, got {self.result_op}")

    def check(self, trajectory: Trajectory) -> CheckResult:
        if len(trajectory) < 2:
            return CheckResult(passed=True)

        vacuity = VacuityReport()
        vacuity.total_checks = len(trajectory) - 1  # Step checks

        for t in range(len(trajectory) - 1):
            state_t = trajectory[t]
            state_t1 = trajectory[t + 1]

            # Check each position
            for i, cell in enumerate(state_t):
                if cell == self.trigger_symbol:
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
                                    "error": "grid_shrunk",
                                },
                                message=f"Grid shrunk: position {i} doesn't exist at t={t+1}",
                            ),
                            vacuity=vacuity,
                        )

                    if not self._check_result(next_cell):
                        return CheckResult(
                            passed=False,
                            violation=Violation(
                                t=t,
                                state=state_t,
                                details={
                                    "position": i,
                                    "trigger_symbol": self.trigger_symbol,
                                    "expected_op": self.result_op.value,
                                    "expected_symbol": self.result_symbol,
                                    "actual_symbol": next_cell,
                                    "state_t": state_t,
                                    "state_t1": state_t1,
                                },
                                message=f"Local transition violated at t={t}, position {i}: "
                                f"cell was '{self.trigger_symbol}', expected {self.result_op.value} "
                                f"'{self.result_symbol}' at t+1 but got '{next_cell}'",
                            ),
                            vacuity=vacuity,
                        )

        vacuity.is_vacuous = vacuity.antecedent_true_count == 0
        return CheckResult(passed=True, vacuity=vacuity)


# Type alias for the unified checker interface
TrajectoryChecker = TemplateChecker
