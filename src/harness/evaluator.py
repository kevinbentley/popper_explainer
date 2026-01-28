"""Law evaluator - runs claims against trajectories."""

from typing import Callable

from src.claims.ast_evaluator import ASTClaimEvaluator, ASTEvaluationError, create_ast_checker
from src.claims.compiler import ClaimCompiler, CompilationError, compile_precondition
from src.claims.schema import CandidateLaw, Template
from src.claims.templates import CheckResult, SymmetryCommutationChecker, TemplateChecker, Violation
from src.harness.case import Case, CaseResult
from src.claims.vacuity import VacuityReport
from src.universe.simulator import run
from src.universe.types import State, Trajectory


class Evaluator:
    """Evaluates laws against test cases.

    Handles:
    - Compiling laws into checkers
    - Running simulations
    - Checking preconditions
    - Tracking vacuity
    """

    def __init__(self, probe_registry=None, scrambler=None):
        self._compiler = ClaimCompiler(probe_registry=probe_registry, scrambler=scrambler)
        self._probe_registry = probe_registry
        self._scrambler = scrambler
        self._checker: TemplateChecker | None = None
        self._ast_checker: ASTClaimEvaluator | None = None
        self._precondition_checkers: list[Callable[[State], bool]] = []
        self._current_law: CandidateLaw | None = None

    def prepare(self, law: CandidateLaw) -> None:
        """Prepare to evaluate a law.

        Uses AST evaluator if claim_ast is provided, otherwise falls back
        to string-based claim compiler.

        Args:
            law: The law to evaluate

        Raises:
            CompilationError: If law compilation fails
            ASTEvaluationError: If AST validation fails
        """
        self._current_law = law
        self._ast_checker = None
        self._checker = None

        # Try AST-based evaluation first (preferred)
        if law.claim_ast is not None:
            # Build probe_observables dict for AST path
            probe_observables: dict[str, Callable] | None = None
            if self._probe_registry is not None:
                probe_observables = {}
                for obs in law.observables:
                    if obs.probe_id:
                        fn = self._compiler._make_probe_evaluator(obs.probe_id)
                        if fn is not None:
                            probe_observables[obs.name] = fn
                if not probe_observables:
                    probe_observables = None
            self._ast_checker = create_ast_checker(law, probe_observables=probe_observables)
        else:
            # Fall back to string-based compilation
            self._checker = self._compiler.compile(law)

        # Compile preconditions
        self._precondition_checkers = []
        for precond in law.preconditions:
            checker = compile_precondition(precond, law)
            self._precondition_checkers.append(checker)

    def check_preconditions(self, state: State) -> bool:
        """Check if all preconditions are satisfied for a state.

        Args:
            state: Initial state to check

        Returns:
            True if all preconditions are met
        """
        for checker in self._precondition_checkers:
            if not checker(state):
                return False
        return True

    def evaluate_case(self, case: Case, time_horizon: int) -> CaseResult:
        """Evaluate a single test case.

        Args:
            case: The test case to evaluate
            time_horizon: Number of time steps to simulate

        Returns:
            CaseResult with pass/fail and violation details
        """
        if self._checker is None and self._ast_checker is None:
            raise RuntimeError("Evaluator not prepared. Call prepare() first.")

        # Check preconditions
        precondition_met = self.check_preconditions(case.initial_state)

        # Run simulation
        trajectory = run(case.initial_state, time_horizon, case.config)

        if not precondition_met:
            # Preconditions not met - case doesn't count as evidence
            return CaseResult(
                case=case,
                trajectory=trajectory,
                passed=True,  # Not a failure, just not applicable
                precondition_met=False,
            )

        # Check the law against the trajectory
        violation_dict = None
        passed = True
        antecedent_hits = 0
        total_checks = len(trajectory)

        if self._ast_checker is not None:
            # Use AST-based evaluation (now returns vacuity)
            passed, t_fail, details, vacuity = self._ast_checker.check(trajectory)
            if not passed:
                violation_dict = {
                    "t": t_fail,
                    "state": trajectory[t_fail] if t_fail is not None and t_fail < len(trajectory) else None,
                    "message": f"Claim failed at t={t_fail}",
                    "details": details,
                }
            # Extract vacuity from AST checker
            antecedent_hits = vacuity.antecedent_true_count
            total_checks = vacuity.total_checks or len(trajectory)
        else:
            # Use string-based evaluation
            # For symmetry_commutation with shift_k, pass k from case metadata
            if (isinstance(self._checker, SymmetryCommutationChecker) and
                self._checker.transform_name == "shift_k"):
                k = case.metadata.get("k") if case.metadata else None
                result = self._checker.check(trajectory, k=k)
            else:
                result = self._checker.check(trajectory)
            passed = result.passed
            if not passed and result.violation:
                violation_dict = {
                    "t": result.violation.t,
                    "state": result.violation.state,
                    "message": result.violation.message,
                    "details": result.violation.details,
                }
            # Extract vacuity from CheckResult
            antecedent_hits = result.vacuity.antecedent_true_count
            total_checks = result.vacuity.total_checks or len(trajectory)

        return CaseResult(
            case=case,
            trajectory=trajectory,
            passed=passed,
            violation=violation_dict,
            precondition_met=True,
            antecedent_hits=antecedent_hits,
            total_checks=total_checks,
        )

    def get_vacuity_report(self, results: list[CaseResult]) -> VacuityReport:
        """Aggregate vacuity information from multiple case results.

        This is relevant for implication and eventually templates.
        For implications, we track:
        - How many times the antecedent (LHS) was true (trigger_count)
        - How many distinct generators produced triggers (trigger_diversity)
        - Which initial states triggered the antecedent

        A test is vacuous if the antecedent is NEVER true.
        """
        if self._current_law is None:
            return VacuityReport()

        # Only track vacuity for implication/eventually templates
        if self._current_law.template not in (
            Template.IMPLICATION_STEP,
            Template.IMPLICATION_STATE,
            Template.EVENTUALLY,
            Template.LOCAL_TRANSITION,  # Also tracks trigger hits
        ):
            return VacuityReport(is_vacuous=False)

        # Aggregate antecedent hits from all cases
        total_antecedent_hits = 0
        total_checks = 0
        triggering_generators: set[str] = set()
        triggering_states: set[str] = set()

        for r in results:
            if r.precondition_met:
                total_antecedent_hits += r.antecedent_hits
                total_checks += r.total_checks

                # Track which generators and states produced triggers
                if r.antecedent_hits > 0:
                    triggering_generators.add(r.case.generator_family)
                    # Use initial state hash for diversity tracking
                    state_hash = r.case.content_hash()
                    triggering_states.add(state_hash)

        is_vacuous = total_antecedent_hits == 0
        trigger_diversity = len(triggering_generators)

        return VacuityReport(
            antecedent_true_count=total_antecedent_hits,
            consequent_evaluated_count=total_antecedent_hits,  # For implications, same as antecedent
            total_checks=total_checks,
            is_vacuous=is_vacuous,
            trigger_diversity=trigger_diversity,
            triggering_generators=triggering_generators,
            triggering_states=triggering_states,
        )


def has_collision(trajectory: Trajectory) -> bool:
    """Check if any state in the trajectory has a collision."""
    return any("X" in state for state in trajectory)


def has_wrapping(initial_state: State, trajectory: Trajectory) -> bool:
    """Check if boundary wrapping occurred.

    Simple heuristic: check if any particle position changed by more
    than would be possible without wrapping.
    """
    if len(trajectory) < 2:
        return False

    n = len(initial_state)

    # Find particle positions in initial and final states
    def particle_positions(state: str) -> list[int]:
        return [i for i, c in enumerate(state) if c in "><X"]

    initial_pos = particle_positions(trajectory[0])
    final_pos = particle_positions(trajectory[-1])

    if not initial_pos or not final_pos:
        return False

    # Check if any position changed dramatically (likely wrapping)
    # This is a heuristic - could be improved
    t = len(trajectory) - 1
    for ip in initial_pos:
        for fp in final_pos:
            # If a particle moved more than t steps, it might have wrapped
            # But this is tricky with collisions... use simple heuristic
            pass

    # Simpler: just check if particles are near opposite edges
    has_left_edge = any(p < 2 for p in final_pos)
    has_right_edge = any(p >= n - 2 for p in final_pos)
    return has_left_edge and has_right_edge


def compute_density(state: State) -> float:
    """Compute particle density of a state."""
    if not state:
        return 0.0
    particles = state.count(">") + state.count("<") + 2 * state.count("X")
    return particles / len(state)
