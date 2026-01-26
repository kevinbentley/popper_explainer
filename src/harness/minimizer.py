"""Counterexample minimization.

When a law fails, try to find a smaller/simpler counterexample
for better auditability.
"""

from src.claims.schema import CandidateLaw
from src.harness.case import Case
from src.harness.evaluator import Evaluator
from src.harness.verdict import Counterexample
from src.universe.simulator import run
from src.universe.types import Config, Symbol


class Minimizer:
    """Minimizes counterexamples for failed laws.

    Strategies:
    1. Reduce grid length while preserving failure
    2. Remove unnecessary particles
    3. Reduce time horizon to earliest failure
    """

    def __init__(self, budget: int = 100):
        """Initialize minimizer.

        Args:
            budget: Maximum minimization attempts
        """
        self.budget = budget
        self._evaluator = Evaluator()

    def minimize(
        self,
        law: CandidateLaw,
        counterexample: Counterexample,
        time_horizon: int,
    ) -> Counterexample:
        """Attempt to minimize a counterexample.

        Args:
            law: The law that was violated
            counterexample: The original counterexample
            time_horizon: Original time horizon

        Returns:
            Minimized counterexample (may be same as original)
        """
        self._evaluator.prepare(law)

        current_state = counterexample.initial_state
        current_config = Config(
            grid_length=counterexample.config.get("grid_length", len(current_state))
        )
        current_t_fail = counterexample.t_fail
        attempts = 0

        # Strategy 1: Reduce time horizon to earliest failure
        min_t = self._find_earliest_failure(
            current_state, current_config, current_t_fail, law
        )
        if min_t < current_t_fail:
            current_t_fail = min_t
        attempts += 1

        # Strategy 2: Try to remove particles
        while attempts < self.budget:
            reduced = self._try_remove_particle(
                current_state, current_config, current_t_fail, law
            )
            if reduced is None:
                break
            current_state = reduced
            attempts += 1

        # Strategy 3: Try to shrink grid (harder, skip for now)
        # This would require more sophisticated state transformation

        # Regenerate trajectory excerpt from minimized state
        # Run enough steps to capture states around the failure (t_fail-2 to t_fail+2)
        min_steps = current_t_fail + 3  # Need at least t_fail+2 states
        trajectory = run(current_state, min_steps, current_config)
        excerpt_start = max(0, current_t_fail - 2)
        excerpt_end = min(len(trajectory), current_t_fail + 3)
        trajectory_excerpt = trajectory[excerpt_start:excerpt_end]

        return Counterexample(
            initial_state=current_state,
            config={"grid_length": current_config.grid_length, "boundary": "periodic"},
            seed=counterexample.seed,
            t_max=current_t_fail,
            t_fail=current_t_fail,
            trajectory_excerpt=trajectory_excerpt,
            observables_at_fail=counterexample.observables_at_fail,
            witness=counterexample.witness,
            minimized=True,
        )

    def _find_earliest_failure(
        self,
        state: str,
        config: Config,
        max_t: int,
        law: CandidateLaw,
    ) -> int:
        """Find the earliest time step where the law fails."""
        # Binary search for earliest failure
        low, high = 1, max_t

        while low < high:
            mid = (low + high) // 2
            case = Case(
                initial_state=state,
                config=config,
                seed=0,
                generator_family="minimizer",
                params_hash="min",
            )
            result = self._evaluator.evaluate_case(case, mid)

            if result.passed:
                low = mid + 1
            else:
                high = mid

        return low

    def _try_remove_particle(
        self,
        state: str,
        config: Config,
        t_fail: int,
        law: CandidateLaw,
    ) -> str | None:
        """Try removing each particle to see if failure persists.

        Returns the reduced state if successful, None otherwise.
        """
        cells = list(state)

        for i, cell in enumerate(cells):
            if cell in (Symbol.RIGHT.value, Symbol.LEFT.value):
                # Try removing this particle
                original = cells[i]
                cells[i] = Symbol.EMPTY.value
                test_state = "".join(cells)

                case = Case(
                    initial_state=test_state,
                    config=config,
                    seed=0,
                    generator_family="minimizer",
                    params_hash="min",
                )
                result = self._evaluator.evaluate_case(case, t_fail)

                if not result.passed and result.precondition_met:
                    # Failure still occurs with particle removed
                    return test_state

                # Restore particle
                cells[i] = original

            elif cell == Symbol.COLLISION.value:
                # For collisions, try replacing with single particle
                for replacement in [Symbol.RIGHT.value, Symbol.LEFT.value, Symbol.EMPTY.value]:
                    cells[i] = replacement
                    test_state = "".join(cells)

                    case = Case(
                        initial_state=test_state,
                        config=config,
                        seed=0,
                        generator_family="minimizer",
                        params_hash="min",
                    )
                    result = self._evaluator.evaluate_case(case, t_fail)

                    if not result.passed and result.precondition_met:
                        return test_state

                # Restore collision
                cells[i] = Symbol.COLLISION.value

        return None
