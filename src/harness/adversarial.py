"""Adversarial search orchestration for enhanced falsification.

The adversarial searcher runs after initial testing to more aggressively
search for counterexamples using mutation-based strategies.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.claims.schema import CandidateLaw
from src.harness.case import Case, CaseResult
from src.harness.generators.adversarial import (
    AdversarialMutationGenerator,
    GuidedAdversarialGenerator,
)
from src.harness.verdict import Counterexample


@dataclass
class AdversarialSearchResult:
    """Result of an adversarial search.

    Attributes:
        found_counterexample: Whether a counterexample was found
        counterexample: The counterexample if found
        cases_tried: Number of adversarial cases tried
        runtime_ms: Time spent on adversarial search
        effective_mutations: Mutations that contributed to findings
        search_phases: Details about search phases run
    """

    found_counterexample: bool = False
    counterexample: Counterexample | None = None
    cases_tried: int = 0
    runtime_ms: int = 0
    effective_mutations: list[str] = field(default_factory=list)
    search_phases: list[dict[str, Any]] = field(default_factory=list)


class AdversarialSearcher:
    """Orchestrates adversarial search for counterexamples.

    The searcher uses multiple phases:
    1. Mutation phase: Mutate promising states from initial testing
    2. Focused phase: Target specific weaknesses (collisions, boundaries)
    3. Intensive phase: Deep mutations on best candidates

    Usage:
        searcher = AdversarialSearcher(budget=1000)
        result = searcher.search(
            law=law,
            evaluate_case=evaluator.evaluate_case,
            seed_results=initial_results,
            time_horizon=50,
        )
    """

    def __init__(
        self,
        budget: int = 1500,
        max_runtime_ms: int = 5000,
    ):
        """Initialize the adversarial searcher.

        Args:
            budget: Maximum number of adversarial cases to try
            max_runtime_ms: Maximum time to spend on search
        """
        self.budget = budget
        self.max_runtime_ms = max_runtime_ms
        self._mutation_gen = AdversarialMutationGenerator()
        self._guided_gen = GuidedAdversarialGenerator()

    def search(
        self,
        law: CandidateLaw,
        evaluate_case: Callable[[Case, int], CaseResult],
        seed_results: list[CaseResult],
        time_horizon: int,
        seed: int = 42,
    ) -> AdversarialSearchResult:
        """Run adversarial search for counterexamples.

        Args:
            law: The law being tested
            evaluate_case: Function to evaluate a case
            seed_results: Results from initial testing (for seeding)
            time_horizon: Time horizon for simulations
            seed: Random seed

        Returns:
            AdversarialSearchResult with findings
        """
        start_time = time.time()
        result = AdversarialSearchResult()
        cases_remaining = self.budget

        # Extract promising seed states from initial results
        seed_states = self._extract_seed_states(seed_results)

        # Phase 1: Basic mutation search
        phase1_result = self._run_phase(
            name="mutation",
            cases_budget=min(cases_remaining // 3, self.budget // 3),
            params={
                "seed_states": seed_states[:20],
                "mutations_per_seed": 5,
                "max_mutations_per_state": 2,
                "grid_lengths": [8, 16, 32],
            },
            evaluate_case=evaluate_case,
            time_horizon=time_horizon,
            seed=seed,
            start_time=start_time,
        )
        result.search_phases.append(phase1_result)
        result.cases_tried += phase1_result["cases_tried"]

        if phase1_result.get("counterexample"):
            result.found_counterexample = True
            result.counterexample = phase1_result["counterexample"]
            result.effective_mutations = phase1_result.get("effective_mutations", [])
            result.runtime_ms = int((time.time() - start_time) * 1000)
            return result

        cases_remaining -= phase1_result["cases_tried"]
        if self._should_stop(start_time, cases_remaining):
            result.runtime_ms = int((time.time() - start_time) * 1000)
            return result

        # Phase 2: Collision-focused search
        phase2_result = self._run_phase(
            name="collision_focused",
            cases_budget=min(cases_remaining // 2, self.budget // 4),
            params={
                "seed_states": seed_states[:10],
                "mutations_per_seed": 8,
                "max_mutations_per_state": 3,
                "focus_collisions": True,
                "grid_lengths": [8, 12, 16],
            },
            evaluate_case=evaluate_case,
            time_horizon=time_horizon,
            seed=seed + 1000,
            start_time=start_time,
        )
        result.search_phases.append(phase2_result)
        result.cases_tried += phase2_result["cases_tried"]

        if phase2_result.get("counterexample"):
            result.found_counterexample = True
            result.counterexample = phase2_result["counterexample"]
            result.effective_mutations = phase2_result.get("effective_mutations", [])
            result.runtime_ms = int((time.time() - start_time) * 1000)
            return result

        cases_remaining -= phase2_result["cases_tried"]
        if self._should_stop(start_time, cases_remaining):
            result.runtime_ms = int((time.time() - start_time) * 1000)
            return result

        # Phase 3: Intensive deep mutations
        phase3_result = self._run_phase(
            name="intensive",
            cases_budget=cases_remaining,
            params={
                "seed_states": seed_states,
                "mutations_per_seed": 10,
                "max_mutations_per_state": 4,
                "grid_lengths": [6, 8, 10, 12, 16],
            },
            evaluate_case=evaluate_case,
            time_horizon=time_horizon,
            seed=seed + 2000,
            start_time=start_time,
        )
        result.search_phases.append(phase3_result)
        result.cases_tried += phase3_result["cases_tried"]

        if phase3_result.get("counterexample"):
            result.found_counterexample = True
            result.counterexample = phase3_result["counterexample"]
            result.effective_mutations = phase3_result.get("effective_mutations", [])

        result.runtime_ms = int((time.time() - start_time) * 1000)
        return result

    def _run_phase(
        self,
        name: str,
        cases_budget: int,
        params: dict[str, Any],
        evaluate_case: Callable[[Case, int], CaseResult],
        time_horizon: int,
        seed: int,
        start_time: float,
    ) -> dict[str, Any]:
        """Run a single search phase.

        Returns:
            Dict with phase results including cases_tried, counterexample if found
        """
        phase_start = time.time()
        cases_tried = 0
        effective_mutations: list[str] = []

        # Generate cases
        cases = self._mutation_gen.generate(params, seed, cases_budget)

        for case in cases:
            if self._should_stop(start_time, self.budget - cases_tried):
                break

            result = evaluate_case(case, time_horizon)
            cases_tried += 1

            if not result.passed and result.precondition_met:
                # Found counterexample!
                mutations = case.generator_params.get("mutations", [])
                effective_mutations.extend(mutations)

                # Record for guided search
                for mut in mutations:
                    self._guided_gen.record_effective_mutation(mut)

                counterexample = self._create_counterexample(result, time_horizon)
                return {
                    "name": name,
                    "cases_tried": cases_tried,
                    "runtime_ms": int((time.time() - phase_start) * 1000),
                    "counterexample": counterexample,
                    "effective_mutations": effective_mutations,
                }
            elif result.precondition_met and result.near_miss_score > 0.5:
                # Record near-miss for guided search
                self._guided_gen.record_near_miss(
                    case.initial_state, result.near_miss_score
                )

        return {
            "name": name,
            "cases_tried": cases_tried,
            "runtime_ms": int((time.time() - phase_start) * 1000),
            "counterexample": None,
            "effective_mutations": [],
        }

    def _extract_seed_states(self, results: list[CaseResult]) -> list[str]:
        """Extract promising seed states from initial results.

        Prioritizes:
        1. States that were close to violations
        2. States with collisions
        3. States with higher density
        """
        # Sort by near_miss_score (higher = more promising)
        sorted_results = sorted(
            [r for r in results if r.precondition_met],
            key=lambda r: r.near_miss_score,
            reverse=True,
        )

        # Take initial states, prioritizing near-misses
        seed_states = [r.case.initial_state for r in sorted_results]

        # Add some states that had collisions
        collision_states = [
            r.case.initial_state
            for r in results
            if r.precondition_met and self._has_collision(r.trajectory)
        ]
        for state in collision_states:
            if state not in seed_states:
                seed_states.append(state)

        return seed_states[:50]  # Limit to top 50

    def _has_collision(self, trajectory: list[str]) -> bool:
        """Check if trajectory contains a collision."""
        for state in trajectory:
            if "X" in state:
                return True
        return False

    def _create_counterexample(
        self, result: CaseResult, time_horizon: int
    ) -> Counterexample:
        """Create a counterexample from a failed case result."""
        t_fail = result.violation.get("t", 0) if result.violation else 0

        # Extract trajectory excerpt around failure
        excerpt_start = max(0, t_fail - 2)
        excerpt_end = min(len(result.trajectory), t_fail + 3)
        trajectory_excerpt = result.trajectory[excerpt_start:excerpt_end]

        return Counterexample(
            initial_state=result.case.initial_state,
            config={
                "grid_length": result.case.config.grid_length,
                "boundary": result.case.config.boundary,
            },
            seed=result.case.seed,
            t_max=time_horizon,
            t_fail=t_fail,
            trajectory_excerpt=trajectory_excerpt,
            observables_at_fail=result.violation.get("details") if result.violation else None,
            witness={
                **(result.violation or {}),
                "adversarial": True,
                "mutations": result.case.generator_params.get("mutations", []),
            },
            minimized=False,
        )

    def _should_stop(self, start_time: float, cases_remaining: int) -> bool:
        """Check if search should stop."""
        elapsed_ms = (time.time() - start_time) * 1000
        return elapsed_ms >= self.max_runtime_ms or cases_remaining <= 0
