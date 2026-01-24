"""Adversarial test generation for prediction verification.

Generates test cases designed to maximize prediction errors,
providing a non-gameable evaluation metric.

Key principle: The adversarial generator searches for states where
the explainer's predictions disagree with the simulator, targeting
edge cases and boundary conditions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from src.orchestration.prediction.test_sets import PredictionTestCase, PredictionTestSet
from src.universe.types import Config, State, Symbol

if TYPE_CHECKING:
    from src.db.repo import Repository


@dataclass
class AdversarialSearchConfig:
    """Configuration for adversarial search.

    Attributes:
        max_mutations: Maximum mutations per state
        mutation_rounds: Number of mutation rounds per seed
        population_size: Size of mutation population
        elite_fraction: Fraction of top performers to keep
        target_disagreement: Stop if this disagreement rate is achieved
    """

    max_mutations: int = 3
    mutation_rounds: int = 10
    population_size: int = 50
    elite_fraction: float = 0.2
    target_disagreement: float = 0.5


class AdversarialPredictionGenerator:
    """Generates adversarial test cases for prediction verification.

    Uses mutation-based search to find states that maximize
    disagreement between predictions and ground truth.

    Search strategies:
    1. Edge case injection: Focus on collision setups and boundaries
    2. Mutation search: Evolve states toward prediction failures
    3. Targeted probing: Focus on observed weak areas
    """

    def __init__(
        self,
        predictor_fn: Callable[[State, int], State] | None = None,
        config: AdversarialSearchConfig | None = None,
    ):
        """Initialize adversarial generator.

        Args:
            predictor_fn: Function that generates predictions (state, horizon) -> predicted_state
            config: Search configuration
        """
        self.predictor_fn = predictor_fn
        self.config = config or AdversarialSearchConfig()
        self._weak_patterns: list[str] = []  # Patterns that caused errors

    def set_predictor(self, predictor_fn: Callable[[State, int], State]) -> None:
        """Set the predictor function to test against.

        Args:
            predictor_fn: Function (state, horizon) -> predicted_state
        """
        self.predictor_fn = predictor_fn

    def record_weak_pattern(self, pattern: str) -> None:
        """Record a pattern that caused prediction errors.

        Args:
            pattern: State pattern or substring that caused errors
        """
        self._weak_patterns.append(pattern)
        # Keep only recent patterns
        if len(self._weak_patterns) > 100:
            self._weak_patterns = self._weak_patterns[-100:]

    def generate_adversarial_set(
        self,
        run_id: str,
        seed: int,
        count: int = 50,
        horizons: list[int] | None = None,
        grid_lengths: list[int] | None = None,
    ) -> PredictionTestSet:
        """Generate an adversarial test set.

        If a predictor is set, uses active search to find disagreements.
        Otherwise, generates edge cases and boundary conditions.

        Args:
            run_id: Orchestration run ID
            seed: Random seed
            count: Number of cases to generate
            horizons: Prediction horizons
            grid_lengths: Grid sizes to use

        Returns:
            PredictionTestSet with adversarial cases
        """
        from src.universe.simulator import run as sim_run

        rng = random.Random(seed)
        horizons = horizons or [1, 2, 3]
        grid_lengths = grid_lengths or [8, 16, 24]

        cases: list[PredictionTestCase] = []

        # Calculate allocation for each strategy
        edge_count = count // 3
        dense_count = count // 3
        remaining_count = count - edge_count - dense_count

        # Strategy 1: Edge cases (collision setups, boundaries)
        edge_cases = self._generate_edge_cases(rng, grid_lengths, edge_count)
        cases.extend(edge_cases)

        # Strategy 2: High-density states (complex interactions)
        dense_cases = self._generate_dense_cases(rng, grid_lengths, dense_count)
        cases.extend(dense_cases)

        # Strategy 3: Mutation search (if predictor available) or random
        if self.predictor_fn:
            adversarial_cases = self._mutation_search(
                rng, grid_lengths, horizons, remaining_count
            )
            cases.extend(adversarial_cases)
        else:
            # Fallback: random cases with bias toward complex states
            random_cases = self._generate_biased_random(rng, grid_lengths, remaining_count)
            cases.extend(random_cases)

        # Fill any remaining slots with biased random
        while len(cases) < count:
            extra = self._generate_biased_random(rng, grid_lengths, count - len(cases))
            cases.extend(extra)
            if len(extra) == 0:
                break  # Prevent infinite loop

        # Add horizons and compute ground truth
        finalized_cases: list[PredictionTestCase] = []
        for i, case in enumerate(cases[:count]):
            horizon = horizons[i % len(horizons)]
            config = case.config

            # Compute ground truth
            trajectory = sim_run(case.initial_state, horizon, config)
            expected_state = trajectory[-1]

            finalized_case = PredictionTestCase(
                initial_state=case.initial_state,
                horizon=horizon,
                config=config,
                expected_state=expected_state,
                metadata={
                    **case.metadata,
                    "adversarial_strategy": case.metadata.get("strategy", "unknown"),
                },
            )
            finalized_cases.append(finalized_case)

        set_id = f"adversarial_{run_id}_{seed}"
        test_set = PredictionTestSet(
            set_id=set_id,
            set_type="adversarial",
            cases=finalized_cases,
            generation_seed=seed,
            locked=True,
        )

        return test_set

    def _generate_edge_cases(
        self,
        rng: random.Random,
        grid_lengths: list[int],
        count: int,
    ) -> list[PredictionTestCase]:
        """Generate edge cases focusing on collision and boundary behavior."""
        cases = []

        edge_patterns = [
            # Imminent collisions
            lambda n: "><" + "." * (n - 2),
            lambda n: "." * (n - 2) + "<>",
            lambda n: ">." * (n // 2),  # Evenly spaced rights
            lambda n: ".<" * (n // 2),  # Evenly spaced lefts
            # Boundary wrapping
            lambda n: ">" + "." * (n - 2) + "<",  # Will collide at boundary
            lambda n: "<" + "." * (n - 2) + ">",  # Wrap in opposite direction
            # Multiple collisions
            lambda n: "><><" + "." * max(0, n - 4),
            lambda n: "X" * min(4, n // 2) + "." * max(0, n - (n // 2)),
            # Dense collision chains
            lambda n: "><><><><"[:n],
            lambda n: "X.X.X.X."[:n],
        ]

        for i in range(count):
            n = grid_lengths[i % len(grid_lengths)]
            pattern_fn = edge_patterns[i % len(edge_patterns)]

            try:
                state = pattern_fn(n)
                # Pad or truncate to exact length
                if len(state) < n:
                    state = state + "." * (n - len(state))
                elif len(state) > n:
                    state = state[:n]

                case = PredictionTestCase(
                    initial_state=state,
                    horizon=1,  # Will be set later
                    config=Config(grid_length=n),
                    metadata={"strategy": "edge_case", "pattern_index": i % len(edge_patterns)},
                )
                cases.append(case)
            except Exception:
                continue

        return cases

    def _generate_dense_cases(
        self,
        rng: random.Random,
        grid_lengths: list[int],
        count: int,
    ) -> list[PredictionTestCase]:
        """Generate high-density states with complex interactions."""
        cases = []

        for i in range(count):
            n = grid_lengths[i % len(grid_lengths)]
            # High density: 60-90% occupied
            density = rng.uniform(0.6, 0.9)
            num_particles = int(n * density)

            cells = [Symbol.EMPTY.value] * n
            positions = list(range(n))
            rng.shuffle(positions)

            for pos in positions[:num_particles]:
                # Include some collisions
                if rng.random() < 0.15:
                    cells[pos] = Symbol.COLLISION.value
                elif rng.random() < 0.5:
                    cells[pos] = Symbol.RIGHT.value
                else:
                    cells[pos] = Symbol.LEFT.value

            state = "".join(cells)
            case = PredictionTestCase(
                initial_state=state,
                horizon=1,
                config=Config(grid_length=n),
                metadata={"strategy": "dense", "density": density},
            )
            cases.append(case)

        return cases

    def _generate_biased_random(
        self,
        rng: random.Random,
        grid_lengths: list[int],
        count: int,
    ) -> list[PredictionTestCase]:
        """Generate random cases biased toward challenging states."""
        cases = []

        for i in range(count):
            n = grid_lengths[i % len(grid_lengths)]

            # Bias toward medium-high density
            density = rng.triangular(0.2, 0.9, 0.5)
            num_particles = int(n * density)

            cells = [Symbol.EMPTY.value] * n
            positions = list(range(n))
            rng.shuffle(positions)

            # Bias particle directions based on position
            for pos in positions[:num_particles]:
                # Particles on left half tend to go right, creating collisions
                if pos < n // 2:
                    cells[pos] = Symbol.RIGHT.value if rng.random() < 0.7 else Symbol.LEFT.value
                else:
                    cells[pos] = Symbol.LEFT.value if rng.random() < 0.7 else Symbol.RIGHT.value

            state = "".join(cells)
            case = PredictionTestCase(
                initial_state=state,
                horizon=1,
                config=Config(grid_length=n),
                metadata={"strategy": "biased_random", "density": density},
            )
            cases.append(case)

        return cases

    def _mutation_search(
        self,
        rng: random.Random,
        grid_lengths: list[int],
        horizons: list[int],
        count: int,
    ) -> list[PredictionTestCase]:
        """Use mutation search to find states that cause prediction errors.

        This method requires a predictor function to be set.
        """
        from src.universe.simulator import run as sim_run

        if not self.predictor_fn:
            return []

        cases: list[PredictionTestCase] = []
        found_disagreements: set[str] = set()

        # Generate initial population
        population: list[tuple[str, int, float]] = []  # (state, grid_length, score)

        for n in grid_lengths:
            for _ in range(self.config.population_size // len(grid_lengths)):
                state = self._random_state(rng, n, rng.uniform(0.3, 0.7))
                score = self._compute_disagreement_score(state, horizons[0])
                population.append((state, n, score))

        # Evolution loop
        for round_idx in range(self.config.mutation_rounds):
            if len(cases) >= count:
                break

            # Sort by disagreement score (higher = more adversarial)
            population.sort(key=lambda x: x[2], reverse=True)

            # Keep elites
            elite_count = int(len(population) * self.config.elite_fraction)
            elites = population[:elite_count]

            # Check if top performers are adversarial enough
            for state, n, score in elites:
                if score > 0.3 and state not in found_disagreements:
                    found_disagreements.add(state)
                    case = PredictionTestCase(
                        initial_state=state,
                        horizon=1,
                        config=Config(grid_length=n),
                        metadata={"strategy": "mutation_search", "disagreement_score": score},
                    )
                    cases.append(case)

                    if len(cases) >= count:
                        break

            # Generate new population through mutation
            new_population = list(elites)
            while len(new_population) < self.config.population_size:
                # Select parent from elites
                parent_state, parent_n, _ = rng.choice(elites)

                # Apply mutations
                mutated = self._mutate_state(rng, parent_state)

                # Score the mutant
                score = self._compute_disagreement_score(mutated, horizons[0])
                new_population.append((mutated, parent_n, score))

            population = new_population

        return cases[:count]

    def _compute_disagreement_score(self, state: State, horizon: int) -> float:
        """Compute how much the predictor disagrees with ground truth.

        Returns a score from 0 (perfect agreement) to 1 (complete disagreement).
        """
        from src.universe.simulator import run as sim_run

        if not self.predictor_fn:
            return 0.0

        try:
            # Get ground truth
            config = Config(grid_length=len(state))
            trajectory = sim_run(state, horizon, config)
            expected = trajectory[-1]

            # Get prediction
            predicted = self.predictor_fn(state, horizon)

            # Compute Hamming distance normalized by length
            if len(predicted) != len(expected):
                return 1.0  # Complete disagreement if lengths differ

            mismatches = sum(1 for a, b in zip(predicted, expected) if a != b)
            return mismatches / len(expected)

        except Exception:
            return 0.0  # Treat errors as no disagreement

    def _random_state(self, rng: random.Random, n: int, density: float) -> State:
        """Generate a random state."""
        num_particles = int(n * density)
        cells = [Symbol.EMPTY.value] * n
        positions = list(range(n))
        rng.shuffle(positions)

        for pos in positions[:num_particles]:
            cells[pos] = rng.choice([Symbol.RIGHT.value, Symbol.LEFT.value])

        return "".join(cells)

    def _mutate_state(self, rng: random.Random, state: State) -> State:
        """Apply random mutations to a state."""
        cells = list(state)
        n = len(cells)

        # Apply 1-3 mutations
        num_mutations = rng.randint(1, min(3, n))

        for _ in range(num_mutations):
            mutation_type = rng.choice(["flip", "add", "remove", "swap"])
            pos = rng.randint(0, n - 1)

            if mutation_type == "flip" and cells[pos] in (Symbol.RIGHT.value, Symbol.LEFT.value):
                cells[pos] = Symbol.LEFT.value if cells[pos] == Symbol.RIGHT.value else Symbol.RIGHT.value
            elif mutation_type == "add" and cells[pos] == Symbol.EMPTY.value:
                cells[pos] = rng.choice([Symbol.RIGHT.value, Symbol.LEFT.value])
            elif mutation_type == "remove" and cells[pos] != Symbol.EMPTY.value:
                cells[pos] = Symbol.EMPTY.value
            elif mutation_type == "swap" and n > 1:
                pos2 = rng.randint(0, n - 1)
                cells[pos], cells[pos2] = cells[pos2], cells[pos]

        return "".join(cells)
