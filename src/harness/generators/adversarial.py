"""Adversarial mutation generator for enhanced falsification.

This generator uses mutation strategies to search for counterexamples
by modifying promising seed states in targeted ways.
"""

import random
from typing import Any

from src.harness.case import Case
from src.harness.generators.base import Generator
from src.universe.types import Config, Symbol


class MutationStrategy:
    """Base class for mutation strategies."""

    def mutate(self, state: str, rng: random.Random) -> str:
        """Apply mutation to a state string."""
        raise NotImplementedError


class FlipDirectionMutation(MutationStrategy):
    """Flip the direction of a random particle."""

    def mutate(self, state: str, rng: random.Random) -> str:
        state_list = list(state)
        # Find positions with particles
        particle_positions = [
            i for i, c in enumerate(state_list)
            if c in (Symbol.RIGHT.value, Symbol.LEFT.value)
        ]
        if not particle_positions:
            return state

        pos = rng.choice(particle_positions)
        if state_list[pos] == Symbol.RIGHT.value:
            state_list[pos] = Symbol.LEFT.value
        else:
            state_list[pos] = Symbol.RIGHT.value
        return "".join(state_list)


class AddParticleMutation(MutationStrategy):
    """Add a particle to an empty cell."""

    def mutate(self, state: str, rng: random.Random) -> str:
        state_list = list(state)
        # Find empty positions
        empty_positions = [
            i for i, c in enumerate(state_list)
            if c == Symbol.EMPTY.value
        ]
        if not empty_positions:
            return state

        pos = rng.choice(empty_positions)
        state_list[pos] = rng.choice([Symbol.RIGHT.value, Symbol.LEFT.value])
        return "".join(state_list)


class RemoveParticleMutation(MutationStrategy):
    """Remove a particle (replace with empty)."""

    def mutate(self, state: str, rng: random.Random) -> str:
        state_list = list(state)
        # Find positions with particles (not collisions)
        particle_positions = [
            i for i, c in enumerate(state_list)
            if c in (Symbol.RIGHT.value, Symbol.LEFT.value)
        ]
        if not particle_positions:
            return state

        pos = rng.choice(particle_positions)
        state_list[pos] = Symbol.EMPTY.value
        return "".join(state_list)


class ShiftParticleMutation(MutationStrategy):
    """Shift a particle to an adjacent empty cell."""

    def mutate(self, state: str, rng: random.Random) -> str:
        state_list = list(state)
        n = len(state_list)

        # Find positions with particles
        particle_positions = [
            i for i, c in enumerate(state_list)
            if c in (Symbol.RIGHT.value, Symbol.LEFT.value)
        ]
        if not particle_positions:
            return state

        pos = rng.choice(particle_positions)
        particle = state_list[pos]

        # Try adjacent positions (with wrapping)
        directions = [(pos - 1) % n, (pos + 1) % n]
        rng.shuffle(directions)

        for new_pos in directions:
            if state_list[new_pos] == Symbol.EMPTY.value:
                state_list[pos] = Symbol.EMPTY.value
                state_list[new_pos] = particle
                return "".join(state_list)

        return state


class CreateCollisionSetupMutation(MutationStrategy):
    """Create a setup where particles will collide."""

    def mutate(self, state: str, rng: random.Random) -> str:
        state_list = list(state)
        n = len(state_list)

        # Find two adjacent empty cells
        for i in range(n):
            pos1 = i
            pos2 = (i + 1) % n
            pos3 = (i + 2) % n  # For proper collision setup

            if (state_list[pos1] == Symbol.EMPTY.value and
                state_list[pos2] == Symbol.EMPTY.value and
                state_list[pos3] == Symbol.EMPTY.value and n > 3):
                # Create approaching particles: >.<
                state_list[pos1] = Symbol.RIGHT.value
                state_list[pos2] = Symbol.EMPTY.value
                state_list[pos3] = Symbol.LEFT.value
                return "".join(state_list)

        return state


class SwapParticlesMutation(MutationStrategy):
    """Swap two particles."""

    def mutate(self, state: str, rng: random.Random) -> str:
        state_list = list(state)

        # Find all particle positions
        particle_positions = [
            i for i, c in enumerate(state_list)
            if c in (Symbol.RIGHT.value, Symbol.LEFT.value)
        ]
        if len(particle_positions) < 2:
            return state

        pos1, pos2 = rng.sample(particle_positions, 2)
        state_list[pos1], state_list[pos2] = state_list[pos2], state_list[pos1]
        return "".join(state_list)


class AdversarialMutationGenerator(Generator):
    """Generator that mutates seed states to find counterexamples.

    This generator takes seed states (either provided or generated)
    and applies various mutations to search for states that might
    violate a law.
    """

    # All available mutation strategies with weights
    MUTATIONS = [
        (FlipDirectionMutation(), 0.25),
        (AddParticleMutation(), 0.15),
        (RemoveParticleMutation(), 0.15),
        (ShiftParticleMutation(), 0.20),
        (CreateCollisionSetupMutation(), 0.15),
        (SwapParticlesMutation(), 0.10),
    ]

    def family_name(self) -> str:
        return "adversarial_mutation_search"

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate mutated test cases.

        Params:
            seed_states: Optional list of initial states to mutate
            mutations_per_seed: Number of mutations per seed state
            grid_lengths: Grid lengths to use for generated seeds
            max_mutations_per_state: Maximum mutations to chain
            focus_collisions: Whether to bias toward collision setups
        """
        rng = random.Random(seed)
        cases: list[Case] = []

        seed_states = params.get("seed_states", [])
        mutations_per_seed = params.get("mutations_per_seed", 3)
        grid_lengths = params.get("grid_lengths", [8, 16, 32])
        max_mutations = params.get("max_mutations_per_state", 2)
        focus_collisions = params.get("focus_collisions", False)

        # If no seed states provided, generate some
        if not seed_states:
            seed_states = self._generate_seed_states(rng, grid_lengths, count // mutations_per_seed + 1)

        # Select mutations based on focus
        mutations = self.MUTATIONS.copy()
        if focus_collisions:
            # Increase weight of collision-creating mutations
            mutations = [
                (mut, weight * 2 if isinstance(mut, CreateCollisionSetupMutation) else weight)
                for mut, weight in mutations
            ]

        # Normalize weights
        total_weight = sum(w for _, w in mutations)
        mutations = [(mut, w / total_weight) for mut, w in mutations]

        for seed_state in seed_states:
            if len(cases) >= count:
                break

            # Generate mutations from this seed
            for _ in range(mutations_per_seed):
                if len(cases) >= count:
                    break

                # Apply 1 to max_mutations mutations
                num_mutations = rng.randint(1, max_mutations)
                mutated_state = seed_state
                mutation_chain = []

                for _ in range(num_mutations):
                    mutation = self._select_mutation(mutations, rng)
                    new_state = mutation.mutate(mutated_state, rng)
                    if new_state != mutated_state:  # Mutation was effective
                        mutated_state = new_state
                        mutation_chain.append(type(mutation).__name__)

                if mutated_state != seed_state:  # Only add if actually mutated
                    config = Config(
                        grid_length=len(mutated_state),
                        boundary="periodic",
                    )
                    case = Case(
                        initial_state=mutated_state,
                        config=config,
                        seed=seed + len(cases),
                        generator_family=self.family_name(),
                        generator_params={
                            "seed_state": seed_state,
                            "mutations": mutation_chain,
                        },
                    )
                    cases.append(case)

        return cases

    def _generate_seed_states(
        self, rng: random.Random, grid_lengths: list[int], count: int
    ) -> list[str]:
        """Generate seed states for mutation."""
        states = []
        for _ in range(count):
            n = rng.choice(grid_lengths)
            # Generate a state with moderate density
            density = rng.uniform(0.2, 0.5)
            state_list = [Symbol.EMPTY.value] * n

            for i in range(n):
                if rng.random() < density:
                    state_list[i] = rng.choice([Symbol.RIGHT.value, Symbol.LEFT.value])

            states.append("".join(state_list))
        return states

    def _select_mutation(
        self, mutations: list[tuple[MutationStrategy, float]], rng: random.Random
    ) -> MutationStrategy:
        """Select a mutation based on weights."""
        r = rng.random()
        cumulative = 0
        for mutation, weight in mutations:
            cumulative += weight
            if r <= cumulative:
                return mutation
        return mutations[-1][0]


class GuidedAdversarialGenerator(Generator):
    """Generator that uses feedback from previous failures to guide search.

    This generator learns from near-misses and successful counterexamples
    to focus the search on promising regions of the state space.
    """

    def family_name(self) -> str:
        return "guided_adversarial_search"

    def __init__(self):
        self._mutation_gen = AdversarialMutationGenerator()
        self._promising_states: list[str] = []
        self._effective_mutations: dict[str, int] = {}

    def record_near_miss(self, state: str, score: float) -> None:
        """Record a state that almost violated the law.

        Args:
            state: The state that was a near-miss
            score: How close it was to violation (0-1, higher = closer)
        """
        if score > 0.5:
            self._promising_states.append(state)
            # Keep only most recent promising states
            if len(self._promising_states) > 100:
                self._promising_states = self._promising_states[-100:]

    def record_effective_mutation(self, mutation_name: str) -> None:
        """Record that a mutation led to a counterexample."""
        self._effective_mutations[mutation_name] = (
            self._effective_mutations.get(mutation_name, 0) + 1
        )

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate cases guided by previous results.

        Params:
            seed_states: Explicit seed states to mutate
            use_promising: Whether to use recorded promising states
            intensity: Search intensity (1=normal, 2=aggressive)
        """
        rng = random.Random(seed)

        use_promising = params.get("use_promising", True)
        intensity = params.get("intensity", 1)

        # Build seed states from various sources
        seed_states = list(params.get("seed_states", []))

        if use_promising and self._promising_states:
            # Add promising states with higher weight
            num_promising = min(len(self._promising_states), count // 2)
            seed_states.extend(rng.sample(self._promising_states, num_promising))

        # Adjust parameters based on intensity
        mutations_per_seed = params.get("mutations_per_seed", 3 * intensity)
        max_mutations = params.get("max_mutations_per_state", 2 + intensity)

        # Use the mutation generator with adjusted params
        mutation_params = {
            "seed_states": seed_states if seed_states else None,
            "mutations_per_seed": mutations_per_seed,
            "max_mutations_per_state": max_mutations,
            "grid_lengths": params.get("grid_lengths", [8, 16, 32]),
            "focus_collisions": params.get("focus_collisions", False),
        }

        return self._mutation_gen.generate(mutation_params, seed, count)

    def reset(self) -> None:
        """Reset the guided search state."""
        self._promising_states = []
        self._effective_mutations = {}
