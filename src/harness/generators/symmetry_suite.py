"""Symmetry metamorphic test generator.

Generates test cases specifically for testing symmetry commutation laws.
"""

import hashlib
import random
from typing import Any

from src.harness.case import Case
from src.harness.generators.base import Generator
from src.universe.types import Config, Symbol


class SymmetryMetamorphicGenerator(Generator):
    """Generate states for metamorphic symmetry testing.

    Creates diverse states designed to exercise symmetry properties:
    - Symmetric states (should be easy cases)
    - Asymmetric states (more likely to find violations)
    - States with collisions (complex dynamics)

    For shift_k transform, ensures:
    - k values are in {1..L-1} (never 0 or L, which are identity)
    - At least 2 different k values per grid length L
    - Logs k, L, state_hash for reproducibility

    Parameters:
        grid_lengths: Grid sizes to test
        transform: Which symmetry transform to target
        bias_asymmetric: Weight toward asymmetric states
        min_k_values_per_length: Minimum distinct k values per grid length (for shift_k)
    """

    def family_name(self) -> str:
        return "symmetry_metamorphic_suite"

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate symmetry test cases.

        Args:
            params: {
                "grid_lengths": [8, 16, 32],
                "transform": "mirror_swap",  # target transform
                "bias_asymmetric": 0.7,  # fraction of asymmetric cases
                "min_k_values_per_length": 2,  # for shift_k: minimum distinct k per L
            }
            seed: Random seed
            count: Number of cases to generate

        Returns:
            List of generated cases
        """
        rng = random.Random(seed)

        grid_lengths = params.get("grid_lengths", [8, 16, 32])
        transform = params.get("transform", "mirror_swap")
        bias_asymmetric = params.get("bias_asymmetric", 0.7)
        min_k_values = params.get("min_k_values_per_length", 2)

        cases = []
        params_hash = self.params_hash(params)

        # For shift_k, track k values used per grid length to ensure diversity
        if transform == "shift_k":
            k_values_used: dict[int, set[int]] = {L: set() for L in grid_lengths}

        for i in range(count):
            grid_length = grid_lengths[i % len(grid_lengths)]

            # Decide whether to generate symmetric or asymmetric state
            if rng.random() < bias_asymmetric:
                state = self._generate_asymmetric_state(rng, grid_length, transform)
                state_type = "asymmetric"
            else:
                state = self._generate_symmetric_state(rng, grid_length, transform)
                state_type = "symmetric"

            case_seed = seed + i
            state_hash = hashlib.sha256(state.encode()).hexdigest()[:8]

            # Build metadata
            metadata = {
                "transform": transform,
                "state_type": state_type,
                "state_hash": state_hash,
            }

            # For shift_k, assign a k value
            if transform == "shift_k":
                k = self._select_k_value(
                    rng, grid_length, k_values_used[grid_length], min_k_values
                )
                k_values_used[grid_length].add(k)
                metadata["k"] = k
                metadata["L"] = grid_length

            cases.append(Case(
                initial_state=state,
                config=Config(grid_length=grid_length),
                seed=case_seed,
                generator_family=self.family_name(),
                params_hash=params_hash,
                metadata=metadata,
            ))

        return cases

    def _select_k_value(
        self,
        rng: random.Random,
        grid_length: int,
        used_k: set[int],
        min_k_values: int,
    ) -> int:
        """Select a k value for shift_k, ensuring diversity.

        Rules:
        - k must be in {1..L-1} (never 0 or L, which are identity)
        - Prioritize unused k values until min_k_values are covered
        - After that, select randomly from valid range

        Args:
            rng: Random number generator
            grid_length: Grid length L
            used_k: Set of k values already used for this L
            min_k_values: Minimum distinct k values to use

        Returns:
            k value in {1..L-1}
        """
        valid_range = list(range(1, grid_length))  # {1..L-1}

        if not valid_range:
            return 1  # Edge case: L=1, only k=0 is valid but that's identity

        # If we haven't used enough distinct k values, prioritize unused ones
        if len(used_k) < min_k_values:
            unused = [k for k in valid_range if k not in used_k]
            if unused:
                return rng.choice(unused)

        # Otherwise, select randomly from valid range
        return rng.choice(valid_range)

    def _generate_symmetric_state(
        self,
        rng: random.Random,
        grid_length: int,
        transform: str,
    ) -> str:
        """Generate a state that is symmetric under the given transform."""
        if transform == "mirror_swap":
            # For mirror_swap symmetry: reverse + direction swap
            # A state S is symmetric if mirror_swap(S) == S
            # This means S[i] and S[n-1-i] must be direction-swapped versions
            return self._generate_mirror_swap_symmetric(rng, grid_length)
        else:
            # Default: just generate random state
            return self._generate_random_state(rng, grid_length, 0.3)

    def _generate_asymmetric_state(
        self,
        rng: random.Random,
        grid_length: int,
        transform: str,
    ) -> str:
        """Generate a state that is asymmetric under the given transform."""
        if transform == "mirror_swap":
            # Deliberately asymmetric: put particles only on one side
            return self._generate_biased_state(rng, grid_length)
        else:
            return self._generate_random_state(rng, grid_length, 0.3)

    def _generate_mirror_swap_symmetric(
        self,
        rng: random.Random,
        grid_length: int,
    ) -> str:
        """Generate a state symmetric under mirror_swap.

        For mirror_swap(S) == S:
        - S[i] = '>' implies S[n-1-i] = '<'
        - S[i] = '<' implies S[n-1-i] = '>'
        - S[i] = '.' implies S[n-1-i] = '.'
        - S[i] = 'X' implies S[n-1-i] = 'X'
        """
        cells = [Symbol.EMPTY.value] * grid_length

        # Fill first half, mirror to second half
        half = grid_length // 2
        for i in range(half):
            if rng.random() < 0.3:  # 30% chance of particle
                if rng.random() < 0.5:
                    cells[i] = Symbol.RIGHT.value
                    cells[grid_length - 1 - i] = Symbol.LEFT.value
                else:
                    cells[i] = Symbol.LEFT.value
                    cells[grid_length - 1 - i] = Symbol.RIGHT.value
            elif rng.random() < 0.05:  # 5% chance of collision
                cells[i] = Symbol.COLLISION.value
                cells[grid_length - 1 - i] = Symbol.COLLISION.value

        # Handle middle element for odd-length grids
        if grid_length % 2 == 1:
            mid = grid_length // 2
            # Middle must be empty or X (self-symmetric under direction swap)
            if rng.random() < 0.1:
                cells[mid] = Symbol.COLLISION.value

        return "".join(cells)

    def _generate_biased_state(
        self,
        rng: random.Random,
        grid_length: int,
    ) -> str:
        """Generate a state biased toward one direction."""
        cells = [Symbol.EMPTY.value] * grid_length

        # Mostly right-moving particles on left side
        for i in range(grid_length // 2):
            if rng.random() < 0.4:
                cells[i] = Symbol.RIGHT.value

        # Maybe one left-moving particle on right side for asymmetry
        if rng.random() < 0.3:
            pos = rng.randint(grid_length // 2, grid_length - 1)
            cells[pos] = Symbol.LEFT.value

        return "".join(cells)

    def _generate_random_state(
        self,
        rng: random.Random,
        grid_length: int,
        density: float,
    ) -> str:
        """Generate a random state at the given density."""
        cells = []
        for _ in range(grid_length):
            if rng.random() < density:
                symbol = rng.choice([Symbol.RIGHT.value, Symbol.LEFT.value])
            else:
                symbol = Symbol.EMPTY.value
            cells.append(symbol)
        return "".join(cells)
