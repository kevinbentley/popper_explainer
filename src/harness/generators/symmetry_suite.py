"""Symmetry metamorphic test generator.

Generates test cases specifically for testing symmetry commutation laws.
"""

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

    Parameters:
        grid_lengths: Grid sizes to test
        transform: Which symmetry transform to target
        bias_asymmetric: Weight toward asymmetric states
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

        cases = []
        params_hash = self.params_hash(params)

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
            cases.append(Case(
                initial_state=state,
                config=Config(grid_length=grid_length),
                seed=case_seed,
                generator_family=self.family_name(),
                params_hash=params_hash,
                metadata={
                    "transform": transform,
                    "state_type": state_type,
                },
            ))

        return cases

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
