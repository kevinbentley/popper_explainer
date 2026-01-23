"""Extreme state generator for satisfying unusual antecedents.

Generates states that satisfy extreme conditions like:
- Full collision grids (all X)
- Full particle grids (all > or all <)
- Maximum density states
- Other edge cases that normal generators rarely produce
"""

import random
from typing import Any

from src.harness.case import Case
from src.harness.generators.base import Generator
from src.universe.types import Config, Symbol


class ExtremeStatesGenerator(Generator):
    """Generate extreme states for antecedent coverage.

    This generator targets states that satisfy unusual antecedents
    that normal generators rarely produce, such as:
    - Full collision grid: "XXXX..." (CollisionCells == GridLength)
    - Full right movers: ">>>>..."
    - Full left movers: "<<<<..."
    - Alternating collisions: "X.X.X..."
    - Near-maximum density states

    Use this generator to ensure implication laws don't pass vacuously
    because their antecedents are never triggered.

    Parameters:
        grid_lengths: Grid sizes to generate
        include_full_collision: Generate full collision grids
        include_full_movers: Generate all-right or all-left grids
        include_alternating: Generate alternating patterns
        include_high_density: Generate high-density random states
    """

    def family_name(self) -> str:
        return "extreme_states"

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate extreme state test cases.

        Args:
            params: {
                "grid_lengths": [4, 8, 16, 32],
                "include_full_collision": True,
                "include_full_movers": True,
                "include_alternating": True,
                "include_high_density": True,
            }
            seed: Random seed
            count: Number of cases to generate

        Returns:
            List of generated cases
        """
        rng = random.Random(seed)

        grid_lengths = params.get("grid_lengths", [4, 8, 16, 32])
        include_full_collision = params.get("include_full_collision", True)
        include_full_movers = params.get("include_full_movers", True)
        include_alternating = params.get("include_alternating", True)
        include_high_density = params.get("include_high_density", True)

        cases = []
        params_hash = self.params_hash(params)

        # Build list of extreme state generators
        generators = []

        if include_full_collision:
            generators.append(("full_collision", self._generate_full_collision))

        if include_full_movers:
            generators.append(("full_right", self._generate_full_right))
            generators.append(("full_left", self._generate_full_left))

        if include_alternating:
            generators.append(("alternating_collision", self._generate_alternating_collision))
            generators.append(("alternating_rl", self._generate_alternating_rl))

        if include_high_density:
            generators.append(("high_density", self._generate_high_density))
            generators.append(("near_full_collision", self._generate_near_full_collision))

        if not generators:
            return cases

        for i in range(count):
            # Cycle through grid lengths
            grid_length = grid_lengths[i % len(grid_lengths)]

            # Cycle through generator types
            gen_name, gen_func = generators[i % len(generators)]

            state = gen_func(rng, grid_length)
            case_seed = seed + i

            cases.append(Case(
                initial_state=state,
                config=Config(grid_length=grid_length),
                seed=case_seed,
                generator_family=self.family_name(),
                params_hash=params_hash,
                metadata={
                    "extreme_type": gen_name,
                    "density": self._compute_density(state),
                    "collision_fraction": state.count("X") / len(state) if state else 0,
                },
            ))

        return cases

    def _generate_full_collision(self, rng: random.Random, length: int) -> str:
        """Generate a full collision grid: XXXX..."""
        return Symbol.COLLISION.value * length

    def _generate_full_right(self, rng: random.Random, length: int) -> str:
        """Generate all right movers: >>>>..."""
        return Symbol.RIGHT.value * length

    def _generate_full_left(self, rng: random.Random, length: int) -> str:
        """Generate all left movers: <<<<..."""
        return Symbol.LEFT.value * length

    def _generate_alternating_collision(self, rng: random.Random, length: int) -> str:
        """Generate alternating collisions: X.X.X..."""
        cells = []
        for i in range(length):
            if i % 2 == 0:
                cells.append(Symbol.COLLISION.value)
            else:
                cells.append(Symbol.EMPTY.value)
        return "".join(cells)

    def _generate_alternating_rl(self, rng: random.Random, length: int) -> str:
        """Generate alternating right-left: ><><..."""
        cells = []
        for i in range(length):
            if i % 2 == 0:
                cells.append(Symbol.RIGHT.value)
            else:
                cells.append(Symbol.LEFT.value)
        return "".join(cells)

    def _generate_high_density(self, rng: random.Random, length: int) -> str:
        """Generate high-density random state (90%+ occupied)."""
        cells = []
        for _ in range(length):
            r = rng.random()
            if r < 0.45:
                cells.append(Symbol.RIGHT.value)
            elif r < 0.90:
                cells.append(Symbol.LEFT.value)
            else:
                cells.append(Symbol.EMPTY.value)
        return "".join(cells)

    def _generate_near_full_collision(self, rng: random.Random, length: int) -> str:
        """Generate mostly-collision state with a few gaps."""
        cells = []
        for i in range(length):
            if rng.random() < 0.85:  # 85% collision
                cells.append(Symbol.COLLISION.value)
            else:
                # Small chance of other symbols
                cells.append(rng.choice([
                    Symbol.EMPTY.value,
                    Symbol.RIGHT.value,
                    Symbol.LEFT.value,
                ]))
        return "".join(cells)

    def _compute_density(self, state: str) -> float:
        """Compute particle density."""
        if not state:
            return 0.0
        particles = state.count(">") + state.count("<") + 2 * state.count("X")
        return particles / len(state)
