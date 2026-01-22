"""Edge wrapping case generator.

Generates states specifically designed to test boundary wrapping behavior.
"""

import random
from typing import Any

from src.harness.case import Case
from src.harness.generators.base import Generator
from src.universe.types import Config, Symbol


class EdgeWrappingGenerator(Generator):
    """Generate states that test boundary wrapping.

    Creates states with particles near edges that will wrap around
    within the test horizon.

    Parameters:
        grid_lengths: Grid sizes to test
        edge_configs: Configurations for edge placement
    """

    def family_name(self) -> str:
        return "edge_wrapping_cases"

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate edge wrapping cases.

        Args:
            params: {
                "grid_lengths": [8, 16, 32],
                "include_collision_at_boundary": True,
            }
            seed: Random seed
            count: Number of cases to generate

        Returns:
            List of generated cases
        """
        rng = random.Random(seed)

        grid_lengths = params.get("grid_lengths", [8, 16, 32])
        include_boundary_collision = params.get("include_collision_at_boundary", True)

        cases = []
        params_hash = self.params_hash(params)

        # Different edge configurations
        edge_configs = [
            "right_at_end",      # > at last position
            "left_at_start",     # < at first position
            "approaching_boundary",  # Particles will collide at boundary
            "diverging_from_boundary",  # Particles moving away from boundary
            "multiple_wrapping",  # Multiple particles will wrap
        ]

        for i in range(count):
            config_name = edge_configs[i % len(edge_configs)]
            grid_length = grid_lengths[i % len(grid_lengths)]

            state = self._generate_edge_state(
                rng, config_name, grid_length, include_boundary_collision
            )

            case_seed = seed + i
            cases.append(Case(
                initial_state=state,
                config=Config(grid_length=grid_length),
                seed=case_seed,
                generator_family=self.family_name(),
                params_hash=params_hash,
                metadata={
                    "edge_config": config_name,
                },
            ))

        return cases

    def _generate_edge_state(
        self,
        rng: random.Random,
        config_name: str,
        grid_length: int,
        include_boundary_collision: bool,
    ) -> str:
        """Generate a state with the specified edge configuration."""
        cells = [Symbol.EMPTY.value] * grid_length

        if config_name == "right_at_end":
            # > at last position - will wrap to first position
            cells[-1] = Symbol.RIGHT.value

        elif config_name == "left_at_start":
            # < at first position - will wrap to last position
            cells[0] = Symbol.LEFT.value

        elif config_name == "approaching_boundary":
            # Particles that will collide at/near boundary
            # < at position 1, > at last position
            # After 1 step: < at 0, > wraps to 0 -> collision
            if grid_length >= 4:
                cells[1] = Symbol.LEFT.value
                cells[-1] = Symbol.RIGHT.value

        elif config_name == "diverging_from_boundary":
            # < at last position (moving away from end)
            # > at first position (moving away from start)
            cells[0] = Symbol.RIGHT.value
            cells[-1] = Symbol.LEFT.value

        elif config_name == "multiple_wrapping":
            # Multiple particles near edges
            if grid_length >= 6:
                cells[0] = Symbol.LEFT.value
                cells[1] = Symbol.LEFT.value
                cells[-2] = Symbol.RIGHT.value
                cells[-1] = Symbol.RIGHT.value

        # Optionally add some interior particles for variation
        if rng.random() < 0.3 and grid_length >= 8:
            mid = grid_length // 2
            if cells[mid] == Symbol.EMPTY.value:
                cells[mid] = rng.choice([Symbol.RIGHT.value, Symbol.LEFT.value])

        return "".join(cells)
