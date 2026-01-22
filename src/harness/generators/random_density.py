"""Random density sweep generator.

Generates random initial states at various particle densities.
"""

import random
from typing import Any

from src.harness.case import Case
from src.harness.generators.base import Generator
from src.universe.types import Config, Symbol


class RandomDensityGenerator(Generator):
    """Generate random states at specified density levels.

    Parameters:
        densities: List of target densities (0.0 to 1.0)
        grid_lengths: List of grid lengths to use
        include_collisions: Whether to include X in initial states
    """

    def family_name(self) -> str:
        return "random_density_sweep"

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate random density cases.

        Args:
            params: {
                "densities": [0.1, 0.3, 0.6],  # particle densities
                "grid_lengths": [10, 20, 50],  # grid sizes
                "include_collisions": False,   # include X states
            }
            seed: Random seed
            count: Number of cases to generate

        Returns:
            List of generated cases
        """
        rng = random.Random(seed)

        densities = params.get("densities", [0.1, 0.3, 0.5])
        grid_lengths = params.get("grid_lengths", [10, 20, 40])
        include_collisions = params.get("include_collisions", False)

        cases = []
        params_hash = self.params_hash(params)

        for i in range(count):
            # Cycle through densities and grid lengths
            density = densities[i % len(densities)]
            grid_length = grid_lengths[i % len(grid_lengths)]

            # Generate random state at this density
            state = self._generate_state(
                rng, grid_length, density, include_collisions
            )

            case_seed = seed + i
            cases.append(Case(
                initial_state=state,
                config=Config(grid_length=grid_length),
                seed=case_seed,
                generator_family=self.family_name(),
                params_hash=params_hash,
                metadata={
                    "density": density,
                    "target_particles": int(grid_length * density),
                },
            ))

        return cases

    def _generate_state(
        self,
        rng: random.Random,
        grid_length: int,
        density: float,
        include_collisions: bool,
    ) -> str:
        """Generate a random state at the given density."""
        # Calculate number of particles
        num_particles = max(0, int(grid_length * density))
        num_particles = min(num_particles, grid_length)  # Can't exceed grid size

        # Start with empty grid
        cells = [Symbol.EMPTY.value] * grid_length

        # Place particles randomly
        available_positions = list(range(grid_length))
        rng.shuffle(available_positions)

        particles_placed = 0
        for pos in available_positions:
            if particles_placed >= num_particles:
                break

            if include_collisions and rng.random() < 0.1 and particles_placed + 2 <= num_particles:
                # Place a collision (counts as 2 particles)
                cells[pos] = Symbol.COLLISION.value
                particles_placed += 2
            else:
                # Place a directional particle
                if rng.random() < 0.5:
                    cells[pos] = Symbol.RIGHT.value
                else:
                    cells[pos] = Symbol.LEFT.value
                particles_placed += 1

        return "".join(cells)
