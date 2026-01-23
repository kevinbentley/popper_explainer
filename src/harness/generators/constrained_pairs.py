"""Constrained pair interaction generator.

Generates states with specific particle configurations designed to
test collision behavior and conservation laws.
"""

import random
from typing import Any

from src.harness.case import Case
from src.harness.generators.base import Generator
from src.universe.types import Config, Symbol


class ConstrainedPairGenerator(Generator):
    """Generate states with controlled particle interactions.

    Creates states with specific patterns:
    - Approaching pairs (>..<) that will collide
    - Diverging pairs (<..>) that won't collide
    - Adjacent particles (><)
    - Collision states (X)

    Parameters:
        patterns: List of patterns to use
        grid_lengths: Grid sizes
        include_collision_state: Whether to include X in initial states
    """

    # Predefined patterns that test specific behaviors
    PATTERNS = {
        "approaching": ">.<",      # Will collide
        "diverging": "<.>",        # Won't collide
        "adjacent": "><",          # Pass through
        "collision": "X",          # Already colliding
        "approaching_2": ">..<",   # Collision in 2 steps
        "triple_right": ">>>",     # Multiple particles, same direction
        "triple_left": "<<<",      # Multiple particles, same direction
        "alternating": "><><",     # Alternating directions
    }

    def family_name(self) -> str:
        return "constrained_pair_interactions"

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate constrained pair cases.

        Args:
            params: {
                "patterns": ["approaching", "collision"],  # which patterns
                "grid_lengths": [8, 16, 32],
                "include_collision_state": True,
            }
            seed: Random seed
            count: Number of cases to generate

        Returns:
            List of generated cases
        """
        rng = random.Random(seed)

        patterns = params.get("patterns", list(self.PATTERNS.keys()))
        grid_lengths = params.get("grid_lengths", [8, 16, 32])
        include_collision = params.get("include_collision_state", True)

        # Validate pattern names - filter out invalid ones
        valid_patterns = [p for p in patterns if p in self.PATTERNS]
        if not valid_patterns:
            # Fall back to defaults if no valid patterns
            valid_patterns = list(self.PATTERNS.keys())
        patterns = valid_patterns

        if not include_collision:
            patterns = [p for p in patterns if p != "collision"]

        cases = []
        params_hash = self.params_hash(params)

        for i in range(count):
            pattern_name = patterns[i % len(patterns)]
            grid_length = grid_lengths[i % len(grid_lengths)]
            pattern = self.PATTERNS[pattern_name]

            # Place pattern in the grid
            state = self._place_pattern(rng, pattern, grid_length)

            case_seed = seed + i
            cases.append(Case(
                initial_state=state,
                config=Config(grid_length=grid_length),
                seed=case_seed,
                generator_family=self.family_name(),
                params_hash=params_hash,
                metadata={
                    "pattern": pattern_name,
                    "pattern_str": pattern,
                },
            ))

        return cases

    def _place_pattern(
        self,
        rng: random.Random,
        pattern: str,
        grid_length: int,
    ) -> str:
        """Place a pattern in a grid, padding with empty cells."""
        if len(pattern) > grid_length:
            # Truncate pattern if too long
            pattern = pattern[:grid_length]

        padding_needed = grid_length - len(pattern)

        if padding_needed == 0:
            return pattern

        # Random position for the pattern
        left_padding = rng.randint(0, padding_needed)
        right_padding = padding_needed - left_padding

        return (
            Symbol.EMPTY.value * left_padding +
            pattern +
            Symbol.EMPTY.value * right_padding
        )
