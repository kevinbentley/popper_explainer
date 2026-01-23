"""Generator for pathological and uniform test cases.

These cases test edge conditions that random generators often miss:
- Uniform grids (all same symbol)
- Alternating patterns
- Single-element grids
- Empty grids

This generator is critical for catching false positives.
"""

from typing import Any

from src.harness.case import Case
from src.harness.generators.base import Generator
from src.universe.types import Config


class PathologicalGenerator(Generator):
    """Generates pathological test cases that stress-test edge conditions.

    These deterministic cases are essential for falsifying laws that
    make implicit assumptions about state diversity.
    """

    def family_name(self) -> str:
        return "pathological_cases"

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate pathological test cases.

        Args:
            params: Generator parameters
                - min_length: Minimum grid length (default: 1)
                - max_length: Maximum grid length (default: 20)
                - include_empty: Include empty grid (default: True)
            seed: Random seed (mostly ignored - cases are deterministic)
            count: Requested count (actual count may differ)

        Returns:
            List of pathological test cases
        """
        min_length = params.get("min_length", 1)
        max_length = params.get("max_length", 20)
        include_empty = params.get("include_empty", True)

        cases = []
        params_hash = self.params_hash(params)
        lengths = [1, 2, 3, 5, 8, 10, 15, 20]
        lengths = [l for l in lengths if min_length <= l <= max_length]

        case_seed = seed

        # Empty grid - note: Config requires grid_length >= 1,
        # so we use a minimal length and empty state
        # The simulator should handle empty strings gracefully
        if include_empty:
            # Skip empty grid as Config requires grid_length >= 1
            # Empty states are typically not valid in this universe
            pass

        for length in lengths:
            # Uniform grids - all same symbol
            # Note: X (collision) is excluded as it's not a valid initial state -
            # collisions are the result of particles meeting, not an initial config
            for symbol, type_name in [
                (">", "uniform_right"),
                ("<", "uniform_left"),
                (".", "uniform_empty"),
            ]:
                cases.append(Case(
                    initial_state=symbol * length,
                    config=Config(grid_length=length),
                    seed=case_seed,
                    generator_family=self.family_name(),
                    params_hash=params_hash,
                    generator_params={"type": type_name, "length": length},
                ))
                case_seed += 1

            # Alternating patterns (for even lengths)
            if length >= 2:
                # ><><><
                alt_rl = "".join(">" if i % 2 == 0 else "<" for i in range(length))
                cases.append(Case(
                    initial_state=alt_rl,
                    config=Config(grid_length=length),
                    seed=case_seed,
                    generator_family=self.family_name(),
                    params_hash=params_hash,
                    generator_params={"type": "alternating_rl", "length": length},
                ))
                case_seed += 1

                # <><><>
                alt_lr = "".join("<" if i % 2 == 0 else ">" for i in range(length))
                cases.append(Case(
                    initial_state=alt_lr,
                    config=Config(grid_length=length),
                    seed=case_seed,
                    generator_family=self.family_name(),
                    params_hash=params_hash,
                    generator_params={"type": "alternating_lr", "length": length},
                ))
                case_seed += 1

            # Two-symbol boundaries
            if length >= 4:
                half = length // 2
                # Half right, half left (collision boundary)
                cases.append(Case(
                    initial_state=">" * half + "<" * (length - half),
                    config=Config(grid_length=length),
                    seed=case_seed,
                    generator_family=self.family_name(),
                    params_hash=params_hash,
                    generator_params={"type": "half_right_left", "length": length},
                ))
                case_seed += 1

                # Half left, half right (diverging)
                cases.append(Case(
                    initial_state="<" * half + ">" * (length - half),
                    config=Config(grid_length=length),
                    seed=case_seed,
                    generator_family=self.family_name(),
                    params_hash=params_hash,
                    generator_params={"type": "half_left_right", "length": length},
                ))
                case_seed += 1

        # Limit to requested count if specified
        if count > 0 and len(cases) > count:
            # Prioritize diverse case types
            return cases[:count]

        return cases
