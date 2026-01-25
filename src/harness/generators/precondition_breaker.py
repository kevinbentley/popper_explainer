"""Precondition-breaking and adversarial case generators.

These generators are designed to stress-test laws by:
1. Violating preconditions to check if laws hold more broadly
2. Testing multiplicity (crowding) scenarios with many-to-one collisions
3. Forcing periodic boundary edge cases
4. Universal stress states that every law must pass

Key insight: If the AI proposes a law with narrow preconditions like
"CollisionCells == 0", we should test whether the law holds when
CollisionCells > 0. This encourages discovering universal laws.

UNIVERSAL STRESS STATES (must always be tested):
- ><><><    : Maximum collision / occupancy limits
- >>.<<     : Multiplicity (4 particles merging into 1 cell)
- XX        : Chain collisions without free movers ("X-Battery")
- ....>     : Circular topology (particle at N-1 going right, wraps to 0)
- <....     : Circular topology (particle at 0 going left, wraps to N-1)
"""

import random
from typing import Any, Callable

from src.harness.case import Case
from src.harness.generators.base import Generator
from src.universe.types import Config, Symbol
from src.universe.observables import (
    count_collision,
    incoming_collisions,
    count_right,
    count_left,
    free_movers,
)


class PreconditionBreakingGenerator(Generator):
    """Generate states that violate common preconditions.

    The goal is to test whether laws hold more broadly than their
    stated preconditions suggest. If a law says "when X == 0, then Y",
    we test states where X > 0 to see if the law still holds.
    """

    def family_name(self) -> str:
        return "precondition_breaking"

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate precondition-breaking cases.

        Args:
            params: {
                "target_precondition": str - which precondition to violate
                    Options: "collision_cells", "incoming_collisions", "both"
                "grid_lengths": list[int] - grid sizes
                "collision_density": float - fraction of cells with collisions
            }
            seed: Random seed
            count: Number of cases

        Returns:
            List of cases that violate common preconditions
        """
        rng = random.Random(seed)
        target = params.get("target_precondition", "both")
        grid_lengths = params.get("grid_lengths", [8, 12, 16, 20])
        collision_density = params.get("collision_density", 0.2)

        cases = []
        params_hash = self.params_hash(params)

        for i in range(count):
            length = grid_lengths[i % len(grid_lengths)]

            if target == "collision_cells" or (target == "both" and i % 2 == 0):
                state = self._generate_with_collisions(rng, length, collision_density)
                case_type = "with_collisions"
            else:
                state = self._generate_with_incoming_collisions(rng, length)
                case_type = "with_incoming_collisions"

            cases.append(Case(
                initial_state=state,
                config=Config(grid_length=length),
                seed=seed + i,
                generator_family=self.family_name(),
                params_hash=params_hash,
                metadata={
                    "case_type": case_type,
                    "collision_cells": count_collision(state),
                    "incoming_collisions": incoming_collisions(state),
                },
            ))

        return cases

    def _generate_with_collisions(
        self, rng: random.Random, length: int, density: float
    ) -> str:
        """Generate a state with existing collision cells (X).

        Note: X cells are normally created by particles colliding.
        We include them directly to test laws that assume no collisions.
        """
        cells = [Symbol.EMPTY.value] * length
        num_collisions = max(1, int(length * density))

        # Place collision cells
        positions = rng.sample(range(length), min(num_collisions, length))
        for pos in positions:
            cells[pos] = Symbol.COLLISION.value

        # Add some free movers in remaining positions
        empty_positions = [i for i in range(length) if cells[i] == Symbol.EMPTY.value]
        num_movers = min(len(empty_positions) // 2, rng.randint(1, 4))
        for pos in rng.sample(empty_positions, num_movers):
            cells[pos] = rng.choice([Symbol.RIGHT.value, Symbol.LEFT.value])

        return "".join(cells)

    def _generate_with_incoming_collisions(
        self, rng: random.Random, length: int
    ) -> str:
        """Generate a state with pending incoming collisions.

        Creates >< patterns that will collide at t+1.
        """
        cells = [Symbol.EMPTY.value] * length

        # Create some collision setups (>.<, ><, or convergent pairs)
        num_setups = rng.randint(1, max(1, length // 4))

        for _ in range(num_setups):
            # Find a position for collision setup
            pos = rng.randint(0, length - 1)
            setup_type = rng.choice(["adjacent", "gap_1", "gap_2"])

            if setup_type == "adjacent":
                # >< - will collide with wrapping considered
                cells[pos] = Symbol.RIGHT.value
                cells[(pos + 1) % length] = Symbol.LEFT.value
            elif setup_type == "gap_1":
                # >.< - will collide at middle
                cells[pos] = Symbol.RIGHT.value
                cells[(pos + 2) % length] = Symbol.LEFT.value
            else:
                # >..< - will collide in 2 steps
                cells[pos] = Symbol.RIGHT.value
                cells[(pos + 3) % length] = Symbol.LEFT.value

        return "".join(cells)


class MultiplicityGenerator(Generator):
    """Generate states with many-to-one collision scenarios (crowding).

    Key insight: The AI often assumes collisions are 1v1. But in reality,
    multiple particles can converge on a single cell. For example:
    - >>.<<  - center receives 2 right-movers and 2 left-movers
    - >>.<< - 4 particles converge on one cell
    - >.X.< - collision cell also participates

    This tests whether laws handle "crowded" collision scenarios.
    """

    def family_name(self) -> str:
        return "multiplicity_crowding"

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate multiplicity test cases.

        Args:
            params: {
                "grid_lengths": list[int] - grid sizes
                "max_convergence": int - max particles converging on one cell
                "include_chain_reactions": bool - include cascading collisions
            }
            seed: Random seed
            count: Number of cases

        Returns:
            List of crowding test cases
        """
        rng = random.Random(seed)
        grid_lengths = params.get("grid_lengths", [8, 12, 16, 24])
        max_convergence = params.get("max_convergence", 4)
        include_chains = params.get("include_chain_reactions", True)

        cases = []
        params_hash = self.params_hash(params)

        # Deterministic patterns for specific convergence scenarios
        patterns = self._get_convergence_patterns()

        for i in range(count):
            length = grid_lengths[i % len(grid_lengths)]

            if i < len(patterns):
                # Use a deterministic pattern
                pattern_name, pattern_fn = patterns[i % len(patterns)]
                state = pattern_fn(length)
                case_type = pattern_name
            elif include_chains and i % 3 == 0:
                # Generate chain reaction setup
                state = self._generate_chain_collision(rng, length)
                case_type = "chain_reaction"
            else:
                # Generate random convergence
                convergence = rng.randint(2, max_convergence)
                state = self._generate_convergence(rng, length, convergence)
                case_type = f"convergence_{convergence}"

            cases.append(Case(
                initial_state=state,
                config=Config(grid_length=length),
                seed=seed + i,
                generator_family=self.family_name(),
                params_hash=params_hash,
                metadata={
                    "case_type": case_type,
                    "incoming_collisions": incoming_collisions(state),
                },
            ))

        return cases

    def _get_convergence_patterns(self) -> list[tuple[str, Callable[[int], str]]]:
        """Get deterministic convergence patterns."""
        return [
            # 4-particle convergence: >>.<<
            ("four_way_collision", lambda n: self._center_pattern(n, ">>", "<<")),

            # 6-particle convergence: >>>.<<<
            ("six_way_collision", lambda n: self._center_pattern(n, ">>>", "<<<")),

            # Multiple convergence points: >.<.>.<
            ("multi_point", lambda n: self._multi_point_pattern(n)),

            # With collision in the mix: >.X.<
            ("collision_in_path", lambda n: self._collision_in_path(n)),

            # All converging to center: >>>>>>..<<<<<<
            ("mass_convergence", lambda n: self._mass_convergence(n)),

            # Alternating approaching: ><><><><
            ("alternating_approach", lambda n: "><" * (n // 2) + (">" if n % 2 else "")),

            # X-Battery: Adjacent X cells emit particles that collide
            # This tests chain reactions without free movers
            ("x_battery_pair", lambda n: self._x_battery(n, 2)),
            ("x_battery_triple", lambda n: self._x_battery(n, 3)),
            ("x_battery_chain", lambda n: self._x_battery(n, max(2, n // 4))),

            # X surrounded by free movers: >XX<
            ("x_squeeze", lambda n: self._x_squeeze(n)),
        ]

    def _center_pattern(self, length: int, left_part: str, right_part: str) -> str:
        """Create a pattern centered in the grid."""
        total = len(left_part) + len(right_part)
        if length < total:
            return left_part[:length//2] + right_part[:length - length//2]

        padding = length - total
        left_pad = padding // 2
        right_pad = padding - left_pad
        return "." * left_pad + left_part + right_part + "." * right_pad

    def _multi_point_pattern(self, length: int) -> str:
        """Create multiple convergence points."""
        if length < 6:
            return "><" + "." * (length - 2)
        # >.<.>.< pattern repeated
        pattern = ">.<"
        result = ""
        while len(result) + len(pattern) <= length:
            result += pattern
        result += "." * (length - len(result))
        return result

    def _collision_in_path(self, length: int) -> str:
        """Create state with existing collision in collision path."""
        if length < 5:
            return ">" + "X" * (length - 2) + "<" if length >= 3 else "><"

        mid = length // 2
        cells = ["."] * length
        cells[mid - 2] = ">"
        cells[mid] = "X"  # Existing collision in path
        cells[mid + 2] = "<"
        return "".join(cells)

    def _mass_convergence(self, length: int) -> str:
        """All particles converging toward center."""
        if length < 4:
            return "><" if length >= 2 else ">"

        half = length // 2
        # Fill left half with >, right half with <
        rights = ">" * (half - 1)
        lefts = "<" * (length - half - 1)
        gap = ".."  # Small gap in center
        return rights + gap + lefts

    def _x_battery(self, length: int, num_x: int) -> str:
        """Create X-Battery: adjacent collision cells that emit particles.

        Key insight: Even with ZERO free movers (> or <), adjacent X cells
        will emit particles that collide, creating chain reactions.

        Physics: Each X emits one > and one <. Adjacent X cells have their
        emissions collide immediately, creating new X cells.

        Example: XX at t=0 -> <X>X< at t=1 (each X emits, inner particles collide)

        This falsifies laws like "no free movers = no new collisions".
        """
        if length < num_x:
            return "X" * length

        # Center the X chain
        padding = length - num_x
        left_pad = padding // 2
        right_pad = padding - left_pad
        return "." * left_pad + "X" * num_x + "." * right_pad

    def _x_squeeze(self, length: int) -> str:
        """Create X cells surrounded by incoming particles: >XX<

        Tests what happens when free movers collide with collision cells
        that are also emitting particles.
        """
        if length < 4:
            return "XX" if length >= 2 else "X"

        # >XX< centered
        num_x = min(3, (length - 2) // 2)
        x_part = "X" * num_x
        total = len(x_part) + 2  # +2 for > and <
        padding = length - total
        left_pad = padding // 2
        right_pad = padding - left_pad
        return "." * left_pad + ">" + x_part + "<" + "." * right_pad

    def _generate_convergence(
        self, rng: random.Random, length: int, num_particles: int
    ) -> str:
        """Generate random convergence scenario."""
        cells = [Symbol.EMPTY.value] * length

        # Pick convergence point
        target = rng.randint(2, length - 3)

        # Place right-movers to the left of target
        num_right = num_particles // 2
        for i in range(num_right):
            pos = (target - 1 - i) % length
            if cells[pos] == Symbol.EMPTY.value:
                cells[pos] = Symbol.RIGHT.value

        # Place left-movers to the right of target
        num_left = num_particles - num_right
        for i in range(num_left):
            pos = (target + 1 + i) % length
            if cells[pos] == Symbol.EMPTY.value:
                cells[pos] = Symbol.LEFT.value

        return "".join(cells)

    def _generate_chain_collision(self, rng: random.Random, length: int) -> str:
        """Generate state that will have cascading collisions."""
        cells = [Symbol.EMPTY.value] * length

        # Create multiple collision points at different times
        # t=1 collision at pos, t=2 collision at pos+2, etc.
        num_chains = min(3, length // 6)

        for chain in range(num_chains):
            base = (chain * length // num_chains) % length

            # Setup: >..< (collides at t=2) followed by >< (collides at t=1)
            cells[base] = Symbol.RIGHT.value
            cells[(base + 3) % length] = Symbol.LEFT.value
            cells[(base + 5) % length] = Symbol.RIGHT.value
            cells[(base + 6) % length] = Symbol.LEFT.value

        return "".join(cells)


class PeriodicBoundaryGenerator(Generator):
    """Enhanced generator for periodic boundary edge cases.

    Tests scenarios where particles wrap around the N-1 to 0 boundary,
    including collisions that span the boundary.
    """

    def family_name(self) -> str:
        return "periodic_boundary_stress"

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate periodic boundary stress tests.

        Args:
            params: {
                "grid_lengths": list[int] - grid sizes (smaller = more wrapping)
                "focus_wrap_collision": bool - focus on collisions at boundary
            }
            seed: Random seed
            count: Number of cases

        Returns:
            List of boundary stress test cases
        """
        rng = random.Random(seed)
        # Use smaller grids to force more wrapping
        grid_lengths = params.get("grid_lengths", [4, 5, 6, 8, 10])
        focus_wrap = params.get("focus_wrap_collision", True)

        cases = []
        params_hash = self.params_hash(params)

        # Deterministic boundary scenarios
        scenarios = self._get_boundary_scenarios()

        for i in range(count):
            length = grid_lengths[i % len(grid_lengths)]

            if i < len(scenarios):
                scenario_name, scenario_fn = scenarios[i % len(scenarios)]
                state = scenario_fn(length)
                case_type = scenario_name
            elif focus_wrap:
                state = self._generate_wrap_collision(rng, length)
                case_type = "wrap_collision"
            else:
                state = self._generate_boundary_stress(rng, length)
                case_type = "boundary_stress"

            cases.append(Case(
                initial_state=state,
                config=Config(grid_length=length),
                seed=seed + i,
                generator_family=self.family_name(),
                params_hash=params_hash,
                metadata={
                    "case_type": case_type,
                    "boundary_collision": self._has_boundary_collision(state),
                },
            ))

        return cases

    def _get_boundary_scenarios(self) -> list[tuple[str, Callable[[int], str]]]:
        """Get deterministic boundary scenarios."""
        return [
            # > at end, will wrap to 0
            ("right_at_end", lambda n: "." * (n - 1) + ">"),

            # < at start, will wrap to n-1
            ("left_at_start", lambda n: "<" + "." * (n - 1)),

            # Collision across boundary: < at 0, > at n-1
            ("boundary_collision", lambda n: "<" + "." * (n - 2) + ">"),

            # Both edges occupied, moving toward boundary
            ("both_edges", lambda n: "<" + "." * (n - 2) + ">"),

            # Particles will pass through boundary
            ("through_boundary", lambda n: ">>" + "." * (n - 4) + "<<" if n >= 4 else "><"),

            # Dense at boundary
            ("dense_boundary", lambda n: "><" + "." * (n - 4) + "<>" if n >= 4 else "><"),

            # Collision at 0
            ("collision_at_zero", self._collision_at_zero),

            # Collision at n-1
            ("collision_at_end", self._collision_at_end),
        ]

    def _collision_at_zero(self, length: int) -> str:
        """Create state where collision will happen at position 0."""
        # Need > at n-1 and < at 1
        if length < 3:
            return "><" if length == 2 else ">"
        cells = ["."] * length
        cells[-1] = ">"  # Will wrap to 0
        cells[1] = "<"   # Will move to 0
        return "".join(cells)

    def _collision_at_end(self, length: int) -> str:
        """Create state where collision will happen at position n-1."""
        # Need > at n-2 and < at 0
        if length < 3:
            return "><" if length == 2 else ">"
        cells = ["."] * length
        cells[-2] = ">"  # Will move to n-1
        cells[0] = "<"   # Will wrap to n-1
        return "".join(cells)

    def _generate_wrap_collision(self, rng: random.Random, length: int) -> str:
        """Generate random wrap collision scenario."""
        cells = [Symbol.EMPTY.value] * length

        # Choose collision target: 0 or n-1
        if rng.random() < 0.5:
            # Collision at 0
            cells[-1] = Symbol.RIGHT.value  # Wraps to 0
            cells[1] = Symbol.LEFT.value    # Moves to 0
        else:
            # Collision at n-1
            cells[-2] = Symbol.RIGHT.value  # Moves to n-1
            cells[0] = Symbol.LEFT.value    # Wraps to n-1

        # Add some interior particles
        num_interior = rng.randint(0, length // 3)
        interior = [i for i in range(2, length - 2) if cells[i] == Symbol.EMPTY.value]
        for pos in rng.sample(interior, min(num_interior, len(interior))):
            cells[pos] = rng.choice([Symbol.RIGHT.value, Symbol.LEFT.value])

        return "".join(cells)

    def _generate_boundary_stress(self, rng: random.Random, length: int) -> str:
        """Generate general boundary stress state."""
        cells = [Symbol.EMPTY.value] * length

        # Place particles near both boundaries
        boundary_zone = max(2, length // 4)

        for i in range(boundary_zone):
            if rng.random() < 0.5:
                cells[i] = rng.choice([Symbol.RIGHT.value, Symbol.LEFT.value])
            if rng.random() < 0.5:
                cells[-(i + 1)] = rng.choice([Symbol.RIGHT.value, Symbol.LEFT.value])

        return "".join(cells)

    def _has_boundary_collision(self, state: str) -> bool:
        """Check if state will have collision at boundary (0 or n-1)."""
        n = len(state)
        if n < 2:
            return False

        # Check position 0: need right-mover at n-1 AND left-mover at 1
        right_sources = {">", "X"}
        left_sources = {"<", "X"}

        collision_at_0 = (state[-1] in right_sources and state[1] in left_sources)
        collision_at_end = (state[-2] in right_sources and state[0] in left_sources)

        return collision_at_0 or collision_at_end


class UniversalStressGenerator(Generator):
    """Generate universal stress states that every law must pass.

    These are MANDATORY test cases that expose common AI blind spots:

    1. ><><>< (max_collision): Maximum collision / occupancy limits
       - Tests laws that can't handle dense collision scenarios

    2. >>.<<  (multiplicity): 4 particles merging into 1 cell
       - Tests laws that assume 1v1 collisions

    3. XX     (x_battery): Chain collisions without free movers
       - Tests laws that assume "no movers = no action"
       - X cells emit particles that collide with each other

    4. ....>  (wrap_right): Particle at N-1 wrapping to 0
       - Tests laws that ignore periodic boundaries

    5. <....  (wrap_left): Particle at 0 wrapping to N-1
       - Tests laws that assume grid has "ends"

    6. <....> (boundary_collision): Collision across the boundary
       - Tests wrap-around collision handling

    Every law submitted to the harness should be tested against these
    states BEFORE random/constrained testing begins.
    """

    # The canonical stress states (at minimum size)
    STRESS_PATTERNS = [
        ("max_collision", "><><><"),       # Dense collisions
        ("multiplicity", ">>.<<"),         # 4-to-1 convergence
        ("x_battery", "XX"),               # X-chain reaction
        ("x_battery_3", "XXX"),            # Longer X chain
        ("wrap_right", "....>"),           # Right-mover at end
        ("wrap_left", "<...."),            # Left-mover at start
        ("boundary_collision", "<....>"),  # Collision across boundary
        ("all_collision", "XXXXXX"),       # Pure collision state
        ("alternating", "><><><><><"),     # Dense alternating
        ("mass_convergence", ">>>...<<<"), # Many particles converging
    ]

    def family_name(self) -> str:
        return "universal_stress"

    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate universal stress test cases.

        Args:
            params: {
                "grid_lengths": list[int] - sizes to test at
                "include_scaled": bool - include scaled versions of patterns
            }
            seed: Random seed (for any randomization needed)
            count: Number of cases (will generate at least len(STRESS_PATTERNS))

        Returns:
            List of mandatory stress test cases
        """
        grid_lengths = params.get("grid_lengths", [6, 8, 10, 12, 16])
        include_scaled = params.get("include_scaled", True)

        cases = []
        params_hash = self.params_hash(params)

        # Always include canonical stress patterns at their natural sizes
        for i, (name, pattern) in enumerate(self.STRESS_PATTERNS):
            natural_length = len(pattern)

            cases.append(Case(
                initial_state=pattern,
                config=Config(grid_length=natural_length),
                seed=seed + i,
                generator_family=self.family_name(),
                params_hash=params_hash,
                metadata={
                    "stress_type": name,
                    "is_canonical": True,
                },
            ))

        # Optionally add scaled versions at different grid sizes
        if include_scaled:
            scale_idx = len(self.STRESS_PATTERNS)
            for length in grid_lengths:
                for name, pattern in self.STRESS_PATTERNS:
                    scaled = self._scale_pattern(pattern, name, length)
                    if scaled != pattern:  # Only add if different
                        cases.append(Case(
                            initial_state=scaled,
                            config=Config(grid_length=length),
                            seed=seed + scale_idx,
                            generator_family=self.family_name(),
                            params_hash=params_hash,
                            metadata={
                                "stress_type": name,
                                "is_canonical": False,
                                "scaled_from": pattern,
                            },
                        ))
                        scale_idx += 1

        return cases[:count] if count < len(cases) else cases

    def _scale_pattern(self, pattern: str, name: str, target_length: int) -> str:
        """Scale a stress pattern to a target length.

        Maintains the essential stress characteristics while fitting
        the pattern to the target grid size.
        """
        if target_length <= len(pattern):
            return pattern[:target_length]

        extra = target_length - len(pattern)

        if name in ("wrap_right", "wrap_left"):
            # Add padding dots
            if name == "wrap_right":
                return "." * extra + pattern
            else:
                return pattern[0] + "." * (target_length - 1)

        elif name == "boundary_collision":
            # <....> scaled: < + dots + >
            return "<" + "." * (target_length - 2) + ">"

        elif name == "max_collision" or name == "alternating":
            # Repeat the pattern
            repeats = target_length // len(pattern)
            remainder = target_length % len(pattern)
            return (pattern * repeats + pattern[:remainder])[:target_length]

        elif name == "multiplicity":
            # >>..<< scaled: add more particles on each side
            half = target_length // 2
            rights = ">" * half
            lefts = "<" * (target_length - half)
            return rights + lefts

        elif name in ("x_battery", "x_battery_3", "all_collision"):
            # Center the X's
            num_x = len([c for c in pattern if c == "X"])
            padding = target_length - num_x
            left_pad = padding // 2
            right_pad = padding - left_pad
            return "." * left_pad + "X" * num_x + "." * right_pad

        elif name == "mass_convergence":
            # Scale the convergence pattern
            third = target_length // 3
            rights = ">" * third
            dots = "." * (target_length - 2 * third)
            lefts = "<" * third
            return rights + dots + lefts

        else:
            # Default: center with padding
            padding = target_length - len(pattern)
            left_pad = padding // 2
            right_pad = padding - left_pad
            return "." * left_pad + pattern + "." * right_pad
