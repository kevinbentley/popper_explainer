"""Tool for requesting sample states and trajectories."""

import json
import logging
import random
from typing import Any

from src.ahc.tools.base import BaseTool, ToolParameter, ToolResult
from src.harness.generators import GeneratorRegistry
from src.universe.simulator import run

logger = logging.getLogger(__name__)


class RequestSamplesTool(BaseTool):
    """Tool for requesting sample states and trajectories.

    Provides access to the generator system to request specific
    types of test cases for exploration and learning.
    """

    @property
    def name(self) -> str:
        return "request_samples"

    @property
    def description(self) -> str:
        return """Request sample states or trajectories for exploration.

Use this to get examples of specific patterns or configurations:
- Random states with controlled density
- States with collisions
- States designed to test specific behaviors
- Trajectories showing how states evolve

Available patterns:
- "random": Random states with specified density
- "collision": States that will have collisions
- "approaching": States with particles approaching each other
- "empty": Mostly empty states
- "dense": High-density states
- "pathological": Edge cases and unusual configurations
- "symmetric": States with mirror symmetry

Returns states and optionally their trajectories."""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="pattern",
                type="string",
                description="Type of samples to request",
                required=True,
                enum=["random", "collision", "approaching", "empty", "dense", "pathological", "symmetric"],
            ),
            ToolParameter(
                name="count",
                type="integer",
                description="Number of samples to generate (default: 5, max: 50)",
                required=False,
                default=5,
            ),
            ToolParameter(
                name="length",
                type="integer",
                description="Desired grid length (default: 10)",
                required=False,
                default=10,
            ),
            ToolParameter(
                name="include_trajectory",
                type="boolean",
                description="Whether to include trajectories for each sample (default: true)",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="trajectory_steps",
                type="integer",
                description="Number of steps in trajectory (default: 5)",
                required=False,
                default=5,
            ),
        ]

    def execute(
        self,
        pattern: str,
        count: int = 5,
        length: int = 10,
        include_trajectory: bool = True,
        trajectory_steps: int = 5,
        **kwargs,
    ) -> ToolResult:
        """Generate samples according to the pattern.

        Args:
            pattern: Type of samples to generate
            count: Number of samples
            length: Grid length
            include_trajectory: Whether to include trajectories
            trajectory_steps: Steps in trajectory

        Returns:
            ToolResult with generated samples
        """
        try:
            # Coerce types (Gemini may return floats)
            count = int(count)
            length = int(length)
            trajectory_steps = int(trajectory_steps)

            # Clamp parameters
            count = min(max(1, count), 50)
            length = min(max(2, length), 100)
            trajectory_steps = min(max(1, trajectory_steps), 20)

            # Generate samples based on pattern
            samples = self._generate_samples(pattern, count, length)

            # Add trajectories if requested
            if include_trajectory:
                for sample in samples:
                    trajectory = run(sample["state"], trajectory_steps)
                    sample["trajectory"] = trajectory

            return ToolResult.ok(
                data={
                    "pattern": pattern,
                    "count": len(samples),
                    "samples": samples,
                }
            )

        except Exception as e:
            logger.exception(f"Failed to generate samples: pattern={pattern}")
            return ToolResult.fail(f"Sample generation failed: {str(e)}")

    def _generate_samples(
        self,
        pattern: str,
        count: int,
        length: int,
    ) -> list[dict[str, Any]]:
        """Generate samples based on pattern type."""
        samples = []
        seed = random.randint(0, 100000)

        if pattern == "random":
            samples = self._generate_random(count, length, seed)
        elif pattern == "collision":
            samples = self._generate_collision(count, length, seed)
        elif pattern == "approaching":
            samples = self._generate_approaching(count, length, seed)
        elif pattern == "empty":
            samples = self._generate_empty(count, length, seed)
        elif pattern == "dense":
            samples = self._generate_dense(count, length, seed)
        elif pattern == "pathological":
            samples = self._generate_pathological(count, length, seed)
        elif pattern == "symmetric":
            samples = self._generate_symmetric(count, length, seed)
        else:
            # Fall back to random
            samples = self._generate_random(count, length, seed)

        return samples

    def _generate_random(self, count: int, length: int, seed: int) -> list[dict]:
        """Generate random states with varying density."""
        random.seed(seed)
        samples = []
        symbols = ['.', '>', '<']

        for i in range(count):
            density = random.uniform(0.1, 0.5)
            state = []
            for _ in range(length):
                if random.random() < density:
                    state.append(random.choice(['>', '<']))
                else:
                    state.append('.')
            samples.append({
                "state": "".join(state),
                "metadata": {"density": density, "seed": seed + i},
            })

        return samples

    def _generate_collision(self, count: int, length: int, seed: int) -> list[dict]:
        """Generate states that will have collisions."""
        random.seed(seed)
        samples = []

        for i in range(count):
            # Place at least one collision-causing pair: >.<
            state = ['.'] * length
            pos = random.randint(0, length - 3)
            state[pos] = '>'
            state[pos + 2] = '<'

            # Maybe add more pairs
            for _ in range(random.randint(0, 2)):
                p = random.randint(0, length - 3)
                if state[p] == '.' and state[p + 2] == '.':
                    state[p] = '>'
                    state[p + 2] = '<'

            samples.append({
                "state": "".join(state),
                "metadata": {"type": "collision", "seed": seed + i},
            })

        return samples

    def _generate_approaching(self, count: int, length: int, seed: int) -> list[dict]:
        """Generate states with particles approaching each other."""
        random.seed(seed)
        samples = []

        for i in range(count):
            state = ['.'] * length
            # Place > on left side, < on right side
            left_pos = random.randint(0, length // 3)
            right_pos = random.randint(2 * length // 3, length - 1)
            state[left_pos] = '>'
            state[right_pos] = '<'

            # Maybe add more
            for _ in range(random.randint(0, 2)):
                p = random.randint(0, length // 2)
                if state[p] == '.':
                    state[p] = '>'
                p = random.randint(length // 2, length - 1)
                if state[p] == '.':
                    state[p] = '<'

            samples.append({
                "state": "".join(state),
                "metadata": {"type": "approaching", "seed": seed + i},
            })

        return samples

    def _generate_empty(self, count: int, length: int, seed: int) -> list[dict]:
        """Generate mostly empty states."""
        random.seed(seed)
        samples = []

        for i in range(count):
            state = ['.'] * length
            # Add 1-2 particles
            num_particles = random.randint(1, 2)
            for _ in range(num_particles):
                pos = random.randint(0, length - 1)
                state[pos] = random.choice(['>', '<'])

            samples.append({
                "state": "".join(state),
                "metadata": {"type": "empty", "seed": seed + i},
            })

        return samples

    def _generate_dense(self, count: int, length: int, seed: int) -> list[dict]:
        """Generate high-density states."""
        random.seed(seed)
        samples = []

        for i in range(count):
            state = []
            for _ in range(length):
                if random.random() < 0.7:  # 70% occupied
                    state.append(random.choice(['>', '<']))
                else:
                    state.append('.')

            samples.append({
                "state": "".join(state),
                "metadata": {"type": "dense", "seed": seed + i},
            })

        return samples

    def _generate_pathological(self, count: int, length: int, seed: int) -> list[dict]:
        """Generate edge cases and unusual configurations."""
        canonical_cases = [
            ("." * length, "all_empty"),
            (">" * length, "all_right"),
            ("<" * length, "all_left"),
            ("><" * (length // 2), "alternating_rl"),
            ("<>" * (length // 2), "alternating_lr"),
            (">..<" + "." * (length - 4) if length >= 4 else ">..<", "single_collision"),
            ("X" + "." * (length - 1) if length >= 1 else "X", "single_X"),
            (">.<>.<" + "." * (length - 6) if length >= 6 else ">.<>.<", "double_collision"),
        ]

        samples = []
        for state, case_type in canonical_cases[:count]:
            # Adjust length if needed
            if len(state) < length:
                state = state + "." * (length - len(state))
            elif len(state) > length:
                state = state[:length]
            samples.append({
                "state": state,
                "metadata": {"type": "pathological", "case": case_type},
            })

        return samples

    def _generate_symmetric(self, count: int, length: int, seed: int) -> list[dict]:
        """Generate mirror-symmetric states."""
        random.seed(seed)
        samples = []

        for i in range(count):
            half_len = length // 2
            half = []
            for _ in range(half_len):
                half.append(random.choice(['.', '.', '>']))

            # Mirror with direction swap
            mirrored = []
            for c in reversed(half):
                if c == '>':
                    mirrored.append('<')
                elif c == '<':
                    mirrored.append('>')
                else:
                    mirrored.append(c)

            state = "".join(half) + ("." if length % 2 else "") + "".join(mirrored)

            samples.append({
                "state": state[:length],  # Ensure exact length
                "metadata": {"type": "symmetric", "seed": seed + i},
            })

        return samples
