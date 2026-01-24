"""Test set management for prediction verification.

Manages held-out and adversarial test sets that the LLM never sees
during iterative refinement, preventing overfitting.

Key principle: These test sets are LOCKED at creation time and
cannot be modified during the explanation refinement loop.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.universe.types import Config, State

if TYPE_CHECKING:
    from src.db.repo import Repository


@dataclass
class PredictionTestCase:
    """A single test case for prediction verification.

    Attributes:
        initial_state: Starting state at t=0
        horizon: Number of steps to predict (default 1)
        config: Universe configuration
        expected_state: Ground truth state at t=horizon (computed by simulator)
        case_id: Unique identifier for this case
        metadata: Additional tracking info
    """

    initial_state: State
    horizon: int
    config: Config
    expected_state: State | None = None
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.case_id is None:
            # Generate deterministic ID from content
            content = f"{self.initial_state}:{self.horizon}:{self.config.grid_length}"
            self.case_id = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class PredictionTestSet:
    """A collection of test cases with metadata.

    Attributes:
        set_id: Unique identifier
        set_type: 'held_out', 'adversarial', or 'regression'
        cases: List of test cases
        generation_seed: Seed used for reproducibility
        locked: Whether the set is frozen
        created_at: Creation timestamp
    """

    set_id: str
    set_type: str  # 'held_out', 'adversarial', 'regression'
    cases: list[PredictionTestCase]
    generation_seed: int
    locked: bool = False
    created_at: str | None = None

    @property
    def case_count(self) -> int:
        return len(self.cases)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "set_id": self.set_id,
            "set_type": self.set_type,
            "cases": [
                {
                    "initial_state": c.initial_state,
                    "horizon": c.horizon,
                    "config": {"grid_length": c.config.grid_length},
                    "expected_state": c.expected_state,
                    "case_id": c.case_id,
                    "metadata": c.metadata,
                }
                for c in self.cases
            ],
            "generation_seed": self.generation_seed,
            "locked": self.locked,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionTestSet:
        """Deserialize from dictionary."""
        cases = [
            PredictionTestCase(
                initial_state=c["initial_state"],
                horizon=c["horizon"],
                config=Config(grid_length=c["config"]["grid_length"]),
                expected_state=c.get("expected_state"),
                case_id=c.get("case_id"),
                metadata=c.get("metadata", {}),
            )
            for c in data.get("cases", [])
        ]
        return cls(
            set_id=data["set_id"],
            set_type=data["set_type"],
            cases=cases,
            generation_seed=data["generation_seed"],
            locked=data.get("locked", False),
            created_at=data.get("created_at"),
        )


class HeldOutSetManager:
    """Manages held-out test sets for non-gameable evaluation.

    The held-out set is generated ONCE at the start of a run and
    NEVER shown to the LLM during refinement. This prevents the
    explanation from overfitting to specific test cases.

    Test set types:
    - held_out: Random states the LLM never sees
    - adversarial: States designed to maximize prediction errors
    - regression: Fixed states for tracking progress over time
    """

    def __init__(
        self,
        repo: Repository | None = None,
        default_grid_lengths: list[int] | None = None,
        default_densities: list[float] | None = None,
    ):
        """Initialize the held-out set manager.

        Args:
            repo: Database repository for persistence
            default_grid_lengths: Default grid sizes to use
            default_densities: Default particle densities
        """
        self.repo = repo
        self.default_grid_lengths = default_grid_lengths or [8, 16, 32]
        self.default_densities = default_densities or [0.1, 0.3, 0.5]
        self._cached_sets: dict[str, PredictionTestSet] = {}

    def create_held_out_set(
        self,
        run_id: str,
        seed: int,
        count: int = 100,
        horizons: list[int] | None = None,
        grid_lengths: list[int] | None = None,
        densities: list[float] | None = None,
    ) -> PredictionTestSet:
        """Create a new held-out test set.

        The set is created, ground truth is computed via simulation,
        and the set is locked to prevent modification.

        Args:
            run_id: Orchestration run ID
            seed: Random seed for reproducibility
            count: Number of test cases to generate
            horizons: Prediction horizons (default [1, 2, 5])
            grid_lengths: Grid sizes to use
            densities: Particle densities to use

        Returns:
            Locked PredictionTestSet with ground truth states
        """
        from src.universe.simulator import run as sim_run
        from src.universe.types import Symbol

        rng = random.Random(seed)
        horizons = horizons or [1, 2, 5]
        grid_lengths = grid_lengths or self.default_grid_lengths
        densities = densities or self.default_densities

        cases: list[PredictionTestCase] = []

        for i in range(count):
            # Select parameters
            horizon = horizons[i % len(horizons)]
            grid_length = grid_lengths[i % len(grid_lengths)]
            density = densities[i % len(densities)]

            # Generate random initial state
            initial_state = self._generate_random_state(
                rng, grid_length, density
            )
            config = Config(grid_length=grid_length)

            # Compute ground truth via simulation
            trajectory = sim_run(initial_state, horizon, config)
            expected_state = trajectory[-1]

            case = PredictionTestCase(
                initial_state=initial_state,
                horizon=horizon,
                config=config,
                expected_state=expected_state,
                metadata={
                    "density": density,
                    "generation_index": i,
                },
            )
            cases.append(case)

        set_id = f"held_out_{run_id}_{seed}"
        test_set = PredictionTestSet(
            set_id=set_id,
            set_type="held_out",
            cases=cases,
            generation_seed=seed,
            locked=True,  # Immediately lock
        )

        # Persist to database
        if self.repo:
            self._persist_test_set(run_id, test_set)

        self._cached_sets[set_id] = test_set
        return test_set

    def create_regression_set(
        self,
        run_id: str,
        seed: int = 42,
    ) -> PredictionTestSet:
        """Create a fixed regression test set.

        This set uses carefully chosen states that cover important
        edge cases and remain constant across runs for comparison.

        Args:
            run_id: Orchestration run ID
            seed: Fixed seed for reproducibility

        Returns:
            Locked PredictionTestSet with fixed regression cases
        """
        from src.universe.simulator import run as sim_run

        # Hand-crafted regression cases covering edge conditions
        regression_states = [
            # Empty states
            ("........", 1),
            ("................", 1),
            # Single particles
            (">.......", 1),
            (".......<", 1),
            # Collisions
            ("><......", 1),  # Imminent collision
            ("X.......", 1),  # Existing collision resolving
            # Multiple collisions
            ("><><><><", 1),  # Dense collisions
            (">.<.>.<.", 2),  # Approaching pairs
            # Edge wrapping
            (">......<", 1),  # Will wrap and collide
            ("<......>", 1),  # Particles wrapping
            # Complex patterns
            (">><><.<<.>", 3),
            (">.>.>.>.<.<.<.<", 5),
        ]

        cases: list[PredictionTestCase] = []
        for state, horizon in regression_states:
            config = Config(grid_length=len(state))
            trajectory = sim_run(state, horizon, config)

            case = PredictionTestCase(
                initial_state=state,
                horizon=horizon,
                config=config,
                expected_state=trajectory[-1],
                metadata={"regression_case": True},
            )
            cases.append(case)

        set_id = f"regression_{run_id}"
        test_set = PredictionTestSet(
            set_id=set_id,
            set_type="regression",
            cases=cases,
            generation_seed=seed,
            locked=True,
        )

        if self.repo:
            self._persist_test_set(run_id, test_set)

        self._cached_sets[set_id] = test_set
        return test_set

    def get_test_set(self, set_id: str) -> PredictionTestSet | None:
        """Retrieve a test set by ID.

        Args:
            set_id: Test set identifier

        Returns:
            PredictionTestSet if found, None otherwise
        """
        if set_id in self._cached_sets:
            return self._cached_sets[set_id]

        if self.repo:
            # Try to load from database
            # TODO: Implement repo.get_held_out_set
            pass

        return None

    def get_sets_for_run(self, run_id: str) -> list[PredictionTestSet]:
        """Get all test sets for a run.

        Args:
            run_id: Orchestration run ID

        Returns:
            List of test sets
        """
        sets = []
        for set_id, test_set in self._cached_sets.items():
            if run_id in set_id:
                sets.append(test_set)
        return sets

    def _generate_random_state(
        self,
        rng: random.Random,
        grid_length: int,
        density: float,
    ) -> State:
        """Generate a random state at the given density."""
        from src.universe.types import Symbol

        num_particles = int(grid_length * density)
        cells = [Symbol.EMPTY.value] * grid_length

        positions = list(range(grid_length))
        rng.shuffle(positions)

        for i, pos in enumerate(positions[:num_particles]):
            if rng.random() < 0.5:
                cells[pos] = Symbol.RIGHT.value
            else:
                cells[pos] = Symbol.LEFT.value

        return "".join(cells)

    def _persist_test_set(self, run_id: str, test_set: PredictionTestSet) -> None:
        """Persist a test set to the database."""
        if not self.repo:
            return

        from src.db.orchestration_models import HeldOutSetRecord

        record = HeldOutSetRecord(
            run_id=run_id,
            set_type=test_set.set_type,
            generation_seed=test_set.generation_seed,
            cases_json=json.dumps(test_set.to_dict()["cases"]),
            case_count=test_set.case_count,
            locked=test_set.locked,
        )
        self.repo.insert_held_out_set(record)
