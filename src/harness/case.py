"""Case contract for test harness.

A Case represents a single experiment configuration that can be
run through the simulator and checked against a law.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from src.universe.types import Config, State, Trajectory


@dataclass
class Case:
    """A test case for law evaluation.

    Attributes:
        initial_state: Starting state for simulation
        config: Universe configuration
        seed: Random seed used to generate this case
        generator_family: Name of the generator that created this case
        params_hash: Hash of generator parameters
        metadata: Additional metadata about case generation
    """

    initial_state: State
    config: Config
    seed: int
    generator_family: str
    params_hash: str = ""
    generator_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Compute a hash of the case content for deduplication."""
        content = {
            "initial_state": self.initial_state,
            "grid_length": self.config.grid_length,
            "boundary": self.config.boundary,
            "seed": self.seed,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "initial_state": self.initial_state,
            "config": {
                "grid_length": self.config.grid_length,
                "boundary": self.config.boundary,
            },
            "seed": self.seed,
            "generator_family": self.generator_family,
            "params_hash": self.params_hash,
            "generator_params": self.generator_params,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Case":
        """Create from dictionary."""
        return cls(
            initial_state=data["initial_state"],
            config=Config(
                grid_length=data["config"]["grid_length"],
                boundary=data["config"].get("boundary", "periodic"),
            ),
            seed=data["seed"],
            generator_family=data["generator_family"],
            params_hash=data.get("params_hash", ""),
            generator_params=data.get("generator_params", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CaseResult:
    """Result of running a single case.

    Attributes:
        case: The test case that was run
        trajectory: The simulated trajectory
        passed: Whether the law held for this case
        violation: Details if the law was violated
        precondition_met: Whether preconditions were satisfied
    """

    case: Case
    trajectory: Trajectory
    passed: bool
    violation: dict[str, Any] | None = None
    precondition_met: bool = True
    near_miss_score: float = 0.0  # 0-1, higher = closer to violation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "case": self.case.to_dict(),
            "trajectory_length": len(self.trajectory),
            "passed": self.passed,
            "violation": self.violation,
            "precondition_met": self.precondition_met,
            "near_miss_score": self.near_miss_score,
        }
