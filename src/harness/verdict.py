"""Law verdict data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.harness.power import PowerMetrics
from src.claims.vacuity import VacuityReport


class ReasonCode(str, Enum):
    """Reason codes for UNKNOWN verdicts."""

    MISSING_OBSERVABLE = "missing_observable"
    MISSING_TRANSFORM = "missing_transform"
    MISSING_GENERATOR = "missing_generator"
    AMBIGUOUS_CLAIM = "ambiguous_claim"
    UNMET_PRECONDITIONS = "unmet_preconditions"
    RESOURCE_LIMIT = "resource_limit"
    INCONCLUSIVE_LOW_POWER = "inconclusive_low_power"
    VACUOUS_PASS = "vacuous_pass"


@dataclass
class Counterexample:
    """A minimal counterexample for a failed law.

    Attributes:
        initial_state: Starting state that leads to violation
        config: Universe configuration
        seed: Random seed (if applicable)
        t_max: Total time simulated
        t_fail: Time step where violation occurred
        trajectory_excerpt: States around the failure
        observables_at_fail: Observable values at failure time
        witness: Additional witness data
        minimized: Whether this has been minimized
    """

    initial_state: str
    config: dict[str, Any]
    seed: int | None
    t_max: int
    t_fail: int
    trajectory_excerpt: list[str] | None = None
    observables_at_fail: dict[str, int] | None = None
    witness: dict[str, Any] | None = None
    minimized: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "initial_state": self.initial_state,
            "config": self.config,
            "seed": self.seed,
            "t_max": self.t_max,
            "t_fail": self.t_fail,
            "trajectory_excerpt": self.trajectory_excerpt,
            "observables_at_fail": self.observables_at_fail,
            "witness": self.witness,
            "minimized": self.minimized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Counterexample":
        """Create from dictionary."""
        return cls(
            initial_state=data["initial_state"],
            config=data["config"],
            seed=data.get("seed"),
            t_max=data["t_max"],
            t_fail=data["t_fail"],
            trajectory_excerpt=data.get("trajectory_excerpt"),
            observables_at_fail=data.get("observables_at_fail"),
            witness=data.get("witness"),
            minimized=data.get("minimized", False),
        )


@dataclass
class LawVerdict:
    """Complete verdict for a law evaluation.

    Attributes:
        law_id: Identifier of the evaluated law
        status: PASS, FAIL, or UNKNOWN
        reason_code: Reason for UNKNOWN (or additional info)
        counterexample: Minimal counterexample if FAIL
        power_metrics: Testing thoroughness metrics
        vacuity: Vacuity report for implication laws
        runtime_ms: Total evaluation time
        tests_run: List of test families executed
        notes: Additional notes or warnings
    """

    law_id: str
    status: str  # "PASS", "FAIL", "UNKNOWN"
    reason_code: ReasonCode | None = None
    counterexample: Counterexample | None = None
    power_metrics: PowerMetrics = field(default_factory=PowerMetrics)
    vacuity: VacuityReport = field(default_factory=VacuityReport)
    runtime_ms: int = 0
    tests_run: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "law_id": self.law_id,
            "status": self.status,
            "reason_code": self.reason_code.value if self.reason_code else None,
            "counterexample": self.counterexample.to_dict() if self.counterexample else None,
            "power_metrics": self.power_metrics.to_dict(),
            "vacuity": self.vacuity.to_dict(),
            "runtime_ms": self.runtime_ms,
            "tests_run": self.tests_run,
            "notes": self.notes,
        }
