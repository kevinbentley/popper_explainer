"""Vacuity tracking for implication and eventually laws.

A test is vacuous if the antecedent (P) is never true, meaning
the consequent (Q) is never actually tested.
"""

from dataclasses import dataclass


@dataclass
class VacuityReport:
    """Report on test vacuity for implication laws.

    Attributes:
        antecedent_true_count: Number of times P was true (trigger_count)
        consequent_evaluated_count: Number of times Q was evaluated
        total_checks: Total number of time steps checked
        is_vacuous: True if antecedent was never true
        trigger_diversity: Number of distinct sources that produced triggers
                          (distinct initial_state hashes or generator families)
        triggering_generators: Set of generator families that produced triggers
        triggering_states: Set of initial state hashes that produced triggers
    """

    antecedent_true_count: int = 0
    consequent_evaluated_count: int = 0
    total_checks: int = 0
    is_vacuous: bool = False
    trigger_diversity: int = 0
    triggering_generators: set[str] = None  # type: ignore
    triggering_states: set[str] = None  # type: ignore

    def __post_init__(self):
        if self.triggering_generators is None:
            self.triggering_generators = set()
        if self.triggering_states is None:
            self.triggering_states = set()

    def merge(self, other: "VacuityReport") -> "VacuityReport":
        """Merge two vacuity reports (e.g., from multiple cases)."""
        merged_generators = self.triggering_generators | other.triggering_generators
        merged_states = self.triggering_states | other.triggering_states
        return VacuityReport(
            antecedent_true_count=self.antecedent_true_count + other.antecedent_true_count,
            consequent_evaluated_count=self.consequent_evaluated_count + other.consequent_evaluated_count,
            total_checks=self.total_checks + other.total_checks,
            is_vacuous=self.is_vacuous and other.is_vacuous,
            trigger_diversity=len(merged_generators),
            triggering_generators=merged_generators,
            triggering_states=merged_states,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "antecedent_true_count": self.antecedent_true_count,
            "consequent_evaluated_count": self.consequent_evaluated_count,
            "total_checks": self.total_checks,
            "is_vacuous": self.is_vacuous,
            "trigger_diversity": self.trigger_diversity,
            "triggering_generators": list(self.triggering_generators) if self.triggering_generators else [],
            "triggering_states_count": len(self.triggering_states) if self.triggering_states else 0,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VacuityReport":
        """Create from dictionary."""
        return cls(
            antecedent_true_count=data.get("antecedent_true_count", 0),
            consequent_evaluated_count=data.get("consequent_evaluated_count", 0),
            total_checks=data.get("total_checks", 0),
            is_vacuous=data.get("is_vacuous", False),
            trigger_diversity=data.get("trigger_diversity", 0),
            triggering_generators=set(data.get("triggering_generators", [])),
            triggering_states=set(),  # Not stored in full, just count
        )
