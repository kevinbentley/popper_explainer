"""Vacuity tracking for implication and eventually laws.

A test is vacuous if the antecedent (P) is never true, meaning
the consequent (Q) is never actually tested.
"""

from dataclasses import dataclass


@dataclass
class VacuityReport:
    """Report on test vacuity for implication laws.

    Attributes:
        antecedent_true_count: Number of times P was true
        consequent_evaluated_count: Number of times Q was evaluated
        total_checks: Total number of time steps checked
        is_vacuous: True if antecedent was never true
    """

    antecedent_true_count: int = 0
    consequent_evaluated_count: int = 0
    total_checks: int = 0
    is_vacuous: bool = False

    def merge(self, other: "VacuityReport") -> "VacuityReport":
        """Merge two vacuity reports (e.g., from multiple cases)."""
        return VacuityReport(
            antecedent_true_count=self.antecedent_true_count + other.antecedent_true_count,
            consequent_evaluated_count=self.consequent_evaluated_count + other.consequent_evaluated_count,
            total_checks=self.total_checks + other.total_checks,
            is_vacuous=self.is_vacuous and other.is_vacuous,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "antecedent_true_count": self.antecedent_true_count,
            "consequent_evaluated_count": self.consequent_evaluated_count,
            "total_checks": self.total_checks,
            "is_vacuous": self.is_vacuous,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VacuityReport":
        """Create from dictionary."""
        return cls(
            antecedent_true_count=data.get("antecedent_true_count", 0),
            consequent_evaluated_count=data.get("consequent_evaluated_count", 0),
            total_checks=data.get("total_checks", 0),
            is_vacuous=data.get("is_vacuous", False),
        )
