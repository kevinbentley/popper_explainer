"""Power metrics for test evaluation.

Power metrics help distinguish a genuine PASS from "we didn't look hard enough."
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PowerMetrics:
    """Metrics indicating how thorough the testing was.

    Attributes:
        cases_attempted: Total cases generated
        cases_used: Cases that satisfied preconditions
        cases_with_collisions: Cases where collisions occurred
        cases_with_wrapping: Cases where boundary wrapping occurred
        density_bins_hit: Particle density ranges tested
        violation_score_max: Highest "near-violation" score in adversarial search
        metamorphic_cases_run: Number of metamorphic (symmetry) tests
        metamorphic_mismatches: Number of symmetry violations found
        vacuous_cases: Cases where antecedent was never true
        coverage_score: Overall coverage estimate (0-1)
    """

    cases_attempted: int = 0
    cases_used: int = 0
    cases_with_collisions: int = 0
    cases_with_wrapping: int = 0
    density_bins_hit: list[float] = field(default_factory=list)
    violation_score_max: float = 0.0
    metamorphic_cases_run: int = 0
    metamorphic_mismatches: int = 0
    vacuous_cases: int = 0
    coverage_score: float = 0.0
    adversarial_cases_tried: int = 0
    adversarial_found: bool = False

    def compute_coverage(self) -> float:
        """Compute an overall coverage score."""
        if self.cases_attempted == 0:
            return 0.0

        # Factors contributing to coverage
        usage_ratio = self.cases_used / max(self.cases_attempted, 1)
        collision_ratio = self.cases_with_collisions / max(self.cases_used, 1)
        wrapping_ratio = self.cases_with_wrapping / max(self.cases_used, 1)
        density_coverage = min(len(self.density_bins_hit) / 5.0, 1.0)  # 5 bins = full coverage
        non_vacuous_ratio = 1.0 - (self.vacuous_cases / max(self.cases_used, 1))

        # Weighted combination
        self.coverage_score = (
            0.3 * usage_ratio +
            0.2 * collision_ratio +
            0.1 * wrapping_ratio +
            0.2 * density_coverage +
            0.2 * non_vacuous_ratio
        )
        return self.coverage_score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cases_attempted": self.cases_attempted,
            "cases_used": self.cases_used,
            "cases_with_collisions": self.cases_with_collisions,
            "cases_with_wrapping": self.cases_with_wrapping,
            "density_bins_hit": self.density_bins_hit,
            "violation_score_max": self.violation_score_max,
            "metamorphic_cases_run": self.metamorphic_cases_run,
            "metamorphic_mismatches": self.metamorphic_mismatches,
            "vacuous_cases": self.vacuous_cases,
            "coverage_score": self.coverage_score,
            "adversarial_cases_tried": self.adversarial_cases_tried,
            "adversarial_found": self.adversarial_found,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PowerMetrics":
        """Create from dictionary."""
        return cls(
            cases_attempted=data.get("cases_attempted", 0),
            cases_used=data.get("cases_used", 0),
            cases_with_collisions=data.get("cases_with_collisions", 0),
            cases_with_wrapping=data.get("cases_with_wrapping", 0),
            density_bins_hit=data.get("density_bins_hit", []),
            violation_score_max=data.get("violation_score_max", 0.0),
            metamorphic_cases_run=data.get("metamorphic_cases_run", 0),
            metamorphic_mismatches=data.get("metamorphic_mismatches", 0),
            vacuous_cases=data.get("vacuous_cases", 0),
            coverage_score=data.get("coverage_score", 0.0),
            adversarial_cases_tried=data.get("adversarial_cases_tried", 0),
            adversarial_found=data.get("adversarial_found", False),
        )
