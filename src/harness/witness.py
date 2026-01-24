"""Witness capture for counterexamples (PHASE-E).

This module provides structured witness capture for FAIL verdicts,
turning raw counterexamples into human-readable formatted witnesses.

A formatted witness captures:
- The violation in human-readable form: "LHS=10, RHS=5, violated by (actual > expected)"
- States at t and t+1 for context
- Observable values at failure time
- Neighborhood hash for diversity tracking
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from src.claims.schema import CandidateLaw


@dataclass
class FormattedWitness:
    """A human-readable witness for a law violation.

    This captures all the information needed to understand why a law failed
    and provides enough context to debug or reproduce the issue.
    """

    law_id: str
    t: int  # Time step where violation occurred
    lhs_expr: str  # Left-hand side expression
    lhs_value: Any  # Evaluated LHS value
    rhs_expr: str  # Right-hand side expression
    rhs_value: Any  # Evaluated RHS value
    violation_description: str  # Human-readable description of the violation
    state_at_t: str  # Universe state at t
    state_at_t1: str | None = None  # Universe state at t+1 (if relevant)
    observables_at_t: dict[str, Any] = field(default_factory=dict)
    observables_at_t1: dict[str, Any] | None = None
    neighborhood_hash: str = ""  # Hash of local state around violation

    def __str__(self) -> str:
        """Return human-readable formatted witness."""
        return (
            f"t={self.t}: {self.lhs_expr}={self.lhs_value}, "
            f"{self.rhs_expr}={self.rhs_value}, "
            f"violated by ({self.violation_description})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "law_id": self.law_id,
            "t": self.t,
            "lhs_expr": self.lhs_expr,
            "lhs_value": self.lhs_value,
            "rhs_expr": self.rhs_expr,
            "rhs_value": self.rhs_value,
            "violation_description": self.violation_description,
            "state_at_t": self.state_at_t,
            "state_at_t1": self.state_at_t1,
            "observables_at_t": self.observables_at_t,
            "observables_at_t1": self.observables_at_t1,
            "neighborhood_hash": self.neighborhood_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FormattedWitness":
        return cls(
            law_id=data["law_id"],
            t=data["t"],
            lhs_expr=data["lhs_expr"],
            lhs_value=data["lhs_value"],
            rhs_expr=data["rhs_expr"],
            rhs_value=data["rhs_value"],
            violation_description=data["violation_description"],
            state_at_t=data["state_at_t"],
            state_at_t1=data.get("state_at_t1"),
            observables_at_t=data.get("observables_at_t", {}),
            observables_at_t1=data.get("observables_at_t1"),
            neighborhood_hash=data.get("neighborhood_hash", ""),
        )


def compute_neighborhood_hash(state: str, position: int | None, radius: int = 3) -> str:
    """Compute a hash of the local neighborhood around a position.

    This is used for witness diversity tracking - witnesses with different
    neighborhood hashes represent structurally different violations.

    Args:
        state: The universe state string
        position: Center position (or None for global hash)
        radius: Radius of neighborhood to include

    Returns:
        SHA256 hash prefix of the neighborhood
    """
    if position is None or position < 0:
        # Global hash for position-independent violations
        content = state
    else:
        # Extract neighborhood around position
        start = max(0, position - radius)
        end = min(len(state), position + radius + 1)
        content = state[start:end]

    return hashlib.sha256(content.encode()).hexdigest()[:16]


def extract_violation_info(
    law: CandidateLaw,
    violation: dict[str, Any] | None,
) -> tuple[str, str, str]:
    """Extract LHS/RHS expressions and violation description from law and violation.

    Args:
        law: The candidate law that was violated
        violation: The violation details from evaluation

    Returns:
        Tuple of (lhs_expr, rhs_expr, violation_description)
    """
    # Default expressions based on template
    template = law.template.value if hasattr(law.template, "value") else str(law.template)

    if template == "invariant":
        lhs_expr = law.claim if hasattr(law, "claim") else "observable"
        rhs_expr = "constant"
        violation_desc = "value changed"
    elif template == "monotone":
        lhs_expr = "obs(t)"
        rhs_expr = "obs(t+1)"
        violation_desc = "monotonicity violated"
    elif template in ("implication_step", "implication_state", "local_transition"):
        lhs_expr = "antecedent"
        rhs_expr = "consequent"
        violation_desc = "implication failed (P true, Q false)"
    elif template == "eventually":
        lhs_expr = "condition"
        rhs_expr = "goal"
        violation_desc = "goal not reached within horizon"
    elif template == "bound":
        lhs_expr = "observable"
        rhs_expr = "bound"
        violation_desc = "bound exceeded"
    elif template == "symmetry_commutation":
        lhs_expr = "transform(step(S))"
        rhs_expr = "step(transform(S))"
        violation_desc = "symmetry broken"
    else:
        lhs_expr = "LHS"
        rhs_expr = "RHS"
        violation_desc = "violation"

    # Override with violation details if available
    if violation:
        if "details" in violation:
            details = violation["details"]
            if isinstance(details, dict):
                if "lhs" in details:
                    lhs_expr = str(details.get("lhs_expr", lhs_expr))
                if "rhs" in details:
                    rhs_expr = str(details.get("rhs_expr", rhs_expr))
                if "reason" in details:
                    violation_desc = details["reason"]

    return lhs_expr, rhs_expr, violation_desc


def build_formatted_witness(
    law: CandidateLaw,
    trajectory: list[str],
    t_fail: int,
    violation: dict[str, Any] | None,
    observables_at_t: dict[str, Any] | None = None,
    observables_at_t1: dict[str, Any] | None = None,
    position: int | None = None,
) -> FormattedWitness:
    """Build a formatted witness from evaluation data.

    Args:
        law: The candidate law that was violated
        trajectory: Full trajectory of states
        t_fail: Time step where violation occurred
        violation: Violation details from evaluation
        observables_at_t: Observable values at t_fail
        observables_at_t1: Observable values at t_fail+1
        position: Position in state where violation occurred (for neighborhood hash)

    Returns:
        FormattedWitness capturing the violation
    """
    # Get states
    state_at_t = trajectory[t_fail] if t_fail < len(trajectory) else ""
    state_at_t1 = trajectory[t_fail + 1] if t_fail + 1 < len(trajectory) else None

    # Extract expression info
    lhs_expr, rhs_expr, violation_desc = extract_violation_info(law, violation)

    # Get values from violation details
    lhs_value = None
    rhs_value = None
    if violation and "details" in violation:
        details = violation["details"]
        if isinstance(details, dict):
            lhs_value = details.get("lhs", details.get("actual"))
            rhs_value = details.get("rhs", details.get("expected"))

    # Compute neighborhood hash
    neighborhood_hash = compute_neighborhood_hash(state_at_t, position)

    return FormattedWitness(
        law_id=law.law_id,
        t=t_fail,
        lhs_expr=lhs_expr,
        lhs_value=lhs_value,
        rhs_expr=rhs_expr,
        rhs_value=rhs_value,
        violation_description=violation_desc,
        state_at_t=state_at_t,
        state_at_t1=state_at_t1,
        observables_at_t=observables_at_t or {},
        observables_at_t1=observables_at_t1,
        neighborhood_hash=neighborhood_hash,
    )


class WitnessCapture:
    """Captures witnesses with diversity tracking.

    This class manages witness collection, ensuring we keep diverse witnesses
    (different neighborhood hashes) rather than many similar ones.
    """

    def __init__(self, max_witnesses_per_law: int = 20):
        self.max_witnesses_per_law = max_witnesses_per_law
        self._witnesses: dict[str, list[FormattedWitness]] = {}
        self._seen_hashes: dict[str, set[str]] = {}

    def add_witness(self, witness: FormattedWitness) -> bool:
        """Add a witness if it's novel or we haven't reached capacity.

        Args:
            witness: The witness to add

        Returns:
            True if witness was added, False if it was a duplicate or at capacity
        """
        law_id = witness.law_id

        # Initialize tracking for this law
        if law_id not in self._witnesses:
            self._witnesses[law_id] = []
            self._seen_hashes[law_id] = set()

        # Check capacity
        if len(self._witnesses[law_id]) >= self.max_witnesses_per_law:
            return False

        # Check novelty (different neighborhood hash)
        if witness.neighborhood_hash in self._seen_hashes[law_id]:
            return False

        # Add witness
        self._witnesses[law_id].append(witness)
        self._seen_hashes[law_id].add(witness.neighborhood_hash)
        return True

    def get_witnesses(self, law_id: str) -> list[FormattedWitness]:
        """Get all captured witnesses for a law."""
        return self._witnesses.get(law_id, [])

    def get_primary_witness(self, law_id: str) -> FormattedWitness | None:
        """Get the first (primary) witness for a law."""
        witnesses = self._witnesses.get(law_id, [])
        return witnesses[0] if witnesses else None

    def get_diversity_count(self, law_id: str) -> int:
        """Get the number of unique neighborhood hashes for a law."""
        return len(self._seen_hashes.get(law_id, set()))

    def clear(self, law_id: str | None = None) -> None:
        """Clear captured witnesses.

        Args:
            law_id: Specific law to clear, or None to clear all
        """
        if law_id is None:
            self._witnesses.clear()
            self._seen_hashes.clear()
        else:
            self._witnesses.pop(law_id, None)
            self._seen_hashes.pop(law_id, None)
