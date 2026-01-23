"""Flip classification types for escalation retests.

A "flip" occurs when a law's verdict changes during escalation testing.
"""

from dataclasses import dataclass
from enum import Enum

from src.harness.verdict import Counterexample, LawVerdict


class FlipType(str, Enum):
    """Classification of verdict changes during escalation.

    Attributes:
        STABLE: Law survived escalation (PASS -> PASS)
        REVOKED: Found counterexample at higher power (PASS -> FAIL)
        DOWNGRADED: Inconclusive at higher power (PASS -> UNKNOWN)
    """

    STABLE = "stable"
    REVOKED = "revoked"
    DOWNGRADED = "downgraded"


@dataclass
class RetestResult:
    """Result of re-testing a single law during escalation.

    Attributes:
        law_id: ID of the law that was retested
        old_status: Status before retest (always "PASS" for escalation)
        new_status: Status after retest ("PASS", "FAIL", or "UNKNOWN")
        flip_type: Classification of the verdict change
        new_verdict: The full verdict from the escalated evaluation
        counterexample: Counterexample if law was revoked (FAIL)
    """

    law_id: str
    old_status: str
    new_status: str
    flip_type: FlipType
    new_verdict: LawVerdict
    counterexample: Counterexample | None = None


def classify_flip(old_status: str, new_status: str) -> FlipType:
    """Classify the type of verdict change.

    Args:
        old_status: Status before escalation (should be "PASS")
        new_status: Status after escalation ("PASS", "FAIL", "UNKNOWN")

    Returns:
        FlipType classification

    Raises:
        ValueError: If old_status is not "PASS" or transition is invalid
    """
    if old_status != "PASS":
        raise ValueError(f"Escalation expects old_status='PASS', got '{old_status}'")

    if new_status == "PASS":
        return FlipType.STABLE
    elif new_status == "FAIL":
        return FlipType.REVOKED
    elif new_status == "UNKNOWN":
        return FlipType.DOWNGRADED
    else:
        raise ValueError(f"Unknown new_status: '{new_status}'")
