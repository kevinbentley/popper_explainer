"""Data models for escalation tracking."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class EscalationRunRecord:
    """Record of a single escalation run.

    Attributes:
        level: Escalation level name (e.g., 'escalation_1')
        harness_config_hash: Hash of the harness configuration used
        sim_hash: Hash of the simulator version
        seed: Random seed used for the run
        laws_tested: Number of laws tested
        stable_count: Laws that remained PASS
        revoked_count: Laws that flipped to FAIL
        downgraded_count: Laws that flipped to UNKNOWN
        runtime_ms: Total runtime in milliseconds
        id: Database ID (None until inserted)
        created_at: Timestamp of creation
    """

    level: str
    harness_config_hash: str
    sim_hash: str
    seed: int
    laws_tested: int
    stable_count: int
    revoked_count: int
    downgraded_count: int
    runtime_ms: int
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class LawRetestRecord:
    """Record of a single law retest within an escalation run.

    Attributes:
        escalation_run_id: FK to escalation_runs table
        law_id: ID of the law being retested
        old_status: Status before retest (always 'PASS' for escalation)
        new_status: Status after retest ('PASS', 'FAIL', 'UNKNOWN')
        flip_type: Classification ('stable', 'revoked', 'downgraded')
        evaluation_id: FK to new evaluation record (if created)
        counterexample_id: FK to counterexample (if FAIL)
        id: Database ID (None until inserted)
        created_at: Timestamp of creation
    """

    escalation_run_id: int
    law_id: str
    old_status: str
    new_status: str
    flip_type: str
    evaluation_id: int | None = None
    counterexample_id: int | None = None
    id: int | None = None
    created_at: datetime | None = None
