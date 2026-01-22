"""Database models for the Popper Explainer persistence layer."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LawRecord:
    """A persisted candidate law."""

    law_id: str
    law_hash: str
    template: str
    law_json: str
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class CaseSetRecord:
    """A cached set of test cases for reproducibility."""

    generator_family: str
    params_hash: str
    seed: int
    cases_json: str
    case_count: int
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class EvaluationRecord:
    """A law evaluation result with full audit trail."""

    law_id: str
    law_hash: str
    status: str  # 'PASS', 'FAIL', 'UNKNOWN'
    harness_config_hash: str
    sim_hash: str
    cases_attempted: int
    cases_used: int
    reason_code: str | None = None
    case_set_id: int | None = None
    power_metrics_json: str | None = None
    vacuity_json: str | None = None
    runtime_ms: int | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class CounterexampleRecord:
    """A minimal counterexample for a failed law."""

    evaluation_id: int
    law_id: str
    initial_state: str
    config_json: str
    t_max: int
    t_fail: int
    seed: int | None = None
    trajectory_excerpt_json: str | None = None
    observables_at_fail_json: str | None = None
    witness_json: str | None = None
    minimized: bool = False
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class TheoryRecord:
    """A theory structure with axioms, theorems, and explanations."""

    theory_id: str
    theory_json: str
    axiom_count: int
    theorem_count: int
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class AuditLogRecord:
    """An audit log entry for tracking operations."""

    operation: str
    entity_type: str
    entity_id: str | None = None
    details_json: str | None = None
    id: int | None = None
    created_at: datetime | None = None
