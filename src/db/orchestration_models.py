"""Database models for orchestration engine (PHASE-F).

These models represent records in the orchestration-related tables:
- orchestration_runs: Top-level run state
- orchestration_iterations: Each iteration within a run
- phase_transitions: Audit trail of phase changes
- readiness_snapshots: Metrics snapshots
- explanations: Mechanistic hypotheses
- predictions: Generated predictions
- prediction_evaluations: Verification results
- held_out_sets: Locked test sets
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class OrchestrationRunRecord:
    """A top-level orchestration run record."""

    run_id: str
    status: str  # 'running', 'completed', 'aborted'
    current_phase: str  # 'discovery', 'theorem', 'explanation', 'prediction', 'finalize'
    config_json: str
    universe_id: str | None = None
    sim_hash: str | None = None
    harness_hash: str | None = None
    discovery_model_id: str | None = None
    tester_model_id: str | None = None
    total_iterations: int = 0
    id: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class OrchestrationIterationRecord:
    """A single iteration within an orchestration run."""

    run_id: str
    iteration_index: int
    phase: str
    status: str = "running"  # 'running', 'completed', 'aborted'
    prompt_hash: str | None = None
    control_block_json: str | None = None
    readiness_metrics_json: str | None = None
    summary_json: str | None = None
    id: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class PhaseTransitionRecord:
    """A phase transition audit record."""

    run_id: str
    iteration_id: int
    from_phase: str
    to_phase: str
    trigger: str  # 'readiness_threshold', 'llm_recommendation', 'plateau_escape', 'manual'
    readiness_score: float | None = None
    evidence_json: str | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class ReadinessSnapshotRecord:
    """A periodic readiness metrics snapshot."""

    run_id: str
    iteration_id: int
    phase: str
    s_pass: float | None = None
    s_stability: float | None = None
    s_novel_cex: float | None = None
    s_harness_health: float | None = None
    s_redundancy: float | None = None
    s_coverage: float | None = None
    s_prediction_accuracy: float | None = None
    s_adversarial_accuracy: float | None = None
    s_held_out_accuracy: float | None = None
    combined_score: float | None = None
    weights_json: str | None = None
    source_counts_json: str | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class ExplanationRecord:
    """A mechanistic explanation record."""

    run_id: str
    explanation_id: str
    hypothesis_text: str
    iteration_id: int | None = None
    mechanism_json: str | None = None
    supporting_theorem_ids_json: str | None = None
    open_questions_json: str | None = None
    criticisms_json: str | None = None
    confidence: float | None = None
    status: str = "proposed"  # 'proposed', 'validated', 'refuted'
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class PredictionRecord:
    """A prediction generated from an explanation."""

    run_id: str
    prediction_id: str
    initial_state: str
    horizon: int
    predicted_state: str
    iteration_id: int | None = None
    explanation_id: str | None = None
    predicted_observables_json: str | None = None
    confidence: float | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class PredictionEvaluationRecord:
    """A prediction evaluation result."""

    prediction_id: str
    run_id: str
    actual_state: str
    is_exact_match: bool
    evaluation_set: str  # 'held_out', 'adversarial', 'regression'
    hamming_distance: int | None = None
    cell_accuracy: float | None = None
    observable_errors_json: str | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class HeldOutSetRecord:
    """A held-out test set record."""

    run_id: str
    set_type: str  # 'random', 'adversarial', 'regression'
    generation_seed: int
    cases_json: str
    case_count: int
    locked: bool = False
    id: int | None = None
    created_at: datetime | None = None
