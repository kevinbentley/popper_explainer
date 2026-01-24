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


@dataclass
class FailureClassificationRecord:
    """A failure classification for convergence detection.

    Types:
        A: Known counterexample class (expected failure)
        B: Novel counterexample class (learning)
        C: Process/harness issue (invalid law, missing observable)
        D: Harness error (show-stopper)
    """

    evaluation_id: int
    law_id: str
    failure_class: str  # 'type_a_known_counterexample', etc.
    is_known_class: bool
    actionable: bool
    counterexample_class_id: str | None = None
    confidence: float | None = None
    features_json: str | None = None
    reasoning: str | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class CounterexampleClassRecord:
    """A known counterexample class in the registry."""

    class_id: str
    description: str | None = None
    example_state: str | None = None
    occurrence_count: int = 0
    id: int | None = None
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None


@dataclass
class LawNoveltyRecord:
    """Novelty classification for a candidate law."""

    law_id: str
    syntactic_fingerprint: str
    is_syntactically_novel: bool
    is_semantically_novel: bool
    is_novel: bool
    is_fully_novel: bool
    semantic_signature_hash: str | None = None
    behavior_summary_json: str | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class NoveltySnapshotRecord:
    """Periodic snapshot of novelty statistics."""

    window_size: int
    total_laws_in_window: int
    syntactically_novel_count: int
    semantically_novel_count: int
    fully_novel_count: int
    syntactic_novelty_rate: float
    semantic_novelty_rate: float
    combined_novelty_rate: float
    is_saturated: bool
    total_laws_seen: int
    unique_syntactic_fingerprints: int
    unique_semantic_signatures: int
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class FailureKeyRecord:
    """A canonical failure key for counterexample patterns."""

    key_hash: str
    canonical_initial: str
    t_fail_relative: int
    canonical_fail_state: str | None = None
    observable_signature_json: str | None = None
    trajectory_signature: str | None = None
    occurrence_count: int = 0
    id: int | None = None
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None


@dataclass
class CounterexampleFailureKeyRecord:
    """Links a counterexample to its failure key."""

    counterexample_id: int
    failure_key_id: int
    law_id: str
    is_novel: bool
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class FailureKeySnapshotRecord:
    """Periodic snapshot of failure key statistics."""

    window_size: int
    total_falsifications: int
    unique_failure_keys: int
    new_cex_rate: float
    repetition_rate: float
    total_keys_seen: int
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class TheoremGenerationArtifactRecord:
    """A theorem generation artifact for reproducibility (PHASE-D)."""

    artifact_hash: str
    snapshot_hash: str
    prompt_template_version: str
    model_name: str
    model_params_json: str
    raw_response: str
    parsed_response_json: str
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class TheoremRunRecord:
    """A theorem generation run record."""

    run_id: str
    status: str  # 'running', 'completed', 'aborted'
    config_json: str
    pass_laws_count: int
    fail_laws_count: int
    theorems_generated: int = 0
    clusters_found: int = 0
    observable_proposals: int = 0
    prompt_hash: str | None = None
    artifact_id: int | None = None  # PHASE-D: link to artifact
    id: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class TheoremRecord:
    """A synthesized theorem record."""

    theorem_run_id: int
    theorem_id: str
    name: str
    status: str  # 'Established', 'Conditional', 'Conjectural'
    claim: str
    support_json: str
    failure_modes_json: str | None = None
    missing_structure_json: str | None = None
    typed_missing_structure_json: str | None = None  # PHASE-D: Typed missing structure
    failure_signature_text: str | None = None
    failure_signature_hash: str | None = None
    role_coded_signature: str | None = None  # PHASE-D: Role-coded signature
    bucket_tags_json: str | None = None  # PHASE-C: Multi-label bucket assignment
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class FailureClusterRecord:
    """A failure cluster record."""

    theorem_run_id: int
    cluster_id: str
    bucket: str  # Primary bucket (backwards compatibility)
    semantic_cluster_idx: int
    theorem_ids_json: str
    cluster_size: int
    centroid_signature: str | None = None
    avg_similarity: float | None = None
    bucket_tags_json: str | None = None  # PHASE-C: Multi-label bucket assignment
    top_keywords_json: str | None = None  # PHASE-C: TF-IDF top terms
    recommended_action: str | None = None  # PHASE-C: SCHEMA_FIX, OBSERVABLE, GATING
    distance_threshold: float | None = None  # PHASE-C: Clustering threshold used
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class ObservableProposalRecord:
    """An observable proposal record."""

    theorem_run_id: int
    cluster_id: str
    proposal_id: str
    observable_name: str
    observable_expr: str
    rationale: str
    priority: str  # 'high', 'medium', 'low'
    action_type: str | None = None  # PHASE-C: SCHEMA_FIX, OBSERVABLE, GATING
    status: str = "proposed"
    id: int | None = None
    created_at: datetime | None = None


# =============================================================================
# PHASE-E: Cluster artifacts and law witnesses
# =============================================================================


@dataclass
class ClusterArtifactRecord:
    """A cluster artifact for reproducibility tracking (PHASE-E)."""

    artifact_hash: str
    snapshot_hash: str
    signature_version: str
    method: str
    params_json: str
    assignments_json: str
    cluster_summaries_json: str
    theorem_run_id: int | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class LawWitnessRecord:
    """A structured witness for a FAIL verdict (PHASE-E)."""

    law_id: str
    evaluation_id: int
    t_fail: int
    formatted_witness: str
    state_at_t: str
    neighborhood_hash: str
    state_at_t1: str | None = None
    observables_at_t_json: str | None = None
    observables_at_t1_json: str | None = None
    is_primary: bool = False
    id: int | None = None
    created_at: datetime | None = None


# =============================================================================
# Web Viewer: LLM Transcripts
# =============================================================================


@dataclass
class LLMTranscriptRecord:
    """A record of an LLM interaction for debugging and audit.

    Attributes:
        component: Which component made the call (law_proposer, theorem_generator, etc.)
        model_name: LLM model used
        prompt: The prompt text sent to the LLM
        raw_response: The raw response from the LLM
        success: Whether the call succeeded
        run_id: Optional orchestration run ID
        iteration_id: Optional iteration index
        phase: Optional phase name
        system_instruction: Optional system instruction
        prompt_hash: Hash of the prompt for deduplication
        prompt_tokens: Estimated prompt tokens
        output_tokens: Estimated output tokens
        thinking_tokens: Thinking tokens (for extended thinking models)
        total_tokens: Total tokens used
        duration_ms: Call duration in milliseconds
        error_message: Error message if call failed
    """

    component: str
    model_name: str
    prompt: str
    raw_response: str
    success: bool = True
    run_id: str | None = None
    iteration_id: int | None = None
    phase: str | None = None
    system_instruction: str | None = None
    prompt_hash: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int = 0
    total_tokens: int | None = None
    duration_ms: int | None = None
    error_message: str | None = None
    id: int | None = None
    created_at: datetime | None = None
