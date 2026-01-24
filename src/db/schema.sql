-- Popper Explainer Database Schema
-- Version: 6 (PHASE-F: Orchestration engine, predictions, held-out sets)

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Laws table: stores candidate laws
CREATE TABLE IF NOT EXISTS laws (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    law_id TEXT NOT NULL UNIQUE,
    law_hash TEXT NOT NULL,
    template TEXT NOT NULL,
    law_json TEXT NOT NULL,  -- Full CandidateLaw JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_laws_hash ON laws(law_hash);
CREATE INDEX IF NOT EXISTS idx_laws_template ON laws(template);

-- Case sets: cached for reproducibility
CREATE TABLE IF NOT EXISTS case_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generator_family TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    seed INTEGER NOT NULL,
    cases_json TEXT NOT NULL,  -- Serialized Case[]
    case_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(generator_family, params_hash, seed)
);

CREATE INDEX IF NOT EXISTS idx_case_sets_lookup
    ON case_sets(generator_family, params_hash, seed);

-- Evaluations: verdict history with full audit trail
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    law_id TEXT NOT NULL,
    law_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('PASS', 'FAIL', 'UNKNOWN')),
    reason_code TEXT,  -- NULL for PASS/FAIL without special reason
    case_set_id INTEGER REFERENCES case_sets(id),
    cases_attempted INTEGER NOT NULL,
    cases_used INTEGER NOT NULL,
    power_metrics_json TEXT,
    vacuity_json TEXT,
    harness_config_hash TEXT NOT NULL,
    sim_hash TEXT NOT NULL,
    runtime_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (law_id) REFERENCES laws(law_id)
);

CREATE INDEX IF NOT EXISTS idx_evaluations_law ON evaluations(law_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_status ON evaluations(status);
CREATE INDEX IF NOT EXISTS idx_evaluations_created ON evaluations(created_at);

-- Counterexamples: minimal reproduction packages
CREATE TABLE IF NOT EXISTS counterexamples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL,
    law_id TEXT NOT NULL,
    initial_state TEXT NOT NULL,
    config_json TEXT NOT NULL,
    seed INTEGER,
    t_max INTEGER NOT NULL,
    t_fail INTEGER NOT NULL,
    trajectory_excerpt_json TEXT,
    observables_at_fail_json TEXT,
    witness_json TEXT,
    minimized BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id),
    FOREIGN KEY (law_id) REFERENCES laws(law_id)
);

CREATE INDEX IF NOT EXISTS idx_counterexamples_law ON counterexamples(law_id);
CREATE INDEX IF NOT EXISTS idx_counterexamples_eval ON counterexamples(evaluation_id);

-- Theories: axiom/theorem/explanation structures
CREATE TABLE IF NOT EXISTS theories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    theory_id TEXT NOT NULL UNIQUE,
    theory_json TEXT NOT NULL,  -- Full Theory object
    axiom_count INTEGER NOT NULL,
    theorem_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit logs: detailed operation history
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,  -- 'evaluate', 'propose', 'verify', etc.
    entity_type TEXT NOT NULL,  -- 'law', 'theory', 'evaluation', etc.
    entity_id TEXT,
    details_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_operation ON audit_logs(operation);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_logs(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at);

-- Escalation runs: one record per escalation event
CREATE TABLE IF NOT EXISTS escalation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL,
    harness_config_hash TEXT NOT NULL,
    sim_hash TEXT NOT NULL,
    seed INTEGER NOT NULL,
    laws_tested INTEGER NOT NULL,
    stable_count INTEGER NOT NULL,
    revoked_count INTEGER NOT NULL,
    downgraded_count INTEGER NOT NULL,
    runtime_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_escalation_runs_level ON escalation_runs(level);
CREATE INDEX IF NOT EXISTS idx_escalation_runs_created ON escalation_runs(created_at);

-- Law retests: one per law per escalation run
CREATE TABLE IF NOT EXISTS law_retests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    escalation_run_id INTEGER NOT NULL,
    law_id TEXT NOT NULL,
    old_status TEXT NOT NULL,
    new_status TEXT NOT NULL,
    flip_type TEXT NOT NULL,
    evaluation_id INTEGER,
    counterexample_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (escalation_run_id) REFERENCES escalation_runs(id),
    FOREIGN KEY (law_id) REFERENCES laws(law_id),
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id),
    FOREIGN KEY (counterexample_id) REFERENCES counterexamples(id)
);

CREATE INDEX IF NOT EXISTS idx_law_retests_run ON law_retests(escalation_run_id);
CREATE INDEX IF NOT EXISTS idx_law_retests_law ON law_retests(law_id);
CREATE INDEX IF NOT EXISTS idx_law_retests_flip ON law_retests(flip_type);

-- Failure classifications: Type A/B/C/D classification for convergence detection
CREATE TABLE IF NOT EXISTS failure_classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL UNIQUE,
    law_id TEXT NOT NULL,
    failure_class TEXT NOT NULL CHECK(failure_class IN (
        'type_a_known_counterexample',
        'type_b_novel_counterexample',
        'type_c_process_issue',
        'type_d_harness_error'
    )),
    counterexample_class_id TEXT,  -- e.g., 'converging_pair', 'x_emission'
    is_known_class BOOLEAN NOT NULL DEFAULT FALSE,
    confidence REAL,  -- Classification confidence 0.0-1.0
    features_json TEXT,  -- Extracted structural features
    reasoning TEXT,  -- Explanation of classification
    actionable BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id),
    FOREIGN KEY (law_id) REFERENCES laws(law_id)
);

CREATE INDEX IF NOT EXISTS idx_failure_class_type ON failure_classifications(failure_class);
CREATE INDEX IF NOT EXISTS idx_failure_class_ce_class ON failure_classifications(counterexample_class_id);
CREATE INDEX IF NOT EXISTS idx_failure_class_law ON failure_classifications(law_id);
CREATE INDEX IF NOT EXISTS idx_failure_class_actionable ON failure_classifications(actionable);

-- Counterexample class registry: tracks known structural patterns
CREATE TABLE IF NOT EXISTS counterexample_classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id TEXT NOT NULL UNIQUE,
    description TEXT,
    example_state TEXT,  -- Representative initial state
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ce_class_id ON counterexample_classes(class_id);
CREATE INDEX IF NOT EXISTS idx_ce_class_count ON counterexample_classes(occurrence_count DESC);

-- Novelty tracking: per-law novelty classification
CREATE TABLE IF NOT EXISTS law_novelty (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    law_id TEXT NOT NULL UNIQUE,
    syntactic_fingerprint TEXT NOT NULL,
    semantic_signature_hash TEXT,
    is_syntactically_novel BOOLEAN NOT NULL,
    is_semantically_novel BOOLEAN NOT NULL,
    is_novel BOOLEAN NOT NULL,  -- novel by either measure
    is_fully_novel BOOLEAN NOT NULL,  -- novel by both measures
    behavior_summary_json TEXT,  -- semantic evaluation details
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (law_id) REFERENCES laws(law_id)
);

CREATE INDEX IF NOT EXISTS idx_law_novelty_syntactic ON law_novelty(syntactic_fingerprint);
CREATE INDEX IF NOT EXISTS idx_law_novelty_semantic ON law_novelty(semantic_signature_hash);
CREATE INDEX IF NOT EXISTS idx_law_novelty_novel ON law_novelty(is_novel);

-- Novelty snapshots: periodic snapshots of novelty statistics
CREATE TABLE IF NOT EXISTS novelty_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_size INTEGER NOT NULL,
    total_laws_in_window INTEGER NOT NULL,
    syntactically_novel_count INTEGER NOT NULL,
    semantically_novel_count INTEGER NOT NULL,
    fully_novel_count INTEGER NOT NULL,
    syntactic_novelty_rate REAL NOT NULL,
    semantic_novelty_rate REAL NOT NULL,
    combined_novelty_rate REAL NOT NULL,
    is_saturated BOOLEAN NOT NULL,
    total_laws_seen INTEGER NOT NULL,
    unique_syntactic_fingerprints INTEGER NOT NULL,
    unique_semantic_signatures INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_novelty_snapshots_created ON novelty_snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_novelty_snapshots_saturated ON novelty_snapshots(is_saturated);

-- Failure keys: canonical keys for counterexample patterns
CREATE TABLE IF NOT EXISTS failure_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash TEXT NOT NULL UNIQUE,
    canonical_initial TEXT NOT NULL,
    canonical_fail_state TEXT,
    t_fail_relative INTEGER NOT NULL,  -- 0=immediate, 1=early, 2=mid, 3=late
    observable_signature_json TEXT,  -- JSON array of [name, value] pairs
    trajectory_signature TEXT,  -- Hash of canonical trajectory snippet
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_failure_keys_hash ON failure_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_failure_keys_initial ON failure_keys(canonical_initial);
CREATE INDEX IF NOT EXISTS idx_failure_keys_count ON failure_keys(occurrence_count DESC);

-- Counterexample failure key links: maps counterexamples to their failure keys
CREATE TABLE IF NOT EXISTS counterexample_failure_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    counterexample_id INTEGER NOT NULL,
    failure_key_id INTEGER NOT NULL,
    law_id TEXT NOT NULL,
    is_novel BOOLEAN NOT NULL,  -- Was this key novel when first seen
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (counterexample_id) REFERENCES counterexamples(id),
    FOREIGN KEY (failure_key_id) REFERENCES failure_keys(id),
    FOREIGN KEY (law_id) REFERENCES laws(law_id)
);

CREATE INDEX IF NOT EXISTS idx_cex_fk_counterexample ON counterexample_failure_keys(counterexample_id);
CREATE INDEX IF NOT EXISTS idx_cex_fk_failure_key ON counterexample_failure_keys(failure_key_id);
CREATE INDEX IF NOT EXISTS idx_cex_fk_law ON counterexample_failure_keys(law_id);

-- Failure key snapshots: periodic snapshots of failure key statistics
CREATE TABLE IF NOT EXISTS failure_key_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_size INTEGER NOT NULL,
    total_falsifications INTEGER NOT NULL,
    unique_failure_keys INTEGER NOT NULL,
    new_cex_rate REAL NOT NULL,
    repetition_rate REAL NOT NULL,
    total_keys_seen INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fk_snapshots_created ON failure_key_snapshots(created_at);

-- Theorem generation artifacts: reproducibility tracking (PHASE-D)
CREATE TABLE IF NOT EXISTS theorem_generation_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_hash TEXT UNIQUE NOT NULL,
    snapshot_hash TEXT NOT NULL,
    prompt_template_version TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_params_json TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    parsed_response_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON theorem_generation_artifacts(artifact_hash);
CREATE INDEX IF NOT EXISTS idx_artifacts_snapshot ON theorem_generation_artifacts(snapshot_hash);
CREATE INDEX IF NOT EXISTS idx_artifacts_created ON theorem_generation_artifacts(created_at);

-- Theorem runs: batch tracking for theorem generation
CREATE TABLE IF NOT EXISTS theorem_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'aborted')),
    config_json TEXT NOT NULL,
    prompt_hash TEXT,
    pass_laws_count INTEGER NOT NULL,
    fail_laws_count INTEGER NOT NULL,
    theorems_generated INTEGER DEFAULT 0,
    clusters_found INTEGER DEFAULT 0,
    observable_proposals INTEGER DEFAULT 0,
    artifact_id INTEGER,  -- PHASE-D: link to theorem_generation_artifacts
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (artifact_id) REFERENCES theorem_generation_artifacts(id)
);

CREATE INDEX IF NOT EXISTS idx_theorem_runs_status ON theorem_runs(status);
CREATE INDEX IF NOT EXISTS idx_theorem_runs_started ON theorem_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_theorem_runs_artifact ON theorem_runs(artifact_id);

-- Theorems: individual synthesized theorems
CREATE TABLE IF NOT EXISTS theorems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    theorem_run_id INTEGER NOT NULL,
    theorem_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('Established', 'Conditional', 'Conjectural')),
    claim TEXT NOT NULL,
    support_json TEXT NOT NULL,
    failure_modes_json TEXT,
    missing_structure_json TEXT,
    typed_missing_structure_json TEXT,  -- PHASE-D: Typed missing structure
    failure_signature_text TEXT,
    failure_signature_hash TEXT,
    role_coded_signature TEXT,  -- PHASE-D: Role-coded signature
    bucket_tags_json TEXT,  -- PHASE-C: Multi-label bucket assignment
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (theorem_run_id) REFERENCES theorem_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_theorems_run ON theorems(theorem_run_id);
CREATE INDEX IF NOT EXISTS idx_theorems_status ON theorems(status);
CREATE INDEX IF NOT EXISTS idx_theorems_signature_hash ON theorems(failure_signature_hash);
CREATE INDEX IF NOT EXISTS idx_theorems_role_coded_sig ON theorems(role_coded_signature);

-- Failure clusters: grouped failure signatures
CREATE TABLE IF NOT EXISTS failure_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    theorem_run_id INTEGER NOT NULL,
    cluster_id TEXT NOT NULL UNIQUE,
    bucket TEXT NOT NULL,  -- Primary bucket (backwards compatibility)
    bucket_tags_json TEXT,  -- PHASE-C: Multi-label bucket assignment
    semantic_cluster_idx INTEGER NOT NULL,
    theorem_ids_json TEXT NOT NULL,
    cluster_size INTEGER NOT NULL,
    centroid_signature TEXT,
    avg_similarity REAL,
    top_keywords_json TEXT,  -- PHASE-C: TF-IDF top terms
    recommended_action TEXT,  -- PHASE-C: SCHEMA_FIX, OBSERVABLE, GATING
    distance_threshold REAL,  -- PHASE-C: Clustering threshold used
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (theorem_run_id) REFERENCES theorem_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_failure_clusters_run ON failure_clusters(theorem_run_id);
CREATE INDEX IF NOT EXISTS idx_failure_clusters_bucket ON failure_clusters(bucket);
CREATE INDEX IF NOT EXISTS idx_failure_clusters_action ON failure_clusters(recommended_action);

-- Observable proposals: suggested new observables from clusters
CREATE TABLE IF NOT EXISTS observable_proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    theorem_run_id INTEGER NOT NULL,
    cluster_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL UNIQUE,
    observable_name TEXT NOT NULL,
    observable_expr TEXT NOT NULL,
    rationale TEXT NOT NULL,
    priority TEXT NOT NULL CHECK(priority IN ('high', 'medium', 'low')),
    action_type TEXT,  -- PHASE-C: SCHEMA_FIX, OBSERVABLE, GATING
    status TEXT NOT NULL DEFAULT 'proposed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (theorem_run_id) REFERENCES theorem_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_observable_proposals_run ON observable_proposals(theorem_run_id);
CREATE INDEX IF NOT EXISTS idx_observable_proposals_cluster ON observable_proposals(cluster_id);
CREATE INDEX IF NOT EXISTS idx_observable_proposals_priority ON observable_proposals(priority);
CREATE INDEX IF NOT EXISTS idx_observable_proposals_status ON observable_proposals(status);
CREATE INDEX IF NOT EXISTS idx_observable_proposals_action ON observable_proposals(action_type);

-- =============================================================================
-- PHASE-E: Deterministic clustering, witness capture, signature versioning
-- =============================================================================

-- Cluster artifacts: reproducibility tracking for clustering runs
CREATE TABLE IF NOT EXISTS cluster_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_hash TEXT UNIQUE NOT NULL,
    theorem_run_id INTEGER,
    snapshot_hash TEXT NOT NULL,
    signature_version TEXT NOT NULL,
    method TEXT NOT NULL,
    params_json TEXT NOT NULL,
    assignments_json TEXT NOT NULL,
    cluster_summaries_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (theorem_run_id) REFERENCES theorem_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_cluster_artifacts_hash ON cluster_artifacts(artifact_hash);
CREATE INDEX IF NOT EXISTS idx_cluster_artifacts_snapshot ON cluster_artifacts(snapshot_hash);
CREATE INDEX IF NOT EXISTS idx_cluster_artifacts_run ON cluster_artifacts(theorem_run_id);

-- Law witnesses: structured witnesses for FAIL verdicts
CREATE TABLE IF NOT EXISTS law_witnesses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    law_id TEXT NOT NULL,
    evaluation_id INTEGER NOT NULL,
    t_fail INTEGER NOT NULL,
    formatted_witness TEXT NOT NULL,
    state_at_t TEXT NOT NULL,
    state_at_t1 TEXT,
    observables_at_t_json TEXT,
    observables_at_t1_json TEXT,
    neighborhood_hash TEXT NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (law_id) REFERENCES laws(law_id),
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
);

CREATE INDEX IF NOT EXISTS idx_law_witnesses_law ON law_witnesses(law_id);
CREATE INDEX IF NOT EXISTS idx_law_witnesses_eval ON law_witnesses(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_law_witnesses_neighborhood ON law_witnesses(neighborhood_hash);
CREATE INDEX IF NOT EXISTS idx_law_witnesses_primary ON law_witnesses(is_primary);

-- =============================================================================
-- PHASE-F: Orchestration engine tables
-- =============================================================================

-- Orchestration runs: top-level container for a full scientific loop
CREATE TABLE IF NOT EXISTS orchestration_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    current_phase TEXT NOT NULL DEFAULT 'discovery',
    config_json TEXT NOT NULL,
    universe_id TEXT,
    sim_hash TEXT,
    harness_hash TEXT,
    discovery_model_id TEXT,
    tester_model_id TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    total_iterations INTEGER DEFAULT 0,
    CONSTRAINT valid_orch_status CHECK (status IN ('running', 'completed', 'aborted')),
    CONSTRAINT valid_orch_phase CHECK (current_phase IN ('discovery', 'theorem', 'explanation', 'prediction', 'finalize'))
);

CREATE INDEX IF NOT EXISTS idx_orch_runs_status ON orchestration_runs(status);
CREATE INDEX IF NOT EXISTS idx_orch_runs_phase ON orchestration_runs(current_phase);

-- Orchestration iterations: each loop step within a run
CREATE TABLE IF NOT EXISTS orchestration_iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration_index INTEGER NOT NULL,
    phase TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    prompt_hash TEXT,
    control_block_json TEXT,
    readiness_metrics_json TEXT,
    summary_json TEXT,
    FOREIGN KEY (run_id) REFERENCES orchestration_runs(run_id),
    UNIQUE (run_id, iteration_index),
    CONSTRAINT valid_iter_status CHECK (status IN ('running', 'completed', 'aborted'))
);

CREATE INDEX IF NOT EXISTS idx_orch_iter_run ON orchestration_iterations(run_id);
CREATE INDEX IF NOT EXISTS idx_orch_iter_phase ON orchestration_iterations(run_id, phase);
CREATE INDEX IF NOT EXISTS idx_orch_iter_status ON orchestration_iterations(status);

-- Phase transitions: audit trail of phase changes
CREATE TABLE IF NOT EXISTS phase_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration_id INTEGER NOT NULL,
    from_phase TEXT NOT NULL,
    to_phase TEXT NOT NULL,
    trigger TEXT NOT NULL,
    readiness_score REAL,
    evidence_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES orchestration_runs(run_id),
    FOREIGN KEY (iteration_id) REFERENCES orchestration_iterations(id)
);

CREATE INDEX IF NOT EXISTS idx_phase_trans_run ON phase_transitions(run_id);
CREATE INDEX IF NOT EXISTS idx_phase_trans_trigger ON phase_transitions(trigger);

-- Readiness snapshots: periodic metrics capture for phase transitions
CREATE TABLE IF NOT EXISTS readiness_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration_id INTEGER NOT NULL,
    phase TEXT NOT NULL,
    s_pass REAL,
    s_stability REAL,
    s_novel_cex REAL,
    s_harness_health REAL,
    s_redundancy REAL,
    s_coverage REAL,
    s_prediction_accuracy REAL,
    s_adversarial_accuracy REAL,
    s_held_out_accuracy REAL,
    combined_score REAL,
    weights_json TEXT,
    source_counts_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES orchestration_runs(run_id),
    FOREIGN KEY (iteration_id) REFERENCES orchestration_iterations(id)
);

CREATE INDEX IF NOT EXISTS idx_readiness_run ON readiness_snapshots(run_id);
CREATE INDEX IF NOT EXISTS idx_readiness_phase ON readiness_snapshots(phase);

-- Explanations: mechanistic hypotheses from explanation phase
CREATE TABLE IF NOT EXISTS explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration_id INTEGER,
    explanation_id TEXT UNIQUE NOT NULL,
    hypothesis_text TEXT NOT NULL,
    mechanism_json TEXT,
    supporting_theorem_ids_json TEXT,
    open_questions_json TEXT,
    criticisms_json TEXT,
    confidence REAL,
    status TEXT DEFAULT 'proposed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES orchestration_runs(run_id),
    CONSTRAINT valid_expl_status CHECK (status IN ('proposed', 'validated', 'refuted'))
);

CREATE INDEX IF NOT EXISTS idx_expl_run ON explanations(run_id);
CREATE INDEX IF NOT EXISTS idx_expl_status ON explanations(status);

-- Predictions: generated by explanation phase for verification
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration_id INTEGER,
    prediction_id TEXT UNIQUE NOT NULL,
    explanation_id TEXT,
    initial_state TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    predicted_state TEXT NOT NULL,
    predicted_observables_json TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES orchestration_runs(run_id),
    FOREIGN KEY (explanation_id) REFERENCES explanations(explanation_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_pred_explanation ON predictions(explanation_id);

-- Prediction evaluations: verification results against simulator
CREATE TABLE IF NOT EXISTS prediction_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    actual_state TEXT NOT NULL,
    is_exact_match INTEGER NOT NULL,
    hamming_distance INTEGER,
    cell_accuracy REAL,
    observable_errors_json TEXT,
    evaluation_set TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
    FOREIGN KEY (run_id) REFERENCES orchestration_runs(run_id),
    UNIQUE (prediction_id, evaluation_set),
    CONSTRAINT valid_eval_set CHECK (evaluation_set IN ('held_out', 'adversarial', 'regression'))
);

CREATE INDEX IF NOT EXISTS idx_pred_eval_run ON prediction_evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_pred_eval_set ON prediction_evaluations(run_id, evaluation_set);
CREATE INDEX IF NOT EXISTS idx_pred_eval_match ON prediction_evaluations(is_exact_match);

-- Held-out test sets: locked evaluation sets for non-gameable testing
CREATE TABLE IF NOT EXISTS held_out_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    set_type TEXT NOT NULL,
    generation_seed INTEGER NOT NULL,
    cases_json TEXT NOT NULL,
    case_count INTEGER NOT NULL,
    locked INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES orchestration_runs(run_id),
    UNIQUE (run_id, set_type),
    CONSTRAINT valid_set_type CHECK (set_type IN ('random', 'adversarial', 'regression'))
);

CREATE INDEX IF NOT EXISTS idx_held_out_run ON held_out_sets(run_id);
CREATE INDEX IF NOT EXISTS idx_held_out_type ON held_out_sets(set_type);
