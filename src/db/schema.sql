-- Popper Explainer Database Schema
-- Version: 1

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
