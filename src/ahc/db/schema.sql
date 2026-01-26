-- AHC-DS Database Schema
-- Separate from popper.db to avoid interference

-- Sessions: Top-level container for AHC runs
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',  -- running, paused, completed, failed
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    config_json TEXT,  -- Full run configuration
    model_id TEXT,  -- LLM model identifier
    seed INTEGER,

    -- Termination metrics
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy REAL DEFAULT 0.0,
    transition_rules_complete INTEGER DEFAULT 0,  -- Boolean: 0 or 1
    terminated_at TEXT,
    termination_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);


-- Journal entries: Chain-of-Thought records
CREATE TABLE IF NOT EXISTS journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    turn_number INTEGER NOT NULL,
    entry_type TEXT NOT NULL,  -- thought, observation, hypothesis, experiment, conclusion
    content TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_journal_session ON journal_entries(session_id);
CREATE INDEX IF NOT EXISTS idx_journal_turn ON journal_entries(session_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_journal_type ON journal_entries(session_id, entry_type);


-- Tool calls: Audit trail of all tool invocations
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    turn_number INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_turn ON tool_calls(session_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(session_id, tool_name);


-- Predictions: run_prediction results for accuracy tracking
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    turn_number INTEGER NOT NULL,
    state_t0 TEXT NOT NULL,  -- Initial state
    predicted_state_t1 TEXT NOT NULL,  -- Agent's prediction
    actual_state_t1 TEXT NOT NULL,  -- Simulator result
    is_correct INTEGER NOT NULL,  -- Boolean: 0 or 1
    prediction_method TEXT,  -- How agent made the prediction
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_correct ON predictions(session_id, is_correct);


-- Theorems: Validated laws stored by the agent
CREATE TABLE IF NOT EXISTS theorems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    name TEXT NOT NULL,
    description TEXT,
    law_ids_json TEXT NOT NULL,  -- JSON array of law IDs
    status TEXT NOT NULL DEFAULT 'proposed',  -- proposed, validated, refuted
    evidence_json TEXT,  -- Supporting evidence
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(session_id, name)
);

CREATE INDEX IF NOT EXISTS idx_theorems_session ON theorems(session_id);
CREATE INDEX IF NOT EXISTS idx_theorems_status ON theorems(session_id, status);


-- Law evaluations: evaluate_laws results with counterexamples
CREATE TABLE IF NOT EXISTS law_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    turn_number INTEGER NOT NULL,
    law_json TEXT NOT NULL,  -- Full CandidateLaw as JSON
    law_id TEXT NOT NULL,
    law_hash TEXT NOT NULL,
    status TEXT NOT NULL,  -- PASS, FAIL, UNKNOWN
    reason_code TEXT,
    counterexample_json TEXT,
    power_metrics_json TEXT,
    runtime_ms INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_law_evals_session ON law_evaluations(session_id);
CREATE INDEX IF NOT EXISTS idx_law_evals_status ON law_evaluations(session_id, status);
CREATE INDEX IF NOT EXISTS idx_law_evals_law_hash ON law_evaluations(law_hash);


-- Trajectory samples: Cached request_samples results
CREATE TABLE IF NOT EXISTS trajectory_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    pattern TEXT NOT NULL,  -- Pattern requested
    count_requested INTEGER NOT NULL,
    samples_json TEXT NOT NULL,  -- JSON array of samples
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_samples_session ON trajectory_samples(session_id);
CREATE INDEX IF NOT EXISTS idx_samples_pattern ON trajectory_samples(session_id, pattern);


-- Transition rules: Incremental symbolic transition function discovery
CREATE TABLE IF NOT EXISTS transition_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    symbol TEXT NOT NULL,  -- The symbol being transitioned (>, <, ., X)
    neighbor_config TEXT NOT NULL,  -- 3-char neighborhood pattern
    coordinate_class TEXT NOT NULL,  -- even, odd, or any
    result_symbol TEXT NOT NULL,  -- The resulting symbol at t+1
    confidence REAL DEFAULT 1.0,  -- Confidence level
    evidence_count INTEGER DEFAULT 0,  -- How many observations support this
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(session_id, symbol, neighbor_config, coordinate_class)
);

CREATE INDEX IF NOT EXISTS idx_transition_session ON transition_rules(session_id);
CREATE INDEX IF NOT EXISTS idx_transition_symbol ON transition_rules(session_id, symbol);


-- Conversation turns: Raw LLM conversation for replay/debugging
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL,  -- system, user, assistant
    content TEXT NOT NULL,
    tool_calls_json TEXT,  -- Tool calls in this turn (if any)
    token_count INTEGER DEFAULT 0,  -- Estimated tokens for context management
    created_at TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(session_id, turn_number, role)
);

CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_turn ON conversation_turns(session_id, turn_number);


-- Meta knowledge: Compaction snapshots for context management
-- Stores synthesized knowledge state after each compaction event
CREATE TABLE IF NOT EXISTS meta_knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    version INTEGER NOT NULL,  -- Increments with each compaction

    -- Knowledge state at compaction time
    theorems_snapshot_json TEXT,  -- Current validated theorems
    negative_knowledge TEXT,      -- Summarized falsifications and ruled-out concepts

    -- Compaction metadata
    last_compacted_turn INTEGER NOT NULL,  -- Turn number where compaction occurred
    turns_compacted INTEGER,               -- Number of turns summarized
    token_count_before INTEGER,            -- Tokens in context before compaction
    token_count_after INTEGER,             -- Tokens in context after compaction

    -- Audit
    compaction_prompt_hash TEXT,  -- Hash of the summarization prompt used
    created_at TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(session_id, version)
);

CREATE INDEX IF NOT EXISTS idx_meta_knowledge_session ON meta_knowledge(session_id);
CREATE INDEX IF NOT EXISTS idx_meta_knowledge_version ON meta_knowledge(session_id, version DESC);
