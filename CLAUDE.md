# CLAUDE.md — Popperian Explainer Project (Law Discovery + Tester Harness)

This repository implements a Popperian-style discovery loop in a simulated universe. The system proposes falsifiable laws, attempts to refute them via simulation-driven experiments, and persists all results so the process can be paused/resumed safely.

This file is written for a coding agent. Follow it as the primary engineering guidance.

---

## 1) Project Goal (High-Level)

Build a non-human agent that can satisfy Popperian constraints for “explanation” in a synthetic universe not present in training data.

Operationally:
- Propose **compileable** candidate laws (claims) using an LLM.
- Test laws via a deterministic simulation harness that attempts falsification.
- Track outcomes as PASS / FAIL / UNKNOWN with counterexamples and power metrics.
- Persist all state in SQLite so long runs can be aborted and resumed.

---

## 2) Design Principles (Non-Negotiable)

### 2.1 Popper Discipline
- A “law” is only a law if it can be falsified.
- Every candidate law must include an explicit **forbidden** condition (counterexample definition).
- “PASS” is not proof. A pass must be **non-vacuous** and meet minimum power thresholds or is downgraded to UNKNOWN.

### 2.2 Strict Claim Language
The discovery stage must only propose laws using a constrained template set:
- invariant
- monotone
- implication_step
- implication_state
- eventually
- symmetry_commutation
- bound

Reject laws outside these templates early as invalid (do not label them “untestable”).

### 2.3 Separation of Concerns
- Law Discovery proposes candidate laws + suggested tests.
- Tester Harness executes experiments + evaluates laws.
- Do not mix “axioms” or “mechanisms” into law discovery. Mechanism synthesis is a separate subsystem.

### 2.4 Determinism and Reproducibility
Given the same:
- sim hash
- harness config
- law
- seed
the harness must produce identical outcomes and counterexamples.

All randomness must be controlled by the harness RNG.

---

## 3) Persistence (SQLite): Required Behavior

### 3.1 Why SQLite
Runs may last hours/days. We need durable state and the ability to resume after abort/crash without losing progress or duplicating work.

### 3.2 Persistence requirements
The system MUST persist:
- every proposed law (even if rejected)
- every evaluation attempt (including config + seed)
- every verdict
- counterexamples (minimal reproduction packages)
- power metrics and evidence summary
- iteration metadata and prompt hashes (audit)

### 3.3 Resume requirements
On startup:
- detect the latest completed iteration (or last durable checkpoint)
- continue from there
- never re-test identical (law_fingerprint, harness_config_hash, sim_hash) unless explicitly requested

---

## 4) Minimal Data Model (SQLite)

Implement at least these tables. Use foreign keys and indexes.

### 4.1 `runs`
One logical end-to-end run.

Columns:
- id (PK)
- created_at
- universe_id
- sim_hash
- harness_hash
- discovery_model_id (string)
- tester_model_id (optional, string)
- config_json (full run config)

### 4.2 `iterations`
Tracks loop steps and makes resume trivial.

Columns:
- id (PK)
- run_id (FK)
- iteration_index (int)
- started_at
- completed_at (nullable)
- status (running|completed|aborted)
- prompt_hash (nullable)
- summary_json (compact memory snapshot used for discovery)

Index:
- (run_id, iteration_index) UNIQUE

### 4.3 `laws`
Canonical law objects (normalized) plus fingerprints.

Columns:
- id (PK)
- run_id (FK)
- law_id (string from LLM)
- law_fingerprint (string UNIQUE per run)
- schema_version
- template
- quantifiers_json
- preconditions_json
- observables_json
- claim_text
- forbidden_text
- proposed_tests_json
- capability_requirements_json
- created_iteration_id (FK)
- raw_llm_json (store original as text)
- normalized_json (store normalized as text)
- status (proposed|rejected_schema|queued|tested)

Indexes:
- (run_id, law_fingerprint) UNIQUE
- (run_id, template)
- (run_id, status)

### 4.4 `law_evaluations`
Every evaluation attempt with deterministic parameters.

Columns:
- id (PK)
- run_id (FK)
- law_id (FK -> laws.id)
- harness_config_hash
- seed
- started_at
- completed_at
- status (PASS|FAIL|UNKNOWN)
- reason_code
- evidence_json
- power_metrics_json
- runtime_ms
- counterexample_id (FK nullable)
- artifacts_json
- notes

Indexes:
- (run_id, law_id, harness_config_hash, seed) UNIQUE
- (run_id, status)

### 4.5 `counterexamples`
Repro packages for FAIL.

Columns:
- id (PK)
- run_id (FK)
- law_evaluation_id (FK)
- initial_state
- config_json
- seed
- T
- t_fail
- witness_json
- trajectory_excerpt_json
- minimized (bool)
- created_at

Index:
- (run_id, law_evaluation_id) UNIQUE

### 4.6 `capability_snapshots`
Track capability changes over time (instrumentation upgrades).

Columns:
- id (PK)
- run_id (FK)
- iteration_id (FK)
- universe_contract_json
- harness_capabilities_json
- created_at

---

## 5) Normalization and Fingerprinting

### 5.1 Normalization
Before inserting a law:
- canonicalize template string
- sort preconditions by (lhs, op, rhs)
- normalize whitespace in claim/forbidden
- normalize commutative expressions where feasible (optional v1)
- stable JSON key ordering

### 5.2 Fingerprint
Compute `law_fingerprint = sha256(normalized_json)`.

Use it to:
- dedupe proposals
- ensure idempotent resume behavior

---

## 6) Control Flow (Main Loop)

### 6.1 Iteration steps (recommended)
1) Build `DiscoveryMemorySnapshot` from DB (recent accepted/falsified/unknown + counterexample gallery + capabilities).
2) Call Law Discovery LLM to propose K candidate laws (JSON only).
3) Validate + normalize + insert into `laws` table (rejected_schema if invalid).
4) Rank and select top M for testing (skip already tested fingerprints under same config hash).
5) For each selected law:
   - evaluate via Tester Harness
   - insert row into `law_evaluations`
   - insert counterexample if FAIL
6) Mark iteration complete with summary_json and prompt_hash.

### 6.2 Abort safety
- Iteration row should be created at start with status=running.
- Mark completed_at + status=completed only after all selected evaluations are committed.
- All inserts must happen in transactions.

---

## 7) Tester Harness Requirements (Operational)

- Must enforce preconditions: cases not satisfying preconditions do not count as evidence.
- Must compute power metrics and downgrade PASS → UNKNOWN if vacuous or low-power.
- Must attempt falsification with:
  - random cases
  - constrained cases
  - metamorphic tests for symmetry laws
  - adversarial mutation search (if enabled)
- On FAIL, must return a counterexample package and attempt bounded minimization.

---

## 8) Logging and Auditability

- Store prompt hashes and the exact JSON returned by LLM calls.
- Store harness config and hashes per evaluation.
- Store first failure witness and trajectory excerpts for FAIL.
- Prefer compact structured JSON over raw text logs.

---

## 9) Suggested Directory Structure

- `src/`
  - `discovery/` (LLM prompt builder, schema validator, normalizer, ranker)
  - `harness/` (claim compiler, generators, evaluator, minimizer)
  - `db/` (schema, migrations, DAO/repository)
  - `sim/` (universe simulator contract + step/run)
  - `cli/` (run/resume commands)
- `tests/fixtures/` (JSON fixtures and expectations)
- `tests/` (unit + integration + regression)

---

## 10) CLI Requirements

Implement at least:
- `run --db path --run-id optional` (start new or resume existing run)
- `resume --run-id X`
- `status --run-id X` (progress summary)
- `retest --law-fingerprint ...` (optional)

---

## 11) Practical Defaults

- Use state strings for universe states (e.g., `"..><.X.."`).
- Store full trajectories only when debugging; otherwise store excerpts around failure.
- Keep K (proposed laws) small at first (e.g., 20–50), M (tested per iter) smaller (e.g., 5–15).

---

## 12) Non-Goals (for now)

- Do not execute arbitrary LLM-generated Python for testing in v1.
- Do not add mechanism synthesis until law discovery + harness stabilize.
- Do not add distributed execution until single-machine resume is solid.

---

## 13) Definition of Done (Law Discovery + Harness v1)

- Regression fixtures pass:
  - True laws PASS non-vacuously
  - False laws FAIL with counterexamples
  - Unknown laws return UNKNOWN with correct reason codes
- System can be interrupted and resumed with no duplicate evaluations.
- All verdicts are traceable in SQLite with minimal reproduction data.
