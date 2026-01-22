Below is a requirements document for the **Law Discovery** portion only (not the tester). It is written so you can hand it directly to a coding agent and start implementing. I’ve included interfaces, data types, acceptance criteria, and “non-goals” to keep it tight.

---

# Requirements Document: Law Discovery Subsystem

## 0. Purpose

The Law Discovery subsystem proposes **testable, falsifiable candidate laws** about a simulated universe based on prior experiment results, while enforcing Popperian discipline:

* Laws must be **compileable** into a constrained claim language.
* Laws must include explicit **preconditions, quantifiers, horizons, observables, and falsifiers**.
* The subsystem must avoid “axiom inflation” (relabeling observations as axioms).
* The subsystem must prioritize laws that are **risky**, **novel**, and **discriminating**.

This subsystem does **not** execute tests; it only proposes laws and ranks them. It consumes summaries produced by the tester.

---

## 1. Scope

### In scope

* Structured **law proposal** generation (LLM-driven)
* **Schema validation** and normalization
* **Redundancy detection** (syntactic + shallow semantic)
* **Ranking/prioritization** for testing
* Maintaining compact, structured **working memory** for prompting

### Out of scope (explicit)

* Running simulations or evaluating laws against traces
* Generating Python test code (belongs to Tester subsystem)
* Mechanism/axiom synthesis (separate subsystem)
* Proof checking / theorem proving (separate subsystem)

---

## 2. Definitions

* **Observation**: A concrete trajectory or metric series produced by the simulator for a specific initial state and configuration.
* **Law (Claim)**: A quantified, falsifiable statement about trajectories, expressed in a constrained template language.
* **Counterexample**: A specific experiment result that violates a law’s forbidden condition.
* **Inconclusive / Untestable**: Not enough tester capability, not enough coverage, or ambiguous claim; law not accepted nor rejected.

---

## 3. Inputs and Outputs

### Inputs (from upstream subsystems)

1. `UniverseContract`

* Symbols, state representation, and allowed observables/transforms (capability list).
* Any simulator configuration knobs exposed (grid length range, boundary variants, etc.) as metadata (even if not all are used yet).

2. `DiscoveryMemorySnapshot`

* A compact summary of:

  * Accepted laws (top K)
  * Falsified laws (top K) with minimal counterexamples
  * Unknown laws (top K) with reason codes
  * Counterexample gallery (top K)
  * Current tester capability set (observables, transforms, generator families)

3. Optional: `Request`

* How many laws to propose (`k`)
* Target law families to explore (e.g., symmetry, invariant) or exclude
* Current “open questions” (capability gaps) the system wants to prioritize

### Outputs

1. `CandidateLawBatch`

* List of candidate laws (validated and normalized), each with metadata:

  * novelty score, expected risk, expected discriminative value, and predicted test families

2. `Rejections`

* Laws rejected due to schema violation, redundancy, or prohibited patterns (with reasons)

---

## 4. Core Requirements

### R1 — Constrained Claim Language (Templates)

All proposed laws MUST be expressible in one of the following templates:

1. `invariant`

* Form: `∀t∈[0..T]: f(state[t]) == f(state[0])`

2. `monotone`

* Form: `∀t∈[0..T-1]: f(t+1) <= f(t)` (or >=)

3. `implication_step`

* Form: `∀t∈[0..T-1]: P(state[t]) → Q(state[t+1])`

4. `implication_state`

* Form: `∀t∈[0..T]: P(state[t]) → Q(state[t])`

5. `eventually`

* Form: `∀t0∈[0..T]: P(state[t0]) → ∃t∈[t0..min(t0+H,T)]: Q(state[t])`

6. `symmetry_commutation`

* Form: `evolve(Transform(S), T) == Transform(evolve(S, T))`

7. `bound`

* Form: `∀t∈[0..T]: f(t) <= k` (or >=)

If the LLM outputs a law not matching these, it is **rejected** (not “untestable”).

---

### R2 — Law Object Must Include Falsifier

Every proposed law must include:

* `forbidden`: a machine-checkable description of what constitutes a counterexample.

Examples:

* `invariant`: “exists t where f(t) != f(0)”
* `implication_step`: “exists t where P(t) true and Q(t+1) false”
* `symmetry_commutation`: “exists t where state_A[t] != transform(state_B[t])”

No law may be accepted into the candidate batch without a falsifier definition.

---

### R3 — Preconditions are Mandatory

Every law must specify applicability conditions as structured predicates over the **initial state and configuration**.

Examples:

* `grid_length >= 5`
* `count('>') >= 1`
* `boundary_type == periodic` (only if boundary variants exist in UniverseContract)
* `collision_count(0) > 0`

If no preconditions are needed, the law must explicitly declare `preconditions: []`.

---

### R4 — Observables Must Be Declared and Must Be Available

Each law must explicitly list:

* `observables`: named metrics used by `f`, `P`, `Q`, or `Transform`.

Each observable must map to one of:

* A primitive observable in `UniverseContract` (e.g., `count(symbol)`)
* A derived observable defined in the law object via a restricted expression language.

If an observable cannot be resolved using current capabilities, the law is still allowed (it may become “unknown”), but it must be labeled:

* `capability_requirements`: what is missing (observable/transform/generator)

This enables the tester to return actionable “unknown reason codes.”

---

### R5 — No Axioms During Law Discovery

The LLM is forbidden from proposing:

* “axioms”
* “mechanisms”
* “definitions of the universe”
* “update rules”

It may only propose **claims** as above.

(Those belong to a separate Mechanism Synthesis subsystem.)

---

### R6 — Redundancy and Triviality Filters

The subsystem must reject or downrank laws that are:

* Duplicate of an already-known accepted/falsified law (syntactic match or normalized match)
* Trivial identities (e.g., “occupied = R + L + X” if “occupied” is defined that way)
* Pure rephrasings (e.g., conservation and “non-decreasing” that is implied by conservation)

Minimum implementation:

* Normalize template + observables + preconditions
* Hash normalized form
* Reject exact duplicates
* Downrank near-duplicates via similarity heuristic (see R10)

---

### R7 — Novelty / Discrimination Requirement

Each law must include at least one of:

* `distinguishes_from`: references to one or more plausible rival hypotheses OR
* `novelty_claim`: why it is not implied by accepted laws

If neither is provided, the law is allowed but receives a **low ranking** by default.

---

### R8 — Proposed Test Families

Each law must propose at least one test family in structured form, from a known list of generator families (declared by tester capability set), e.g.:

* `random_density_sweep`
* `constrained_pair_interactions`
* `edge_wrapping_cases`
* `symmetry_metamorphic_suite`
* `adversarial_mutation_search`

If the system proposes a test family not currently supported, it must declare it as a capability requirement.

---

### R9 — Deterministic Prompt Construction

Prompting must be constructed deterministically from inputs:

* stable ordering
* fixed formatting
* explicit token budgeting

Rationale: makes behavior reproducible for debugging and ablation.

---

### R10 — Ranking Model

The subsystem must produce a ranked list of candidate laws using a deterministic scoring function:

`score = w1*risk + w2*novelty + w3*discrimination + w4*testability - w5*redundancy`

Where:

* `risk`: how easily falsifiable it is (strong prohibitions, broad preconditions)
* `novelty`: distance from known laws
* `discrimination`: likelihood to separate rival mechanisms
* `testability`: based on current tester capabilities
* `redundancy`: similarity to existing items

Initial implementation may be heuristic (no ML required).

---

## 5. Data Types

### 5.1 UniverseContract (input)

```json
{
  "symbols": [".", ">", "<", "X"],
  "state_representation": "string|array",
  "capabilities": {
    "primitive_observables": ["count(symbol)", "grid_length", "cell_at(i)"],
    "derived_observables_allowed": true,
    "transforms": ["mirror_swap", "shift_k", "swap_only", "mirror_only"],
    "generator_families": ["random_density_sweep", "constrained_pair_interactions", "symmetry_metamorphic_suite"]
  },
  "config_knobs": {
    "grid_length_range": [4, 200]
  }
}
```

### 5.2 CandidateLaw (output)

```json
{
  "law_id": "string",
  "template": "invariant|monotone|implication_step|implication_state|eventually|symmetry_commutation|bound",
  "quantifiers": {
    "type": "forall|exists|mixed",
    "T": 50,
    "H": 10
  },
  "preconditions": [
    {"lhs": "grid_length", "op": ">=", "rhs": 5}
  ],
  "observables": [
    {"name": "R_total", "expr": "count('>') + count('X')"}
  ],
  "claim": "string (restricted expression form)",
  "forbidden": "string (restricted expression form)",
  "distinguishes_from": ["optional ids or descriptions"],
  "proposed_tests": [
    {"family": "random_density_sweep", "params": {"cases": 200, "densities": [0.1,0.3,0.6]}}
  ],
  "capability_requirements": {
    "missing_observables": [],
    "missing_transforms": [],
    "missing_generators": []
  },
  "ranking_features": {
    "risk": 0.0,
    "novelty": 0.0,
    "discrimination": 0.0,
    "testability": 0.0,
    "redundancy": 0.0
  }
}
```

### 5.3 DiscoveryMemorySnapshot (input)

Provide as structured lists of laws and counterexamples, not raw logs.

---

## 6. Prompting Requirements (LLM)

### P1 — Strict output format

The LLM must output:

* JSON array of CandidateLaw objects ONLY
* No prose
* No markdown

### P2 — Include “Do not propose axioms” rule

Prompt must explicitly ban:

* axioms
* rules of evolution
* redefining metrics
* restating accepted laws

### P3 — Context packaging

Prompt must include:

* UniverseContract capability summary
* Accepted laws (top K, normalized)
* Falsified laws with minimal counterexamples
* Unknown laws with reason codes
* Counterexample gallery (small)

K must be configurable; default K=30 accepted, 20 falsified, 20 unknown, 20 counterexamples.

---

## 7. Acceptance Criteria

Law Discovery subsystem is considered correct when:

1. **100%** of proposed laws validate against schema and compile into one of the templates (or are rejected with a reason).
2. Duplicate proposals across iterations drop significantly (redundancy filter works).
3. “Untestable” is reduced by:

   * rejecting non-compileable laws early, and
   * surfacing capability requirements for the rest.
4. Proposed laws increasingly target:

   * symmetry commutation tests
   * invariants involving derived observables
   * implication-based event laws
5. The top-ranked laws are measurably more “killable” (higher falsification rate) than random proposals (requires tester metrics; but discovery must output ranking features).

---

## 8. Logging and Audit

The subsystem must log per iteration:

* Prompt hash + prompt size
* Candidate batch before/after normalization
* Rejection reasons distribution
* Ranking scores and final order
* Redundancy matches (what it was similar to)

Logs must be sufficient to reconstruct why a law was proposed and why it was prioritized.

---

## 9. Non-Functional Requirements

* Deterministic behavior aside from LLM sampling (seedable)
* Configurable token budgets and K values
* Safe JSON parsing and strict validation
* Backwards-compatible schema evolution (version field recommended)

---

## 10. Implementation Notes (recommended, not required)

* Implement a `ClaimCompiler` that converts `template + observables + claim/forbidden` into an internal AST used by the tester later.
* Implement normalization rules (commutative reordering, canonical precondition ordering, whitespace stripping).
* Keep a law fingerprint index for fast duplicate detection.

---
