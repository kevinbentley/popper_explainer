Great — here are two deliverables:

1. **API contract** between Law Discovery ↔ Tester Harness (schemas, endpoints/functions, error handling, versioning)
2. **Regression suite spec** (a “green bar” suite of known-true/known-false/known-unknown laws with expected outcomes + counterexample characteristics)

I’m going to assume a Python implementation, but this is language-agnostic.

---

# 1) API Contract: Law Discovery ↔ Tester Harness

## 1.1 Goals

* Deterministic, reproducible evaluation of `CandidateLaw` objects
* Clear separation of concerns:

  * Discovery proposes **compileable laws**
  * Harness evaluates laws using existing capabilities
* Results are:

  * machine-readable
  * audit-friendly
  * stable across schema evolution

---

## 1.2 Versioning and Compatibility

### Schema version fields

Every message object exchanged MUST include:

* `schema_version`: semantic version string (e.g., `"1.0.0"`)
* `universe_id`: stable identifier for the universe (e.g., hash of universe contract + sim hash)
* `sim_hash`: hash of simulator implementation/version used by harness
* `harness_hash`: hash/version of harness implementation

### Backwards compatibility rules

* Additive fields are allowed (consumers must ignore unknown fields).
* Breaking changes require bumping major schema version.
* Harness MUST reject unknown `template` values.

---

## 1.3 Data Structures

### UniverseContract (shared)

Minimal required fields (Discovery and Harness must agree):

```json
{
  "schema_version": "1.0.0",
  "universe_id": "string",
  "symbols": [".", ">", "<", "X"],
  "state_representation": "string|array",
  "capabilities": {
    "primitive_observables": ["count(symbol)", "grid_length"],
    "derived_observables_allowed": true,
    "transforms": ["mirror_swap", "shift_k", "swap_only", "mirror_only"],
    "generator_families": [
      "random_density_sweep",
      "constrained_pair_interactions",
      "edge_wrapping_cases",
      "symmetry_metamorphic_suite",
      "adversarial_mutation_search"
    ]
  },
  "config_knobs": {
    "grid_length_range": [4, 200],
    "boundary_types": ["periodic"]
  }
}
```

### CandidateLaw (Discovery → Harness)

Hard requirements: compileable template, falsifier, explicit horizon.

```json
{
  "schema_version": "1.0.0",
  "law_id": "string",
  "template": "invariant|monotone|implication_step|implication_state|eventually|symmetry_commutation|bound",
  "quantifiers": { "T": 50, "H": 10 },
  "preconditions": [
    {"lhs":"grid_length","op":">=","rhs":5}
  ],
  "observables": [
    {"name":"R_total","expr":"count('>') + count('X')"}
  ],
  "claim": "restricted_expression_string",
  "forbidden": "restricted_expression_string",
  "proposed_tests": [
    {"family":"random_density_sweep","params":{"cases":200,"densities":[0.1,0.3,0.6]}}
  ],
  "capability_requirements": {
    "missing_observables": [],
    "missing_transforms": [],
    "missing_generators": []
  }
}
```

### HarnessConfig (caller → Harness)

```json
{
  "schema_version": "1.0.0",
  "seed": 12345,
  "max_runtime_ms_per_law": 2500,
  "max_cases": 500,
  "default_T": 50,
  "max_T": 200,
  "min_cases_used_for_pass": 50,
  "enable_adversarial_search": true,
  "adversarial_budget": 2000,
  "enable_counterexample_minimization": true,
  "minimization_budget": 500,
  "store_full_trajectories": false,
  "require_non_vacuous": true
}
```

### LawVerdict (Harness → Discovery)

```json
{
  "schema_version": "1.0.0",
  "universe_id": "string",
  "sim_hash": "string",
  "harness_hash": "string",
  "law_id": "string",
  "law_fingerprint": "string",
  "status": "PASS|FAIL|UNKNOWN",
  "reason_code": "missing_observable|missing_transform|missing_generator_knob|ambiguous_claim|unmet_preconditions|resource_limit|inconclusive_low_power|null",
  "evidence": {
    "cases_attempted": 500,
    "cases_used": 240,
    "families_run": ["random_density_sweep","constrained_pair_interactions","adversarial_mutation_search"],
    "runtime_ms": 812
  },
  "power_metrics": {
    "vacuous": false,
    "collision_events_seen": 120,
    "wrap_events_seen": 35,
    "density_bins_hit": [0.1,0.3,0.6],
    "violation_score_max": 0.0
  },
  "counterexample": null,
  "artifacts": {
    "first_failure": null,
    "trajectory_excerpt": null
  }
}
```

### Counterexample (when FAIL)

```json
{
  "initial_state": "string",
  "config": {"grid_length": 10, "boundary": "periodic"},
  "seed": 9981,
  "T": 12,
  "t_fail": 3,
  "witness": {
    "message": "explain mismatch succinctly",
    "state_before": "string",
    "state_after": "string",
    "observables_at_t0": {"R_total": 3},
    "observables_at_tfail": {"R_total": 4}
  }
}
```

---

## 1.4 Primary Interfaces

You can implement these as functions, RPC, REST, CLI—doesn’t matter. This is the logical contract.

### `evaluate_laws(...)`

**Purpose:** batch evaluation with shared config and contract.

**Signature (conceptual):**

* Input: `UniverseContract`, `HarnessConfig`, `CandidateLaw[]`
* Output: `LawVerdict[]` (one per input law, in same order)

**Behavior requirements:**

* Must be deterministic given:

  * contract + sim_hash + harness_hash
  * config + seed
  * laws
* Must evaluate laws independently (no cross-law interference), but may reuse generated cases for efficiency (must log reuse).

### `evaluate_law(...)`

Single-law variant (same semantics).

### `get_capabilities()`

Returns harness-supported:

* templates
* generator families
* transforms
* primitive observables
* expression features (operators/functions allowed)

Discovery uses this to avoid proposing impossible laws.

---

## 1.5 Expression Language Contract (Restricted)

Because your entire pipeline depends on this being stable, define it explicitly.

### Allowed primitives

* `count('<symbol>')` where symbol in contract
* `grid_length`
* Optionally: `occupied_cells = grid_length - count('.')` (or as derived)
* Optionally event metrics if implemented later: `collision_events`, `wrap_events`

### Allowed operators

* arithmetic: `+ - *`
* comparisons: `== != < <= > >=`
* boolean: `and or not`
* parentheses

### Disallowed

* loops, comprehensions, indexing arbitrary state unless explicitly exposed
* non-deterministic functions

If a law uses disallowed expressions, harness returns:

* `UNKNOWN`, `reason_code=ambiguous_claim`, `details`

---

## 1.6 Error Handling

Harness must not throw unhandled exceptions across the contract boundary.

Instead, for each law:

* return `UNKNOWN` with:

  * `reason_code=resource_limit` (timeouts)
  * `reason_code=ambiguous_claim` (compile failure)
  * plus `artifacts.first_failure.message` with the error string

Batch call should always return a verdict for every law.

---

## 1.7 Discovery’s Required Response to Verdicts

Discovery subsystem should treat verdicts as:

* `FAIL`: store law in falsified set + store minimal counterexample
* `PASS`: store law in accepted set **only if** `power_metrics` passes thresholds
* `UNKNOWN`: store in unknown set with reason_code; optionally propose:

  * capability upgrades
  * rewritten claims

---

# 2) Regression Suite Spec

The goal is a set of laws that give you a stable “green bar” as you iterate on harness behavior, plus a set that is expected to fail (red bar), and a set that should return UNKNOWN for capability reasons.

This suite should be run on every commit to:

* claim compiler
* generators
* metamorphic testing
* counterexample minimization

## 2.1 Regression Suite Organization

### Test groups

1. **Schema/Compiler tests**
2. **True-law tests** (must PASS non-vacuously)
3. **False-law tests** (must FAIL with counterexample)
4. **Unknown-law tests** (must return UNKNOWN with specific reason_code)
5. **Metamorphic symmetry tests** (PASS/FAIL depending on truth)
6. **Reproducibility tests** (same seed gives same counterexample)

### Naming convention

* `T_*` for true laws
* `F_*` for false laws
* `U_*` for unknown laws

---

## 2.2 Assumptions about Your Universe

Based on your described universe:

* Symbols: `. > < X`
* `X` represents co-location of `>` and `<`
* particles move at v=1
* `X` resolves in one step into separated particles (`<.>` pattern)
* boundaries are periodic
* “full mirror symmetry” is mirror+swap commutation

If any of these change, update the suite.

---

## 2.3 Canonical Observables for Suite

Define these derived observables in the suite (and ensure harness supports them):

* `R_total = count('>') + count('X')`
* `L_total = count('<') + count('X')`
* `particle_count = count('>') + count('<') + 2*count('X')`
* `momentum = R_total - L_total`
* `occupied_cells = count('>') + count('<') + count('X')`
* `collision_count = count('X')`

---

## 2.4 True Laws (must PASS)

### T1 — Right-component conservation

* Template: invariant
* Preconditions: none (or grid_length>=4)
* Claim: `R_total(t) == R_total(0)` for all t≤T
* Expect: PASS, non-vacuous (cases_used >= threshold)

### T2 — Left-component conservation

Same as T1 for `L_total`.

### T3 — Particle count conservation

* Claim: `particle_count(t) == particle_count(0)` for all t≤T
* Expect: PASS

### T4 — Momentum conservation

* Claim: `momentum(t) == momentum(0)` for all t≤T
* Expect: PASS

### T5 — Collision is ephemeral (if your rule guarantees 1-step resolution)

* Template: implication_step
* Preconditions: `collision_count(0) > 0` (or more generally: “exists X at time t”)
* Claim: `collision_count(t) > 0 -> collision_count(t+1) < collision_count(t)` is too strong.
  Better: **local**:

  * `collision_count(t) > 0 -> collision_count(t+1) >= 0` is trivial.
    Instead use:
* Template: implication_step
* Claim: `count('X')(t) > 0 -> count('X')(t+1) == 0` **only** if your universe prevents multiple collisions. If multiple X possible, restrict:

  * Preconditions: “exactly one X and no adjacent interactions” is hard.
    So for regression, pick a *specific deterministic case* and test via `constrained_pair_interactions`.
* Expect: PASS for the specific family; otherwise skip if too brittle.

### T6 — Full mirror symmetry (mirror+swap commutation)

* Template: symmetry_commutation
* Transform: `mirror_swap`
* Expect: PASS

---

## 2.5 False Laws (must FAIL)

These should reliably produce counterexamples with small cases.

### F1 — Collision count conserved

* Template: invariant
* Claim: `collision_count(t) == collision_count(0)` for all t≤T
* Expected: FAIL
* Expected counterexample pattern: a case where `X` resolves so collision_count decreases (e.g., initial includes `X`)

### F2 — Occupied cells conserved (globally)

* Template: invariant
* Claim: `occupied_cells(t) == occupied_cells(0)` for all t≤T
* Expected: FAIL
* Counterexample: `X` resolves into `<.>` increasing occupied cells (1→2)

### F3 — Spatial mirror without swap is a symmetry

* Template: symmetry_commutation
* Transform: `mirror_only`
* Expected: FAIL
* Counterexample: a state like `>..` vs `..>` evolves differently without swapping directions

### F4 — Time reversal symmetry (if your universe is not reversible)

* Template: symmetry_commutation (or separate “preimage uniqueness” if you implement it later)
* If you don’t support preimage search yet, classify as UNKNOWN (see U group). If you *do* support it:

  * Expected: FAIL (witness: two distinct predecessors mapping to same state, or inability to reconstruct)

### F5 — “No collision change implies pure-right increases” (wrong)

* Template: implication_step
* Preconditions: `collision_count(t) == collision_count(t+1)` (hard to express unless you expose observables across steps)
* This one is annoying in a limited claim language; keep it out unless your expression language supports referencing t+1 observables in precondition.
  Better false implication:
* Template: implication_step
* Claim: `collision_count(t) > 0 -> occupied_cells(t+1) <= occupied_cells(t)`
* Expected: FAIL, since resolution expands occupied cells.

---

## 2.6 Unknown Laws (must return UNKNOWN with specific reason_code)

These are important to prevent the harness from lying.

### U1 — Requires missing observable

* Claim references `collision_events_seen` if you haven’t implemented it
* Expected: UNKNOWN, `reason_code=missing_observable`

### U2 — Requires missing transform

* Symmetry with transform not in contract (e.g., `rotate_2d`)
* Expected: UNKNOWN, `reason_code=missing_transform`

### U3 — Ambiguous claim

* Non-compileable template or uses unsupported expression features (indexing, loops)
* Expected: UNKNOWN, `reason_code=ambiguous_claim`

### U4 — Unmet preconditions

* Preconditions impossible to satisfy given generator limits (e.g., grid_length < 0)
* Expected: UNKNOWN, `reason_code=unmet_preconditions`

### U5 — Resource limit

* Very large T and max_cases beyond configured caps
* Expected: UNKNOWN, `reason_code=resource_limit`

---

## 2.7 Reproducibility Tests

### RPT1 — Counterexample reproducibility

* Pick a false law (F1)
* Run twice with same seed/config
* Expect identical `counterexample.initial_state`, `t_fail`, and witness observables

### RPT2 — Minimization stability

* With minimization enabled, FAIL result should produce *same or smaller* grid length than without minimization, and still fail.

---

## 2.8 Test Runner Behavior Requirements

The regression runner must:

* set fixed `HarnessConfig.seed`
* clamp all randomness through harness RNG only
* print a short diff-friendly report

Example expected outputs per case:

* `PASS (power ok)`
* `FAIL (counterexample found: N=10, t_fail=3)`
* `UNKNOWN (missing_observable: collision_events)`

---

## 2.9 “Green Bar” Thresholds

For PASS tests, require minimum power:

* `cases_used >= 50` (configurable)
* `vacuous == false`
* at least one “relevant event” for certain laws:

  * for collision-related laws: `collision_events_seen > 0` or at least `count('X')` seen in trajectories

If not met, suite expects UNKNOWN with `inconclusive_low_power` rather than PASS.

---

# Next step (optional but strongly helpful)

If you want, I can also provide a **starter set of JSON law fixtures** for T/F/U that exactly match the `CandidateLaw` schema above, so your coding agent can implement:

* compiler
* generators
* checker
* verdict objects

and immediately run the regression suite.

Tell me whether your simulator state is represented as a string (e.g., `"..><.X.."`) or as an array/list—then I’ll format the fixtures accordingly.
