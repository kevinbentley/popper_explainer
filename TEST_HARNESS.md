Below is a requirements document for the **Tester Harness** subsystem. This is the component that takes a `CandidateLaw` (from Law Discovery) plus the simulator contract, generates/chooses experiments, runs them, and returns **PASS / FAIL / UNKNOWN** with **counterexamples** and **power metrics**.

I am assuming your simulator already exists as a deterministic `step()` / `run()` function over states (or will), and that you are not yet committing to running arbitrary LLM-written Python. I include both modes: **Plan-Driven** (recommended first) and **Sandboxed Code** (optional extension).

---

# Requirements Document: Tester Harness Subsystem

## 0. Purpose

The Tester Harness evaluates candidate laws Popper-style by attempting to **falsify** them under explicit preconditions and within specified horizons, using:

* deterministic simulation execution
* structured experiment families (generators)
* metamorphic tests (symmetry commutation)
* adversarial search for counterexamples
* capability-aware reporting (why something is unknown)

It must produce:

* concrete counterexamples when a law fails
* “unknown” with actionable reason codes when it cannot decide
* test power / coverage metrics so “pass” is not mistaken for “proven”

---

## 1. Scope

### In scope

* compiling claim templates into executable predicates over trajectories
* experiment case generation (random + constrained)
* running simulations and collecting observables
* checking claims and producing counterexamples
* adversarial counterexample search (property-based + hillclimb)
* metamorphic tests for symmetry claims
* UNKNOWN classification with reason codes
* logging and reproducibility (seeds, configs, traces)

### Out of scope

* discovering laws (Law Discovery)
* mechanism/axiom synthesis and proof derivations
* editing/learning the simulator rule itself
* high-level agent orchestration across subsystems (though Harness should be callable)

---

## 2. Inputs and Outputs

### Inputs

1. `UniverseContract`

* symbols, state format, config knobs
* available observables/transforms/generator families
* resource limits defaults

2. `CandidateLaw` (from Law Discovery)

* template, quantifiers, horizon
* preconditions
* observables and definitions
* claim + forbidden
* proposed test families (optional but used for prioritization)

3. `HarnessConfig`

* budgets, seeds, timeouts, max cases, max horizon
* adversarial search parameters
* logging verbosity

### Outputs

1. `LawVerdict`

* `status`: `PASS | FAIL | UNKNOWN`
* `reason_code`: for UNKNOWN (and optionally PASS/FAIL metadata)
* `counterexample`: if FAIL (minimal reproducible case)
* `evidence`: summary of runs and checks
* `power_metrics`: coverage and discriminative strength
* `artifacts`: optional traces, metrics time series, metamorphic mismatch witness

---

## 3. Core Requirements

### R1 — Claim Compilation

The Harness must support the claim template set defined in Law Discovery:

* invariant
* monotone
* implication_step
* implication_state
* eventually
* symmetry_commutation
* bound

It must compile each `CandidateLaw` into a checkable predicate over:

* `initial_state`
* `trajectory` (states 0..T)
* `observables(t)` for declared observables

If compilation fails (unknown template, invalid expression), verdict is UNKNOWN with `reason_code = ambiguous_claim`.

---

### R2 — Preconditions Enforcement

For each candidate experiment case, the harness must evaluate `preconditions` on the initial state/config:

* Cases that do not satisfy preconditions are not allowed to count as evidence.
* If after budgeted generation/search the harness cannot produce *any* precondition-satisfying case, verdict is UNKNOWN with `reason_code = unmet_preconditions`.

Must record: how many attempted cases were rejected for preconditions.

---

### R3 — Observable Resolution and Instrumentation

Harness must compute declared observables via:

* primitive observables in UniverseContract (e.g., `count(symbol)`, `grid_length`)
* derived observables defined by restricted expressions over primitives (e.g., `R_total = count('>') + count('X')`)

If an observable requires missing capability (e.g., collision events, wrap events, preimage existence) the harness returns UNKNOWN with:

* `reason_code = missing_observable`
* `missing = [...]`

---

### R4 — Experiment Families (Generators)

Harness must support a modular generator framework:

#### Required built-in families (minimum viable)

1. `random_density_sweep`

* random initial states at densities or count constraints

2. `constrained_pair_interactions`

* place localized patterns (e.g., `><`, `>.<`, etc.) at controlled separations

3. `edge_wrapping_cases`

* ensure particles cross boundaries within horizon (for boundary claims)

4. `symmetry_metamorphic_suite`

* generate random states and test commutation with available transforms

5. `adversarial_mutation_search`

* mutate an initial state to maximize violation score / find counterexample

Each family must accept a standard parameter schema and be seedable.

If a law proposes families not implemented, harness may ignore them and use defaults, but must record unmet family support. If the law *requires* an unsupported family (capability requirement), verdict may become UNKNOWN with `reason_code = missing_generator_knob`.

---

### R5 — Default Test Strategy

If a law does not specify tests, harness must still attempt falsification using a default multi-stage strategy:

1. Quick smoke: N small random cases
2. Structured: constrained interactions and edge cases
3. Metamorphic: if applicable (symmetry templates or laws mentioning transforms)
4. Adversarial: mutation/hillclimb to search for counterexample

This ensures “pass” is not just “we didn’t look.”

---

### R6 — Counterexample Search (Popper pressure)

For each law, harness must attempt to find a counterexample with a bounded search budget.

Minimum capabilities:

* property-based random search
* mutation-based hillclimbing with a violation score
* optional: small exhaustive enumeration for tiny grid sizes (useful for refuting absolutes)

If a counterexample is found, verdict must be FAIL and include a minimal reproduction package (see R11).

---

### R7 — Symmetry Testing as Commutation

For `symmetry_commutation` template, harness must test:

`evolve(Transform(S), T) == Transform(evolve(S, T))`

Where:

* `Transform` is one of the declared transforms (mirror+swap, shift, etc.)

Harness must:

* generate cases that exercise the symmetry (not only empty or trivial states)
* produce a witness at the first time step where mismatch occurs
* return FAIL with witness if mismatch found

If transform not available, UNKNOWN with `reason_code = missing_transform`.

---

### R8 — Eventuality Claims Must Be Checked Properly

For `eventually` template:

`∀t0: P(t0) -> ∃t in [t0..t0+H] Q(t)`

Harness must:

* check all `t0` where P holds in the trajectory
* verify existence of t satisfying Q within window
* return FAIL with specific t0 and missing window witness if violated

If law’s `H` is too large for resource limit, allow clamping but must record that the test was incomplete; verdict becomes UNKNOWN unless configured otherwise.

---

### R9 — UNKNOWN Must Be Actionable (Reason Codes)

Harness must never return UNKNOWN without a reason code and “minimal capability needed.”

Required reason codes:

* `missing_observable`
* `missing_transform`
* `missing_generator_knob`
* `ambiguous_claim`
* `unmet_preconditions`
* `resource_limit`
* `inconclusive_low_power` (tests ran, but power too low)

Each UNKNOWN must include:

* `details`: what was missing or why inconclusive
* `suggested_upgrade`: what capability would likely make it decidable

---

### R10 — Power Metrics (Don’t confuse PASS with proof)

Every PASS or UNKNOWN must include metrics that indicate how strong the testing was.

Required power metrics:

* `cases_attempted`, `cases_used` (preconditions satisfied)
* `coverage` (simple proxies, e.g., density bins hit, collision events observed, boundary wraps observed)
* `violation_score_max` achieved during adversarial search
* `metamorphic_cases` run and mismatch counts
* `rival_discrimination` (optional, see R14)
* `vacuity_checks`: whether P ever held for implication/eventually templates

PASS is only valid if:

* `cases_used >= min_cases_used`
* `vacuity_checks` pass (i.e., not all tests were vacuous)
  Otherwise PASS must be downgraded to UNKNOWN with `reason_code = inconclusive_low_power`.

---

### R11 — Counterexample Minimization

When FAIL occurs, harness must attempt to minimize the counterexample to improve auditability:

* reduce grid length if possible
* reduce number of particles/symbols while preserving failure
* reduce horizon to first failing step
* produce canonical formatting of initial state and config

This can be heuristic and bounded by budget, but must exist.

Output must include:

* `initial_state`
* `config` (grid length, boundary type, etc.)
* `seed` (if generator-based)
* `T`, and failing time index `t_fail`
* `trajectory_excerpt` around failure
* computed observables at failure

---

### R12 — Reproducibility and Determinism

Given the same:

* simulator version/hash
* harness config
* random seed
* initial state + config

the harness must reproduce identical verdict and counterexample.

Harness must log:

* `sim_hash`, `harness_hash`, `law_id`, `seed`, `config`

---

### R13 — Safety and Resource Limits

Harness must enforce hard limits:

* max runtime per law
* max steps per simulation (T cap)
* max number of cases
* max memory for stored traces (store excerpts by default)

If limits are hit before sufficient power, verdict is UNKNOWN with `reason_code = resource_limit`.

---

## 4. Optional Extension: Sandboxed LLM Test Code

This is optional and should be Phase 2+.

### R14 — Sandboxed Code Interface

If enabled, the harness may execute tester-provided code, but only inside a strict harness API:

Allowed user code functions:

* `generate_cases(api) -> list[Case]` (optional)
* `check(api, case) -> CheckResult` (optional)

User code must not:

* import modules
* access filesystem/network/subprocess
* define or override simulator
* call eval/exec
* run unbounded loops

### R15 — AST Validation

Before execution, harness must parse code into AST and reject disallowed nodes and patterns.

Minimum disallow list:

* `Import`, `ImportFrom`
* `Exec`, `Eval`, `Compile` (or Python equivalents)
* `open`, `__import__`, `globals`, `locals`, `getattr` (unless allowlisted)
* attribute access outside allowlisted API objects
* while loops without static bounds
* recursion (or set recursion depth to 0)

### R16 — Isolation

Run code in a subprocess with:

* CPU time limit
* memory cap
* no network
* no file access
* kill on timeout

If sandbox triggers, verdict UNKNOWN with `reason_code = resource_limit` (and include sandbox error detail).

---

## 5. Rival Discrimination (Recommended but Optional)

### R17 — Discriminative Testing Against Rival Simulators

To prevent “tests that always pass,” harness can optionally run the same test suite against one or more **rival universes** (small rule variations).

If a law passes on true simulator but also passes on all rivals, `rival_discrimination` is low.

This should not change PASS/FAIL for the real simulator, but should affect power and can downgrade PASS → UNKNOWN if discrimination is required.

---

## 6. Interfaces and Data Types

### 6.1 HarnessConfig

```json
{
  "seed": 12345,
  "max_runtime_ms_per_law": 2000,
  "max_cases": 500,
  "default_T": 50,
  "max_T": 200,
  "min_cases_used_for_pass": 50,
  "enable_adversarial_search": true,
  "adversarial_budget": 2000,
  "enable_counterexample_minimization": true,
  "minimization_budget": 500,
  "store_full_trajectories": false
}
```

### 6.2 Case

```json
{
  "initial_state": "string|array",
  "config": {
    "grid_length": 20,
    "boundary": "periodic"
  },
  "metadata": {
    "generator_family": "random_density_sweep",
    "seed": 12345
  }
}
```

### 6.3 LawVerdict

```json
{
  "law_id": "momentum_conservation",
  "status": "PASS|FAIL|UNKNOWN",
  "reason_code": "missing_observable|missing_transform|missing_generator_knob|ambiguous_claim|unmet_preconditions|resource_limit|inconclusive_low_power|null",
  "evidence": {
    "cases_attempted": 500,
    "cases_used": 240,
    "tests_run": ["random_density_sweep", "constrained_pair_interactions", "adversarial_mutation_search"],
    "notes": []
  },
  "power_metrics": {
    "collision_events_seen": 120,
    "wrap_events_seen": 35,
    "density_bins_hit": [0.1, 0.3, 0.6],
    "violation_score_max": 0.0,
    "vacuous": false,
    "rival_discrimination": 0.4
  },
  "counterexample": null,
  "artifacts": {
    "first_failure": null,
    "trajectory_excerpt": null
  }
}
```

### 6.4 Counterexample (when FAIL)

```json
{
  "initial_state": "....><....",
  "config": {"grid_length": 10, "boundary": "periodic"},
  "seed": 9981,
  "T": 12,
  "t_fail": 3,
  "witness": {
    "state_before": "....><....",
    "state_after":  "....X.....",
    "observables": {"R_total_0": 1, "R_total_3": 2}
  }
}
```

---

## 7. Acceptance Criteria

Tester Harness is considered correct when:

1. It can evaluate all claim templates in R1 and returns correct FAIL on known falsifiable examples.
2. For any FAIL verdict, it provides a **reproducible minimal counterexample** (R11, R12).
3. UNKNOWN verdicts always include actionable reason codes and upgrade hints (R9).
4. PASS verdicts are not vacuous:

   * preconditions satisfied in enough cases
   * relevant events occurred when needed (e.g., collisions for collision laws)
   * otherwise PASS becomes UNKNOWN (R10)
5. Harness is deterministic and reproducible across runs with the same seed/config (R12).
6. Adversarial search finds counterexamples for a seeded set of “false laws” more reliably than pure random testing (measurable in regression suite).

---

## 8. Logging and Audit Requirements

Harness must log per law evaluation:

* law hash + law_id
* selected test families + parameters
* seed and config(s)
* number of generated cases, precondition pass rate
* first failing witness (if any)
* power metrics summary
* minimization steps (if enabled)
* runtime and resource usage

Logs must be human-auditable and sufficient to reconstruct the evaluation.

---

## 9. Suggested Implementation PhasesTEST_

### Phase 1 (recommended MVP)

* claim compiler + predicate checker
* random + constrained generators
* metamorphic commutation tests
* UNKNOWN reason codes
* basic power metrics
* counterexample minimization (simple)

### Phase 2

* adversarial mutation search (hillclimb)
* small exhaustive enumeration for tiny N
* richer instrumentation (event logs)

### Phase 3 (optional)

* sandboxed test-code execution
* rival simulator discrimination scoring

---
