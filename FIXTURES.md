Here’s a **starter fixtures pack** (law JSON objects + a simple regression runner contract) assuming **states are strings** like `"..><.X.."`.

I’m going to give you:

1. **Fixture file layout** (what to commit)
2. **Canonical derived observables** you should support in the harness expression language
3. **Law fixtures** for T/F/U (as JSON you can drop into files)
4. **Expected outcomes** for the regression runner

I’ll keep the fixtures strictly within the claim-template vocabulary from the requirements docs.

---

## 1) Suggested fixture layout

```
tests/
  fixtures/
    universe_contract.json
    harness_config.json
    laws_true.json
    laws_false.json
    laws_unknown.json
    expectations.json
```

---

## 2) Canonical observable defs (needed by harness)

Your harness should allow `observables[].expr` with:

* `count('<symbol>')` where symbol is one of `. > < X`
* numeric ops `+ - *`
* parentheses

Derived observables used in fixtures:

* `R_total = count('>') + count('X')`
* `L_total = count('<') + count('X')`
* `particle_count = count('>') + count('<') + 2*count('X')`
* `momentum = (count('>') + count('X')) - (count('<') + count('X'))`  (simplifies to `count('>') - count('<')`, but keep explicit)
* `occupied_cells = count('>') + count('<') + count('X')`
* `collision_count = count('X')`

---

## 3) Fixtures

### 3.1 `universe_contract.json`

```json
{
  "schema_version": "1.0.0",
  "universe_id": "kinetic_grid_v1",
  "symbols": [".", ">", "<", "X"],
  "state_representation": "string",
  "capabilities": {
    "primitive_observables": ["count(symbol)", "grid_length"],
    "derived_observables_allowed": true,
    "transforms": ["mirror_swap", "mirror_only", "swap_only", "shift_k"],
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

### 3.2 `harness_config.json`

```json
{
  "schema_version": "1.0.0",
  "seed": 1337,
  "max_runtime_ms_per_law": 2500,
  "max_cases": 300,
  "default_T": 50,
  "max_T": 200,
  "min_cases_used_for_pass": 50,
  "enable_adversarial_search": true,
  "adversarial_budget": 1500,
  "enable_counterexample_minimization": true,
  "minimization_budget": 300,
  "store_full_trajectories": false,
  "require_non_vacuous": true
}
```

---

## 4) True laws (`laws_true.json`)

These should PASS given your described universe.

```json
[
  {
    "schema_version": "1.0.0",
    "law_id": "T1_right_component_conservation",
    "template": "invariant",
    "quantifiers": { "T": 60 },
    "preconditions": [
      { "lhs": "grid_length", "op": ">=", "rhs": 4 }
    ],
    "observables": [
      { "name": "R_total", "expr": "count('>') + count('X')" }
    ],
    "claim": "R_total(t) == R_total(0)",
    "forbidden": "exists t where R_total(t) != R_total(0)",
    "proposed_tests": [
      { "family": "random_density_sweep", "params": { "cases": 150, "densities": [0.1, 0.3, 0.6] } },
      { "family": "constrained_pair_interactions", "params": { "cases": 50 } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": [], "missing_generators": [] }
  },
  {
    "schema_version": "1.0.0",
    "law_id": "T2_left_component_conservation",
    "template": "invariant",
    "quantifiers": { "T": 60 },
    "preconditions": [
      { "lhs": "grid_length", "op": ">=", "rhs": 4 }
    ],
    "observables": [
      { "name": "L_total", "expr": "count('<') + count('X')" }
    ],
    "claim": "L_total(t) == L_total(0)",
    "forbidden": "exists t where L_total(t) != L_total(0)",
    "proposed_tests": [
      { "family": "random_density_sweep", "params": { "cases": 150, "densities": [0.1, 0.3, 0.6] } },
      { "family": "constrained_pair_interactions", "params": { "cases": 50 } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": [], "missing_generators": [] }
  },
  {
    "schema_version": "1.0.0",
    "law_id": "T3_particle_count_conservation",
    "template": "invariant",
    "quantifiers": { "T": 60 },
    "preconditions": [
      { "lhs": "grid_length", "op": ">=", "rhs": 4 }
    ],
    "observables": [
      { "name": "particle_count", "expr": "count('>') + count('<') + 2*count('X')" }
    ],
    "claim": "particle_count(t) == particle_count(0)",
    "forbidden": "exists t where particle_count(t) != particle_count(0)",
    "proposed_tests": [
      { "family": "random_density_sweep", "params": { "cases": 200, "densities": [0.1, 0.3, 0.6] } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": [], "missing_generators": [] }
  },
  {
    "schema_version": "1.0.0",
    "law_id": "T4_momentum_conservation",
    "template": "invariant",
    "quantifiers": { "T": 60 },
    "preconditions": [
      { "lhs": "grid_length", "op": ">=", "rhs": 4 }
    ],
    "observables": [
      {
        "name": "momentum",
        "expr": "(count('>') + count('X')) - (count('<') + count('X'))"
      }
    ],
    "claim": "momentum(t) == momentum(0)",
    "forbidden": "exists t where momentum(t) != momentum(0)",
    "proposed_tests": [
      { "family": "random_density_sweep", "params": { "cases": 200, "densities": [0.1, 0.3, 0.6] } },
      { "family": "constrained_pair_interactions", "params": { "cases": 50 } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": [], "missing_generators": [] }
  },
  {
    "schema_version": "1.0.0",
    "law_id": "T5_full_mirror_symmetry_commutation",
    "template": "symmetry_commutation",
    "quantifiers": { "T": 40 },
    "preconditions": [
      { "lhs": "grid_length", "op": ">=", "rhs": 4 }
    ],
    "observables": [],
    "claim": "commutes(transform='mirror_swap')",
    "forbidden": "exists t where evolve(mirror_swap(S), t) != mirror_swap(evolve(S, t))",
    "proposed_tests": [
      { "family": "symmetry_metamorphic_suite", "params": { "cases": 200, "transform": "mirror_swap" } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": [], "missing_generators": [] }
  }
]
```

> Note: for symmetry laws I used `claim: "commutes(transform='mirror_swap')"` as a convention. Your claim compiler can treat `symmetry_commutation` templates as special and ignore `claim` in favor of structured fields (if you prefer, add `transform: "mirror_swap"` to the schema). The `forbidden` is still explicit.

---

## 5) False laws (`laws_false.json`)

These should FAIL with counterexamples (small ones exist).

```json
[
  {
    "schema_version": "1.0.0",
    "law_id": "F1_collision_count_conserved",
    "template": "invariant",
    "quantifiers": { "T": 10 },
    "preconditions": [
      { "lhs": "grid_length", "op": ">=", "rhs": 4 }
    ],
    "observables": [
      { "name": "collision_count", "expr": "count('X')" }
    ],
    "claim": "collision_count(t) == collision_count(0)",
    "forbidden": "exists t where collision_count(t) != collision_count(0)",
    "proposed_tests": [
      { "family": "constrained_pair_interactions", "params": { "cases": 50, "include_collision_state": true } },
      { "family": "adversarial_mutation_search", "params": { "budget": 500 } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": [], "missing_generators": [] }
  },
  {
    "schema_version": "1.0.0",
    "law_id": "F2_occupied_cells_conserved",
    "template": "invariant",
    "quantifiers": { "T": 10 },
    "preconditions": [
      { "lhs": "grid_length", "op": ">=", "rhs": 4 }
    ],
    "observables": [
      { "name": "occupied_cells", "expr": "count('>') + count('<') + count('X')" }
    ],
    "claim": "occupied_cells(t) == occupied_cells(0)",
    "forbidden": "exists t where occupied_cells(t) != occupied_cells(0)",
    "proposed_tests": [
      { "family": "constrained_pair_interactions", "params": { "cases": 50, "include_collision_state": true } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": [], "missing_generators": [] }
  },
  {
    "schema_version": "1.0.0",
    "law_id": "F3_spatial_mirror_only_is_symmetry",
    "template": "symmetry_commutation",
    "quantifiers": { "T": 20 },
    "preconditions": [
      { "lhs": "grid_length", "op": ">=", "rhs": 4 }
    ],
    "observables": [],
    "claim": "commutes(transform='mirror_only')",
    "forbidden": "exists t where evolve(mirror_only(S), t) != mirror_only(evolve(S, t))",
    "proposed_tests": [
      { "family": "symmetry_metamorphic_suite", "params": { "cases": 200, "transform": "mirror_only" } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": [], "missing_generators": [] }
  },
  {
    "schema_version": "1.0.0",
    "law_id": "F4_collision_implies_occupied_nonincrease",
    "template": "implication_step",
    "quantifiers": { "T": 15 },
    "preconditions": [
      { "lhs": "grid_length", "op": ">=", "rhs": 4 }
    ],
    "observables": [
      { "name": "collision_count", "expr": "count('X')" },
      { "name": "occupied_cells", "expr": "count('>') + count('<') + count('X')" }
    ],
    "claim": "collision_count(t) > 0 -> occupied_cells(t+1) <= occupied_cells(t)",
    "forbidden": "exists t where collision_count(t) > 0 and occupied_cells(t+1) > occupied_cells(t)",
    "proposed_tests": [
      { "family": "constrained_pair_interactions", "params": { "cases": 80, "include_collision_state": true } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": [], "missing_generators": [] }
  }
]
```

---

## 6) Unknown laws (`laws_unknown.json`)

These should return UNKNOWN with specific reason codes, regardless of the simulator, because they require capabilities you likely will not implement at first.

```json
[
  {
    "schema_version": "1.0.0",
    "law_id": "U1_requires_missing_observable_collision_events",
    "template": "bound",
    "quantifiers": { "T": 20 },
    "preconditions": [
      { "lhs": "grid_length", "op": ">=", "rhs": 4 }
    ],
    "observables": [
      { "name": "collision_events", "expr": "collision_events" }
    ],
    "claim": "collision_events(t) <= 100",
    "forbidden": "exists t where collision_events(t) > 100",
    "proposed_tests": [
      { "family": "random_density_sweep", "params": { "cases": 50, "densities": [0.3, 0.6] } }
    ],
    "capability_requirements": { "missing_observables": ["collision_events"], "missing_transforms": [], "missing_generators": [] }
  },
  {
    "schema_version": "1.0.0",
    "law_id": "U2_requires_missing_transform_rotate_2d",
    "template": "symmetry_commutation",
    "quantifiers": { "T": 10 },
    "preconditions": [],
    "observables": [],
    "claim": "commutes(transform='rotate_2d')",
    "forbidden": "exists t where evolve(rotate_2d(S), t) != rotate_2d(evolve(S, t))",
    "proposed_tests": [
      { "family": "symmetry_metamorphic_suite", "params": { "cases": 10, "transform": "rotate_2d" } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": ["rotate_2d"], "missing_generators": [] }
  },
  {
    "schema_version": "1.0.0",
    "law_id": "U3_ambiguous_claim_uses_indexing",
    "template": "implication_state",
    "quantifiers": { "T": 10 },
    "preconditions": [],
    "observables": [
      { "name": "cell0", "expr": "state[0]" }
    ],
    "claim": "cell0(t) == '.' -> cell0(t) == '.'",
    "forbidden": "exists t where cell0(t) == '.' and cell0(t) != '.'",
    "proposed_tests": [
      { "family": "random_density_sweep", "params": { "cases": 10, "densities": [0.3] } }
    ],
    "capability_requirements": { "missing_observables": ["cell0"], "missing_transforms": [], "missing_generators": [] }
  },
  {
    "schema_version": "1.0.0",
    "law_id": "U4_unmet_preconditions_impossible_grid",
    "template": "invariant",
    "quantifiers": { "T": 10 },
    "preconditions": [
      { "lhs": "grid_length", "op": "<", "rhs": 0 }
    ],
    "observables": [
      { "name": "occupied_cells", "expr": "count('>') + count('<') + count('X')" }
    ],
    "claim": "occupied_cells(t) == occupied_cells(0)",
    "forbidden": "exists t where occupied_cells(t) != occupied_cells(0)",
    "proposed_tests": [
      { "family": "random_density_sweep", "params": { "cases": 10, "densities": [0.3] } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": [], "missing_generators": [] }
  },
  {
    "schema_version": "1.0.0",
    "law_id": "U5_resource_limit_huge_T",
    "template": "invariant",
    "quantifiers": { "T": 1000000 },
    "preconditions": [
      { "lhs": "grid_length", "op": ">=", "rhs": 4 }
    ],
    "observables": [
      { "name": "particle_count", "expr": "count('>') + count('<') + 2*count('X')" }
    ],
    "claim": "particle_count(t) == particle_count(0)",
    "forbidden": "exists t where particle_count(t) != particle_count(0)",
    "proposed_tests": [
      { "family": "random_density_sweep", "params": { "cases": 10, "densities": [0.3] } }
    ],
    "capability_requirements": { "missing_observables": [], "missing_transforms": [], "missing_generators": [] }
  }
]
```

---

## 7) Expectations (`expectations.json`)

This tells your regression runner what to assert.

```json
{
  "schema_version": "1.0.0",
  "expected": [
    { "law_id": "T1_right_component_conservation", "status": "PASS" },
    { "law_id": "T2_left_component_conservation", "status": "PASS" },
    { "law_id": "T3_particle_count_conservation", "status": "PASS" },
    { "law_id": "T4_momentum_conservation", "status": "PASS" },
    { "law_id": "T5_full_mirror_symmetry_commutation", "status": "PASS" },

    { "law_id": "F1_collision_count_conserved", "status": "FAIL" },
    { "law_id": "F2_occupied_cells_conserved", "status": "FAIL" },
    { "law_id": "F3_spatial_mirror_only_is_symmetry", "status": "FAIL" },
    { "law_id": "F4_collision_implies_occupied_nonincrease", "status": "FAIL" },

    { "law_id": "U1_requires_missing_observable_collision_events", "status": "UNKNOWN", "reason_code": "missing_observable" },
    { "law_id": "U2_requires_missing_transform_rotate_2d", "status": "UNKNOWN", "reason_code": "missing_transform" },
    { "law_id": "U3_ambiguous_claim_uses_indexing", "status": "UNKNOWN", "reason_code": "missing_observable" },
    { "law_id": "U4_unmet_preconditions_impossible_grid", "status": "UNKNOWN", "reason_code": "unmet_preconditions" },
    { "law_id": "U5_resource_limit_huge_T", "status": "UNKNOWN", "reason_code": "resource_limit" }
  ],

  "pass_requirements": {
    "min_cases_used": 50,
    "require_non_vacuous": true
  },

  "fail_requirements": {
    "require_counterexample": true,
    "minimized_or_equal": true
  }
}
```

---

## 8) A couple of implementation notes (so your coding agent doesn’t get stuck)

### A) Handling `f(t+1)` in expressions

Some fixtures use `occupied_cells(t+1)` in the implication templates. Your claim compiler should allow:

* `observable(t)` and `observable(t+1)` references **only** for implication_step and monotone templates.

Internally, easiest is:

* compile implication_step into a function that loops t in `[0..T-1]` and evaluates expressions against `obs[t]` and `obs[t+1]`.

### B) Constrained generator parameter `include_collision_state`

For F1/F2/F4, ensure the generator can produce initial states containing `X` (since those falsify collision-count and occupied invariants quickly). If your natural dynamics never has `X` at t=0, you still want this for regression.

If you don’t like allowing X in initial states, replace those false laws with adjacency setups that create an X at t=1. But allowing `X` at t=0 is a cleaner regression hammer.

### C) Symmetry commutation claim encoding

I used a “string convention” for `claim`. If you prefer, extend schema with:

* `transform: "mirror_swap"`
  and ignore `claim`/`forbidden` for symmetry template (but keep forbidden in output logs).

---

If you’d like, I can also sketch a **tiny reference implementation** of:

* `mirror_swap(state_str)`
* `mirror_only(state_str)`
* `swap_only(state_str)`
  and a minimal constrained generator for `><` and `X` injection, purely as pseudo-code to align the team/coding agent.
