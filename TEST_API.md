````markdown
# TEST_API.md — LLM Python Test Generator Interface

This document defines the **only** interface available to LLM-generated Python tests.

It is intended to be included (in condensed form) in the test-generator prompt, and used verbatim by the harness implementation.

## 1. Purpose

LLM-generated tests provide additional expressiveness when fixed claim templates are insufficient. The generated Python code must interact with the universe **only** through the `TestAPI` object described here.

**Key properties**
- Deterministic and replayable given the same inputs.
- Safe to run in a sandbox (no filesystem/network access assumed).
- Auditable: tests must return structured evidence and power metrics.

---

## 2. Execution Contract

Generated code must define exactly one function:

```python
def run_test(api) -> dict:
    """
    Run an experiment attempting to falsify a candidate law.

    Returns a dict following TEST_RESULT_SCHEMA.md (minimum required keys below).
    """
````

No work should happen at import time (no top-level execution). The harness will call `run_test(api)` once per run.

---

## 3. `TestAPI` Object

### 3.1 Determinism

The API is deterministic. Given the same:

* simulator version (`api.meta.sim_hash`)
* harness configuration (`api.meta.harness_config_hash`)
* run seed (passed to your RNG)
* initial states you generate

the outcomes must be identical.

### 3.2 Symbols and State Representation

A universe state is a string of length `N`, consisting only of:

* `.` empty cell
* `>` right-moving particle
* `<` left-moving particle
* `X` collision cell

Example:

```text
..><.X..
```

---

## 4. Simulation Methods

### `api.validate(state: str) -> None`

Validate that `state` is syntactically valid (correct length, allowed symbols). Raises `ValueError` on invalid states.

**Example**

```python
api.validate("..><..")
```

---

### `api.step(state: str) -> str`

Advance the universe by exactly one time step.

**Example**

```python
s1 = api.step(s0)
```

---

### `api.run(state: str, T: int) -> list[str]`

Run the universe for `T` steps and return the trajectory as a list of states.

* The returned list **includes the initial state** at index 0.
* Length of returned list is `T + 1`.

**Example**

```python
traj = api.run("..><..", T=5)
assert traj[0] == "..><.."
assert len(traj) == 6
```

---

## 5. Observables

### `api.count(state: str, symbol: str) -> int`

Count occurrences of a given symbol in `state`.

* `symbol` must be one of `.`, `>`, `<`, `X`.

**Example**

```python
x = api.count(state, "X")
r = api.count(state, ">")
l = api.count(state, "<")
```

---

### `api.grid_length(state: str) -> int`

Return the grid length `N` (length of the state string). Provided for clarity.

**Example**

```python
N = api.grid_length(state)
```

---

## 6. Transforms (Metamorphic Testing)

These transforms are useful for symmetry and metamorphic tests.

### `api.mirror_swap(state: str) -> str`

Reverse the string AND swap particle directions:

* `>` ↔ `<`
* `.` stays `.`
* `X` stays `X`

**Example**

```python
s2 = api.mirror_swap("..><.X..")
```

---

### `api.mirror_only(state: str) -> str`

Reverse the string without swapping directions.

**Example**

```python
s2 = api.mirror_only("..><..")
```

---

### `api.swap_only(state: str) -> str`

Swap particle directions without reversing:

* `>` ↔ `<`
* `.` stays `.`
* `X` stays `X`

**Example**

```python
s2 = api.swap_only("..><..")
```

---

### `api.shift_k(state: str, k: int) -> str`

Cyclically shift (rotate) the grid by `k` cells under periodic boundaries.

* Positive `k` shifts right, negative shifts left (implementation-defined but stable).
* Use to test translation invariance.

**Example**

```python
s2 = api.shift_k("..><..", k=2)
```

---

## 7. RNG and Case Generation Helpers

### `api.rng(seed: int)`

Returns a deterministic RNG object (similar to `random.Random`) scoped to your test.

**Example**

```python
rng = api.rng(12345)
x = rng.random()
i = rng.randrange(0, 10)
```

---

### `api.random_state(n: int, p_empty: float, p_right: float, p_left: float, p_X: float = 0.0, rng=None) -> str`

Generate a random state of length `n` with approximate per-cell probabilities.

* Probabilities should sum to ~1.0 (minor floating error tolerated).
* If `rng` is None, the API will use an internal deterministic RNG (not recommended; pass your own rng).
* Use `p_X` sparingly; tests should not assume semantic meaning of `X` beyond what is observed.

**Example**

```python
rng = api.rng(1)
s = api.random_state(n=12, p_empty=0.7, p_right=0.15, p_left=0.15, p_X=0.0, rng=rng)
api.validate(s)
```

---

### `api.constrained_case(kind: str, n: int, rng=None) -> str`

Generate a state designed to trigger particular interactions.

Supported `kind` strings (minimum set; may expand):

* `"empty"`: all `.`
* `"single_right"`: one `>` and rest `.`
* `"single_left"`: one `<` and rest `.`
* `"adjacent_pair"`: contains `><` adjacent somewhere (likely to form collision)
* `"collision_seed"`: contains an `X` somewhere
* `"edge_wrap"`: places a mover at an edge to exercise wrapping

**Example**

```python
rng = api.rng(7)
s = api.constrained_case("adjacent_pair", n=12, rng=rng)
```

---

### `api.mutate_state(state: str, edits: int, rng=None) -> str`

Randomly mutate a state by performing `edits` local changes (flip a cell symbol, swap two cells, etc.).

* The mutation operator is implementation-defined but stable.
* Useful for adversarial search and local neighborhood exploration.

**Example**

```python
rng = api.rng(9)
s2 = api.mutate_state(s, edits=3, rng=rng)
api.validate(s2)
```

---

## 8. Limits and Metadata

### `api.limits`

An object with enforced limits (read-only):

* `api.limits.max_cases` (int)
* `api.limits.max_T` (int)
* `api.limits.max_runtime_ms` (int)

Your test must respect these, and may adapt based on them.

**Example**

```python
T = min(20, api.limits.max_T)
```

---

### `api.meta`

Run metadata (read-only):

* `api.meta.universe_id` (str)
* `api.meta.sim_hash` (str)
* `api.meta.harness_config_hash` (str)

Use these only for reporting, not branching logic.

---

## 9. Assertion Helpers (Optional but Recommended)

### `api.assert_true(cond: bool, msg: str = "") -> None`

Raises `AssertionError` if `cond` is false.

### `api.assert_equal(a, b, msg: str = "") -> None`

Raises `AssertionError` if `a != b`.

These are convenience helpers; you may also raise exceptions directly.

---

## 10. Required Return Structure (Minimum)

Your returned dict must include at least:

```python
{
  "status": "PASS" | "FAIL" | "UNKNOWN",
  "reason_code": "<short_machine_readable_reason>",
  "evidence": { ... },
  "power_metrics": {
      "vacuous": <bool>,
      "cases_attempted": <int>,
      "cases_used": <int>,
      "precondition_hits": <int>
  },
  "counterexample": None | {
      "initial_state": "<state string>",
      "seed": <int>,
      "T": <int>,
      "t_fail": <int>,
      "witness": { ... }
  }
}
```

### Vacuity rule

If `precondition_hits == 0`, you must set:

* `power_metrics.vacuous = True`
* `status = "UNKNOWN"`
* `reason_code = "inconclusive_vacuous"` (or similar)

### FAIL rule

If `status == "FAIL"`, `counterexample` must be non-null.

---

## 11. Minimal Example Test

This example checks a simple invariant over multiple random cases.

```python
def run_test(api):
    seed = 123
    rng = api.rng(seed)

    max_cases = min(50, api.limits.max_cases)
    T = min(20, api.limits.max_T)

    cases_attempted = 0
    cases_used = 0
    precondition_hits = 0

    # Example property: count('X') is always <= grid length (a trivial bound)
    for _ in range(max_cases):
        cases_attempted += 1
        s0 = api.random_state(n=12, p_empty=0.7, p_right=0.15, p_left=0.15, p_X=0.0, rng=rng)
        api.validate(s0)

        traj = api.run(s0, T=T)
        cases_used += 1
        precondition_hits += 1  # no preconditions in this example

        N = api.grid_length(s0)
        for t, st in enumerate(traj):
            if api.count(st, "X") > N:
                return {
                    "status": "FAIL",
                    "reason_code": "bound_violated",
                    "evidence": {"N": N, "t": t, "x": api.count(st, "X")},
                    "power_metrics": {
                        "vacuous": False,
                        "cases_attempted": cases_attempted,
                        "cases_used": cases_used,
                        "precondition_hits": precondition_hits
                    },
                    "counterexample": {
                        "initial_state": s0,
                        "seed": seed,
                        "T": T,
                        "t_fail": t,
                        "witness": {"state": st, "x": api.count(st, "X"), "N": N}
                    }
                }

    return {
        "status": "PASS",
        "reason_code": "no_counterexample_found",
        "evidence": {"checked_cases": cases_used, "T": T},
        "power_metrics": {
            "vacuous": (precondition_hits == 0),
            "cases_attempted": cases_attempted,
            "cases_used": cases_used,
            "precondition_hits": precondition_hits
        },
        "counterexample": None
    }
```

---

## 12. What Not To Do

Generated tests must NOT:

* import modules
* access filesystem or network
* call `eval`, `exec`, or `__import__`
* rely on timing or nondeterminism
* return PASS without reporting power metrics
* return FAIL without a counterexample witness

The harness may reject tests that violate these rules.

---

```
```
