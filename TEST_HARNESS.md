# Test Harness Subsystem — Implementation Guide

This document describes the **Tester Harness** subsystem: the component that evaluates `CandidateLaw` objects by attempting falsification via simulation, returning **PASS / FAIL / UNKNOWN** verdicts with counterexamples and power metrics.

---

## Quick Reference

| Component | File | Purpose |
|-----------|------|---------|
| Main orchestrator | `src/harness/harness.py` | `Harness.evaluate(law) → LawVerdict` |
| Configuration | `src/harness/config.py` | `HarnessConfig` dataclass |
| Verdicts | `src/harness/verdict.py` | `LawVerdict`, `Counterexample`, `ReasonCode` |
| Evaluator | `src/harness/evaluator.py` | Per-case evaluation with precondition checking |
| Generators | `src/harness/generators/` | Case generation strategies (6 families) |
| Templates | `src/claims/templates.py` | 8 template checkers (Invariant, Monotone, etc.) |
| Power metrics | `src/harness/power.py` | `PowerMetrics` for coverage tracking |
| Adversarial | `src/harness/adversarial.py` | `AdversarialSearcher` for mutation-based search |
| Minimizer | `src/harness/minimizer.py` | Delta-debugging counterexample minimization |
| Witness | `src/harness/witness.py` | PHASE-E formatted witness capture |

---

## 1. Architecture Overview

```
Harness.evaluate(law: CandidateLaw) → LawVerdict
    │
    ├─→ _check_capabilities(law)     # Missing observables/transforms?
    ├─→ _evaluator.prepare(law)      # Compile law to TemplateChecker
    ├─→ _generate_cases(law)         # Multi-strategy case generation
    │
    ├─→ For each case:
    │     ├─→ is_valid_initial_state()  # No X at t=0
    │     ├─→ _evaluator.evaluate_case(case, T)
    │     ├─→ Track power metrics
    │     └─→ If failed: create Counterexample
    │
    ├─→ _adversarial.search()        # Mutation-based falsification
    ├─→ _minimizer.minimize()        # Reduce counterexample
    └─→ _determine_verdict()         # PASS/FAIL/UNKNOWN with reason
```

---

## 2. Inputs and Outputs

### Inputs

**CandidateLaw** (from `src/claims/schema.py`):
- `template`: One of 8 supported templates
- `quantifiers`: Time horizon T, eventually horizon H
- `preconditions`: Conditions required for test validity
- `observables`: Observable definitions
- `claim_ast`: Structured claim (AST-based evaluation)
- `proposed_tests`: Optional test hints from discovery
- `capability_requirements`: Missing observables/transforms/generators

**HarnessConfig** (from `src/harness/config.py`):
```python
@dataclass
class HarnessConfig:
    seed: int = 42
    max_runtime_ms_per_law: int = 5000
    max_cases: int = 300
    default_T: int = 50
    max_T: int = 200
    min_cases_used_for_pass: int = 50
    enable_adversarial_search: bool = True
    adversarial_budget: int = 1500
    enable_counterexample_minimization: bool = True
    minimization_budget: int = 300
    require_non_vacuous: bool = True
    min_antecedent_triggers: int = 20
    min_trigger_diversity: int = 2
    generator_weights: dict[str, float]  # See section 4
```

### Outputs

**LawVerdict** (from `src/harness/verdict.py`):
```python
@dataclass
class LawVerdict:
    law_id: str
    status: str  # "PASS", "FAIL", "UNKNOWN"
    reason_code: ReasonCode | None
    failure_type: FailureType | None
    counterexample: Counterexample | None
    power_metrics: PowerMetrics
    vacuity: VacuityReport
    runtime_ms: int
    tests_run: list[str]
    notes: list[str]
```

**Counterexample** (when FAIL):
```python
@dataclass
class Counterexample:
    initial_state: str           # e.g., "..><.."
    config: dict[str, Any]       # grid_length, boundary
    seed: int | None
    t_max: int                   # Total time simulated
    t_fail: int                  # Time step of violation
    trajectory_excerpt: list[str] | None
    observables_at_fail: dict[str, int] | None
    witness: dict[str, Any] | None
    minimized: bool
    # PHASE-E additions:
    formatted_witness: str | None
    observables_at_t: dict[str, Any] | None
    observables_at_t1: dict[str, Any] | None
    neighborhood_hash: str | None
```

---

## 3. Supported Law Templates

All templates are implemented in `src/claims/templates.py`:

| Template | Checker Class | Semantics |
|----------|---------------|-----------|
| `invariant` | `InvariantChecker` | `∀t∈[0..T]: f(t) == f(0)` |
| `monotone` | `MonotoneChecker` | `∀t∈[0..T-1]: f(t+1) ≤ f(t)` or `≥` |
| `implication_step` | `ImplicationStepChecker` | `∀t∈[0..T-1]: P(t) → Q(t+1)` |
| `implication_state` | `ImplicationStateChecker` | `∀t∈[0..T]: P(t) → Q(t)` |
| `eventually` | `EventuallyChecker` | `∀t0: P(t0) → ∃t∈[t0..t0+H]: Q(t)` |
| `symmetry_commutation` | `SymmetryCommutationChecker` | `evolve(T(S), t) == T(evolve(S, t))` |
| `bound` | `BoundChecker` | `∀t∈[0..T]: f(t) ≤ k` or `≥` |
| `local_transition` | `LocalTransitionChecker` | `∀t,i: state[i]==P → state[i+1] satisfies Q` |

### Template Compilation

The `ClaimCompiler` (`src/claims/compiler.py`) transforms `CandidateLaw` objects into `TemplateChecker` instances:

1. Validates template-specific required fields
2. Parses observable expressions via `ExprParser` (`src/claims/expr_parser.py`)
3. Constructs precondition predicates
4. Returns compiled checker or raises `CompilationError`

---

## 4. Case Generation

### Generator Registry

Generators are registered via `GeneratorRegistry` (`src/harness/generators/base.py`). Each generator implements:

```python
class Generator(ABC):
    @abstractmethod
    def generate(self, params: dict, seed: int, count: int) -> list[Case]
```

### Implemented Generators

| Family | File | Purpose | Default Weight |
|--------|------|---------|----------------|
| `random_density_sweep` | `random_density.py` | Random states at target densities | 30% |
| `constrained_pair_interactions` | `constrained_pairs.py` | Controlled particle pair setups | 20% |
| `edge_wrapping_cases` | `edge_wrapping.py` | Boundary-crossing scenarios | 15% |
| `symmetry_metamorphic_suite` | `symmetry_suite.py` | Metamorphic symmetry tests | 15% |
| `pathological_cases` | `pathological.py` | Uniform grids, alternating patterns | 10% |
| `extreme_states` | `extreme_states.py` | Max density, full collision grids | 10% |

### Default Generation Strategy

When a law provides no `proposed_tests`, the harness uses a multi-stage strategy:

1. **Pathological baseline** (always first): ~20 cases from `pathological_cases`
2. **Weighted distribution**: Remaining cases allocated by `generator_weights`

This ensures uniform grids (which vacuously satisfy many laws) are always tested.

---

## 5. Evaluation Flow

### Case Evaluation (`src/harness/evaluator.py`)

```python
def evaluate_case(self, case: Case, time_horizon: int) -> CaseResult:
    # 1. Check preconditions
    if not self._check_preconditions(case.initial_state):
        return CaseResult(passed=True, precondition_met=False, ...)

    # 2. Run simulation
    trajectory = run(case.initial_state, time_horizon)

    # 3. Check law against trajectory
    check_result = self._checker.check(trajectory)

    # 4. Return result with vacuity tracking
    return CaseResult(
        passed=check_result.passed,
        precondition_met=True,
        violation=check_result.violation,
        trajectory=trajectory,
        vacuity=check_result.vacuity,
    )
```

### Initial State Validation

Before evaluation, the harness validates that initial states conform to universe rules:
- No `X` cells at t=0 (collisions can only form during evolution)
- Invalid states are skipped and logged as generator bugs

---

## 6. Verdict Determination

The harness determines verdicts in `_determine_verdict()` (`src/harness/harness.py:391`):

### FAIL Conditions
- Counterexample found (via regular testing or adversarial search)
- Returns `FailureType.LAW_COUNTEREXAMPLE` with minimized counterexample

### UNKNOWN Conditions (with ReasonCodes)

| ReasonCode | Condition |
|------------|-----------|
| `MISSING_OBSERVABLE` | Law requires unavailable observable |
| `MISSING_TRANSFORM` | Law requires unavailable transform |
| `AMBIGUOUS_CLAIM` | Compilation failed |
| `UNMET_PRECONDITIONS` | `cases_used < min_cases_used_for_pass` (50) |
| `VACUOUS_PASS` | Implication antecedent never held |
| `INCONCLUSIVE_LOW_POWER` | `antecedent_true_count < 20` OR `trigger_diversity < 2` OR `coverage_score < 0.3` |

### PASS Conditions
All of the following must be true:
- No counterexample found
- `cases_used >= min_cases_used_for_pass` (50)
- For implication templates: non-vacuous with sufficient trigger diversity
- `coverage_score >= 0.3`
- For `shift_k` symmetry: diverse k values tested

---

## 7. Power Metrics

`PowerMetrics` (`src/harness/power.py`) tracks testing thoroughness:

```python
@dataclass
class PowerMetrics:
    cases_attempted: int = 0
    cases_used: int = 0                    # Preconditions satisfied
    cases_with_collisions: int = 0
    cases_with_wrapping: int = 0
    density_bins_hit: list[float] = []     # 0.0-1.0 in 0.1 increments
    violation_score_max: float = 0.0
    adversarial_cases_tried: int = 0
    adversarial_found: bool = False
    coverage_score: float = 0.0            # Computed weighted metric
```

### Coverage Score Calculation

```python
coverage_score = (
    0.3 * usage_ratio +        # cases_used / cases_attempted
    0.2 * collision_ratio +    # collision cases / used
    0.1 * wrapping_ratio +     # wrapping cases / used
    0.2 * density_coverage +   # density_bins_hit / 5
    0.2 * non_vacuous_ratio    # 1 - vacuous_cases / used
)
```

---

## 8. Vacuity Tracking

For implication templates (`implication_step`, `implication_state`, `eventually`, `local_transition`), the harness tracks vacuity via `VacuityReport` (`src/claims/vacuity.py`):

```python
@dataclass
class VacuityReport:
    total_checks: int = 0
    antecedent_true_count: int = 0
    consequent_evaluated_count: int = 0
    is_vacuous: bool = False
    trigger_diversity: int = 0             # Distinct generator families
    triggering_generators: set[str] = set()
    triggering_states: set[str] = set()    # Distinct triggering states
```

A PASS is downgraded to UNKNOWN if:
- `is_vacuous` is True (antecedent never held)
- `antecedent_true_count < min_antecedent_triggers` (default: 20)
- `trigger_diversity < min_trigger_diversity` (default: 2 generator families)

---

## 9. Adversarial Search

`AdversarialSearcher` (`src/harness/adversarial.py`) performs mutation-based counterexample search:

1. Seeds from cases that "almost" failed (high violation scores)
2. Applies mutations: symbol flips, insertions, deletions, shifts
3. Evaluates mutated states against the law
4. Terminates on counterexample or budget exhaustion

Configuration:
- `adversarial_budget`: 1500 mutations (default)
- `max_runtime_ms`: Half of `max_runtime_ms_per_law`

---

## 10. Counterexample Minimization

`Minimizer` (`src/harness/minimizer.py`) reduces counterexamples using delta-debugging:

1. **Grid reduction**: Remove cells while preserving failure
2. **Particle reduction**: Remove particles while preserving failure
3. **Horizon reduction**: Find earliest failing time step

Budget: `minimization_budget` (default: 300 attempts)

---

## 11. PHASE-E: Witness Capture

The harness captures formatted witnesses for human-readable failure analysis:

**Files**:
- `src/harness/witness.py`: `build_formatted_witness()`, `compute_neighborhood_hash()`

**Output Fields** (in `Counterexample`):
- `formatted_witness`: Human-readable violation description
- `observables_at_t`: Observable values at failure time
- `observables_at_t1`: Observable values at t+1
- `neighborhood_hash`: Content-based hash for diversity tracking

---

## 12. Persistence

When a `Repository` is provided, the harness persists:

1. **Case sets** (`case_sets` table): Generated test cases
2. **Evaluations** (`evaluations` table): Verdict + power metrics
3. **Counterexamples** (`counterexamples` table): Failure reproduction data
4. **Law witnesses** (`law_witnesses` table): PHASE-E formatted witnesses
5. **Audit log**: Operation tracking

---

## 13. Usage Example

```python
from src.harness.harness import Harness
from src.harness.config import HarnessConfig
from src.claims.schema import CandidateLaw

# Configure harness
config = HarnessConfig(
    seed=42,
    max_cases=300,
    min_cases_used_for_pass=50,
    enable_adversarial_search=True,
)

# Create harness
harness = Harness(config)

# Evaluate a law
verdict = harness.evaluate(law)

if verdict.status == "FAIL":
    print(f"Falsified at t={verdict.counterexample.t_fail}")
    print(f"Initial state: {verdict.counterexample.initial_state}")
elif verdict.status == "PASS":
    print(f"Passed with coverage {verdict.power_metrics.coverage_score:.2f}")
else:
    print(f"Unknown: {verdict.reason_code.value}")
    print(f"Notes: {verdict.notes}")
```

---

## 14. Determinism and Reproducibility

Given identical:
- `sim_hash` (universe simulator version)
- `HarnessConfig` (especially seed)
- `CandidateLaw`

The harness produces identical verdicts and counterexamples.

The harness logs:
- `sim_hash`: From `src/universe/simulator.version_hash()`
- `harness_config_hash`: From `HarnessConfig.content_hash()`
- All random seeds used

---

## 15. Acceptance Criteria

The harness is correct when:

1. **True laws PASS non-vacuously**: Known conservation laws pass with sufficient power
2. **False laws FAIL with counterexamples**: Known invalid laws are falsified with reproducible witnesses
3. **UNKNOWN is actionable**: Every UNKNOWN includes a reason code and specific notes
4. **Deterministic**: Same inputs → same outputs across runs
5. **Vacuity protected**: Implication laws require antecedent triggers from multiple generators

---

## 16. Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| All 8 templates | ✅ Complete | Including LOCAL_TRANSITION |
| 6 generator families | ✅ Complete | With configurable weights |
| Precondition enforcement | ✅ Complete | Cases skipped if unmet |
| Power metrics | ✅ Complete | Coverage score computed |
| Vacuity detection | ✅ Complete | Trigger diversity tracked |
| Adversarial search | ✅ Complete | Mutation-based |
| Counterexample minimization | ✅ Complete | Delta-debugging |
| PHASE-E witness capture | ✅ Complete | Formatted + neighborhood hash |
| Database persistence | ✅ Complete | All tables implemented |
| Reproducibility | ✅ Complete | Seeded RNG + version hashing |

---

## 17. Future Extensions (Not Implemented)

- **Sandboxed LLM test code**: Execute tester-provided Python in isolation
- **Rival simulator discrimination**: Test against rule variations
- **Distributed execution**: Parallel harness across machines
