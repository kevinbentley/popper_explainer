# Claims Subsystem — Theory and Implementation Guide

This document describes the **Claims** subsystem: the module that defines, parses, compiles, and evaluates candidate laws in the Popperian discovery system.

---

## Quick Reference

| Component | File | Purpose |
|-----------|------|---------|
| Schema | `src/claims/schema.py` | Pydantic models: `CandidateLaw`, `Template`, `Observable`, etc. |
| Templates | `src/claims/templates.py` | 8 template checkers: `InvariantChecker`, `MonotoneChecker`, etc. |
| Compiler | `src/claims/compiler.py` | `ClaimCompiler`: transforms `CandidateLaw` → `TemplateChecker` |
| Expression AST | `src/claims/expr_ast.py` | AST node types: `Count`, `Literal`, `BinOp`, etc. |
| Expression Parser | `src/claims/expr_parser.py` | `ExpressionParser`: string → AST |
| Expression Evaluator | `src/claims/expr_evaluator.py` | `evaluate_expression()`: AST × State → int |
| Claim AST | `src/claims/ast_schema.py` | Structured JSON claim ASTs with validation |
| AST Evaluator | `src/claims/ast_evaluator.py` | `ASTClaimEvaluator`: evaluates structured claims |
| Fingerprint | `src/claims/fingerprint.py` | Semantic fingerprinting for deduplication |
| Semantic Linter | `src/claims/semantic_linter.py` | `SemanticLinter`: detects mismatches |
| Quantity Types | `src/claims/quantity_types.py` | Type inference for observables |
| Vacuity | `src/claims/vacuity.py` | `VacuityReport`: tracks test vacuity |

---

## 1. Theoretical Background

### Popperian Falsifiability

A law in this system must be **falsifiable**: there must exist an observable condition that, if witnessed, would refute the law. The claims subsystem encodes this by requiring:

1. **Template**: A structured pattern (invariant, implication, etc.) with clear checking semantics
2. **Observables**: Measurable quantities on universe states
3. **Preconditions**: When the law is applicable
4. **Forbidden condition**: What constitutes a counterexample

### Claim Templates

Rather than allowing arbitrary logical expressions, laws must fit one of 8 **templates**. This constraint ensures:
- Every law has well-defined checking semantics
- Vacuity can be detected (for implications)
- Power metrics can be computed
- Counterexamples are interpretable

---

## 2. Core Data Structure: CandidateLaw

**Location:** `src/claims/schema.py`

The `CandidateLaw` Pydantic model is the central data structure:

```python
class CandidateLaw(BaseModel):
    # Identity
    schema_version: str = "1.0.0"
    law_id: str                          # Unique identifier

    # Core structure
    template: Template                   # Which of 8 templates
    quantifiers: Quantifiers             # T (time horizon), H (eventuality horizon)

    # Semantic content
    preconditions: list[Precondition]    # When law applies
    observables: list[Observable]        # Named measurements
    claim: str                           # Human-readable claim
    forbidden: str                       # What constitutes counterexample

    # Structured AST (preferred)
    claim_ast: dict[str, Any] | None     # JSON AST for unambiguous semantics

    # Template-specific fields
    transform: str | None                # For symmetry_commutation
    direction: MonotoneDirection | None  # For monotone (>= or <=)
    bound_value: int | None              # For bound template
    bound_op: ComparisonOp | None        # For bound template
    trigger_symbol: str | None           # For local_transition
    result_op: ComparisonOp | None       # For local_transition
    result_symbol: str | None            # For local_transition

    # Metadata
    proposed_tests: list[ProposedTest]   # Suggested test families
    capability_requirements: CapabilityRequirements
    distinguishes_from: list[str]        # Rival hypotheses
    novelty_claim: str | None
    ranking_features: RankingFeatures    # For prioritization
```

### Key Methods

- `get_observable_names() → set[str]`: Returns defined observable names
- `content_hash() → str`: SHA256 hash (16 chars) for duplicate detection

---

## 3. Templates and Their Semantics

**Location:** `src/claims/schema.py` (enum), `src/claims/templates.py` (checkers)

| Template | Enum Value | Formal Semantics | Checker Class |
|----------|------------|------------------|---------------|
| Invariant | `invariant` | `∀t∈[0..T]: f(t) == f(0)` | `InvariantChecker` |
| Monotone | `monotone` | `∀t∈[0..T-1]: f(t+1) ≤ f(t)` (or `≥`) | `MonotoneChecker` |
| Implication (step) | `implication_step` | `∀t∈[0..T-1]: P(t) → Q(t+1)` | `ImplicationStepChecker` |
| Implication (state) | `implication_state` | `∀t∈[0..T]: P(t) → Q(t)` | `ImplicationStateChecker` |
| Eventually | `eventually` | `∀t0: P(t0) → ∃t∈[t0..t0+H]: Q(t)` | `EventuallyChecker` |
| Symmetry | `symmetry_commutation` | `evolve(T(S), t) == T(evolve(S, t))` | `SymmetryCommutationChecker` |
| Bound | `bound` | `∀t∈[0..T]: f(t) op k` | `BoundChecker` |
| Local Transition | `local_transition` | `∀t,i: state[i]==P → state[i] at t+1 satisfies Q` | `LocalTransitionChecker` |

### Template-Specific Required Fields

| Template | Required Fields |
|----------|-----------------|
| `invariant` | 1 observable |
| `monotone` | 1 observable, `direction` |
| `bound` | 1 observable, `bound_value`, `bound_op` |
| `implication_step` | ≥1 observable, claim with `->` |
| `implication_state` | ≥1 observable, claim with `->` |
| `eventually` | ≥1 observable, `quantifiers.H`, claim with `->` |
| `symmetry_commutation` | `transform` |
| `local_transition` | `trigger_symbol`, `result_op`, `result_symbol` |

---

## 4. Observables and the Expression Language

### Observable Definition

```python
class Observable(BaseModel):
    name: str   # e.g., "R_total"
    expr: str   # e.g., "count('>') + count('X')"
```

### Expression Language

**Location:** `src/claims/expr_ast.py`, `src/claims/expr_parser.py`

The expression language supports measuring state properties:

| Function | Syntax | Description |
|----------|--------|-------------|
| Count | `count('<symbol>')` | Count occurrences of symbol |
| Grid length | `grid_length` | Length of state string |
| Leftmost | `leftmost('<symbol>')` | Position of first occurrence (-1 if none) |
| Rightmost | `rightmost('<symbol>')` | Position of last occurrence (-1 if none) |
| Max gap | `max_gap('<symbol>')` | Longest contiguous run |
| Adjacent pairs | `adjacent_pairs('<s1>', '<s2>')` | Count of s1 immediately followed by s2 |
| Gap pairs | `gap_pairs('<s1>', '<s2>', gap)` | Count with `gap` cells between |
| Incoming collisions | `incoming_collisions` | Cells that will collide at t+1 |
| Spread | `spread('<symbol>')` | rightmost - leftmost position |

**Valid Symbols:** `.` (empty), `>` (right mover), `<` (left mover), `X` (collision)

**Operators:** `+`, `-`, `*` with standard precedence

### AST Node Types

```python
# Primitives
Literal(value: int)           # Integer constant
Count(symbol: str)            # count('>')
GridLength()                  # grid_length
IncomingCollisions()          # incoming_collisions

# Position
Leftmost(symbol: str)         # leftmost('X')
Rightmost(symbol: str)        # rightmost('>')

# Derived
MaxGap(symbol: str)           # max_gap('.')
Spread(symbol: str)           # spread('>')
AdjacentPairs(s1: str, s2: str)
GapPairs(s1: str, s2: str, gap: int)

# Composition
BinOp(left: Expr, op: Operator, right: Expr)
```

### Expression Evaluation

**Location:** `src/claims/expr_evaluator.py`

```python
def evaluate_expression(expr: Expr, state: State) -> int:
    """Evaluate an expression AST against a universe state."""
```

Evaluation is pure and deterministic—same expression + state always yields same integer.

---

## 5. Preconditions

Preconditions restrict when a law applies. Cases not satisfying preconditions don't count as evidence.

```python
class Precondition(BaseModel):
    lhs: str              # Observable name or "grid_length"
    op: ComparisonOp      # ==, !=, <, <=, >, >=
    rhs: int | str        # Value or observable name
```

### Example Preconditions

```python
# Grid must have at least 5 cells
Precondition(lhs="grid_length", op=">=", rhs=5)

# Must have at least one right-mover
Precondition(lhs="R_count", op=">", rhs=0)

# R and L counts must be equal
Precondition(lhs="R_count", op="==", rhs="L_count")
```

### Precondition Compilation

**Location:** `src/claims/compiler.py:323`

```python
def compile_precondition(precondition: Precondition, law: CandidateLaw) -> Callable[[State], bool]:
    """Compile a precondition into a callable predicate."""
```

Preconditions are evaluated on the **initial state only**, before simulation.

---

## 6. The Claim Compiler

**Location:** `src/claims/compiler.py`

The `ClaimCompiler` transforms a `CandidateLaw` into an executable `TemplateChecker`:

```python
class ClaimCompiler:
    def compile(self, law: CandidateLaw) -> TemplateChecker:
        # 1. Parse all observable expressions
        for obs in law.observables:
            expr = self._parser.parse(obs.expr)
            self._compiled_observables[obs.name] = expr

        # 2. Dispatch to template-specific compiler
        if law.template == Template.INVARIANT:
            return self._compile_invariant(law)
        elif law.template == Template.MONOTONE:
            return self._compile_monotone(law)
        # ... etc for each template
```

### Compilation Errors

`CompilationError` is raised for:
- Unknown template
- Missing required fields (e.g., `direction` for monotone)
- Observable parse failure
- Invalid claim format (e.g., implication without `->`)
- Unknown transform name

---

## 7. Template Checkers

**Location:** `src/claims/templates.py`

All checkers implement:

```python
class TemplateChecker(ABC):
    @abstractmethod
    def check(self, trajectory: Trajectory) -> CheckResult:
        """Check the law against a trajectory."""
        pass

@dataclass
class CheckResult:
    passed: bool
    violation: Violation | None = None
    vacuity: VacuityReport = field(default_factory=VacuityReport)

@dataclass
class Violation:
    t: int                          # Time step of violation
    state: State                    # State at time t
    details: dict[str, Any] = {}    # Observable values, etc.
    message: str = ""               # Human-readable description
```

### Checker Details

#### InvariantChecker
- Evaluates observable at t=0, then at each subsequent t
- Violation if any `f(t) != f(0)`
- Returns first divergence point

#### MonotoneChecker
- Evaluates at consecutive timesteps
- Checks `f(t+1) <= f(t)` (or `>=` based on direction)
- Violation includes prev/curr values

#### BoundChecker
- Evaluates observable at each timestep
- Checks `f(t) op bound_value`
- Supports all comparison operators

#### ImplicationStepChecker
- For each t, if `antecedent(state[t])` is true, checks `consequent(state[t+1])`
- Tracks vacuity (when antecedent never holds)
- Returns `VacuityReport` with trigger counts

#### ImplicationStateChecker
- Same as step, but checks `consequent(state[t])` (same timestep)
- Also tracks vacuity

#### EventuallyChecker
- For each t0 where antecedent holds, searches `[t0..t0+H]` for consequent
- Violation includes horizon and search bounds
- Tracks trigger diversity

#### SymmetryCommutationChecker
- Computes two paths: `evolve(transform(S))` and `transform(evolve(S))`
- Compares at each timestep
- Supports parameterized transforms (e.g., `shift_k` with custom k)

#### LocalTransitionChecker
- For each position i where `state[i] == trigger_symbol`
- Checks `state[i]` at t+1 satisfies result condition
- Tracks per-position violations

---

## 8. Structured Claim ASTs

**Location:** `src/claims/ast_schema.py`, `src/claims/ast_evaluator.py`

For unambiguous claim semantics, use structured JSON ASTs instead of string claims.

### AST Node Types

```json
// Constant
{"const": 5}

// Variable (time)
{"var": "t"}

// Observable at time
{"obs": "N", "t": {"var": "t"}}
{"obs": "R_total", "t": {"const": 0}}    // At t=0
{"obs": "collisions", "t": "t_plus_1"}   // Shorthand for t+1

// Binary operation
{"op": "+", "lhs": {...}, "rhs": {...}}
{"op": "==", "lhs": {...}, "rhs": {...}}
{"op": "=>", "lhs": {...}, "rhs": {...}}  // Implication
```

### Supported Operators

- Arithmetic: `+`, `-`, `*`, `/`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Logical: `=>` (implication), `and`, `or`, `not`

### AST Evaluator

```python
class ASTClaimEvaluator:
    def __init__(self, law: CandidateLaw):
        # Compile observable expressions

    def check(self, trajectory: Trajectory) -> tuple[bool, int | None, dict, VacuityReport]:
        # Dispatch by template type
```

---

## 9. Vacuity Tracking

**Location:** `src/claims/vacuity.py`

For implication templates, a test is **vacuous** if the antecedent is never true.

```python
@dataclass
class VacuityReport:
    total_checks: int = 0              # Time steps examined
    antecedent_true_count: int = 0     # Times P was true
    consequent_evaluated_count: int = 0 # Times Q was checked
    is_vacuous: bool = False           # True iff antecedent never held
    trigger_diversity: int = 0         # Distinct generator families
    triggering_generators: set[str]    # Which generators triggered
    triggering_states: set[str]        # Which initial states triggered
```

### Vacuity Detection Logic

A PASS is downgraded to UNKNOWN if:
- `is_vacuous == True` (antecedent never held)
- `antecedent_true_count < min_antecedent_triggers` (default: 20)
- `trigger_diversity < min_trigger_diversity` (default: 2)

---

## 10. Fingerprinting and Deduplication

**Location:** `src/claims/fingerprint.py`

Semantic fingerprinting detects when two laws are equivalent despite different names.

### Key Functions

```python
def compute_semantic_fingerprint(law: CandidateLaw) -> str:
    """Compute 24-char hex fingerprint for semantic deduplication."""

def canonicalize_ast(ast: dict) -> dict:
    """Normalize AST for comparison (sort commutative ops, etc.)."""

def canonicalize_observable_expr(expr: str) -> str:
    """Normalize expression string (whitespace, quotes, case)."""

def extract_observable_semantics(obs: Observable) -> str:
    """Map observable to canonical semantic identifier."""
```

### Canonical Forms

```python
CANONICAL_FORMS = {
    "count('>')": "count_right",
    "count('<')": "count_left",
    "count('x')": "count_collision",
    "count('>') + count('<')": "count_movers",
    "count('>') - count('<')": "momentum",
    "count('>')+count('<')+2*count('x')": "total_particles",
    "grid_length": "grid_length",
}
```

### Canonicalization Rules

1. Sort operands of commutative operators (`+`, `*`, `and`, `or`, `==`, `!=`)
2. Normalize comparisons (larger expression on left)
3. Algebraic normalization: `(X*2) <= Y` → `X <= (Y/2)`
4. Recursive processing of sub-expressions

---

## 11. Semantic Linter

**Location:** `src/claims/semantic_linter.py`

Detects mismatches between law names and their actual semantics.

```python
class SemanticLinter:
    def lint(self, law: CandidateLaw) -> LintResult:
        """Check law for semantic consistency."""

@dataclass
class LintResult:
    law_id: str
    warnings: list[LintWarning]
    observable_types: list[tuple[str, TypedQuantity]]
    has_errors: bool

@dataclass
class LintWarning:
    law_id: str
    message: str
    severity: str  # "warning" or "error"
    suggested_fix: str | None
```

### Checks Performed

1. **Observable type vs name**: Does `particle_count` actually count particles?
2. **Conservation laws**: Should use truly conserved quantities
3. **Bound laws**: Don't compare incompatible types (e.g., momentum vs count)
4. **Canonical names**: Suggest better observable names
5. **Type propagation**: Verify types through expressions

---

## 12. Quantity Type System

**Location:** `src/claims/quantity_types.py`

Type inference for observable expressions.

```python
class QuantityType(str, Enum):
    CELL_COUNT = "cell_count"           # Counts cells (X=1 cell)
    PARTICLE_COUNT = "particle_count"   # Counts particles (X=2 particles)
    COMPONENT_COUNT = "component_count" # R+X or L+X
    MOMENTUM_LIKE = "momentum_like"     # Signed difference (R-L)
    POSITION = "position"               # Grid positions
    LENGTH = "length"                   # Lengths/distances
    RATIO = "ratio"                     # Dimensionless
    PAIR_COUNT = "pair_count"           # Adjacent/gap pairs
    UNKNOWN = "unknown"
```

### Type Inference Rules

| Expression Pattern | Inferred Type |
|--------------------|---------------|
| `count('.')` | `CELL_COUNT` |
| `count('X')` | `CELL_COUNT` |
| `count('>')`, `count('<')` | `COMPONENT_COUNT` |
| `count('>') - count('<')` | `MOMENTUM_LIKE` |
| `count('>') + count('X')` | `COMPONENT_COUNT` (right component) |
| `count('<') + count('X')` | `COMPONENT_COUNT` (left component) |
| `count('>') + count('<') + 2*count('X')` | `PARTICLE_COUNT` |
| `leftmost`, `rightmost` | `POSITION` |
| `grid_length`, `spread`, `max_gap` | `LENGTH` |
| `adjacent_pairs`, `gap_pairs` | `PAIR_COUNT` |

---

## 13. Compilation Pipeline

```
CandidateLaw (JSON/Pydantic)
    │
    ├── observables[].expr (string)
    │   └── ExpressionParser.parse() → Expr (AST)
    │       └── evaluate_expression(expr, state) → int
    │
    ├── preconditions[]
    │   └── compile_precondition() → Callable[[State], bool]
    │
    ├── claim_ast (optional JSON)
    │   └── ASTClaimEvaluator
    │
    └── template
        └── ClaimCompiler.compile() → TemplateChecker
            └── check(trajectory) → CheckResult
```

---

## 14. Usage Examples

### Creating a Law

```python
from src.claims.schema import CandidateLaw, Template, Quantifiers, Observable

law = CandidateLaw(
    law_id="particle_conservation",
    template=Template.INVARIANT,
    quantifiers=Quantifiers(T=50),
    observables=[
        Observable(
            name="total_particles",
            expr="count('>') + count('<') + 2*count('X')"
        )
    ],
    claim="Total particle count is conserved",
    forbidden="total_particles changes"
)
```

### Compiling a Law

```python
from src.claims.compiler import ClaimCompiler

compiler = ClaimCompiler()
checker = compiler.compile(law)
```

### Evaluating Against a Trajectory

```python
from src.universe.simulator import run

trajectory = run(initial_state, T=50)
result = checker.check(trajectory)

if result.passed:
    print("Law passed")
else:
    print(f"Violation at t={result.violation.t}")
    print(f"Details: {result.violation.details}")
```

### Fingerprinting for Deduplication

```python
from src.claims.fingerprint import compute_semantic_fingerprint

fp1 = compute_semantic_fingerprint(law1)
fp2 = compute_semantic_fingerprint(law2)

if fp1 == fp2:
    print("Laws are semantically equivalent")
```

---

## 15. Design Decisions

### Why Templates?

1. **Clarity**: Each template has unambiguous checking semantics
2. **Vacuity**: Implication templates can detect vacuous tests
3. **Power**: Different templates need different testing strategies
4. **Interpretability**: Counterexamples are meaningful

### Why Two Claim Formats?

1. **String claims** (`claim`): Human-readable, flexible, but ambiguous
2. **Structured ASTs** (`claim_ast`): Machine-precise, no parsing ambiguity

The system supports both; `claim_ast` takes precedence if provided.

### Why Separate Observables from Claims?

1. **Reusability**: Same observable in multiple laws
2. **Testability**: Observables can be validated independently
3. **Clarity**: Named observables document intent

---

## 16. Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| 8 templates | ✅ Complete | All with checkers |
| Expression parser | ✅ Complete | 10 functions + operators |
| Claim compiler | ✅ Complete | All templates |
| AST evaluator | ✅ Complete | Structured claims |
| Fingerprinting | ✅ Complete | Semantic dedup |
| Semantic linter | ✅ Complete | Type checking |
| Vacuity tracking | ✅ Complete | Trigger diversity |

---

## 17. Adding New Templates

To add a new template:

1. **Add enum value** in `src/claims/schema.py`:
   ```python
   class Template(str, Enum):
       NEW_TEMPLATE = "new_template"
   ```

2. **Create checker** in `src/claims/templates.py`:
   ```python
   class NewTemplateChecker(TemplateChecker):
       def check(self, trajectory: Trajectory) -> CheckResult:
           # Implementation
   ```

3. **Add compilation** in `src/claims/compiler.py`:
   ```python
   def _compile_new_template(self, law: CandidateLaw) -> NewTemplateChecker:
       # Validate required fields
       # Return checker
   ```

4. **Update dispatcher** in `ClaimCompiler.compile()`

5. **Add tests** in `tests/`
