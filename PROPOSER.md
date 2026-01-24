# PROPOSER.md — Law Discovery Proposer System

This document describes the proposer system that orchestrates LLM-driven law discovery. The proposer generates falsifiable candidate laws about the simulated universe by interacting with an LLM, parsing responses, filtering duplicates, and ranking proposals by expected scientific value.

---

## 1. System Overview

### Purpose

The proposer implements the "conjecture" phase of Popperian scientific discovery:
1. Generate falsifiable candidate laws via LLM
2. Parse and validate proposed laws
3. Filter redundant proposals
4. Rank by scientific value (risk, novelty, discrimination, testability)
5. Return ranked laws for testing by the harness

### Core Flow

```
DiscoveryMemory → PromptBuilder → GeminiClient → ResponseParser → RedundancyDetector → RankingModel → ProposalBatch
```

### Source Location

All proposer components are in `src/proposer/`:

| File | Component | Purpose |
|------|-----------|---------|
| `proposer.py` | LawProposer | Main orchestrator |
| `client.py` | GeminiClient | LLM API interface |
| `prompt.py` | PromptBuilder | Prompt construction |
| `parser.py` | ResponseParser | JSON parsing/validation |
| `ranking.py` | RankingModel | Multi-factor scoring |
| `redundancy.py` | RedundancyDetector | Duplicate filtering |
| `memory.py` | DiscoveryMemory | State tracking |

---

## 2. Architecture

### Component Relationships

```
                    ┌─────────────────┐
                    │  DiscoveryMemory │
                    │   (state store)  │
                    └────────┬────────┘
                             │ get_snapshot()
                             ▼
┌──────────────┐    ┌─────────────────┐    ┌─────────────┐
│ UniverseContract │──▶│  PromptBuilder   │──▶│ GeminiClient │
│  (capabilities)  │    │ (prompt + sysins) │    │   (LLM API)  │
└──────────────┘    └─────────────────┘    └──────┬──────┘
                                                   │ response
                                                   ▼
                                          ┌─────────────────┐
                                          │ ResponseParser   │
                                          │ (JSON → Laws)    │
                                          └────────┬────────┘
                                                   │ parsed laws
                                                   ▼
                                          ┌─────────────────┐
                                          │RedundancyDetector│
                                          │ (filter dupes)   │
                                          └────────┬────────┘
                                                   │ unique laws
                                                   ▼
                                          ┌─────────────────┐
                                          │  RankingModel    │
                                          │ (score & sort)   │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │  ProposalBatch   │
                                          │  (final output)  │
                                          └─────────────────┘
```

---

## 3. LawProposer Orchestrator

The `LawProposer` class in `src/proposer/proposer.py` is the main entry point.

### Configuration

```python
@dataclass
class ProposerConfig:
    max_token_budget: int = 8000      # Max tokens for prompts
    include_counterexamples: bool = True  # Include counterexample gallery
    strict_parsing: bool = False      # Reject vs. attempt to fix invalid laws
    add_to_redundancy_filter: bool = True  # Auto-learn proposed laws
    verbose: bool = False             # Store full prompts/responses
```

### Core Method: propose()

```python
def propose(
    self,
    memory: DiscoveryMemorySnapshot | DiscoveryMemory,
    request: ProposalRequest | None = None,
) -> ProposalBatch:
```

**Workflow:**
1. Build prompt from contract + memory + request
2. Call LLM with system instruction
3. Parse JSON response into CandidateLaw objects
4. Filter redundant laws
5. Rank by multi-factor score
6. Return ProposalBatch

### ProposalRequest

```python
@dataclass
class ProposalRequest:
    count: int = 5                    # Number of laws to request
    target_templates: list[str] | None = None  # Focus on specific templates
    exclude_templates: list[str] | None = None  # Avoid specific templates
    temperature: float | None = None  # Override LLM temperature
```

### ProposalBatch (Output)

```python
@dataclass
class ProposalBatch:
    laws: list[CandidateLaw]          # Ranked candidate laws
    features: list[RankingFeatures]   # Scores for each law
    rejections: list[tuple[dict, str]] # Failed parsing with reasons
    redundant: list[tuple[CandidateLaw, RedundancyMatch]]  # Filtered duplicates
    prompt_hash: str                  # For audit/caching
    prompt_tokens: int
    response_tokens: int
    runtime_ms: int
    warnings: list[str]
```

---

## 4. LLM Client

The `GeminiClient` in `src/proposer/client.py` provides the LLM interface.

### Configuration

```python
@dataclass
class GeminiConfig:
    api_key: str | None = None       # From GEMINI_API_KEY or GOOGLE_API_KEY env
    model: str = "gemini-2.5-flash"  # Model name
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    json_mode: bool = True           # Enforce JSON output
    thinking_budget: int | None = None  # Extended reasoning tokens
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `generate(prompt, system_instruction, temperature)` | Send prompt, get text response |
| `generate_json(prompt, system_instruction)` | Parse response as JSON |
| `count_tokens(text)` | Token counting (with fallback estimation) |

### Token Usage Tracking

```python
@dataclass
class TokenUsage:
    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int = 0
```

Access via `client.last_usage` after each call.

---

## 5. Prompt Engineering

The `PromptBuilder` in `src/proposer/prompt.py` constructs prompts with two sections:

### Prompt Structure

```
=== STATIC SECTION (cached by API) ===
- Universe capabilities
- Expression language reminder
- Request specification

=== DYNAMIC SECTION (changes each iteration) ===
- Accepted laws (do not repeat)
- Falsified laws with counterexamples
- Unknown laws with reason codes
- Counterexample gallery
```

### System Instruction

The system instruction (~200 lines) enforces:

1. **Output format**: JSON array only, no prose
2. **Template constraints**: Must use one of 8 templates exactly
3. **Falsifiability**: Every law must have "forbidden" field
4. **AST format**: Claims must use structured JSON AST (claim_ast)
5. **Observable definitions**: Use canonical names and expressions
6. **Collision physics**: Correct understanding of X behavior

### UniverseContract

```python
@dataclass
class UniverseContract:
    universe_id: str = "kinetic_grid_v1"
    symbols: list[str] = [".", ">", "<", "X"]
    state_representation: str = "string"
    capabilities: dict = {
        "primitive_observables": ["count(symbol)", "grid_length", "incoming_collisions"],
        "transforms": ["mirror_swap", "shift_k", "swap_only", "mirror_only"],
        "generator_families": [
            "random_density_sweep",
            "constrained_pair_interactions",
            "edge_wrapping_cases",
            "symmetry_metamorphic_suite",
            "adversarial_mutation_search",
        ],
    }
    config_knobs: dict = {
        "grid_length_range": [4, 200],
        "boundary": "periodic",
    }
```

---

## 6. Law Schema

### CandidateLaw Structure

From `src/claims/schema.py`:

```python
class CandidateLaw:
    schema_version: str               # "1.0.0"
    law_id: str                       # Unique identifier
    template: Template                # One of 8 templates
    quantifiers: Quantifiers          # T (time horizon), H (eventuality horizon)
    preconditions: list[Precondition] # Applicability constraints
    observables: list[Observable]     # Named expressions
    claim: str                        # Human-readable summary
    forbidden: str                    # What falsifies it (REQUIRED)
    claim_ast: dict                   # JSON AST of claim (REQUIRED)
    transform: str | None             # For symmetry template
    direction: MonotoneDirection | None  # For monotone template
    bound_value: int | None           # For bound template
    bound_op: ComparisonOp | None     # For bound template
    proposed_tests: list[ProposedTest]
    capability_requirements: CapabilityRequirements
```

### Allowed Templates

| Template | Semantics |
|----------|-----------|
| `invariant` | ∀t∈[0..T]: claim holds |
| `monotone` | ∀t∈[0..T-1]: f(t+1) ≤ f(t) or ≥ |
| `implication_step` | ∀t∈[0..T-1]: P(t) → Q(t+1) |
| `implication_state` | ∀t∈[0..T]: P(t) → Q(t) |
| `eventually` | ∀t0: P(t0) → ∃t∈[t0..t0+H]: Q(t) |
| `symmetry_commutation` | evolve(T(S), n) == T(evolve(S, n)) |
| `bound` | ∀t∈[0..T]: f(t) op k |
| `local_transition` | ∀t,i: state[i]==trigger → state[i] satisfies result at t+1 |

### Claim AST Format

Claims are structured JSON ASTs:

```json
// TotalParticles(t) == TotalParticles(0)
{
  "op": "==",
  "lhs": {"obs": "TotalParticles", "t": {"var": "t"}},
  "rhs": {"obs": "TotalParticles", "t": {"const": 0}}
}
```

**AST Node Types:**
- Constant: `{"const": 5}`
- Time variable: `{"var": "t"}`
- Time t+1: `{"t_plus_1": true}`
- Observable: `{"obs": "<name>", "t": <time_node>}`
- Binary op: `{"op": "<op>", "lhs": <node>, "rhs": <node>}`
- Unary not: `{"op": "not", "arg": <node>}`

**Operators:** `+`, `-`, `*`, `/`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `=>`, `and`, `or`, `not`

### Canonical Observables

**CONSERVED (use for conservation laws):**
| Name | Expression |
|------|------------|
| TotalParticles | `count('>') + count('<') + 2*count('X')` |
| RightComponent | `count('>') + count('X')` |
| LeftComponent | `count('<') + count('X')` |
| Momentum | `count('>') - count('<')` |

**NOT CONSERVED (change during collisions):**
| Name | Expression |
|------|------------|
| FreeMovers | `count('>') + count('<')` |
| OccupiedCells | `count('>') + count('<') + count('X')` |
| CollisionCells | `count('X')` |
| IncomingCollisions | `incoming_collisions` |

---

## 7. Response Parsing

The `ResponseParser` in `src/proposer/parser.py` handles LLM output.

### Extraction

1. Extract JSON from markdown code blocks (` ```json ... ``` `)
2. Handle bare JSON arrays

### Sanitization

Fixes common LLM JSON errors:
- Stray backticks outside strings
- Trailing commas before `]` or `}`
- JavaScript-style comments (`//` and `/* */`)

### Validation

For each law object:
1. Required fields: `template`, `forbidden`, `claim_ast` or `claim`
2. Template must be in allowed enum
3. AST validation via `validate_claim_ast()`
4. Observable expressions parsed
5. Preconditions validated

### ParseResult

```python
@dataclass
class ParseResult:
    laws: list[CandidateLaw]           # Successfully parsed
    rejections: list[tuple[dict, str]] # Failed with reasons
    warnings: list[str]                # Non-fatal issues
```

---

## 8. Ranking Algorithm

The `RankingModel` in `src/proposer/ranking.py` scores laws by scientific value.

### Ranking Factors

| Factor | Weight | Meaning |
|--------|--------|---------|
| risk | 0.25 | How easily falsifiable (higher = better) |
| novelty | 0.20 | Distance from known laws (higher = better) |
| discrimination | 0.20 | Separates rival mechanisms (higher = better) |
| testability | 0.25 | Capabilities available (higher = better) |
| redundancy | -0.10 | Similarity to existing laws (higher = worse) |

### Formula

```
overall_score = 0.25*risk + 0.20*novelty + 0.20*discrimination
              + 0.25*testability - 0.10*redundancy
```

### Risk Scoring

- +0.2 if no preconditions (broader applicability)
- +0.2 if invariant or symmetry template
- +0.1 if strong claims (equality in forbidden)

### Novelty Scoring

- Starts at 0.8
- -0.1 for each accepted law with same template
- -0.1 for each accepted law with overlapping observables

### Discrimination Scoring

- +0.3 for symmetry template
- +0.2 for implication templates
- +0.1 if specific transform specified

### Testability Scoring

- Starts at 1.0
- -0.3 per missing observable
- -0.3 per missing transform
- -0.2 per missing generator
- -0.1 per unavailable proposed test

---

## 9. Redundancy Detection

The `RedundancyDetector` in `src/proposer/redundancy.py` prevents duplicate proposals.

### Three-Tier Matching

1. **Exact hash**: Identical content via `content_hash()`
2. **Semantic fingerprint**: Same meaning, different names via `compute_semantic_fingerprint()`
3. **Normalized form**: Sorted observables, lowercase, whitespace removed

### Key Methods

| Method | Purpose |
|--------|---------|
| `add_known_law(law)` | Learn a law (adds to all indexes) |
| `check(law)` | Test single law for redundancy |
| `filter_batch(laws)` | Filter batch, catch within-batch duplicates |

### RedundancyMatch

```python
@dataclass
class RedundancyMatch:
    matched_law_id: str
    similarity: float        # 0.0 to 1.0
    match_type: str          # "exact", "fingerprint", "normalized", etc.
    details: str
```

---

## 10. Discovery Memory

The `DiscoveryMemory` in `src/proposer/memory.py` tracks discovery state for feedback.

### Memory Limits

```python
max_accepted: int = 30       # Recent PASS laws
max_falsified: int = 20      # Recent FAIL laws
max_unknown: int = 20        # Recent UNKNOWN laws
max_counterexamples: int = 20  # Gallery entries
```

Trims oldest entries when limits exceeded.

### Recording Evaluations

```python
def record_evaluation(self, law: CandidateLaw, verdict: LawVerdict) -> None:
```

Routes to appropriate list based on verdict status (PASS/FAIL/UNKNOWN).

### DiscoveryMemorySnapshot

Immutable snapshot for prompting:

```python
@dataclass
class DiscoveryMemorySnapshot:
    accepted_laws: list[dict]      # Passed with coverage metrics
    falsified_laws: list[dict]     # Failed with counterexamples
    unknown_laws: list[dict]       # Untestable with reason codes
    counterexamples: list[dict]    # Gallery of witness states
    capabilities: dict             # Current harness capabilities
```

---

## 11. Discovery Loop Integration

### Main Loop (scripts/run_discovery.py)

```python
# Per iteration:
1. Load existing state from DB
2. Build memory snapshot
3. proposer.propose(memory, request) → ProposalBatch
4. For each law in batch:
   - harness.evaluate(law) → verdict
   - memory.record_evaluation(law, verdict)
   - persist to SQLite
5. Log counterexamples
6. Periodic escalation testing
```

### Parallel Execution

Multiple workers run independently, synchronizing via shared SQLite database.

### Persistence

Laws and evaluations saved to SQLite for resume capability and audit trail.

---

## 12. Configuration Reference

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Primary API key |
| `GOOGLE_API_KEY` | Fallback API key |

### Default Configuration

```python
# GeminiClient
model = "gemini-2.5-flash"
temperature = 0.7
max_output_tokens = 65535
json_mode = True

# ProposerConfig
max_token_budget = 16000
include_counterexamples = True
strict_parsing = False
add_to_redundancy_filter = True

# DiscoveryMemory
max_accepted = 30
max_falsified = 20
max_unknown = 20
max_counterexamples = 20

# RankingWeights
risk = 0.25
novelty = 0.20
discrimination = 0.20
testability = 0.25
redundancy = 0.10
```

---

## 13. Agent Usage Notes

### Key Invariants

1. **Every law must have a "forbidden" field** - this is what makes it falsifiable
2. **Claims must use claim_ast** - string claims are deprecated
3. **Templates are strict** - only 8 allowed, no variations
4. **Observables define before use** - expressions go in observables array
5. **Redundancy filter learns** - proposed laws are auto-added if configured

### Common Pitfalls

1. **FreeMovers is NOT conserved** - use TotalParticles, RightComponent, LeftComponent, or Momentum
2. **adjacent_pairs('>', '<') does NOT predict collisions** - use incoming_collisions
3. **X is ephemeral** - each X resolves in one step but new X can form
4. **Division not supported** - rewrite algebraically (X <= L/2 becomes 2*X <= L)

### Debugging Tips

1. Use `proposer.get_last_exchange()` to inspect prompt/response
2. Check `proposer.stats` for iteration metrics
3. Review `batch.rejections` for parsing failures
4. Review `batch.redundant` for filtered duplicates
5. Enable `verbose=True` in ProposerConfig for full logging

### Testing Without API

Use `MockGeminiClient` for deterministic testing without API calls:

```python
from src.proposer.client import MockGeminiClient

client = MockGeminiClient()
proposer = LawProposer(client=client)
```
