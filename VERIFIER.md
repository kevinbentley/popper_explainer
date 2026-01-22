# VERIFIER.md — Explanation Verification Criteria (Popperian Standard)

This document defines how to evaluate whether a discovered theory or law set qualifies as a **Popper-valid explanation**, rather than a mere descriptive regularity.

It is used by:
- verifier agents
- meta-evaluation code
- auditors and researchers

It MUST NOT be provided to the Law Discovery agent.

---

## 1. Purpose

The verifier answers a single question:

> Does this theory explain why the observed laws hold, rather than merely restating them?

The verifier does **not** decide truth (that is the tester’s job).  
It decides **explanatory adequacy**.

---

## 2. Definitions

### 2.1 Law
A falsifiable universal claim about system behavior.

Examples:
- “Particle count is conserved”
- “Mirror+swap commutes with evolution”

### 2.2 Explanation
A structured account that:
- identifies underlying constraints or mechanisms
- makes the law *necessary*, not coincidental
- would fail if the law failed

### 2.3 Non-Explanation (Anti-Patterns)
- restatement (“X is conserved because it is conserved”)
- correlation without constraint
- brute-force enumeration
- memorized invariants without generative reason

---

## 3. Verifier Inputs

The verifier is given:

1) **Theory object**
   - axioms
   - derived theorems
   - dependency graph (axioms → theorems)
   - explanations attached to theorems

2) **Law record**
   - the specific law being explained
   - tester verdict (PASS/FAIL/UNKNOWN)
   - counterexamples if any

3) **Universe contract**
   - semantic meaning of primitives
   - allowed operations and constraints

---

## 4. Explanation Validity Criteria

An explanation must satisfy **all** of the following.

---

### V1 — Non-Circularity

**Requirement**  
The explanation must not directly or indirectly restate the law it explains.

**Fail if**:
- the law appears verbatim in its own explanation
- the explanation depends on a theorem that is equivalent to the law
- the dependency graph contains a cycle involving the law

**Example (FAIL)**  
“Momentum is conserved because momentum does not change.”

**Example (PASS)**  
“Momentum is conserved because right- and left-moving components are independently conserved, and momentum is their difference.”

---

### V2 — Dependency Asymmetry

**Requirement**  
The explanation must depend on **more primitive claims** than the law itself.

**Operational test**:
- every explanation must reference only:
  - axioms
  - strictly lower-complexity theorems

Complexity may be approximated by:
- number of observables involved
- number of quantifiers
- dependency depth

**Fail if**:
- the law is primitive (axiom) but presented as explained
- explanation depends on an equal-or-higher-level theorem

---

### V3 — Counterfactual Sensitivity

**Requirement**  
If the explanation were false, the law would not hold.

**Operational proxy**  
The verifier must be able to imagine (or construct via a rival universe):

- a modified rule where the explanation breaks
- and show that the law fails under that modification

This does NOT require running a real alternative simulator — a *conceptual* counterfactual suffices if grounded.

**Example (PASS)**  
“If particles did not move at fixed speed v=1, collision separation would not be guaranteed, and particle count conservation could fail.”

**Example (FAIL)**  
“This law holds because the simulator enforces it.”

---

### V4 — Constraint Identification

**Requirement**  
The explanation must identify a **constraint** or **necessity**, not a pattern.

Acceptable constraint types:
- conservation of components
- causal locality
- symmetry invariance
- structural impossibility
- topological necessity

**Fail if**:
- explanation only lists observations
- explanation appeals to frequency or probability
- explanation is purely statistical

---

### V5 — Compression Gain

**Requirement**  
The explanation must reduce descriptive complexity.

**Operational test**:
- explanation should allow *multiple* laws to be derived from fewer principles
- explanation should eliminate the need to separately state some laws

**Example (PASS)**  
“One conservation principle explains particle count, momentum, and collision resolution.”

**Example (FAIL)**  
Each law explained independently with no shared structure.

---

### V6 — Falsifiability of the Explanation

**Requirement**  
The explanation itself must be falsifiable.

That is:
- it must imply at least one testable consequence
- there must exist a conceivable counterexample that would refute the explanation

**Fail if**:
- explanation is metaphysical or tautological
- explanation cannot be operationalized into testable claims

---

## 5. Explanation Scoring (Optional but Recommended)

The verifier may compute a **Why-Score** to compare explanations.

Suggested dimensions (0–2 each):

| Dimension | Meaning |
|---------|--------|
| Depth | How far below the law the explanation reaches |
| Necessity | How unavoidable the law becomes given explanation |
| Compression | How many laws are unified |
| Counterfactual | How sharply failure cases are defined |
| Simplicity | Fewer primitives, fewer assumptions |

Total score: 0–10  
Threshold for “explanatory”: ≥7 (configurable)

Scores are **diagnostic**, not authoritative.

---

## 6. Handling Partial Explanations

Explanations may be marked:

- **Incomplete**: explains *that* but not *why*
- **Local**: applies only under narrow conditions
- **Provisional**: depends on unverified axioms

These should not be rejected outright, but must be labeled.

---

## 7. Relationship to Tester Verdicts

| Tester Verdict | Verifier Role |
|---------------|---------------|
| PASS | Judge explanation quality |
| FAIL | Explanation invalid (law false) |
| UNKNOWN | Explanation may be conceptually valid but empirically unconfirmed |

Verifier must never upgrade UNKNOWN → PASS.

---

## 8. Forbidden Verifier Behaviors

The verifier MUST NOT:
- inject ground-truth facts into explanations
- “repair” explanations silently
- downgrade valid explanations because they resemble known physics
- assume human intuition as a standard

---

## 9. Output Format (Suggested)

Verifier outputs structured judgments:

```json
{
  "law_id": "momentum_conservation",
  "explanation_valid": true,
  "why_score": 8,
  "failed_criteria": [],
  "notes": [
    "Explanation identifies independent conservation of components",
    "Counterfactual dependency is clear"
  ]
}
