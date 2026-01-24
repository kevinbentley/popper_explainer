# KG_UNIVERSE.md — Formal Specification of the Kinetic Grid Universe

This document provides a complete, formal definition of the Kinetic Grid Universe physics for use by agents requiring authoritative knowledge of universal laws and behaviors.

---

## 1. FOUNDATIONAL AXIOMS

### 1.1 Grid Structure

| Property | Value |
|----------|-------|
| Dimensionality | 1 (linear) |
| Length | N cells (finite, positive integer) |
| Cell indices | 0 to N-1 |
| Topology | Circular (periodic boundary) |
| Time | Discrete, non-negative integers |

### 1.2 Alphabet

The state of each cell is exactly one symbol from the alphabet:

```
ALPHABET = { '.', '>', '<', 'X' }
```

### 1.3 Symbol Semantics

| Symbol | Type | Contains Right-Mover | Contains Left-Mover | Total Particles |
|--------|------|----------------------|---------------------|-----------------|
| `.` | Empty | 0 | 0 | 0 |
| `>` | Particle | 1 | 0 | 1 |
| `<` | Particle | 0 | 1 | 1 |
| `X` | Collision | 1 | 1 | 2 |

### 1.4 State Definition

A **state** S is a string of length N where each character is in ALPHABET.

```
S : [0, N-1] → ALPHABET
S = s₀s₁s₂...sₙ₋₁
```

---

## 2. PARTICLE DEFINITIONS

### 2.1 Component Functions

For any state S, define:

```
R(S) = count of '>' in S
L(S) = count of '<' in S
X(S) = count of 'X' in S
E(S) = count of '.' in S
```

### 2.2 Derived Counts

| Name | Symbol | Formula | Description |
|------|--------|---------|-------------|
| Right components | N_R | R(S) + X(S) | Total right-moving entities |
| Left components | N_L | L(S) + X(S) | Total left-moving entities |
| Total particles | N_P | R(S) + L(S) + 2·X(S) | Total particle count |
| Net momentum | M | N_R - N_L | Signed momentum |

### 2.3 Particle Properties

| Property | Value |
|----------|-------|
| Speed | 1 cell per timestep |
| Direction | Fixed (never changes) |
| Creation | Forbidden |
| Destruction | Forbidden |
| Size | Point-like (occupies single cell) |

---

## 3. STATE TRANSITION FUNCTION

### 3.1 Overview

The universe evolves deterministically:

```
EVOLVE : STATE × TIME → STATE
S(t+1) = EVOLVE(S(t))
```

### 3.2 Contribution Phase

At time t, each cell contributes particles to neighboring cells:

```
CONTRIBUTE(cell, position) → {(destination, direction)}

CONTRIBUTE('.', i) = {}
CONTRIBUTE('>', i) = {((i+1) mod N, RIGHT)}
CONTRIBUTE('<', i) = {((i-1) mod N, LEFT)}
CONTRIBUTE('X', i) = {((i+1) mod N, RIGHT), ((i-1) mod N, LEFT)}
```

### 3.3 Collection Phase

For each cell j at time t+1, collect incoming contributions:

```
r(j) = count of RIGHT contributions arriving at j
l(j) = count of LEFT contributions arriving at j
```

### 3.4 Resolution Phase

Determine the new cell state based on incoming counts:

```
RESOLVE(r, l) → symbol

RESOLVE(0, 0) = '.'
RESOLVE(r, 0) = '>'  where r > 0
RESOLVE(0, l) = '<'  where l > 0
RESOLVE(r, l) = 'X'  where r > 0 and l > 0
```

### 3.5 Complete Transition

```
For each position j in [0, N-1]:
    r(j) = |{i : S(t)[i] ∈ {'>', 'X'} and (i+1) mod N = j}|
    l(j) = |{i : S(t)[i] ∈ {'<', 'X'} and (i-1) mod N = j}|
    S(t+1)[j] = RESOLVE(r(j), l(j))
```

---

## 4. BOUNDARY CONDITIONS

### 4.1 Periodic Wrapping

```
Position arithmetic uses modular arithmetic:
    (i + 1) mod N  for rightward movement
    (i - 1) mod N  for leftward movement (equivalent to (i + N - 1) mod N)
```

### 4.2 Consequences

- No edges, walls, or boundaries
- No particle sinks or sources
- Grid forms a closed ring
- Every cell has exactly two neighbors

---

## 5. COLLISION MECHANICS

### 5.1 Collision Formation

An X forms at position j at time t+1 if and only if:
```
r(j) > 0 AND l(j) > 0
```

### 5.2 Collision Resolution

X at position i at time t:
- Contributes one right-mover to (i+1) mod N
- Contributes one left-mover to (i-1) mod N
- Does NOT persist (ephemeral)

### 5.3 Collision Chains

After X resolution, new collisions may form if emitted particles intercept:
- Other emitted particles from adjacent X cells
- Free particles approaching the emission destinations

---

## 6. UNIVERSAL INVARIANTS (Conservation Laws)

These quantities are preserved for ALL valid state transitions:

### 6.1 Conservation of Right Components

```
INVARIANT: N_R(S(t)) = N_R(S(t+1)) for all t
```

**Proof sketch**: Each '>' and each 'X' emits exactly one right-mover. Each right-mover becomes part of exactly one '>' or 'X' at t+1.

### 6.2 Conservation of Left Components

```
INVARIANT: N_L(S(t)) = N_L(S(t+1)) for all t
```

### 6.3 Conservation of Total Particles

```
INVARIANT: N_P(S(t)) = N_P(S(t+1)) for all t
```

Since N_P = N_R + N_L and both are conserved.

### 6.4 Conservation of Momentum

```
INVARIANT: M(S(t)) = M(S(t+1)) for all t
```

Since M = N_R - N_L and both N_R and N_L are independently conserved.

---

## 7. NON-INVARIANTS

These quantities may change during evolution:

| Quantity | Why Not Conserved |
|----------|-------------------|
| Number of '>' symbols | Collisions create X from > and < |
| Number of '<' symbols | Collisions create X from > and < |
| Number of 'X' symbols | Collisions form and resolve |
| Number of '.' symbols | Particle motion changes occupancy |
| Occupied cell count | Collisions merge/split occupancy |
| Local density | Particles move and collide |

---

## 8. SYMMETRY LAWS

### 8.1 Mirror-Swap Symmetry (TRUE SYMMETRY)

Define the mirror-swap transformation MS:
```
MS(S) = reverse(swap(S))

where:
    reverse(s₀s₁...sₙ₋₁) = sₙ₋₁...s₁s₀
    swap(c) = '>' if c = '<'
            = '<' if c = '>'
            = c   otherwise
```

**Symmetry Law**:
```
EVOLVE(MS(S)) = MS(EVOLVE(S))
```

This transformation commutes with time evolution.

### 8.2 Non-Symmetries (IMPORTANT)

The following are NOT symmetries:

| Transformation | Why Not a Symmetry |
|----------------|-------------------|
| Spatial reverse only | Particles move wrong direction |
| Direction swap only | Breaks spatial relationships |
| Time reversal | Collisions are not invertible |
| Translation | Breaks position-dependent patterns |

---

## 9. VALID STATE CONSTRAINTS

### 9.1 Initial State Constraints

For valid initial states:
```
VALID_INITIAL(S) ⟺ ∀i: S[i] ∈ {'.', '>', '<'}
```

**X is forbidden in initial states.**

### 9.2 Evolved State Constraints

After evolution, X may appear:
```
VALID_EVOLVED(S) ⟺ ∀i: S[i] ∈ {'.', '>', '<', 'X'}
```

### 9.3 Reachability

A state S is **reachable** if there exists a valid initial state S₀ and time t ≥ 0 such that:
```
S = EVOLVE^t(S₀)
```

---

## 10. CAUSAL STRUCTURE

### 10.1 Light Cone

The causal influence radius is exactly 1 cell per timestep:
```
CAUSAL_RADIUS = 1
```

Cell j at time t+1 depends only on cells {(j-1) mod N, j, (j+1) mod N} at time t.

### 10.2 Information Propagation

Maximum information speed: 1 cell per timestep (particle velocity).

### 10.3 Local Determinism

Each cell's next state is determined entirely by local information (3-cell neighborhood).

---

## 11. OBSERVABLE DEFINITIONS

### 11.1 Core Observables

| Observable | Formula | Domain |
|------------|---------|--------|
| `right_count` | R(S) | [0, N] |
| `left_count` | L(S) | [0, N] |
| `collision_count` | X(S) | [0, N/2] |
| `empty_count` | E(S) | [0, N] |
| `right_components` | R(S) + X(S) | [0, N] |
| `left_components` | L(S) + X(S) | [0, N] |
| `total_particles` | R(S) + L(S) + 2·X(S) | [0, 2N] |
| `momentum` | N_R - N_L | [-N, N] |
| `density` | N_P / N | [0, 2] |

### 11.2 Local Observables

| Observable | Definition |
|------------|------------|
| `cell_at(i)` | S[i] |
| `is_empty(i)` | S[i] = '.' |
| `is_collision(i)` | S[i] = 'X' |
| `has_right_mover(i)` | S[i] ∈ {'>', 'X'} |
| `has_left_mover(i)` | S[i] ∈ {'<', 'X'} |

### 11.3 Pattern Observables

| Observable | Definition |
|------------|------------|
| `adjacent_pair(i)` | True if S[i] and S[(i+1) mod N] are both occupied |
| `converging_pair(i)` | S[i] = '>' and S[(i+1) mod N] = '<' |
| `diverging_pair(i)` | S[i] = '<' and S[(i+1) mod N] = '>' |
| `gap_before_right(i)` | Distance to nearest '>' or 'X' to the left |

---

## 12. COMMON PATTERNS AND BEHAVIORS

### 12.1 Collision Formation Pattern

A collision at position j at time t+1 requires:
```
∃i₁,i₂ : (i₁+1) mod N = j AND S(t)[i₁] ∈ {'>', 'X'}
     AND (i₂-1) mod N = j AND S(t)[i₂] ∈ {'<', 'X'}
```

### 12.2 Converging Pair

Two particles `><` at positions i, i+1 will collide at position i+1 at time t+1.

### 12.3 Persistent Collision Chain

If X cells are adjacent, they may form a persistent chain of collisions as emitted particles re-intercept.

### 12.4 Free Particle Motion

A '>' with no approaching '<' will move right indefinitely.
A '<' with no approaching '>' will move left indefinitely.

---

## 13. EDGE CASES AND SPECIAL STATES

### 13.1 Empty Universe

```
S = "......" (all dots)
EVOLVE(S) = S (fixed point)
```

### 13.2 Single Particle

```
S = "..>..."
EVOLVE(S) = "...>.." (circulates)
Period = N
```

### 13.3 Symmetric Pair

```
S = ".><..." (converging)
EVOLVE(S) = "..X..."
EVOLVE²(S) = ".><..." (returns to similar pattern, shifted)
```

### 13.4 Maximum Collision

```
S = "><><><" (alternating)
All pairs collide simultaneously
Maximum collision density
```

---

## 14. FORMAL PROPERTIES SUMMARY

### 14.1 System Classification

| Property | Value |
|----------|-------|
| Deterministic | Yes |
| Reversible | No |
| Conservative | Yes (particles, components, momentum) |
| Chaotic | No |
| Periodic | Yes (finite state space) |
| Ergodic | State-dependent |

### 14.2 Computational Properties

| Property | Value |
|----------|-------|
| State space size | 4^N |
| Transition computable | O(N) time |
| Period upper bound | 4^N |
| Hidden state | None |

---

## 15. FALSIFICATION CRITERIA

A proposed law is **falsified** if there exists:
1. A valid initial state S₀
2. A time t ≥ 0
3. A trajectory S₀ → S₁ → ... → Sₜ

Such that the law's claim is violated at some step.

A proposed law is **unfalsifiable** (and thus unscientific) if:
- It makes no testable predictions
- It holds vacuously (no states satisfy preconditions)
- It depends on unobservable quantities

---

## 16. QUICK REFERENCE TABLES

### 16.1 Symbol Quick Reference

| Symbol | Right-Movers | Left-Movers | Total | Emits Right | Emits Left |
|--------|--------------|-------------|-------|-------------|------------|
| `.` | 0 | 0 | 0 | No | No |
| `>` | 1 | 0 | 1 | Yes | No |
| `<` | 0 | 1 | 1 | No | Yes |
| `X` | 1 | 1 | 2 | Yes | Yes |

### 16.2 Resolution Quick Reference

| Incoming Right | Incoming Left | Result |
|----------------|---------------|--------|
| 0 | 0 | `.` |
| ≥1 | 0 | `>` |
| 0 | ≥1 | `<` |
| ≥1 | ≥1 | `X` |

### 16.3 Conservation Quick Reference

| Quantity | Conserved | Formula |
|----------|-----------|---------|
| Right components | YES | R + X |
| Left components | YES | L + X |
| Total particles | YES | R + L + 2X |
| Momentum | YES | (R + X) - (L + X) |
| Collision count | NO | X |
| Empty count | NO | E |
| Occupied count | NO | R + L + X |

---

## 17. AGENT USAGE NOTES

1. **All conservation laws are consequences** of the transition rules, not axioms.

2. **X is ephemeral** - it exists for exactly one timestep unless re-formed by new incoming particles.

3. **Collisions absorb multiplicity** - if multiple right-movers arrive at a cell with left-movers, the result is still a single X containing one of each.

4. **Mirror-swap is the only true symmetry** - do not assume other symmetries.

5. **Time is not reversible** - given a state with X, there may be multiple predecessor states.

6. **Initial states cannot contain X** - this is a validity constraint, not a physical law.

7. **Period depends on initial conditions** - simple states may have short periods; complex states may have very long periods.
