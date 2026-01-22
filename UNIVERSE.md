# UNIVERSE.md — Kinetic Grid Universe (Authoritative Specification)

This document defines the ground-truth universe in which laws are to be discovered and tested.

It exists to:
- define the simulator’s semantics unambiguously
- constrain what counts as a valid explanation
- support rigorous falsification and auditing
- prevent “theory drift” by agents

⚠️ IMPORTANT  
- This document MUST NOT be provided to the Law Discovery LLM.  
- It MAY be provided to tester, verifier, and auditing agents.  
- The simulator implementation MUST conform to this document.

---

## 1. Universe Overview

The universe is a **1-dimensional discrete cellular universe** with:

- finite length `N`
- periodic boundary conditions
- discrete time steps
- deterministic evolution
- point-like particles with fixed velocities

The universe is intentionally simple but nontrivial, designed to support:
- conservation laws
- symmetry laws
- falsifiable counterexamples
- explanatory structure (WHY, not just WHAT)

---

## 2. State Representation

The universe state at time `t` is a **string of length N**.

Each character represents one cell and must be exactly one of:

| Symbol | Meaning |
|------|--------|
| `.`  | Empty cell |
| `>`  | Right-moving particle |
| `<`  | Left-moving particle |
| `X`  | Collision cell (simultaneous occupation by `>` and `<`) |

Example state:
..><.X..


---

## 3. Fundamental Entities

### 3.1 Particles

There are exactly two particle types:
- `>` : right-moving
- `<` : left-moving

Particles:
- are never created or destroyed
- always move at speed **1 cell per time step**
- retain their direction indefinitely

### 3.2 Collision State (`X`)

`X` is **not a particle**.  
It represents **two particles occupying the same cell simultaneously**:
- one `>`
- one `<`

Properties of `X`:
- contains exactly two particles
- has zero net momentum
- is ephemeral (exists for exactly one time step)
- must resolve deterministically

---

## 4. Time Evolution Rules

Time advances in discrete steps: `t → t+1`.

Evolution is **deterministic** and proceeds as follows:

### 4.1 Movement Phase

Each particle attempts to move simultaneously:

- `>` moves one cell to the right
- `<` moves one cell to the left
- movement wraps around at boundaries (periodic)

### 4.2 Collision Formation

If a `>` and `<` attempt to occupy the same cell in the same time step:
- they form an `X` at that cell
- no other collision types exist
- collisions only occur between opposite-direction particles

### 4.3 Collision Resolution

Every `X` resolves **in the next time step**:

- the `>` exits to the right
- the `<` exits to the left
- the original cell becomes empty (`.`)

Resulting pattern (schematic):
X → <.>

Resolution is:
- mandatory
- deterministic
- completes in exactly one step
- never produces another `X`

---

## 5. Boundary Conditions

The universe uses **periodic boundaries**:

- moving off the right edge wraps to the left
- moving off the left edge wraps to the right

There are no walls, sinks, or sources.

---

## 6. Determinism and Causality

- The universe is fully deterministic.
- Identical initial states always produce identical trajectories.
- The causal radius is exactly **1 cell per time step**.

There is:
- no randomness
- no hidden state
- no long-range interaction

---

## 7. Symmetries (Ground Truth)

The universe exhibits the following true symmetries:

### 7.1 Full Mirror Symmetry

The following transformation **commutes with evolution**:

- reverse the grid spatially
- swap particle directions (`>` ↔ `<`)

Formally:
evolve(mirror_swap(S), t) == mirror_swap(evolve(S, t))

### 7.2 Non-Symmetries (Important!)

The following are **NOT** symmetries:
- spatial mirror **without** swapping directions
- direction swap **without** spatial mirror
- time reversal (the system is not invertible)

---

## 8. Conserved Quantities (Ground Truth)

The following quantities are conserved across all time steps:

| Quantity | Definition |
|--------|------------|
| Right components | `count('>') + count('X')` |
| Left components  | `count('<') + count('X')` |
| Particle count   | `count('>') + count('<') + 2*count('X')` |
| Momentum         | Right − Left |

These are consequences of the movement and collision rules, not axioms.

---

## 9. Non-Conserved Quantities

The following are **not conserved**:

- number of occupied cells
- number of empty cells
- number of collision cells
- local densities

These may change during evolution.

---

## 10. What Counts as an Explanation

An explanation must:
- apply universally across all valid initial states
- survive adversarial counterexample search
- identify **why** a law holds (not just that it does)
- be falsifiable by a concrete counterexample

Merely restating observations is insufficient.

---

## 11. Forbidden Knowledge for Discovery Agents

The Law Discovery agent must NOT be told:

- collision resolution rules
- conservation laws
- symmetry properties
- the meaning of `X`
- movement speed or radius
- boundary conditions

Discovery must infer laws *only* from experiment outcomes.

---

## 12. Allowed Knowledge for Tester Agents

Tester and verifier agents MAY use this document to:
- generate valid experiments
- check semantic correctness of tests
- assess explanation quality
- reject invalid “explanations”

---

## 13. Contract with Simulator

The simulator implementation MUST:
- conform exactly to this document
- be deterministic
- reject invalid states
- not implement undocumented behavior

If simulator behavior diverges from this document, **the simulator is wrong**.

---

## 14. Why This Universe Exists

This universe is designed to answer one question:

> Can a non-human system generate Popper-valid explanations for laws in a universe it was not trained on?

The simplicity is intentional. The explanatory burden is not.

---
