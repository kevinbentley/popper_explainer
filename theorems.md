# Theorems from thm_run_46b961cfa8be

**Run Status:** completed
**Total Theorems:** 10
**Generated:** 2026-01-24 04:32:44

---

## Theorem 1: Grid Structural Invariants and Empty Space Bounds

**Status:** Established
**ID:** `thm_02605397e070`

### Claim

The total count of particles and empty cells in the system is a fundamental invariant, always summing to the grid length. While various bounds exist for the maximum contiguous empty gaps based on the total empty cells, the system does not guarantee that a grid full of particles will eliminate all empty gaps, nor that empty cells will always outnumber free-moving particles.

### Support (7 laws)

- `particle_empty_cell_sum_identity`: confirms
- `occupied_cells_bound_by_grid_length`: confirms
- `max_empty_gap_bounded_by_available_empty_cells`: confirms
- `max_empty_gap_bounded_by_total_empty_cells`: confirms
- `max_empty_gap_bound_by_empty_cells`: confirms
- `max_empty_gap_zero_if_total_particles_equals_grid_length`: constrains
- `empty_cells_gte_free_movers`: constrains

### Failure Modes

- The sum of particles and empty cells deviates from grid length.
- Any of the established empty gap bounds are violated.
- A grid full of particles is observed to have no empty gaps, contradicting the counterexample for max_empty_gap_zero_if_total_particles_equals_grid_length.

### Missing Structure

- A precise definition of 'empty gap' and 'free movers' from the laws themselves.
- The exact mechanisms by which particles occupy or vacate cells.

---

## Theorem 2: Bounded Momentum and Directional Asymmetry

**Status:** Established
**ID:** `thm_9dad40df9f5b`

### Claim

The system's total momentum is universally bounded by both the total number of particles and the grid length, and also by the available free space. The individual counts of right-moving and left-moving particles are also bounded by the grid length. However, the system's dynamics are not symmetric under a simple momentum swap, suggesting a fundamental directional bias or asymmetrical interaction rules.

### Support (6 laws)

- `abs_momentum_bounded_by_total_particles`: confirms
- `momentum_bounded_by_grid_length`: confirms
- `momentum_bounded_by_free_space`: confirms
- `right_left_component_bounded_by_grid_length`: confirms
- `right_component_bounded_by_grid_length`: confirms
- `momentum_swap_symmetry`: constrains

### Failure Modes

- Observed momentum exceeds any of the established bounds.
- The system is shown to be symmetric under a momentum swap.

### Missing Structure

- A precise definition of 'momentum' and 'free space'.
- The specific rules governing particle movement that lead to momentum.

---

## Theorem 3: Conditional Collision Prevention and Insufficient Conditions

**Status:** Established
**ID:** `thm_e3d08909877a`

### Claim

The complete absence of left-moving particles guarantees no incoming collisions. Furthermore, if there are no free right-moving particles, incoming collisions are bounded by the count of 'X' cells. However, the absence of empty cells, any free movers, or 'X' cells generally, is insufficient to prevent incoming collisions, indicating that collisions can occur in dense or complex configurations with non-free movers.

### Support (7 laws)

- `no_incoming_collisions_if_no_left_component`: confirms
- `no_free_right_movers_bounds_incoming_collisions_by_x_cells`: confirms
- `no_incoming_collisions_if_all_occupied`: constrains
- `no_empty_implies_no_incoming_collisions`: constrains
- `no_free_movers_implies_no_collisions`: constrains
- `no_free_right_movers_implies_no_incoming_collisions`: constrains
- `no_x_cells_implies_no_incoming_collisions`: constrains

### Failure Modes

- Incoming collisions occur when no left-moving particles are present.
- Incoming collisions exceed the 'X' cell bound when no free right-movers are present.
- No collisions are observed when empty cells or free movers are absent, contradicting the failed laws.

### Missing Structure

- A precise definition of 'incoming collision', 'free movers', 'left component', 'right component', and 'X' cells.

---

## Theorem 4: Specific Particle Conservation vs. General Spatial Invariance

**Status:** Established
**ID:** `thm_0335ba4ad490`

### Claim

The total count of left-moving particles and free right-moving particles are conserved in the absence of collisions. However, this 'no collisions' condition is insufficient to conserve many other spatial arrangements or derived quantities such as particle-empty adjacent pairs, empty spread, or maximum empty gaps. This implies that the definition of 'collisions' relevant to particle counts is more specific than what affects general spatial configurations.

### Support (8 laws)

- `left_movers_conserved_if_no_collisions`: confirms
- `free_right_movers_conserved_if_no_collisions`: confirms
- `spread_right_movers_conserved_if_no_collisions`: constrains
- `particle_empty_adjacent_pairs_conserved_if_no_collisions`: constrains
- `empty_spread_conserved_if_no_collisions`: constrains
- `adjacent_dot_pairs_conserved_if_no_collisions`: constrains
- `passing_pairs_conserved_if_no_collisions`: constrains
- `max_empty_gap_conserved_if_no_collisions`: constrains

### Failure Modes

- Left-moving or free right-moving particles are not conserved when no collisions occur.
- Any of the quantities identified as 'not conserved' *are* observed to be conserved in the absence of collisions.

### Missing Structure

- A precise definition of 'collisions' and how they interact with different particle types and spatial arrangements.
- Definitions of 'spread', 'particle-empty adjacent pairs', 'empty spread', 'adjacent dot pairs', 'passing pairs', and 'max empty gap'.

---

## Theorem 5: Predictable Rates of Change Amidst Non-Monotonicity

**Status:** Established
**ID:** `thm_9af9fe270c35`

### Claim

The system exhibits specific, consistent rates of change for occupied cells and left-moving particles, and the rate of change for right-moving particles is an exact invariant. However, many other properties, such as adjacent 'gx' pairs, total occupied cells, or collision cells, do not exhibit simple monotonic behavior (e.g., non-increasing). This implies that while some aggregate properties evolve linearly, local interactions and collisions can cause fluctuations or increases in other system characteristics.

### Support (7 laws)

- `occupied_cells_change_rate_alternative`: confirms
- `left_movers_change_rate`: confirms
- `right_movers_change_rate_invariant_exact`: confirms
- `adjacent_gx_pairs_non_increasing`: constrains
- `occupied_cells_non_increasing`: constrains
- `collision_cells_non_increasing_at_high_density`: constrains
- `collision_cells_never_increase`: constrains

### Failure Modes

- The change rates for occupied cells or left-movers are not consistent.
- The right-movers change rate is not invariant.
- Any of the quantities identified as 'not monotonic' are observed to be strictly monotonic.

### Missing Structure

- The exact mathematical rules for the 'alternative' occupied cell change rate and left-movers change rate.
- A definition of 'adjacent gx pairs' and 'collision cells'.

---

## Theorem 6: Fundamental Invariants and Translational Symmetry

**Status:** Established
**ID:** `thm_43df1c1459ce`

### Claim

The system possesses fundamental invariants, including the total count of particles and empty cells, a specific relationship governing collision formation, and the exact rate of change of right-moving particles. Furthermore, the system's dynamics are invariant under spatial translations. However, the system does not exhibit simpler symmetries such as reflection (mirror) or a direct swap of momentum, indicating inherent directional and particle-specific characteristics.

### Support (6 laws)

- `particle_empty_cell_sum_identity`: confirms
- `dynamics_symmetric_under_shift`: confirms
- `bridge_law_collision_formation`: confirms
- `right_movers_change_rate_invariant_exact`: confirms
- `dynamics_symmetric_under_mirror_only`: constrains
- `momentum_swap_symmetry`: constrains

### Failure Modes

- Any of the identified invariants are violated.
- The system dynamics are not invariant under spatial shifts.
- The system is observed to be symmetric under mirror reflection or momentum swap.

### Missing Structure

- The explicit definition of the 'bridge law' for collision formation.
- The precise functional form of the 'right movers change rate invariant'.

---

## Theorem 7: Comprehensive Bounds on Incoming Collisions

**Status:** Established
**ID:** `thm_ef38843fd86e`

### Claim

Incoming collisions are comprehensively bounded by a variety of factors, including the sum of empty cells and free movers, the count of left-moving particles, half the total particles, the total particle components, the count of right-moving particles, and the maximum number of adjacent passing pairs. However, simpler bounds based solely on the count of empty cells, right-moving particles, or a ratio of free movers are insufficient to generally constrain incoming collisions, suggesting that complex interactions and configurations are key.

### Support (9 laws)

- `incoming_collisions_bounded_by_empty_plus_free_movers`: confirms
- `incoming_collisions_bounded_by_left_component`: confirms
- `incoming_collisions_bounded_by_half_total_particles`: confirms
- `incoming_collisions_bounded_by_components`: confirms
- `incoming_collisions_bounded_by_right_component`: confirms
- `max_adjacent_passing_pairs_bound`: confirms
- `incoming_collisions_bounded_by_empty_cells`: constrains
- `incoming_collisions_bounded_by_right_movers`: constrains
- `incoming_collisions_bound_by_free_movers_ratio`: constrains

### Failure Modes

- Any of the established bounds for incoming collisions are violated.
- A simpler, rejected bound is found to hold universally.

### Missing Structure

- Precise definitions for 'free movers', 'particle components', 'adjacent passing pairs', and what constitutes an 'incoming collision'.

---

## Theorem 8: Limited Predictive Power of Zero-Count Conditions

**Status:** Conditional
**ID:** `thm_790468098bc5`

### Claim

While the absence of free right-moving particles can specifically bound incoming collisions by 'X' cells, the general absence of free movers or empty cells does not imply a full resolution of 'X' cells, nor does it guarantee the conservation of free movers or that they will remain zero if initially absent. This indicates that zero counts of certain particle types or empty spaces do not simplify system dynamics as broadly as might be intuitively expected.

### Support (4 laws)

- `no_free_right_movers_bounds_incoming_collisions_by_x_cells`: confirms
- `no_free_movers_implies_full_x_resolution`: constrains
- `no_empty_cells_implies_free_movers_unchanged`: constrains
- `free_movers_zero_implies_zero`: constrains

### Failure Modes

- The bound on incoming collisions (given no free right-movers) is violated.
- The absence of free movers *does* lead to full X resolution.
- Free movers are observed to remain unchanged when empty cells are absent.

### Missing Structure

- Clear definitions of 'free movers', 'X cells', and 'full X resolution'.

---

## Theorem 9: The Unpredictability of Eventual System States

**Status:** Conjectural
**ID:** `thm_a5ffcc54ed53`

### Claim

The system's long-term behavior, such as the eventual appearance or disappearance of free movers, the resolution of collision cells, or the cessation of all collisions, is largely unpredictable from simple initial conditions. Despite a predictable rate of change for occupied cells, the overall trend is not necessarily monotonic, and local interactions prevent simple long-term predictions or global monotonic behavior for many system properties.

### Support (7 laws)

- `occupied_cells_change_rate_alternative`: confirms
- `free_movers_eventually_appear_if_zero`: constrains
- `free_movers_eventually_collide_or_form`: constrains
- `no_free_movers_eventually_no_collisions`: constrains
- `collision_cells_eventually_resolve`: constrains
- `collisions_eventually_cease`: constrains
- `occupied_cells_non_increasing`: constrains

### Failure Modes

- Any of the 'eventually' laws are confirmed to hold.
- A consistent long-term pattern for free movers or collisions is discovered.

### Missing Structure

- The precise rules governing collision resolution and particle transformation.
- A more complete understanding of the conditions under which 'eventual' behaviors might hold.

---

## Theorem 10: Specificity of Collision Triggers vs. General Presence

**Status:** Established
**ID:** `thm_e981b4401eb8`

### Claim

The absence of specific particle types, such as left-moving particles, can definitively prevent incoming collisions. Conversely, the mere presence of *both* free right and left movers does not guarantee incoming collisions, nor does the absence of 'X' cells simplify collision counts to just `gt_lt` pairs. This indicates that precise configurations and specific particle interactions, rather than just the general existence or absence of broad categories of particles, are crucial for triggering collision events.

### Support (3 laws)

- `no_incoming_collisions_if_no_left_component`: confirms
- `presence_of_both_free_movers_implies_incoming_collisions`: constrains
- `incoming_collisions_equals_gt_lt_pairs_if_no_x`: constrains

### Failure Modes

- Incoming collisions are observed when no left-moving particles are present.
- Incoming collisions are consistently observed whenever both free right and left movers are present.
- Incoming collisions are found to exactly equal `gt_lt` pairs in the absence of 'X' cells.

### Missing Structure

- A definition of 'gt_lt' pairs and 'X' cells to fully understand the context of the failed law.
- Detailed rules for how particle types interact to form collisions.

---
