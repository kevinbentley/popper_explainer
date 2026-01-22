"""Tests for the Kinetic Grid Universe simulator (Phase 1)."""

import pytest

from src.universe.types import Config, Symbol, VALID_SYMBOLS
from src.universe.validation import InvalidStateError, is_valid_state, validate_state
from src.universe.simulator import run, step, version_hash
from src.universe.transforms import mirror_only, mirror_swap, shift_k, swap_only
from src.universe.observables import (
    count_symbol,
    grid_length,
    left_component,
    momentum,
    occupied_cells,
    particle_count,
    right_component,
)


class TestTypes:
    def test_valid_symbols(self):
        assert "." in VALID_SYMBOLS
        assert ">" in VALID_SYMBOLS
        assert "<" in VALID_SYMBOLS
        assert "X" in VALID_SYMBOLS
        assert len(VALID_SYMBOLS) == 4

    def test_config_valid(self):
        config = Config(grid_length=10)
        assert config.grid_length == 10
        assert config.boundary == "periodic"

    def test_config_invalid_length(self):
        with pytest.raises(ValueError):
            Config(grid_length=0)

        with pytest.raises(ValueError):
            Config(grid_length=-5)

    def test_config_invalid_boundary(self):
        with pytest.raises(ValueError):
            Config(grid_length=10, boundary="reflecting")


class TestValidation:
    def test_valid_states(self):
        assert is_valid_state("........")
        assert is_valid_state("><")
        assert is_valid_state("..X..")
        assert is_valid_state(">.<.X.>")

    def test_invalid_empty_state(self):
        assert not is_valid_state("")

    def test_invalid_characters(self):
        assert not is_valid_state("..a..")
        assert not is_valid_state("..1..")
        assert not is_valid_state(".. ..")

    def test_validate_state_raises(self):
        with pytest.raises(InvalidStateError):
            validate_state("")

        with pytest.raises(InvalidStateError):
            validate_state("..a..")

    def test_validate_with_config(self):
        config = Config(grid_length=5)
        validate_state(".....", config)  # Should not raise

        with pytest.raises(InvalidStateError):
            validate_state("....", config)  # Wrong length


class TestSimulatorBasics:
    def test_version_hash_stable(self):
        h1 = version_hash()
        h2 = version_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_empty_universe(self):
        state = "........"
        next_state = step(state)
        assert next_state == "........"

    def test_single_right_particle(self):
        state = ">......."
        next_state = step(state)
        assert next_state == ".>......"

    def test_single_left_particle(self):
        state = ".......<"
        next_state = step(state)
        assert next_state == "......<."

    def test_right_particle_wraps(self):
        state = ".......>"
        next_state = step(state)
        assert next_state == ">......."

    def test_left_particle_wraps(self):
        state = "<......."
        next_state = step(state)
        assert next_state == ".......<"


class TestCollisions:
    def test_collision_forms(self):
        # Particles approaching each other - need 2 cells apart for collision
        # > at 2 moves to 3, < at 4 moves to 3 -> collision at 3
        state = "..>.<..."
        next_state = step(state)
        assert next_state == "...X...."

    def test_collision_resolves(self):
        # Collision resolves: > exits right, < exits left
        state = "...X...."
        next_state = step(state)
        assert next_state == "..<.>..."

    def test_collision_full_cycle(self):
        # Start with particles 2 apart (they will collide)
        s0 = ".>.<...."
        s1 = step(s0)  # Collision forms
        assert s1 == "..X.....", f"Expected collision, got {s1}"
        s2 = step(s1)  # Collision resolves
        assert "X" not in s2
        assert s2 == ".<.>....", f"Expected resolved, got {s2}"
        # Particles should have passed through each other
        assert s2.count(">") == 1
        assert s2.count("<") == 1

    def test_adjacent_particles_swap(self):
        # Particles directly adjacent - they swap positions, no collision
        # > at 0 goes to 1, < at 1 goes to 0
        state = "><"
        next_state = step(state)
        assert next_state == "<>"

    def test_particles_pass_through(self):
        # When adjacent (1 cell apart), particles swap positions
        state = "...><..."
        next_state = step(state)
        # > at 3 goes to 4, < at 4 goes to 3
        assert next_state == "...<>..."

    def test_collision_at_boundary(self):
        # Collision can happen at boundary due to wrapping
        # > at 7 moves to 0, < at 1 moves to 0 -> collision at 0
        state = ".<.....>"
        next_state = step(state)
        assert next_state == "X......."


class TestConservation:
    """Test that conserved quantities are actually conserved."""

    def test_right_component_conservation(self):
        state = "..><.X.."
        traj = run(state, 20)

        r0 = right_component(traj[0])
        for t, s in enumerate(traj):
            assert right_component(s) == r0, f"Right component changed at t={t}"

    def test_left_component_conservation(self):
        state = "..><.X.."
        traj = run(state, 20)

        l0 = left_component(traj[0])
        for t, s in enumerate(traj):
            assert left_component(s) == l0, f"Left component changed at t={t}"

    def test_particle_count_conservation(self):
        state = "><..><..X."
        traj = run(state, 30)

        p0 = particle_count(traj[0])
        for t, s in enumerate(traj):
            assert particle_count(s) == p0, f"Particle count changed at t={t}"

    def test_momentum_conservation(self):
        state = ">>><..X.."
        traj = run(state, 25)

        m0 = momentum(traj[0])
        for t, s in enumerate(traj):
            assert momentum(s) == m0, f"Momentum changed at t={t}"


class TestNonConservation:
    """Test that non-conserved quantities do change."""

    def test_occupied_cells_changes(self):
        # Start with collision state
        state = "..X.."
        s1 = step(state)
        # Collision resolves to two separate cells
        assert occupied_cells(state) == 1
        assert occupied_cells(s1) == 2

    def test_collision_count_changes(self):
        # Particles 2 apart will collide
        state = ".>.<.."
        s1 = step(state)
        assert count_symbol(state, "X") == 0
        assert count_symbol(s1, "X") == 1


class TestRun:
    def test_run_returns_trajectory(self):
        state = ">.."
        traj = run(state, 3)
        assert len(traj) == 4  # t=0, 1, 2, 3
        assert traj[0] == ">.."
        assert traj[1] == ".>."
        assert traj[2] == "..>"
        assert traj[3] == ">.."  # Wraps

    def test_run_with_config(self):
        config = Config(grid_length=5)
        traj = run(".....", 2, config)
        assert len(traj) == 3

    def test_run_invalid_t(self):
        with pytest.raises(ValueError):
            run("...", -1)


class TestMirrorSwapSymmetry:
    """Test that mirror_swap is a true symmetry (commutes with evolution)."""

    def test_mirror_swap_basic(self):
        # mirror_swap: swap directions then reverse spatially
        # "..><.." -> swap directions: "..<>.." -> reverse: "..><.."
        # Symmetric state stays the same
        assert mirror_swap("..><..") == "..><.."

        # Asymmetric state:
        # ">...<." -> swap directions: "<...>." -> reverse: ".>...<"
        state = ">...<."
        swapped = mirror_swap(state)
        assert swapped == ".>...<"

        # Another example:
        # ">.." -> swap: "<.." -> reverse: "..<"
        assert mirror_swap(">..") == "..<"

    def test_mirror_swap_commutes(self):
        """evolve(mirror_swap(S), t) == mirror_swap(evolve(S, t))"""
        test_states = [
            "..><..",
            ">....<",
            "..X..",
            ">>><<<",
            ".>.<.X..",
        ]

        for state in test_states:
            for t in [1, 5, 10]:
                # Path 1: transform then evolve
                transformed = mirror_swap(state)
                traj1 = run(transformed, t)
                result1 = traj1[-1]

                # Path 2: evolve then transform
                traj2 = run(state, t)
                result2 = mirror_swap(traj2[-1])

                assert result1 == result2, (
                    f"mirror_swap symmetry broken for state={state}, t={t}\n"
                    f"evolve(transform(S)) = {result1}\n"
                    f"transform(evolve(S)) = {result2}"
                )


class TestFalseSymmetries:
    """Test that false symmetries do NOT commute with evolution."""

    def test_mirror_only_does_not_commute(self):
        # Find a state where mirror_only breaks commutation
        state = ">...."
        t = 2

        result1 = run(mirror_only(state), t)[-1]
        result2 = mirror_only(run(state, t)[-1])

        # These should differ for asymmetric states
        assert result1 != result2

    def test_swap_only_does_not_commute(self):
        state = ">...."
        t = 2

        result1 = run(swap_only(state), t)[-1]
        result2 = swap_only(run(state, t)[-1])

        assert result1 != result2


class TestShiftSymmetry:
    """Test that shift_k is a true symmetry."""

    def test_shift_basic(self):
        # Use valid state symbols
        # shift_k(state, k) takes last k chars and moves them to front
        # ">..<." with k=1: last char "." + first 4 ">..<" = ".>..<"
        assert shift_k(">..<.", 1) == ".>..<"
        # ">..<." with k=2: last 2 chars "<." + first 3 ">.." = "<.>.."
        assert shift_k(">..<.", 2) == "<.>.."
        assert shift_k(">..<.", 0) == ">..<."
        assert shift_k(">..<.", 5) == ">..<."  # Full rotation

    def test_shift_commutes(self):
        test_states = [
            "..><..",
            ">....<",
            "..X..",
        ]

        for state in test_states:
            for k in [1, 2, len(state) // 2]:
                for t in [1, 5]:
                    result1 = run(shift_k(state, k), t)[-1]
                    result2 = shift_k(run(state, t)[-1], k)

                    assert result1 == result2, (
                        f"shift_k symmetry broken for state={state}, k={k}, t={t}"
                    )


class TestObservables:
    def test_count_symbol(self):
        state = "..><.X.."
        assert count_symbol(state, ".") == 5
        assert count_symbol(state, ">") == 1
        assert count_symbol(state, "<") == 1
        assert count_symbol(state, "X") == 1

    def test_count_invalid_symbol(self):
        with pytest.raises(ValueError):
            count_symbol("...", "a")

    def test_grid_length(self):
        assert grid_length("....") == 4
        assert grid_length("") == 0

    def test_derived_observables(self):
        state = "..><.X.."  # 1 >, 1 <, 1 X
        assert right_component(state) == 2  # > + X
        assert left_component(state) == 2   # < + X
        assert particle_count(state) == 4   # > + < + 2*X
        assert momentum(state) == 0         # > - <
        assert occupied_cells(state) == 3   # > + < + X


class TestDeterminism:
    """Verify the simulator is deterministic."""

    def test_same_input_same_output(self):
        state = "..><.X..>>><<<."
        traj1 = run(state, 50)
        traj2 = run(state, 50)
        assert traj1 == traj2

    def test_step_deterministic(self):
        state = "..X..><."
        assert step(state) == step(state)
