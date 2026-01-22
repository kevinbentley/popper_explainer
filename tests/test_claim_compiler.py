"""Tests for claim schema and template compilation (Phase 2B)."""

import pytest

from src.claims.compiler import ClaimCompiler, CompilationError, compile_precondition
from src.claims.schema import (
    CandidateLaw,
    CapabilityRequirements,
    ComparisonOp,
    MonotoneDirection,
    Observable,
    Precondition,
    Quantifiers,
    Template,
)
from src.claims.templates import (
    BoundChecker,
    CheckResult,
    EventuallyChecker,
    ImplicationStateChecker,
    ImplicationStepChecker,
    InvariantChecker,
    MonotoneChecker,
    SymmetryCommutationChecker,
)
from src.universe.simulator import run


class TestSchema:
    """Tests for the CandidateLaw schema."""

    def test_create_invariant_law(self):
        law = CandidateLaw(
            law_id="test_invariant",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[
                Observable(name="R_total", expr="count('>') + count('X')")
            ],
            claim="R_total(t) == R_total(0)",
            forbidden="exists t where R_total(t) != R_total(0)",
        )
        assert law.law_id == "test_invariant"
        assert law.template == Template.INVARIANT

    def test_create_symmetry_law(self):
        law = CandidateLaw(
            law_id="test_symmetry",
            template=Template.SYMMETRY_COMMUTATION,
            quantifiers=Quantifiers(T=40),
            observables=[],
            claim="commutes(transform='mirror_swap')",
            forbidden="exists t where evolve(mirror_swap(S), t) != mirror_swap(evolve(S, t))",
            transform="mirror_swap",
        )
        assert law.transform == "mirror_swap"

    def test_content_hash_stable(self):
        law = CandidateLaw(
            law_id="hash_test",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[Observable(name="R", expr="count('>')")],
            claim="R(t) == R(0)",
            forbidden="exists t where R(t) != R(0)",
        )
        h1 = law.content_hash()
        h2 = law.content_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_content_hash_differs_for_different_laws(self):
        law1 = CandidateLaw(
            law_id="law1",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[Observable(name="R", expr="count('>')")],
            claim="R(t) == R(0)",
            forbidden="exists t",
        )
        law2 = CandidateLaw(
            law_id="law2",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[Observable(name="L", expr="count('<')")],
            claim="L(t) == L(0)",
            forbidden="exists t",
        )
        assert law1.content_hash() != law2.content_hash()


class TestInvariantChecker:
    """Tests for invariant template."""

    def test_invariant_passes_for_conserved_quantity(self):
        # Right component is conserved
        law = CandidateLaw(
            law_id="r_conserved",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="R_total", expr="count('>') + count('X')")],
            claim="R_total(t) == R_total(0)",
            forbidden="exists t where R_total(t) != R_total(0)",
        )

        compiler = ClaimCompiler()
        checker = compiler.compile(law)
        assert isinstance(checker, InvariantChecker)

        # Run simulation
        trajectory = run("..><.X..", 20)
        result = checker.check(trajectory)
        assert result.passed

    def test_invariant_fails_for_non_conserved_quantity(self):
        # Collision count is NOT conserved
        law = CandidateLaw(
            law_id="collision_not_conserved",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="collision_count", expr="count('X')")],
            claim="collision_count(t) == collision_count(0)",
            forbidden="exists t where collision_count(t) != collision_count(0)",
        )

        compiler = ClaimCompiler()
        checker = compiler.compile(law)

        # Start with a state that will create/destroy collisions
        trajectory = run(".>.<..", 10)  # Collision will form
        result = checker.check(trajectory)
        assert not result.passed
        assert result.violation is not None


class TestMonotoneChecker:
    """Tests for monotone template."""

    def test_monotone_compilation(self):
        law = CandidateLaw(
            law_id="monotone_test",
            template=Template.MONOTONE,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="empty_count", expr="count('.')")],
            claim="empty_count(t+1) <= empty_count(t)",
            forbidden="exists t where empty_count(t+1) > empty_count(t)",
            direction=MonotoneDirection.NON_INCREASING,
        )

        compiler = ClaimCompiler()
        checker = compiler.compile(law)
        assert isinstance(checker, MonotoneChecker)

    def test_monotone_requires_direction(self):
        law = CandidateLaw(
            law_id="no_direction",
            template=Template.MONOTONE,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="x", expr="count('X')")],
            claim="x(t+1) <= x(t)",
            forbidden="exists t",
            # direction is missing
        )

        compiler = ClaimCompiler()
        with pytest.raises(CompilationError, match="direction"):
            compiler.compile(law)


class TestBoundChecker:
    """Tests for bound template."""

    def test_bound_passes(self):
        law = CandidateLaw(
            law_id="bound_test",
            template=Template.BOUND,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="particles", expr="count('>') + count('<')")],
            claim="particles(t) <= 10",
            forbidden="exists t where particles(t) > 10",
            bound_value=10,
            bound_op=ComparisonOp.LE,
        )

        compiler = ClaimCompiler()
        checker = compiler.compile(law)
        assert isinstance(checker, BoundChecker)

        trajectory = run(">.<.", 20)  # 2 particles
        result = checker.check(trajectory)
        assert result.passed

    def test_bound_fails(self):
        law = CandidateLaw(
            law_id="bound_fail",
            template=Template.BOUND,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="particles", expr="count('>') + count('<')")],
            claim="particles(t) <= 1",
            forbidden="exists t where particles(t) > 1",
            bound_value=1,
            bound_op=ComparisonOp.LE,
        )

        compiler = ClaimCompiler()
        checker = compiler.compile(law)

        trajectory = run(">.<.", 10)  # 2 particles > 1
        result = checker.check(trajectory)
        assert not result.passed


class TestImplicationCheckers:
    """Tests for implication templates."""

    def test_implication_state(self):
        # If collision_count > 0, then particle_count >= 2
        law = CandidateLaw(
            law_id="collision_implies_particles",
            template=Template.IMPLICATION_STATE,
            quantifiers=Quantifiers(T=20),
            observables=[
                Observable(name="collision_count", expr="count('X')"),
                Observable(name="particle_count", expr="count('>') + count('<') + 2 * count('X')"),
            ],
            claim="collision_count(t) > 0 -> particle_count(t) >= 2",
            forbidden="exists t where collision_count(t) > 0 and particle_count(t) < 2",
        )

        compiler = ClaimCompiler()
        checker = compiler.compile(law)
        assert isinstance(checker, ImplicationStateChecker)

        trajectory = run("..X..", 20)
        result = checker.check(trajectory)
        assert result.passed

    def test_implication_step(self):
        law = CandidateLaw(
            law_id="step_implication",
            template=Template.IMPLICATION_STEP,
            quantifiers=Quantifiers(T=20),
            observables=[
                Observable(name="collision_count", expr="count('X')"),
            ],
            claim="collision_count(t) > 0 -> collision_count(t+1) == 0",
            forbidden="exists t where collision_count(t) > 0 and collision_count(t+1) > 0",
        )

        compiler = ClaimCompiler()
        checker = compiler.compile(law)
        assert isinstance(checker, ImplicationStepChecker)

    def test_vacuity_tracking(self):
        # Test with a condition that's never true
        law = CandidateLaw(
            law_id="vacuous_test",
            template=Template.IMPLICATION_STATE,
            quantifiers=Quantifiers(T=10),
            observables=[
                Observable(name="collision_count", expr="count('X')"),
            ],
            claim="collision_count(t) > 100 -> collision_count(t) > 0",
            forbidden="exists t where collision_count(t) > 100 and collision_count(t) <= 0",
        )

        compiler = ClaimCompiler()
        checker = compiler.compile(law)

        # No state will have collision_count > 100
        trajectory = run("..X..", 10)
        result = checker.check(trajectory)
        assert result.passed
        assert result.vacuity.is_vacuous  # Antecedent was never true


class TestEventuallyChecker:
    """Tests for eventually template."""

    def test_eventually_compilation(self):
        # Simplified claim format: antecedent -> consequent
        # The "eventually" semantics are implied by the template type
        law = CandidateLaw(
            law_id="eventually_test",
            template=Template.EVENTUALLY,
            quantifiers=Quantifiers(T=20, H=5),
            observables=[
                Observable(name="collision_count", expr="count('X')"),
            ],
            claim="collision_count(t) > 0 -> collision_count(t) == 0",
            forbidden="exists t0 where collision_count(t0) > 0 and forall t in [t0..t0+H]: collision_count(t) > 0",
        )

        compiler = ClaimCompiler()
        checker = compiler.compile(law)
        assert isinstance(checker, EventuallyChecker)

    def test_eventually_requires_H(self):
        law = CandidateLaw(
            law_id="no_H",
            template=Template.EVENTUALLY,
            quantifiers=Quantifiers(T=20),  # H is missing
            observables=[Observable(name="x", expr="count('X')")],
            claim="x(t) > 0 -> eventually x == 0",
            forbidden="...",
        )

        compiler = ClaimCompiler()
        with pytest.raises(CompilationError, match="H"):
            compiler.compile(law)


class TestSymmetryChecker:
    """Tests for symmetry commutation template."""

    def test_symmetry_true_symmetry_passes(self):
        law = CandidateLaw(
            law_id="mirror_swap_symmetry",
            template=Template.SYMMETRY_COMMUTATION,
            quantifiers=Quantifiers(T=20),
            observables=[],
            claim="commutes(transform='mirror_swap')",
            forbidden="exists t where evolve(mirror_swap(S), t) != mirror_swap(evolve(S, t))",
            transform="mirror_swap",
        )

        compiler = ClaimCompiler()
        checker = compiler.compile(law)
        assert isinstance(checker, SymmetryCommutationChecker)

        # Test with various states
        for state in ["..><..", ">....<", "..X..", ".>.<.."]:
            trajectory = [state]  # Just the initial state
            result = checker.check(trajectory)
            assert result.passed, f"mirror_swap should commute for state {state}"

    def test_symmetry_false_symmetry_fails(self):
        law = CandidateLaw(
            law_id="mirror_only_not_symmetry",
            template=Template.SYMMETRY_COMMUTATION,
            quantifiers=Quantifiers(T=10),
            observables=[],
            claim="commutes(transform='mirror_only')",
            forbidden="exists t where evolve(mirror_only(S), t) != mirror_only(evolve(S, t))",
            transform="mirror_only",
        )

        compiler = ClaimCompiler()
        checker = compiler.compile(law)

        # mirror_only is NOT a true symmetry - should fail for asymmetric states
        trajectory = [">...."]
        result = checker.check(trajectory)
        assert not result.passed

    def test_symmetry_requires_transform(self):
        law = CandidateLaw(
            law_id="no_transform",
            template=Template.SYMMETRY_COMMUTATION,
            quantifiers=Quantifiers(T=10),
            observables=[],
            claim="commutes",
            forbidden="...",
            # transform is missing
        )

        compiler = ClaimCompiler()
        with pytest.raises(CompilationError, match="transform"):
            compiler.compile(law)

    def test_symmetry_unknown_transform_fails(self):
        law = CandidateLaw(
            law_id="unknown_transform",
            template=Template.SYMMETRY_COMMUTATION,
            quantifiers=Quantifiers(T=10),
            observables=[],
            claim="commutes(transform='rotate_2d')",
            forbidden="...",
            transform="rotate_2d",
        )

        compiler = ClaimCompiler()
        with pytest.raises(CompilationError, match="Unknown transform"):
            compiler.compile(law)


class TestPreconditionCompilation:
    """Tests for precondition compilation."""

    def test_precondition_grid_length(self):
        law = CandidateLaw(
            law_id="prec_test",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            preconditions=[Precondition(lhs="grid_length", op=ComparisonOp.GE, rhs=4)],
            observables=[Observable(name="R", expr="count('>')")],
            claim="R(t) == R(0)",
            forbidden="...",
        )

        checker = compile_precondition(law.preconditions[0], law)

        assert checker("....") is True  # length 4 >= 4
        assert checker(".....") is True  # length 5 >= 4
        assert checker("...") is False  # length 3 < 4

    def test_precondition_observable(self):
        law = CandidateLaw(
            law_id="prec_obs_test",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            preconditions=[Precondition(lhs="count('>')", op=ComparisonOp.GE, rhs=1)],
            observables=[Observable(name="R", expr="count('>')")],
            claim="R(t) == R(0)",
            forbidden="...",
        )

        checker = compile_precondition(law.preconditions[0], law)

        assert checker(">...") is True  # has > particle
        assert checker("....") is False  # no > particles


class TestCompilerErrors:
    """Tests for compiler error handling."""

    def test_invalid_observable_expression(self):
        law = CandidateLaw(
            law_id="invalid_expr",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="bad", expr="invalid syntax here")],
            claim="bad(t) == bad(0)",
            forbidden="...",
        )

        compiler = ClaimCompiler()
        with pytest.raises(CompilationError, match="Failed to parse"):
            compiler.compile(law)

    def test_invariant_requires_one_observable(self):
        law = CandidateLaw(
            law_id="no_obs",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            observables=[],  # No observables
            claim="something",
            forbidden="...",
        )

        compiler = ClaimCompiler()
        with pytest.raises(CompilationError, match="exactly 1 observable"):
            compiler.compile(law)
