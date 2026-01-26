"""Tests for the integrity checking and invariant unmasking module."""

import pytest

from src.claims.schema import (
    CandidateLaw,
    ComparisonOp,
    Observable,
    Precondition,
    Quantifiers,
    Template,
)
from src.harness.evaluator import Evaluator
from src.harness.integrity import (
    IntegrityChecker,
    IntegrityViolation,
    InvariantUnmasker,
    HALLUCINATION_BREAKERS,
    check_theorem_integrity,
)


class TestIntegrityChecker:
    """Tests for IntegrityChecker."""

    @pytest.fixture
    def checker(self):
        return IntegrityChecker()

    def test_ephemeral_violation_detected(self, checker):
        """Test that claims about X being static are flagged."""
        law = CandidateLaw(
            law_id="x_static_test",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[Observable(name="X", expr="count('X')")],
            claim="X remains static throughout evolution",
            forbidden="X changes",
        )

        violations = checker.check_law(law)

        # Should find ephemeral violation
        ephemeral_violations = [v for v in violations if "ephemeral" in v.violation_type]
        assert len(ephemeral_violations) > 0
        assert ephemeral_violations[0].severity == "error"

    def test_component_confusion_warning(self, checker):
        """Test that using count('>') for conservation is flagged."""
        law = CandidateLaw(
            law_id="right_conserved_wrong",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[Observable(name="RightMoversConserved", expr="count('>')")],
            claim="RightMoversConserved(t) == RightMoversConserved(0)",
            forbidden="Right movers change",
        )

        violations = checker.check_law(law)

        # Should find component confusion
        confusion_violations = [v for v in violations if "component" in v.violation_type]
        assert len(confusion_violations) > 0
        assert "RightComponent" in confusion_violations[0].suggested_fix

    def test_narrow_preconditions_info(self, checker):
        """Test that laws with many preconditions are flagged."""
        law = CandidateLaw(
            law_id="narrow_law",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            preconditions=[
                Precondition(lhs="CollisionCells", op=ComparisonOp.EQ, rhs=0),
                Precondition(lhs="IncomingCollisions", op=ComparisonOp.EQ, rhs=0),
            ],
            observables=[Observable(name="FreeMovers", expr="count('>') + count('<')")],
            claim="FreeMovers(t) == FreeMovers(0)",
            forbidden="Free movers change",
        )

        violations = checker.check_law(law)

        # Should find narrow preconditions warning
        narrow_violations = [v for v in violations if "narrow" in v.violation_type]
        assert len(narrow_violations) > 0

    def test_collision_avoidance_warning(self, checker):
        """Test that collision-avoiding preconditions are flagged."""
        law = CandidateLaw(
            law_id="collision_avoider",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            preconditions=[
                Precondition(lhs="CollisionCells", op=ComparisonOp.EQ, rhs=0),
            ],
            observables=[Observable(name="N", expr="count('>') + count('<')")],
            claim="N(t) == N(0)",
            forbidden="N changes",
        )

        violations = checker.check_law(law)

        # Should find collision avoidance warning
        avoidance_violations = [v for v in violations if "collision_avoidance" in v.violation_type]
        assert len(avoidance_violations) > 0

    def test_no_violations_for_correct_law(self, checker):
        """Test that correctly formulated laws pass integrity checks."""
        law = CandidateLaw(
            law_id="total_particles_conserved",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[
                Observable(name="TotalParticles", expr="count('>') + count('<') + 2*count('X')")
            ],
            claim="TotalParticles(t) == TotalParticles(0)",
            forbidden="Total particles change",
        )

        violations = checker.check_law(law)

        # Should have no errors or warnings (info is ok)
        errors_and_warnings = [v for v in violations if v.severity in ("error", "warning")]
        assert len(errors_and_warnings) == 0


class TestInvariantUnmasker:
    """Tests for InvariantUnmasker."""

    @pytest.fixture
    def evaluator(self):
        return Evaluator()

    @pytest.fixture
    def unmasker(self, evaluator):
        return InvariantUnmasker(evaluator)

    def test_should_unmask_preconditioned_law(self, unmasker):
        """Test that laws with collision-avoiding preconditions should be unmasked."""
        law = CandidateLaw(
            law_id="conditional_conservation",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            preconditions=[
                Precondition(lhs="CollisionCells", op=ComparisonOp.EQ, rhs=0),
            ],
            observables=[Observable(name="N", expr="count('>') + count('<')")],
            claim="N(t) == N(0)",
            forbidden="N changes",
        )

        assert unmasker.should_unmask(law) is True

    def test_should_not_unmask_unpreconditioned_law(self, unmasker):
        """Test that laws without preconditions are not unmasked."""
        law = CandidateLaw(
            law_id="universal_conservation",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[
                Observable(name="TotalParticles", expr="count('>') + count('<') + 2*count('X')")
            ],
            claim="TotalParticles(t) == TotalParticles(0)",
            forbidden="Total particles change",
        )

        assert unmasker.should_unmask(law) is False

    def test_should_not_unmask_implication(self, unmasker):
        """Test that implication laws are not unmasked (not universal templates)."""
        law = CandidateLaw(
            law_id="conditional_implication",
            template=Template.IMPLICATION_STEP,
            quantifiers=Quantifiers(T=50),
            preconditions=[
                Precondition(lhs="CollisionCells", op=ComparisonOp.EQ, rhs=0),
            ],
            observables=[Observable(name="N", expr="count('>') + count('<')")],
            claim="N(t) > 0 => N(t+1) > 0",
            forbidden="N becomes zero",
        )

        # Implication is not a universal template
        assert unmasker.should_unmask(law) is False


class TestHallucinationBreakers:
    """Tests for hallucination breaker patterns."""

    def test_x_persistent_breakers_exist(self):
        """Test that X-persistent hallucination breakers are defined."""
        assert "x_persistent" in HALLUCINATION_BREAKERS
        breaker = HALLUCINATION_BREAKERS["x_persistent"]
        assert "XX" in breaker["states"] or "...XX..." in breaker["states"]

    def test_conditional_conservation_breakers_exist(self):
        """Test that conditional conservation breakers are defined."""
        assert "conditional_conservation" in HALLUCINATION_BREAKERS
        breaker = HALLUCINATION_BREAKERS["conditional_conservation"]
        assert ">>.<<" in breaker["states"]

    def test_grid_boundary_breakers_exist(self):
        """Test that grid boundary breakers are defined."""
        assert "grid_boundaries" in HALLUCINATION_BREAKERS
        breaker = HALLUCINATION_BREAKERS["grid_boundaries"]
        assert "....>" in breaker["states"]
        assert "<...." in breaker["states"]


class TestTheoremIntegrityCheck:
    """Tests for theorem integrity checking."""

    def test_ephemeral_violation_in_theorem(self):
        """Test that theorems claiming X is static are flagged."""
        claim = "X is static and does not change during evolution"
        violations = check_theorem_integrity(claim, [])

        assert len(violations) > 0
        assert any("ephemeral" in v.violation_type for v in violations)

    def test_component_confusion_in_theorem(self):
        """Test that theorems confusing symbols with components are flagged."""
        claim = "count('>') is conserved during all time steps"
        violations = check_theorem_integrity(claim, [])

        assert len(violations) > 0
        assert any("component" in v.violation_type for v in violations)

    def test_no_violations_for_correct_theorem(self):
        """Test that correct theorems pass integrity checks."""
        claim = "TotalParticles = count('>') + count('<') + 2*count('X') is conserved"
        violations = check_theorem_integrity(claim, [])

        # Should have no violations
        assert len(violations) == 0
