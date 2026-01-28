"""End-to-end tests for the probe system.

Tests the full cycle: define probe -> propose law -> evaluate -> verdict.
"""

import pytest

from src.claims.schema import (
    CandidateLaw,
    ComparisonOp,
    MonotoneDirection,
    Observable,
    Precondition,
    Quantifiers,
    Template,
)
from src.claims.compiler import ClaimCompiler
from src.claims.templates import InvariantChecker, MonotoneChecker, BoundChecker
from src.probes.registry import ProbeRegistry
from src.probes.sandbox import ProbeValidationError
from src.universe.simulator import run as sim_run


class TestProbeInvariantLaw:
    """Test: define probe for total particle count, propose invariant, evaluate."""

    def test_true_invariant_with_probe_passes(self):
        """Total non-background cells is conserved -> should PASS."""
        reg = ProbeRegistry()
        defn = reg.register(
            probe_id="total_particles",
            source="def probe(S):\n    return sum(1 for c in S if c != '.')",
            hypothesis="Total particle count",
        )
        assert defn.status == "active"

        # Create an invariant law referencing the probe
        law = CandidateLaw(
            law_id="test_inv_probe",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="P", probe_id="total_particles")],
            claim="P(t) == P(0) for all t",
            forbidden="exists t where P(t) != P(0)",
        )

        # Compile with probe registry
        compiler = ClaimCompiler(probe_registry=reg)
        checker = compiler.compile(law)
        assert isinstance(checker, InvariantChecker)

        # Run simulation and check
        initial_state = "..><.."
        trajectory = sim_run(initial_state, 20)
        result = checker.check(trajectory)
        # Total particles (>, <, X) should be conserved in kinetic grid
        assert result.passed is True

    def test_false_invariant_with_probe_fails(self):
        """Count of '>' alone is NOT conserved (collisions create X) -> should FAIL."""
        reg = ProbeRegistry()
        reg.register(
            probe_id="count_right",
            source="def probe(S):\n    return sum(1 for c in S if c == '>')",
            hypothesis="Count of rightward-moving particles",
        )

        law = CandidateLaw(
            law_id="test_false_inv",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="R", probe_id="count_right")],
            claim="R(t) == R(0) for all t",
            forbidden="exists t where R(t) != R(0)",
        )

        compiler = ClaimCompiler(probe_registry=reg)
        checker = compiler.compile(law)

        # Use a state that will produce collisions:
        # ">.<" creates collision at t=1: ".X."
        initial_state = ">.<"
        trajectory = sim_run(initial_state, 10)
        result = checker.check(trajectory)
        # At t=1, '>' count drops from 1 to 0 (collision produces X)
        assert result.passed is False
        assert result.violation is not None


class TestProbeBoundLaw:

    def test_bound_with_probe(self):
        """Grid length is always >= 1."""
        reg = ProbeRegistry()
        reg.register(
            probe_id="grid_len",
            source="def probe(S):\n    return len(S)",
        )

        law = CandidateLaw(
            law_id="test_bound",
            template=Template.BOUND,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="L", probe_id="grid_len")],
            claim="L(t) >= 1 for all t",
            forbidden="exists t where L(t) < 1",
            bound_value=1,
            bound_op=ComparisonOp.GE,
        )

        compiler = ClaimCompiler(probe_registry=reg)
        checker = compiler.compile(law)
        assert isinstance(checker, BoundChecker)

        trajectory = sim_run("..><..", 10)
        result = checker.check(trajectory)
        assert result.passed is True


class TestProbeMonotoneLaw:

    def test_monotone_with_probe(self):
        """Background count is NOT monotone (collisions create X then separate)."""
        reg = ProbeRegistry()
        reg.register(
            probe_id="bg_count",
            source="def probe(S):\n    return sum(1 for c in S if c == '.')",
        )

        law = CandidateLaw(
            law_id="test_mono",
            template=Template.MONOTONE,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="BG", probe_id="bg_count")],
            claim="BG is non-increasing",
            forbidden="exists t where BG(t+1) > BG(t)",
            direction=MonotoneDirection.NON_INCREASING,
        )

        compiler = ClaimCompiler(probe_registry=reg)
        checker = compiler.compile(law)

        # ">.<" has collision then separation:
        # t=0: ">.<" -> bg=1
        # t=1: ".X." -> bg=2 (increases!)
        # t=2: "<.>" -> bg=1 (decreases)
        # So background count oscillates non-monotonically
        trajectory = sim_run(">.<", 10)
        result = checker.check(trajectory)
        # This should fail because bg goes 1 -> 2 (increase violates non-increasing)
        assert result.passed is False


class TestProbeErrorHandling:

    def test_errored_probe_prevents_compilation(self):
        """Law with errored probe should fail to compile."""
        reg = ProbeRegistry()
        defn = reg.register(
            probe_id="bad_probe",
            source="import os\ndef probe(S):\n    return 1",
        )
        assert defn.status == "error"

        law = CandidateLaw(
            law_id="test_error",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="X", probe_id="bad_probe")],
            claim="X(t) == X(0)",
            forbidden="exists t where X(t) != X(0)",
        )

        compiler = ClaimCompiler(probe_registry=reg)
        # _make_probe_evaluator returns None for errored probes
        # And the observable has no expr, so compilation should fail
        from src.claims.compiler import CompilationError
        with pytest.raises(CompilationError):
            compiler.compile(law)


class TestProbeWithExpressionFallback:

    def test_mixed_observables(self):
        """Law with both probe-based and expression-based observables."""
        reg = ProbeRegistry()
        reg.register(
            probe_id="custom_metric",
            source="def probe(S):\n    return sum(1 for c in S if c != '.')",
        )

        law = CandidateLaw(
            law_id="test_mixed",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            # Only one observable for invariant, but test that probe-based works
            observables=[Observable(name="P", probe_id="custom_metric")],
            claim="P(t) == P(0) for all t",
            forbidden="exists t where P(t) != P(0)",
        )

        compiler = ClaimCompiler(probe_registry=reg)
        checker = compiler.compile(law)

        trajectory = sim_run("..><..", 10)
        result = checker.check(trajectory)
        assert result.passed is True


class TestProbeOutputTable:

    def test_execute_all_active_on_trajectory(self):
        """Run all active probes on each state of a trajectory."""
        reg = ProbeRegistry()
        reg.register("count_bg", "def probe(S):\n    return sum(1 for c in S if c == '.')")
        reg.register("count_right", "def probe(S):\n    return sum(1 for c in S if c == '>')")
        reg.register("grid_len", "def probe(S):\n    return len(S)")

        trajectory = sim_run("..><..", 5)

        # Build probe output table
        table = {}
        for state in trajectory:
            results = reg.execute_all_active(list(state))
            for pid, val in results.items():
                table.setdefault(pid, []).append(val)

        assert "count_bg" in table
        assert "count_right" in table
        assert "grid_len" in table
        assert len(table["grid_len"]) == len(trajectory)
        # Grid length is constant
        assert all(v == 6 for v in table["grid_len"])


# ---------------------------------------------------------------------------
# Temporal probe end-to-end tests
# ---------------------------------------------------------------------------

class TestTemporalProbeInvariant:
    """Test temporal probes in invariant laws."""

    def test_temporal_invariant_pass(self):
        """A temporal probe measuring a conserved transition quantity -> PASS.

        The number of cells that remain unchanged between any two consecutive
        steps plus the number of cells that change should equal grid length.
        This is trivially an invariant.
        """
        reg = ProbeRegistry()
        defn = reg.register(
            probe_id="total_changes_plus_stable",
            source="def probe(S_cur, S_nxt):\n    return len(S_cur)",
            hypothesis="Grid length measured via transition (always grid length)",
        )
        assert defn.status == "active"
        assert defn.arity == 2

        law = CandidateLaw(
            law_id="test_temporal_inv_pass",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="M", probe_id="total_changes_plus_stable")],
            claim="M(t) == M(0) for all t",
            forbidden="exists t where M(t) != M(0)",
        )

        compiler = ClaimCompiler(probe_registry=reg)
        checker = compiler.compile(law)
        assert isinstance(checker, InvariantChecker)

        trajectory = sim_run("..><..", 20)
        result = checker.check(trajectory)
        assert result.passed is True

    def test_temporal_invariant_fail(self):
        """A temporal probe measuring a non-conserved transition -> FAIL.

        Number of cells that changed should vary across timesteps for
        a trajectory with collisions.
        """
        reg = ProbeRegistry()
        reg.register(
            probe_id="changed_cell_count",
            source="def probe(S_cur, S_nxt):\n    return sum(1 for a, b in zip(S_cur, S_nxt) if a != b)",
            hypothesis="Counts cells that changed",
        )

        law = CandidateLaw(
            law_id="test_temporal_inv_fail",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="D", probe_id="changed_cell_count")],
            claim="D(t) == D(0) for all t",
            forbidden="exists t where D(t) != D(0)",
        )

        compiler = ClaimCompiler(probe_registry=reg)
        checker = compiler.compile(law)

        # ">.<" produces collision then separation -> varying change counts
        trajectory = sim_run(">.<", 10)
        result = checker.check(trajectory)
        # The number of changed cells varies, so invariant should fail
        assert result.passed is False

    def test_mixed_temporal_and_single(self):
        """Verify that a single-state probe still works alongside temporal probes."""
        reg = ProbeRegistry()
        reg.register(
            "single_count",
            "def probe(S):\n    return sum(1 for c in S if c != '.')",
        )
        reg.register(
            "temporal_len",
            "def probe(S_cur, S_nxt):\n    return len(S_cur)",
        )

        # Test single-state probe in a law
        law_single = CandidateLaw(
            law_id="test_single",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="P", probe_id="single_count")],
            claim="P(t) == P(0) for all t",
            forbidden="exists t where P(t) != P(0)",
        )

        compiler = ClaimCompiler(probe_registry=reg)
        checker = compiler.compile(law_single)
        trajectory = sim_run("..><..", 10)
        result = checker.check(trajectory)
        assert result.passed is True

        # Test temporal probe in a law
        law_temporal = CandidateLaw(
            law_id="test_temporal",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="M", probe_id="temporal_len")],
            claim="M(t) == M(0) for all t",
            forbidden="exists t where M(t) != M(0)",
        )

        checker2 = compiler.compile(law_temporal)
        result2 = checker2.check(trajectory)
        assert result2.passed is True


class TestTemporalProbeBound:

    def test_temporal_bound_pass(self):
        """Change count is bounded above by grid length."""
        reg = ProbeRegistry()
        reg.register(
            "change_count",
            "def probe(S_cur, S_nxt):\n    return sum(1 for a, b in zip(S_cur, S_nxt) if a != b)",
        )

        law = CandidateLaw(
            law_id="test_temporal_bound",
            template=Template.BOUND,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="D", probe_id="change_count")],
            claim="D(t) <= 6 for all t",
            forbidden="exists t where D(t) > 6",
            bound_value=6,
            bound_op=ComparisonOp.LE,
        )

        compiler = ClaimCompiler(probe_registry=reg)
        checker = compiler.compile(law)
        trajectory = sim_run("..><..", 10)
        result = checker.check(trajectory)
        assert result.passed is True
