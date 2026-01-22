"""Tests for the test harness (Phase 3)."""

import tempfile
from pathlib import Path

import pytest

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
from src.db.repo import Repository
from src.harness.case import Case
from src.harness.config import HarnessConfig
from src.harness.generators import (
    ConstrainedPairGenerator,
    EdgeWrappingGenerator,
    GeneratorRegistry,
    RandomDensityGenerator,
    SymmetryMetamorphicGenerator,
)
from src.harness.harness import Harness
from src.harness.power import PowerMetrics
from src.harness.verdict import LawVerdict, ReasonCode
from src.universe.types import Config


class TestCase:
    """Tests for Case data structure."""

    def test_case_creation(self):
        case = Case(
            initial_state="..><..",
            config=Config(grid_length=6),
            seed=42,
            generator_family="test",
            params_hash="abc123",
        )
        assert case.initial_state == "..><.."
        assert case.config.grid_length == 6

    def test_case_content_hash(self):
        case1 = Case("..><..", Config(6), 42, "test", "abc")
        case2 = Case("..><..", Config(6), 42, "test", "abc")
        case3 = Case("..><..", Config(6), 43, "test", "abc")  # Different seed

        assert case1.content_hash() == case2.content_hash()
        assert case1.content_hash() != case3.content_hash()

    def test_case_serialization(self):
        case = Case("..><..", Config(6), 42, "test", "abc", {"key": "value"})
        d = case.to_dict()
        restored = Case.from_dict(d)

        assert restored.initial_state == case.initial_state
        assert restored.config.grid_length == case.config.grid_length
        assert restored.metadata == case.metadata


class TestGenerators:
    """Tests for case generators."""

    def test_random_density_generator(self):
        gen = RandomDensityGenerator()
        assert gen.family_name() == "random_density_sweep"

        cases = gen.generate(
            {"densities": [0.3], "grid_lengths": [20]},
            seed=42,
            count=10,
        )

        assert len(cases) == 10
        for case in cases:
            assert len(case.initial_state) == 20
            assert case.generator_family == "random_density_sweep"

    def test_random_density_reproducible(self):
        gen = RandomDensityGenerator()
        cases1 = gen.generate({"densities": [0.3]}, seed=42, count=5)
        cases2 = gen.generate({"densities": [0.3]}, seed=42, count=5)

        for c1, c2 in zip(cases1, cases2):
            assert c1.initial_state == c2.initial_state

    def test_constrained_pairs_generator(self):
        gen = ConstrainedPairGenerator()
        assert gen.family_name() == "constrained_pair_interactions"

        cases = gen.generate(
            {"patterns": ["approaching", "collision"], "grid_lengths": [16]},
            seed=42,
            count=10,
        )

        assert len(cases) == 10
        # Should contain the patterns
        patterns_found = set()
        for case in cases:
            if ">.<" in case.initial_state or ">..<" in case.initial_state:
                patterns_found.add("approaching")
            if "X" in case.initial_state:
                patterns_found.add("collision")

        assert "approaching" in patterns_found or "collision" in patterns_found

    def test_edge_wrapping_generator(self):
        gen = EdgeWrappingGenerator()
        assert gen.family_name() == "edge_wrapping_cases"

        cases = gen.generate({"grid_lengths": [16]}, seed=42, count=10)

        assert len(cases) == 10
        # Check that some cases have particles near edges
        edge_cases = 0
        for case in cases:
            state = case.initial_state
            if state[0] in "><" or state[-1] in "><":
                edge_cases += 1
        assert edge_cases > 0

    def test_symmetry_metamorphic_generator(self):
        gen = SymmetryMetamorphicGenerator()
        assert gen.family_name() == "symmetry_metamorphic_suite"

        cases = gen.generate(
            {"grid_lengths": [16], "transform": "mirror_swap"},
            seed=42,
            count=10,
        )

        assert len(cases) == 10

    def test_generator_registry(self):
        assert "random_density_sweep" in GeneratorRegistry.list_available()
        assert "constrained_pair_interactions" in GeneratorRegistry.list_available()

        gen = GeneratorRegistry.create("random_density_sweep")
        assert gen is not None
        assert isinstance(gen, RandomDensityGenerator)


class TestHarnessConfig:
    """Tests for harness configuration."""

    def test_default_config(self):
        config = HarnessConfig()
        assert config.seed == 42
        assert config.max_cases == 300
        assert config.min_cases_used_for_pass == 50

    def test_config_hash_stable(self):
        config = HarnessConfig(seed=100)
        h1 = config.content_hash()
        h2 = config.content_hash()
        assert h1 == h2

    def test_config_hash_changes(self):
        config1 = HarnessConfig(seed=100)
        config2 = HarnessConfig(seed=200)
        assert config1.content_hash() != config2.content_hash()


class TestPowerMetrics:
    """Tests for power metrics."""

    def test_coverage_computation(self):
        metrics = PowerMetrics(
            cases_attempted=100,
            cases_used=80,
            cases_with_collisions=40,
            density_bins_hit=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        coverage = metrics.compute_coverage()
        assert 0 <= coverage <= 1
        assert coverage > 0.5  # Good coverage

    def test_serialization(self):
        metrics = PowerMetrics(cases_attempted=50, cases_used=40)
        d = metrics.to_dict()
        restored = PowerMetrics.from_dict(d)
        assert restored.cases_attempted == 50
        assert restored.cases_used == 40


class TestHarnessEvaluation:
    """Integration tests for harness evaluation."""

    @pytest.fixture
    def harness(self):
        config = HarnessConfig(
            seed=42,
            max_cases=100,
            min_cases_used_for_pass=20,
            default_T=30,
        )
        return Harness(config)

    def test_evaluate_true_invariant(self, harness):
        """Test that a true conservation law passes."""
        law = CandidateLaw(
            law_id="r_conservation",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=30),
            observables=[Observable(name="R_total", expr="count('>') + count('X')")],
            claim="R_total(t) == R_total(0)",
            forbidden="exists t where R_total(t) != R_total(0)",
        )

        verdict = harness.evaluate(law)
        assert verdict.status == "PASS"
        assert verdict.counterexample is None
        assert verdict.power_metrics.cases_used > 0

    def test_evaluate_false_invariant(self, harness):
        """Test that a false invariant fails with counterexample."""
        law = CandidateLaw(
            law_id="collision_count_invariant",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="X_count", expr="count('X')")],
            claim="X_count(t) == X_count(0)",
            forbidden="exists t where X_count(t) != X_count(0)",
        )

        verdict = harness.evaluate(law)
        assert verdict.status == "FAIL"
        assert verdict.counterexample is not None
        assert verdict.counterexample.t_fail >= 0

    def test_evaluate_true_symmetry(self, harness):
        """Test that mirror_swap symmetry passes."""
        law = CandidateLaw(
            law_id="mirror_swap_symmetry",
            template=Template.SYMMETRY_COMMUTATION,
            quantifiers=Quantifiers(T=20),
            observables=[],
            claim="commutes(transform='mirror_swap')",
            forbidden="...",
            transform="mirror_swap",
        )

        verdict = harness.evaluate(law)
        assert verdict.status == "PASS"

    def test_evaluate_false_symmetry(self, harness):
        """Test that mirror_only (false symmetry) fails."""
        law = CandidateLaw(
            law_id="mirror_only_not_symmetry",
            template=Template.SYMMETRY_COMMUTATION,
            quantifiers=Quantifiers(T=10),
            observables=[],
            claim="commutes(transform='mirror_only')",
            forbidden="...",
            transform="mirror_only",
        )

        verdict = harness.evaluate(law)
        assert verdict.status == "FAIL"
        assert verdict.counterexample is not None

    def test_missing_transform_returns_unknown(self, harness):
        """Test that missing transform returns UNKNOWN."""
        law = CandidateLaw(
            law_id="missing_transform",
            template=Template.SYMMETRY_COMMUTATION,
            quantifiers=Quantifiers(T=10),
            observables=[],
            claim="commutes(transform='rotate_2d')",
            forbidden="...",
            transform="rotate_2d",
            capability_requirements=CapabilityRequirements(missing_transforms=["rotate_2d"]),
        )

        verdict = harness.evaluate(law)
        assert verdict.status == "UNKNOWN"
        assert verdict.reason_code == ReasonCode.MISSING_TRANSFORM

    def test_missing_observable_returns_unknown(self, harness):
        """Test that missing observable returns UNKNOWN."""
        law = CandidateLaw(
            law_id="missing_obs",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="weird", expr="count('X')")],
            claim="weird(t) == weird(0)",
            forbidden="...",
            capability_requirements=CapabilityRequirements(
                missing_observables=["collision_events"]
            ),
        )

        verdict = harness.evaluate(law)
        assert verdict.status == "UNKNOWN"
        assert verdict.reason_code == ReasonCode.MISSING_OBSERVABLE

    def test_precondition_filtering(self, harness):
        """Test that preconditions filter cases correctly."""
        law = CandidateLaw(
            law_id="large_grid_only",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=20),
            preconditions=[
                Precondition(lhs="grid_length", op=ComparisonOp.GE, rhs=100)
            ],
            observables=[Observable(name="R", expr="count('>')")],
            claim="R(t) == R(0)",
            forbidden="...",
        )

        # With default generators using small grids, preconditions won't be met
        config = HarnessConfig(
            seed=42,
            max_cases=50,
            min_cases_used_for_pass=10,
        )
        harness_strict = Harness(config)
        verdict = harness_strict.evaluate(law)

        # Should be UNKNOWN due to unmet preconditions
        assert verdict.status == "UNKNOWN"
        assert verdict.reason_code == ReasonCode.UNMET_PRECONDITIONS


class TestHarnessWithPersistence:
    """Tests for harness with database persistence."""

    @pytest.fixture
    def db_harness(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        repo = Repository(db_path)
        repo.connect()

        config = HarnessConfig(seed=42, max_cases=50, min_cases_used_for_pass=10)
        harness = Harness(config, repo)

        yield harness, repo

        repo.close()
        db_path.unlink()

    def test_evaluation_persisted(self, db_harness):
        harness, repo = db_harness

        law = CandidateLaw(
            law_id="persist_test",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="R", expr="count('>')")],
            claim="R(t) == R(0)",
            forbidden="...",
        )

        # First, persist the law
        from src.db.models import LawRecord
        repo.insert_law(LawRecord(
            law_id=law.law_id,
            law_hash=law.content_hash(),
            template=law.template.value,
            law_json="{}",
        ))

        verdict = harness.evaluate(law)

        # Check that evaluation was persisted
        eval_record = repo.get_latest_evaluation(law.law_id)
        assert eval_record is not None
        assert eval_record.status == verdict.status

        # Check audit log
        logs = repo.get_audit_logs(operation="evaluate")
        assert len(logs) > 0


class TestBoundTemplate:
    """Tests for bound template evaluation."""

    def test_bound_passes(self):
        config = HarnessConfig(seed=42, max_cases=50, min_cases_used_for_pass=10)
        harness = Harness(config)

        law = CandidateLaw(
            law_id="bound_test",
            template=Template.BOUND,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="particles", expr="count('>') + count('<')")],
            claim="particles(t) <= 100",
            forbidden="exists t where particles(t) > 100",
            bound_value=100,
            bound_op=ComparisonOp.LE,
        )

        verdict = harness.evaluate(law)
        assert verdict.status == "PASS"

    def test_bound_fails(self):
        config = HarnessConfig(seed=42, max_cases=50, min_cases_used_for_pass=10)
        harness = Harness(config)

        # Very restrictive bound that will be violated
        law = CandidateLaw(
            law_id="bound_fail_test",
            template=Template.BOUND,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="particles", expr="count('>') + count('<')")],
            claim="particles(t) <= 0",
            forbidden="exists t where particles(t) > 0",
            bound_value=0,
            bound_op=ComparisonOp.LE,
        )

        verdict = harness.evaluate(law)
        # Should fail because random density generator will create particles
        assert verdict.status == "FAIL"


class TestMonotoneTemplate:
    """Tests for monotone template evaluation."""

    def test_monotone_compilation_and_eval(self):
        config = HarnessConfig(seed=42, max_cases=50, min_cases_used_for_pass=10)
        harness = Harness(config)

        # This monotone law is FALSE - collision count can increase
        law = CandidateLaw(
            law_id="monotone_test",
            template=Template.MONOTONE,
            quantifiers=Quantifiers(T=20),
            observables=[Observable(name="X_count", expr="count('X')")],
            claim="X_count(t+1) <= X_count(t)",
            forbidden="exists t where X_count(t+1) > X_count(t)",
            direction=MonotoneDirection.NON_INCREASING,
        )

        verdict = harness.evaluate(law)
        # Collision count can increase when particles collide
        assert verdict.status == "FAIL"
