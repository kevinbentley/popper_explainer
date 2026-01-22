"""Regression tests using fixture laws.

These tests verify that:
- True laws (T1-T5) pass
- False laws (F1-F4) fail with counterexamples
- Unknown laws (U1-U5) return UNKNOWN with correct reason codes
"""

from pathlib import Path

import pytest

from src.harness.config import HarnessConfig
from src.harness.fixtures import FixtureLoader
from src.harness.harness import Harness
from src.harness.verdict import ReasonCode


# Fixture directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def loader():
    """Create a fixture loader."""
    return FixtureLoader(FIXTURES_DIR)


@pytest.fixture
def harness(loader):
    """Create a harness with fixture config."""
    config = loader.load_harness_config()
    # Reduce cases for faster tests
    config.max_cases = 150
    config.min_cases_used_for_pass = 30
    return Harness(config)


class TestTrueLaws:
    """Tests for laws that should PASS."""

    def test_t1_right_component_conservation(self, harness, loader):
        """Right component (count('>') + count('X')) is conserved."""
        laws = loader.load_true_laws()
        law = next(l for l in laws if l.law_id == "T1_right_component_conservation")

        verdict = harness.evaluate(law)

        assert verdict.status == "PASS", f"Expected PASS, got {verdict.status}: {verdict.notes}"
        assert verdict.counterexample is None
        assert verdict.power_metrics.cases_used >= 30

    def test_t2_left_component_conservation(self, harness, loader):
        """Left component (count('<') + count('X')) is conserved."""
        laws = loader.load_true_laws()
        law = next(l for l in laws if l.law_id == "T2_left_component_conservation")

        verdict = harness.evaluate(law)

        assert verdict.status == "PASS", f"Expected PASS, got {verdict.status}: {verdict.notes}"
        assert verdict.counterexample is None

    def test_t3_particle_count_conservation(self, harness, loader):
        """Total particle count is conserved."""
        laws = loader.load_true_laws()
        law = next(l for l in laws if l.law_id == "T3_particle_count_conservation")

        verdict = harness.evaluate(law)

        assert verdict.status == "PASS", f"Expected PASS, got {verdict.status}: {verdict.notes}"

    def test_t4_momentum_conservation(self, harness, loader):
        """Momentum (R - L) is conserved."""
        laws = loader.load_true_laws()
        law = next(l for l in laws if l.law_id == "T4_momentum_conservation")

        verdict = harness.evaluate(law)

        assert verdict.status == "PASS", f"Expected PASS, got {verdict.status}: {verdict.notes}"

    def test_t5_full_mirror_symmetry_commutation(self, harness, loader):
        """mirror_swap is a true symmetry (commutes with evolution)."""
        laws = loader.load_true_laws()
        law = next(l for l in laws if l.law_id == "T5_full_mirror_symmetry_commutation")

        verdict = harness.evaluate(law)

        assert verdict.status == "PASS", f"Expected PASS, got {verdict.status}: {verdict.notes}"


class TestFalseLaws:
    """Tests for laws that should FAIL with counterexamples."""

    def test_f1_collision_count_conserved(self, harness, loader):
        """Collision count is NOT conserved - should fail."""
        laws = loader.load_false_laws()
        law = next(l for l in laws if l.law_id == "F1_collision_count_conserved")

        verdict = harness.evaluate(law)

        assert verdict.status == "FAIL", f"Expected FAIL, got {verdict.status}"
        assert verdict.counterexample is not None, "Expected a counterexample"
        # Verify counterexample has required fields
        assert verdict.counterexample.initial_state is not None
        assert verdict.counterexample.t_fail >= 0

    def test_f2_occupied_cells_conserved(self, harness, loader):
        """Occupied cell count is NOT conserved - should fail."""
        laws = loader.load_false_laws()
        law = next(l for l in laws if l.law_id == "F2_occupied_cells_conserved")

        verdict = harness.evaluate(law)

        assert verdict.status == "FAIL", f"Expected FAIL, got {verdict.status}"
        assert verdict.counterexample is not None

    def test_f3_spatial_mirror_only_is_symmetry(self, harness, loader):
        """mirror_only is NOT a symmetry - should fail."""
        laws = loader.load_false_laws()
        law = next(l for l in laws if l.law_id == "F3_spatial_mirror_only_is_symmetry")

        verdict = harness.evaluate(law)

        assert verdict.status == "FAIL", f"Expected FAIL, got {verdict.status}"
        assert verdict.counterexample is not None

    def test_f4_swap_only_is_symmetry(self, harness, loader):
        """swap_only is NOT a symmetry - should fail."""
        laws = loader.load_false_laws()
        law = next(l for l in laws if l.law_id == "F4_swap_only_is_symmetry")

        verdict = harness.evaluate(law)

        assert verdict.status == "FAIL", f"Expected FAIL, got {verdict.status}"
        assert verdict.counterexample is not None


class TestUnknownLaws:
    """Tests for laws that should return UNKNOWN with specific reason codes."""

    def test_u1_missing_observable(self, harness, loader):
        """Missing observable 'collision_events' returns UNKNOWN."""
        laws = loader.load_unknown_laws()
        law = next(l for l in laws if l.law_id == "U1_requires_missing_observable_collision_events")

        verdict = harness.evaluate(law)

        assert verdict.status == "UNKNOWN", f"Expected UNKNOWN, got {verdict.status}"
        assert verdict.reason_code == ReasonCode.MISSING_OBSERVABLE

    def test_u2_missing_transform(self, harness, loader):
        """Missing transform 'rotate_2d' returns UNKNOWN."""
        laws = loader.load_unknown_laws()
        law = next(l for l in laws if l.law_id == "U2_requires_missing_transform_rotate_2d")

        verdict = harness.evaluate(law)

        assert verdict.status == "UNKNOWN", f"Expected UNKNOWN, got {verdict.status}"
        assert verdict.reason_code == ReasonCode.MISSING_TRANSFORM

    def test_u3_ambiguous_claim(self, harness, loader):
        """Ambiguous claim using indexing returns UNKNOWN."""
        laws = loader.load_unknown_laws()
        law = next(l for l in laws if l.law_id == "U3_ambiguous_claim_uses_indexing")

        verdict = harness.evaluate(law)

        assert verdict.status == "UNKNOWN", f"Expected UNKNOWN, got {verdict.status}"
        # Could be MISSING_OBSERVABLE or AMBIGUOUS_CLAIM
        assert verdict.reason_code in (
            ReasonCode.MISSING_OBSERVABLE,
            ReasonCode.AMBIGUOUS_CLAIM,
        )

    def test_u4_unmet_preconditions(self, harness, loader):
        """Impossible preconditions (grid_length < 0) returns UNKNOWN."""
        laws = loader.load_unknown_laws()
        law = next(l for l in laws if l.law_id == "U4_unmet_preconditions_impossible_grid")

        verdict = harness.evaluate(law)

        assert verdict.status == "UNKNOWN", f"Expected UNKNOWN, got {verdict.status}"
        assert verdict.reason_code == ReasonCode.UNMET_PRECONDITIONS

    def test_u5_resource_limit(self, harness, loader):
        """Huge T value gets capped by max_T, so actually passes.

        Note: The expectation says PASS because max_T caps the horizon.
        The law is actually TRUE (particle conservation), just with huge T.
        """
        laws = loader.load_unknown_laws()
        law = next(l for l in laws if l.law_id == "U5_resource_limit_huge_T")

        verdict = harness.evaluate(law)

        # This actually passes because:
        # 1. The law (particle conservation) is TRUE
        # 2. max_T caps the time horizon to a reasonable value
        assert verdict.status == "PASS", f"Expected PASS (T capped), got {verdict.status}"


class TestRegressionSuite:
    """Full regression suite running all fixture laws."""

    def test_all_true_laws_pass(self, harness, loader):
        """Verify all true laws pass."""
        laws = loader.load_true_laws()
        expectations = loader.load_expectations()

        for law in laws:
            verdict = harness.evaluate(law)
            expected = expectations.get(law.law_id, {})

            assert verdict.status == expected.get("status", "PASS"), (
                f"Law {law.law_id}: expected {expected.get('status')}, got {verdict.status}"
            )

    def test_all_false_laws_fail(self, harness, loader):
        """Verify all false laws fail with counterexamples."""
        laws = loader.load_false_laws()
        expectations = loader.load_expectations()

        for law in laws:
            verdict = harness.evaluate(law)
            expected = expectations.get(law.law_id, {})

            assert verdict.status == "FAIL", (
                f"Law {law.law_id}: expected FAIL, got {verdict.status}"
            )
            assert verdict.counterexample is not None, (
                f"Law {law.law_id}: expected counterexample"
            )

    def test_unknown_laws_return_unknown_or_expected(self, harness, loader):
        """Verify unknown laws return UNKNOWN (or expected status if different)."""
        laws = loader.load_unknown_laws()
        expectations = loader.load_expectations()

        for law in laws:
            verdict = harness.evaluate(law)
            expected = expectations.get(law.law_id, {})
            expected_status = expected.get("status", "UNKNOWN")

            assert verdict.status == expected_status, (
                f"Law {law.law_id}: expected {expected_status}, got {verdict.status}"
            )

            # Check reason code if expected
            expected_reason = expected.get("reason_code")
            if expected_reason and verdict.status == "UNKNOWN":
                assert verdict.reason_code is not None
                assert verdict.reason_code.value == expected_reason, (
                    f"Law {law.law_id}: expected reason {expected_reason}, "
                    f"got {verdict.reason_code.value}"
                )


class TestFixtureLoading:
    """Tests for fixture loading utilities."""

    def test_load_universe_contract(self, loader):
        contract = loader.load_universe_contract()
        assert contract["universe_id"] == "kinetic_grid_v1"
        assert "." in contract["symbols"]
        assert "mirror_swap" in contract["capabilities"]["transforms"]

    def test_load_harness_config(self, loader):
        config = loader.load_harness_config()
        assert config.seed == 1337
        assert config.max_cases == 300

    def test_load_true_laws(self, loader):
        laws = loader.load_true_laws()
        assert len(laws) == 5
        law_ids = [l.law_id for l in laws]
        assert "T1_right_component_conservation" in law_ids

    def test_load_false_laws(self, loader):
        laws = loader.load_false_laws()
        assert len(laws) == 4
        law_ids = [l.law_id for l in laws]
        assert "F1_collision_count_conserved" in law_ids

    def test_load_unknown_laws(self, loader):
        laws = loader.load_unknown_laws()
        assert len(laws) == 5

    def test_load_expectations(self, loader):
        expectations = loader.load_expectations()
        assert "T1_right_component_conservation" in expectations
        assert expectations["T1_right_component_conservation"]["status"] == "PASS"
        assert expectations["F1_collision_count_conserved"]["status"] == "FAIL"
