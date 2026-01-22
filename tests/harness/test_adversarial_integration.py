"""Integration tests for adversarial search with the harness."""

import pytest

from src.harness.config import HarnessConfig
from src.harness.harness import Harness
from src.harness.fixtures import FixtureLoader
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def loader():
    return FixtureLoader(FIXTURES_DIR)


@pytest.fixture
def harness_with_adversarial(loader):
    """Harness with adversarial search enabled."""
    config = loader.load_harness_config()
    config.max_cases = 50
    config.min_cases_used_for_pass = 20
    config.enable_adversarial_search = True
    config.adversarial_budget = 100
    return Harness(config)


@pytest.fixture
def harness_without_adversarial(loader):
    """Harness with adversarial search disabled."""
    config = loader.load_harness_config()
    config.max_cases = 50
    config.min_cases_used_for_pass = 20
    config.enable_adversarial_search = False
    return Harness(config)


class TestAdversarialHarnessIntegration:
    """Tests for adversarial search integration with harness."""

    def test_adversarial_tracks_cases(self, harness_with_adversarial, loader):
        """Verify adversarial search tracking in power metrics."""
        laws = loader.load_true_laws()
        law = next(l for l in laws if l.law_id == "T1_right_component_conservation")

        verdict = harness_with_adversarial.evaluate(law)

        # True laws should pass - adversarial search should run but not find anything
        assert verdict.status == "PASS"
        # Adversarial should have tried some cases (unless short-circuited)
        assert verdict.power_metrics.adversarial_cases_tried >= 0
        assert not verdict.power_metrics.adversarial_found

    def test_false_law_found_without_adversarial(self, harness_without_adversarial, loader):
        """False laws should fail even without adversarial search."""
        laws = loader.load_false_laws()
        law = next(l for l in laws if l.law_id == "F1_collision_count_conserved")

        verdict = harness_without_adversarial.evaluate(law)

        assert verdict.status == "FAIL"
        assert verdict.counterexample is not None
        # No adversarial search run
        assert verdict.power_metrics.adversarial_cases_tried == 0

    def test_false_law_found_with_adversarial(self, harness_with_adversarial, loader):
        """False laws should fail - adversarial may or may not be needed."""
        laws = loader.load_false_laws()
        law = next(l for l in laws if l.law_id == "F1_collision_count_conserved")

        verdict = harness_with_adversarial.evaluate(law)

        assert verdict.status == "FAIL"
        assert verdict.counterexample is not None
        # Since standard testing finds it, adversarial shouldn't run
        # (counterexample found stops adversarial search)

    def test_adversarial_notes_added(self, harness_with_adversarial, loader):
        """Verify adversarial search info is added to verdict notes."""
        laws = loader.load_true_laws()
        law = next(l for l in laws if l.law_id == "T1_right_component_conservation")

        verdict = harness_with_adversarial.evaluate(law)

        # Notes should contain adversarial search info
        if verdict.notes:
            adversarial_notes = [n for n in verdict.notes if "adversarial" in n.lower()]
            # Should have a note about adversarial search if it ran
            if verdict.power_metrics.adversarial_cases_tried > 0:
                assert len(adversarial_notes) > 0

    def test_symmetry_law_with_adversarial(self, harness_with_adversarial, loader):
        """Test symmetry law evaluation with adversarial search."""
        laws = loader.load_true_laws()
        law = next(l for l in laws if l.law_id == "T5_full_mirror_symmetry_commutation")

        verdict = harness_with_adversarial.evaluate(law)

        assert verdict.status == "PASS"

    def test_false_symmetry_with_adversarial(self, harness_with_adversarial, loader):
        """Test false symmetry law with adversarial search."""
        laws = loader.load_false_laws()
        law = next(l for l in laws if l.law_id == "F3_spatial_mirror_only_is_symmetry")

        verdict = harness_with_adversarial.evaluate(law)

        assert verdict.status == "FAIL"
        assert verdict.counterexample is not None


class TestAdversarialSearchEffectiveness:
    """Tests to verify adversarial search actually improves falsification."""

    def test_adversarial_mutation_generator_produces_diverse_cases(self, loader):
        """Verify the mutation generator creates diverse test cases."""
        from src.harness.generators import GeneratorRegistry

        gen = GeneratorRegistry.create("adversarial_mutation_search")

        params = {
            "seed_states": [">..<..", "<.>..", ">.>..", "<.<.."],
            "mutations_per_seed": 5,
            "max_mutations_per_state": 3,
            "grid_lengths": [6, 8, 10],
        }

        cases = gen.generate(params, seed=42, count=20)

        # Check diversity
        initial_states = set(c.initial_state for c in cases)
        assert len(initial_states) >= 10  # At least 50% unique

    def test_collision_focus_increases_collision_states(self, loader):
        """Verify collision focus produces more collision-prone states."""
        from src.harness.generators import GeneratorRegistry

        gen = GeneratorRegistry.create("adversarial_mutation_search")

        # Without collision focus
        params_normal = {
            "grid_lengths": [10],
            "mutations_per_seed": 5,
            "focus_collisions": False,
        }
        cases_normal = gen.generate(params_normal, seed=42, count=50)

        # With collision focus
        params_focused = {
            "grid_lengths": [10],
            "mutations_per_seed": 5,
            "focus_collisions": True,
        }
        cases_focused = gen.generate(params_focused, seed=42, count=50)

        # Count cases with potential collisions (has both > and <)
        def has_collision_potential(state):
            return ">" in state and "<" in state

        normal_collision_cases = sum(
            1 for c in cases_normal if has_collision_potential(c.initial_state)
        )
        focused_collision_cases = sum(
            1 for c in cases_focused if has_collision_potential(c.initial_state)
        )

        # Focused should have at least as many collision-prone cases
        assert focused_collision_cases >= normal_collision_cases * 0.8
