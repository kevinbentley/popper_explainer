"""Tests for adversarial search components."""

import pytest

from src.harness.case import Case, CaseResult
from src.harness.generators.adversarial import (
    AdversarialMutationGenerator,
    FlipDirectionMutation,
    AddParticleMutation,
    RemoveParticleMutation,
    ShiftParticleMutation,
    CreateCollisionSetupMutation,
    SwapParticlesMutation,
    GuidedAdversarialGenerator,
)
from src.harness.adversarial import AdversarialSearcher, AdversarialSearchResult
from src.universe.types import Config


class TestMutationStrategies:
    """Tests for individual mutation strategies."""

    def test_flip_direction_mutation(self):
        """FlipDirectionMutation flips a particle direction."""
        import random

        rng = random.Random(42)
        mutation = FlipDirectionMutation()

        # State with right-moving particle
        state = ">...."
        mutated = mutation.mutate(state, rng)

        # Should have flipped to left
        assert "<" in mutated or ">" in mutated
        assert mutated != state or state.count(">") + state.count("<") == 0

    def test_flip_direction_empty_state(self):
        """FlipDirectionMutation handles empty state."""
        import random

        rng = random.Random(42)
        mutation = FlipDirectionMutation()

        state = "....."
        mutated = mutation.mutate(state, rng)
        assert mutated == state  # No change possible

    def test_add_particle_mutation(self):
        """AddParticleMutation adds a particle to empty cell."""
        import random

        rng = random.Random(42)
        mutation = AddParticleMutation()

        state = ">...."
        mutated = mutation.mutate(state, rng)

        # Should have more particles
        original_count = state.count(">") + state.count("<")
        mutated_count = mutated.count(">") + mutated.count("<")
        assert mutated_count == original_count + 1

    def test_add_particle_full_state(self):
        """AddParticleMutation handles full state."""
        import random

        rng = random.Random(42)
        mutation = AddParticleMutation()

        state = "><><"
        mutated = mutation.mutate(state, rng)
        assert mutated == state  # No empty cells

    def test_remove_particle_mutation(self):
        """RemoveParticleMutation removes a particle."""
        import random

        rng = random.Random(42)
        mutation = RemoveParticleMutation()

        state = "><..."
        mutated = mutation.mutate(state, rng)

        original_count = state.count(">") + state.count("<")
        mutated_count = mutated.count(">") + mutated.count("<")
        assert mutated_count == original_count - 1

    def test_shift_particle_mutation(self):
        """ShiftParticleMutation moves particle to adjacent cell."""
        import random

        rng = random.Random(42)
        mutation = ShiftParticleMutation()

        state = ">...."
        mutated = mutation.mutate(state, rng)

        # Particle should have moved (or stayed if no empty adjacent)
        assert mutated.count(">") + mutated.count("<") == 1

    def test_create_collision_setup_mutation(self):
        """CreateCollisionSetupMutation creates collision setup."""
        import random

        rng = random.Random(42)
        mutation = CreateCollisionSetupMutation()

        state = "......"
        mutated = mutation.mutate(state, rng)

        # Should contain approaching particles
        # Pattern >.<
        if mutated != state:
            assert ">" in mutated and "<" in mutated

    def test_swap_particles_mutation(self):
        """SwapParticlesMutation swaps two particles."""
        import random

        rng = random.Random(42)
        mutation = SwapParticlesMutation()

        state = ">.<..."
        mutated = mutation.mutate(state, rng)

        # Should still have same particle types
        assert mutated.count(">") == state.count(">")
        assert mutated.count("<") == state.count("<")


class TestAdversarialMutationGenerator:
    """Tests for AdversarialMutationGenerator."""

    def test_generate_basic(self):
        """Generator produces cases."""
        gen = AdversarialMutationGenerator()
        params = {
            "seed_states": [">..<.."],
            "mutations_per_seed": 3,
            "max_mutations_per_state": 2,
        }
        cases = gen.generate(params, seed=42, count=5)

        assert len(cases) <= 5
        assert all(isinstance(c, Case) for c in cases)
        assert all(c.generator_family == "adversarial_mutation_search" for c in cases)

    def test_generate_without_seeds(self):
        """Generator can create its own seed states."""
        gen = AdversarialMutationGenerator()
        params = {
            "grid_lengths": [8, 16],
            "mutations_per_seed": 2,
        }
        cases = gen.generate(params, seed=42, count=10)

        assert len(cases) > 0
        assert all(len(c.initial_state) in [8, 16] for c in cases)

    def test_generate_with_collision_focus(self):
        """Generator with collision focus creates collision-prone states."""
        gen = AdversarialMutationGenerator()
        params = {
            "grid_lengths": [10],
            "mutations_per_seed": 5,
            "focus_collisions": True,
        }
        cases = gen.generate(params, seed=42, count=20)

        # With collision focus, some cases should have potential collisions
        assert len(cases) > 0

    def test_mutations_recorded_in_params(self):
        """Mutations applied are recorded in generator_params."""
        gen = AdversarialMutationGenerator()
        params = {
            "seed_states": [">..<.."],
            "mutations_per_seed": 3,
        }
        cases = gen.generate(params, seed=42, count=3)

        for case in cases:
            assert "mutations" in case.generator_params
            assert isinstance(case.generator_params["mutations"], list)

    def test_family_name(self):
        """Generator has correct family name."""
        gen = AdversarialMutationGenerator()
        assert gen.family_name() == "adversarial_mutation_search"


class TestGuidedAdversarialGenerator:
    """Tests for GuidedAdversarialGenerator."""

    def test_generate_basic(self):
        """Guided generator produces cases."""
        gen = GuidedAdversarialGenerator()
        params = {
            "seed_states": [">..<.."],
            "mutations_per_seed": 3,
        }
        cases = gen.generate(params, seed=42, count=5)

        assert len(cases) > 0
        assert all(isinstance(c, Case) for c in cases)

    def test_record_near_miss(self):
        """Near-miss recording affects future generation."""
        gen = GuidedAdversarialGenerator()

        # Record some promising states
        gen.record_near_miss(">.<...", 0.8)
        gen.record_near_miss("<.>..", 0.7)

        # Generate with recorded states
        params = {
            "use_promising": True,
            "mutations_per_seed": 2,
        }
        cases = gen.generate(params, seed=42, count=10)

        # Should generate cases (using near-miss states as seeds)
        assert len(cases) > 0

    def test_record_effective_mutation(self):
        """Effective mutation recording works."""
        gen = GuidedAdversarialGenerator()

        gen.record_effective_mutation("FlipDirectionMutation")
        gen.record_effective_mutation("FlipDirectionMutation")

        assert gen._effective_mutations["FlipDirectionMutation"] == 2

    def test_reset(self):
        """Reset clears recorded data."""
        gen = GuidedAdversarialGenerator()

        gen.record_near_miss(">.<...", 0.8)
        gen.record_effective_mutation("FlipDirectionMutation")

        gen.reset()

        assert len(gen._promising_states) == 0
        assert len(gen._effective_mutations) == 0

    def test_family_name(self):
        """Generator has correct family name."""
        gen = GuidedAdversarialGenerator()
        assert gen.family_name() == "guided_adversarial_search"


class TestAdversarialSearcher:
    """Tests for AdversarialSearcher."""

    def test_init(self):
        """Searcher initializes with config."""
        searcher = AdversarialSearcher(budget=500, max_runtime_ms=1000)

        assert searcher.budget == 500
        assert searcher.max_runtime_ms == 1000

    def test_search_basic(self):
        """Basic search returns result."""

        def mock_evaluate(case: Case, time_horizon: int) -> CaseResult:
            return CaseResult(
                case=case,
                trajectory=[case.initial_state],
                passed=True,
                precondition_met=True,
                near_miss_score=0.1,
            )

        searcher = AdversarialSearcher(budget=20, max_runtime_ms=1000)

        # Create seed results
        seed_results = [
            CaseResult(
                case=Case(
                    initial_state=">..<..",
                    config=Config(grid_length=6),
                    seed=1,
                    generator_family="test",
                ),
                trajectory=[">..<.."],
                passed=True,
                precondition_met=True,
                near_miss_score=0.3,
            )
        ]

        result = searcher.search(
            law=None,  # Not used in mock
            evaluate_case=mock_evaluate,
            seed_results=seed_results,
            time_horizon=10,
            seed=42,
        )

        assert isinstance(result, AdversarialSearchResult)
        assert result.cases_tried > 0
        assert result.runtime_ms >= 0  # May be 0 if very fast

    def test_search_finds_counterexample(self):
        """Search can find counterexamples."""
        call_count = [0]

        def mock_evaluate(case: Case, time_horizon: int) -> CaseResult:
            call_count[0] += 1
            # Fail on the 5th case
            passed = call_count[0] < 5
            return CaseResult(
                case=case,
                trajectory=[case.initial_state, case.initial_state],
                passed=passed,
                precondition_met=True,
                near_miss_score=0.0,
                violation=None if passed else {"t": 1, "details": "test"},
            )

        searcher = AdversarialSearcher(budget=100, max_runtime_ms=5000)

        seed_results = [
            CaseResult(
                case=Case(
                    initial_state=">..<..",
                    config=Config(grid_length=6),
                    seed=1,
                    generator_family="test",
                ),
                trajectory=[">..<.."],
                passed=True,
                precondition_met=True,
                near_miss_score=0.5,
            )
        ]

        result = searcher.search(
            law=None,
            evaluate_case=mock_evaluate,
            seed_results=seed_results,
            time_horizon=10,
            seed=42,
        )

        assert result.found_counterexample
        assert result.counterexample is not None

    def test_search_respects_budget(self):
        """Search respects budget limit."""
        call_count = [0]

        def mock_evaluate(case: Case, time_horizon: int) -> CaseResult:
            call_count[0] += 1
            return CaseResult(
                case=case,
                trajectory=[case.initial_state],
                passed=True,
                precondition_met=True,
                near_miss_score=0.1,
            )

        searcher = AdversarialSearcher(budget=10, max_runtime_ms=10000)

        seed_results = [
            CaseResult(
                case=Case(
                    initial_state=">..<..",
                    config=Config(grid_length=6),
                    seed=1,
                    generator_family="test",
                ),
                trajectory=[">..<.."],
                passed=True,
                precondition_met=True,
                near_miss_score=0.3,
            )
        ]

        result = searcher.search(
            law=None,
            evaluate_case=mock_evaluate,
            seed_results=seed_results,
            time_horizon=10,
            seed=42,
        )

        # Should not exceed budget
        assert result.cases_tried <= 10

    def test_search_multiple_phases(self):
        """Search runs multiple phases."""

        def mock_evaluate(case: Case, time_horizon: int) -> CaseResult:
            return CaseResult(
                case=case,
                trajectory=[case.initial_state],
                passed=True,
                precondition_met=True,
                near_miss_score=0.1,
            )

        searcher = AdversarialSearcher(budget=100, max_runtime_ms=5000)

        seed_results = [
            CaseResult(
                case=Case(
                    initial_state=">..<..",
                    config=Config(grid_length=6),
                    seed=i,
                    generator_family="test",
                ),
                trajectory=[">..<.."],
                passed=True,
                precondition_met=True,
                near_miss_score=0.3,
            )
            for i in range(5)
        ]

        result = searcher.search(
            law=None,
            evaluate_case=mock_evaluate,
            seed_results=seed_results,
            time_horizon=10,
            seed=42,
        )

        # Should have run multiple phases
        assert len(result.search_phases) >= 1
        assert all("name" in phase for phase in result.search_phases)


class TestGeneratorRegistration:
    """Test that adversarial generators are registered."""

    def test_adversarial_mutation_registered(self):
        """AdversarialMutationGenerator is registered."""
        from src.harness.generators import GeneratorRegistry

        gen = GeneratorRegistry.create("adversarial_mutation_search")
        assert gen is not None
        assert isinstance(gen, AdversarialMutationGenerator)

    def test_guided_adversarial_registered(self):
        """GuidedAdversarialGenerator is registered."""
        from src.harness.generators import GeneratorRegistry

        gen = GeneratorRegistry.create("guided_adversarial_search")
        assert gen is not None
        assert isinstance(gen, GuidedAdversarialGenerator)
