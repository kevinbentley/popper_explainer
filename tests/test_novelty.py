"""Tests for novelty tracking system."""

import pytest

from src.claims.schema import (
    CandidateLaw,
    Observable,
    Quantifiers,
    Template,
)
from src.discovery.novelty import (
    NoveltyResult,
    NoveltyStats,
    NoveltyTracker,
    ProbeSuite,
    SemanticEvaluator,
    SemanticSignature,
    check_law_novelty,
    get_novelty_stats,
    is_discovery_saturated,
    reset_novelty_tracker,
)


def make_invariant_law(
    law_id: str,
    observable_name: str,
    observable_expr: str,
) -> CandidateLaw:
    """Helper to create a simple invariant law."""
    return CandidateLaw(
        law_id=law_id,
        template=Template.INVARIANT,
        quantifiers=Quantifiers(T=10),
        observables=[Observable(name=observable_name, expr=observable_expr)],
        claim=f"{observable_name}(t) == {observable_name}(0)",
        forbidden=f"{observable_name} changes over time",
        claim_ast={
            "op": "==",
            "lhs": {"obs": observable_name, "t": {"var": "t"}},
            "rhs": {"obs": observable_name, "t": {"const": 0}},
        },
    )


def make_implication_law(
    law_id: str,
    obs1_name: str,
    obs1_expr: str,
    obs2_name: str,
    obs2_expr: str,
) -> CandidateLaw:
    """Helper to create a simple implication law."""
    return CandidateLaw(
        law_id=law_id,
        template=Template.IMPLICATION_STEP,
        quantifiers=Quantifiers(T=10),
        observables=[
            Observable(name=obs1_name, expr=obs1_expr),
            Observable(name=obs2_name, expr=obs2_expr),
        ],
        claim=f"{obs1_name}(t) > 0 => {obs2_name}(t+1) == 0",
        forbidden="Antecedent true but consequent false",
        claim_ast={
            "op": "=>",
            "lhs": {"op": ">", "lhs": {"obs": obs1_name, "t": {"var": "t"}}, "rhs": {"const": 0}},
            "rhs": {"op": "==", "lhs": {"obs": obs2_name, "t": {"t_plus_1": True}}, "rhs": {"const": 0}},
        },
    )


class TestProbeSuite:
    """Tests for probe suite generation."""

    def test_probe_suite_creation(self):
        """Probe suite should generate trajectories for all initial states."""
        suite = ProbeSuite()
        trajectories = suite.get_all_trajectories()

        # Should have trajectories
        assert len(trajectories) > 0

        # Each trajectory should have expected length
        for state, traj in trajectories.items():
            if state:  # Non-empty
                assert len(traj) > 1

    def test_probe_suite_custom_states(self):
        """Probe suite should work with custom initial states."""
        custom_states = [">.<", "..><..", ">..."]
        suite = ProbeSuite(initial_states=custom_states, time_horizon=5)

        trajectories = suite.get_all_trajectories()
        assert len(trajectories) == len(custom_states)

    def test_probe_suite_trajectory_retrieval(self):
        """Should retrieve specific trajectories by initial state."""
        suite = ProbeSuite()
        traj = suite.get_trajectory(">.....")

        assert traj is not None
        assert traj[0] == ">....."
        assert len(traj) > 1


class TestSemanticEvaluator:
    """Tests for semantic evaluation."""

    def test_compute_signature_invariant(self):
        """Should compute semantic signature for invariant law."""
        evaluator = SemanticEvaluator()
        law = make_invariant_law("test_law", "R", "count('>')")

        signature = evaluator.compute_signature(law)

        assert isinstance(signature, SemanticSignature)
        assert signature.signature_hash is not None
        assert len(signature.signature_hash) == 24

    def test_compute_signature_implication(self):
        """Should compute semantic signature for implication law."""
        evaluator = SemanticEvaluator()
        law = make_implication_law(
            "test_law",
            "X_count", "count('X')",
            "R_count", "count('>')",
        )

        signature = evaluator.compute_signature(law)

        assert isinstance(signature, SemanticSignature)
        assert "results" in signature.behavior_summary

    def test_different_laws_different_signatures(self):
        """Different laws should have different semantic signatures."""
        evaluator = SemanticEvaluator()

        # Use different templates for genuinely different behavior
        # Invariant: N(t) == N(0)
        # Monotone: N(t+1) >= N(t)
        law1 = make_invariant_law("law1", "R", "count('>')")

        # Create a monotone law manually
        law2 = CandidateLaw(
            law_id="law2",
            template=Template.MONOTONE,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="R", expr="count('>')")],
            claim="R(t+1) >= R(t)",
            forbidden="R decreases",
            claim_ast={
                "op": ">=",
                "lhs": {"obs": "R", "t": {"t_plus_1": True}},
                "rhs": {"obs": "R", "t": {"var": "t"}},
            },
        )

        sig1 = evaluator.compute_signature(law1)
        sig2 = evaluator.compute_signature(law2)

        # Different templates should produce different signatures
        assert sig1.signature_hash != sig2.signature_hash

    def test_equivalent_laws_same_signature(self):
        """Semantically equivalent laws should have same signature."""
        evaluator = SemanticEvaluator()

        # Same observable with different names
        law1 = make_invariant_law("law1", "RightCount", "count('>')")
        law2 = make_invariant_law("law2", "R_total", "count('>')")

        sig1 = evaluator.compute_signature(law1)
        sig2 = evaluator.compute_signature(law2)

        # These should be the same (same observable expression)
        assert sig1.signature_hash == sig2.signature_hash


class TestNoveltyTracker:
    """Tests for novelty tracking."""

    def test_first_law_is_novel(self):
        """First law should always be novel."""
        tracker = NoveltyTracker()

        law = make_invariant_law("first_law", "R", "count('>')")
        result = tracker.check_novelty(law)

        assert result.is_novel
        assert result.is_syntactically_novel
        assert result.is_semantically_novel
        assert result.is_fully_novel

    def test_duplicate_law_not_novel(self):
        """Exact duplicate should not be novel."""
        tracker = NoveltyTracker()

        law1 = make_invariant_law("law1", "R", "count('>')")
        law2 = make_invariant_law("law2", "R", "count('>')")  # Same observable

        result1 = tracker.check_novelty(law1)
        result2 = tracker.check_novelty(law2)

        assert result1.is_novel
        assert not result2.is_syntactically_novel or not result2.is_semantically_novel

    def test_syntactically_different_semantically_same(self):
        """Laws with different syntax but same semantics should be detected."""
        tracker = NoveltyTracker()

        # Same semantic meaning, different names
        law1 = make_invariant_law("law1", "R", "count('>')")
        law2 = make_invariant_law("law2", "RightParticles", "count('>')")

        result1 = tracker.check_novelty(law1)
        result2 = tracker.check_novelty(law2)

        assert result1.is_novel
        # Second law should have same semantic signature
        assert not result2.is_semantically_novel

    def test_window_stats(self):
        """Should compute correct window statistics."""
        tracker = NoveltyTracker(window_size=10)

        # Add some laws
        for i in range(5):
            law = make_invariant_law(f"law_{i}", f"Obs{i}", f"count('>') + {i}")
            tracker.check_novelty(law)

        stats = tracker.get_window_stats()

        assert stats.total_laws == 5
        assert stats.window_size == 10

    def test_saturation_detection(self):
        """Should detect saturation when novelty rate drops."""
        tracker = NoveltyTracker(window_size=10, saturation_threshold=0.3)

        # Add some unique laws first
        for i in range(5):
            law = make_invariant_law(f"unique_{i}", f"Obs{i}", f"count('>') + {i}")
            tracker.check_novelty(law)

        assert not tracker.is_saturated()

        # Now add duplicates (same observable expression)
        for i in range(10):
            law = make_invariant_law(f"dup_{i}", "R", "count('>')")
            tracker.check_novelty(law)

        # Should be saturated now
        assert tracker.is_saturated()

    def test_saturation_report(self):
        """Should generate saturation report."""
        tracker = NoveltyTracker()

        law = make_invariant_law("test_law", "R", "count('>')")
        tracker.check_novelty(law)

        report = tracker.get_saturation_report()

        assert "is_saturated" in report
        assert "saturation_threshold" in report
        assert "window_stats" in report
        assert "total_laws_seen" in report

    def test_reset(self):
        """Should reset all tracking state."""
        tracker = NoveltyTracker()

        law = make_invariant_law("test_law", "R", "count('>')")
        tracker.check_novelty(law)

        assert tracker._total_laws == 1

        tracker.reset()

        assert tracker._total_laws == 0
        assert len(tracker._seen_syntactic) == 0
        assert len(tracker._seen_semantic) == 0

    def test_seed_known_fingerprints(self):
        """Should seed with known fingerprints from previous runs."""
        tracker = NoveltyTracker()

        # Seed with a known fingerprint
        tracker.seed_known_fingerprints(
            syntactic_fps={"known_fp_123"},
            semantic_hashes={"known_hash_456"},
        )

        # These should be in the sets
        assert "known_fp_123" in tracker._seen_syntactic
        assert "known_hash_456" in tracker._seen_semantic


class TestNoveltyStats:
    """Tests for NoveltyStats calculations."""

    def test_novelty_rates(self):
        """Should calculate novelty rates correctly."""
        stats = NoveltyStats(
            window_size=10,
            total_laws=10,
            syntactically_novel=8,
            semantically_novel=6,
            fully_novel=5,
        )

        assert stats.syntactic_novelty_rate == 0.8
        assert stats.semantic_novelty_rate == 0.6
        assert stats.full_novelty_rate == 0.5

    def test_combined_novelty_rate(self):
        """Should calculate combined novelty rate correctly."""
        stats = NoveltyStats(
            window_size=10,
            total_laws=10,
            syntactically_novel=8,
            semantically_novel=6,
            fully_novel=4,
        )

        # novel_either = 8 + 6 - 4 = 10
        assert stats.combined_novelty_rate == 1.0

    def test_empty_stats(self):
        """Should handle empty stats gracefully."""
        stats = NoveltyStats(
            window_size=10,
            total_laws=0,
            syntactically_novel=0,
            semantically_novel=0,
            fully_novel=0,
        )

        assert stats.syntactic_novelty_rate == 1.0  # Default when no data
        assert stats.semantic_novelty_rate == 1.0
        assert stats.combined_novelty_rate == 1.0

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        stats = NoveltyStats(
            window_size=10,
            total_laws=5,
            syntactically_novel=4,
            semantically_novel=3,
            fully_novel=2,
        )

        d = stats.to_dict()

        assert d["window_size"] == 10
        assert d["total_laws"] == 5
        assert "syntactic_novelty_rate" in d
        assert "semantic_novelty_rate" in d


class TestNoveltyResultSerialization:
    """Tests for serialization of novelty results."""

    def test_novelty_result_to_dict(self):
        """NoveltyResult should serialize correctly."""
        result = NoveltyResult(
            syntactic_fingerprint="abc123",
            is_syntactically_novel=True,
            semantic_signature=SemanticSignature(
                signature_hash="def456",
                behavior_summary={"test": True},
            ),
            is_semantically_novel=True,
            is_novel=True,
            is_fully_novel=True,
            reason="test reason",
        )

        d = result.to_dict()

        assert d["syntactic_fingerprint"] == "abc123"
        assert d["is_syntactically_novel"] is True
        assert d["semantic_signature"]["signature_hash"] == "def456"
        assert d["is_novel"] is True


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset tracker before each test."""
        reset_novelty_tracker()

    def test_check_law_novelty(self):
        """Module-level novelty check should work."""
        law = make_invariant_law("test_law", "R", "count('>')")
        result = check_law_novelty(law)

        assert isinstance(result, NoveltyResult)
        assert result.is_novel

    def test_get_novelty_stats(self):
        """Module-level stats retrieval should work."""
        law = make_invariant_law("test_law", "R", "count('>')")
        check_law_novelty(law)

        stats = get_novelty_stats()

        assert isinstance(stats, NoveltyStats)
        assert stats.total_laws == 1

    def test_is_discovery_saturated(self):
        """Module-level saturation check should work."""
        assert not is_discovery_saturated()


class TestEdgeCases:
    """Tests for edge cases in novelty tracking."""

    def test_law_without_ast(self):
        """Should handle laws without claim_ast."""
        tracker = NoveltyTracker()

        law = CandidateLaw(
            law_id="no_ast_law",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="R", expr="count('>')")],
            claim="R(t) == R(0)",
            forbidden="R changes",
            claim_ast=None,  # No AST
        )

        result = tracker.check_novelty(law)

        # Should still produce a result
        assert result is not None
        assert result.syntactic_fingerprint is not None

    def test_law_with_invalid_observable(self):
        """Should handle laws with invalid observables gracefully."""
        tracker = NoveltyTracker()

        law = CandidateLaw(
            law_id="invalid_obs_law",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            observables=[Observable(name="R", expr="invalid_function('x')")],
            claim="R(t) == R(0)",
            forbidden="R changes",
            claim_ast={
                "op": "==",
                "lhs": {"obs": "R", "t": {"var": "t"}},
                "rhs": {"obs": "R", "t": {"const": 0}},
            },
        )

        result = tracker.check_novelty(law)

        # Should produce a result even if evaluation fails
        assert result is not None
        assert result.semantic_signature is not None
        # Error signatures should still be unique enough
        assert result.semantic_signature.signature_hash.startswith("err_")

    def test_empty_observables(self):
        """Should handle laws with no observables."""
        tracker = NoveltyTracker()

        law = CandidateLaw(
            law_id="empty_obs_law",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=10),
            observables=[],  # No observables
            claim="true",
            forbidden="false",
            claim_ast={"const": 1},
        )

        result = tracker.check_novelty(law)

        assert result is not None
