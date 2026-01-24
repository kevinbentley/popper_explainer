"""Tests for counterexample canonicalization and failure key tracking."""

import pytest

from src.harness.counterexample_key import (
    FailureKey,
    FailureKeyResult,
    FailureKeyStats,
    FailureKeyTracker,
    canonical_rotation,
    canonical_trajectory_snippet,
    compute_failure_key,
    get_failure_key_stats,
    get_failure_key_tracker,
    group_failures_by_structure,
    observable_vector,
    reset_failure_key_tracker,
    track_failure,
)
from src.harness.verdict import Counterexample


class TestCanonicalRotation:
    """Tests for shift-invariant canonicalization."""

    def test_already_minimal(self):
        """State with dots at start - but rotation finds smaller."""
        # "..><." rotations: "..><.", ".><..", "><...", "<...>", "...><"
        # "...><" is lexicographically smallest (more dots before '>')
        assert canonical_rotation("..><.") == "...><"

    def test_needs_rotation(self):
        """State needs rotation to be minimal."""
        assert canonical_rotation(">.<..") == "..>.<"

    def test_single_particle(self):
        """Single particle finds minimal position."""
        assert canonical_rotation(">....") == "....>"

    def test_collision_position(self):
        """X particle in canonical position."""
        assert canonical_rotation("X....") == "....X"

    def test_empty_state(self):
        """Empty state is unchanged."""
        assert canonical_rotation("") == ""

    def test_single_char(self):
        """Single character is unchanged."""
        assert canonical_rotation(">") == ">"
        assert canonical_rotation("X") == "X"

    def test_all_same(self):
        """All same characters - any rotation is equivalent."""
        result = canonical_rotation(">>>")
        assert result == ">>>"

    def test_complex_state(self):
        """Complex state with multiple particles."""
        # "><.X." has rotations: "><.X.", "X.><.", ".><.X", etc.
        # Lexicographically smallest should start with '.'
        result = canonical_rotation("><.X.")
        assert result[0] == "."  # Should start with empty

    def test_symmetric_state(self):
        """Symmetric patterns."""
        # "><><" rotated: "><><", "<><>", "><><", "<><>"
        # '<' (ASCII 60) comes before '>' (ASCII 62), so "<><>" is smallest
        result = canonical_rotation("><><")
        assert result == "<><>"


class TestCanonicalTrajectorySnippet:
    """Tests for trajectory snippet canonicalization."""

    def test_basic_snippet(self):
        """Basic trajectory snippet extraction."""
        trajectory = [">....", ".>...", "..>..", "...>.", "....>"]
        snippet = canonical_trajectory_snippet(trajectory, center_t=2, window=1)

        # Should get 3 states centered on t=2
        assert len(snippet) == 3
        # Each should be canonicalized
        assert all(s == canonical_rotation(s) for s in snippet)

    def test_at_start(self):
        """Snippet at trajectory start."""
        trajectory = [">....", ".>...", "..>.."]
        snippet = canonical_trajectory_snippet(trajectory, center_t=0, window=2)

        # Should only have states from 0 to 2
        assert len(snippet) == 3

    def test_at_end(self):
        """Snippet at trajectory end."""
        trajectory = [">....", ".>...", "..>..", "...>.", "....>"]
        snippet = canonical_trajectory_snippet(trajectory, center_t=4, window=2)

        # Should only have states from 2 to 4
        assert len(snippet) == 3

    def test_empty_trajectory(self):
        """Empty trajectory returns empty snippet."""
        assert canonical_trajectory_snippet([], center_t=0, window=2) == []


class TestObservableVector:
    """Tests for observable vector creation."""

    def test_basic_vector(self):
        """Basic observable vector creation."""
        obs = {"R": 3, "L": 2, "X": 1}
        vec = observable_vector(obs)

        assert len(vec) == 3
        # Should be sorted by key
        assert vec[0][0] == "L"
        assert vec[1][0] == "R"
        assert vec[2][0] == "X"

    def test_none_observables(self):
        """None observables returns empty tuple."""
        assert observable_vector(None) == ()

    def test_empty_observables(self):
        """Empty dict returns empty tuple."""
        assert observable_vector({}) == ()

    def test_hashable(self):
        """Vector should be hashable."""
        obs = {"R": 3, "L": 2}
        vec = observable_vector(obs)
        # Should not raise
        hash(vec)


class TestComputeFailureKey:
    """Tests for failure key computation."""

    def test_basic_failure_key(self):
        """Compute basic failure key."""
        ce = Counterexample(
            initial_state=">.<..",
            config={"grid_length": 5, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=1,
            trajectory_excerpt=[">.<..", ".X..."],
            observables_at_fail={"R": 1, "L": 1, "X": 1},
        )
        key = compute_failure_key(ce)

        assert key.canonical_initial == canonical_rotation(">.<..")
        assert key.t_fail_relative == 1  # Early (t=1)
        assert key.key_hash is not None
        assert len(key.key_hash) == 24

    def test_same_pattern_same_key(self):
        """Same pattern at different positions should have same key."""
        ce1 = Counterexample(
            initial_state=">.<..",
            config={"grid_length": 5, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=1,
        )
        ce2 = Counterexample(
            initial_state="..>.<",  # Same pattern, rotated
            config={"grid_length": 5, "boundary": "periodic"},
            seed=99,
            t_max=10,
            t_fail=1,
        )

        key1 = compute_failure_key(ce1, include_trajectory=False, include_observables=False)
        key2 = compute_failure_key(ce2, include_trajectory=False, include_observables=False)

        # Should have same canonical initial
        assert key1.canonical_initial == key2.canonical_initial
        # And same key hash
        assert key1.key_hash == key2.key_hash

    def test_different_timing_different_key(self):
        """Different failure timing should produce different keys."""
        ce1 = Counterexample(
            initial_state=">.<..",
            config={"grid_length": 5, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=0,  # Immediate
        )
        ce2 = Counterexample(
            initial_state=">.<..",
            config={"grid_length": 5, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=10,  # Late
        )

        key1 = compute_failure_key(ce1, include_trajectory=False, include_observables=False)
        key2 = compute_failure_key(ce2, include_trajectory=False, include_observables=False)

        # Different timing classes
        assert key1.t_fail_relative != key2.t_fail_relative
        assert key1.key_hash != key2.key_hash

    def test_timing_classification(self):
        """Test timing classification buckets."""
        def get_timing(t_fail):
            ce = Counterexample(
                initial_state=">.<..",
                config={},
                seed=None,
                t_max=20,
                t_fail=t_fail,
            )
            return compute_failure_key(ce).t_fail_relative

        assert get_timing(0) == 0  # Immediate
        assert get_timing(1) == 1  # Early
        assert get_timing(2) == 1  # Early
        assert get_timing(3) == 2  # Mid
        assert get_timing(5) == 2  # Mid
        assert get_timing(6) == 3  # Late
        assert get_timing(100) == 3  # Late

    def test_observable_filter(self):
        """Observable filter should only include specified observables."""
        ce = Counterexample(
            initial_state=">.<..",
            config={},
            seed=None,
            t_max=10,
            t_fail=1,
            observables_at_fail={"R": 1, "L": 1, "X": 1, "total": 3},
        )

        key_all = compute_failure_key(ce, observable_filter=None)
        key_filtered = compute_failure_key(ce, observable_filter={"R", "L"})

        # Filtered should have fewer observables
        assert len(key_filtered.observable_signature) < len(key_all.observable_signature)


class TestFailureKeyTracker:
    """Tests for failure key tracking."""

    def test_first_failure_is_novel(self):
        """First failure should be novel."""
        tracker = FailureKeyTracker()

        ce = Counterexample(
            initial_state=">.<..",
            config={"grid_length": 5},
            seed=42,
            t_max=10,
            t_fail=1,
        )
        result = tracker.track(ce)

        assert result.is_novel
        assert result.occurrence_count == 1

    def test_duplicate_failure_not_novel(self):
        """Duplicate failure should not be novel."""
        tracker = FailureKeyTracker()

        ce1 = Counterexample(
            initial_state=">.<..",
            config={"grid_length": 5},
            seed=42,
            t_max=10,
            t_fail=1,
        )
        ce2 = Counterexample(
            initial_state="..>.<",  # Same pattern rotated
            config={"grid_length": 5},
            seed=99,
            t_max=10,
            t_fail=1,
        )

        result1 = tracker.track(ce1)
        result2 = tracker.track(ce2)

        assert result1.is_novel
        assert not result2.is_novel
        assert result2.occurrence_count == 2

    def test_window_stats(self):
        """Should compute correct window statistics."""
        tracker = FailureKeyTracker(window_size=10)

        # Add some unique failures
        for i in range(5):
            ce = Counterexample(
                initial_state=">" + "." * i + "<",
                config={},
                seed=None,
                t_max=10,
                t_fail=1,
            )
            tracker.track(ce)

        stats = tracker.get_window_stats()

        assert stats.total_falsifications == 5
        # All unique
        assert stats.new_cex_rate == 1.0

    def test_repetition_detection(self):
        """Should detect repetition rate."""
        tracker = FailureKeyTracker(window_size=10)

        # Add one unique, then repeat it 4 times
        for i in range(5):
            ce = Counterexample(
                initial_state=">.<..",  # Same every time
                config={},
                seed=None,
                t_max=10,
                t_fail=1,
            )
            tracker.track(ce)

        stats = tracker.get_window_stats()

        assert stats.total_falsifications == 5
        assert stats.unique_failure_keys == 1
        assert stats.new_cex_rate == 0.2  # 1/5
        assert stats.repetition_rate == 0.8  # 4/5

    def test_top_failure_patterns(self):
        """Should return top failure patterns."""
        tracker = FailureKeyTracker()

        # Add pattern A 3 times
        for _ in range(3):
            tracker.track(Counterexample(">.<..", {}, None, 10, 1))

        # Add pattern B 2 times
        for _ in range(2):
            tracker.track(Counterexample(">..<.", {}, None, 10, 1))

        # Add pattern C 1 time
        tracker.track(Counterexample(">...<", {}, None, 10, 1))

        top = tracker.get_top_failure_patterns(limit=3)

        assert len(top) == 3
        # Most common first
        assert top[0][1] == 3
        assert top[1][1] == 2
        assert top[2][1] == 1

    def test_similar_keys_detection(self):
        """Should detect similar keys by initial state."""
        tracker = FailureKeyTracker()

        # Add two keys with same canonical initial but different timing
        ce1 = Counterexample(">.<..", {}, None, 10, 0)  # Immediate
        ce2 = Counterexample(">.<..", {}, None, 10, 10)  # Late

        result1 = tracker.track(ce1)
        result2 = tracker.track(ce2)

        # Second should see first as similar
        assert result1.failure_key.key_hash in result2.similar_keys or len(result2.similar_keys) > 0

    def test_reset(self):
        """Should reset all state."""
        tracker = FailureKeyTracker()

        tracker.track(Counterexample(">.<..", {}, None, 10, 1))
        assert tracker._total_falsifications == 1

        tracker.reset()

        assert tracker._total_falsifications == 0
        assert len(tracker._seen_keys) == 0

    def test_summary(self):
        """Should generate summary report."""
        tracker = FailureKeyTracker()

        for i in range(3):
            tracker.track(Counterexample(">" + "." * i + "<", {}, None, 10, 1))

        summary = tracker.get_summary()

        assert "total_falsifications" in summary
        assert "unique_failure_keys_total" in summary
        assert "window_stats" in summary
        assert "top_patterns" in summary


class TestFailureKeyStats:
    """Tests for failure key statistics."""

    def test_new_cex_rate(self):
        """Should calculate new_cex_rate correctly."""
        stats = FailureKeyStats(
            window_size=10,
            total_falsifications=10,
            unique_failure_keys=4,
        )

        assert stats.new_cex_rate == 0.4
        assert stats.repetition_rate == 0.6

    def test_empty_stats(self):
        """Should handle empty stats."""
        stats = FailureKeyStats(
            window_size=10,
            total_falsifications=0,
            unique_failure_keys=0,
        )

        assert stats.new_cex_rate == 1.0  # Default when no data
        assert stats.repetition_rate == 0.0

    def test_to_dict(self):
        """Should serialize correctly."""
        stats = FailureKeyStats(
            window_size=10,
            total_falsifications=5,
            unique_failure_keys=3,
        )

        d = stats.to_dict()

        assert d["window_size"] == 10
        assert d["total_falsifications"] == 5
        assert d["unique_failure_keys"] == 3
        assert "new_cex_rate" in d
        assert "repetition_rate" in d


class TestGroupFailuresByStructure:
    """Tests for grouping failures by structure."""

    def test_group_by_initial_state(self):
        """Should group by canonical initial state."""
        keys = [
            compute_failure_key(Counterexample(">.<..", {}, None, 10, 0)),
            compute_failure_key(Counterexample(">.<..", {}, None, 10, 1)),
            compute_failure_key(Counterexample("..>.<", {}, None, 10, 0)),  # Same as first
            compute_failure_key(Counterexample(">....<", {}, None, 10, 0)),  # Different
        ]

        groups = group_failures_by_structure(keys)

        # Should have groups for different patterns and timings
        assert len(groups) >= 2


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset tracker before each test."""
        reset_failure_key_tracker()

    def test_track_failure(self):
        """Module-level track should work."""
        ce = Counterexample(">.<..", {}, None, 10, 1)
        result = track_failure(ce)

        assert isinstance(result, FailureKeyResult)
        assert result.is_novel

    def test_get_failure_key_stats(self):
        """Module-level stats should work."""
        ce = Counterexample(">.<..", {}, None, 10, 1)
        track_failure(ce)

        stats = get_failure_key_stats()

        assert isinstance(stats, FailureKeyStats)
        assert stats.total_falsifications == 1

    def test_get_tracker(self):
        """Module-level tracker access should work."""
        tracker = get_failure_key_tracker()
        assert isinstance(tracker, FailureKeyTracker)


class TestFailureKeySerialization:
    """Tests for failure key serialization."""

    def test_failure_key_to_dict(self):
        """FailureKey should serialize correctly."""
        key = compute_failure_key(
            Counterexample(
                initial_state=">.<..",
                config={},
                seed=None,
                t_max=10,
                t_fail=1,
                observables_at_fail={"R": 1, "L": 1},
            )
        )

        d = key.to_dict()

        assert "canonical_initial" in d
        assert "t_fail_relative" in d
        assert "key_hash" in d
        assert "observable_signature" in d

    def test_failure_key_from_dict(self):
        """FailureKey should deserialize correctly."""
        original = compute_failure_key(
            Counterexample(">.<..", {}, None, 10, 1)
        )

        d = original.to_dict()
        restored = FailureKey.from_dict(d)

        assert restored.canonical_initial == original.canonical_initial
        assert restored.key_hash == original.key_hash
        assert restored.t_fail_relative == original.t_fail_relative

    def test_failure_key_result_to_dict(self):
        """FailureKeyResult should serialize correctly."""
        tracker = FailureKeyTracker()
        result = tracker.track(Counterexample(">.<..", {}, None, 10, 1))

        d = result.to_dict()

        assert "failure_key" in d
        assert "is_novel" in d
        assert "occurrence_count" in d


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_initial_state(self):
        """Should handle empty initial state."""
        ce = Counterexample(
            initial_state="",
            config={},
            seed=None,
            t_max=10,
            t_fail=0,
        )
        key = compute_failure_key(ce)

        assert key.canonical_initial == ""

    def test_single_char_state(self):
        """Should handle single character state."""
        ce = Counterexample(
            initial_state=">",
            config={},
            seed=None,
            t_max=10,
            t_fail=0,
        )
        key = compute_failure_key(ce)

        assert key.canonical_initial == ">"

    def test_no_trajectory(self):
        """Should handle missing trajectory."""
        ce = Counterexample(
            initial_state=">.<..",
            config={},
            seed=None,
            t_max=10,
            t_fail=1,
            trajectory_excerpt=None,
        )
        key = compute_failure_key(ce)

        assert key.trajectory_signature is None
        assert key.canonical_fail_state is None

    def test_no_observables(self):
        """Should handle missing observables."""
        ce = Counterexample(
            initial_state=">.<..",
            config={},
            seed=None,
            t_max=10,
            t_fail=1,
            observables_at_fail=None,
        )
        key = compute_failure_key(ce)

        assert key.observable_signature == ()
