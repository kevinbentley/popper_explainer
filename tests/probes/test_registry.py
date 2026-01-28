"""Tests for the probe registry: registration, execution, retirement, prompt summary."""

import pytest

from src.probes.registry import ProbeDefinition, ProbeRegistry
from src.probes.sandbox import ProbeError, ProbeRuntimeError


class TestProbeRegistration:

    def test_register_valid_probe(self):
        reg = ProbeRegistry()
        defn = reg.register(
            probe_id="count_right",
            source="def probe(S):\n    return sum(1 for c in S if c == '>')",
            hypothesis="Counts rightward-moving particles",
        )
        assert defn.status == "active"
        assert defn.probe_id == "count_right"
        assert defn.hypothesis == "Counts rightward-moving particles"
        assert defn.source_hash != ""

    def test_register_invalid_probe_gets_error_status(self):
        reg = ProbeRegistry()
        defn = reg.register(
            probe_id="bad_probe",
            source="import os\ndef probe(S):\n    return 1",
        )
        assert defn.status == "error"
        assert defn.error_message is not None
        assert "Validation failed" in defn.error_message

    def test_register_runtime_error_probe(self):
        reg = ProbeRegistry()
        defn = reg.register(
            probe_id="div_zero",
            source="def probe(S):\n    return 1 / 0",
        )
        assert defn.status == "error"
        assert "Test-run failed" in defn.error_message

    def test_register_non_numeric_probe(self):
        reg = ProbeRegistry()
        defn = reg.register(
            probe_id="string_return",
            source="def probe(S):\n    return 'hello'",
        )
        assert defn.status == "error"
        assert "Test-run failed" in defn.error_message

    def test_dedup_by_source_hash(self):
        reg = ProbeRegistry()
        defn1 = reg.register(
            probe_id="probe_a",
            source="def probe(S):\n    return len(S)",
        )
        # Same source, different ID -> returns existing
        defn2 = reg.register(
            probe_id="probe_b",
            source="def probe(S):\n    return len(S)",
        )
        assert defn2.probe_id == "probe_a"  # returns existing

    def test_dedup_ignores_errored_probes(self):
        reg = ProbeRegistry()
        # Register with error
        defn1 = reg.register(
            probe_id="bad",
            source="def probe(S):\n    return 1/0",
        )
        assert defn1.status == "error"

        # Same source, new ID - should NOT deduplicate against errored
        defn2 = reg.register(
            probe_id="bad2",
            source="def probe(S):\n    return 1/0",
        )
        assert defn2.probe_id == "bad2"


class TestProbeRetrieval:

    def test_get_existing(self):
        reg = ProbeRegistry()
        reg.register("my_probe", "def probe(S):\n    return len(S)")
        defn = reg.get("my_probe")
        assert defn is not None
        assert defn.probe_id == "my_probe"

    def test_get_nonexistent(self):
        reg = ProbeRegistry()
        assert reg.get("nonexistent") is None

    def test_list_active(self):
        reg = ProbeRegistry()
        reg.register("p1", "def probe(S):\n    return len(S)")
        reg.register("p2", "def probe(S):\n    return sum(1 for c in S if c == '>')")
        reg.register("p3", "def probe(S):\n    return 1/0")  # error

        active = reg.list_active()
        assert len(active) == 2
        ids = {p.probe_id for p in active}
        assert ids == {"p1", "p2"}

    def test_list_all(self):
        reg = ProbeRegistry()
        reg.register("p1", "def probe(S):\n    return len(S)")
        reg.register("p2", "def probe(S):\n    return 1/0")

        all_probes = reg.list_all()
        assert len(all_probes) == 2


class TestProbeExecution:

    def test_execute_single_probe(self):
        reg = ProbeRegistry()
        reg.register("grid_len", "def probe(S):\n    return len(S)")
        result = reg.execute("grid_len", list("..><.."))
        assert result == 6

    def test_execute_nonexistent_raises(self):
        reg = ProbeRegistry()
        with pytest.raises(KeyError):
            reg.execute("nonexistent", list(".."))

    def test_execute_errored_probe_raises(self):
        reg = ProbeRegistry()
        reg.register("bad", "def probe(S):\n    return 1/0")
        with pytest.raises(ProbeRuntimeError):
            reg.execute("bad", list(".."))

    def test_execute_all_active(self):
        reg = ProbeRegistry()
        reg.register("count_bg", "def probe(S):\n    return sum(1 for c in S if c == '.')")
        reg.register("count_right", "def probe(S):\n    return sum(1 for c in S if c == '>')")
        reg.register("bad", "def probe(S):\n    return 1/0")  # error, should be skipped

        state = list("..><..")
        results = reg.execute_all_active(state)
        assert "count_bg" in results
        assert "count_right" in results
        assert "bad" not in results
        assert results["count_bg"] == 4
        assert results["count_right"] == 1


class TestProbeRetirement:

    def test_retire_probe(self):
        reg = ProbeRegistry()
        reg.register("p1", "def probe(S):\n    return len(S)")
        assert reg.get("p1").status == "active"

        reg.retire("p1")
        assert reg.get("p1").status == "retired"

        # Retired probe should not appear in list_active
        assert len(reg.list_active()) == 0

    def test_retire_nonexistent_is_noop(self):
        reg = ProbeRegistry()
        reg.retire("nonexistent")  # should not raise


class TestPromptSummary:

    def test_no_probes_message(self):
        reg = ProbeRegistry()
        summary = reg.to_prompt_summary()
        assert "No probes defined" in summary

    def test_summary_includes_active_probes(self):
        reg = ProbeRegistry()
        reg.register(
            "count_right",
            "def probe(S):\n    return sum(1 for c in S if c == '>')",
            hypothesis="Counts rightward particles",
        )
        summary = reg.to_prompt_summary()
        assert "count_right" in summary
        assert "Counts rightward particles" in summary
        assert "def probe(S)" in summary

    def test_summary_excludes_errored_and_retired(self):
        reg = ProbeRegistry()
        reg.register("active", "def probe(S):\n    return len(S)")
        reg.register("errored", "def probe(S):\n    return 1/0")
        reg.register("to_retire", "def probe(S):\n    return 0")
        reg.retire("to_retire")

        summary = reg.to_prompt_summary()
        assert "active" in summary
        assert "errored" not in summary
        assert "to_retire" not in summary


# ---------------------------------------------------------------------------
# Temporal (2-param) probe tests
# ---------------------------------------------------------------------------

class TestTemporalProbeRegistration:

    def test_register_temporal_probe(self):
        reg = ProbeRegistry()
        defn = reg.register(
            probe_id="changed_cells",
            source="def probe(S_cur, S_nxt):\n    return sum(a != b for a, b in zip(S_cur, S_nxt))",
            hypothesis="Counts cells that changed between timesteps",
        )
        assert defn.status == "active"
        assert defn.arity == 2

    def test_register_single_param_has_arity_one(self):
        reg = ProbeRegistry()
        defn = reg.register("p1", "def probe(S):\n    return len(S)")
        assert defn.arity == 1

    def test_execute_temporal_probe(self):
        reg = ProbeRegistry()
        reg.register(
            "diff_count",
            "def probe(S_cur, S_nxt):\n    return sum(a != b for a, b in zip(S_cur, S_nxt))",
        )
        result = reg.execute("diff_count", list("..><.."), next_state=list("..X..."))
        assert result == 2

    def test_temporal_probe_dedup(self):
        reg = ProbeRegistry()
        source = "def probe(S_cur, S_nxt):\n    return 0"
        defn1 = reg.register("tp1", source)
        defn2 = reg.register("tp2", source)
        # Same source -> dedup returns existing
        assert defn2.probe_id == "tp1"
        assert defn2.arity == 2

    def test_prompt_summary_temporal(self):
        reg = ProbeRegistry()
        reg.register(
            "changes",
            "def probe(S_cur, S_nxt):\n    return sum(a != b for a, b in zip(S_cur, S_nxt))",
        )
        summary = reg.to_prompt_summary()
        assert "temporal" in summary.lower()
        assert "changes" in summary

    def test_execute_all_active_with_temporal(self):
        """Temporal probes should be included when next_state is provided."""
        reg = ProbeRegistry()
        reg.register("grid_len", "def probe(S):\n    return len(S)")
        reg.register(
            "changes",
            "def probe(S_cur, S_nxt):\n    return sum(a != b for a, b in zip(S_cur, S_nxt))",
        )
        results = reg.execute_all_active(list("..><.."), next_state=list("..X..."))
        assert "grid_len" in results
        assert "changes" in results
        assert results["grid_len"] == 6
        assert results["changes"] == 2

    def test_execute_all_active_skips_temporal_without_next_state(self):
        """Temporal probes should be skipped when no next_state provided."""
        reg = ProbeRegistry()
        reg.register("grid_len", "def probe(S):\n    return len(S)")
        reg.register(
            "changes",
            "def probe(S_cur, S_nxt):\n    return sum(a != b for a, b in zip(S_cur, S_nxt))",
        )
        results = reg.execute_all_active(list("..><.."))
        assert "grid_len" in results
        assert "changes" not in results  # skipped
