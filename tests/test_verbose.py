"""Tests for the verbose logging module."""

import io
import tempfile
from pathlib import Path

from src.verbose import VerboseLogger


class TestVerboseLoggerLLMFile:
    """Test that LLM exchanges are written to the log file."""

    def test_llm_exchange_written_to_file(self, tmp_path):
        log_file = tmp_path / "verbose.log"
        vl = VerboseLogger(log_file=log_file)

        vl.log_llm_exchange(
            component="law_proposer",
            system_instruction="You are a scientist.",
            prompt="Propose laws about particles.",
            response='{"candidate_laws": []}',
            duration_ms=1234,
            success=True,
        )

        content = log_file.read_text()
        assert "law_proposer" in content
        assert "SYSTEM INSTRUCTION" in content
        assert "You are a scientist." in content
        assert "PROMPT" in content
        assert "Propose laws about particles." in content
        assert "RESPONSE" in content
        assert '{"candidate_laws": []}' in content
        assert "1234ms" in content

    def test_llm_exchange_appends(self, tmp_path):
        log_file = tmp_path / "verbose.log"
        vl = VerboseLogger(log_file=log_file)

        vl.log_llm_exchange(
            component="first",
            system_instruction=None,
            prompt="prompt1",
            response="response1",
        )
        vl.log_llm_exchange(
            component="second",
            system_instruction=None,
            prompt="prompt2",
            response="response2",
        )

        content = log_file.read_text()
        assert "first" in content
        assert "second" in content
        assert "prompt1" in content
        assert "prompt2" in content

    def test_no_file_means_no_write(self):
        vl = VerboseLogger(log_file=None)
        # Should not raise
        vl.log_llm_exchange(
            component="test",
            system_instruction=None,
            prompt="p",
            response="r",
        )

    def test_error_exchange(self, tmp_path):
        log_file = tmp_path / "verbose.log"
        vl = VerboseLogger(log_file=log_file)

        vl.log_llm_exchange(
            component="law_proposer",
            system_instruction="sys",
            prompt="prompt",
            response="",
            success=False,
            error_message="API timeout",
        )

        content = log_file.read_text()
        assert "Success: False" in content
        assert "API timeout" in content


class TestVerboseLoggerProbeConsole:
    """Test that probe events are written to the console stream."""

    def test_probe_registered_active(self):
        buf = io.StringIO()
        vl = VerboseLogger(console=buf)

        vl.log_probe_registered(
            probe_id="count_right",
            source="def probe(S):\n    return sum(1 for c in S if c == '>')",
            status="active",
        )

        output = buf.getvalue()
        assert "PROBE REGISTERED" in output
        assert "count_right" in output
        assert "active" in output
        assert "def probe(S):" in output

    def test_probe_registered_error(self):
        buf = io.StringIO()
        vl = VerboseLogger(console=buf)

        vl.log_probe_registered(
            probe_id="bad_probe",
            source="import os\ndef probe(S):\n    return 1",
            status="error",
            error_message="Validation failed: import not allowed",
        )

        output = buf.getvalue()
        assert "PROBE REGISTERED" in output
        assert "bad_probe" in output
        assert "status=error" in output
        assert "Validation failed" in output

    def test_probe_called_success(self):
        buf = io.StringIO()
        vl = VerboseLogger(console=buf)

        vl.log_probe_called(
            probe_id="count_right",
            state_repr="..><..",
            result=1,
        )

        output = buf.getvalue()
        assert "PROBE CALL" in output
        assert "count_right" in output
        assert "'..><..'" in output
        assert "1" in output

    def test_probe_called_error(self):
        buf = io.StringIO()
        vl = VerboseLogger(console=buf)

        vl.log_probe_called(
            probe_id="bad",
            state_repr="..",
            error="division by zero",
        )

        output = buf.getvalue()
        assert "ERROR" in output
        assert "division by zero" in output

    def test_probe_dedup(self):
        buf = io.StringIO()
        vl = VerboseLogger(console=buf)

        vl.log_probe_dedup("new_probe", "existing_probe")

        output = buf.getvalue()
        assert "PROBE DEDUP" in output
        assert "new_probe" in output
        assert "existing_probe" in output


class TestVerboseLoggerIntegration:
    """Test verbose logger wired into ProbeRegistry."""

    def test_registry_logs_registration(self):
        buf = io.StringIO()
        vl = VerboseLogger(console=buf)

        from src.probes.registry import ProbeRegistry
        reg = ProbeRegistry(verbose_logger=vl)
        reg.register("my_probe", "def probe(S):\n    return len(S)")

        output = buf.getvalue()
        assert "PROBE REGISTERED" in output
        assert "my_probe" in output

    def test_registry_logs_execution(self):
        buf = io.StringIO()
        vl = VerboseLogger(console=buf)

        from src.probes.registry import ProbeRegistry
        reg = ProbeRegistry(verbose_logger=vl)
        reg.register("grid_len", "def probe(S):\n    return len(S)")

        # Clear registration output
        buf.truncate(0)
        buf.seek(0)

        result = reg.execute("grid_len", list("..><.."))
        assert result == 6

        output = buf.getvalue()
        assert "PROBE CALL" in output
        assert "grid_len" in output
        assert "'..><..'" in output

    def test_registry_logs_execute_all_active(self):
        buf = io.StringIO()
        vl = VerboseLogger(console=buf)

        from src.probes.registry import ProbeRegistry
        reg = ProbeRegistry(verbose_logger=vl)
        reg.register("p1", "def probe(S):\n    return len(S)")
        reg.register("p2", "def probe(S):\n    return sum(1 for c in S if c == '>')")

        buf.truncate(0)
        buf.seek(0)

        results = reg.execute_all_active(list("..><.."))
        assert len(results) == 2

        output = buf.getvalue()
        assert output.count("PROBE CALL") == 2

    def test_registry_logs_dedup(self):
        buf = io.StringIO()
        vl = VerboseLogger(console=buf)

        from src.probes.registry import ProbeRegistry
        reg = ProbeRegistry(verbose_logger=vl)
        reg.register("probe_a", "def probe(S):\n    return len(S)")
        reg.register("probe_b", "def probe(S):\n    return len(S)")  # duplicate

        output = buf.getvalue()
        assert "PROBE DEDUP" in output
        assert "probe_b" in output
        assert "probe_a" in output

    def test_registry_logs_error_probe(self):
        buf = io.StringIO()
        vl = VerboseLogger(console=buf)

        from src.probes.registry import ProbeRegistry
        reg = ProbeRegistry(verbose_logger=vl)
        reg.register("bad", "import os\ndef probe(S):\n    return 1")

        output = buf.getvalue()
        assert "status=error" in output
        assert "Validation failed" in output
