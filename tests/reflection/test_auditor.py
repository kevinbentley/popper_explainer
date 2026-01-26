"""Tests for the Auditor task."""

import json
import pytest

from src.proposer.client import MockGeminiClient
from src.reflection.auditor import AuditorTask
from src.reflection.models import AuditorResult


def _make_auditor_response(
    conflicts=None, archives=None, deductive_issues=None, summary=""
):
    """Build a mock auditor JSON response."""
    return json.dumps({
        "conflicts": conflicts or [],
        "archives": archives or [],
        "deductive_issues": deductive_issues or [],
        "summary": summary,
    })


class TestAuditorTask:
    def test_no_conflicts_found(self):
        """Auditor returns empty result when no issues detected."""
        response = _make_auditor_response(summary="No issues found.")
        client = MockGeminiClient(responses=[response])
        auditor = AuditorTask(client=client, temperature=0.3)

        result = auditor.run(
            fixed_laws=[{"law_id": "L1", "claim": "count(A) == count(B)"}],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
        )

        assert isinstance(result, AuditorResult)
        assert len(result.conflicts) == 0
        assert len(result.archives) == 0
        assert result.summary == "No issues found."

    def test_conflict_detected_by_llm(self):
        """Auditor detects a conflict via LLM analysis."""
        response = _make_auditor_response(
            conflicts=[{
                "law_id": "L1",
                "conflicting_law_id": None,
                "counterexample_law_id": "L2",
                "description": "L1 claims total is conserved but L2 disproves it",
                "severity": "high",
            }],
            summary="One conflict found.",
        )
        client = MockGeminiClient(responses=[response])
        auditor = AuditorTask(client=client)

        result = auditor.run(
            fixed_laws=[{"law_id": "L1", "claim": "total is conserved"}],
            graveyard=[{"law_id": "L2", "claim": "total is conserved", "counterexample": {}}],
            anomalies=[],
            research_log_entries=[],
        )

        assert len(result.conflicts) == 1
        assert result.conflicts[0].law_id == "L1"
        assert result.conflicts[0].severity == "high"

    def test_python_conflict_detection(self):
        """Pure Python detects matching claims in fixed and graveyard."""
        # LLM finds nothing, but Python should detect the duplicate
        response = _make_auditor_response(summary="Looks clean.")
        client = MockGeminiClient(responses=[response])
        auditor = AuditorTask(client=client)

        result = auditor.run(
            fixed_laws=[{"law_id": "L1", "claim": "count(A) == count(B)"}],
            graveyard=[{"law_id": "L2", "claim": "count(A) == count(B)"}],
            anomalies=[],
            research_log_entries=[],
        )

        # Python should have found a conflict
        assert len(result.conflicts) >= 1
        conflict = next(c for c in result.conflicts if c.law_id == "L1")
        assert conflict.severity == "high"
        assert conflict.counterexample_law_id == "L2"

    def test_archive_recommendation(self):
        """Auditor recommends archiving redundant laws."""
        response = _make_auditor_response(
            archives=[{
                "law_id": "L2",
                "reason": "subsumed",
                "subsumed_by": "L1",
            }],
            summary="L2 is subsumed by L1.",
        )
        client = MockGeminiClient(responses=[response])
        auditor = AuditorTask(client=client)

        result = auditor.run(
            fixed_laws=[
                {"law_id": "L1", "claim": "count(A) + count(B) == N"},
                {"law_id": "L2", "claim": "count(A) + count(B) >= N"},
            ],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
        )

        assert len(result.archives) == 1
        assert result.archives[0].law_id == "L2"
        assert result.archives[0].reason == "subsumed"
        assert result.archives[0].subsumed_by == "L1"

    def test_llm_failure_graceful(self):
        """Auditor handles LLM call failure gracefully."""
        client = MockGeminiClient(responses=["not valid json at all"])
        auditor = AuditorTask(client=client)

        result = auditor.run(
            fixed_laws=[{"law_id": "L1", "claim": "test"}],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
        )

        assert isinstance(result, AuditorResult)
        assert "Failed to parse" in result.summary

    def test_merge_python_and_llm_conflicts(self):
        """Python and LLM conflicts are merged without duplicates."""
        response = _make_auditor_response(
            conflicts=[{
                "law_id": "L1",
                "conflicting_law_id": None,
                "counterexample_law_id": "L2",
                "description": "LLM also found this",
                "severity": "medium",
            }],
        )
        client = MockGeminiClient(responses=[response])
        auditor = AuditorTask(client=client)

        # Both Python and LLM should find a conflict for L1
        result = auditor.run(
            fixed_laws=[{"law_id": "L1", "claim": "test claim"}],
            graveyard=[{"law_id": "L2", "claim": "test claim"}],
            anomalies=[],
            research_log_entries=[],
        )

        # Should not duplicate L1 conflicts
        l1_conflicts = [c for c in result.conflicts if c.law_id == "L1"]
        assert len(l1_conflicts) == 1  # deduplicated

    def test_truncation_of_long_entries(self):
        """Research log entries are truncated when over budget."""
        response = _make_auditor_response(summary="OK")
        client = MockGeminiClient(responses=[response])
        auditor = AuditorTask(client=client, max_input_tokens=100)

        # Create very long entries
        long_entries = ["x" * 10000 for _ in range(10)]

        result = auditor.run(
            fixed_laws=[],
            graveyard=[],
            anomalies=[],
            research_log_entries=long_entries,
        )

        # Should still succeed despite truncation
        assert isinstance(result, AuditorResult)
        # Verify prompt was sent (client was called)
        assert len(client.calls) == 1
