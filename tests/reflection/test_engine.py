"""Tests for the Reflection Engine (full integration)."""

import json
import os
import tempfile
import pytest

from src.proposer.client import MockGeminiClient
from src.reflection.engine import ReflectionEngine
from src.reflection.models import ReflectionResult, StandardModel


def _make_auditor_response():
    return json.dumps({
        "conflicts": [],
        "archives": [{
            "law_id": "L_old",
            "reason": "redundant",
            "subsumed_by": "L1",
        }],
        "deductive_issues": [],
        "summary": "L_old is redundant with L1.",
    })


def _make_theorist_response():
    return json.dumps({
        "derived_observables": [{
            "name": "net_flow",
            "expression": "count(A) - count(B)",
            "rationale": "A and B move in opposite directions",
            "source_laws": ["L1"],
        }],
        "hidden_variables": [{
            "name": "collision_rate",
            "description": "Rate of A-B collisions",
            "evidence": "Many laws about A+B fail near K symbols",
            "testable_prediction": "Filter states with K present vs absent",
        }],
        "causal_narrative": "A and B are complementary movers. K represents collisions.",
        "k_decomposition": "Established: A+B conservation. Open: K mechanics.",
        "confidence": 0.65,
        "severe_test_suggestions": [{
            "command_type": "initial_condition",
            "description": "Test with no K symbols to isolate A-B dynamics",
            "target_law_id": "L1",
            "initial_conditions": ["AABB", "ABAB"],
            "grid_lengths": [4, 8],
        }],
    })


class TestReflectionEngine:
    def test_full_cycle(self):
        """Full reflection cycle produces expected outputs."""
        client = MockGeminiClient(responses=[
            _make_auditor_response(),
            _make_theorist_response(),
        ])
        engine = ReflectionEngine(client=client)

        result = engine.run(
            fixed_laws=[
                {"law_id": "L1", "claim": "count(A) + count(B) == N"},
                {"law_id": "L_old", "claim": "count(A) >= 0"},
            ],
            graveyard=[
                {"law_id": "L2", "claim": "count(A) always increases",
                 "counterexample": {"initial_state": "AB", "t_fail": 1}},
            ],
            anomalies=[],
            research_log_entries=["Iteration 1: tested conservation"],
        )

        assert isinstance(result, ReflectionResult)

        # Auditor results
        assert len(result.auditor_result.archives) == 1
        assert result.auditor_result.archives[0].law_id == "L_old"

        # Theorist results
        assert len(result.theorist_result.derived_observables) == 1
        assert result.theorist_result.derived_observables[0].name == "net_flow"
        assert len(result.theorist_result.hidden_variables) == 1
        assert result.theorist_result.confidence == 0.65

        # Standard model
        assert "L_old" in result.standard_model.archived_laws
        assert "L1" in result.standard_model.fixed_laws
        assert result.standard_model.version == 1

        # Severe test commands
        assert len(result.severe_test_commands) >= 1

        # Research log addendum
        assert "REFLECTION ENGINE REPORT" in result.research_log_addendum
        assert result.runtime_ms >= 0

    def test_with_existing_standard_model(self):
        """Engine builds on existing standard model."""
        client = MockGeminiClient(responses=[
            _make_auditor_response(),
            _make_theorist_response(),
        ])
        engine = ReflectionEngine(client=client)

        previous_model = StandardModel(
            fixed_laws=["L1"],
            archived_laws=["L_ancient"],
            version=3,
        )

        result = engine.run(
            fixed_laws=[
                {"law_id": "L1", "claim": "test"},
                {"law_id": "L_old", "claim": "old"},
            ],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
            current_standard_model=previous_model,
        )

        # Version should increment
        assert result.standard_model.version == 4

        # Archives should merge with previous
        assert "L_ancient" in result.standard_model.archived_laws
        assert "L_old" in result.standard_model.archived_laws

    def test_two_llm_calls_made(self):
        """Engine makes exactly 2 LLM calls (auditor + theorist)."""
        client = MockGeminiClient(responses=[
            _make_auditor_response(),
            _make_theorist_response(),
        ])
        engine = ReflectionEngine(client=client)

        engine.run(
            fixed_laws=[{"law_id": "L1", "claim": "test"}],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
        )

        assert len(client.calls) == 2
        # First call should be auditor (lower temperature)
        assert client.calls[0]["temperature"] == 0.3
        # Second call should be theorist (higher temperature)
        assert client.calls[1]["temperature"] == 0.7

    def test_auditor_failure_continues_to_theorist(self):
        """If auditor produces bad JSON, engine still runs theorist."""
        client = MockGeminiClient(responses=[
            "not valid json",
            _make_theorist_response(),
        ])
        engine = ReflectionEngine(client=client)

        result = engine.run(
            fixed_laws=[{"law_id": "L1", "claim": "test"}],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
        )

        # Should still have theorist results
        assert len(result.theorist_result.derived_observables) == 1
        # Auditor should have graceful failure
        assert isinstance(result.auditor_result.summary, str)

    def test_research_log_addendum_content(self):
        """Addendum contains auditor and theorist summaries."""
        client = MockGeminiClient(responses=[
            _make_auditor_response(),
            _make_theorist_response(),
        ])
        engine = ReflectionEngine(client=client)

        result = engine.run(
            fixed_laws=[{"law_id": "L1", "claim": "test"}],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
        )

        addendum = result.research_log_addendum
        assert "AUDITOR" in addendum
        assert "THEORIST" in addendum
        assert "SEVERE TESTS" in addendum

    def test_hidden_variable_generates_severe_test(self):
        """Hidden variables from theorist produce severe test commands."""
        client = MockGeminiClient(responses=[
            json.dumps({"conflicts": [], "archives": [], "deductive_issues": [], "summary": ""}),
            json.dumps({
                "derived_observables": [],
                "hidden_variables": [{
                    "name": "phase_boundary",
                    "description": "Phase transitions at grid boundaries",
                    "evidence": "Laws fail near edges",
                    "testable_prediction": "Test with periodic vs reflective boundary",
                }],
                "causal_narrative": "",
                "k_decomposition": "",
                "confidence": 0.3,
                "severe_test_suggestions": [],
            }),
        ])
        engine = ReflectionEngine(client=client)

        result = engine.run(
            fixed_laws=[],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
        )

        # Hidden variable should generate a severe test command
        assert len(result.severe_test_commands) >= 1
        assert "phase_boundary" in result.severe_test_commands[0].description

    def test_scrambler_applied(self):
        """Symbol scrambler is applied to inputs when provided."""
        from src.proposer.scrambler import SymbolScrambler

        client = MockGeminiClient(responses=[
            json.dumps({"conflicts": [], "archives": [], "deductive_issues": [], "summary": ""}),
            json.dumps({
                "derived_observables": [], "hidden_variables": [],
                "causal_narrative": "", "k_decomposition": "",
                "confidence": 0.5, "severe_test_suggestions": [],
            }),
        ])
        scrambler = SymbolScrambler()
        engine = ReflectionEngine(client=client, scrambler=scrambler)

        engine.run(
            fixed_laws=[{"law_id": "L1", "claim": "count('>') == count('<')"}],
            graveyard=[{
                "law_id": "L2",
                "claim": "count('.') > 0",
                "counterexample": {"initial_state": "..><..", "t_fail": 3},
            }],
            anomalies=[],
            research_log_entries=["Found that > and < interact"],
        )

        # Prompts should use abstract symbols
        auditor_prompt = client.calls[0]["prompt"]
        assert ">" not in auditor_prompt or "count('>')" not in auditor_prompt
        # Physical symbols should be replaced
        assert "A" in auditor_prompt or "B" in auditor_prompt

    def test_llm_calls_logged_to_transcripts(self):
        """LLM calls are logged to the database when llm_logger is provided."""
        from src.db.llm_logger import LLMLogger
        from src.db.repo import Repository

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            repo = Repository(db_path=path)
            repo.connect()

            llm_logger = LLMLogger(
                repo=repo,
                component="reflection",  # base component (overridden per sub-task)
                model_name="mock-model",
            )

            client = MockGeminiClient(responses=[
                _make_auditor_response(),
                _make_theorist_response(),
            ])
            engine = ReflectionEngine(client=client, llm_logger=llm_logger)

            # Set context like the discovery handler would
            engine.set_logger_context(
                run_id="test-run", iteration_id=5, phase="discovery",
            )

            engine.run(
                fixed_laws=[{"law_id": "L1", "claim": "test"}],
                graveyard=[],
                anomalies=[],
                research_log_entries=[],
            )

            # Query all transcripts (list returns newest first, so reverse)
            transcripts = repo.list_llm_transcripts()
            assert len(transcripts) == 2

            # Sort by id to get chronological order
            transcripts.sort(key=lambda t: t.id or 0)

            # First call: auditor
            assert transcripts[0].component == "reflection_auditor"
            assert transcripts[0].model_name == "mock-model"
            assert transcripts[0].run_id == "test-run"
            assert transcripts[0].iteration_id == 5
            assert transcripts[0].phase == "discovery"
            assert transcripts[0].success is True

            # Second call: theorist
            assert transcripts[1].component == "reflection_theorist"
            assert transcripts[1].model_name == "mock-model"
            assert transcripts[1].run_id == "test-run"
            assert transcripts[1].iteration_id == 5
            assert transcripts[1].phase == "discovery"
            assert transcripts[1].success is True

            repo.close()
        finally:
            os.unlink(path)

    def test_llm_logging_captures_prompt_and_response(self):
        """Logged transcripts contain the actual prompt and response text."""
        from src.db.llm_logger import LLMLogger
        from src.db.repo import Repository

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            repo = Repository(db_path=path)
            repo.connect()

            llm_logger = LLMLogger(
                repo=repo, component="reflection", model_name="mock-model",
            )

            client = MockGeminiClient(responses=[
                _make_auditor_response(),
                _make_theorist_response(),
            ])
            engine = ReflectionEngine(client=client, llm_logger=llm_logger)

            engine.run(
                fixed_laws=[{"law_id": "L1", "claim": "count(A) == 5"}],
                graveyard=[],
                anomalies=[],
                research_log_entries=[],
            )

            transcripts = repo.list_llm_transcripts()
            assert len(transcripts) == 2

            # Sort by id to get chronological order
            transcripts.sort(key=lambda t: t.id or 0)

            # Auditor prompt should contain the law claim
            assert "count(A) == 5" in transcripts[0].prompt
            # Auditor response should be the mock response
            assert "L_old is redundant" in transcripts[0].raw_response

            # Theorist response should contain the mock narrative
            assert "complementary movers" in transcripts[1].raw_response

            repo.close()
        finally:
            os.unlink(path)
