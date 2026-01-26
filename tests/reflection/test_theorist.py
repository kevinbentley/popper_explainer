"""Tests for the Theorist task."""

import json
import pytest

from src.proposer.client import MockGeminiClient
from src.reflection.models import TheoristResult
from src.reflection.theorist import TheoristTask


def _make_theorist_response(
    derived_observables=None,
    hidden_variables=None,
    causal_narrative="",
    k_decomposition="",
    confidence=0.5,
    severe_test_suggestions=None,
):
    """Build a mock theorist JSON response."""
    return json.dumps({
        "derived_observables": derived_observables or [],
        "hidden_variables": hidden_variables or [],
        "causal_narrative": causal_narrative,
        "k_decomposition": k_decomposition,
        "confidence": confidence,
        "severe_test_suggestions": severe_test_suggestions or [],
    })


class TestTheoristTask:
    def test_basic_synthesis(self):
        """Theorist produces derived observables and narrative."""
        response = _make_theorist_response(
            derived_observables=[{
                "name": "net_momentum",
                "expression": "count(A) - count(B)",
                "rationale": "A and B appear to be complementary",
                "source_laws": ["L1", "L3"],
            }],
            causal_narrative="The universe appears to conserve total particle count.",
            confidence=0.7,
        )
        client = MockGeminiClient(responses=[response])
        theorist = TheoristTask(client=client, temperature=0.7)

        result = theorist.run(
            fixed_laws=[
                {"law_id": "L1", "claim": "count(A) + count(B) == N"},
                {"law_id": "L3", "claim": "grid_length == const"},
            ],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
        )

        assert isinstance(result, TheoristResult)
        assert len(result.derived_observables) == 1
        assert result.derived_observables[0].name == "net_momentum"
        assert result.confidence == 0.7
        assert "conserve" in result.causal_narrative

    def test_hidden_variable_postulation(self):
        """Theorist postulates hidden variables from anomalies."""
        response = _make_theorist_response(
            hidden_variables=[{
                "name": "boundary_interaction_count",
                "description": "Count of interactions at grid boundary",
                "evidence": "Many laws fail near grid edges",
                "testable_prediction": "Laws should hold when tested only on interior cells",
            }],
            confidence=0.4,
        )
        client = MockGeminiClient(responses=[response])
        theorist = TheoristTask(client=client)

        result = theorist.run(
            fixed_laws=[],
            graveyard=[{"law_id": "L2", "claim": "count fails at boundary"}],
            anomalies=[{"law_id": "L5", "reason_code": "low_power"}],
            research_log_entries=[],
        )

        assert len(result.hidden_variables) == 1
        hv = result.hidden_variables[0]
        assert hv.name == "boundary_interaction_count"
        assert "boundary" in hv.testable_prediction.lower() or "interior" in hv.testable_prediction.lower()

    def test_severe_test_suggestions(self):
        """Theorist suggests severe tests."""
        response = _make_theorist_response(
            severe_test_suggestions=[{
                "command_type": "initial_condition",
                "description": "Test with all-A grid to stress conservation",
                "target_law_id": "L1",
                "initial_conditions": ["AAAA", "AAAAAA"],
                "grid_lengths": [4, 8, 16],
            }],
        )
        client = MockGeminiClient(responses=[response])
        theorist = TheoristTask(client=client)

        result = theorist.run(
            fixed_laws=[{"law_id": "L1", "claim": "count preserved"}],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
        )

        assert len(result.severe_test_suggestions) == 1
        sug = result.severe_test_suggestions[0]
        assert sug["command_type"] == "initial_condition"
        assert sug["target_law_id"] == "L1"

    def test_with_existing_model(self):
        """Theorist receives and builds on existing standard model."""
        response = _make_theorist_response(
            causal_narrative="Building on prior model, we now see...",
            confidence=0.8,
        )
        client = MockGeminiClient(responses=[response])
        theorist = TheoristTask(client=client)

        result = theorist.run(
            fixed_laws=[],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
            current_model_summary={
                "version": 1,
                "causal_narrative_excerpt": "Universe conserves total count.",
                "derived_observables": [{"name": "net", "expression": "A-B"}],
                "hidden_variables": [],
                "confidence": 0.6,
            },
        )

        assert result.confidence == 0.8
        # Verify the model summary was included in the prompt
        assert len(client.calls) == 1
        prompt = client.calls[0]["prompt"]
        assert "CURRENT STANDARD MODEL" in prompt
        assert "Universe conserves total count" in prompt

    def test_llm_failure_graceful(self):
        """Theorist handles LLM failure gracefully."""
        client = MockGeminiClient(responses=["{{invalid json}}"])
        theorist = TheoristTask(client=client)

        result = theorist.run(
            fixed_laws=[],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
        )

        assert isinstance(result, TheoristResult)
        assert result.confidence == 0.0
        assert "Failed to parse" in result.causal_narrative

    def test_k_decomposition(self):
        """Theorist produces knowledge decomposition."""
        response = _make_theorist_response(
            k_decomposition="Established: total conservation. Conditional: parity effects. Open: boundary behavior.",
            confidence=0.6,
        )
        client = MockGeminiClient(responses=[response])
        theorist = TheoristTask(client=client)

        result = theorist.run(
            fixed_laws=[],
            graveyard=[],
            anomalies=[],
            research_log_entries=[],
        )

        assert "Established" in result.k_decomposition
        assert "Open" in result.k_decomposition
