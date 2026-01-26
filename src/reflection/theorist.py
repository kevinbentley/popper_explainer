"""Theorist task for the Reflection Engine.

The theorist synthesizes a coherent causal narrative from empirical
evidence, derives new observables, postulates hidden variables, and
suggests severe tests. Runs as the second step of the reflection cycle.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from src.reflection.models import (
    DerivedObservable,
    HiddenVariable,
    TheoristResult,
)
from src.reflection.prompts import THEORIST_SYSTEM_INSTRUCTION, build_theorist_prompt

if TYPE_CHECKING:
    from src.proposer.client import GeminiClient, MockGeminiClient

logger = logging.getLogger(__name__)


class TheoristTask:
    """Synthesizes theory from empirical evidence.

    Usage:
        theorist = TheoristTask(client)
        result = theorist.run(fixed_laws, graveyard, anomalies,
                              research_log_entries, current_model_summary)
    """

    def __init__(
        self,
        client: GeminiClient | MockGeminiClient,
        temperature: float = 0.7,
        max_input_tokens: int = 100_000,
    ):
        self.client = client
        self.temperature = temperature
        self.max_input_tokens = max_input_tokens

    def run(
        self,
        fixed_laws: list[dict[str, Any]],
        graveyard: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        research_log_entries: list[str],
        current_model_summary: dict[str, Any] | None = None,
    ) -> TheoristResult:
        """Run the theorist analysis.

        Args:
            fixed_laws: Accepted (PASS) laws post-audit
            graveyard: Falsified (FAIL) laws with counterexamples
            anomalies: UNKNOWN laws
            research_log_entries: Research log from all prior iterations
            current_model_summary: Previous standard model summary

        Returns:
            TheoristResult with derived observables, hidden variables, etc.
        """
        prompt = build_theorist_prompt(
            fixed_laws=fixed_laws,
            graveyard=graveyard,
            anomalies=anomalies,
            research_log_entries=self._truncate_entries(research_log_entries),
            current_standard_model=current_model_summary,
        )

        prompt = self._truncate_prompt(prompt)

        try:
            response_text = self.client.generate(
                prompt,
                system_instruction=THEORIST_SYSTEM_INSTRUCTION,
                temperature=self.temperature,
            )
            return self._parse_response(response_text)
        except Exception as e:
            logger.error(f"Theorist LLM call failed: {e}")
            return TheoristResult(
                causal_narrative=f"Theorist LLM call failed: {e}",
                confidence=0.0,
            )

    def _parse_response(self, response_text: str) -> TheoristResult:
        """Parse the LLM response into a TheoristResult."""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                first_nl = text.find("\n")
                last_fence = text.rfind("```")
                if first_nl != -1 and last_fence > first_nl:
                    text = text[first_nl + 1:last_fence].strip()

            data = json.loads(text)

            derived_observables = [
                DerivedObservable(
                    name=d.get("name", ""),
                    expression=d.get("expression", ""),
                    rationale=d.get("rationale", ""),
                    source_laws=d.get("source_laws", []),
                )
                for d in data.get("derived_observables", [])
            ]

            hidden_variables = [
                HiddenVariable(
                    name=h.get("name", ""),
                    description=h.get("description", ""),
                    evidence=h.get("evidence", ""),
                    testable_prediction=h.get("testable_prediction", ""),
                )
                for h in data.get("hidden_variables", [])
            ]

            return TheoristResult(
                derived_observables=derived_observables,
                hidden_variables=hidden_variables,
                causal_narrative=data.get("causal_narrative", ""),
                k_decomposition=data.get("k_decomposition", ""),
                confidence=data.get("confidence", 0.5),
                severe_test_suggestions=data.get("severe_test_suggestions", []),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse theorist response: {e}")
            return TheoristResult(
                causal_narrative=f"Failed to parse theorist response: {e}",
                confidence=0.0,
            )

    def _truncate_entries(self, entries: list[str]) -> list[str]:
        """Truncate research log entries if total exceeds budget."""
        total_chars = sum(len(e) for e in entries)
        if total_chars / 4 <= self.max_input_tokens * 0.3:
            return entries

        result = []
        budget = int(self.max_input_tokens * 0.3 * 4)
        for entry in reversed(entries):
            if budget <= 0:
                break
            result.append(entry)
            budget -= len(entry)
        return list(reversed(result))

    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt if over token budget."""
        estimated_tokens = len(prompt) // 4
        if estimated_tokens <= self.max_input_tokens:
            return prompt

        target_chars = self.max_input_tokens * 4
        half = target_chars // 2
        return (
            prompt[:half]
            + "\n\n[... truncated for token budget ...]\n\n"
            + prompt[-half:]
        )
