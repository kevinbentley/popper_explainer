"""Auditor task for the Reflection Engine.

The auditor detects conflicts between fixed laws and the falsification
graveyard, identifies tautologies and redundancies, and flags deductive
issues. This runs as the first step of the reflection cycle.

Part of the conflict detection is done in pure Python (comparing law
claims against counterexample states) before the LLM call, so the LLM
can focus on tautology/redundancy identification.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from src.reflection.models import (
    ArchiveDecision,
    AuditorResult,
    ConflictEntry,
)
from src.reflection.prompts import AUDITOR_SYSTEM_INSTRUCTION, build_auditor_prompt

if TYPE_CHECKING:
    from src.proposer.client import GeminiClient, MockGeminiClient

logger = logging.getLogger(__name__)


class AuditorTask:
    """Audits fixed laws for conflicts, tautologies, and redundancy.

    Usage:
        auditor = AuditorTask(client)
        result = auditor.run(fixed_laws, graveyard, anomalies, research_log_entries)
    """

    def __init__(
        self,
        client: GeminiClient | MockGeminiClient,
        temperature: float = 0.3,
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
    ) -> AuditorResult:
        """Run the auditor analysis.

        Args:
            fixed_laws: Accepted (PASS) laws
            graveyard: Falsified (FAIL) laws with counterexamples
            anomalies: UNKNOWN laws
            research_log_entries: Research log from all prior iterations

        Returns:
            AuditorResult with conflicts, archives, and issues
        """
        # Phase 1: Pure Python conflict pre-check
        python_conflicts = self._detect_obvious_conflicts(fixed_laws, graveyard)

        # Phase 2: LLM analysis for tautology/redundancy
        prompt = build_auditor_prompt(
            fixed_laws=fixed_laws,
            graveyard=graveyard,
            anomalies=anomalies,
            research_log_entries=self._truncate_entries(research_log_entries),
        )

        # Truncate prompt if over token budget
        prompt = self._truncate_prompt(prompt)

        try:
            response_text = self.client.generate(
                prompt,
                system_instruction=AUDITOR_SYSTEM_INSTRUCTION,
                temperature=self.temperature,
            )
            llm_result = self._parse_response(response_text)
        except Exception as e:
            logger.error(f"Auditor LLM call failed: {e}")
            llm_result = AuditorResult(
                summary=f"Auditor LLM call failed: {e}",
            )

        # Merge python conflicts with LLM results
        return self._merge_results(python_conflicts, llm_result)

    def _detect_obvious_conflicts(
        self,
        fixed_laws: list[dict[str, Any]],
        graveyard: list[dict[str, Any]],
    ) -> list[ConflictEntry]:
        """Pure Python detection of obvious conflicts.

        Checks if any fixed law has the same template and claim text
        as a graveyard law (meaning the exact same law was both passed
        and failed, indicating a test inconsistency or flaky harness).
        """
        conflicts = []

        # Build lookup of graveyard law claims
        graveyard_claims: dict[str, str] = {}  # normalized_claim -> law_id
        for law in graveyard:
            claim = law.get("claim", "").strip().lower()
            if claim:
                graveyard_claims[claim] = law.get("law_id", "")

        # Check fixed laws against graveyard
        for law in fixed_laws:
            claim = law.get("claim", "").strip().lower()
            if claim in graveyard_claims:
                conflicts.append(ConflictEntry(
                    law_id=law.get("law_id", ""),
                    counterexample_law_id=graveyard_claims[claim],
                    description=(
                        f"Law '{law.get('law_id', '')}' is accepted (PASS) but an "
                        f"identical or near-identical claim was falsified in the graveyard "
                        f"(law '{graveyard_claims[claim]}'). This may indicate test "
                        f"inconsistency or a non-deterministic harness."
                    ),
                    severity="high",
                ))

        return conflicts

    def _parse_response(self, response_text: str) -> AuditorResult:
        """Parse the LLM response into an AuditorResult."""
        try:
            # Strip markdown code fences if present
            text = response_text.strip()
            if text.startswith("```"):
                first_nl = text.find("\n")
                last_fence = text.rfind("```")
                if first_nl != -1 and last_fence > first_nl:
                    text = text[first_nl + 1:last_fence].strip()

            data = json.loads(text)
            return AuditorResult.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse auditor response: {e}")
            return AuditorResult(
                summary=f"Failed to parse auditor response: {e}",
            )

    def _merge_results(
        self,
        python_conflicts: list[ConflictEntry],
        llm_result: AuditorResult,
    ) -> AuditorResult:
        """Merge Python-detected conflicts with LLM analysis."""
        # Deduplicate by law_id
        seen_law_ids = {c.law_id for c in llm_result.conflicts}
        for conflict in python_conflicts:
            if conflict.law_id not in seen_law_ids:
                llm_result.conflicts.append(conflict)
                seen_law_ids.add(conflict.law_id)

        return llm_result

    def _truncate_entries(self, entries: list[str]) -> list[str]:
        """Truncate research log entries if total exceeds budget."""
        total_chars = sum(len(e) for e in entries)
        # Rough estimate: 4 chars per token
        if total_chars / 4 <= self.max_input_tokens * 0.3:
            return entries

        # Keep most recent entries, drop oldest
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

        # Truncate from the middle (keep start and end)
        target_chars = self.max_input_tokens * 4
        half = target_chars // 2
        return (
            prompt[:half]
            + "\n\n[... truncated for token budget ...]\n\n"
            + prompt[-half:]
        )
