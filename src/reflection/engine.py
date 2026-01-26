"""Reflection Engine — orchestrates auditor + theorist + severe test generation.

The engine runs as a periodic sub-routine within the Discovery phase,
invoked every N iterations by the DiscoveryPhaseHandler.

Two separate LLM calls:
1. Auditor (low temperature ~0.3): conflict detection, tautology/redundancy pruning
2. Theorist (higher temperature ~0.7): derived observables, hidden variables, narrative

The engine produces a ReflectionResult that is persisted to the database
and feeds back into the next discovery iteration.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from src.reflection.auditor import AuditorTask
from src.reflection.models import (
    AuditorResult,
    ReflectionResult,
    StandardModel,
    TheoristResult,
)
from src.reflection.persistence import build_standard_model_summary
from src.reflection.severe_test import SevereTestGenerator
from src.reflection.theorist import TheoristTask

if TYPE_CHECKING:
    from src.db.llm_logger import LLMLogger
    from src.proposer.client import GeminiClient, MockGeminiClient
    from src.proposer.scrambler import SymbolScrambler

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """Orchestrates the full reflection cycle.

    Usage:
        engine = ReflectionEngine(client=gemini_client)
        result = engine.run(
            fixed_laws=[...],
            graveyard=[...],
            anomalies=[...],
            research_log_entries=[...],
            current_standard_model=model_or_none,
        )
    """

    def __init__(
        self,
        client: GeminiClient | MockGeminiClient,
        auditor_temperature: float = 0.3,
        theorist_temperature: float = 0.7,
        max_input_tokens: int = 100_000,
        scrambler: SymbolScrambler | None = None,
        llm_logger: LLMLogger | None = None,
    ):
        """Initialize the reflection engine.

        Args:
            client: LLM client for both auditor and theorist calls
            auditor_temperature: Temperature for auditor (lower = more conservative)
            theorist_temperature: Temperature for theorist (higher = more creative)
            max_input_tokens: Maximum input tokens per LLM call
            scrambler: Optional symbol scrambler for abstract/physical translation
            llm_logger: Optional LLM logger for transcript logging. If provided,
                creates sub-loggers for auditor ("reflection_auditor") and theorist
                ("reflection_theorist") components and wraps the client to automatically
                log all LLM calls.
        """
        self.client = client
        self.scrambler = scrambler
        self._auditor_logger: LLMLogger | None = None
        self._theorist_logger: LLMLogger | None = None

        # If logger provided, create component-specific sub-loggers and wrap the client
        auditor_client = client
        theorist_client = client
        if llm_logger is not None:
            from src.db.llm_logger import LLMLogger as _LLMLogger

            self._auditor_logger = _LLMLogger(
                repo=llm_logger.repo,
                component="reflection_auditor",
                model_name=llm_logger.model_name,
            )
            self._theorist_logger = _LLMLogger(
                repo=llm_logger.repo,
                component="reflection_theorist",
                model_name=llm_logger.model_name,
            )
            auditor_client = self._auditor_logger.wrap_client(client)
            theorist_client = self._theorist_logger.wrap_client(client)

        self.auditor = AuditorTask(
            client=auditor_client,
            temperature=auditor_temperature,
            max_input_tokens=max_input_tokens,
        )
        self.theorist = TheoristTask(
            client=theorist_client,
            temperature=theorist_temperature,
            max_input_tokens=max_input_tokens,
        )
        self.severe_test_generator = SevereTestGenerator()

    def set_logger_context(
        self,
        run_id: str | None = None,
        iteration_id: int | None = None,
        phase: str | None = None,
    ) -> None:
        """Update the logging context for both sub-loggers.

        Call this before run() to tag LLM transcript records with the
        current run/iteration/phase.
        """
        if self._auditor_logger:
            self._auditor_logger.set_context(
                run_id=run_id, iteration_id=iteration_id, phase=phase,
            )
        if self._theorist_logger:
            self._theorist_logger.set_context(
                run_id=run_id, iteration_id=iteration_id, phase=phase,
            )

    def run(
        self,
        fixed_laws: list[dict[str, Any]],
        graveyard: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        research_log_entries: list[str],
        current_standard_model: StandardModel | None = None,
    ) -> ReflectionResult:
        """Run a full reflection cycle.

        Args:
            fixed_laws: Accepted (PASS) laws with details
            graveyard: Falsified (FAIL) laws with counterexamples
            anomalies: UNKNOWN laws with reason codes
            research_log_entries: All research log entries from prior iterations
            current_standard_model: Previous standard model (if exists)

        Returns:
            ReflectionResult with auditor output, theorist output,
            updated standard model, and severe test commands
        """
        start_time = time.time()

        # Scramble inputs if scrambler is available
        scrambled_fixed = self._scramble_laws(fixed_laws)
        scrambled_graveyard = self._scramble_laws(graveyard)
        scrambled_anomalies = self._scramble_laws(anomalies)
        scrambled_logs = self._scramble_log_entries(research_log_entries)

        # Build current model summary for theorist
        model_summary = None
        if current_standard_model:
            model_summary = build_standard_model_summary(current_standard_model)

        # Step 1: Run auditor
        logger.info("Reflection: running auditor...")
        auditor_result = self.auditor.run(
            fixed_laws=scrambled_fixed,
            graveyard=scrambled_graveyard,
            anomalies=scrambled_anomalies,
            research_log_entries=scrambled_logs,
        )
        logger.info(
            f"Auditor found {len(auditor_result.conflicts)} conflicts, "
            f"{len(auditor_result.archives)} archive candidates"
        )

        # Step 2: Apply audit results to get post-audit fixed laws
        archived_law_ids = {a.law_id for a in auditor_result.archives}
        post_audit_fixed = [
            law for law in scrambled_fixed
            if law.get("law_id") not in archived_law_ids
        ]

        # Step 3: Run theorist on post-audit data
        logger.info("Reflection: running theorist...")
        theorist_result = self.theorist.run(
            fixed_laws=post_audit_fixed,
            graveyard=scrambled_graveyard,
            anomalies=scrambled_anomalies,
            research_log_entries=scrambled_logs,
            current_model_summary=model_summary,
        )
        logger.info(
            f"Theorist produced {len(theorist_result.derived_observables)} derived observables, "
            f"{len(theorist_result.hidden_variables)} hidden variables"
        )

        # Step 4: Generate severe test commands
        severe_commands = self.severe_test_generator.generate(theorist_result)

        # Unscramble initial conditions in severe test commands if needed
        if self.scrambler:
            for cmd in severe_commands:
                if cmd.initial_conditions_json:
                    try:
                        ics = json.loads(cmd.initial_conditions_json)
                        ics = [self.scrambler.to_physical(ic) for ic in ics]
                        cmd.initial_conditions_json = json.dumps(ics)
                    except (json.JSONDecodeError, TypeError):
                        pass

        # Step 5: Build updated standard model
        standard_model = self._build_standard_model(
            fixed_laws=fixed_laws,  # Use original (unscrambled) law_ids
            auditor_result=auditor_result,
            theorist_result=theorist_result,
            previous_model=current_standard_model,
        )

        # Step 6: Build research log addendum
        addendum = self._build_research_log_addendum(
            auditor_result, theorist_result, severe_commands
        )

        runtime_ms = int((time.time() - start_time) * 1000)

        return ReflectionResult(
            auditor_result=auditor_result,
            theorist_result=theorist_result,
            standard_model=standard_model,
            severe_test_commands=severe_commands,
            research_log_addendum=addendum,
            runtime_ms=runtime_ms,
        )

    def _build_standard_model(
        self,
        fixed_laws: list[dict[str, Any]],
        auditor_result: AuditorResult,
        theorist_result: TheoristResult,
        previous_model: StandardModel | None,
    ) -> StandardModel:
        """Build an updated StandardModel from reflection outputs."""
        archived_law_ids = [a.law_id for a in auditor_result.archives]

        # Merge with previous archives
        if previous_model:
            existing_archives = set(previous_model.archived_laws)
            for lid in archived_law_ids:
                existing_archives.add(lid)
            archived_law_ids = list(existing_archives)

        # Fixed laws = all PASS laws not archived
        archived_set = set(archived_law_ids)
        active_fixed = [
            law.get("law_id", "") for law in fixed_laws
            if law.get("law_id") not in archived_set
        ]

        version = (previous_model.version + 1) if previous_model else 1

        return StandardModel(
            fixed_laws=active_fixed,
            archived_laws=archived_law_ids,
            derived_observables=theorist_result.derived_observables,
            causal_narrative=theorist_result.causal_narrative,
            hidden_variables=theorist_result.hidden_variables,
            k_decomposition=theorist_result.k_decomposition,
            confidence=theorist_result.confidence,
            version=version,
        )

    def _build_research_log_addendum(
        self,
        auditor_result: AuditorResult,
        theorist_result: TheoristResult,
        severe_commands: list,
    ) -> str:
        """Build a research log addendum summarizing the reflection cycle."""
        lines = []
        lines.append("\n--- REFLECTION ENGINE REPORT ---\n")

        if auditor_result.conflicts:
            lines.append(f"AUDITOR: Found {len(auditor_result.conflicts)} conflict(s):")
            for c in auditor_result.conflicts:
                lines.append(f"  - {c.law_id}: {c.description}")

        if auditor_result.archives:
            lines.append(f"AUDITOR: Recommended archiving {len(auditor_result.archives)} law(s):")
            for a in auditor_result.archives:
                lines.append(f"  - {a.law_id}: {a.reason}")

        if auditor_result.summary:
            lines.append(f"AUDITOR SUMMARY: {auditor_result.summary}")

        if theorist_result.derived_observables:
            lines.append(f"\nTHEORIST: Proposed {len(theorist_result.derived_observables)} derived observable(s):")
            for d in theorist_result.derived_observables:
                lines.append(f"  - {d.name}: {d.expression}")

        if theorist_result.hidden_variables:
            lines.append(f"THEORIST: Postulated {len(theorist_result.hidden_variables)} hidden variable(s):")
            for h in theorist_result.hidden_variables:
                lines.append(f"  - {h.name}: {h.testable_prediction}")

        if severe_commands:
            lines.append(f"\nSEVERE TESTS: {len(severe_commands)} priority research direction(s):")
            for cmd in severe_commands:
                lines.append(f"  - [{cmd.priority}] {cmd.description}")

        lines.append("\n--- END REFLECTION REPORT ---\n")
        return "\n".join(lines)

    def _scramble_laws(self, laws: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Scramble physical symbols to abstract in law dicts."""
        if not self.scrambler:
            return laws

        scrambled = []
        for law in laws:
            s = law.copy()
            if "claim" in s and isinstance(s["claim"], str):
                s["claim"] = self.scrambler.to_abstract(s["claim"])
            if "forbidden" in s and isinstance(s["forbidden"], str):
                s["forbidden"] = self.scrambler.to_abstract(s["forbidden"])
            if "observables" in s and isinstance(s["observables"], list):
                s["observables"] = [
                    {
                        **obs,
                        "expr": self.scrambler.to_abstract(obs.get("expr", ""))
                        if obs and "expr" in obs else obs.get("expr", ""),
                    }
                    for obs in s["observables"] if obs
                ]
            cx = s.get("counterexample")
            if cx and isinstance(cx, dict):
                cx = cx.copy()
                if "initial_state" in cx:
                    cx["initial_state"] = self.scrambler.to_abstract(cx["initial_state"])
                if "trajectory_excerpt" in cx:
                    exc = cx["trajectory_excerpt"]
                    if isinstance(exc, list):
                        cx["trajectory_excerpt"] = [
                            self.scrambler.to_abstract(t) for t in exc
                        ]
                    elif isinstance(exc, str):
                        cx["trajectory_excerpt"] = self.scrambler.to_abstract(exc)
                s["counterexample"] = cx
            scrambled.append(s)
        return scrambled

    def _scramble_log_entries(self, entries: list[str]) -> list[str]:
        """Scramble physical symbols in research log entries."""
        if not self.scrambler:
            return entries
        return [self.scrambler.to_abstract(e) for e in entries]
