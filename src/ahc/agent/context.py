"""Context builder for dynamic prompt construction.

Implements the three-tier memory architecture from HC_CONTEXT_MGR.md:
- Tier 1: Permanent core (system prompt, tools, grammar)
- Tier 2: Evolving theory (theorems, negative knowledge)
- Tier 3: Rolling journal (last N turns)
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.ahc.db.repo import AHCRepository
from src.ahc.db.models import TheoremStatus

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context building."""
    rolling_window_size: int = 15  # Number of recent turns to include
    include_tier2: bool = True  # Whether to inject knowledge state
    max_theorems_in_context: int = 20  # Max theorems to inject
    max_counterexamples_in_gallery: int = 10  # Max counterexamples in gallery


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string.

    Uses a simple heuristic of ~4 characters per token.
    This is approximate but sufficient for context management.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4


class ContextBuilder:
    """Builds dynamic prompts from the three-tier memory.

    This class constructs the conversation context for each LLM call,
    combining:
    - Tier 1: Static system prompt
    - Tier 2: Dynamic knowledge state (theorems, negative knowledge)
    - Tier 3: Rolling window of recent turns

    The goal is to keep context size manageable while preserving
    all knowledge the agent has accumulated.
    """

    def __init__(
        self,
        repo: AHCRepository,
        session_id: int,
        config: ContextConfig | None = None,
    ):
        """Initialize the context builder.

        Args:
            repo: Database repository
            session_id: Current session ID
            config: Optional configuration
        """
        self.repo = repo
        self.session_id = session_id
        self.config = config or ContextConfig()
        self._system_prompt: str | None = None

    def set_system_prompt(self, prompt: str) -> None:
        """Set the Tier 1 system prompt.

        Args:
            prompt: The static system prompt
        """
        self._system_prompt = prompt

    def build_messages(self) -> list[dict[str, Any]]:
        """Build the full message list for an LLM call.

        Returns:
            List of messages with dynamic context injection
        """
        messages = []

        # Tier 1: System prompt
        if self._system_prompt:
            messages.append({
                "role": "system",
                "content": self._system_prompt,
            })

        # Tier 2: Knowledge state injection
        if self.config.include_tier2:
            knowledge_msg = self._build_knowledge_injection()
            if knowledge_msg:
                messages.append(knowledge_msg)

        # Tier 3: Rolling window of recent turns
        recent_turns = self._get_rolling_window()
        for turn in recent_turns:
            messages.append({
                "role": turn.role,
                "content": turn.content,
            })

        return messages

    def _build_knowledge_injection(self) -> dict[str, Any] | None:
        """Build the Tier 2 knowledge state injection.

        Returns:
            User message containing current knowledge state, or None
        """
        sections = []

        # Always include tool syntax reminder (common source of errors)
        sections.append("## TOOL SYNTAX REMINDER\n")
        sections.append("**IMPORTANT**: The symbol `_` means EMPTY CELL (not a blank). Double-check your trigger_symbol matches your intent!")
        sections.append("")
        sections.append("For `local_transition`: use trigger_symbol, result_op, result_symbol (+ optional left_neighbor/right_neighbor)")
        sections.append("For `invariant`/`bound`/`monotone`/`implication_step`/`implication_state`: use observables + claim_ast")
        sections.append("For `symmetry_commutation`: use transform")
        sections.append("")
        sections.append("**Reading counterexamples**: When a law FAILs, `states` shows ~5 consecutive states centered on t_fail. Use `states_t_offset` to find t_fail in the list (it's at index `-states_t_offset`). For local_transition: `fail_position` is the cell index, `actual_result` is what it became, `neighborhood_at_t` shows the 3-cell pattern {left, center, right} at the failure, and `neighborhood_at_t_minus_1` shows the pattern one step earlier — compare these to understand what caused the transition. For non-local templates: `observables_at_t` and `observables_at_t1` show the observable values at the failure time.")
        sections.append("")

        # Get validated theorems
        theorems = self.repo.get_theorems(
            self.session_id,
            status=TheoremStatus.VALIDATED,
        )

        if theorems:
            sections.append("## YOUR CURRENT KNOWLEDGE STATE\n")
            sections.append("### Established Theorems (Previously Validated)\n")
            for t in theorems[:self.config.max_theorems_in_context]:
                sections.append(f"- **{t.name}**: {t.description or 'No description'}")
            sections.append("")

        # Get negative knowledge from last compaction
        meta = self.repo.get_latest_meta_knowledge(self.session_id)
        if meta and meta.negative_knowledge:
            sections.append("### What You've Ruled Out (Falsified Concepts)\n")
            sections.append(meta.negative_knowledge)
            sections.append("")

        # Counterexample gallery
        gallery = self._build_counterexample_gallery()
        if gallery:
            sections.append(gallery)

        # Template distribution warning
        template_dist = self._build_template_distribution()
        if template_dist:
            sections.append(template_dist)

        # Methodology and strategy reminder (reinforces system prompt; critical after
        # compaction when the data above can overwhelm behavioral instructions)
        sections.append("---")
        sections.append("REMEMBER: Write RESULT + LAB NOTE analysis BEFORE every tool call. No exceptions.")
        sections.append("")
        sections.append("STRATEGY: You have a limited tool call budget. Prefer BOLD macro-level laws "
                        "(invariant, symmetry_commutation, implication_step, bound) over exhaustive "
                        "local_transition enumeration. One invariant test covers every cell at every "
                        "timestep — far more informative than a single neighborhood rule.")
        sections.append("")

        return {
            "role": "user",
            "content": "\n".join(sections),
        }

    def _build_counterexample_gallery(self) -> str | None:
        """Build the counterexample gallery from recent FAIL evaluations.

        Returns:
            Formatted gallery string, or None if no failures exist
        """
        failed_evals = self.repo.get_recent_failed_evaluations(
            self.session_id,
            limit=self.config.max_counterexamples_in_gallery,
        )

        if not failed_evals:
            return None

        from src.proposer.scrambler import get_default_scrambler
        scrambler = get_default_scrambler()

        lines = [
            "### COUNTEREXAMPLE GALLERY (Your Most Valuable Data)\n",
            "Each counterexample is a FACT. Study the states to understand WHY the law failed.\n",
        ]

        for i, ev in enumerate(failed_evals, 1):
            # Parse the stored counterexample JSON
            if not ev.counterexample_json:
                continue

            try:
                cx = json.loads(ev.counterexample_json)
            except (json.JSONDecodeError, TypeError):
                continue

            # Parse the law JSON to get the tested echo
            tested_echo = ""
            if ev.law_json:
                try:
                    law = json.loads(ev.law_json)
                    template = law.get("template") or "?"
                    trigger = law.get("trigger_symbol") or None
                    result_sym = law.get("result_symbol") or None
                    result_op = law.get("result_op") or "=="
                    neighbor_pattern = law.get("neighbor_pattern") or None

                    if trigger is not None:
                        # local_transition style echo
                        trigger_abs = scrambler.to_abstract(trigger)
                        result_abs = scrambler.to_abstract(result_sym) if result_sym else "?"

                        if neighbor_pattern:
                            pattern_abs = scrambler.to_abstract(neighbor_pattern)
                            tested_echo = f"trigger={trigger_abs} with left={pattern_abs[0]} right={pattern_abs[2]} -> {result_op} {result_abs}"
                        else:
                            tested_echo = f"trigger={trigger_abs} (any neighbors) -> {result_op} {result_abs}"
                    else:
                        # Non-local-transition: show template + claim summary
                        claim = law.get("claim") or ""
                        claim_short = (claim[:60] + "...") if len(claim) > 60 else claim
                        tested_echo = f"[{template}] {claim_short}"
                except (json.JSONDecodeError, TypeError, IndexError):
                    tested_echo = "(unknown law)"

            # Format the gallery entry
            t_fail = cx.get("t_fail", "?")
            fail_pos = cx.get("fail_position", "?")
            actual = cx.get("actual_result") or None

            lines.append(f"{i}. TESTED: {tested_echo}")
            lines.append(f"   FAILED at t={t_fail}, position {fail_pos}")

            # Show states trajectory (scrambled to abstract)
            states = cx.get("states")
            if states and isinstance(states, list):
                scrambled_states = [scrambler.to_abstract(s) for s in states if s is not None]
                if scrambled_states:
                    lines.append(f"   States: {' -> '.join(scrambled_states)}")

            # Show actual result
            if actual is not None:
                actual_abs = scrambler.to_abstract(actual)
                lines.append(f"   Cell {fail_pos} became {actual_abs}")

            lines.append("")

        return "\n".join(lines)

    def _build_template_distribution(self) -> str | None:
        """Build a template distribution summary from recent evaluations.

        If the agent is overwhelmingly using a single template, adds a warning.

        Returns:
            Formatted distribution summary with warning, or None
        """
        evaluations = self.repo.get_law_evaluations(self.session_id, limit=50)
        if len(evaluations) < 10:
            return None

        template_counts: dict[str, int] = {}
        for ev in evaluations:
            if not ev.law_json:
                continue
            try:
                law = json.loads(ev.law_json)
                template = law.get("template", "unknown")
                template_counts[template] = template_counts.get(template, 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue

        total = sum(template_counts.values())
        if total == 0:
            return None

        local_count = template_counts.get("local_transition", 0)
        local_pct = (local_count / total) * 100

        lines = ["### Template Distribution (Last 50 Evaluations)\n"]
        for template, count in sorted(template_counts.items(), key=lambda x: -x[1]):
            pct = (count / total) * 100
            lines.append(f"- {template}: {count} ({pct:.0f}%)")

        if local_pct > 80:
            lines.append("")
            lines.append(
                "WARNING: Over 80% of your tests are local_transition. "
                "You are missing macro-level structure. Prioritize: "
                "invariant, bound, symmetry_commutation, implication_step."
            )

        lines.append("")
        return "\n".join(lines)

    def _get_rolling_window(self) -> list:
        """Get the Tier 3 rolling window of recent turns.

        Returns:
            List of recent ConversationTurnRecords
        """
        return self.repo.get_recent_turns(
            self.session_id,
            limit=self.config.rolling_window_size,
        )

    def estimate_context_tokens(self) -> int:
        """Estimate total tokens in the current context.

        Returns:
            Estimated token count
        """
        messages = self.build_messages()
        total = 0
        for msg in messages:
            total += estimate_tokens(msg.get("content", ""))
        return total

    def get_context_stats(self) -> dict[str, Any]:
        """Get statistics about the current context.

        Returns:
            Dict with context statistics
        """
        messages = self.build_messages()

        stats = {
            "message_count": len(messages),
            "estimated_tokens": 0,
            "tier1_tokens": 0,
            "tier2_tokens": 0,
            "tier3_tokens": 0,
        }

        for msg in messages:
            tokens = estimate_tokens(msg.get("content", ""))
            stats["estimated_tokens"] += tokens

            role = msg.get("role", "")
            if role == "system":
                stats["tier1_tokens"] += tokens
            elif "CURRENT KNOWLEDGE STATE" in msg.get("content", ""):
                stats["tier2_tokens"] += tokens
            else:
                stats["tier3_tokens"] += tokens

        return stats
