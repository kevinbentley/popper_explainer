"""Context compaction manager.

Handles periodic summarization of conversation history to prevent
context overflow while preserving learned knowledge.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from src.ahc.db.repo import AHCRepository
from src.ahc.db.models import MetaKnowledgeRecord, TheoremStatus
from src.ahc.agent.context import estimate_tokens

logger = logging.getLogger(__name__)


@dataclass
class CompactionConfig:
    """Configuration for compaction manager."""
    token_threshold: int = 100000  # Trigger compaction at this token count
    turn_threshold: int = 50  # Or after this many turns since last compaction
    min_turns_between: int = 10  # Minimum turns between compactions


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    success: bool
    version: int
    turns_compacted: int
    tokens_before: int
    tokens_after: int
    new_negative_knowledge: str | None = None
    error: str | None = None


COMPACTION_PROMPT = """You are analyzing a sequence of experimental turns from a Popperian discovery session.
Your task is to extract and summarize key learnings.

## Instructions

1. **Identify Falsified Concepts**: Look for patterns that were consistently FAILED or ruled out.
   These become "negative knowledge" - things the agent now knows are NOT true.

2. **Identify Candidate Theorems**: Look for laws that consistently PASSED with high power.
   These are candidates for promotion to validated theorems.

3. **Summarize Concisely**: The negative knowledge should be a brief bullet list
   of ruled-out concepts, not a full history.

4. **Assess Research Strategy**: Check whether the agent is stuck in a narrow pattern,
   such as exhaustively enumerating neighborhood combinations using only local_transition.
   If so, include a bullet in negative_knowledge noting this and recommending bolder
   macro-level templates (invariant, symmetry_commutation, implication_step, bound).
   Popperian science favors bold, risky conjectures — not brute-force enumeration.

## Output Format

Return a JSON object:
```json
{
  "negative_knowledge": "- Concept X is false because...\n- Y does not hold when...\n- STRATEGY: ...",
  "theorem_candidates": [
    {"name": "suggested name", "description": "what it claims", "evidence": "why it passed"}
  ]
}
```

## Turns to Analyze

"""


class CompactionManager:
    """Manages periodic context compaction.

    When the conversation history grows too large, this manager:
    1. Calls the LLM to summarize key learnings
    2. Stores a compaction snapshot in meta_knowledge
    3. Allows the context builder to use the summary instead of raw history

    This enables indefinitely long sessions without hitting context limits.
    """

    def __init__(
        self,
        repo: AHCRepository,
        llm_client: Any,
        session_id: int,
        config: CompactionConfig | None = None,
    ):
        """Initialize the compaction manager.

        Args:
            repo: Database repository
            llm_client: LLM client for summarization calls
            session_id: Current session ID
            config: Optional configuration
        """
        self.repo = repo
        self.llm = llm_client
        self.session_id = session_id
        self.config = config or CompactionConfig()

    def should_compact(self) -> bool:
        """Check if compaction is needed.

        Returns:
            True if compaction should be triggered
        """
        # Get current stats
        total_tokens = self.repo.get_total_token_count(self.session_id)
        turns_since = self.repo.get_turn_count_since_compaction(self.session_id)

        # Check thresholds
        if total_tokens > self.config.token_threshold:
            logger.info(
                f"Compaction triggered: {total_tokens} tokens > {self.config.token_threshold}"
            )
            return True

        if turns_since > self.config.turn_threshold:
            logger.info(
                f"Compaction triggered: {turns_since} turns > {self.config.turn_threshold}"
            )
            return True

        return False

    def compact(self) -> CompactionResult:
        """Run the compaction workflow.

        Returns:
            CompactionResult with outcome details
        """
        try:
            # Get turns since last compaction
            turns = self.repo.get_turns_since_compaction(self.session_id)

            if len(turns) < self.config.min_turns_between:
                return CompactionResult(
                    success=False,
                    version=0,
                    turns_compacted=0,
                    tokens_before=0,
                    tokens_after=0,
                    error=f"Only {len(turns)} turns since last compaction, minimum is {self.config.min_turns_between}",
                )

            # Calculate tokens before
            tokens_before = sum(estimate_tokens(t.content) for t in turns)

            # Build summarization prompt
            turns_text = self._format_turns_for_summary(turns)
            full_prompt = COMPACTION_PROMPT + turns_text

            # Call LLM for summarization
            summary = self._call_summarization_llm(full_prompt)

            # Get existing negative knowledge to merge
            existing_meta = self.repo.get_latest_meta_knowledge(self.session_id)
            merged_negative = self._merge_negative_knowledge(
                existing_meta.negative_knowledge if existing_meta else None,
                summary.get("negative_knowledge"),
            )

            # Get current theorems snapshot
            theorems = self.repo.get_theorems(
                self.session_id,
                status=TheoremStatus.VALIDATED,
            )
            theorems_json = json.dumps([
                {"name": t.name, "description": t.description}
                for t in theorems
            ])

            # Calculate tokens after (estimate)
            tokens_after = estimate_tokens(merged_negative or "") + estimate_tokens(theorems_json)

            # Get next version
            version = self.repo.get_next_meta_knowledge_version(self.session_id)

            # Create and store meta knowledge record
            record = MetaKnowledgeRecord(
                session_id=self.session_id,
                version=version,
                theorems_snapshot_json=theorems_json,
                negative_knowledge=merged_negative,
                last_compacted_turn=turns[-1].turn_number if turns else 0,
                turns_compacted=len(turns),
                token_count_before=tokens_before,
                token_count_after=tokens_after,
                compaction_prompt_hash=hashlib.sha256(full_prompt.encode()).hexdigest()[:16],
            )
            self.repo.insert_meta_knowledge(record)

            logger.info(
                f"Compaction complete: v{version}, {len(turns)} turns, "
                f"{tokens_before} -> {tokens_after} tokens"
            )

            return CompactionResult(
                success=True,
                version=version,
                turns_compacted=len(turns),
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                new_negative_knowledge=summary.get("negative_knowledge"),
            )

        except Exception as e:
            logger.exception("Compaction failed")
            return CompactionResult(
                success=False,
                version=0,
                turns_compacted=0,
                tokens_before=0,
                tokens_after=0,
                error=str(e),
            )

    def _format_turns_for_summary(self, turns: list) -> str:
        """Format turns for the summarization prompt.

        Args:
            turns: List of ConversationTurnRecords

        Returns:
            Formatted string of turns
        """
        lines = []
        for t in turns:
            role_label = t.role.upper()
            # Truncate very long content
            content = t.content[:2000] if len(t.content) > 2000 else t.content
            lines.append(f"[Turn {t.turn_number}] {role_label}:\n{content}\n")
        return "\n".join(lines)

    def _call_summarization_llm(self, prompt: str) -> dict[str, Any]:
        """Call the LLM to summarize turns.

        Args:
            prompt: Full summarization prompt

        Returns:
            Parsed summary dict with negative_knowledge and theorem_candidates
        """
        if self.llm is None:
            logger.warning("No LLM client for compaction, using empty summary")
            return {"negative_knowledge": None, "theorem_candidates": []}

        try:
            # Use generate_with_tools with no tools for simple completion
            response = self.llm.generate_with_tools(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                temperature=0.3,  # Lower temperature for summarization
            )

            content = response.get("content", "")

            # Try to extract JSON from response
            return self._parse_summary_response(content)

        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return {"negative_knowledge": None, "theorem_candidates": []}

    def _parse_summary_response(self, content: str) -> dict[str, Any]:
        """Parse the LLM's summary response.

        Args:
            content: Raw LLM response

        Returns:
            Parsed dict with negative_knowledge and theorem_candidates
        """
        # Try to find JSON in the response
        try:
            # Look for JSON block
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "{" in content:
                # Find the JSON object
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
            else:
                return {"negative_knowledge": None, "theorem_candidates": []}

            result = json.loads(json_str)

            # Ensure negative_knowledge is a string (LLM might return a list)
            neg_knowledge = result.get("negative_knowledge")
            if isinstance(neg_knowledge, list):
                result["negative_knowledge"] = "\n".join(f"- {item}" for item in neg_knowledge)
            elif neg_knowledge is not None and not isinstance(neg_knowledge, str):
                result["negative_knowledge"] = str(neg_knowledge)

            return result

        except json.JSONDecodeError:
            logger.warning("Failed to parse summary JSON")
            return {"negative_knowledge": None, "theorem_candidates": []}

    def _merge_negative_knowledge(
        self,
        existing: str | None,
        new: str | list | None,
    ) -> str | None:
        """Merge existing and new negative knowledge.

        Args:
            existing: Previous negative knowledge
            new: Newly extracted negative knowledge (may be string or list)

        Returns:
            Merged negative knowledge string
        """
        # Convert list to string if needed
        if isinstance(new, list):
            new = "\n".join(f"- {item}" for item in new)

        if not existing and not new:
            return None
        if not existing:
            return new
        if not new:
            return existing

        # Simple concatenation with deduplication could be improved
        # For now, just append new knowledge
        return f"{existing}\n\n### Additional Learnings:\n{new}"
