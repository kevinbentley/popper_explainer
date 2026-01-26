"""Journal manager for chain-of-thought persistence."""

import json
import logging
from datetime import datetime
from typing import Any

from src.ahc.db.models import JournalEntry, JournalEntryType
from src.ahc.db.repo import AHCRepository

logger = logging.getLogger(__name__)


class JournalManager:
    """Manages chain-of-thought journal entries.

    The journal provides a persistent record of the agent's
    reasoning process, observations, and conclusions.
    """

    def __init__(self, repo: AHCRepository, session_id: int):
        """Initialize the journal manager.

        Args:
            repo: Database repository
            session_id: Session ID for this journal
        """
        self._repo = repo
        self._session_id = session_id
        self._current_turn = 0

    def set_turn(self, turn_number: int) -> None:
        """Set the current turn number.

        Args:
            turn_number: Current turn
        """
        self._current_turn = turn_number

    def log_thought(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Log a thought entry.

        Args:
            content: The thought content
            metadata: Optional metadata

        Returns:
            Entry ID
        """
        return self._log_entry(JournalEntryType.THOUGHT, content, metadata)

    def log_observation(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Log an observation entry.

        Args:
            content: The observation content
            metadata: Optional metadata

        Returns:
            Entry ID
        """
        return self._log_entry(JournalEntryType.OBSERVATION, content, metadata)

    def log_hypothesis(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Log a hypothesis entry.

        Args:
            content: The hypothesis content
            metadata: Optional metadata

        Returns:
            Entry ID
        """
        return self._log_entry(JournalEntryType.HYPOTHESIS, content, metadata)

    def log_experiment(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Log an experiment entry.

        Args:
            content: The experiment content
            metadata: Optional metadata

        Returns:
            Entry ID
        """
        return self._log_entry(JournalEntryType.EXPERIMENT, content, metadata)

    def log_conclusion(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Log a conclusion entry.

        Args:
            content: The conclusion content
            metadata: Optional metadata

        Returns:
            Entry ID
        """
        return self._log_entry(JournalEntryType.CONCLUSION, content, metadata)

    def log_error(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Log an error entry.

        Args:
            content: The error content
            metadata: Optional metadata

        Returns:
            Entry ID
        """
        return self._log_entry(JournalEntryType.ERROR, content, metadata)

    def _log_entry(
        self,
        entry_type: JournalEntryType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Log a journal entry.

        Args:
            entry_type: Type of entry
            content: Entry content
            metadata: Optional metadata

        Returns:
            Entry ID
        """
        entry = JournalEntry(
            session_id=self._session_id,
            turn_number=self._current_turn,
            entry_type=entry_type,
            content=content,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        return self._repo.insert_journal_entry(entry)

    def get_recent_entries(
        self,
        count: int = 10,
        entry_type: JournalEntryType | None = None,
    ) -> list[JournalEntry]:
        """Get recent journal entries.

        Args:
            count: Number of entries to retrieve
            entry_type: Optional type filter

        Returns:
            List of journal entries
        """
        return self._repo.get_journal_entries(
            self._session_id,
            entry_type=entry_type,
            limit=count,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the journal.

        Returns:
            Summary statistics
        """
        entries = self._repo.get_journal_entries(self._session_id)

        type_counts = {}
        for entry in entries:
            t = entry.entry_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_entries": len(entries),
            "by_type": type_counts,
            "turns_covered": len(set(e.turn_number for e in entries)),
        }

    def format_for_context(self, max_entries: int = 20) -> str:
        """Format recent entries for LLM context.

        Args:
            max_entries: Maximum entries to include

        Returns:
            Formatted string for context injection
        """
        entries = self.get_recent_entries(count=max_entries)

        if not entries:
            return "No journal entries yet."

        lines = ["## Recent Journal Entries\n"]
        current_turn = None

        for entry in entries:
            if entry.turn_number != current_turn:
                current_turn = entry.turn_number
                lines.append(f"\n### Turn {current_turn}\n")

            type_icon = {
                JournalEntryType.THOUGHT: "💭",
                JournalEntryType.OBSERVATION: "👁",
                JournalEntryType.HYPOTHESIS: "❓",
                JournalEntryType.EXPERIMENT: "🧪",
                JournalEntryType.CONCLUSION: "✅",
                JournalEntryType.ERROR: "❌",
            }.get(entry.entry_type, "•")

            lines.append(f"{type_icon} **{entry.entry_type.value.upper()}**: {entry.content}")

        return "\n".join(lines)
