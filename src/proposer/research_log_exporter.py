"""Research Log Exporter.

Exports research logs to text files for human review of the LLM's
thinking process across discovery iterations.

The research log is the LLM's internal notebook - it contains:
- Hypotheses being tested
- Lessons learned from falsifications
- Patterns discovered
- Next steps planned

By reviewing these logs, humans can understand:
- How the AI's understanding evolves over time
- Whether it's learning from counterexamples
- What hypotheses it's pursuing and why
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ResearchLogEntry:
    """A single research log entry with metadata."""

    content: str
    iteration: int
    timestamp: datetime
    run_id: str | None = None
    phase: str | None = None
    laws_proposed: int = 0
    laws_accepted: int = 0
    laws_rejected: int = 0

    def to_text(self) -> str:
        """Format as human-readable text."""
        lines = [
            "=" * 72,
            f"RESEARCH LOG - Iteration {self.iteration}",
            "=" * 72,
            "",
            f"Timestamp: {self.timestamp.isoformat()}",
        ]

        if self.run_id:
            lines.append(f"Run ID: {self.run_id}")
        if self.phase:
            lines.append(f"Phase: {self.phase}")

        lines.extend([
            f"Laws proposed this iteration: {self.laws_proposed}",
            f"Laws accepted: {self.laws_accepted}",
            f"Laws rejected (syntax/schema): {self.laws_rejected}",
            "",
            "-" * 72,
            "LLM'S RESEARCH NOTES:",
            "-" * 72,
            "",
            self.content,
            "",
            "=" * 72,
        ])

        return "\n".join(lines)


class ResearchLogExporter:
    """Exports research logs to text files for human review.

    Creates a directory structure:
        output_dir/
            research_logs/
                run_<run_id>/
                    iteration_001.txt
                    iteration_002.txt
                    ...
                    combined.txt  (all logs concatenated)

    Usage:
        exporter = ResearchLogExporter("results/")

        # After each iteration
        exporter.export_log(
            content=batch.research_log,
            iteration=iteration_num,
            run_id="run_001",
            laws_proposed=len(batch.laws),
            laws_accepted=accepted_count,
        )

        # At end of run, create combined file
        exporter.export_combined(run_id="run_001")
    """

    def __init__(self, output_dir: str | Path):
        """Initialize the exporter.

        Args:
            output_dir: Base directory for output files
        """
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "research_logs"
        self._entries: dict[str, list[ResearchLogEntry]] = {}

    def _get_run_dir(self, run_id: str | None) -> Path:
        """Get the directory for a specific run."""
        if run_id:
            return self.logs_dir / f"run_{run_id}"
        else:
            # Use date-based directory if no run_id
            date_str = datetime.now().strftime("%Y%m%d")
            return self.logs_dir / f"session_{date_str}"

    def export_log(
        self,
        content: str | None,
        iteration: int,
        run_id: str | None = None,
        phase: str | None = None,
        laws_proposed: int = 0,
        laws_accepted: int = 0,
        laws_rejected: int = 0,
        timestamp: datetime | None = None,
    ) -> Path | None:
        """Export a single research log to a text file.

        Args:
            content: The research log content (may be None if LLM didn't provide one)
            iteration: Iteration number (1-indexed)
            run_id: Optional run identifier
            phase: Optional phase name (e.g., "law_discovery", "theorem")
            laws_proposed: Number of laws proposed this iteration
            laws_accepted: Number that passed schema validation
            laws_rejected: Number rejected for syntax/schema errors
            timestamp: When the log was generated (defaults to now)

        Returns:
            Path to the exported file, or None if content was empty/None
        """
        if not content or not content.strip():
            return None

        timestamp = timestamp or datetime.now()

        entry = ResearchLogEntry(
            content=content.strip(),
            iteration=iteration,
            timestamp=timestamp,
            run_id=run_id,
            phase=phase,
            laws_proposed=laws_proposed,
            laws_accepted=laws_accepted,
            laws_rejected=laws_rejected,
        )

        # Track for combined export
        run_key = run_id or "default"
        if run_key not in self._entries:
            self._entries[run_key] = []
        self._entries[run_key].append(entry)

        # Create directory
        run_dir = self._get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write individual file
        filename = f"iteration_{iteration:03d}.txt"
        filepath = run_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(entry.to_text())

        return filepath

    def export_combined(self, run_id: str | None = None) -> Path | None:
        """Create a combined file with all logs for a run.

        This is useful for reviewing the entire thinking process
        in a single document.

        Args:
            run_id: The run to export

        Returns:
            Path to the combined file, or None if no logs exist
        """
        run_key = run_id or "default"
        entries = self._entries.get(run_key, [])

        if not entries:
            return None

        # Sort by iteration
        entries = sorted(entries, key=lambda e: e.iteration)

        run_dir = self._get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        filepath = run_dir / "combined.txt"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("RESEARCH LOG COMPILATION\n")
            f.write(f"Run: {run_id or 'default'}\n")
            f.write(f"Total iterations: {len(entries)}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("\n" + "=" * 72 + "\n\n")

            for entry in entries:
                f.write(entry.to_text())
                f.write("\n\n")

        return filepath

    def export_summary(self, run_id: str | None = None) -> dict[str, Any]:
        """Generate a summary of research log evolution.

        Returns statistics about how the research log evolved over time.

        Args:
            run_id: The run to summarize

        Returns:
            Dictionary with summary statistics
        """
        run_key = run_id or "default"
        entries = self._entries.get(run_key, [])

        if not entries:
            return {"error": "No logs found"}

        entries = sorted(entries, key=lambda e: e.iteration)

        # Analyze content evolution
        word_counts = [len(e.content.split()) for e in entries]
        total_laws_proposed = sum(e.laws_proposed for e in entries)
        total_laws_accepted = sum(e.laws_accepted for e in entries)

        return {
            "run_id": run_id,
            "total_iterations": len(entries),
            "first_iteration": entries[0].iteration,
            "last_iteration": entries[-1].iteration,
            "avg_log_words": sum(word_counts) / len(word_counts) if word_counts else 0,
            "min_log_words": min(word_counts) if word_counts else 0,
            "max_log_words": max(word_counts) if word_counts else 0,
            "total_laws_proposed": total_laws_proposed,
            "total_laws_accepted": total_laws_accepted,
            "acceptance_rate": total_laws_accepted / total_laws_proposed if total_laws_proposed > 0 else 0,
        }


def export_from_database(
    db_path: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
) -> Path | None:
    """Export research logs from the database.

    This extracts research logs from the llm_transcripts table where they're
    stored during law proposer calls.

    Args:
        db_path: Path to the SQLite database
        output_dir: Directory to write output files
        run_id: Optional run ID to filter (exports all if None)

    Returns:
        Path to the combined log file, or None if no logs found
    """
    import sqlite3
    from datetime import datetime

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    exporter = ResearchLogExporter(output_dir)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        # Check if llm_transcripts table exists and has research_log column
        cursor.execute("PRAGMA table_info(llm_transcripts)")
        columns = {row["name"] for row in cursor.fetchall()}

        if "research_log" not in columns:
            print("Warning: No research_log column found in llm_transcripts table")
            print("Try running: sqlite3 your.db < src/db/schema.sql")
            print("Or add column: ALTER TABLE llm_transcripts ADD COLUMN research_log TEXT")
            return None

        # Query LLM transcripts for law_proposer with research logs
        query = """
            SELECT
                t.iteration_id,
                t.run_id,
                t.phase,
                t.research_log,
                t.created_at,
                t.id as transcript_id
            FROM llm_transcripts t
            WHERE t.component = 'law_proposer'
              AND t.research_log IS NOT NULL
              AND t.research_log != ''
        """
        params: list[Any] = []

        if run_id:
            query += " AND t.run_id = ?"
            params.append(run_id)

        query += " ORDER BY t.run_id, t.iteration_id, t.id"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            print("No research logs found in llm_transcripts")
            return None

        for row in rows:
            research_log = row["research_log"]
            if not research_log:
                continue

            # Parse timestamp
            created_at = row["created_at"]
            if isinstance(created_at, str):
                try:
                    timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            iteration = row["iteration_id"] if row["iteration_id"] else row["transcript_id"]

            exporter.export_log(
                content=research_log,
                iteration=iteration,
                run_id=row["run_id"],
                phase=row["phase"],
                timestamp=timestamp,
            )

        # Create combined files for each run
        for rkey in exporter._entries:
            exporter.export_combined(run_id=rkey if rkey != "default" else None)

        # Return the path to the first run's combined file
        first_run = list(exporter._entries.keys())[0] if exporter._entries else None
        if first_run:
            return exporter._get_run_dir(first_run if first_run != "default" else None) / "combined.txt"

        return None

    finally:
        conn.close()


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export research logs from discovery database"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="results/discovery.db",
        help="Path to database file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run ID to export (exports all if not specified)",
    )

    args = parser.parse_args()

    try:
        output_path = export_from_database(
            db_path=args.db,
            output_dir=args.output,
            run_id=args.run_id,
        )
        if output_path:
            print(f"Research logs exported to: {output_path}")
        else:
            print("No research logs found to export")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
