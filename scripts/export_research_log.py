#!/usr/bin/env python3
"""Export LLM research logs from the discovery database.

Usage:
    python scripts/export_research_log.py [--db PATH] [--output PATH] [--format FORMAT]

Examples:
    python scripts/export_research_log.py
    python scripts/export_research_log.py --db results/discovery.db --output research_log.txt
    python scripts/export_research_log.py --format json
"""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path


def get_research_logs(db_path: str, component: str = "law_proposer") -> list[dict]:
    """Extract research logs from the database.

    Args:
        db_path: Path to the SQLite database
        component: Component to filter by (default: law_proposer)

    Returns:
        List of dicts with iteration info and research_log content
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Try to get research_log from raw_response JSON
    cursor.execute("""
        SELECT
            id,
            run_id,
            iteration_id,
            phase,
            component,
            created_at,
            raw_response
        FROM llm_transcripts
        WHERE component = ?
        ORDER BY created_at ASC
    """, (component,))

    results = []
    for row in cursor.fetchall():
        entry = {
            "id": row["id"],
            "run_id": row["run_id"],
            "iteration_id": row["iteration_id"],
            "phase": row["phase"],
            "component": row["component"],
            "created_at": row["created_at"],
            "research_log": None,
        }

        # Try to extract research_log from raw_response JSON
        raw_response = row["raw_response"]
        if raw_response:
            try:
                data = json.loads(raw_response)
                if isinstance(data, dict):
                    entry["research_log"] = data.get("research_log")
            except json.JSONDecodeError:
                pass

        if entry["research_log"]:
            results.append(entry)

    conn.close()
    return results


def format_as_text(logs: list[dict]) -> str:
    """Format research logs as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("RESEARCH LOG HISTORY")
    lines.append(f"Exported: {datetime.now().isoformat()}")
    lines.append(f"Total entries: {len(logs)}")
    lines.append("=" * 80)
    lines.append("")

    for i, log in enumerate(logs, 1):
        lines.append(f"--- Entry {i} ---")
        lines.append(f"Iteration: {log['iteration_id']}")
        lines.append(f"Phase: {log['phase']}")
        lines.append(f"Time: {log['created_at']}")
        lines.append("")
        research_log = log["research_log"]
        if isinstance(research_log, dict):
            lines.append(json.dumps(research_log, indent=2))
        else:
            lines.append(str(research_log))
        lines.append("")
        lines.append("")

    return "\n".join(lines)


def format_as_markdown(logs: list[dict]) -> str:
    """Format research logs as Markdown."""
    lines = []
    lines.append("# Research Log History")
    lines.append("")
    lines.append(f"**Exported:** {datetime.now().isoformat()}")
    lines.append(f"**Total entries:** {len(logs)}")
    lines.append("")

    for i, log in enumerate(logs, 1):
        lines.append(f"## Iteration {log['iteration_id'] or i}")
        lines.append("")
        lines.append(f"*Phase: {log['phase']} | Time: {log['created_at']}*")
        lines.append("")
        research_log = log["research_log"]
        if isinstance(research_log, dict):
            lines.append("```json")
            lines.append(json.dumps(research_log, indent=2))
            lines.append("```")
        else:
            lines.append(str(research_log))
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def format_as_json(logs: list[dict]) -> str:
    """Format research logs as JSON."""
    return json.dumps(logs, indent=2)


def export_research_log(
    db_path: str = "results/discovery.db",
    output_path: str | None = None,
    format: str = "text",
    component: str = "law_proposer",
) -> str:
    """Export research logs to a file or return as string.

    Args:
        db_path: Path to the SQLite database
        output_path: Output file path (if None, returns string)
        format: Output format ('text', 'markdown', 'json')
        component: Component to filter by

    Returns:
        Formatted research log content
    """
    logs = get_research_logs(db_path, component)

    if format == "json":
        content = format_as_json(logs)
    elif format == "markdown" or format == "md":
        content = format_as_markdown(logs)
    else:
        content = format_as_text(logs)

    if output_path:
        Path(output_path).write_text(content)
        print(f"Exported {len(logs)} entries to {output_path}")

    return content


def main():
    parser = argparse.ArgumentParser(
        description="Export LLM research logs from the discovery database"
    )
    parser.add_argument(
        "--db",
        default="results/discovery.db",
        help="Path to the SQLite database (default: results/discovery.db)"
    )
    parser.add_argument(
        "--output", "-o",
        default="research_log_history.txt",
        help="Output file path (default: research_log_history.txt)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "markdown", "md", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--component", "-c",
        default="law_proposer",
        help="Component to filter by (default: law_proposer)"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of file"
    )

    args = parser.parse_args()

    output_path = None if args.stdout else args.output
    content = export_research_log(
        db_path=args.db,
        output_path=output_path,
        format=args.format,
        component=args.component,
    )

    if args.stdout:
        print(content)


if __name__ == "__main__":
    main()
