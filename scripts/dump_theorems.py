#!/usr/bin/env python3
"""Dump theorems from the database to a text file.

Usage:
    python scripts/dump_theorems.py --db results/discovery.db
    python scripts/dump_theorems.py --db results/discovery.db --output theorems.txt
    python scripts/dump_theorems.py --db results/discovery.db --run-id thm_run_abc123
    python scripts/dump_theorems.py --db results/discovery.db --format markdown
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.repo import Repository


def format_theorem_text(theorem, index: int) -> str:
    """Format a single theorem as plain text."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"THEOREM {index}: {theorem.name}")
    lines.append(f"{'=' * 70}")
    lines.append(f"Status: {theorem.status}")
    lines.append(f"ID: {theorem.theorem_id}")
    lines.append("")
    lines.append("CLAIM:")
    lines.append(theorem.claim)
    lines.append("")

    # Support
    support = json.loads(theorem.support_json) if theorem.support_json else []
    lines.append(f"SUPPORT ({len(support)} laws):")
    for s in support:
        lines.append(f"  - {s['law_id']}: {s.get('role', 'confirms')}")
    lines.append("")

    # Failure modes
    failure_modes = json.loads(theorem.failure_modes_json) if theorem.failure_modes_json else []
    if failure_modes:
        lines.append(f"FAILURE MODES ({len(failure_modes)}):")
        for fm in failure_modes:
            lines.append(f"  - {fm}")
        lines.append("")

    # Missing structure
    missing = json.loads(theorem.missing_structure_json) if theorem.missing_structure_json else []
    if missing:
        lines.append(f"MISSING STRUCTURE ({len(missing)}):")
        for ms in missing:
            lines.append(f"  - {ms}")
        lines.append("")

    # Signature
    if theorem.failure_signature_hash:
        lines.append(f"Signature Hash: {theorem.failure_signature_hash}")
        lines.append("")

    return "\n".join(lines)


def format_theorem_markdown(theorem, index: int) -> str:
    """Format a single theorem as markdown."""
    lines = []
    lines.append(f"## Theorem {index}: {theorem.name}")
    lines.append("")
    lines.append(f"**Status:** {theorem.status}")
    lines.append(f"**ID:** `{theorem.theorem_id}`")
    lines.append("")
    lines.append("### Claim")
    lines.append("")
    lines.append(theorem.claim)
    lines.append("")

    # Support
    support = json.loads(theorem.support_json) if theorem.support_json else []
    lines.append(f"### Support ({len(support)} laws)")
    lines.append("")
    for s in support:
        lines.append(f"- `{s['law_id']}`: {s.get('role', 'confirms')}")
    lines.append("")

    # Failure modes
    failure_modes = json.loads(theorem.failure_modes_json) if theorem.failure_modes_json else []
    if failure_modes:
        lines.append(f"### Failure Modes")
        lines.append("")
        for fm in failure_modes:
            lines.append(f"- {fm}")
        lines.append("")

    # Missing structure
    missing = json.loads(theorem.missing_structure_json) if theorem.missing_structure_json else []
    if missing:
        lines.append(f"### Missing Structure")
        lines.append("")
        for ms in missing:
            lines.append(f"- {ms}")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Dump theorems from the database to a text file"
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to the SQLite database",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: theorems_<timestamp>.txt)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific theorem run ID to dump (default: latest)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["Established", "Conditional", "Conjectural"],
        default=None,
        help="Filter by theorem status",
    )

    args = parser.parse_args()

    # Check database exists
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    # Connect to database
    repo = Repository(db_path)
    repo.connect()

    try:
        # Get theorem run
        if args.run_id:
            run = repo.get_theorem_run(args.run_id)
            if not run:
                print(f"Error: Theorem run '{args.run_id}' not found")
                sys.exit(1)
        else:
            runs = repo.list_theorem_runs(limit=1)
            if not runs:
                print("Error: No theorem runs found in database")
                sys.exit(1)
            run = runs[0]

        # Get theorems
        theorems = repo.get_theorems_by_run(run.id)
        if args.status:
            theorems = [t for t in theorems if t.status == args.status]

        if not theorems:
            print(f"No theorems found for run {run.run_id}")
            sys.exit(0)

        # Determine output file
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "md" if args.format == "markdown" else "json" if args.format == "json" else "txt"
            output_path = Path(f"theorems_{timestamp}.{ext}")

        # Format output
        if args.format == "json":
            output_data = {
                "run_id": run.run_id,
                "status": run.status,
                "generated_at": run.started_at,
                "theorems": []
            }
            for t in theorems:
                output_data["theorems"].append({
                    "theorem_id": t.theorem_id,
                    "name": t.name,
                    "status": t.status,
                    "claim": t.claim,
                    "support": json.loads(t.support_json) if t.support_json else [],
                    "failure_modes": json.loads(t.failure_modes_json) if t.failure_modes_json else [],
                    "missing_structure": json.loads(t.missing_structure_json) if t.missing_structure_json else [],
                    "failure_signature_hash": t.failure_signature_hash,
                })
            content = json.dumps(output_data, indent=2)
        else:
            lines = []

            # Header
            if args.format == "markdown":
                lines.append(f"# Theorems from {run.run_id}")
                lines.append("")
                lines.append(f"**Run Status:** {run.status}")
                lines.append(f"**Total Theorems:** {len(theorems)}")
                lines.append(f"**Generated:** {run.started_at}")
                lines.append("")
                lines.append("---")
                lines.append("")
            else:
                lines.append(f"THEOREM DUMP")
                lines.append(f"Run ID: {run.run_id}")
                lines.append(f"Run Status: {run.status}")
                lines.append(f"Total Theorems: {len(theorems)}")
                lines.append(f"Generated: {run.started_at}")
                lines.append("")

            # Theorems
            format_fn = format_theorem_markdown if args.format == "markdown" else format_theorem_text
            for i, theorem in enumerate(theorems, 1):
                lines.append(format_fn(theorem, i))

            content = "\n".join(lines)

        # Write output
        output_path.write_text(content)
        print(f"Dumped {len(theorems)} theorems to {output_path}")

        # Also print summary
        print(f"\nSummary:")
        print(f"  Run: {run.run_id}")
        print(f"  Theorems: {len(theorems)}")
        by_status = {}
        for t in theorems:
            by_status[t.status] = by_status.get(t.status, 0) + 1
        for status, count in sorted(by_status.items()):
            print(f"    {status}: {count}")

    finally:
        repo.close()


if __name__ == "__main__":
    main()
