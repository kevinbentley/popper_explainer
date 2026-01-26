#!/usr/bin/env python3
"""Export research logs to text files for human review.

Usage:
    python scripts/export_research_logs.py
    python scripts/export_research_logs.py --db results/discovery.db
    python scripts/export_research_logs.py --output exports/
    python scripts/export_research_logs.py --run-id my_run_123

The exported files will be organized as:
    output_dir/
        research_logs/
            run_<run_id>/
                iteration_001.txt
                iteration_002.txt
                ...
                combined.txt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.proposer.research_log_exporter import (
    ResearchLogExporter,
    export_from_database,
)


def main():
    parser = argparse.ArgumentParser(
        description="Export research logs from discovery database for human review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        type=str,
        default="results/discovery.db",
        help="Path to database file (default: results/discovery.db)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run ID to export (exports all runs if not specified)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        print("\nTo export research logs, you need a database from a discovery run.")
        print("Try running a discovery session first:")
        print("    python scripts/run_discovery.py")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Reading from: {db_path}")
        print(f"Output dir: {output_dir}")
        if args.run_id:
            print(f"Run ID filter: {args.run_id}")

    try:
        output_path = export_from_database(
            db_path=db_path,
            output_dir=output_dir,
            run_id=args.run_id,
        )

        if output_path and output_path.exists():
            print(f"\nResearch logs exported successfully!")
            print(f"Combined log: {output_path}")

            # Show directory contents
            logs_dir = output_path.parent
            files = sorted(logs_dir.glob("*.txt"))
            print(f"\nFiles in {logs_dir}/:")
            for f in files:
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name} ({size_kb:.1f} KB)")
        else:
            print("\nNo research logs found to export.")
            print("\nThis could mean:")
            print("  1. The database has no completed iterations yet")
            print("  2. The iterations don't have research logs in their summary")
            print("  3. The research_log feature wasn't enabled during the run")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error exporting logs: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
