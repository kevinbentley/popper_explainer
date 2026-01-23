#!/usr/bin/env python3
"""Lint laws in the database for semantic consistency.

Checks for mismatches between law names and quantity types:
- "particle_conserved" that uses cell_count instead of particle_count
- "momentum_..." that uses cell_count instead of momentum_like
- Observable names that don't match their expressions
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.claims.schema import CandidateLaw
from src.claims.semantic_linter import SemanticLinter
from src.claims.quantity_types import QuantityType
from src.db.repo import Repository


def main():
    parser = argparse.ArgumentParser(description="Lint laws for semantic consistency")
    parser.add_argument(
        "--db", "-d",
        type=str,
        default="results/discovery.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--show-types",
        action="store_true",
        help="Show inferred quantity types for all laws",
    )
    parser.add_argument(
        "--law-id",
        type=str,
        help="Lint a specific law only",
    )
    parser.add_argument(
        "--suggest-fixes",
        action="store_true",
        help="Suggest name fixes for misnamed laws",
    )
    parser.add_argument(
        "--min-severity",
        type=str,
        choices=["info", "warning", "error"],
        default="warning",
        help="Minimum severity to show (default: warning)",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    repo = Repository(db_path)
    repo.connect()

    linter = SemanticLinter(strict=args.strict)

    # Get laws to lint
    if args.law_id:
        law_record = repo.get_law(args.law_id)
        if not law_record:
            print(f"ERROR: Law not found: {args.law_id}")
            sys.exit(1)
        law_records = [law_record]
    else:
        # Get all laws
        cursor = repo._conn.cursor()
        cursor.execute("SELECT law_id, law_json FROM laws ORDER BY law_id")
        law_records = cursor.fetchall()

    print(f"Linting {len(law_records)} laws...")
    print()

    # Track statistics
    total_warnings = 0
    total_errors = 0
    laws_with_issues = 0
    type_counts = {qt: 0 for qt in QuantityType}

    for record in law_records:
        # Handle both tuple (from raw query) and object (from repo method)
        if hasattr(record, 'law_id'):
            law_id = record.law_id
            law_json = json.loads(record.law_json)
        else:
            law_id, law_json_str = record[0], record[1]
            law_json = json.loads(law_json_str)

        try:
            law = CandidateLaw.model_validate(law_json)
        except Exception as e:
            print(f"ERROR: Could not parse law {law_id}: {e}")
            continue

        result = linter.lint(law)

        # Count types
        for _, typed in result.observable_types:
            type_counts[typed.quantity_type] += 1

        # Show types if requested
        if args.show_types:
            print(f"{law_id}:")
            for obs_name, typed in result.observable_types:
                print(f"  {obs_name}: {typed.quantity_type.value} ({typed.description})")
            print()

        # Filter warnings by severity
        severity_order = {"info": 0, "warning": 1, "error": 2}
        min_severity = severity_order.get(args.min_severity, 1)
        filtered_warnings = [
            w for w in result.warnings
            if severity_order.get(w.severity, 1) >= min_severity
        ]

        # Show warnings
        if filtered_warnings:
            laws_with_issues += 1
            if not args.show_types:  # Avoid double-printing
                print(f"{law_id}:")
            for warning in filtered_warnings:
                if warning.severity == "error":
                    prefix = "  ERROR:"
                elif warning.severity == "warning":
                    prefix = "  WARNING:"
                else:
                    prefix = "  INFO:"
                print(f"{prefix} {warning.message}")
                if warning.suggested_fix and args.suggest_fixes:
                    print(f"    Suggested fix: {warning.suggested_fix}")
                if warning.severity == "error":
                    total_errors += 1
                else:
                    total_warnings += 1
            print()

    repo.close()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Laws linted: {len(law_records)}")
    print(f"Laws with issues: {laws_with_issues}")
    print(f"Total warnings: {total_warnings}")
    print(f"Total errors: {total_errors}")
    print()

    if args.show_types:
        print("Quantity type distribution:")
        for qt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"  {qt.value}: {count}")

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
