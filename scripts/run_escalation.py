#!/usr/bin/env python3
"""Run power escalation on accepted laws.

Escalation re-tests laws that have passed under baseline configuration
with progressively stronger harness settings to identify false positives.

Usage:
    python scripts/run_escalation.py --level escalation_1
    python scripts/run_escalation.py --level escalation_2 --db results/discovery.db
    python scripts/run_escalation.py --level escalation_3 --seed 12345 -v
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.repo import Repository
from src.harness.escalation import (
    EscalationLevel,
    FlipType,
    RetestResult,
    get_escalation_summary,
    get_preset,
    list_levels,
    run_escalation,
)


def progress_callback(current: int, total: int, result: RetestResult, verbose: bool) -> None:
    """Print progress during escalation."""
    status_symbol = {
        FlipType.STABLE: "+",
        FlipType.REVOKED: "X",
        FlipType.DOWNGRADED: "?",
    }[result.flip_type]

    if verbose:
        print(f"  [{current}/{total}] {result.law_id}: {result.old_status} -> {result.new_status} ({status_symbol})")
    elif current % 10 == 0 or current == total:
        print(f"  Progress: {current}/{total} laws tested")


def print_level_info() -> None:
    """Print information about available escalation levels."""
    print("\nAvailable escalation levels:\n")
    for preset in list_levels():
        config = preset.config
        print(f"  {preset.level.value}:")
        print(f"    Description: {preset.description}")
        print(f"    Cost factor: {preset.cost_factor}x")
        print(f"    max_cases: {config.max_cases}")
        print(f"    default_T: {config.default_T}, max_T: {config.max_T}")
        print(f"    adversarial_budget: {config.adversarial_budget}")
        print(f"    min_cases_used_for_pass: {config.min_cases_used_for_pass}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run power escalation on accepted laws",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --level escalation_1                    # Run level 1 escalation
  %(prog)s --level escalation_2 --db results/my.db # Use specific database
  %(prog)s --list-levels                           # Show available levels
  %(prog)s -l escalation_1 -v                      # Verbose output
        """,
    )

    parser.add_argument(
        "--level", "-l",
        type=str,
        choices=["escalation_1", "escalation_2", "escalation_3"],
        help="Escalation level to run",
    )
    parser.add_argument(
        "--db", "-d",
        type=str,
        default="results/discovery.db",
        help="Path to SQLite database (default: results/discovery.db)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed override for reproducibility",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress for each law",
    )
    parser.add_argument(
        "--list-levels",
        action="store_true",
        help="List available escalation levels and exit",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show escalation summary from database and exit",
    )

    args = parser.parse_args()

    # Handle --list-levels
    if args.list_levels:
        print_level_info()
        return 0

    # Handle --summary
    if args.summary:
        db_path = Path(args.db)
        if not db_path.exists():
            print(f"Error: Database not found: {db_path}")
            return 1

        with Repository(db_path) as repo:
            summary = get_escalation_summary(repo)
            if not summary:
                print("No escalation runs found in database.")
                return 0

            print("\nEscalation Summary:\n")
            for level, counts in sorted(summary.items()):
                print(f"  {level}:")
                print(f"    Total runs: {counts['total_runs']}")
                print(f"    Stable (PASS->PASS): {counts['stable']}")
                print(f"    Revoked (PASS->FAIL): {counts['revoked']}")
                print(f"    Downgraded (PASS->UNKNOWN): {counts['downgraded']}")
                print()
        return 0

    # Validate --level is provided for escalation run
    if not args.level:
        parser.error("--level is required (or use --list-levels or --summary)")

    # Validate database exists
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        print("Run discovery first: python scripts/run_discovery.py --iterations 5 --db results/discovery.db")
        return 1

    level = EscalationLevel(args.level)
    preset = get_preset(level)

    print(f"\n{'='*60}")
    print(f"POWER ESCALATION: {level.value}")
    print(f"{'='*60}")
    print(f"\nLevel: {preset.description}")
    print(f"Cost factor: {preset.cost_factor}x baseline")
    print(f"Database: {db_path}")
    if args.seed:
        print(f"Seed override: {args.seed}")
    print()

    # Run escalation
    with Repository(db_path) as repo:
        # Check for accepted laws
        accepted = repo.get_laws_with_status("PASS")
        print(f"Total accepted laws in database: {len(accepted)}")

        # Check how many need testing at this level
        needs_testing = repo.get_laws_needing_escalation(level.value)
        print(f"Laws needing {level.value} testing: {len(needs_testing)}")

        if not needs_testing:
            print("\nNo laws need testing at this level. All accepted laws have already been escalated.")
            return 0

        print(f"\nStarting escalation...\n")

        # Create progress callback
        def on_progress(current: int, total: int, result: RetestResult) -> None:
            progress_callback(current, total, result, args.verbose)

        result = run_escalation(
            level=level,
            repo=repo,
            seed=args.seed,
            on_progress=on_progress,
        )

    # Print summary
    print(f"\n{'='*60}")
    print("ESCALATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nRun ID: {result.run_id}")
    print(f"Level: {result.level.value}")
    print(f"Runtime: {result.runtime_ms / 1000:.1f}s")
    print(f"\nResults:")
    print(f"  Laws tested: {result.laws_tested}")
    print(f"  Stable (PASS->PASS): {result.stable_count}")
    print(f"  Revoked (PASS->FAIL): {result.revoked_count}")
    print(f"  Downgraded (PASS->UNKNOWN): {result.downgraded_count}")

    # Print revoked laws if any
    if result.revoked_count > 0:
        print(f"\n{'!'*60}")
        print("REVOKED LAWS (found counterexamples at higher power):")
        print(f"{'!'*60}")
        for retest in result.retests:
            if retest.flip_type == FlipType.REVOKED:
                print(f"\n  {retest.law_id}:")
                if retest.counterexample:
                    cx = retest.counterexample
                    print(f"    Initial state: {cx.initial_state}")
                    print(f"    Failed at t={cx.t_fail} (of t_max={cx.t_max})")

    # Print downgraded laws if any
    if result.downgraded_count > 0 and args.verbose:
        print(f"\n{'?'*60}")
        print("DOWNGRADED LAWS (inconclusive at higher power):")
        print(f"{'?'*60}")
        for retest in result.retests:
            if retest.flip_type == FlipType.DOWNGRADED:
                reason = retest.new_verdict.reason_code
                print(f"  {retest.law_id}: {reason.value if reason else 'unknown reason'}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
