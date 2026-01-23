#!/usr/bin/env python3
"""Export discoveries to CSV format.

Usage:
    python scripts/export_discoveries.py [--db PATH] [--output PATH]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.claims.ast_schema import ast_to_string
from src.claims.fingerprint import compute_semantic_fingerprint
from src.claims.schema import CandidateLaw
from src.db.repo import Repository
from src.harness.escalation import EscalationLevel


def get_promotion_status(repo: Repository, law_id: str) -> str:
    """Determine the promotion status of a law.

    Returns one of:
    - 'promoted': Passed escalation_1 (fully accepted)
    - 'provisional': Passed baseline only
    - 'revoked': Failed during escalation
    - 'failed': Failed at baseline
    - 'unknown': Unknown status
    """
    # Get latest evaluation
    evals = list(repo.conn.execute(
        "SELECT status FROM evaluations WHERE law_id = ? ORDER BY created_at DESC LIMIT 1",
        (law_id,)
    ))

    if not evals:
        return "unknown"

    latest_status = evals[0][0]

    if latest_status == "FAIL":
        # Check if it was revoked during escalation
        retests = list(repo.conn.execute(
            "SELECT flip_type FROM law_retests WHERE law_id = ? AND flip_type = 'revoked'",
            (law_id,)
        ))
        if retests:
            return "revoked"
        return "failed"

    if latest_status == "UNKNOWN":
        return "unknown"

    # Status is PASS - check if promoted
    retests = list(repo.conn.execute(
        """SELECT lr.flip_type, er.level
           FROM law_retests lr
           JOIN escalation_runs er ON lr.escalation_run_id = er.id
           WHERE lr.law_id = ? AND lr.flip_type = 'stable'
           AND er.level IN ('escalation_1', 'escalation_2', 'escalation_3')""",
        (law_id,)
    ))

    if retests:
        return "promoted"

    return "provisional"


def export_discoveries(db_path: Path, output_path: Path) -> None:
    """Export discoveries to CSV."""

    with Repository(db_path) as repo:
        # Get all laws with their latest evaluation
        rows = list(repo.conn.execute("""
            SELECT DISTINCT l.law_id, l.template, l.law_json, e.status
            FROM laws l
            JOIN evaluations e ON l.law_id = e.law_id
            WHERE e.id = (
                SELECT id FROM evaluations
                WHERE law_id = l.law_id
                ORDER BY created_at DESC LIMIT 1
            )
            ORDER BY l.created_at
        """))

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['law_id', 'template', 'claim', 'status', 'promotion_status', 'fingerprint'])

            for law_id, template, law_json, status in rows:
                # Parse law to get claim and fingerprint
                try:
                    law = CandidateLaw.model_validate_json(law_json)
                    if law.claim_ast:
                        claim = ast_to_string(law.claim_ast)
                    else:
                        claim = law.claim
                    fingerprint = compute_semantic_fingerprint(law)
                except Exception:
                    claim = "(parse error)"
                    fingerprint = ""

                # Get promotion status
                promotion_status = get_promotion_status(repo, law_id)

                writer.writerow([law_id, template, claim, status, promotion_status, fingerprint])

        print(f"Exported {len(rows)} laws to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export discoveries to CSV")
    parser.add_argument(
        "--db", "-d",
        type=str,
        default="results/discovery.db",
        help="Path to SQLite database (default: results/discovery.db)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/discoveries.csv",
        help="Output CSV path (default: results/discoveries.csv)",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_discoveries(db_path, output_path)


if __name__ == "__main__":
    main()
