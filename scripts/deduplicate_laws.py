#!/usr/bin/env python3
"""Analyze and deduplicate laws in the database using semantic fingerprinting.

Usage:
    python scripts/deduplicate_laws.py --db results/discovery.db [--apply]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.claims.fingerprint import compute_semantic_fingerprint, fingerprint_description
from src.claims.schema import CandidateLaw
from src.db.repo import Repository


def analyze_duplicates(db_path: Path, apply: bool = False) -> None:
    """Analyze and optionally deduplicate laws.

    Args:
        db_path: Path to database
        apply: If True, mark duplicates in the database
    """
    with Repository(db_path) as repo:
        # Get all laws with their latest status
        rows = list(repo.conn.execute("""
            SELECT l.law_id, l.law_json, e.status
            FROM laws l
            JOIN evaluations e ON l.law_id = e.law_id
            WHERE e.id = (
                SELECT id FROM evaluations
                WHERE law_id = l.law_id
                ORDER BY created_at DESC LIMIT 1
            )
            ORDER BY l.created_at
        """))

        print(f"Analyzing {len(rows)} laws...\n")

        # Group by fingerprint
        fingerprint_groups: dict[str, list[tuple[str, CandidateLaw, str]]] = defaultdict(list)

        for law_id, law_json, status in rows:
            try:
                law = CandidateLaw.model_validate_json(law_json)
                fp = compute_semantic_fingerprint(law)
                fingerprint_groups[fp].append((law_id, law, status))
            except Exception as e:
                print(f"Warning: Could not parse {law_id}: {e}")

        # Find groups with duplicates
        duplicate_groups = {fp: laws for fp, laws in fingerprint_groups.items() if len(laws) > 1}

        if not duplicate_groups:
            print("No semantic duplicates found.")
            return

        print(f"Found {len(duplicate_groups)} groups of semantic duplicates:\n")
        print("=" * 70)

        total_duplicates = 0
        duplicates_to_remove = []

        for fp, laws in sorted(duplicate_groups.items(), key=lambda x: -len(x[1])):
            print(f"\nFingerprint: {fp}")
            print(f"  Group size: {len(laws)}")

            # Sort by status (PASS first) and creation time
            status_order = {"PASS": 0, "FAIL": 1, "UNKNOWN": 2}
            laws_sorted = sorted(laws, key=lambda x: (status_order.get(x[2], 3), x[0]))

            # First one is the canonical (keep this one)
            canonical_id, canonical_law, canonical_status = laws_sorted[0]
            print(f"\n  CANONICAL (keep): {canonical_id} [{canonical_status}]")

            if canonical_law.claim_ast:
                from src.claims.ast_schema import ast_to_string
                claim_str = ast_to_string(canonical_law.claim_ast)
            else:
                claim_str = canonical_law.claim
            print(f"    Claim: {claim_str[:60]}{'...' if len(claim_str) > 60 else ''}")

            # Rest are duplicates
            for dup_id, dup_law, dup_status in laws_sorted[1:]:
                print(f"  DUPLICATE: {dup_id} [{dup_status}]")
                if dup_law.claim_ast:
                    dup_claim = ast_to_string(dup_law.claim_ast)
                else:
                    dup_claim = dup_law.claim
                print(f"    Claim: {dup_claim[:60]}{'...' if len(dup_claim) > 60 else ''}")
                duplicates_to_remove.append((dup_id, canonical_id))
                total_duplicates += 1

        print("\n" + "=" * 70)
        print(f"\nSummary:")
        print(f"  Total laws: {len(rows)}")
        print(f"  Unique (by fingerprint): {len(fingerprint_groups)}")
        print(f"  Duplicate groups: {len(duplicate_groups)}")
        print(f"  Duplicates to remove: {total_duplicates}")

        if apply and duplicates_to_remove:
            print(f"\nApplying deduplication...")
            for dup_id, canonical_id in duplicates_to_remove:
                # Log the deduplication
                repo.log_audit(
                    operation="deduplicate",
                    entity_type="law",
                    entity_id=dup_id,
                    details={"canonical_id": canonical_id, "reason": "semantic_fingerprint_match"},
                )
                # Mark the duplicate in the database (add a flag, don't delete)
                repo.conn.execute(
                    "UPDATE laws SET notes = ? WHERE law_id = ?",
                    (f"DUPLICATE_OF:{canonical_id}", dup_id)
                )
            repo.conn.commit()
            print(f"  Marked {len(duplicates_to_remove)} laws as duplicates.")
        elif duplicates_to_remove:
            print(f"\nRun with --apply to mark duplicates in the database.")


def main():
    parser = argparse.ArgumentParser(description="Analyze and deduplicate laws")
    parser.add_argument(
        "--db", "-d",
        type=str,
        default="results/discovery.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually mark duplicates in the database",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    analyze_duplicates(db_path, apply=args.apply)


if __name__ == "__main__":
    main()
