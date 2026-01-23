#!/usr/bin/env python3
"""Re-evaluate suspected false positive laws with pathological cases.

This script targets laws that may have passed due to generator coverage gaps,
and re-tests them with pathological cases (uniform grids, etc).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.repo import Repository
from src.harness.config import HarnessConfig
from src.harness.harness import Harness
from src.claims.schema import CandidateLaw


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate suspected false positive laws")
    parser.add_argument(
        "--db", "-d",
        type=str,
        default="results/discovery.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--law-id",
        type=str,
        help="Specific law ID to re-evaluate (default: all suspected false positives)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without updating database",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    # Create harness with pathological-heavy config
    config = HarnessConfig(
        seed=42,
        max_cases=200,
        min_cases_used_for_pass=30,
        default_T=50,
        max_T=100,
        enable_adversarial_search=True,
        adversarial_budget=500,
        # Heavy emphasis on pathological cases
        generator_weights={
            "pathological_cases": 0.5,  # 50% pathological
            "random_density_sweep": 0.25,
            "constrained_pair_interactions": 0.15,
            "edge_wrapping_cases": 0.10,
        },
    )
    harness = Harness(config)

    # Known suspected false positives
    suspected_laws = [
        "full_grid_implies_collision",  # Fails on uniform grids like ">>>>>"
    ]

    repo = Repository(db_path)
    repo.connect()

    try:
        if args.law_id:
            law_ids = [args.law_id]
        else:
            law_ids = suspected_laws

        for law_id in law_ids:
            print(f"\n{'='*60}")
            print(f"Re-evaluating: {law_id}")
            print(f"{'='*60}")

            # Load law from database
            law_record = repo.get_law(law_id)
            if not law_record:
                print(f"  Law not found in database")
                continue

            # Reconstruct CandidateLaw
            import json
            law_json = json.loads(law_record.law_json)
            try:
                law = CandidateLaw.model_validate(law_json)
            except Exception as e:
                print(f"  Error reconstructing law: {e}")
                continue

            # Re-evaluate
            try:
                verdict = harness.evaluate(law)
            except Exception as e:
                print(f"  Evaluation error: {e}")
                continue

            print(f"  Status: {verdict.status}")
            if verdict.counterexample:
                cx = verdict.counterexample
                print(f"  Counterexample found!")
                print(f"    Initial state: {repr(cx.initial_state)}")
                print(f"    T_fail: {cx.t_fail}")
                if cx.trajectory_excerpt:
                    print(f"    Trajectory:")
                    for i, state in enumerate(cx.trajectory_excerpt[:5]):
                        print(f"      t={i}: {repr(state)}")

            if verdict.status == "FAIL" and not args.dry_run:
                print(f"\n  Updating database to mark as FALSIFIED...")
                # Insert new evaluation record
                from src.db.models import EvaluationRecord, CounterexampleRecord
                eval_record = EvaluationRecord(
                    law_id=law_id,
                    law_hash=law_record.law_hash,
                    status="FAIL",
                    harness_config_hash=config.content_hash(),
                    sim_hash="",
                    cases_attempted=verdict.power_metrics.cases_attempted if verdict.power_metrics else 0,
                    cases_used=verdict.power_metrics.cases_used if verdict.power_metrics else 0,
                    reason_code="counterexample_found",
                    power_metrics_json=json.dumps(verdict.power_metrics.to_dict()) if verdict.power_metrics else None,
                )
                eval_id = repo.insert_evaluation(eval_record)

                # Insert counterexample
                if verdict.counterexample:
                    cx = verdict.counterexample
                    cx_record = CounterexampleRecord(
                        evaluation_id=eval_id,
                        law_id=law_id,
                        initial_state=cx.initial_state,
                        config_json=json.dumps({"grid_length": len(cx.initial_state), "boundary": "periodic"}),
                        t_max=len(cx.trajectory_excerpt) - 1 if cx.trajectory_excerpt else cx.t_fail,
                        t_fail=cx.t_fail,
                        trajectory_excerpt_json=json.dumps(cx.trajectory_excerpt) if cx.trajectory_excerpt else None,
                    )
                    repo.insert_counterexample(cx_record)

                print(f"  Database updated. Law is now FALSIFIED.")

    finally:
        repo.close()


if __name__ == "__main__":
    main()
