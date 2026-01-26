#!/usr/bin/env python3
"""Dump the state of a run to a JSON file.

Exports laws, theorems, explanations, and counterexamples from a database
to a structured JSON file for inspection, sharing, or analysis.

Usage:
    # Dump all state
    python scripts/dump_run_state.py --db popper.db --output run_state.json

    # Dump only passed laws
    python scripts/dump_run_state.py --db popper.db --status PASS --output passed_laws.json

    # Dump with full counterexample details
    python scripts/dump_run_state.py --db popper.db --include-trajectories --output full_dump.json

    # Dump a specific run's data (if using orchestration)
    python scripts/dump_run_state.py --db popper.db --run-id orch_abc123 --output run_abc123.json

    # Pretty print to stdout
    python scripts/dump_run_state.py --db popper.db --pretty
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.repo import Repository


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Dump run state (laws, theorems, explanations) to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db",
        type=str,
        default="popper.db",
        help="Database path (default: popper.db)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file (default: stdout)",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Filter to specific orchestration run ID",
    )

    parser.add_argument(
        "--status",
        type=str,
        choices=["PASS", "FAIL", "UNKNOWN", "all"],
        default="all",
        help="Filter laws by evaluation status (default: all)",
    )

    parser.add_argument(
        "--include-trajectories",
        action="store_true",
        help="Include full trajectory excerpts in counterexamples",
    )

    parser.add_argument(
        "--include-witnesses",
        action="store_true",
        help="Include structured witnesses for failed laws",
    )

    parser.add_argument(
        "--include-transcripts",
        action="store_true",
        help="Include LLM transcripts (can be very large)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum items per category (default: 1000)",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only output summary statistics, not full data",
    )

    return parser


def dump_laws(repo: Repository, status: str, limit: int, include_witnesses: bool) -> dict:
    """Dump laws with their evaluations and counterexamples."""
    result = {
        "passed": [],
        "failed": [],
        "unknown": [],
    }

    statuses_to_fetch = ["PASS", "FAIL", "UNKNOWN"] if status == "all" else [status]

    for s in statuses_to_fetch:
        laws_with_evals = repo.get_laws_with_status(s, limit=limit)

        for law_record, eval_record in laws_with_evals:
            # Parse law JSON
            try:
                law_data = json.loads(law_record.law_json)
            except json.JSONDecodeError:
                law_data = {"raw": law_record.law_json}

            entry = {
                "law_id": law_record.law_id,
                "law_hash": law_record.law_hash,
                "template": law_record.template,
                "law": law_data,
                "evaluation": {
                    "status": eval_record.status,
                    "reason_code": eval_record.reason_code,
                    "cases_attempted": eval_record.cases_attempted,
                    "cases_used": eval_record.cases_used,
                    "runtime_ms": eval_record.runtime_ms,
                    "harness_config_hash": eval_record.harness_config_hash,
                    "sim_hash": eval_record.sim_hash,
                },
                "created_at": str(law_record.created_at) if law_record.created_at else None,
            }

            # Add power metrics if available
            if eval_record.power_metrics_json:
                try:
                    entry["evaluation"]["power_metrics"] = json.loads(eval_record.power_metrics_json)
                except json.JSONDecodeError:
                    pass

            # Add counterexamples for failed laws
            if s == "FAIL":
                counterexamples = repo.get_counterexamples_for_law(law_record.law_id)
                if counterexamples:
                    entry["counterexamples"] = []
                    for cx in counterexamples[:5]:  # Limit counterexamples per law
                        cx_entry = {
                            "initial_state": cx.initial_state,
                            "t_fail": cx.t_fail,
                            "t_max": cx.t_max,
                            "minimized": cx.minimized,
                        }
                        if cx.trajectory_excerpt_json:
                            try:
                                cx_entry["trajectory_excerpt"] = json.loads(cx.trajectory_excerpt_json)
                            except json.JSONDecodeError:
                                pass
                        if cx.observables_at_fail_json:
                            try:
                                cx_entry["observables_at_fail"] = json.loads(cx.observables_at_fail_json)
                            except json.JSONDecodeError:
                                pass
                        entry["counterexamples"].append(cx_entry)

                # Add witnesses if requested
                if include_witnesses:
                    witnesses = repo.get_witnesses_for_law(law_record.law_id, limit=5)
                    if witnesses:
                        entry["witnesses"] = []
                        for w in witnesses:
                            entry["witnesses"].append({
                                "t_fail": w.t_fail,
                                "state_at_t": w.state_at_t,
                                "state_at_t1": w.state_at_t1,
                                "formatted_witness": w.formatted_witness,
                                "neighborhood_hash": w.neighborhood_hash,
                                "is_primary": w.is_primary,
                            })

            # Categorize by status
            if s == "PASS":
                result["passed"].append(entry)
            elif s == "FAIL":
                result["failed"].append(entry)
            else:
                result["unknown"].append(entry)

    return result


def dump_theorems(repo: Repository, run_id: str | None, limit: int) -> list[dict]:
    """Dump theorems from theorem runs."""
    theorems = []

    # Get theorem runs
    theorem_runs = repo.list_theorem_runs(limit=limit)

    for run in theorem_runs:
        run_theorems = repo.get_theorems_by_run(run.id)

        for t in run_theorems:
            entry = {
                "theorem_id": t.theorem_id,
                "name": t.name,
                "status": t.status,
                "claim": t.claim,
                "theorem_run_id": run.run_id,
                "created_at": str(t.created_at) if t.created_at else None,
            }

            # Parse support JSON
            if t.support_json:
                try:
                    entry["support"] = json.loads(t.support_json)
                except json.JSONDecodeError:
                    pass

            # Parse failure modes
            if t.failure_modes_json:
                try:
                    entry["failure_modes"] = json.loads(t.failure_modes_json)
                except json.JSONDecodeError:
                    pass

            # Parse bucket tags
            if t.bucket_tags_json:
                try:
                    entry["bucket_tags"] = json.loads(t.bucket_tags_json)
                except json.JSONDecodeError:
                    pass

            theorems.append(entry)

    return theorems


def dump_explanations(repo: Repository, run_id: str | None, limit: int) -> list[dict]:
    """Dump explanations."""
    explanations = []

    # If run_id is specified, get explanations for that run
    if run_id:
        exp_records = repo.list_explanations_for_run(run_id, limit=limit)
    else:
        # Get all explanations (need to query directly since no list_all method)
        cursor = repo.conn.cursor()
        cursor.execute(
            "SELECT * FROM explanations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        exp_records = [repo._row_to_explanation(row) for row in rows] if rows else []

    for exp in exp_records:
        entry = {
            "explanation_id": exp.explanation_id,
            "run_id": exp.run_id,
            "iteration": exp.iteration,
            "explanation_type": exp.explanation_type,
            "summary": exp.summary,
            "status": exp.status,
            "confidence": exp.confidence,
            "created_at": str(exp.created_at) if exp.created_at else None,
        }

        # Parse JSON fields
        if exp.axioms_json:
            try:
                entry["axioms"] = json.loads(exp.axioms_json)
            except json.JSONDecodeError:
                pass

        if exp.derivations_json:
            try:
                entry["derivations"] = json.loads(exp.derivations_json)
            except json.JSONDecodeError:
                pass

        if exp.predictions_json:
            try:
                entry["predictions"] = json.loads(exp.predictions_json)
            except json.JSONDecodeError:
                pass

        if exp.supporting_laws_json:
            try:
                entry["supporting_laws"] = json.loads(exp.supporting_laws_json)
            except json.JSONDecodeError:
                pass

        explanations.append(entry)

    return explanations


def dump_run_metadata(repo: Repository, run_id: str | None) -> dict | None:
    """Dump orchestration run metadata if available."""
    if not run_id:
        return None

    run = repo.get_orchestration_run(run_id)
    if not run:
        return None

    entry = {
        "run_id": run.run_id,
        "status": run.status,
        "current_phase": run.current_phase,
        "current_iteration": run.current_iteration,
        "started_at": str(run.started_at) if run.started_at else None,
        "completed_at": str(run.completed_at) if run.completed_at else None,
    }

    if run.config_json:
        try:
            entry["config"] = json.loads(run.config_json)
        except json.JSONDecodeError:
            pass

    if run.final_report_json:
        try:
            entry["final_report"] = json.loads(run.final_report_json)
        except json.JSONDecodeError:
            pass

    return entry


def compute_summary(laws: dict, theorems: list, explanations: list) -> dict:
    """Compute summary statistics."""
    return {
        "laws": {
            "total": len(laws["passed"]) + len(laws["failed"]) + len(laws["unknown"]),
            "passed": len(laws["passed"]),
            "failed": len(laws["failed"]),
            "unknown": len(laws["unknown"]),
        },
        "theorems": {
            "total": len(theorems),
            "by_status": {},
        },
        "explanations": {
            "total": len(explanations),
            "by_type": {},
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Check database exists
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    # Open repository
    repo = Repository(str(db_path))
    repo.connect()

    # Collect data
    print(f"Reading from {args.db}...", file=sys.stderr)

    laws = dump_laws(
        repo,
        status=args.status,
        limit=args.limit,
        include_witnesses=args.include_witnesses,
    )

    theorems = dump_theorems(repo, run_id=args.run_id, limit=args.limit)
    explanations = dump_explanations(repo, run_id=args.run_id, limit=args.limit)
    run_metadata = dump_run_metadata(repo, run_id=args.run_id)

    # Compute summary
    summary = compute_summary(laws, theorems, explanations)

    # Build output
    if args.summary_only:
        output = {
            "summary": summary,
            "run_metadata": run_metadata,
        }
    else:
        output = {
            "summary": summary,
            "run_metadata": run_metadata,
            "laws": laws,
            "theorems": theorems,
            "explanations": explanations,
        }

    # Add transcripts if requested
    if args.include_transcripts and not args.summary_only:
        cursor = repo.conn.cursor()
        cursor.execute(
            "SELECT * FROM llm_transcripts ORDER BY created_at DESC LIMIT ?",
            (args.limit,),
        )
        rows = cursor.fetchall()
        if rows:
            output["llm_transcripts"] = []
            for row in rows:
                output["llm_transcripts"].append({
                    "id": row["id"],
                    "component": row["component"],
                    "model_name": row["model_name"],
                    "prompt": row["prompt"][:500] + "..." if len(row["prompt"]) > 500 else row["prompt"],
                    "response_preview": row["raw_response"][:500] + "..." if len(row["raw_response"]) > 500 else row["raw_response"],
                    "success": bool(row["success"]),
                    "total_tokens": row["total_tokens"],
                    "duration_ms": row["duration_ms"],
                    "created_at": row["created_at"],
                })

    # Output
    indent = 2 if args.pretty else None

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=indent, default=str)
        print(f"Wrote {args.output}", file=sys.stderr)
        print(f"  Laws: {summary['laws']['total']} (PASS: {summary['laws']['passed']}, FAIL: {summary['laws']['failed']}, UNKNOWN: {summary['laws']['unknown']})", file=sys.stderr)
        print(f"  Theorems: {summary['theorems']['total']}", file=sys.stderr)
        print(f"  Explanations: {summary['explanations']['total']}", file=sys.stderr)
    else:
        print(json.dumps(output, indent=indent, default=str))

    repo.close()


if __name__ == "__main__":
    main()
