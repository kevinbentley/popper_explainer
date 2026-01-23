"""Main escalation runner for re-testing accepted laws.

Provides the run_escalation() function that orchestrates power escalation
on all accepted laws that haven't been tested at the specified level.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from src.claims.schema import CandidateLaw
from src.db.escalation_models import EscalationRunRecord, LawRetestRecord
from src.db.models import CounterexampleRecord, EvaluationRecord
from src.db.repo import Repository
from src.harness.escalation.flip import FlipType, RetestResult, classify_flip
from src.harness.escalation.levels import EscalationLevel, get_config
from src.harness.harness import Harness
from src.harness.verdict import LawVerdict
from src.universe.simulator import version_hash as sim_version_hash


@dataclass
class EscalationRunResult:
    """Complete result of an escalation run.

    Attributes:
        run_id: Database ID of the escalation run
        level: The escalation level used
        timestamp: When the run started
        laws_tested: Number of laws tested
        stable_count: Laws that remained PASS
        revoked_count: Laws that flipped to FAIL
        downgraded_count: Laws that flipped to UNKNOWN
        retests: Individual retest results
        runtime_ms: Total runtime in milliseconds
    """

    run_id: int
    level: EscalationLevel
    timestamp: datetime
    laws_tested: int
    stable_count: int
    revoked_count: int
    downgraded_count: int
    retests: list[RetestResult]
    runtime_ms: int


def run_escalation(
    level: EscalationLevel,
    repo: Repository,
    seed: int | None = None,
    law_filter: Callable[[CandidateLaw], bool] | None = None,
    law_ids: list[str] | None = None,
    on_progress: Callable[[int, int, RetestResult], None] | None = None,
) -> EscalationRunResult:
    """Run escalation testing on accepted laws.

    Re-tests laws that have PASS status but haven't been tested at the
    specified escalation level. Records all results to the database.

    Args:
        level: The escalation level to use
        repo: Database repository for reading laws and writing results
        seed: Optional seed override for reproducibility
        law_filter: Optional filter function to select subset of laws
        law_ids: Optional list of specific law IDs to test (for scoped escalation)
        on_progress: Optional callback for progress updates (current, total, result)

    Returns:
        EscalationRunResult with all retests and summary counts
    """
    start_time = time.time()
    timestamp = datetime.now()

    # Get escalated harness config
    config = get_config(level, seed=seed)
    effective_seed = config.seed

    # Get accepted laws that need testing at this level
    laws_to_test = repo.get_laws_needing_escalation(level.value)

    # Apply law_ids filter if provided (for scoped escalation)
    if law_ids is not None:
        law_ids_set = set(law_ids)
        laws_to_test = [
            (law_rec, eval_rec)
            for law_rec, eval_rec in laws_to_test
            if law_rec.law_id in law_ids_set
        ]

    # Apply optional callable filter
    if law_filter:
        filtered = []
        for law_record, eval_record in laws_to_test:
            try:
                law = CandidateLaw.model_validate_json(law_record.law_json)
                if law_filter(law):
                    filtered.append((law_record, eval_record))
            except Exception:
                # Skip laws that can't be parsed
                pass
        laws_to_test = filtered

    # Create escalation run record (with initial zero counts)
    run_record = EscalationRunRecord(
        level=level.value,
        harness_config_hash=config.content_hash(),
        sim_hash=sim_version_hash(),
        seed=effective_seed,
        laws_tested=len(laws_to_test),
        stable_count=0,
        revoked_count=0,
        downgraded_count=0,
        runtime_ms=0,
    )
    run_id = repo.insert_escalation_run(run_record)

    # Create harness with escalated config
    harness = Harness(config, repo)

    # Track results
    retests: list[RetestResult] = []
    stable_count = 0
    revoked_count = 0
    downgraded_count = 0

    # Test each law
    for i, (law_record, old_eval) in enumerate(laws_to_test):
        try:
            # Parse the law
            law = CandidateLaw.model_validate_json(law_record.law_json)

            # Evaluate with escalated harness
            new_verdict = harness.evaluate(law)

            # Classify the flip
            flip_type = classify_flip(old_eval.status, new_verdict.status)

            # Create retest result
            retest = RetestResult(
                law_id=law.law_id,
                old_status=old_eval.status,
                new_status=new_verdict.status,
                flip_type=flip_type,
                new_verdict=new_verdict,
                counterexample=new_verdict.counterexample,
            )
            retests.append(retest)

            # Update counts
            if flip_type == FlipType.STABLE:
                stable_count += 1
            elif flip_type == FlipType.REVOKED:
                revoked_count += 1
            elif flip_type == FlipType.DOWNGRADED:
                downgraded_count += 1

            # Persist the retest result
            eval_id, cx_id = _persist_retest(repo, run_id, law, new_verdict, flip_type, config)

            # Insert law retest record
            retest_record = LawRetestRecord(
                escalation_run_id=run_id,
                law_id=law.law_id,
                old_status=old_eval.status,
                new_status=new_verdict.status,
                flip_type=flip_type.value,
                evaluation_id=eval_id,
                counterexample_id=cx_id,
            )
            repo.insert_law_retest(retest_record)

            # Progress callback
            if on_progress:
                on_progress(i + 1, len(laws_to_test), retest)

        except Exception as e:
            # Log error but continue with other laws
            print(f"Error testing law {law_record.law_id}: {e}")
            continue

    # Calculate runtime
    runtime_ms = int((time.time() - start_time) * 1000)

    # Update escalation run with final counts
    repo.update_escalation_run(
        run_id,
        stable_count=stable_count,
        revoked_count=revoked_count,
        downgraded_count=downgraded_count,
        runtime_ms=runtime_ms,
    )

    return EscalationRunResult(
        run_id=run_id,
        level=level,
        timestamp=timestamp,
        laws_tested=len(laws_to_test),
        stable_count=stable_count,
        revoked_count=revoked_count,
        downgraded_count=downgraded_count,
        retests=retests,
        runtime_ms=runtime_ms,
    )


def _persist_retest(
    repo: Repository,
    run_id: int,
    law: CandidateLaw,
    verdict: LawVerdict,
    flip_type: FlipType,
    config,
) -> tuple[int, int | None]:
    """Persist the retest results to the database.

    Creates new evaluation record and counterexample if applicable.
    Also logs to audit trail.

    Args:
        repo: Database repository
        run_id: Escalation run ID
        law: The law being retested
        verdict: The new verdict
        flip_type: Classification of the flip
        config: Harness config used

    Returns:
        Tuple of (evaluation_id, counterexample_id or None)
    """
    # Create evaluation record
    eval_record = EvaluationRecord(
        law_id=law.law_id,
        law_hash=law.content_hash(),
        status=verdict.status,
        reason_code=verdict.reason_code.value if verdict.reason_code else None,
        cases_attempted=verdict.power_metrics.cases_attempted if verdict.power_metrics else 0,
        cases_used=verdict.power_metrics.cases_used if verdict.power_metrics else 0,
        power_metrics_json=json.dumps(verdict.power_metrics.to_dict()) if verdict.power_metrics else None,
        vacuity_json=json.dumps(verdict.vacuity.to_dict()) if verdict.vacuity else None,
        harness_config_hash=config.content_hash(),
        sim_hash=sim_version_hash(),
        runtime_ms=verdict.runtime_ms,
    )
    eval_id = repo.insert_evaluation(eval_record)

    # Create counterexample if FAIL
    cx_id = None
    if verdict.counterexample:
        cx = verdict.counterexample
        cx_record = CounterexampleRecord(
            evaluation_id=eval_id,
            law_id=law.law_id,
            initial_state=cx.initial_state,
            config_json=json.dumps(cx.config),
            seed=cx.seed,
            t_max=cx.t_max,
            t_fail=cx.t_fail,
            trajectory_excerpt_json=json.dumps(cx.trajectory_excerpt) if cx.trajectory_excerpt else None,
            observables_at_fail_json=json.dumps(cx.observables_at_fail) if cx.observables_at_fail else None,
            witness_json=json.dumps(cx.witness) if cx.witness else None,
            minimized=cx.minimized,
        )
        cx_id = repo.insert_counterexample(cx_record)

    # Log to audit trail
    operation = {
        FlipType.STABLE: "escalation_confirm",
        FlipType.REVOKED: "escalation_revoke",
        FlipType.DOWNGRADED: "escalation_downgrade",
    }[flip_type]

    repo.log_audit(
        operation=operation,
        entity_type="law",
        entity_id=law.law_id,
        details={
            "escalation_run_id": run_id,
            "flip_type": flip_type.value,
            "old_status": "PASS",
            "new_status": verdict.status,
            "has_counterexample": verdict.counterexample is not None,
        },
    )

    return eval_id, cx_id


def get_escalation_summary(repo: Repository) -> dict[str, dict[str, int]]:
    """Get a summary of escalation results by level.

    Args:
        repo: Database repository

    Returns:
        Dict mapping level to {stable, revoked, downgraded} counts
    """
    summary: dict[str, dict[str, int]] = {}

    for level in EscalationLevel:
        runs = repo.list_escalation_runs(level=level.value)
        if runs:
            # Sum across all runs for this level
            stable = sum(r.stable_count for r in runs)
            revoked = sum(r.revoked_count for r in runs)
            downgraded = sum(r.downgraded_count for r in runs)
            summary[level.value] = {
                "stable": stable,
                "revoked": revoked,
                "downgraded": downgraded,
                "total_runs": len(runs),
            }

    return summary
