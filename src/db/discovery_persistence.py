"""Persistence layer for law discovery sessions.

Provides easy save/load of accepted and falsified laws to resume
discovery from where it left off.
"""

import json
from pathlib import Path
from typing import Any

from src.claims.schema import CandidateLaw
from src.db.models import CounterexampleRecord, EvaluationRecord, LawRecord
from src.db.repo import Repository
from src.harness.verdict import Counterexample, LawVerdict
from src.universe.simulator import version_hash as sim_version_hash


class DiscoveryPersistence:
    """Manages persistence of discovery session state.

    Usage:
        persistence = DiscoveryPersistence("results/discovery.db")
        persistence.connect()

        # Load existing state
        accepted, falsified = persistence.load_existing_laws()

        # Save new results
        persistence.save_accepted(law, verdict)
        persistence.save_falsified(law, verdict)

        persistence.close()
    """

    def __init__(self, db_path: str | Path = "results/discovery.db", enable_wal: bool = True):
        """Initialize persistence layer.

        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable WAL mode for better concurrent access
        """
        self.db_path = Path(db_path)
        self._repo: Repository | None = None
        self._enable_wal = enable_wal

    def connect(self) -> None:
        """Open database connection."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._repo = Repository(self.db_path)
        self._repo.connect(enable_wal=self._enable_wal)

    def close(self) -> None:
        """Close database connection."""
        if self._repo:
            self._repo.close()
            self._repo = None

    def __enter__(self) -> "DiscoveryPersistence":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @property
    def repo(self) -> Repository:
        if self._repo is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._repo

    def load_existing_laws(self) -> tuple[list[CandidateLaw], list[tuple[CandidateLaw, Counterexample | None]]]:
        """Load existing accepted and falsified laws from the database.

        Returns:
            Tuple of (accepted_laws, falsified_laws_with_counterexamples)
        """
        accepted_laws = []
        falsified_laws = []

        # Load accepted laws (PASS status)
        for law_record, _ in self.repo.get_laws_with_status("PASS"):
            try:
                law = self._law_from_record(law_record)
                accepted_laws.append(law)
            except Exception as e:
                print(f"Warning: Could not load accepted law {law_record.law_id}: {e}")

        # Load falsified laws (FAIL status) with counterexamples
        for law_record, eval_record in self.repo.get_laws_with_status("FAIL"):
            try:
                law = self._law_from_record(law_record)
                # Get the counterexample if available
                cx_records = self.repo.get_counterexamples_for_law(law_record.law_id)
                cx = self._counterexample_from_record(cx_records[0]) if cx_records else None
                falsified_laws.append((law, cx))
            except Exception as e:
                print(f"Warning: Could not load falsified law {law_record.law_id}: {e}")

        return accepted_laws, falsified_laws

    def save_accepted(self, law: CandidateLaw, verdict: LawVerdict, harness_config_hash: str = "") -> None:
        """Save an accepted law to the database.

        Args:
            law: The accepted law
            verdict: The evaluation verdict
            harness_config_hash: Hash of harness configuration
        """
        # Save the law
        law_record = self._law_to_record(law)
        self.repo.upsert_law(law_record)

        # Save the evaluation
        eval_record = EvaluationRecord(
            law_id=law.law_id,
            law_hash=law.content_hash(),
            status="PASS",
            reason_code=verdict.reason_code.value if verdict.reason_code else None,
            cases_attempted=verdict.power_metrics.cases_attempted if verdict.power_metrics else 0,
            cases_used=verdict.power_metrics.cases_used if verdict.power_metrics else 0,
            power_metrics_json=json.dumps(verdict.power_metrics.to_dict()) if verdict.power_metrics else None,
            harness_config_hash=harness_config_hash,
            sim_hash=sim_version_hash(),
        )
        self.repo.insert_evaluation(eval_record)

        # Log the operation
        self.repo.log_audit(
            operation="accept_law",
            entity_type="law",
            entity_id=law.law_id,
            details={"template": law.template.value},
        )

    def save_falsified(self, law: CandidateLaw, verdict: LawVerdict, harness_config_hash: str = "") -> None:
        """Save a falsified law with its counterexample to the database.

        Args:
            law: The falsified law
            verdict: The evaluation verdict (must have counterexample)
            harness_config_hash: Hash of harness configuration
        """
        # Save the law
        law_record = self._law_to_record(law)
        self.repo.upsert_law(law_record)

        # Save the evaluation
        eval_record = EvaluationRecord(
            law_id=law.law_id,
            law_hash=law.content_hash(),
            status="FAIL",
            reason_code=verdict.reason_code.value if verdict.reason_code else None,
            cases_attempted=verdict.power_metrics.cases_attempted if verdict.power_metrics else 0,
            cases_used=verdict.power_metrics.cases_used if verdict.power_metrics else 0,
            power_metrics_json=json.dumps(verdict.power_metrics.to_dict()) if verdict.power_metrics else None,
            harness_config_hash=harness_config_hash,
            sim_hash=sim_version_hash(),
        )
        eval_id = self.repo.insert_evaluation(eval_record)

        # Save the counterexample if present
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
            self.repo.insert_counterexample(cx_record)

        # Log the operation
        self.repo.log_audit(
            operation="falsify_law",
            entity_type="law",
            entity_id=law.law_id,
            details={
                "template": law.template.value,
                "has_counterexample": verdict.counterexample is not None,
            },
        )

    def law_exists(self, law: CandidateLaw) -> bool:
        """Check if a law with the same content hash already exists."""
        return self.repo.law_exists(law.content_hash())

    def get_summary(self) -> dict[str, int]:
        """Get summary counts of laws by status."""
        return self.repo.get_evaluation_summary()

    def _law_to_record(self, law: CandidateLaw) -> LawRecord:
        """Convert a CandidateLaw to a LawRecord."""
        return LawRecord(
            law_id=law.law_id,
            law_hash=law.content_hash(),
            template=law.template.value,
            law_json=law.model_dump_json(),
        )

    def _law_from_record(self, record: LawRecord) -> CandidateLaw:
        """Convert a LawRecord to a CandidateLaw."""
        return CandidateLaw.model_validate_json(record.law_json)

    def _counterexample_from_record(self, record: CounterexampleRecord) -> Counterexample:
        """Convert a CounterexampleRecord to a Counterexample."""
        return Counterexample(
            initial_state=record.initial_state,
            config=json.loads(record.config_json) if record.config_json else {},
            seed=record.seed,
            t_max=record.t_max,
            t_fail=record.t_fail,
            trajectory_excerpt=json.loads(record.trajectory_excerpt_json) if record.trajectory_excerpt_json else None,
            observables_at_fail=json.loads(record.observables_at_fail_json) if record.observables_at_fail_json else None,
            witness=json.loads(record.witness_json) if record.witness_json else None,
            minimized=record.minimized,
        )
