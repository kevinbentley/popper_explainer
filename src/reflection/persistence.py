"""Persistence helpers for the Reflection Engine.

Translates between domain models (src/reflection/models.py) and
database records (src/db/models.py), handling JSON serialization
and symbol scrambling.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.db.models import (
    ReflectionSessionRecord,
    SevereTestCommandRecord,
    StandardModelRecord,
)
from src.reflection.models import (
    AuditorResult,
    DerivedObservable,
    HiddenVariable,
    ReflectionResult,
    SevereTestCommand,
    StandardModel,
    TheoristResult,
)

if TYPE_CHECKING:
    from src.db.repo import Repository


def save_reflection_result(
    repo: Repository,
    run_id: str,
    iteration_index: int,
    result: ReflectionResult,
    trigger_reason: str = "periodic",
    prompt_hash: str | None = None,
) -> tuple[int, int]:
    """Persist a full reflection result to the database.

    Args:
        repo: Database repository
        run_id: Orchestration run ID
        iteration_index: Current iteration index
        result: ReflectionResult from the engine
        trigger_reason: Why reflection was triggered
        prompt_hash: Hash of the prompt(s) used

    Returns:
        Tuple of (reflection_session_id, standard_model_id)
    """
    # 1. Persist the standard model
    next_version = repo.get_next_standard_model_version(run_id)
    sm_record = StandardModelRecord(
        run_id=run_id,
        iteration_id=iteration_index,
        version=next_version,
        fixed_laws_json=json.dumps(result.standard_model.fixed_laws),
        archived_laws_json=json.dumps(result.standard_model.archived_laws),
        derived_observables_json=json.dumps([
            d.to_dict() if hasattr(d, "to_dict") else {
                "name": d.name,
                "expression": d.expression,
                "rationale": d.rationale,
                "source_laws": d.source_laws,
            }
            for d in result.standard_model.derived_observables
        ]),
        causal_narrative=result.standard_model.causal_narrative,
        hidden_variables_json=json.dumps([
            {
                "name": h.name,
                "description": h.description,
                "evidence": h.evidence,
                "testable_prediction": h.testable_prediction,
            }
            for h in result.standard_model.hidden_variables
        ]),
        k_decomposition=result.standard_model.k_decomposition,
        confidence=result.standard_model.confidence,
    )
    sm_id = repo.insert_standard_model(sm_record)

    # 2. Persist the reflection session
    session_record = ReflectionSessionRecord(
        run_id=run_id,
        iteration_index=iteration_index,
        trigger_reason=trigger_reason,
        auditor_result_json=json.dumps(result.auditor_result.to_dict()),
        theorist_result_json=json.dumps(result.theorist_result.to_dict()),
        severe_test_json=json.dumps([
            c.to_dict() for c in result.severe_test_commands
        ]),
        conflicts_found=len(result.auditor_result.conflicts),
        laws_archived=len(result.auditor_result.archives),
        hidden_variables_postulated=len(result.theorist_result.hidden_variables),
        standard_model_version=next_version,
        prompt_hash=prompt_hash,
        runtime_ms=result.runtime_ms,
    )
    session_id = repo.insert_reflection_session(session_record)

    # 3. Persist severe test commands
    for cmd in result.severe_test_commands:
        cmd_record = SevereTestCommandRecord(
            run_id=run_id,
            reflection_session_id=session_id,
            command_type=cmd.command_type,
            target_law_id=cmd.target_law_id,
            description=cmd.description,
            initial_conditions_json=cmd.initial_conditions_json,
            grid_lengths_json=cmd.grid_lengths_json,
            priority=cmd.priority,
        )
        repo.insert_severe_test_command(cmd_record)

    return session_id, sm_id


def load_latest_standard_model(
    repo: Repository,
    run_id: str,
) -> StandardModel | None:
    """Load the latest standard model from the database.

    Args:
        repo: Database repository
        run_id: Orchestration run ID

    Returns:
        StandardModel domain object, or None if no model exists
    """
    record = repo.get_latest_standard_model(run_id)
    if record is None:
        return None

    return _record_to_standard_model(record)


def _record_to_standard_model(record: StandardModelRecord) -> StandardModel:
    """Convert a database record to a domain StandardModel."""
    derived_observables = []
    if record.derived_observables_json:
        for d in json.loads(record.derived_observables_json):
            derived_observables.append(DerivedObservable(
                name=d["name"],
                expression=d["expression"],
                rationale=d["rationale"],
                source_laws=d.get("source_laws", []),
            ))

    hidden_variables = []
    if record.hidden_variables_json:
        for h in json.loads(record.hidden_variables_json):
            hidden_variables.append(HiddenVariable(
                name=h["name"],
                description=h["description"],
                evidence=h["evidence"],
                testable_prediction=h["testable_prediction"],
            ))

    return StandardModel(
        fixed_laws=json.loads(record.fixed_laws_json),
        archived_laws=json.loads(record.archived_laws_json),
        derived_observables=derived_observables,
        causal_narrative=record.causal_narrative or "",
        hidden_variables=hidden_variables,
        k_decomposition=record.k_decomposition or "",
        confidence=record.confidence or 0.5,
        version=record.version,
    )


def build_standard_model_summary(model: StandardModel) -> dict[str, Any]:
    """Build a compact summary of the standard model for memory snapshots.

    Args:
        model: StandardModel domain object

    Returns:
        Dictionary suitable for inclusion in DiscoveryMemorySnapshot
    """
    return {
        "version": model.version,
        "fixed_law_count": len(model.fixed_laws),
        "archived_law_count": len(model.archived_laws),
        "archived_law_ids": model.archived_laws,
        "derived_observables": [
            {"name": d.name, "expression": d.expression}
            for d in model.derived_observables
        ],
        "hidden_variables": [
            {"name": h.name, "testable_prediction": h.testable_prediction}
            for h in model.hidden_variables
        ],
        "causal_narrative_excerpt": (
            model.causal_narrative[:500] if model.causal_narrative else ""
        ),
        "confidence": model.confidence,
    }
