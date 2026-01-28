"""Probe lifecycle management: registration, execution, retirement.

The ProbeRegistry is the central authority for probe definitions. It
validates probes on registration, test-runs them on a dummy state, and
provides execution helpers for the harness and orchestration layers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Union

from src.probes.sandbox import (
    ProbeError,
    ProbeReturnTypeError,
    ProbeRuntimeError,
    ProbeTimeoutError,
    ProbeValidationError,
    detect_probe_arity,
    execute_probe,
    validate_probe_source,
)


@dataclass
class ProbeDefinition:
    """A registered probe with metadata."""
    probe_id: str
    source: str
    hypothesis: str = ""
    return_type: str = "float"
    source_hash: str = ""
    status: str = "active"       # active | retired | error
    error_message: str | None = None
    created_iteration: int | None = None
    db_id: int | None = None     # primary key from DB, if persisted
    arity: int = 1               # 1 = single-state, 2 = temporal (transition)

    def __post_init__(self):
        if not self.source_hash:
            self.source_hash = _hash_source(self.source)


def _hash_source(source: str) -> str:
    """Deterministic hash of probe source for deduplication."""
    normalized = source.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# Default dummy state for test-running probes at registration time
_DUMMY_STATE = list("..><.X..")


class ProbeRegistry:
    """Central registry for probe definitions.

    Manages the full lifecycle: validate -> test-run -> store -> execute -> retire.
    """

    def __init__(self, timeout_ms: int = 100, verbose_logger=None) -> None:
        self._probes: dict[str, ProbeDefinition] = {}
        self._timeout_ms = timeout_ms
        self._verbose_logger = verbose_logger

    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------

    def register(
        self,
        probe_id: str,
        source: str,
        hypothesis: str = "",
        return_type: str = "float",
        created_iteration: int | None = None,
    ) -> ProbeDefinition:
        """Validate, test-run, and store a probe definition.

        If the probe fails validation or test-run, it is still registered
        with status='error' and the error message recorded.

        Returns:
            The ProbeDefinition (status='active' or 'error').
        """
        source_hash = _hash_source(source)

        # Check for duplicate by source hash (return existing if active)
        for existing in self._probes.values():
            if existing.source_hash == source_hash and existing.status == "active":
                if self._verbose_logger is not None:
                    self._verbose_logger.log_probe_dedup(probe_id, existing.probe_id)
                return existing

        defn = ProbeDefinition(
            probe_id=probe_id,
            source=source,
            hypothesis=hypothesis,
            return_type=return_type,
            source_hash=source_hash,
            created_iteration=created_iteration,
        )

        # Validate AST
        valid, error = validate_probe_source(source)
        if not valid:
            defn.status = "error"
            defn.error_message = f"Validation failed: {error}"
            self._probes[probe_id] = defn
            self._log_registered(defn)
            return defn

        # Detect arity (1-param or 2-param temporal probe)
        try:
            defn.arity = detect_probe_arity(source)
        except Exception as exc:
            defn.status = "error"
            defn.error_message = f"Arity detection failed: {exc}"
            self._probes[probe_id] = defn
            self._log_registered(defn)
            return defn

        # Test-run on dummy state(s)
        try:
            if defn.arity == 2:
                execute_probe(source, _DUMMY_STATE, timeout_ms=self._timeout_ms, next_state=_DUMMY_STATE)
            else:
                execute_probe(source, _DUMMY_STATE, timeout_ms=self._timeout_ms)
        except ProbeReturnTypeError as exc:
            defn.status = "error"
            defn.error_message = f"Test-run failed: {exc}"
            self._probes[probe_id] = defn
            self._log_registered(defn)
            return defn
        except ProbeTimeoutError as exc:
            defn.status = "error"
            defn.error_message = f"Test-run failed: {exc}"
            self._probes[probe_id] = defn
            self._log_registered(defn)
            return defn
        except ProbeRuntimeError as exc:
            defn.status = "error"
            defn.error_message = f"Test-run failed: {exc}"
            self._probes[probe_id] = defn
            self._log_registered(defn)
            return defn
        except ProbeError as exc:
            defn.status = "error"
            defn.error_message = f"Test-run failed: {exc}"
            self._probes[probe_id] = defn
            self._log_registered(defn)
            return defn

        defn.status = "active"
        self._probes[probe_id] = defn
        self._log_registered(defn)
        return defn

    def _log_registered(self, defn: ProbeDefinition) -> None:
        """Emit verbose console log for a registration event."""
        if self._verbose_logger is not None:
            self._verbose_logger.log_probe_registered(
                probe_id=defn.probe_id,
                source=defn.source,
                status=defn.status,
                error_message=defn.error_message,
            )

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------

    def get(self, probe_id: str) -> ProbeDefinition | None:
        """Get a probe definition by ID."""
        return self._probes.get(probe_id)

    def list_active(self) -> list[ProbeDefinition]:
        """Return all probes with status='active'."""
        return [p for p in self._probes.values() if p.status == "active"]

    def list_all(self) -> list[ProbeDefinition]:
        """Return all probes regardless of status."""
        return list(self._probes.values())

    # -----------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------

    def execute(
        self,
        probe_id: str,
        state: list[str],
        next_state: list[str] | None = None,
    ) -> Union[float, int]:
        """Execute a single probe on the given state.

        Args:
            probe_id: ID of the probe to execute.
            state: Current universe state.
            next_state: Next universe state (required for arity-2 temporal probes).

        Raises:
            KeyError: If probe_id not found.
            ProbeError: If probe is not active or execution fails.
        """
        defn = self._probes.get(probe_id)
        if defn is None:
            raise KeyError(f"Probe '{probe_id}' not found in registry")
        if defn.status != "active":
            raise ProbeRuntimeError(
                f"Probe '{probe_id}' has status '{defn.status}': "
                f"{defn.error_message}"
            )
        state_repr = "".join(state)
        try:
            result = execute_probe(
                defn.source, state, timeout_ms=self._timeout_ms,
                next_state=next_state,
            )
        except ProbeError as exc:
            if self._verbose_logger is not None:
                self._verbose_logger.log_probe_called(
                    probe_id=probe_id,
                    state_repr=state_repr,
                    error=str(exc),
                )
            raise
        if self._verbose_logger is not None:
            self._verbose_logger.log_probe_called(
                probe_id=probe_id,
                state_repr=state_repr,
                result=result,
            )
        return result

    def execute_all_active(
        self,
        state: list[str],
        next_state: list[str] | None = None,
    ) -> dict[str, Union[float, int]]:
        """Execute all active probes on the given state.

        Args:
            state: Current universe state.
            next_state: Next universe state (required for arity-2 temporal probes;
                        arity-1 probes ignore this).

        Returns a dict mapping probe_id -> result. Probes that error are
        omitted (not raised). Temporal probes that lack next_state are skipped.
        """
        state_repr = "".join(state)
        results = {}
        for defn in self.list_active():
            try:
                if defn.arity == 2 and next_state is None:
                    continue  # skip temporal probes when no next_state available
                val = execute_probe(
                    defn.source, state, timeout_ms=self._timeout_ms,
                    next_state=next_state if defn.arity == 2 else None,
                )
                results[defn.probe_id] = val
                if self._verbose_logger is not None:
                    self._verbose_logger.log_probe_called(
                        probe_id=defn.probe_id,
                        state_repr=state_repr,
                        result=val,
                    )
            except ProbeError as exc:
                if self._verbose_logger is not None:
                    self._verbose_logger.log_probe_called(
                        probe_id=defn.probe_id,
                        state_repr=state_repr,
                        error=str(exc),
                    )
        return results

    # -----------------------------------------------------------------
    # Retirement
    # -----------------------------------------------------------------

    def retire(self, probe_id: str) -> None:
        """Mark a probe as retired (no longer active)."""
        defn = self._probes.get(probe_id)
        if defn is not None:
            defn.status = "retired"

    # -----------------------------------------------------------------
    # Prompt summary (for LLM context)
    # -----------------------------------------------------------------

    def to_prompt_summary(self) -> str:
        """Format the active probe library for inclusion in LLM prompts.

        Returns a human-readable summary of all active probes with their
        source code and hypotheses.
        """
        active = self.list_active()
        if not active:
            return "No probes defined yet. You may propose new probes."

        lines = ["ACTIVE PROBES (you may reference these by probe_id in laws):"]
        lines.append("")
        for defn in active:
            lines.append(f"  probe_id: {defn.probe_id}")
            if defn.arity == 2:
                lines.append(f"  temporal: true  (measures transitions between timesteps)")
            if defn.hypothesis:
                lines.append(f"  hypothesis: {defn.hypothesis}")
            lines.append(f"  code:")
            for src_line in defn.source.strip().splitlines():
                lines.append(f"    {src_line}")
            lines.append("")
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Persistence bridge
    # -----------------------------------------------------------------

    def load_from_db(self, repo) -> None:
        """Load all probes from the database into the registry.

        Args:
            repo: Repository instance with list_probes() method.
        """
        records = repo.list_probes()
        for rec in records:
            defn = ProbeDefinition(
                probe_id=rec.probe_id,
                source=rec.source,
                hypothesis=rec.hypothesis or "",
                return_type=rec.return_type or "float",
                source_hash=rec.source_hash,
                status=rec.status,
                error_message=rec.error_message,
                created_iteration=rec.created_iteration,
                db_id=rec.id,
                arity=rec.arity,
            )
            self._probes[defn.probe_id] = defn

    def save_to_db(self, repo) -> None:
        """Save all probes to the database (upsert logic).

        Args:
            repo: Repository instance with insert_probe() and
                  update_probe_status() methods.
        """
        for defn in self._probes.values():
            existing = repo.get_probe(defn.probe_id)
            if existing is None:
                from src.db.models import ProbeRecord
                record = ProbeRecord(
                    probe_id=defn.probe_id,
                    source=defn.source,
                    hypothesis=defn.hypothesis,
                    return_type=defn.return_type,
                    source_hash=defn.source_hash,
                    status=defn.status,
                    error_message=defn.error_message,
                    created_iteration=defn.created_iteration,
                    arity=defn.arity,
                )
                defn.db_id = repo.insert_probe(record)
            else:
                if existing.status != defn.status:
                    repo.update_probe_status(
                        defn.probe_id, defn.status, defn.error_message
                    )
