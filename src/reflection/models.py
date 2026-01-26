"""Data models for the Reflection Engine.

These are the in-memory domain models used during reflection processing.
For database records, see src/db/models.py (StandardModelRecord, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConflictEntry:
    """A detected conflict between a fixed law and graveyard evidence."""

    law_id: str
    conflicting_law_id: str | None = None
    counterexample_law_id: str | None = None
    description: str = ""
    severity: str = "low"  # 'low', 'medium', 'high'


@dataclass
class ArchiveDecision:
    """Decision to archive (demote) a law from the active set."""

    law_id: str
    reason: str  # 'tautology', 'redundant', 'subsumed', 'conflict'
    subsumed_by: str | None = None  # law_id that subsumes this one


@dataclass
class AuditorResult:
    """Output of the Auditor task.

    The auditor checks fixed laws against the graveyard,
    identifies tautologies and redundancies, and flags conflicts.
    """

    conflicts: list[ConflictEntry] = field(default_factory=list)
    archives: list[ArchiveDecision] = field(default_factory=list)
    deductive_issues: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflicts": [
                {
                    "law_id": c.law_id,
                    "conflicting_law_id": c.conflicting_law_id,
                    "counterexample_law_id": c.counterexample_law_id,
                    "description": c.description,
                    "severity": c.severity,
                }
                for c in self.conflicts
            ],
            "archives": [
                {
                    "law_id": a.law_id,
                    "reason": a.reason,
                    "subsumed_by": a.subsumed_by,
                }
                for a in self.archives
            ],
            "deductive_issues": self.deductive_issues,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditorResult:
        return cls(
            conflicts=[
                ConflictEntry(**c) for c in data.get("conflicts", [])
            ],
            archives=[
                ArchiveDecision(**a) for a in data.get("archives", [])
            ],
            deductive_issues=data.get("deductive_issues", []),
            summary=data.get("summary", ""),
        )


@dataclass
class DerivedObservable:
    """A derived observable suggested by the theorist."""

    name: str
    expression: str
    rationale: str
    source_laws: list[str] = field(default_factory=list)  # law_ids


@dataclass
class HiddenVariable:
    """A postulated hidden variable."""

    name: str
    description: str
    evidence: str  # What anomalies suggest this
    testable_prediction: str  # How to detect it


@dataclass
class TheoristResult:
    """Output of the Theorist task.

    The theorist synthesizes a causal narrative, derives new observables,
    and postulates hidden variables based on anomalies.
    """

    derived_observables: list[DerivedObservable] = field(default_factory=list)
    hidden_variables: list[HiddenVariable] = field(default_factory=list)
    causal_narrative: str = ""
    k_decomposition: str = ""  # Knowledge decomposition summary
    confidence: float = 0.5
    severe_test_suggestions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "derived_observables": [
                {
                    "name": d.name,
                    "expression": d.expression,
                    "rationale": d.rationale,
                    "source_laws": d.source_laws,
                }
                for d in self.derived_observables
            ],
            "hidden_variables": [
                {
                    "name": h.name,
                    "description": h.description,
                    "evidence": h.evidence,
                    "testable_prediction": h.testable_prediction,
                }
                for h in self.hidden_variables
            ],
            "causal_narrative": self.causal_narrative,
            "k_decomposition": self.k_decomposition,
            "confidence": self.confidence,
            "severe_test_suggestions": self.severe_test_suggestions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TheoristResult:
        return cls(
            derived_observables=[
                DerivedObservable(**d) for d in data.get("derived_observables", [])
            ],
            hidden_variables=[
                HiddenVariable(**h) for h in data.get("hidden_variables", [])
            ],
            causal_narrative=data.get("causal_narrative", ""),
            k_decomposition=data.get("k_decomposition", ""),
            confidence=data.get("confidence", 0.5),
            severe_test_suggestions=data.get("severe_test_suggestions", []),
        )


@dataclass
class SevereTestCommand:
    """A priority research direction for the next discovery cycle."""

    command_type: str  # 'initial_condition', 'topology_test', 'parity_challenge'
    description: str
    target_law_id: str | None = None
    initial_conditions_json: str | None = None  # JSON of specific ICs to test
    grid_lengths_json: str | None = None  # JSON of grid lengths to try
    priority: str = "medium"  # 'high', 'medium', 'low'

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_type": self.command_type,
            "description": self.description,
            "target_law_id": self.target_law_id,
            "initial_conditions_json": self.initial_conditions_json,
            "grid_lengths_json": self.grid_lengths_json,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SevereTestCommand:
        return cls(
            command_type=data.get("command_type", "initial_condition"),
            description=data.get("description", ""),
            target_law_id=data.get("target_law_id"),
            initial_conditions_json=data.get("initial_conditions_json"),
            grid_lengths_json=data.get("grid_lengths_json"),
            priority=data.get("priority", "medium"),
        )


@dataclass
class StandardModel:
    """The agent's evolving best theory.

    Distinct from the Falsification Graveyard — this is the current
    coherent picture of how the universe works.
    """

    fixed_laws: list[str] = field(default_factory=list)  # law_ids of accepted laws
    archived_laws: list[str] = field(default_factory=list)  # law_ids demoted by auditor
    derived_observables: list[DerivedObservable] = field(default_factory=list)
    causal_narrative: str = ""
    hidden_variables: list[HiddenVariable] = field(default_factory=list)
    k_decomposition: str = ""
    confidence: float = 0.5
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed_laws": self.fixed_laws,
            "archived_laws": self.archived_laws,
            "derived_observables": [
                {
                    "name": d.name,
                    "expression": d.expression,
                    "rationale": d.rationale,
                    "source_laws": d.source_laws,
                }
                for d in self.derived_observables
            ],
            "causal_narrative": self.causal_narrative,
            "hidden_variables": [
                {
                    "name": h.name,
                    "description": h.description,
                    "evidence": h.evidence,
                    "testable_prediction": h.testable_prediction,
                }
                for h in self.hidden_variables
            ],
            "k_decomposition": self.k_decomposition,
            "confidence": self.confidence,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StandardModel:
        return cls(
            fixed_laws=data.get("fixed_laws", []),
            archived_laws=data.get("archived_laws", []),
            derived_observables=[
                DerivedObservable(**d)
                for d in data.get("derived_observables", [])
            ],
            causal_narrative=data.get("causal_narrative", ""),
            hidden_variables=[
                HiddenVariable(**h) for h in data.get("hidden_variables", [])
            ],
            k_decomposition=data.get("k_decomposition", ""),
            confidence=data.get("confidence", 0.5),
            version=data.get("version", 1),
        )


@dataclass
class ReflectionResult:
    """Combined output of a full reflection cycle."""

    auditor_result: AuditorResult
    theorist_result: TheoristResult
    standard_model: StandardModel
    severe_test_commands: list[SevereTestCommand] = field(default_factory=list)
    research_log_addendum: str = ""
    runtime_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "auditor_result": self.auditor_result.to_dict(),
            "theorist_result": self.theorist_result.to_dict(),
            "standard_model": self.standard_model.to_dict(),
            "severe_test_commands": [
                c.to_dict() for c in self.severe_test_commands
            ],
            "research_log_addendum": self.research_log_addendum,
            "runtime_ms": self.runtime_ms,
        }
