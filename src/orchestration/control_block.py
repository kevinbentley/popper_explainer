"""Control block schema for phase outputs.

Every phase handler emits a ControlBlock alongside its phase-specific outputs.
The orchestrator uses control blocks to make deterministic transition decisions
based on objective metrics and LLM recommendations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PhaseRecommendation(str, Enum):
    """LLM's recommendation for phase transition."""

    STAY = "stay"  # Continue in current phase
    ADVANCE = "advance"  # Move to next phase
    RETREAT = "retreat"  # Return to previous phase for refinement


class StopReason(str, Enum):
    """Specific reason for the phase recommendation.

    These are machine-checkable reasons that the orchestrator can use
    to validate the LLM's recommendation.
    """

    # Discovery phase reasons
    SATURATION = "saturation"  # Novelty rate dropped below threshold
    HIGH_REDUNDANCY = "high_redundancy"  # Most new laws are redundant
    STABLE_CORE_SUITE = "stable_core_suite"  # Core law suite is stable
    TARGETED_SEARCH_COMPLETE = "targeted_search_complete"  # Finished targeted falsification

    # Theorem phase reasons
    NEEDS_MORE_LAWS = "needs_more_laws"  # Insufficient laws for theorems
    GAPS_IDENTIFIED = "gaps_identified"  # Missing lemmas or ambiguous observables
    THEOREMS_STABLE = "theorems_stable"  # Theorem set is stable
    CONTRADICTION_FOUND = "contradiction_found"  # Theorems contradict each other

    # Explanation phase reasons
    PREDICTIONS_LOW = "predictions_low"  # Not enough predictions generated
    MECHANISM_INCOMPLETE = "mechanism_incomplete"  # Mechanism is not well-defined
    READY_FOR_VALIDATION = "ready_for_validation"  # Ready for prediction testing
    OPEN_QUESTIONS = "open_questions"  # Unresolved questions need theorem help

    # Prediction phase reasons
    ACCURACY_THRESHOLD_MET = "accuracy_threshold_met"  # Hit accuracy targets
    ADVERSARIAL_FAILURES = "adversarial_failures"  # Failing on adversarial cases
    HELD_OUT_FAILURES = "held_out_failures"  # Failing on held-out set

    # Finalize phase reasons
    REPORT_COMPLETE = "report_complete"  # Final report generated
    ARTIFACTS_FROZEN = "artifacts_frozen"  # All artifacts frozen

    # General reasons
    RESOURCE_LIMIT = "resource_limit"  # Hit iteration or time limit
    MANUAL_OVERRIDE = "manual_override"  # User requested phase change
    CONTINUING = "continuing"  # Normal operation, no phase change


@dataclass
class EvidenceReference:
    """Reference to a specific artifact supporting a claim.

    All LLM claims about readiness must be backed by artifact references.
    This makes the control block auditable.
    """

    artifact_type: str  # 'law', 'theorem', 'counterexample', 'prediction', 'evaluation'
    artifact_id: str  # Unique ID of the artifact
    role: str  # 'supports', 'contradicts', 'requires', 'refutes'
    note: str = ""  # Optional explanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "role": self.role,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceReference:
        return cls(
            artifact_type=data["artifact_type"],
            artifact_id=data["artifact_id"],
            role=data["role"],
            note=data.get("note", ""),
        )


@dataclass
class PhaseRequest:
    """Request from one phase to another.

    Used for targeted refinement: e.g., theorem phase requests
    discovery phase to falsify a specific conjecture.
    """

    request_type: str  # 'test_law', 'falsify_theorem', 'clarify_observable', 'add_lemma'
    target_id: str | None  # ID of target artifact (if applicable)
    description: str  # Human-readable description
    priority: str = "medium"  # 'high', 'medium', 'low'

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_type": self.request_type,
            "target_id": self.target_id,
            "description": self.description,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseRequest:
        return cls(
            request_type=data["request_type"],
            target_id=data.get("target_id"),
            description=data["description"],
            priority=data.get("priority", "medium"),
        )


@dataclass
class ProposedTransition:
    """A proposed phase transition with rationale."""

    target_phase: str  # Phase enum value as string
    reason: str  # Why this transition is proposed
    confidence: float = 0.8  # LLM's confidence in this transition (0-1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_phase": self.target_phase,
            "reason": self.reason,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProposedTransition:
        return cls(
            target_phase=data["target_phase"],
            reason=data["reason"],
            confidence=data.get("confidence", 0.8),
        )


@dataclass
class ControlBlock:
    """Structured output from each phase's LLM call.

    This is the standardized format that every phase handler must emit.
    The orchestrator uses this to make deterministic transition decisions
    based on a combination of:
    1. Objective readiness metrics (computed from harness data)
    2. LLM's readiness suggestion (advisory)
    3. Hysteresis rules (prevent oscillation)
    """

    # LLM's readiness assessment (0-100, advisory only)
    readiness_score_suggestion: int

    # Why the LLM suggests this score
    readiness_justification: str

    # LLM's recommendation for phase transition
    phase_recommendation: PhaseRecommendation

    # Specific reason for recommendation
    stop_reason: StopReason

    # Artifact references supporting this recommendation
    evidence: list[EvidenceReference] = field(default_factory=list)

    # Requests for other phases (e.g., targeted falsification)
    requests: list[PhaseRequest] = field(default_factory=list)

    # Proposed transitions if recommending a phase change
    proposed_transitions: list[ProposedTransition] = field(default_factory=list)

    # Phase-specific outputs (laws, theorems, predictions, etc.)
    phase_outputs: dict[str, Any] = field(default_factory=dict)

    # Metadata
    phase_name: str = ""
    iteration_number: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "readiness_score_suggestion": self.readiness_score_suggestion,
            "readiness_justification": self.readiness_justification,
            "phase_recommendation": self.phase_recommendation.value,
            "stop_reason": self.stop_reason.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "requests": [r.to_dict() for r in self.requests],
            "proposed_transitions": [t.to_dict() for t in self.proposed_transitions],
            "phase_outputs": self.phase_outputs,
            "phase_name": self.phase_name,
            "iteration_number": self.iteration_number,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlBlock:
        """Deserialize from dictionary."""
        return cls(
            readiness_score_suggestion=data["readiness_score_suggestion"],
            readiness_justification=data["readiness_justification"],
            phase_recommendation=PhaseRecommendation(data["phase_recommendation"]),
            stop_reason=StopReason(data["stop_reason"]),
            evidence=[EvidenceReference.from_dict(e) for e in data.get("evidence", [])],
            requests=[PhaseRequest.from_dict(r) for r in data.get("requests", [])],
            proposed_transitions=[
                ProposedTransition.from_dict(t)
                for t in data.get("proposed_transitions", [])
            ],
            phase_outputs=data.get("phase_outputs", {}),
            phase_name=data.get("phase_name", ""),
            iteration_number=data.get("iteration_number", 0),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.utcnow(),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ControlBlock:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def validate(self) -> list[str]:
        """Validate the control block for consistency.

        Returns a list of validation errors (empty if valid).
        """
        errors: list[str] = []

        # Readiness score must be 0-100
        if not 0 <= self.readiness_score_suggestion <= 100:
            errors.append(
                f"readiness_score_suggestion must be 0-100, got {self.readiness_score_suggestion}"
            )

        # If recommending ADVANCE, should have high readiness
        if (
            self.phase_recommendation == PhaseRecommendation.ADVANCE
            and self.readiness_score_suggestion < 50
        ):
            errors.append(
                "ADVANCE recommendation with low readiness score - suspicious"
            )

        # If recommending RETREAT, should have requests
        if (
            self.phase_recommendation == PhaseRecommendation.RETREAT
            and not self.requests
        ):
            errors.append(
                "RETREAT recommendation should include requests for what to refine"
            )

        # Evidence should reference valid artifact types
        valid_artifact_types = {
            "law",
            "theorem",
            "counterexample",
            "prediction",
            "evaluation",
            "explanation",
            "cluster",
        }
        for ev in self.evidence:
            if ev.artifact_type not in valid_artifact_types:
                errors.append(f"Unknown artifact type: {ev.artifact_type}")

        # Request priorities should be valid
        valid_priorities = {"high", "medium", "low"}
        for req in self.requests:
            if req.priority not in valid_priorities:
                errors.append(f"Invalid request priority: {req.priority}")

        return errors


def create_control_block_from_llm_output(
    llm_output: dict[str, Any],
    phase_name: str,
    iteration_number: int,
) -> ControlBlock:
    """Parse LLM output into a ControlBlock.

    This is the bridge between raw LLM JSON output and the typed ControlBlock.
    It handles missing fields gracefully with sensible defaults.
    """
    # Extract control_block section if nested
    cb_data = llm_output.get("control_block", llm_output)

    # Parse with defaults for missing fields
    return ControlBlock(
        readiness_score_suggestion=cb_data.get("readiness_score_suggestion", 50),
        readiness_justification=cb_data.get(
            "readiness_justification", "No justification provided"
        ),
        phase_recommendation=PhaseRecommendation(
            cb_data.get("phase_recommendation", "stay")
        ),
        stop_reason=StopReason(cb_data.get("stop_reason", "continuing")),
        evidence=[
            EvidenceReference.from_dict(e) for e in cb_data.get("evidence", [])
        ],
        requests=[PhaseRequest.from_dict(r) for r in cb_data.get("requests", [])],
        proposed_transitions=[
            ProposedTransition.from_dict(t)
            for t in cb_data.get("proposed_transitions", [])
        ],
        phase_outputs=llm_output.get("phase_outputs", {}),
        phase_name=phase_name,
        iteration_number=iteration_number,
        timestamp=datetime.utcnow(),
    )
