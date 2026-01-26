"""Domain models for theorem generation and failure clustering."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# =============================================================================
# PHASE-D: Support Role and Missing Structure Enums
# =============================================================================


class SupportRole(str, Enum):
    """Role of a supporting law in a theorem."""

    CONFIRMS = "confirms"
    CONSTRAINS = "constrains"
    REFUTES_ALTERNATIVE = "refutes_alternative"

    @classmethod
    def _missing_(cls, value: str) -> "SupportRole":
        """Handle legacy/unknown role names with fallback."""
        # Normalize common variations
        normalized = value.lower().strip().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        # Default fallback
        return cls.CONFIRMS


class MissingStructureType(str, Enum):
    """Typed categories for missing structure in theorems."""

    DEFINITION_MISSING = "DEFINITION_MISSING"
    LOCAL_STRUCTURE_MISSING = "LOCAL_STRUCTURE_MISSING"
    TEMPORAL_STRUCTURE_MISSING = "TEMPORAL_STRUCTURE_MISSING"
    MECHANISM_MISSING = "MECHANISM_MISSING"

    @classmethod
    def classify(cls, text: str) -> "MissingStructureType":
        """Auto-classify a string into a typed category based on keywords.

        Used for backward compatibility with old format strings.
        """
        lower = text.lower()

        # Definition/observable missing
        if any(
            kw in lower
            for kw in [
                "definition",
                "observable",
                "undefined",
                "ambiguous",
                "clarify",
                "what is",
                "meaning of",
            ]
        ):
            return cls.DEFINITION_MISSING

        # Temporal/eventual structure
        if any(
            kw in lower
            for kw in [
                "temporal",
                "eventually",
                "asymptotic",
                "long-term",
                "over time",
                "sequence",
                "after",
                "before",
            ]
        ):
            return cls.TEMPORAL_STRUCTURE_MISSING

        # Local/spatial structure
        if any(
            kw in lower
            for kw in [
                "local",
                "spatial",
                "adjacent",
                "neighbor",
                "position",
                "pattern",
                "configuration",
                "arrangement",
            ]
        ):
            return cls.LOCAL_STRUCTURE_MISSING

        # Mechanism/causal structure
        if any(
            kw in lower
            for kw in [
                "mechanism",
                "cause",
                "why",
                "how",
                "process",
                "underlying",
                "reason",
            ]
        ):
            return cls.MECHANISM_MISSING

        # Default to definition missing
        return cls.DEFINITION_MISSING


@dataclass
class TypedMissingStructure:
    """A typed missing structure with category, target, and optional note."""

    type: MissingStructureType
    target: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "target": self.target,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedMissingStructure":
        return cls(
            type=MissingStructureType(data["type"]),
            target=data["target"],
            note=data.get("note", ""),
        )

    @classmethod
    def from_string(cls, text: str) -> "TypedMissingStructure":
        """Convert an old-format string to typed structure.

        Auto-classifies the type based on keywords.
        """
        structure_type = MissingStructureType.classify(text)
        return cls(type=structure_type, target=text, note="")


class TheoremStatus(str, Enum):
    """Status levels for theorems."""

    ESTABLISHED = "Established"  # Strong support, no contradictions
    CONDITIONAL = "Conditional"  # Support with known limitations
    CONJECTURAL = "Conjectural"  # Plausible but needs more evidence


class FailureBucket(str, Enum):
    """Deterministic bucket categories for failure clustering.

    PHASE-C taxonomy (6 refined buckets):
    - DEFINITION_GAP: Missing/ambiguous definitions
    - COLLISION_TRIGGERS: Incoming collision, trigger, bound issues
    - LOCAL_PATTERN: Configuration, arrangement, adjacent patterns
    - EVENTUALITY: Eventually, long-term, asymptotic behaviors
    - MONOTONICITY: Monotonic, non-increasing/decreasing trends
    - SYMMETRY: Mirror, swap, reflection, translation invariance
    - OTHER: Default bucket
    """

    DEFINITION_GAP = "DEFINITION_GAP"
    COLLISION_TRIGGERS = "COLLISION_TRIGGERS"
    LOCAL_PATTERN = "LOCAL_PATTERN"
    EVENTUALITY = "EVENTUALITY"
    MONOTONICITY = "MONOTONICITY"
    SYMMETRY = "SYMMETRY"
    OTHER = "OTHER"

    # Legacy aliases for backwards compatibility
    @classmethod
    def _missing_(cls, value: str) -> "FailureBucket":
        """Handle legacy bucket names."""
        legacy_map = {
            "TEMPORAL_EVENTUAL": cls.EVENTUALITY,
            "SYMMETRY_MISAPPLIED": cls.SYMMETRY,
            "COUNT_ONLY_TRAP": cls.LOCAL_PATTERN,  # Map to closest
            "FEATURE_MISMATCH": cls.DEFINITION_GAP,  # Map to closest
        }
        if value in legacy_map:
            return legacy_map[value]
        return cls.OTHER


@dataclass
class LawSupport:
    """A law that supports or constrains a theorem."""

    law_id: str
    role: SupportRole | str  # SupportRole enum or string for backward compat

    def __post_init__(self) -> None:
        """Normalize role to SupportRole enum."""
        if isinstance(self.role, str):
            self.role = SupportRole(self.role)

    @property
    def role_value(self) -> str:
        """Get role as string value."""
        if isinstance(self.role, SupportRole):
            return self.role.value
        return self.role

    def to_dict(self) -> dict[str, Any]:
        return {"law_id": self.law_id, "role": self.role_value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LawSupport":
        role_str = data.get("role", "confirms")
        return cls(law_id=data["law_id"], role=SupportRole(role_str))


@dataclass
class Theorem:
    """A synthesized theorem from multiple laws."""

    theorem_id: str
    name: str
    status: TheoremStatus
    claim: str
    support: list[LawSupport]
    failure_modes: list[str] = field(default_factory=list)
    missing_structure: list[str] = field(default_factory=list)
    typed_missing_structure: list[TypedMissingStructure] = field(default_factory=list)
    bucket_tags: list[str] = field(default_factory=list)
    signature_version: str | None = None  # PHASE-E: Track signature format version

    def __post_init__(self) -> None:
        """Auto-populate typed_missing_structure from missing_structure if empty."""
        if self.missing_structure and not self.typed_missing_structure:
            self.typed_missing_structure = [
                TypedMissingStructure.from_string(ms) for ms in self.missing_structure
            ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "theorem_id": self.theorem_id,
            "name": self.name,
            "status": self.status.value,
            "claim": self.claim,
            "support": [s.to_dict() for s in self.support],
            "failure_modes": self.failure_modes,
            "missing_structure": self.missing_structure,
            "typed_missing_structure": [tms.to_dict() for tms in self.typed_missing_structure],
            "bucket_tags": self.bucket_tags,
            "signature_version": self.signature_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Theorem":
        # Parse typed_missing_structure if present
        typed_ms: list[TypedMissingStructure] = []
        if "typed_missing_structure" in data:
            for item in data["typed_missing_structure"]:
                if isinstance(item, dict):
                    typed_ms.append(TypedMissingStructure.from_dict(item))
                elif isinstance(item, str):
                    typed_ms.append(TypedMissingStructure.from_string(item))

        return cls(
            theorem_id=data["theorem_id"],
            name=data["name"],
            status=TheoremStatus(data["status"]),
            claim=data["claim"],
            support=[LawSupport.from_dict(s) for s in data.get("support", [])],
            failure_modes=data.get("failure_modes", []),
            missing_structure=data.get("missing_structure", []),
            typed_missing_structure=typed_ms,
            bucket_tags=data.get("bucket_tags", []),
            signature_version=data.get("signature_version"),
        )


@dataclass
class TheoremBatch:
    """Result of a theorem generation batch."""

    theorems: list[Theorem]
    rejections: list[tuple[dict[str, Any], str]]  # (raw_data, rejection_reason)
    prompt_hash: str
    runtime_ms: int
    warnings: list[str] = field(default_factory=list)
    research_log: str | None = None  # LLM's theoretical notebook for continuity

    @property
    def accepted_count(self) -> int:
        return len(self.theorems)

    @property
    def rejected_count(self) -> int:
        return len(self.rejections)


@dataclass
class TheoremGenerationArtifact:
    """Artifact capturing all data needed to reproduce theorem generation.

    This is the core reproducibility structure for PHASE-D.
    """

    artifact_hash: str  # SHA256 of content (raw_response + snapshot_hash)
    snapshot_hash: str  # Hash of law snapshots used as input
    prompt_template_version: str  # Version of the prompt template
    model_name: str  # Model name (e.g., "gemini-2.5-flash")
    model_params: dict[str, Any]  # Model parameters (temp, max_tokens, etc.)
    raw_response: str  # Raw LLM response before parsing
    parsed_response: list[dict[str, Any]]  # Parsed JSON response
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_hash": self.artifact_hash,
            "snapshot_hash": self.snapshot_hash,
            "prompt_template_version": self.prompt_template_version,
            "model_name": self.model_name,
            "model_params": self.model_params,
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TheoremGenerationArtifact":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            artifact_hash=data["artifact_hash"],
            snapshot_hash=data["snapshot_hash"],
            prompt_template_version=data["prompt_template_version"],
            model_name=data["model_name"],
            model_params=data.get("model_params", {}),
            raw_response=data["raw_response"],
            parsed_response=data.get("parsed_response", []),
            created_at=created_at,
        )


@dataclass
class LawSnapshot:
    """Snapshot of a law for theorem generation context."""

    law_id: str
    template: str
    claim: str
    status: str  # 'PASS' or 'FAIL'
    observables: list[dict[str, Any]] = field(default_factory=list)
    counterexample: dict[str, Any] | None = None
    power_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "law_id": self.law_id,
            "template": self.template,
            "claim": self.claim,
            "status": self.status,
            "observables": self.observables,
            "counterexample": self.counterexample,
            "power_metrics": self.power_metrics,
        }


@dataclass
class FailureCluster:
    """A cluster of theorems with similar failure signatures."""

    cluster_id: str
    bucket_tags: list[str]  # Multi-label bucket assignment
    semantic_cluster_idx: int
    theorem_ids: list[str]
    centroid_signature: str
    avg_similarity: float
    top_keywords: list[tuple[str, float]] = field(default_factory=list)  # TF-IDF top terms
    recommended_action: str = "OBSERVABLE"  # SCHEMA_FIX, OBSERVABLE, GATING
    distance_threshold: float = 0.6  # Threshold used for clustering

    # Legacy: single bucket (primary tag) for backwards compatibility
    @property
    def bucket(self) -> FailureBucket:
        """Return primary bucket for backwards compatibility."""
        if not self.bucket_tags:
            return FailureBucket.OTHER
        try:
            return FailureBucket(self.bucket_tags[0])
        except ValueError:
            return FailureBucket.OTHER

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "bucket": self.bucket.value,  # For backwards compatibility
            "bucket_tags": self.bucket_tags,
            "semantic_cluster_idx": self.semantic_cluster_idx,
            "theorem_ids": self.theorem_ids,
            "centroid_signature": self.centroid_signature,
            "avg_similarity": self.avg_similarity,
            "top_keywords": self.top_keywords,
            "recommended_action": self.recommended_action,
            "distance_threshold": self.distance_threshold,
        }


@dataclass
class ObservableProposal:
    """A proposed new observable to address failure modes."""

    proposal_id: str
    cluster_id: str
    observable_name: str
    observable_expr: str
    rationale: str
    priority: str  # 'high', 'medium', 'low'
    action_type: str = "OBSERVABLE"  # SCHEMA_FIX, OBSERVABLE, GATING

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "cluster_id": self.cluster_id,
            "observable_name": self.observable_name,
            "observable_expr": self.observable_expr,
            "rationale": self.rationale,
            "priority": self.priority,
            "action_type": self.action_type,
        }
