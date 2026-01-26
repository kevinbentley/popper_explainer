"""Domain models for mechanistic explanations.

An explanation consists of:
1. A hypothesis about HOW the universe works (mechanism)
2. Derived rules that can be used for prediction
3. Open questions and criticisms
4. Confidence level based on theorem support
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ExplanationStatus(str, Enum):
    """Status of an explanation."""

    PROPOSED = "proposed"  # Newly generated, not yet tested
    TESTING = "testing"  # Currently being tested
    VALIDATED = "validated"  # Predictions meet thresholds
    REFUTED = "refuted"  # Predictions failed significantly
    SUPERSEDED = "superseded"  # Replaced by better explanation


class MechanismType(str, Enum):
    """Types of mechanistic rules."""

    MOVEMENT = "movement"  # How entities move
    INTERACTION = "interaction"  # How entities interact
    TRANSFORMATION = "transformation"  # How entities change state
    BOUNDARY = "boundary"  # Boundary conditions
    CONSERVATION = "conservation"  # Conserved quantities


@dataclass
class MechanismRule:
    """A single rule in a mechanistic explanation.

    Attributes:
        rule_id: Unique identifier
        rule_type: Type of mechanism (movement, interaction, etc.)
        condition: When this rule applies
        effect: What happens when the rule fires
        priority: Order of rule application (lower = earlier)
        supporting_laws: Law IDs that support this rule
    """

    rule_id: str
    rule_type: MechanismType
    condition: str
    effect: str
    priority: int = 0
    supporting_laws: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "condition": self.condition,
            "effect": self.effect,
            "priority": self.priority,
            "supporting_laws": self.supporting_laws,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MechanismRule:
        return cls(
            rule_id=data["rule_id"],
            rule_type=MechanismType(data["rule_type"]),
            condition=data["condition"],
            effect=data["effect"],
            priority=data.get("priority", 0),
            supporting_laws=data.get("supporting_laws", []),
        )


@dataclass
class Mechanism:
    """A mechanistic model of the universe.

    The mechanism describes HOW the universe works through
    a set of prioritized rules.

    Attributes:
        rules: Ordered list of mechanism rules
        description: Natural language description
        assumptions: Underlying assumptions
        limitations: Known limitations
    """

    rules: list[MechanismRule]
    description: str
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": [r.to_dict() for r in self.rules],
            "description": self.description,
            "assumptions": self.assumptions,
            "limitations": self.limitations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mechanism:
        return cls(
            rules=[MechanismRule.from_dict(r) for r in data.get("rules", [])],
            description=data.get("description", ""),
            assumptions=data.get("assumptions", []),
            limitations=data.get("limitations", []),
        )

    def get_rules_by_type(self, rule_type: MechanismType) -> list[MechanismRule]:
        """Get all rules of a specific type."""
        return [r for r in self.rules if r.rule_type == rule_type]


@dataclass
class OpenQuestion:
    """An open question about the explanation.

    Represents gaps in understanding or areas needing
    further investigation.
    """

    question: str
    category: str  # 'definition', 'mechanism', 'boundary', 'prediction'
    priority: str = "medium"  # 'high', 'medium', 'low'
    related_theorems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "category": self.category,
            "priority": self.priority,
            "related_theorems": self.related_theorems,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenQuestion:
        return cls(
            question=data["question"],
            category=data.get("category", "mechanism"),
            priority=data.get("priority", "medium"),
            related_theorems=data.get("related_theorems", []),
        )


@dataclass
class Criticism:
    """A criticism of the current explanation.

    Represents potential flaws or weaknesses that need
    to be addressed.
    """

    criticism: str
    severity: str  # 'critical', 'major', 'minor'
    source: str  # 'theorem', 'prediction', 'counterexample', 'llm'
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "criticism": self.criticism,
            "severity": self.severity,
            "source": self.source,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Criticism:
        return cls(
            criticism=data["criticism"],
            severity=data.get("severity", "minor"),
            source=data.get("source", "llm"),
            evidence=data.get("evidence", ""),
        )


@dataclass
class Explanation:
    """A complete mechanistic explanation.

    An explanation synthesizes validated theorems into a
    coherent mechanistic model that can make predictions.

    Attributes:
        explanation_id: Unique identifier
        hypothesis_text: Natural language hypothesis
        mechanism: Structured mechanism model
        supporting_theorems: Theorem IDs that support this explanation
        open_questions: Gaps in understanding
        criticisms: Known weaknesses
        confidence: Overall confidence score (0-1)
        status: Current status
        iteration_id: Iteration when created
        prediction_accuracy: Most recent prediction accuracy
    """

    explanation_id: str
    hypothesis_text: str
    mechanism: Mechanism
    supporting_theorems: list[str] = field(default_factory=list)
    open_questions: list[OpenQuestion] = field(default_factory=list)
    criticisms: list[Criticism] = field(default_factory=list)
    confidence: float = 0.5
    status: ExplanationStatus = ExplanationStatus.PROPOSED
    iteration_id: int | None = None
    prediction_accuracy: float | None = None
    created_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def fingerprint(self) -> str:
        """Compute a fingerprint for deduplication."""
        content = json.dumps(self.mechanism.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanation_id": self.explanation_id,
            "hypothesis_text": self.hypothesis_text,
            "mechanism": self.mechanism.to_dict(),
            "supporting_theorems": self.supporting_theorems,
            "open_questions": [q.to_dict() for q in self.open_questions],
            "criticisms": [c.to_dict() for c in self.criticisms],
            "confidence": self.confidence,
            "status": self.status.value,
            "iteration_id": self.iteration_id,
            "prediction_accuracy": self.prediction_accuracy,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Explanation:
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            explanation_id=data["explanation_id"],
            hypothesis_text=data["hypothesis_text"],
            mechanism=Mechanism.from_dict(data.get("mechanism", {})),
            supporting_theorems=data.get("supporting_theorems", []),
            open_questions=[
                OpenQuestion.from_dict(q) for q in data.get("open_questions", [])
            ],
            criticisms=[Criticism.from_dict(c) for c in data.get("criticisms", [])],
            confidence=data.get("confidence", 0.5),
            status=ExplanationStatus(data.get("status", "proposed")),
            iteration_id=data.get("iteration_id"),
            prediction_accuracy=data.get("prediction_accuracy"),
            created_at=created_at,
        )

    def has_critical_issues(self) -> bool:
        """Check if explanation has critical criticisms."""
        return any(c.severity == "critical" for c in self.criticisms)

    def has_high_priority_questions(self) -> bool:
        """Check if explanation has high-priority open questions."""
        return any(q.priority == "high" for q in self.open_questions)


@dataclass
class ExplanationBatch:
    """Result of an explanation generation iteration."""

    explanations: list[Explanation]
    prompt_hash: str
    runtime_ms: int
    warnings: list[str] = field(default_factory=list)
    research_log: str | None = None  # LLM's mechanistic notebook for continuity

    @property
    def count(self) -> int:
        return len(self.explanations)
