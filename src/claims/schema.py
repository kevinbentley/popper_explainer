"""Schema definitions for candidate laws.

These Pydantic models define the structure of law proposals
from the Law Discovery subsystem.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Template(str, Enum):
    """Supported claim templates.

    This is the authoritative list - no other templates are allowed.
    """

    INVARIANT = "invariant"  # ∀t∈[0..T]: f(t) == f(0)
    MONOTONE = "monotone"  # ∀t∈[0..T-1]: f(t+1) <= f(t) or >=
    IMPLICATION_STEP = "implication_step"  # ∀t∈[0..T-1]: P(t) → Q(t+1)
    IMPLICATION_STATE = "implication_state"  # ∀t∈[0..T]: P(t) → Q(t)
    EVENTUALLY = "eventually"  # ∀t0: P(t0) → ∃t∈[t0..t0+H]: Q(t)
    SYMMETRY_COMMUTATION = "symmetry_commutation"  # evolve(T(S)) == T(evolve(S))
    BOUND = "bound"  # ∀t∈[0..T]: f(t) <= k or >=


class ComparisonOp(str, Enum):
    """Comparison operators for preconditions and bounds."""

    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


class MonotoneDirection(str, Enum):
    """Direction for monotone laws."""

    NON_INCREASING = "<="  # f(t+1) <= f(t)
    NON_DECREASING = ">="  # f(t+1) >= f(t)


class Quantifiers(BaseModel):
    """Quantifier bounds for laws."""

    T: int = Field(ge=1, description="Time horizon")
    H: int | None = Field(
        default=None, ge=1, description="Eventuality horizon (for eventually template)"
    )


class Precondition(BaseModel):
    """A precondition for law applicability.

    Evaluated on initial state and configuration.
    """

    lhs: str = Field(description="Left-hand side (observable name or 'grid_length')")
    op: ComparisonOp = Field(description="Comparison operator")
    rhs: int | str = Field(description="Right-hand side (value or observable name)")


class Observable(BaseModel):
    """A named observable with its expression definition."""

    name: str = Field(description="Observable name (e.g., 'R_total')")
    expr: str = Field(description="Expression (e.g., \"count('>') + count('X')\")")


class ProposedTest(BaseModel):
    """A proposed test family for evaluating the law."""

    family: str = Field(description="Generator family name")
    params: dict[str, Any] = Field(default_factory=dict, description="Generator parameters")


class CapabilityRequirements(BaseModel):
    """Capabilities required to test this law."""

    missing_observables: list[str] = Field(default_factory=list)
    missing_transforms: list[str] = Field(default_factory=list)
    missing_generators: list[str] = Field(default_factory=list)

    def is_satisfied(self) -> bool:
        """Check if all requirements are met (nothing missing)."""
        return (
            len(self.missing_observables) == 0
            and len(self.missing_transforms) == 0
            and len(self.missing_generators) == 0
        )


class RankingFeatures(BaseModel):
    """Features used for law ranking/prioritization."""

    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    discrimination: float = Field(default=0.0, ge=0.0, le=1.0)
    testability: float = Field(default=0.0, ge=0.0, le=1.0)
    redundancy: float = Field(default=0.0, ge=0.0, le=1.0)


class CandidateLaw(BaseModel):
    """A candidate law proposed by the Law Discovery subsystem.

    This is the core data structure that flows through the system.

    Claims can be specified in two ways:
    1. Structured AST (preferred): Use claim_ast with a JSON expression tree
    2. String format (legacy): Use claim as a string expression

    If claim_ast is provided, it takes precedence over claim string.
    """

    schema_version: str = Field(default="1.0.0")
    law_id: str = Field(description="Unique identifier for this law")
    template: Template = Field(description="Claim template type")
    quantifiers: Quantifiers = Field(description="Time bounds")
    preconditions: list[Precondition] = Field(
        default_factory=list, description="Applicability conditions"
    )
    observables: list[Observable] = Field(
        default_factory=list, description="Observable definitions"
    )
    claim: str = Field(description="Human-readable claim statement")
    forbidden: str = Field(description="What constitutes a counterexample")

    # Structured claim AST (preferred over string claim)
    claim_ast: dict[str, Any] | None = Field(
        default=None,
        description="Structured claim as JSON AST. Takes precedence over claim string."
    )

    # Template-specific fields
    transform: str | None = Field(
        default=None, description="Transform name for symmetry_commutation template"
    )
    direction: MonotoneDirection | None = Field(
        default=None, description="Direction for monotone template"
    )
    bound_value: int | None = Field(
        default=None, description="Bound value for bound template"
    )
    bound_op: ComparisonOp | None = Field(
        default=None, description="Comparison operator for bound template"
    )

    # Metadata
    proposed_tests: list[ProposedTest] = Field(
        default_factory=list, description="Suggested test families"
    )
    capability_requirements: CapabilityRequirements = Field(
        default_factory=CapabilityRequirements
    )
    distinguishes_from: list[str] = Field(
        default_factory=list, description="Rival hypotheses this law distinguishes from"
    )
    novelty_claim: str | None = Field(
        default=None, description="Why this law is novel"
    )
    ranking_features: RankingFeatures = Field(default_factory=RankingFeatures)

    @field_validator("transform")
    @classmethod
    def validate_transform(cls, v: str | None, info) -> str | None:
        """Ensure transform is provided for symmetry_commutation template."""
        # Note: We can't access other fields easily in Pydantic v2 validators
        # This validation is done in the compiler instead
        return v

    def get_observable_names(self) -> set[str]:
        """Get all observable names defined by this law."""
        return {obs.name for obs in self.observables}

    def content_hash(self) -> str:
        """Compute a content hash for duplicate detection."""
        import hashlib
        import json

        # Normalize and hash the essential content
        content = {
            "template": self.template.value,
            "quantifiers": self.quantifiers.model_dump(),
            "preconditions": [p.model_dump() for p in sorted(self.preconditions, key=lambda x: x.lhs)],
            "observables": [o.model_dump() for o in sorted(self.observables, key=lambda x: x.name)],
            "claim": self.claim,
            "transform": self.transform,
            "direction": self.direction.value if self.direction else None,
            "bound_value": self.bound_value,
            "bound_op": self.bound_op.value if self.bound_op else None,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
