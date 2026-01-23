"""Semantic linter for law proposals.

Checks for mismatches between law names and the quantities they use.
For example, a law named "particle_conserved" that uses cell counts
instead of particle counts should be flagged.
"""

from dataclasses import dataclass

from src.claims.expr_ast import expr_to_string
from src.claims.expr_parser import parse_expression, ParseError
from src.claims.quantity_types import (
    LintWarning,
    QuantityType,
    TypedQuantity,
    infer_quantity_type,
    lint_law_name,
    lint_observable_expression,
)
from src.claims.schema import CandidateLaw
from src.universe.observables import (
    CANONICAL_OBSERVABLES,
    ConservationStatus,
    suggest_canonical_observable,
)


@dataclass
class LintResult:
    """Result of linting a law."""

    law_id: str
    warnings: list[LintWarning]
    observable_types: list[tuple[str, TypedQuantity]]
    has_errors: bool = False

    @property
    def is_clean(self) -> bool:
        return len(self.warnings) == 0


class SemanticLinter:
    """Lints law proposals for semantic consistency.

    Checks:
    1. Observable names match their expression types
    2. Law names match the quantities used
    3. Common semantic errors (e.g., "particle" with cell_count)
    """

    def __init__(self, strict: bool = False):
        """Initialize the linter.

        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict

    def lint(self, law: CandidateLaw) -> LintResult:
        """Lint a single law.

        Args:
            law: The law to lint

        Returns:
            LintResult with warnings and type information
        """
        warnings = []
        observable_types = []

        # Analyze each observable's expression
        for obs in law.observables:
            try:
                expr = parse_expression(obs.expr)
                typed = infer_quantity_type(expr)
                observable_types.append((obs.name, typed))

                # Check observable name against expression type
                obs_warnings = lint_observable_expression(obs.name, expr)
                warnings.extend(obs_warnings)

            except ParseError as e:
                warnings.append(LintWarning(
                    law_id=law.law_id,
                    message=f"Could not parse observable '{obs.name}': {e}",
                    severity="error",
                ))

        # Check law name against observable types
        typed_quantities = [t for _, t in observable_types]
        name_warnings = lint_law_name(law.law_id, typed_quantities)
        warnings.extend(name_warnings)

        # Additional semantic checks
        warnings.extend(self._check_semantic_consistency(law, observable_types))

        has_errors = any(w.severity == "error" for w in warnings)
        if self.strict:
            has_errors = has_errors or len(warnings) > 0

        return LintResult(
            law_id=law.law_id,
            warnings=warnings,
            observable_types=observable_types,
            has_errors=has_errors,
        )

    def _check_semantic_consistency(
        self,
        law: CandidateLaw,
        observable_types: list[tuple[str, TypedQuantity]],
    ) -> list[LintWarning]:
        """Additional semantic consistency checks."""
        warnings = []
        law_id_lower = law.law_id.lower()

        # Check observable names against canonical observables
        for obs in law.observables:
            suggestion = suggest_canonical_observable(obs.name, obs.expr)
            if suggestion:
                canonical = CANONICAL_OBSERVABLES.get(suggestion)
                if canonical:
                    warnings.append(LintWarning(
                        law_id=law.law_id,
                        message=f"Observable '{obs.name}' could use canonical name '{suggestion}' "
                                f"({canonical.description})",
                        severity="info",
                        suggested_fix=f"Rename '{obs.name}' to '{suggestion}'",
                    ))

        # Check for conserved quantities
        if "conserv" in law_id_lower:
            # Conservation laws should use conserved quantities
            for obs in law.observables:
                suggestion = suggest_canonical_observable(obs.name, obs.expr)
                if suggestion:
                    canonical = CANONICAL_OBSERVABLES.get(suggestion)
                    if canonical and canonical.conservation == ConservationStatus.NOT_CONSERVED:
                        warnings.append(LintWarning(
                            law_id=law.law_id,
                            message=f"Conservation law uses '{obs.name}' which maps to "
                                    f"'{suggestion}' - this is NOT conserved: {canonical.notes}",
                            severity="warning",
                            suggested_fix=f"Consider using TotalParticles, RightComponent, "
                                          f"LeftComponent, or Momentum instead",
                        ))

            # Also check by inferred type
            types_used = {t.quantity_type for _, t in observable_types}
            if QuantityType.CELL_COUNT in types_used:
                # Check if we already warned via canonical check
                already_warned = any(
                    "NOT conserved" in w.message
                    for w in warnings
                )
                if not already_warned:
                    warnings.append(LintWarning(
                        law_id=law.law_id,
                        message="Conservation law uses CELL_COUNT which is not conserved "
                                "(cell counts change during collisions)",
                        severity="warning",
                        suggested_fix="Consider using TotalParticles (particle_count) or "
                                      "RightComponent/LeftComponent (component_count)",
                    ))

        # Check for bound laws
        if "bound" in law_id_lower or "max" in law_id_lower:
            # Bound laws comparing different types might be suspicious
            if len(observable_types) >= 2:
                types = [t.quantity_type for _, t in observable_types]
                unique_types = set(types) - {QuantityType.UNKNOWN, QuantityType.LENGTH}
                if len(unique_types) > 1:
                    types_str = ", ".join(t.value for t in unique_types)
                    warnings.append(LintWarning(
                        law_id=law.law_id,
                        message=f"Bound law compares different quantity types: {types_str}",
                        severity="info" if QuantityType.LENGTH in types else "warning",
                    ))

        return warnings

    def lint_batch(
        self, laws: list[CandidateLaw]
    ) -> tuple[list[CandidateLaw], list[tuple[CandidateLaw, LintResult]]]:
        """Lint a batch of laws.

        Args:
            laws: Laws to lint

        Returns:
            Tuple of (clean laws, problematic laws with results)
        """
        clean = []
        problematic = []

        for law in laws:
            result = self.lint(law)
            if result.has_errors:
                problematic.append((law, result))
            else:
                clean.append(law)

        return clean, problematic

    def suggest_name_fix(self, law: CandidateLaw) -> str | None:
        """Suggest a fixed name for a law based on its actual quantities.

        Args:
            law: The law to analyze

        Returns:
            Suggested name or None if no fix needed
        """
        result = self.lint(law)

        # Find any suggested fixes
        for warning in result.warnings:
            if warning.suggested_fix:
                return warning.suggested_fix

        # Generate a fix based on types
        if not result.observable_types:
            return None

        types_used = {t.quantity_type for _, t in result.observable_types}
        types_used.discard(QuantityType.UNKNOWN)

        if not types_used:
            return None

        law_id_lower = law.law_id.lower()

        # Map types to preferred keywords
        type_keywords = {
            QuantityType.PARTICLE_COUNT: "particle",
            QuantityType.COMPONENT_COUNT: "component",
            QuantityType.CELL_COUNT: "cell",
            QuantityType.MOMENTUM_LIKE: "momentum",
        }

        # Check for mismatches and suggest fixes
        for wrong_keyword in ["particle", "cell", "component"]:
            if wrong_keyword in law_id_lower:
                for qty_type, correct_keyword in type_keywords.items():
                    if qty_type in types_used and wrong_keyword != correct_keyword:
                        return law.law_id.replace(wrong_keyword, correct_keyword)

        return None


def auto_relabel_law(law: CandidateLaw) -> CandidateLaw:
    """Automatically relabel a law's name based on its actual quantities.

    If the law uses cell_count but is named "particle_...", rename it.

    Args:
        law: The law to potentially relabel

    Returns:
        New law with corrected name, or original if no change needed
    """
    linter = SemanticLinter()
    suggested = linter.suggest_name_fix(law)

    if suggested and suggested != law.law_id:
        # Create a new law with the corrected name
        return CandidateLaw(
            schema_version=law.schema_version,
            law_id=suggested,
            template=law.template,
            quantifiers=law.quantifiers,
            preconditions=law.preconditions,
            observables=law.observables,
            claim=law.claim,
            claim_ast=law.claim_ast,
            forbidden=law.forbidden,
            transform=law.transform,
            direction=law.direction,
            bound_value=law.bound_value,
            bound_op=law.bound_op,
            proposed_tests=law.proposed_tests,
            capability_requirements=law.capability_requirements,
            distinguishes_from=law.distinguishes_from,
            novelty_claim=law.novelty_claim,
            ranking_features=law.ranking_features,
        )

    return law
