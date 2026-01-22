"""Response parser for LLM law proposals."""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.claims.ast_schema import validate_claim_ast
from src.claims.schema import (
    CandidateLaw,
    CapabilityRequirements,
    ComparisonOp,
    MonotoneDirection,
    Observable,
    Precondition,
    ProposedTest,
    Quantifiers,
    Template,
)


class ParseError(Exception):
    """Error parsing LLM response."""

    def __init__(self, message: str, law_data: dict[str, Any] | None = None):
        super().__init__(message)
        self.law_data = law_data


@dataclass
class ParseResult:
    """Result of parsing an LLM response.

    Attributes:
        laws: Successfully parsed laws
        rejections: Laws that failed parsing with reasons
        warnings: Non-fatal issues encountered
    """

    laws: list[CandidateLaw] = field(default_factory=list)
    rejections: list[tuple[dict[str, Any], str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ResponseParser:
    """Parses and validates LLM responses into CandidateLaw objects."""

    VALID_TEMPLATES = {t.value for t in Template}
    VALID_OPS = {"==", "!=", "<", "<=", ">", ">="}

    def __init__(self, strict: bool = False):
        """Initialize parser.

        Args:
            strict: If True, reject laws with any issues. If False, try to fix minor issues.
        """
        self.strict = strict

    def parse(self, response: str) -> ParseResult:
        """Parse an LLM response.

        Args:
            response: Raw response text from LLM

        Returns:
            ParseResult with parsed laws and rejections
        """
        result = ParseResult()

        # Extract JSON from response
        try:
            data = self._extract_json(response)
        except json.JSONDecodeError as e:
            result.warnings.append(f"JSON parse error: {e}")
            return result

        # Validate top-level structure
        if not isinstance(data, list):
            result.warnings.append("Response is not a JSON array")
            if isinstance(data, dict):
                data = [data]  # Wrap single object in list
            else:
                return result

        # Parse each law
        for i, law_data in enumerate(data):
            if not isinstance(law_data, dict):
                result.rejections.append(({"index": i}, "Law is not a JSON object"))
                continue

            try:
                law = self._parse_law(law_data, index=i)
                result.laws.append(law)
            except ParseError as e:
                result.rejections.append((law_data, str(e)))

        return result

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()

        # Check for markdown code blocks
        if "```" in text:
            # Try to extract from code block
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        return json.loads(text)

    def _parse_law(self, data: dict[str, Any], index: int) -> CandidateLaw:
        """Parse a single law from JSON data.

        Args:
            data: Law data dictionary
            index: Index in the response array

        Returns:
            Parsed CandidateLaw

        Raises:
            ParseError: If law cannot be parsed
        """
        # Required fields
        if "law_id" not in data:
            data["law_id"] = f"proposed_{index}"

        if "template" not in data:
            raise ParseError("Missing required field: template", data)

        if "forbidden" not in data:
            raise ParseError("Missing required field: forbidden", data)

        # Either claim_ast or claim is required
        if "claim_ast" not in data and "claim" not in data:
            raise ParseError("Missing required field: claim_ast (or claim)", data)

        # Validate template
        template_str = data["template"]
        if template_str not in self.VALID_TEMPLATES:
            raise ParseError(f"Invalid template: {template_str}", data)

        template = Template(template_str)

        # Parse quantifiers
        quantifiers = self._parse_quantifiers(data.get("quantifiers", {}))

        # Parse preconditions
        preconditions = self._parse_preconditions(data.get("preconditions", []))

        # Parse observables
        observables = self._parse_observables(data.get("observables", []))

        # Validate claim_ast if provided
        claim_ast = data.get("claim_ast")
        if claim_ast is not None:
            obs_names = {o.name for o in observables}
            is_valid, errors = validate_claim_ast(claim_ast, obs_names)
            if not is_valid:
                error_msgs = [f"{e.path}: {e.message}" for e in errors]
                raise ParseError(f"Invalid claim_ast: {'; '.join(error_msgs)}", data)

        # Parse proposed tests
        proposed_tests = self._parse_proposed_tests(data.get("proposed_tests", []))

        # Parse capability requirements
        capability_requirements = self._parse_capability_requirements(
            data.get("capability_requirements", {})
        )

        # Parse direction for monotone
        direction = None
        if "direction" in data:
            try:
                direction = MonotoneDirection(data["direction"])
            except ValueError:
                if not self.strict:
                    direction = MonotoneDirection.NON_INCREASING

        # Parse bound fields
        bound_value = data.get("bound_value")
        bound_op = None
        if "bound_op" in data:
            bound_op = self._parse_comparison_op(data["bound_op"])

        # Handle transform field - try to extract from claim if not present
        transform = data.get("transform")
        claim_str = data.get("claim", "")
        if transform is None and template == Template.SYMMETRY_COMMUTATION:
            transform = self._extract_transform_from_claim(claim_str)

        return CandidateLaw(
            schema_version=data.get("schema_version", "1.0.0"),
            law_id=data["law_id"],
            template=template,
            quantifiers=quantifiers,
            preconditions=preconditions,
            observables=observables,
            claim=claim_str if claim_str else "(see claim_ast)",
            forbidden=data["forbidden"],
            claim_ast=claim_ast,
            transform=transform,
            direction=direction,
            bound_value=bound_value,
            bound_op=bound_op,
            proposed_tests=proposed_tests,
            capability_requirements=capability_requirements,
        )

    def _parse_quantifiers(self, data: dict[str, Any]) -> Quantifiers:
        """Parse quantifiers from data."""
        return Quantifiers(
            T=data.get("T", 50),
            H=data.get("H"),
        )

    def _parse_preconditions(self, data: list[dict[str, Any]]) -> list[Precondition]:
        """Parse preconditions from data."""
        result = []
        for p in data:
            if not isinstance(p, dict):
                continue
            if "lhs" not in p or "op" not in p or "rhs" not in p:
                continue

            op = self._parse_comparison_op(p["op"])
            result.append(Precondition(
                lhs=str(p["lhs"]),
                op=op,
                rhs=p["rhs"],
            ))
        return result

    def _parse_observables(self, data: list[dict[str, Any]]) -> list[Observable]:
        """Parse observables from data."""
        result = []
        for o in data:
            if not isinstance(o, dict):
                continue
            if "name" not in o or "expr" not in o:
                continue

            result.append(Observable(
                name=str(o["name"]),
                expr=str(o["expr"]),
            ))
        return result

    def _parse_proposed_tests(self, data: list[dict[str, Any]]) -> list[ProposedTest]:
        """Parse proposed tests from data."""
        result = []
        for t in data:
            if not isinstance(t, dict):
                continue
            if "family" not in t:
                continue

            result.append(ProposedTest(
                family=str(t["family"]),
                params=t.get("params", {}),
            ))
        return result

    def _parse_capability_requirements(
        self, data: dict[str, Any]
    ) -> CapabilityRequirements:
        """Parse capability requirements from data."""
        return CapabilityRequirements(
            missing_observables=data.get("missing_observables", []),
            missing_transforms=data.get("missing_transforms", []),
            missing_generators=data.get("missing_generators", []),
        )

    def _parse_comparison_op(self, op_str: str) -> ComparisonOp:
        """Parse a comparison operator string."""
        mapping = {
            "==": ComparisonOp.EQ,
            "!=": ComparisonOp.NE,
            "<": ComparisonOp.LT,
            "<=": ComparisonOp.LE,
            ">": ComparisonOp.GT,
            ">=": ComparisonOp.GE,
        }
        return mapping.get(op_str, ComparisonOp.EQ)

    def _extract_transform_from_claim(self, claim: str) -> str | None:
        """Try to extract transform name from symmetry claim.

        Handles patterns like:
        - "commutes(transform='mirror_swap')"
        - "evolve(mirror_swap(S), T) == mirror_swap(evolve(S, T))"
        """
        known_transforms = ["mirror_swap", "shift_k", "mirror_only", "swap_only"]

        # Try to find a known transform in the claim
        claim_lower = claim.lower()
        for transform in known_transforms:
            if transform in claim_lower:
                return transform

        return None
