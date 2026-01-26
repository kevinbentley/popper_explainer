"""Parser for theorem generation responses."""

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.theorem.models import (
    LawSupport,
    MissingStructureType,
    SupportRole,
    Theorem,
    TheoremStatus,
    TypedMissingStructure,
)


@dataclass
class TheoremParseResult:
    """Result of parsing a theorem generation response."""

    theorems: list[Theorem]
    rejections: list[tuple[dict[str, Any], str]]  # (raw_data, rejection_reason)
    warnings: list[str] = field(default_factory=list)
    research_log: str | None = None  # LLM's theoretical notebook


class TheoremParser:
    """Parser for theorem generation LLM responses."""

    VALID_STATUSES = {"Established", "Conditional", "Conjectural"}
    VALID_ROLES = {r.value for r in SupportRole}
    VALID_MISSING_STRUCTURE_TYPES = {t.value for t in MissingStructureType}

    def parse(self, response: str) -> TheoremParseResult:
        """Parse an LLM response into theorems.

        Handles two formats:
        1. New format: {"research_log": "...", "theorems": [...]}
        2. Legacy format: [...] (just a list of theorems)

        Args:
            response: Raw LLM response text

        Returns:
            TheoremParseResult with parsed theorems, rejections, and research_log
        """
        theorems: list[Theorem] = []
        rejections: list[tuple[dict[str, Any], str]] = []
        warnings: list[str] = []
        research_log: str | None = None

        # Extract JSON from response
        try:
            data = self._extract_json(response)
        except ValueError as e:
            warnings.append(f"JSON extraction failed: {e}")
            return TheoremParseResult(
                theorems=[],
                rejections=[],
                warnings=warnings,
            )

        # Handle new format: object with research_log and theorems
        theorems_data: list[Any] = []
        if isinstance(data, dict):
            # Extract research_log if present
            if "research_log" in data:
                log_value = data["research_log"]
                if isinstance(log_value, str):
                    research_log = log_value
                else:
                    warnings.append("research_log should be a string")

            # Extract theorems array
            if "theorems" in data:
                theorems_list = data["theorems"]
                if isinstance(theorems_list, list):
                    theorems_data = theorems_list
                else:
                    warnings.append("theorems should be a list")
            else:
                # Maybe legacy format with theorem fields at top level
                if "name" in data or "claim" in data:
                    theorems_data = [data]
                else:
                    warnings.append("Response object missing 'theorems' field")
        elif isinstance(data, list):
            # Legacy format: just a list of theorems
            theorems_data = data
        else:
            warnings.append(f"Expected JSON object or array, got {type(data).__name__}")
            return TheoremParseResult(
                theorems=[],
                rejections=[],
                warnings=warnings,
            )

        # Parse each theorem
        for i, item in enumerate(theorems_data):
            if not isinstance(item, dict):
                rejections.append(
                    ({"index": i, "value": item}, f"Item {i}: not a dict")
                )
                continue

            try:
                theorem = self._parse_theorem(item, i)
                theorems.append(theorem)
            except ValueError as e:
                rejections.append((item, str(e)))

        return TheoremParseResult(
            theorems=theorems,
            rejections=rejections,
            warnings=warnings,
            research_log=research_log,
        )

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from response text.

        Handles cases where JSON is wrapped in markdown code blocks.
        Supports both object format (new) and array format (legacy).
        """
        text = text.strip()

        # Try to find JSON in code blocks (object or array)
        code_block_match = re.search(
            r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*?\])\s*```",
            text,
            re.DOTALL,
        )
        if code_block_match:
            text = code_block_match.group(1)

        # Try to find bare JSON object or array
        if not text.startswith("{") and not text.startswith("["):
            # Look for object start first (new format)
            obj_start = text.find("{")
            array_start = text.find("[")

            # Prefer object format if it comes first or array not found
            if obj_start != -1 and (array_start == -1 or obj_start < array_start):
                # Find matching end brace
                depth = 0
                for i, char in enumerate(text[obj_start:], obj_start):
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            text = text[obj_start : i + 1]
                            break
            elif array_start != -1:
                # Find matching end bracket
                depth = 0
                for i, char in enumerate(text[array_start:], array_start):
                    if char == "[":
                        depth += 1
                    elif char == "]":
                        depth -= 1
                        if depth == 0:
                            text = text[array_start : i + 1]
                            break

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

    def _parse_theorem(self, data: dict[str, Any], index: int) -> Theorem:
        """Parse a single theorem from a dict.

        Args:
            data: Raw theorem data
            index: Index in the response (for error messages)

        Returns:
            Parsed Theorem

        Raises:
            ValueError: If validation fails
        """
        # Required fields
        name = data.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"Theorem {index}: missing or invalid 'name'")

        status_str = data.get("status")
        if status_str not in self.VALID_STATUSES:
            raise ValueError(
                f"Theorem {index}: invalid status '{status_str}', "
                f"must be one of {self.VALID_STATUSES}"
            )
        status = TheoremStatus(status_str)

        claim = data.get("claim")
        if not claim or not isinstance(claim, str):
            raise ValueError(f"Theorem {index}: missing or invalid 'claim'")

        # Parse support
        support_data = data.get("support", [])
        if not isinstance(support_data, list):
            raise ValueError(f"Theorem {index}: 'support' must be a list")

        if len(support_data) < 2:
            raise ValueError(
                f"Theorem {index}: 'support' must have at least 2 law references"
            )

        support: list[LawSupport] = []
        for j, sup in enumerate(support_data):
            if not isinstance(sup, dict):
                raise ValueError(f"Theorem {index}: support[{j}] must be a dict")

            law_id = sup.get("law_id")
            if not law_id or not isinstance(law_id, str):
                raise ValueError(f"Theorem {index}: support[{j}] missing 'law_id'")

            role = sup.get("role", "confirms")
            if role not in self.VALID_ROLES:
                # Be lenient: normalize to closest match or default
                role = "confirms"

            support.append(LawSupport(law_id=law_id, role=SupportRole(role)))

        # Optional fields with validation
        failure_modes = data.get("failure_modes", [])
        if not isinstance(failure_modes, list):
            raise ValueError(f"Theorem {index}: 'failure_modes' must be a list")
        failure_modes = [str(fm) for fm in failure_modes]

        # Parse missing_structure - support both old and new formats
        missing_structure_raw = data.get("missing_structure", [])
        if not isinstance(missing_structure_raw, list):
            raise ValueError(f"Theorem {index}: 'missing_structure' must be a list")

        missing_structure: list[str] = []
        typed_missing_structure: list[TypedMissingStructure] = []

        for item in missing_structure_raw:
            if isinstance(item, str):
                # Old format: plain string
                missing_structure.append(item)
                typed_missing_structure.append(
                    TypedMissingStructure.from_string(item)
                )
            elif isinstance(item, dict):
                # New format: typed object
                type_str = item.get("type", "DEFINITION_MISSING")
                target = item.get("target", "")
                note = item.get("note", "")

                # Validate type
                if type_str not in self.VALID_MISSING_STRUCTURE_TYPES:
                    # Auto-classify from target text
                    type_str = MissingStructureType.classify(target).value

                typed_ms = TypedMissingStructure(
                    type=MissingStructureType(type_str),
                    target=target,
                    note=note,
                )
                typed_missing_structure.append(typed_ms)
                # Also store as string for backward compat
                missing_structure.append(target)
            else:
                # Unknown format, convert to string
                missing_structure.append(str(item))
                typed_missing_structure.append(
                    TypedMissingStructure.from_string(str(item))
                )

        # Generate theorem ID
        theorem_id = data.get("theorem_id") or f"thm_{uuid.uuid4().hex[:12]}"

        return Theorem(
            theorem_id=theorem_id,
            name=name,
            status=status,
            claim=claim,
            support=support,
            failure_modes=failure_modes,
            missing_structure=missing_structure,
            typed_missing_structure=typed_missing_structure,
        )
