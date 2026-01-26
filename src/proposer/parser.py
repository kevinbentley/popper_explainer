"""Response parser for LLM law proposals."""

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from src.proposer.scrambler import SymbolScrambler


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
        research_log: LLM's research notes for continuity across iterations
    """

    laws: list[CandidateLaw] = field(default_factory=list)
    rejections: list[tuple[dict[str, Any], str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    research_log: str | None = None


class ResponseParser:
    """Parses and validates LLM responses into CandidateLaw objects."""

    VALID_TEMPLATES = {t.value for t in Template}
    VALID_OPS = {"==", "!=", "<", "<=", ">", ">="}

    def __init__(self, strict: bool = False, scrambler: "SymbolScrambler | None" = None):
        """Initialize parser.

        Args:
            strict: If True, reject laws with any issues. If False, try to fix minor issues.
            scrambler: Symbol scrambler for translating abstract→physical symbols.
                      If provided, all parsed laws will have their symbols translated.
        """
        self.strict = strict
        self._warnings: list[str] = []  # Accumulated warnings from parsing

        # Initialize scrambler (use default if not provided)
        if scrambler is None:
            from src.proposer.scrambler import get_default_scrambler
            self._scrambler = get_default_scrambler()
        else:
            self._scrambler = scrambler

    def parse(self, response: str) -> ParseResult:
        """Parse an LLM response.

        Handles two formats:
        1. New format: {"research_log": "...", "candidate_laws": [...]}
        2. Legacy format: [...] (just a list of laws)

        Args:
            response: Raw response text from LLM

        Returns:
            ParseResult with parsed laws, rejections, and research_log
        """
        self._warnings = []  # Reset warnings for each parse call
        result = ParseResult()

        # Extract JSON from response
        try:
            data = self._extract_json(response)
        except json.JSONDecodeError as e:
            result.warnings.append(f"JSON parse error: {e}")
            return result

        # Handle new format: object with research_log and candidate_laws
        laws_data: list[Any] = []
        if isinstance(data, dict):
            # Extract research_log if present
            if "research_log" in data:
                research_log = data["research_log"]
                if isinstance(research_log, str):
                    result.research_log = research_log
                else:
                    result.warnings.append("research_log should be a string")

            # Extract candidate_laws
            if "candidate_laws" in data:
                candidate_laws = data["candidate_laws"]
                if isinstance(candidate_laws, list):
                    laws_data = candidate_laws
                else:
                    result.warnings.append("candidate_laws should be a list")
                    laws_data = []
            else:
                # Maybe it's a single law object (legacy format)
                if "law_id" in data or "template" in data:
                    laws_data = [data]
                else:
                    result.warnings.append("Response object missing candidate_laws field")
        elif isinstance(data, list):
            # Legacy format: just a list of laws
            laws_data = data
        else:
            result.warnings.append("Response is neither a JSON object nor array")
            return result

        # Parse each law
        for i, law_data in enumerate(laws_data):
            if not isinstance(law_data, dict):
                result.rejections.append(({"index": i}, "Law is not a JSON object"))
                continue

            try:
                law = self._parse_law(law_data, index=i)
                result.laws.append(law)
            except ParseError as e:
                result.rejections.append((law_data, str(e)))

        # Add accumulated warnings from individual law parsing
        result.warnings.extend(self._warnings)
        return result

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from text, handling markdown code blocks and common LLM errors."""
        text = text.strip()

        # Check for markdown code blocks
        if "```" in text:
            # Try to extract from code block
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Sanitize common LLM JSON errors
        text = self._sanitize_json(text)

        return json.loads(text)

    def _sanitize_json(self, text: str) -> str:
        """Sanitize common LLM JSON formatting errors.

        Handles:
        - Stray backticks from markdown formatting
        - Trailing commas before closing brackets
        - JavaScript-style comments
        """
        # Remove stray backticks (common markdown artifact)
        # But preserve backticks that are part of string content
        # Strategy: remove backticks that appear outside of quoted strings
        result = []
        in_string = False
        escape_next = False
        i = 0

        while i < len(text):
            char = text[i]

            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue

            if char == '\\' and in_string:
                result.append(char)
                escape_next = True
                i += 1
                continue

            if char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
                continue

            if char == '`' and not in_string:
                # Skip stray backticks outside strings
                i += 1
                continue

            result.append(char)
            i += 1

        text = ''.join(result)

        # Remove trailing commas before ] or }
        # Pattern: comma followed by optional whitespace and closing bracket
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        # Remove JavaScript-style comments (// and /* */)
        # Single-line comments
        text = re.sub(r'//[^\n]*', '', text)
        # Multi-line comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

        return text.strip()

    def _parse_law(self, data: dict[str, Any], index: int) -> CandidateLaw:
        """Parse a single law from JSON data.

        Args:
            data: Law data dictionary (may use abstract symbols from scrambler)
            index: Index in the response array

        Returns:
            Parsed CandidateLaw (with physical symbols for harness testing)

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

        # Translate abstract symbols to physical symbols
        # The LLM uses abstract symbols (_, A, B, K) to prevent inferring physics
        # The harness needs physical symbols (., >, <, X) for simulation
        data = self._scrambler.translate_law_to_physical(data)

        # Validate template
        template_str = data["template"]
        if template_str not in self.VALID_TEMPLATES:
            raise ParseError(f"Invalid template: {template_str}", data)

        template = Template(template_str)

        # Parse quantifiers
        # Use `or {}` to handle explicit null values from LLM (dict.get only
        # returns the default when the key is absent, not when it's null)
        quantifiers = self._parse_quantifiers(data.get("quantifiers") or {})

        # Parse preconditions
        preconditions = self._parse_preconditions(data.get("preconditions") or [])

        # Parse observables
        observables = self._parse_observables(data.get("observables") or [])

        # Validate claim_ast if provided
        claim_ast = data.get("claim_ast")
        if claim_ast is not None:
            obs_names = {o.name for o in observables}
            is_valid, errors = validate_claim_ast(claim_ast, obs_names)
            if not is_valid:
                error_msgs = [f"{e.path}: {e.message}" for e in errors]
                raise ParseError(f"Invalid claim_ast: {'; '.join(error_msgs)}", data)

        # Parse proposed tests
        proposed_tests = self._parse_proposed_tests(data.get("proposed_tests") or [])

        # Parse capability requirements
        capability_requirements = self._parse_capability_requirements(
            data.get("capability_requirements") or {}
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
        transform = self._normalize_transform(data.get("transform"))
        claim_str = data.get("claim", "")
        if transform is None and template == Template.SYMMETRY_COMMUTATION:
            transform = self._extract_transform_from_claim(claim_str)

        # Handle local_transition fields
        trigger_symbol = data.get("trigger_symbol")
        result_symbol = data.get("result_symbol")
        neighbor_pattern = data.get("neighbor_pattern")
        required_parity = data.get("required_parity")
        result_op = None
        if "result_op" in data:
            result_op = self._parse_comparison_op(data["result_op"])

        # Validate neighbor_pattern if provided
        if neighbor_pattern is not None:
            if not isinstance(neighbor_pattern, str) or len(neighbor_pattern) != 3:
                self._warnings.append(
                    f"Invalid neighbor_pattern '{neighbor_pattern}' - must be 3 characters"
                )
                neighbor_pattern = None
            # Accept both physical (.><X) and abstract (WABK) symbols
            elif not all(c in '.><XWABK' for c in neighbor_pattern):
                self._warnings.append(
                    f"Invalid neighbor_pattern '{neighbor_pattern}' - must contain only valid symbols"
                )
                neighbor_pattern = None

        # Validate required_parity if provided
        if required_parity is not None:
            if not isinstance(required_parity, int) or required_parity not in (0, 1):
                self._warnings.append(
                    f"Invalid required_parity '{required_parity}' - must be 0 (even) or 1 (odd)"
                )
                required_parity = None

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
            trigger_symbol=trigger_symbol,
            neighbor_pattern=neighbor_pattern,
            required_parity=required_parity,
            result_op=result_op,
            result_symbol=result_symbol,
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
        """Parse preconditions from data.

        Handles both simple format and AST format for lhs/rhs:
        - Simple: {"lhs": "count('>')", "op": ">", "rhs": 0}
        - AST: {"lhs": {"obs": "R"}, "op": ">", "rhs": {"const": 0}}
        """
        result = []
        for p in data:
            if not isinstance(p, dict):
                continue
            if "lhs" not in p or "op" not in p or "rhs" not in p:
                continue

            op = self._parse_comparison_op(p["op"])

            # Handle lhs that might be in AST format
            lhs = p["lhs"]
            if isinstance(lhs, dict):
                if "obs" in lhs:
                    lhs = lhs["obs"]
                else:
                    lhs = self._ast_expr_to_string(lhs)
            lhs = str(lhs)

            # Handle rhs that might be in AST format like {"const": 0}
            rhs = p["rhs"]
            if isinstance(rhs, dict):
                if "const" in rhs:
                    rhs = rhs["const"]
                elif "value" in rhs:
                    rhs = rhs["value"]
                elif "obs" in rhs:
                    rhs = rhs["obs"]
                else:
                    # Try to convert AST to string for observable references
                    rhs = self._ast_expr_to_string(rhs)

            result.append(Precondition(
                lhs=lhs,
                op=op,
                rhs=rhs,
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

            # Handle expr that might be a dict (AST format) instead of a string
            expr = o["expr"]
            if isinstance(expr, dict):
                expr = self._ast_expr_to_string(expr)
            else:
                expr = str(expr)

            result.append(Observable(
                name=str(o["name"]),
                expr=expr,
            ))
        return result

    def _ast_expr_to_string(self, ast: dict[str, Any]) -> str:
        """Convert an AST-style expression dict to a string expression.

        Handles common patterns from LLM responses like:
        - {"op": "count", "symbol": ">"} -> "count('>')"
        - {"op": "add", "left": {...}, "right": {...}} -> "(left) + (right)"
        - {"op": "sub", "left": {...}, "right": {...}} -> "(left) - (right)"
        - {"op": "mul", "left": {...}, "right": {...}} -> "(left) * (right)"
        - {"function": "count", "args": [">"]} -> "count('>')"
        - {"name": "TotalParticles"} -> "TotalParticles"
        """
        if not isinstance(ast, dict):
            return str(ast)

        # Handle simple name reference
        if "name" in ast and len(ast) == 1:
            return str(ast["name"])

        # Handle function call format: {"function": "count", "args": [">"]}
        if "function" in ast:
            func = ast["function"]
            args = ast.get("args", [])
            if func == "count" and args:
                symbol = args[0] if args else ">"
                return f"count('{symbol}')"
            # Generic function call
            arg_strs = [self._ast_expr_to_string(a) if isinstance(a, dict) else repr(a) for a in args]
            return f"{func}({', '.join(arg_strs)})"

        # Handle operator format: {"op": "count", "symbol": ">"}
        op = ast.get("op")

        if op == "count":
            symbol = ast.get("symbol", ast.get("arg", ">"))
            return f"count('{symbol}')"

        if op == "grid_length":
            return "grid_length"

        if op in ("add", "+"):
            left = self._ast_expr_to_string(ast.get("left", ast.get("operands", [{}])[0]))
            right = self._ast_expr_to_string(ast.get("right", ast.get("operands", [{}])[1] if len(ast.get("operands", [])) > 1 else {}))
            return f"({left}) + ({right})"

        if op in ("sub", "-"):
            left = self._ast_expr_to_string(ast.get("left", {}))
            right = self._ast_expr_to_string(ast.get("right", {}))
            return f"({left}) - ({right})"

        if op in ("mul", "*"):
            left = self._ast_expr_to_string(ast.get("left", {}))
            right = self._ast_expr_to_string(ast.get("right", {}))
            return f"({left}) * ({right})"

        if op in ("div", "/"):
            left = self._ast_expr_to_string(ast.get("left", {}))
            right = self._ast_expr_to_string(ast.get("right", {}))
            return f"({left}) / ({right})"

        # Handle literal/constant
        if op == "literal" or op == "const":
            return str(ast.get("value", 0))

        # Handle operands list format: {"op": "+", "operands": [...]}
        if "operands" in ast and op:
            operands = [self._ast_expr_to_string(o) for o in ast["operands"]]
            if op in ("+", "add"):
                return " + ".join(f"({o})" for o in operands)
            if op in ("-", "sub"):
                return " - ".join(f"({o})" for o in operands)
            if op in ("*", "mul"):
                return " * ".join(f"({o})" for o in operands)

        # Fallback: try to make something reasonable
        # If it has 'expr' field, use that
        if "expr" in ast:
            return self._ast_expr_to_string(ast["expr"]) if isinstance(ast["expr"], dict) else str(ast["expr"])

        # Last resort: return a placeholder that won't crash but will fail validation
        return f"UNPARSED_AST({ast})"

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

    def _normalize_transform(self, transform: Any) -> str | None:
        """Normalize a transform value to a string.

        Handles multiple input formats:
        - String: "mirror_swap" -> "mirror_swap"
        - String with param: "shift_1" -> "shift_1"
        - Dict: {"name": "shift_k", "params": {"k": 1}} -> "shift_1"
        - Dict: {"name": "mirror_swap"} -> "mirror_swap"

        For shift_k transforms with k parameter, the k value is encoded
        in the transform name as "shift_N" where N is the k value.
        """
        if transform is None:
            return None

        if isinstance(transform, str):
            return transform

        if isinstance(transform, dict):
            name = transform.get("name")
            if not name:
                return None

            # Handle parameterized transforms
            params = transform.get("params", {})

            if name == "shift_k" and "k" in params:
                # Encode k value in the transform name
                k_value = params["k"]
                if isinstance(k_value, int):
                    return f"shift_{k_value}"
                else:
                    # Try to parse if it's a string
                    try:
                        return f"shift_{int(k_value)}"
                    except (ValueError, TypeError):
                        return "shift_k"  # Fall back to generic

            # For other transforms, just return the name
            return str(name)

        # Unknown type, try to convert to string
        return str(transform) if transform else None

    def _extract_transform_from_claim(self, claim: str) -> str | None:
        """Try to extract transform name from symmetry claim.

        Handles patterns like:
        - "commutes(transform='mirror_swap')"
        - "evolve(mirror_swap(S), T) == mirror_swap(evolve(S, T))"
        - "shift_k(S, k=2)"
        """
        known_transforms = ["mirror_swap", "shift_k", "mirror_only", "swap_only"]

        # Try to find a known transform in the claim
        claim_lower = claim.lower()

        # Check for shift_k with explicit k value like "shift_k(S, k=2)" or "shift_k(k=1)"
        shift_match = re.search(r'shift_k\s*\([^)]*k\s*=\s*(\d+)', claim_lower)
        if shift_match:
            k_value = int(shift_match.group(1))
            return f"shift_{k_value}"

        # Check for "shift_N" pattern (e.g., "shift_1", "shift_2")
        shift_n_match = re.search(r'shift_(\d+)', claim_lower)
        if shift_n_match:
            return f"shift_{shift_n_match.group(1)}"

        for transform in known_transforms:
            if transform in claim_lower:
                return transform

        return None
