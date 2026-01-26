"""Tool for evaluating candidate laws against the simulator."""

import hashlib
import json
import logging
from typing import Any

from src.ahc.db.models import LawEvaluationRecord
from src.ahc.db.repo import AHCRepository
from src.ahc.tools.base import BaseTool, ToolParameter, ToolResult
from src.claims.schema import CandidateLaw, Template, Quantifiers
from src.harness.config import HarnessConfig
from src.harness.harness import Harness

logger = logging.getLogger(__name__)


class EvaluateLawsTool(BaseTool):
    """Tool for evaluating candidate laws using the test harness.

    Wraps the existing Harness.evaluate() method to provide law evaluation
    capabilities to the AHC agent.
    """

    MIN_BATCH_SIZE = 5
    MAX_BATCH_SIZE = 15

    def __init__(
        self,
        repo: AHCRepository | None = None,
        config: HarnessConfig | None = None,
        max_grid_length: int = 50,
    ):
        """Initialize the tool.

        Args:
            repo: Optional database repository for persisting evaluations
            config: Optional harness configuration
            max_grid_length: Maximum grid length for tautology detection
        """
        self._repo = repo
        self._config = config or HarnessConfig()
        self._harness = Harness(config=self._config)
        self._session_id: int | None = None
        self._turn_number: int = 0
        self._max_grid_length = max_grid_length

    def set_context(self, session_id: int, turn_number: int = 0) -> None:
        """Set the current session context for logging.

        Args:
            session_id: Database session ID
            turn_number: Current turn number (default: 0)
        """
        self._session_id = session_id
        self._turn_number = turn_number

    def set_turn(self, turn_number: int) -> None:
        """Update the current turn number.

        Args:
            turn_number: Current turn number
        """
        self._turn_number = turn_number

    @property
    def name(self) -> str:
        return "evaluate_laws"

    @property
    def description(self) -> str:
        return """IMPORTANT: You MUST submit between 5 and 15 laws per call. Single-law submissions will be REJECTED.
Trivially true laws (tautologies) will be REJECTED. Counts are always in [0, grid_length].

Evaluate candidate laws against the physics simulator.

Supports ALL law templates: local_transition, invariant, monotone, implication_step, implication_state, eventually, symmetry_commutation, bound.

For local_transition: use trigger_symbol, result_op, result_symbol (and optionally left_neighbor/right_neighbor).
For all other templates: use observables (array of {name, expr}) and claim_ast (JSON expression tree). Some templates need additional fields: direction (monotone), bound_value/bound_op (bound), transform (symmetry_commutation).

Returns PASS, FAIL, or UNKNOWN with evidence.
For FAIL verdicts, the counterexample includes:
- initial_state: the starting configuration
- t_fail: the time step where the violation occurred
- states: up to 5 consecutive state strings centered on t_fail
- states_t_offset: time offset of states[0] relative to t_fail
- fail_position: the 0-based cell index where the violation occurred (local_transition)
- actual_result: what the cell actually became (local_transition)"""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="laws",
                type="array",
                description="""Array of candidate laws to evaluate (MINIMUM 5, MAXIMUM 15). Template determines required fields:
- local_transition: trigger_symbol, result_op, result_symbol (+ optional left_neighbor/right_neighbor)
- invariant/monotone/implication_step/implication_state/eventually/bound: observables + claim_ast
- monotone: also requires direction ("<=" or ">=")
- bound: also requires bound_value and bound_op
- symmetry_commutation: requires transform
- eventually: quantifiers must include H (horizon)""",
                required=True,
                items={
                    "type": "object",
                    "properties": {
                        "law_id": {"type": "string", "description": "Unique identifier for the law"},
                        "template": {
                            "type": "string",
                            "description": "Law template type",
                            "enum": ["local_transition", "invariant", "monotone", "implication_step", "implication_state", "eventually", "symmetry_commutation", "bound"],
                        },
                        "claim": {"type": "string", "description": "Human-readable law statement"},
                        "forbidden": {"type": "string", "description": "What constitutes a counterexample"},
                        "quantifiers": {
                            "type": "object",
                            "description": "Time bounds (default: T=50). For eventually, include H.",
                            "properties": {
                                "T": {"type": "integer", "description": "Time horizon"},
                                "H": {"type": "integer", "description": "Eventuality horizon (eventually template)"},
                            },
                        },
                        # === Fields for AST-based templates (invariant, monotone, implication, bound, eventually) ===
                        "observables": {
                            "type": "array",
                            "description": "Observable definitions. Each is {name, expr} where expr uses count('A'), transition_indicator, etc.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Observable name referenced in claim_ast"},
                                    "expr": {"type": "string", "description": "Expression: count('A'), transition_indicator, count('A')+count('B'), etc."},
                                },
                            },
                        },
                        "claim_ast": {
                            "type": "object",
                            "description": "Structured claim as JSON AST. Nodes: {const: N}, {var: 't'}, {t_plus_1: true}, {obs: 'name', t: <time>}, {op: '+', lhs: <n>, rhs: <n>}",
                        },
                        # === local_transition specific fields ===
                        "trigger_symbol": {
                            "type": "string",
                            "description": "For local_transition: cell symbol at time t (W, A, B, or K)",
                        },
                        "result_op": {
                            "type": "string",
                            "description": "For local_transition: comparison operator (== or !=)",
                            "enum": ["==", "!="],
                        },
                        "result_symbol": {
                            "type": "string",
                            "description": "For local_transition: expected symbol at t+1 (W, A, B, or K)",
                        },
                        "left_neighbor": {
                            "type": "string",
                            "description": "For local_transition: symbol to LEFT of trigger (W, A, B, or K)",
                        },
                        "right_neighbor": {
                            "type": "string",
                            "description": "For local_transition: symbol to RIGHT of trigger (W, A, B, or K)",
                        },
                        "neighbor_pattern": {
                            "type": "string",
                            "description": "DEPRECATED: Use left_neighbor and right_neighbor instead.",
                        },
                        "required_parity": {
                            "type": "string",
                            "description": "For local_transition: '0' = even indices only, '1' = odd indices only. Tests parity-dependent rules.",
                            "enum": ["0", "1"],
                        },
                        # === monotone specific ===
                        "direction": {
                            "type": "string",
                            "description": "For monotone: '<=' (non-increasing) or '>=' (non-decreasing)",
                            "enum": ["<=", ">="],
                        },
                        # === bound specific ===
                        "bound_value": {
                            "type": "integer",
                            "description": "For bound: the bound value k",
                        },
                        "bound_op": {
                            "type": "string",
                            "description": "For bound: comparison operator (<=, >=, <, >, ==, !=)",
                            "enum": ["<=", ">=", "<", ">", "==", "!="],
                        },
                        # === symmetry specific ===
                        "transform": {
                            "type": "string",
                            "description": "For symmetry_commutation: transform name",
                            "enum": ["mirror_swap", "shift_k", "mirror_only", "swap_only"],
                        },
                    },
                    "required": ["law_id", "template"],
                },
            ),
        ]

    def execute(self, laws: list[dict[str, Any]], **kwargs) -> ToolResult:
        """Execute law evaluation.

        Args:
            laws: List of law dictionaries to evaluate

        Returns:
            ToolResult with evaluation results for each law
        """
        if not laws:
            return ToolResult.fail("No laws provided for evaluation")

        if len(laws) < self.MIN_BATCH_SIZE:
            return ToolResult.fail(
                f"BATCH SIZE TOO SMALL: You submitted {len(laws)} law(s) but the minimum is "
                f"{self.MIN_BATCH_SIZE}. Submit {self.MIN_BATCH_SIZE}-{self.MAX_BATCH_SIZE} diverse "
                f"hypotheses (different templates, different observables) together."
            )
        if len(laws) > self.MAX_BATCH_SIZE:
            return ToolResult.fail(
                f"BATCH SIZE TOO LARGE: {len(laws)} laws exceeds the maximum of {self.MAX_BATCH_SIZE}."
            )

        results = []
        for law_dict in laws:
            try:
                result = self._evaluate_single_law(law_dict)
                results.append(result)
            except Exception as e:
                logger.exception(f"Failed to evaluate law: {law_dict.get('law_id', 'unknown')}")
                # Scramble law_id even in error case to prevent information leakage
                original_id = law_dict.get("law_id", "unknown")
                scrambled_id = self._scramble_law_id(original_id) if original_id != "unknown" else "unknown"
                results.append({
                    "law_id": scrambled_id,
                    "status": "ERROR",
                    "error": str(e),
                })

        # Summarize results
        summary = {
            "total": len(results),
            "passed": sum(1 for r in results if r.get("status") == "PASS"),
            "failed": sum(1 for r in results if r.get("status") == "FAIL"),
            "unknown": sum(1 for r in results if r.get("status") == "UNKNOWN"),
            "errors": sum(1 for r in results if r.get("status") == "ERROR"),
        }

        return ToolResult.ok(
            data={
                "results": results,
                "summary": summary,
            }
        )

    def _scramble_law_id(self, law_id: str) -> str:
        """Scramble a law_id to prevent information leakage.

        The LLM might encode semantic information in law_ids like "A_persists"
        or "empty_becomes_K". We hash them to prevent this leakage while
        still allowing the LLM to track which law is which.

        Args:
            law_id: Original law identifier

        Returns:
            Scrambled identifier like "law_a3f2b1"
        """
        hash_bytes = hashlib.sha256(law_id.encode()).hexdigest()[:6]
        return f"law_{hash_bytes}"

    def _evaluate_single_law(self, law_dict: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a single law.

        Args:
            law_dict: Law specification dictionary

        Returns:
            Evaluation result dictionary
        """
        # Parse the law dictionary into a CandidateLaw
        law = self._parse_law(law_dict)

        # Check for tautological claims before running the harness
        tautology_reason = self._detect_tautology(law)
        if tautology_reason:
            scrambled_id = self._scramble_law_id(law.law_id)
            return {
                "law_id": scrambled_id,
                "status": "UNKNOWN",
                "reason_code": "tautological_claim",
                "rejected_reason": tautology_reason,
            }

        # Run evaluation
        verdict = self._harness.evaluate(law)

        # Scramble the law_id to prevent semantic leakage
        scrambled_id = self._scramble_law_id(law.law_id)

        # Build result with echo of what was tested (helps LLM catch its own errors)
        # Convert back to abstract symbols for the echo
        from src.proposer.scrambler import get_default_scrambler
        scrambler = get_default_scrambler()

        # Build human-readable summary of what was tested
        if law.template == Template.LOCAL_TRANSITION:
            trigger_abstract = scrambler.to_abstract(law.trigger_symbol) if law.trigger_symbol else "?"
            result_abstract = scrambler.to_abstract(law.result_symbol) if law.result_symbol else "?"
            op_str = law.result_op.value if hasattr(law.result_op, 'value') else str(law.result_op)

            if law.neighbor_pattern:
                pattern_abstract = scrambler.to_abstract(law.neighbor_pattern)
                tested = f"trigger={trigger_abstract} with left={pattern_abstract[0]} right={pattern_abstract[2]} -> {op_str} {result_abstract}"
            else:
                tested = f"trigger={trigger_abstract} (any neighbors) -> {op_str} {result_abstract}"
        else:
            # Generic echo for non-local-transition templates
            template_str = law.template.value if hasattr(law.template, 'value') else str(law.template)
            obs_str = ""
            if law.observables:
                obs_names = [o.name for o in law.observables]
                obs_str = f" observables=[{', '.join(obs_names)}]"
            extra = ""
            if law.transform:
                extra += f" transform={law.transform}"
            if law.direction:
                dir_str = law.direction.value if hasattr(law.direction, 'value') else str(law.direction)
                extra += f" direction={dir_str}"
            if law.bound_value is not None:
                op_str = law.bound_op.value if law.bound_op and hasattr(law.bound_op, 'value') else str(law.bound_op)
                extra += f" bound={op_str}{law.bound_value}"
            claim_short = (law.claim[:80] + "...") if len(law.claim) > 80 else law.claim
            tested = f"[{template_str}]{obs_str}{extra}: {claim_short}"

        result = {
            "law_id": scrambled_id,
            "tested": tested,  # Echo what was actually tested
            "status": verdict.status,
        }

        if verdict.reason_code:
            result["reason_code"] = verdict.reason_code.value

        # NOTE: We intentionally exclude all notes from the output.
        # Notes can leak information about universe mechanics or harness internals.
        # The Popperian approach requires the agent to learn solely through
        # counterexamples, not through hints or explanations.

        if verdict.power_metrics:
            result["power_metrics"] = {
                "cases_attempted": verdict.power_metrics.cases_attempted,
                "cases_used": verdict.power_metrics.cases_used,
                "coverage_score": verdict.power_metrics.coverage_score,
            }

        if verdict.counterexample:
            cx = verdict.counterexample
            # Provide enough info for Popperian learning:
            # - initial_state: starting configuration
            # - t_fail: when the violation occurred
            # - states: up to 5 consecutive states centered on t_fail
            # - states_t_offset: time offset of states[0] relative to t_fail
            # - fail_position: which cell index (0-based) violated the law
            # - actual_result: what the cell actually became at t+1
            #
            # We show the full trajectory excerpt (~5 states) so the agent can
            # see the evolution context around the failure point.
            cx_data: dict[str, Any] = {
                "initial_state": cx.initial_state,
                "t_fail": cx.t_fail,
            }

            # Include full trajectory excerpt as states list
            if cx.trajectory_excerpt:
                # The excerpt starts at max(0, t_fail - 2)
                excerpt_start = max(0, cx.t_fail - 2)
                states_t_offset = excerpt_start - cx.t_fail  # Negative or zero

                cx_data["states"] = list(cx.trajectory_excerpt)
                cx_data["states_t_offset"] = states_t_offset

            # Extract cell-level failure details from violation data
            if cx.observables_at_fail and isinstance(cx.observables_at_fail, dict):
                pos = cx.observables_at_fail.get("position")
                if pos is not None:
                    cx_data["fail_position"] = pos
                actual = cx.observables_at_fail.get("actual_symbol")
                if actual is not None:
                    cx_data["actual_result"] = actual

            result["counterexample"] = cx_data

        # Persist to database if repo available
        if self._repo and self._session_id is not None:
            self._persist_evaluation(law, verdict, result)

        return result

    def _detect_tautology(self, law: CandidateLaw) -> str | None:
        """Detect if a law is tautologically true (unfalsifiable).

        Currently checks bound-template laws where count-based observables
        have trivially satisfiable bounds (e.g., count >= 0).

        Args:
            law: Parsed CandidateLaw

        Returns:
            Human-readable rejection reason if tautological, None otherwise
        """
        if law.template != Template.BOUND:
            return None

        # Need bound_op and bound_value
        if law.bound_op is None or law.bound_value is None:
            return None

        # Check each observable for count-type expressions
        count_prefixes = ("count(", "count_even(", "count_odd(", "adjacent_pairs(", "count_pattern(", "count_at_parity(")
        if not law.observables:
            return None

        for obs in law.observables:
            expr = obs.expr if hasattr(obs, 'expr') else str(obs)
            is_count_type = any(expr.strip().startswith(p) for p in count_prefixes)
            if not is_count_type:
                continue

            result = self._check_count_bound_tautology(
                str(law.bound_op.value) if hasattr(law.bound_op, 'value') else str(law.bound_op),
                law.bound_value,
                obs.name if hasattr(obs, 'name') else str(obs),
            )
            if result:
                return result

        return None

    def _check_count_bound_tautology(self, op: str, bound_val: int, obs_name: str) -> str | None:
        """Check if a count-based bound is tautologically true.

        Counts are always in [0, grid_length]. Bounds that are trivially
        satisfied for ALL possible count values are tautologies.

        Args:
            op: Comparison operator string (<=, >=, <, >, ==, !=)
            bound_val: The bound value
            obs_name: Observable name for error message

        Returns:
            Human-readable reason if tautological, None otherwise
        """
        gl = self._max_grid_length

        # count >= 0 or count >= negative → always true
        if op == ">=" and bound_val <= 0:
            return (
                f"Tautology: {obs_name} >= {bound_val} is always true. "
                f"Counts are non-negative integers (minimum 0). "
                f"Propose a meaningful lower bound (e.g., >= 1)."
            )

        # count <= grid_length or count <= huge → always true
        if op == "<=" and bound_val >= gl:
            return (
                f"Tautology: {obs_name} <= {bound_val} is always true. "
                f"Counts cannot exceed grid_length ({gl}). "
                f"Propose a tighter upper bound."
            )

        # count < 1 (i.e., count < non-positive+1 is not useful, but count < 0 is impossible to satisfy differently)
        # count > grid_length is impossible → but that's vacuous, not tautological
        # Focus on the clear cases:

        # count > negative → always true (count >= 0 > negative)
        if op == ">" and bound_val < 0:
            return (
                f"Tautology: {obs_name} > {bound_val} is always true. "
                f"Counts are non-negative integers. "
                f"Propose a meaningful bound."
            )

        # count < grid_length+1 or larger → always true
        if op == "<" and bound_val > gl:
            return (
                f"Tautology: {obs_name} < {bound_val} is always true. "
                f"Counts cannot exceed grid_length ({gl}). "
                f"Propose a tighter bound."
            )

        # count != negative → always true
        if op == "!=" and bound_val < 0:
            return (
                f"Tautology: {obs_name} != {bound_val} is always true. "
                f"Counts are non-negative integers and can never equal {bound_val}. "
                f"Propose a falsifiable inequality."
            )

        # count != huge → always true (can never reach it)
        if op == "!=" and bound_val > gl:
            return (
                f"Tautology: {obs_name} != {bound_val} is always true. "
                f"Counts cannot exceed grid_length ({gl}) so they can never equal {bound_val}. "
                f"Propose a falsifiable inequality."
            )

        return None

    def _parse_law(self, law_dict: dict[str, Any]) -> CandidateLaw:
        """Parse a law dictionary into a CandidateLaw object.

        Handles both full CandidateLaw format and simplified formats.
        """
        from src.claims.schema import ComparisonOp, MonotoneDirection

        # Ensure required fields
        if "law_id" not in law_dict:
            law_dict["law_id"] = f"law_{hash(json.dumps(law_dict, sort_keys=True)) % 10000}"

        if "template" not in law_dict:
            law_dict["template"] = "invariant"  # Default template

        if "claim" not in law_dict:
            law_dict["claim"] = law_dict.get("description", "")

        if "forbidden" not in law_dict:
            law_dict["forbidden"] = "Counterexample found"

        if "quantifiers" not in law_dict:
            law_dict["quantifiers"] = {"T": 50}

        # Parse template string to enum
        template_str = law_dict.get("template")
        if isinstance(template_str, str):
            try:
                law_dict["template"] = Template(template_str)
            except ValueError:
                logger.warning(f"Invalid template string '{template_str}', defaulting to INVARIANT")
                law_dict["template"] = Template.INVARIANT

        # Parse direction string to MonotoneDirection enum for monotone template
        if law_dict.get("template") == Template.MONOTONE:
            if isinstance(law_dict.get("direction"), str):
                try:
                    law_dict["direction"] = MonotoneDirection(law_dict["direction"])
                except ValueError:
                    logger.warning(f"Invalid direction, defaulting to <=")
                    law_dict["direction"] = MonotoneDirection.NON_INCREASING

        # Parse bound_op string to ComparisonOp enum for bound template
        if law_dict.get("template") == Template.BOUND:
            if isinstance(law_dict.get("bound_op"), str):
                try:
                    law_dict["bound_op"] = ComparisonOp(law_dict["bound_op"])
                except ValueError:
                    logger.warning(f"Invalid bound_op, defaulting to <=")
                    law_dict["bound_op"] = ComparisonOp.LE

        # Parse result_op string to ComparisonOp enum for local_transition
        if law_dict.get("template") == Template.LOCAL_TRANSITION:
            if isinstance(law_dict.get("result_op"), str):
                try:
                    law_dict["result_op"] = ComparisonOp(law_dict["result_op"])
                except ValueError:
                    logger.warning(f"Invalid result_op, defaulting to ==")
                    law_dict["result_op"] = ComparisonOp.EQ

            # Coerce required_parity to int (Gemini may return as string)
            if "required_parity" in law_dict and law_dict["required_parity"] is not None:
                try:
                    law_dict["required_parity"] = int(law_dict["required_parity"])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid required_parity value, removing")
                    del law_dict["required_parity"]

            # Handle structured neighbor specification (preferred)
            # Both left_neighbor and right_neighbor must be provided together
            left_neighbor = law_dict.get("left_neighbor")
            right_neighbor = law_dict.get("right_neighbor")
            trigger = law_dict.get("trigger_symbol")

            if (left_neighbor is not None) != (right_neighbor is not None):
                # Only one neighbor specified — this would silently drop the constraint
                missing = "left_neighbor" if left_neighbor is None else "right_neighbor"
                provided = "right_neighbor" if left_neighbor is None else "left_neighbor"
                raise ValueError(
                    f"You specified {provided} but not {missing}. "
                    f"You MUST specify BOTH left_neighbor AND right_neighbor together. "
                    f"If you don't care about one side, use each of the 4 symbols (_, A, B, K) "
                    f"in separate tests."
                )

            if left_neighbor is not None and right_neighbor is not None and trigger:
                # Construct pattern from structured fields
                law_dict["neighbor_pattern"] = f"{left_neighbor}{trigger}{right_neighbor}"
                logger.debug(f"Constructed neighbor_pattern from L/R: {law_dict['neighbor_pattern']}")

            # Validate neighbor_pattern format
            # Note: By this point, symbols have been unscrambled to physical form (. > < X)
            neighbor_pattern = law_dict.get("neighbor_pattern")
            if neighbor_pattern is not None:
                # Physical symbols after unscrambling
                valid_symbols = {".", ">", "<", "X"}
                if len(neighbor_pattern) != 3:
                    # Convert to abstract for helpful error
                    from src.proposer.scrambler import get_default_scrambler
                    scrambler = get_default_scrambler()
                    abstract_pattern = scrambler.to_abstract(neighbor_pattern)
                    trigger = law_dict.get("trigger_symbol")
                    abstract_trigger = scrambler.to_abstract(trigger) if trigger else "?"

                    # Suggest the correct pattern
                    if len(neighbor_pattern) == 2:
                        suggested = f"{abstract_pattern[0]}{abstract_trigger}{abstract_pattern[1]}"
                        hint = f"You wrote '{abstract_pattern}' (2 chars). Did you mean '{suggested}'? The trigger must be in the MIDDLE."
                    else:
                        hint = f"You wrote '{abstract_pattern}' ({len(neighbor_pattern)} chars)."

                    raise ValueError(
                        f"Invalid neighbor_pattern: must be EXACTLY 3 characters [LEFT][TRIGGER][RIGHT]. "
                        f"{hint} "
                        f"Example: for trigger '{abstract_trigger}' with left='A' and right='K', use 'A{abstract_trigger}K'."
                    )
                invalid_chars = [c for c in neighbor_pattern if c not in valid_symbols]
                if invalid_chars:
                    raise ValueError(
                        f"Invalid neighbor_pattern: contains invalid characters. "
                        f"Use only valid symbols: _, A, B, K"
                    )
                # Check that middle character matches trigger_symbol if provided
                # Both are in physical form at this point
                trigger = law_dict.get("trigger_symbol")
                if trigger and neighbor_pattern[1] != trigger:
                    # Convert back to abstract for error message
                    from src.proposer.scrambler import get_default_scrambler
                    scrambler = get_default_scrambler()
                    abstract_trigger = scrambler.to_abstract(trigger)
                    abstract_middle = scrambler.to_abstract(neighbor_pattern[1])
                    raise ValueError(
                        f"Invalid neighbor_pattern: middle character must be the trigger_symbol. "
                        f"You specified trigger_symbol='{abstract_trigger}' but the middle of your pattern is '{abstract_middle}'. "
                        f"Pattern format is [left_neighbor, TRIGGER, right_neighbor]. "
                        f"Example: for trigger '{abstract_trigger}' with empty neighbors, use '_{abstract_trigger}_'."
                    )

        # Create the CandidateLaw
        return CandidateLaw(**law_dict)

    def _persist_evaluation(
        self,
        law: CandidateLaw,
        verdict,
        result: dict[str, Any],
    ) -> None:
        """Persist evaluation result to database."""
        try:
            record = LawEvaluationRecord(
                session_id=self._session_id,
                turn_number=self._turn_number,
                law_json=law.model_dump_json(),
                law_id=law.law_id,
                law_hash=law.content_hash(),
                status=verdict.status,
                reason_code=verdict.reason_code.value if verdict.reason_code else None,
                counterexample_json=json.dumps(result.get("counterexample")) if result.get("counterexample") else None,
                power_metrics_json=json.dumps(result.get("power_metrics")) if result.get("power_metrics") else None,
                runtime_ms=verdict.runtime_ms,
            )
            self._repo.insert_law_evaluation(record)
        except Exception as e:
            logger.warning(f"Failed to persist evaluation: {e}")
