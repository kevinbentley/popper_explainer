"""Theorem integrity checks and invariant unmasking.

This module implements logical guardrails to detect:
1. Narrowly defined laws that could be universal invariants
2. Laws that contradict known physics (e.g., claiming X is persistent)
3. Confusion between symbols and components

Key concepts:
- INVARIANT UNMASKING: When a law has preconditions, test if the core claim
  holds universally without those preconditions.
- EPHEMERAL CHECK: X cells are ephemeral - flag laws claiming X is static.
- COMPONENT VS SYMBOL: RightComponent = count('>') + count('X'), not just count('>').
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from src.claims.schema import CandidateLaw, Precondition, Template

if TYPE_CHECKING:
    from src.harness.case import Case, CaseResult
    from src.harness.evaluator import Evaluator

logger = logging.getLogger(__name__)


@dataclass
class IntegrityViolation:
    """A detected integrity violation in a law or theorem.

    Attributes:
        violation_type: Type of violation detected
        severity: 'warning', 'error', or 'info'
        message: Human-readable explanation
        evidence: Supporting data (states, values, etc.)
        suggested_fix: How to address the violation
    """

    violation_type: str
    severity: str  # 'warning', 'error', 'info'
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    suggested_fix: str | None = None


@dataclass
class UnmaskingResult:
    """Result of testing a law without its preconditions.

    Attributes:
        original_law_id: The original law being tested
        stripped_preconditions: Preconditions that were removed
        core_claim_passed: Whether the bare claim passed globally
        test_cases_used: Number of test cases
        counterexample: If failed, the counterexample found
        recommendation: What to do based on results
    """

    original_law_id: str
    stripped_preconditions: list[str]
    core_claim_passed: bool
    test_cases_used: int = 0
    counterexample: dict[str, Any] | None = None
    recommendation: str = ""


# Adversarial stress states for specific hallucination types
HALLUCINATION_BREAKERS = {
    # "X is Persistent/Frozen" - Adjacent X cells emit and collide
    "x_persistent": {
        "states": ["...XX...", "..XXX..", "XX", ".XX.", "XXX"],
        "description": "Adjacent X cells emit particles that collide, proving X is ephemeral",
    },
    # "Conservation is Conditional" - Multiplicity collision preserves components
    "conditional_conservation": {
        "states": [">>.<<", ">>><<<", ">>>.<<<", ">>>>.<<<<"],
        "description": "Multiplicity collisions preserve RightComponent and LeftComponent",
    },
    # "Grid has boundaries" - Wrap-around at N-1 to 0
    "grid_boundaries": {
        "states": ["....>", "<....", "<....>", ">....<"],
        "description": "Particles wrap around periodic boundary",
    },
    # "Collisions are 1v1" - Multiple particles converging
    "one_v_one_collision": {
        "states": [">>.<<", ">>>.<<<", "><><><", ">>>..<<<"],
        "description": "Multiple particles can converge on single cell",
    },
    # "Fragile Conservation" - count('>') changes but RightComponent stays constant
    # AI hallucination: "count('>') is conserved *except* during collisions"
    # Reality: RightComponent = count('>') + count('X') is ALWAYS conserved
    "fragile_conservation": {
        "states": [
            "><",      # t=0: 1>, 1<, 0X → t=1: 0>, 0<, 1X (count changes, components same)
            ".><.",    # Watch count('>') drop when collision forms
            ">><<",    # 2>, 2<, 0X → 0>, 0<, 2X
            ">...<",   # Slow approach: observe count('>') stable until collision
            "..XX..",  # X emits: count('>') increases but RightComponent same
            ">X<",     # All symbols present
        ],
        "description": (
            "count('>') and count('<') CHANGE during collisions, but "
            "RightComponent = count('>') + count('X') is ALWAYS conserved. "
            "AI should define: RightComponent, LeftComponent, TotalParticles"
        ),
        "component_formulas": {
            "RightComponent": "count('>') + count('X')",
            "LeftComponent": "count('<') + count('X')",
            "TotalParticles": "count('>') + count('<') + 2*count('X')",
        },
    },
    # "Homogeneity Stressor" - Distinguishes partial symmetry (shift_k) from universal (mirror_swap)
    # shift_k: state commutation PASSES but local probe invariance FAILS for non-uniform states
    # mirror_swap: PASSES both state commutation AND local probe invariance
    "homogeneity_stressor": {
        "states": [
            # Non-uniform states where shift_k changes local observations
            ">......",  # cell_at(0) = '>' but shift_1 makes cell_at(0) = '.'
            ".>.....",  # Asymmetric - shift changes which cell contains '>'
            "<......",
            "..X....",
            ">..<...",  # Multiple particles - shift changes local structure
            # Uniform states where shift_k preserves everything (trivially)
            ">>>>>>>",  # Homogeneous - shift_k is trivially a symmetry
            "<<<<<<<",
            ".......",
            "XXXXXXX",
            # States that highlight shift vs mirror difference
            ">..",     # shift_k(1): ".>." vs mirror_swap: "..<"
            ".><",     # shift_k changes relative positions but mirror_swap preserves dynamics
        ],
        "description": (
            "Non-uniform states break shift_k symmetry for local probes. "
            "cell_at(i) in original ≠ cell_at(i) after shift_k unless state is homogeneous. "
            "mirror_swap is a UNIVERSAL symmetry; shift_k is only PARTIAL (state commutation only)."
        ),
    },
}


class IntegrityChecker:
    """Checks laws and theorems for logical integrity.

    Implements:
    1. Ephemeral check: X cells cannot be claimed as static
    2. Component vs symbol check: Track N_R = R + X, not just R
    3. Precondition necessity check: Are preconditions actually needed?
    """

    def __init__(self):
        """Initialize integrity checker."""
        # Patterns that indicate problematic claims
        self._ephemeral_violation_patterns = [
            "X is static",
            "X is stable",
            "X is persistent",
            "X remains",
            "X does not change",
            "collision cells are permanent",
            "collision is frozen",
        ]

        self._component_confusion_patterns = [
            # Claims about > count being conserved (should be RightComponent)
            ("count('>').*conserved", "Use RightComponent = count('>') + count('X')"),
            ("count('<').*conserved", "Use LeftComponent = count('<') + count('X')"),
            # Claims about > disappearing during collision
            ("count('>').*decrease.*collision", "Right-movers absorbed into X, RightComponent conserved"),
            ("count('<').*decrease.*collision", "Left-movers absorbed into X, LeftComponent conserved"),
        ]

    def check_law(self, law: CandidateLaw) -> list[IntegrityViolation]:
        """Check a candidate law for integrity violations.

        Args:
            law: The law to check

        Returns:
            List of detected violations
        """
        violations = []

        # Check for ephemeral violations (claiming X is static)
        violations.extend(self._check_ephemeral_claims(law))

        # Check for component vs symbol confusion
        violations.extend(self._check_component_confusion(law))

        # Check for overly narrow preconditions
        violations.extend(self._check_narrow_preconditions(law))

        return violations

    def _check_ephemeral_claims(self, law: CandidateLaw) -> list[IntegrityViolation]:
        """Check if law incorrectly claims X cells are persistent.

        Axiom: X cells are ephemeral - they emit particles and disappear.
        """
        violations = []

        # Check claim text and forbidden text
        texts_to_check = [
            law.claim or "",
            law.forbidden or "",
        ]

        for text in texts_to_check:
            text_lower = text.lower()
            for pattern in self._ephemeral_violation_patterns:
                if pattern.lower() in text_lower:
                    violations.append(IntegrityViolation(
                        violation_type="ephemeral_violation",
                        severity="error",
                        message=f"Law claims X is static/persistent, but X cells are ephemeral",
                        evidence={
                            "pattern_found": pattern,
                            "in_text": text[:100],
                        },
                        suggested_fix="X cells emit one > and one < then disappear. "
                                      "Use local_transition template for per-cell behavior.",
                    ))

        # Check if law template is about X being invariant
        if law.template == Template.INVARIANT:
            # Check observables for pure X count
            for obs in law.observables:
                if obs.expr.strip() == "count('X')" and "CollisionCells" in obs.name:
                    # Check if claim says CollisionCells is constant
                    if law.claim_ast:
                        claim_str = str(law.claim_ast)
                        if "==" in claim_str and "CollisionCells" in claim_str:
                            # Likely claiming X count is invariant
                            violations.append(IntegrityViolation(
                                violation_type="x_invariant_claim",
                                severity="warning",
                                message="CollisionCells (count of X) is NOT an invariant",
                                evidence={
                                    "observable": obs.name,
                                    "expr": obs.expr,
                                },
                                suggested_fix="X cells appear and disappear. Only TotalParticles, "
                                              "RightComponent, LeftComponent, and Momentum are conserved.",
                            ))

        return violations

    def _check_component_confusion(self, law: CandidateLaw) -> list[IntegrityViolation]:
        """Check if law confuses symbol counts with component counts.

        Common mistake: claiming count('>') is conserved
        Reality: RightComponent = count('>') + count('X') is conserved

        CRITICAL PHYSICS:
        - When '>' and '<' collide, they form 'X' (collision cell)
        - X contains BOTH a right-component AND a left-component
        - count('>') DECREASES when collision forms, but RightComponent stays constant
        - RightComponent = count('>') + count('X')  # Right-movers + right-components in collisions
        - LeftComponent = count('<') + count('X')   # Left-movers + left-components in collisions
        - TotalParticles = count('>') + count('<') + 2*count('X')  # Each X contains 2 particles
        """
        violations = []

        # Check observables for component confusion
        has_right_movers = False
        has_right_component = False
        has_left_movers = False
        has_left_component = False
        has_total_particles = False

        for obs in law.observables:
            expr = obs.expr.strip()
            name = obs.name.lower()

            # Detect naive count('>') usage
            if expr == "count('>')":
                has_right_movers = True
                if "conserv" in name or "invariant" in name:
                    violations.append(IntegrityViolation(
                        violation_type="component_confusion",
                        severity="warning",
                        message="count('>') is NOT conserved; RightComponent = count('>') + count('X') IS",
                        evidence={
                            "observable_name": obs.name,
                            "observable_expr": expr,
                            "counterexample": "State '><' has count('>')=1, but after collision '><'→'X', count('>')=0",
                        },
                        suggested_fix="Define: RightComponent = count('>') + count('X'). This IS conserved.",
                    ))

            # Detect naive count('<') usage
            if expr == "count('<')":
                has_left_movers = True
                if "conserv" in name or "invariant" in name:
                    violations.append(IntegrityViolation(
                        violation_type="component_confusion",
                        severity="warning",
                        message="count('<') is NOT conserved; LeftComponent = count('<') + count('X') IS",
                        evidence={
                            "observable_name": obs.name,
                            "observable_expr": expr,
                            "counterexample": "State '><' has count('<')=1, but after collision '><'→'X', count('<')=0",
                        },
                        suggested_fix="Define: LeftComponent = count('<') + count('X'). This IS conserved.",
                    ))

            # Check for proper component definitions
            if "count('>') + count('X')" in expr or "count('X') + count('>')" in expr:
                has_right_component = True

            if "count('<') + count('X')" in expr or "count('X') + count('<')" in expr:
                has_left_component = True

            # Check for proper TotalParticles definition
            if "2*count('X')" in expr or "2 * count('X')" in expr:
                has_total_particles = True

        # Check claim text for "Fragile Conservation" hallucination
        claim_text = (law.claim or "").lower()
        forbidden_text = (law.forbidden or "").lower()
        all_text = claim_text + " " + forbidden_text

        # Detect "conserved except during collision" pattern
        fragile_patterns = [
            "conserved except",
            "conserved unless",
            "conserved when no collision",
            "conserved if no collision",
            "preserved except",
            "preserved unless",
            "constant except during collision",
            "stable except when",
        ]
        for pattern in fragile_patterns:
            if pattern in all_text:
                violations.append(IntegrityViolation(
                    violation_type="fragile_conservation",
                    severity="error",
                    message=(
                        "FRAGILE CONSERVATION HALLUCINATION: The law claims something is "
                        "'conserved except during collisions'. This is NOT how conservation works. "
                        "True conservation quantities are ALWAYS conserved, including during collisions."
                    ),
                    evidence={
                        "pattern_found": pattern,
                        "in_text": all_text[:200],
                    },
                    suggested_fix=(
                        "Use proper component observables that are ALWAYS conserved:\n"
                        "  • RightComponent = count('>') + count('X')  # Always conserved\n"
                        "  • LeftComponent = count('<') + count('X')   # Always conserved\n"
                        "  • TotalParticles = count('>') + count('<') + 2*count('X')  # Always conserved\n"
                        "These quantities do NOT change during collisions because X 'contains' the particles."
                    ),
                ))

        # If law is about conservation and uses raw counts, suggest components
        if law.template == Template.INVARIANT:
            if has_right_movers and not has_right_component:
                violations.append(IntegrityViolation(
                    violation_type="missing_component",
                    severity="info",
                    message="For conservation laws, use RightComponent instead of count('>')",
                    evidence={
                        "has_right_movers": True,
                        "has_right_component": False,
                        "example": "count('>') changes: '><'→'X' drops from 1 to 0",
                    },
                    suggested_fix=(
                        "RightComponent = count('>') + count('X') is conserved.\n"
                        "When '>' enters a collision, it becomes part of 'X', preserving the component count."
                    ),
                ))

            if has_left_movers and not has_left_component:
                violations.append(IntegrityViolation(
                    violation_type="missing_component",
                    severity="info",
                    message="For conservation laws, use LeftComponent instead of count('<')",
                    evidence={
                        "has_left_movers": True,
                        "has_left_component": False,
                        "example": "count('<') changes: '><'→'X' drops from 1 to 0",
                    },
                    suggested_fix=(
                        "LeftComponent = count('<') + count('X') is conserved.\n"
                        "When '<' enters a collision, it becomes part of 'X', preserving the component count."
                    ),
                ))

        # Check for implication laws that avoid collisions
        if law.template in (Template.IMPLICATION_STEP, Template.IMPLICATION_STATE):
            collision_avoiding_preconds = []
            for p in law.preconditions:
                lhs_lower = str(p.lhs).lower()
                if any(term in lhs_lower for term in ["collision", "x", "incoming"]):
                    collision_avoiding_preconds.append(f"{p.lhs} {p.op.value} {p.rhs}")

            if collision_avoiding_preconds and has_right_movers and not has_right_component:
                violations.append(IntegrityViolation(
                    violation_type="collision_avoidance_with_raw_counts",
                    severity="warning",
                    message=(
                        "Law uses collision-avoiding preconditions with raw symbol counts. "
                        "This suggests trying to work around the fact that count('>') isn't conserved."
                    ),
                    evidence={
                        "collision_avoiding": collision_avoiding_preconds,
                        "observables_with_raw_counts": [
                            obs.name for obs in law.observables
                            if obs.expr.strip() in ("count('>')", "count('<')")
                        ],
                    },
                    suggested_fix=(
                        "Instead of avoiding collisions, use component observables:\n"
                        "  • RightComponent = count('>') + count('X')  # Conserved ALWAYS\n"
                        "  • LeftComponent = count('<') + count('X')   # Conserved ALWAYS\n"
                        "Then the law can hold universally without preconditions."
                    ),
                ))

        return violations

    def _check_narrow_preconditions(self, law: CandidateLaw) -> list[IntegrityViolation]:
        """Check for potentially unnecessary preconditions.

        If a law has many preconditions, it might be avoiding the real physics.
        """
        violations = []

        if len(law.preconditions) >= 2:
            violations.append(IntegrityViolation(
                violation_type="narrow_preconditions",
                severity="info",
                message=f"Law has {len(law.preconditions)} preconditions - consider if all are necessary",
                evidence={
                    "precondition_count": len(law.preconditions),
                    "preconditions": [
                        f"{p.lhs} {p.op.value} {p.rhs}" for p in law.preconditions
                    ],
                },
                suggested_fix="Try testing without preconditions to see if law holds universally",
            ))

        # Check for collision-avoiding preconditions
        collision_avoiding = []
        for p in law.preconditions:
            # Precondition.lhs is the observable name
            lhs_lower = str(p.lhs).lower()
            if any(term in lhs_lower for term in ["collision", "incoming", "x"]):
                collision_avoiding.append(p)

        if collision_avoiding:
            violations.append(IntegrityViolation(
                violation_type="collision_avoidance",
                severity="warning",
                message="Law has preconditions that avoid collision scenarios",
                evidence={
                    "collision_avoiding_preconditions": [
                        f"{p.lhs} {p.op.value} {p.rhs}" for p in collision_avoiding
                    ],
                },
                suggested_fix="Many laws that seem conditional on 'no collisions' are actually "
                              "universal when using RightComponent/LeftComponent instead of raw counts",
            ))

        return violations


class InvariantUnmasker:
    """Tests laws without their preconditions to find universal invariants.

    When a law passes with preconditions, we test if it would pass without them.
    If it does, the law is "narrowly defined" and should be reformulated.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        stress_states: list[str] | None = None,
    ):
        """Initialize unmasker.

        Args:
            evaluator: The evaluator to use for testing
            stress_states: Additional stress states to test
        """
        self._evaluator = evaluator
        self._stress_states = stress_states or self._default_stress_states()

    def _default_stress_states(self) -> list[str]:
        """Get default stress states for unmasking tests."""
        states = []

        # Collect from hallucination breakers
        for breaker in HALLUCINATION_BREAKERS.values():
            states.extend(breaker["states"])

        # Add more stress states
        states.extend([
            # Dense collision scenarios
            "><><><><",
            ">>><<<",
            ">>.<<",
            # X-battery scenarios
            "XX",
            "XXX",
            "...XX...",
            # Boundary scenarios
            "....>",
            "<....",
            "<....>",
            # Mixed scenarios
            ">X<",
            ".>X<.",
            ">>X<<",
        ])

        return list(set(states))  # Dedupe

    def should_unmask(self, law: CandidateLaw) -> bool:
        """Check if a law should be tested without preconditions.

        Args:
            law: The law to check

        Returns:
            True if the law has preconditions worth stripping
        """
        # Must have preconditions
        if not law.preconditions:
            return False

        # Must be a template that could be universal
        universal_templates = {
            Template.INVARIANT,
            Template.MONOTONE,
            Template.BOUND,
        }
        if law.template not in universal_templates:
            return False

        # Check if preconditions are about avoiding collisions/X
        for p in law.preconditions:
            # Precondition.lhs is the observable name
            lhs_lower = str(p.lhs).lower()
            if any(term in lhs_lower for term in ["collision", "incoming", "x", "freemover"]):
                return True

        return False

    def unmask(
        self,
        law: CandidateLaw,
        time_horizon: int,
    ) -> UnmaskingResult:
        """Test a law without its preconditions.

        Args:
            law: The law to unmask
            time_horizon: Time horizon for evaluation

        Returns:
            UnmaskingResult with findings
        """
        if not law.preconditions:
            return UnmaskingResult(
                original_law_id=law.law_id,
                stripped_preconditions=[],
                core_claim_passed=True,
                recommendation="No preconditions to strip",
            )

        # Create stripped version of law
        stripped_law = self._strip_preconditions(law)

        # Record what we stripped
        stripped_preconds = [
            f"{p.observable} {p.op.value} {p.value}"
            for p in law.preconditions
        ]

        # Test against stress states
        passed_count = 0
        failed_count = 0
        counterexample = None

        for state in self._stress_states:
            # Skip states that don't match grid length requirements
            if len(state) < 2:
                continue

            from src.harness.case import Case
            from src.universe.types import Config

            case = Case(
                initial_state=state,
                config=Config(grid_length=len(state)),
                seed=42,
                generator_family="unmasking_test",
                params_hash="unmask",
            )

            result = self._evaluator.evaluate_case(case, time_horizon)

            if result.passed:
                passed_count += 1
            else:
                failed_count += 1
                if counterexample is None and result.violation:
                    counterexample = {
                        "initial_state": state,
                        "t_fail": result.violation.get("t", 0),
                        "violation": result.violation,
                    }

        # Determine recommendation
        total = passed_count + failed_count
        core_claim_passed = failed_count == 0

        if core_claim_passed:
            recommendation = (
                f"NARROWLY DEFINED: Core claim passed ALL {total} stress tests without preconditions. "
                f"Reformulate as universal invariant."
            )
        elif passed_count > failed_count * 2:
            recommendation = (
                f"PARTIALLY UNIVERSAL: Core claim passed {passed_count}/{total} tests. "
                f"Preconditions may be too narrow. Investigate failures."
            )
        else:
            recommendation = (
                f"PRECONDITIONS JUSTIFIED: Core claim failed {failed_count}/{total} tests. "
                f"Preconditions are necessary for this claim."
            )

        return UnmaskingResult(
            original_law_id=law.law_id,
            stripped_preconditions=stripped_preconds,
            core_claim_passed=core_claim_passed,
            test_cases_used=total,
            counterexample=counterexample,
            recommendation=recommendation,
        )

    def _strip_preconditions(self, law: CandidateLaw) -> CandidateLaw:
        """Create a copy of the law with preconditions removed.

        Args:
            law: The original law

        Returns:
            A copy with empty preconditions
        """
        # Deep copy the law
        stripped = copy.deepcopy(law)
        stripped.preconditions = []
        stripped.law_id = f"{law.law_id}_unmasked"

        return stripped


def check_theorem_integrity(
    claim: str,
    supporting_laws: list[dict[str, Any]],
) -> list[IntegrityViolation]:
    """Check a synthesized theorem for integrity violations.

    Args:
        claim: The theorem claim text
        supporting_laws: Laws supporting the theorem

    Returns:
        List of violations found
    """
    violations = []
    claim_lower = claim.lower()

    # Check for ephemeral violations
    ephemeral_patterns = [
        "x is static", "x remains", "x persists", "collision is frozen",
        "x does not change", "x is stable",
    ]
    for pattern in ephemeral_patterns:
        if pattern in claim_lower:
            violations.append(IntegrityViolation(
                violation_type="theorem_ephemeral_violation",
                severity="error",
                message=f"Theorem claims X is static, but X cells are ephemeral (emit and disappear)",
                evidence={"pattern": pattern, "claim_excerpt": claim[:200]},
                suggested_fix="X cells emit one > and one < then resolve. They are not persistent.",
            ))

    # Check for "Fragile Conservation" hallucination
    fragile_patterns = [
        "conserved except",
        "conserved unless",
        "conserved when no collision",
        "conserved if no collision",
        "preserved except",
        "preserved unless",
        "constant except during collision",
        "stable except when",
        "invariant except",
    ]
    for pattern in fragile_patterns:
        if pattern in claim_lower:
            violations.append(IntegrityViolation(
                violation_type="theorem_fragile_conservation",
                severity="error",
                message=(
                    "FRAGILE CONSERVATION HALLUCINATION: Theorem claims something is "
                    "'conserved except during collisions'. True conservation quantities "
                    "are ALWAYS conserved, including during collisions."
                ),
                evidence={"pattern": pattern, "claim_excerpt": claim[:200]},
                suggested_fix=(
                    "Use proper component observables that are ALWAYS conserved:\n"
                    "  • RightComponent = count('>') + count('X')  # Always conserved\n"
                    "  • LeftComponent = count('<') + count('X')   # Always conserved\n"
                    "  • TotalParticles = count('>') + count('<') + 2*count('X')  # Always conserved\n"
                    "  • Momentum = count('>') - count('<')  # Always conserved\n"
                    "These don't change during collisions because X 'contains' the particles."
                ),
            ))

    # Check for component confusion
    # Only flag if count('>') or count('<') is claimed conserved BY ITSELF
    # (not as part of a sum like TotalParticles or RightComponent)
    if "count('>')" in claim and "conserved" in claim_lower:
        # Check if it's used alone or as part of a proper formula
        # If it's in a formula with count('X'), it's probably correct
        if "count('X')" not in claim:
            violations.append(IntegrityViolation(
                violation_type="theorem_component_confusion",
                severity="warning",
                message="Theorem confuses symbol count with component: count('>') is NOT conserved",
                evidence={
                    "claim_excerpt": claim[:200],
                    "counterexample": "State '><' has count('>')=1, after collision '><'→'X', count('>')=0",
                },
                suggested_fix=(
                    "RightComponent = count('>') + count('X') IS conserved.\n"
                    "When > collides with <, they form X. The > doesn't disappear—it's now inside X."
                ),
            ))

    if "count('<')" in claim and "conserved" in claim_lower:
        # Same check - only flag if count('<') is used without count('X')
        if "count('X')" not in claim:
            violations.append(IntegrityViolation(
                violation_type="theorem_component_confusion",
                severity="warning",
                message="Theorem confuses symbol count with component: count('<') is NOT conserved",
                evidence={
                    "claim_excerpt": claim[:200],
                    "counterexample": "State '><' has count('<')=1, after collision '><'→'X', count('<')=0",
                },
                suggested_fix=(
                    "LeftComponent = count('<') + count('X') IS conserved.\n"
                    "When < collides with >, they form X. The < doesn't disappear—it's now inside X."
                ),
            ))

    # Check for "Fundamental Asymmetry" hallucination regarding shift symmetry
    asymmetry_patterns = [
        "fundamental asymmetry",
        "shift is not a symmetry",
        "shift_k breaks",
        "translation symmetry is broken",
    ]
    for pattern in asymmetry_patterns:
        if pattern in claim_lower:
            violations.append(IntegrityViolation(
                violation_type="theorem_symmetry_confusion",
                severity="info",
                message=(
                    "Note on shift symmetry: shift_k is a PARTIAL symmetry. "
                    "It passes state commutation (dynamics are shift-invariant) but "
                    "fails local probe invariance for non-uniform states."
                ),
                evidence={"pattern": pattern, "claim_excerpt": claim[:200]},
                suggested_fix=(
                    "Distinguish between:\n"
                    "  • UNIVERSAL symmetry (mirror_swap): passes BOTH state commutation AND local probe invariance\n"
                    "  • PARTIAL symmetry (shift_k): passes state commutation, fails local probe invariance\n"
                    "shift_k: step(shift_k(s)) == shift_k(step(s)) ✓ (dynamics are shift-invariant)\n"
                    "But: cell_at(i, shift_k(s)) ≠ cell_at(i, s) for non-uniform states (local probes change)"
                ),
            ))

    return violations


def get_component_observable_guidance() -> dict[str, Any]:
    """Return guidance for AI on how to define proper component observables.

    This can be included in discovery prompts to help the AI understand
    the correct conservation quantities.
    """
    return {
        "title": "Component Observables (CRITICAL for Conservation Laws)",
        "explanation": (
            "In this universe, '>' and '<' particles collide to form 'X' collision cells. "
            "The collision cell X contains BOTH a right-moving component AND a left-moving component. "
            "This means simple symbol counts like count('>') are NOT conserved—they decrease when "
            "collisions form. To discover true conservation laws, you must use COMPONENT observables."
        ),
        "formulas": {
            "RightComponent": {
                "expr": "count('>') + count('X')",
                "meaning": "Total right-moving particles (free + trapped in collisions)",
                "conserved": True,
            },
            "LeftComponent": {
                "expr": "count('<') + count('X')",
                "meaning": "Total left-moving particles (free + trapped in collisions)",
                "conserved": True,
            },
            "TotalParticles": {
                "expr": "count('>') + count('<') + 2*count('X')",
                "meaning": "Total particle count (each X contains 2 particles)",
                "conserved": True,
            },
            "Momentum": {
                "expr": "count('>') - count('<')",
                "meaning": "Net rightward momentum (+1 per >, -1 per <, X contributes 0)",
                "conserved": True,
            },
        },
        "counterexample": {
            "state": "'.><.'",
            "t0": {"count_right": 1, "count_left": 1, "count_X": 0, "RightComponent": 1, "LeftComponent": 1},
            "t1": {"count_right": 0, "count_left": 0, "count_X": 1, "RightComponent": 1, "LeftComponent": 1},
            "observation": "count('>') dropped from 1 to 0, but RightComponent stayed at 1",
        },
        "common_hallucination": (
            "'Fragile Conservation': claiming count('>') is conserved 'except during collisions'. "
            "This is wrong. True conservation quantities don't have exceptions. "
            "If your law needs 'no collision' preconditions, you're probably using the wrong observable."
        ),
    }
