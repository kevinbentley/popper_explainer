"""Prompt builder for law proposal."""

import json
from dataclasses import dataclass, field
from typing import Any

from src.proposer.memory import DiscoveryMemorySnapshot


@dataclass
class UniverseContract:
    """Contract describing universe capabilities.

    Attributes:
        universe_id: Unique identifier for the universe
        symbols: Valid state symbols
        state_representation: How state is represented
        capabilities: Available observables, transforms, generators
        config_knobs: Configurable parameters
    """

    universe_id: str = "kinetic_grid_v1"
    symbols: list[str] = field(default_factory=lambda: [".", ">", "<", "X"])
    state_representation: str = "string"
    capabilities: dict[str, Any] = field(default_factory=lambda: {
        "primitive_observables": ["count(symbol)", "grid_length"],
        "derived_observables_allowed": True,
        "transforms": ["mirror_swap", "shift_k", "swap_only", "mirror_only"],
        "generator_families": [
            "random_density_sweep",
            "constrained_pair_interactions",
            "edge_wrapping_cases",
            "symmetry_metamorphic_suite",
            "adversarial_mutation_search",
        ],
    })
    config_knobs: dict[str, Any] = field(default_factory=lambda: {
        "grid_length_range": [4, 200],
        "boundary": "periodic",
    })

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "universe_id": self.universe_id,
            "symbols": self.symbols,
            "state_representation": self.state_representation,
            "capabilities": self.capabilities,
            "config_knobs": self.config_knobs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UniverseContract":
        """Create from dictionary."""
        return cls(
            universe_id=data.get("universe_id", "kinetic_grid_v1"),
            symbols=data.get("symbols", [".", ">", "<", "X"]),
            state_representation=data.get("state_representation", "string"),
            capabilities=data.get("capabilities", {}),
            config_knobs=data.get("config_knobs", {}),
        )


SYSTEM_INSTRUCTION = """You are a scientific law discovery system. Your task is to propose testable, falsifiable candidate laws about a simulated universe.

CRITICAL RULES:
1. You must ONLY output a JSON array of CandidateLaw objects. No prose, no markdown, no explanations.
2. You must NOT propose axioms, mechanisms, definitions, or update rules. Only propose CLAIMS that can be tested.
3. Every law must include a "forbidden" field describing what would falsify it.
4. Every law must use one of the allowed templates exactly.
5. Do NOT restate or rephrase already-accepted laws.
6. Prioritize RISKY laws that are easily falsifiable over safe, trivial laws.
7. ALL claims MUST use the structured JSON AST format (claim_ast field), NOT string expressions.

ALLOWED TEMPLATES:
- invariant: ∀t∈[0..T]: claim_ast holds for all t
- monotone: ∀t∈[0..T-1]: claim_ast holds (typically comparing f(t+1) to f(t))
- implication_step: ∀t∈[0..T-1]: antecedent(t) → consequent(t+1)
- implication_state: ∀t∈[0..T]: antecedent(t) → consequent(t)
- eventually: ∀t0: antecedent(t0) → ∃t∈[t0..t0+H]: consequent(t)
- symmetry_commutation: evolve(Transform(S), T) == Transform(evolve(S, T))
- bound: ∀t∈[0..T]: claim_ast holds (typically f(t) op k)
- local_transition: ∀t,i: state[i]==trigger → state[i] satisfies result at t+1 (per-cell rule)

MICRO vs MACRO LAWS - IMPORTANT DISTINCTION:
Most templates express MACRO (aggregate) laws about global counts: count(X), count(>), etc.
The local_transition template expresses MICRO (per-cell) laws about individual cell behavior.

Example of the difference:
- MACRO (FALSE): "If count(X) > 0 then count(X) at t+1 == 0" - FAILS because new X can form
- MICRO (TRUE): "Each X cell becomes non-X at the same position" - PASSES because individual X resolves

Use local_transition for rules about what happens to individual cells, not aggregate counts.

LOCAL_TRANSITION TEMPLATE:
- Requires fields: trigger_symbol, result_op (== or !=), result_symbol
- Expresses: ∀t,i: if state[i] == trigger_symbol at t, then state[i] result_op result_symbol at t+1
- Example: trigger_symbol="X", result_op="!=", result_symbol="X" means each X cell resolves

OBSERVABLES (defined in "observables" field):
Define observables using only these primitives:
- count('>') - right-moving particles (free, not in collision)
- count('<') - left-moving particles (free, not in collision)
- count('X') - collision cells (each X contains one > and one <)
- count('.') - empty cells
- grid_length - grid size (constant)
- leftmost(symbol), rightmost(symbol) - position of first/last occurrence
- max_gap(symbol) - longest contiguous run of symbol
- spread(symbol) - rightmost - leftmost position
- adjacent_pairs(sym1, sym2) - count of sym1 immediately followed by sym2

CANONICAL OBSERVABLES - USE THESE NAMES for semantic clarity:

CONSERVED quantities (invariant under time evolution):
- TotalParticles:  count('>') + count('<') + 2*count('X')  -- each X contains 2 particles
- RightComponent:  count('>') + count('X')                 -- right-moving component
- LeftComponent:   count('<') + count('X')                 -- left-moving component
- Momentum:        count('>') - count('<')                 -- net momentum

NOT CONSERVED (change during collisions):
- FreeMovers:      count('>') + count('<')                 -- decreases when colliding
- OccupiedCells:   count('>') + count('<') + count('X')    -- 2 cells become 1 during collision
- EmptyCells:      count('.')                              -- changes during collisions
- CollisionCells:  count('X')                              -- X cells resolve next step

CONDITIONALLY CONSERVED (only if no collisions occur):
- RightMovers:     count('>')
- LeftMovers:      count('<')

IMPORTANT: For conservation laws, use TotalParticles, RightComponent, LeftComponent, or Momentum.
Do NOT use FreeMovers, OccupiedCells, or CollisionCells for conservation laws - they are not conserved!

Example observables:
- {"name": "TotalParticles", "expr": "count('>') + count('<') + 2*count('X')"}
- {"name": "RightComponent", "expr": "count('>') + count('X')"}
- {"name": "Momentum", "expr": "count('>') - count('<')"}
- {"name": "FreeMovers", "expr": "count('>') + count('<')"}
- {"name": "CollisionCells", "expr": "count('X')"}

STRUCTURED CLAIM AST (REQUIRED):
Claims must be JSON ASTs with these node types:
- Constant: {"const": <number>}
- Time variable: {"var": "t"}
- Time t+1: {"t_plus_1": true}
- Observable at time: {"obs": "<name>", "t": <time_node>}
- Binary op: {"op": "<op>", "lhs": <node>, "rhs": <node>}
- Unary not: {"op": "not", "arg": <node>}

Valid operators: +, -, *, /, ==, !=, <, <=, >, >=, =>, and, or, not

EXAMPLES:
1. Invariant "N(t) == N(0)":
   {"op": "==", "lhs": {"obs": "N", "t": {"var": "t"}}, "rhs": {"obs": "N", "t": {"const": 0}}}

2. Monotone "X_count(t+1) <= X_count(t)":
   {"op": "<=", "lhs": {"obs": "X_count", "t": {"t_plus_1": true}}, "rhs": {"obs": "X_count", "t": {"var": "t"}}}

3. Implication (implication_step or implication_state) - MUST have "=>" at root:
   {"op": "=>",
    "lhs": {"op": ">", "lhs": {"obs": "X_count", "t": {"var": "t"}}, "rhs": {"const": 0}},
    "rhs": {"op": "==", "lhs": {"obs": "X_count", "t": {"t_plus_1": true}}, "rhs": {"const": 0}}}

4. Eventually - MUST have "=>" at root (antecedent => consequent):
   {"op": "=>",
    "lhs": {"op": ">", "lhs": {"obs": "X_count", "t": {"var": "t"}}, "rhs": {"const": 0}},
    "rhs": {"op": "==", "lhs": {"obs": "X_count", "t": {"var": "t"}}, "rhs": {"const": 0}}}
   (means: if X_count > 0 at t0, then eventually X_count == 0 within horizon H)

4. Bound "N(t) <= grid_length":
   {"op": "<=", "lhs": {"obs": "N", "t": {"var": "t"}}, "rhs": {"obs": "grid_len", "t": {"var": "t"}}}
   (requires: {"name": "grid_len", "expr": "grid_length"})

For symmetry_commutation template:
- Include "transform" field with: "mirror_swap", "shift_k", "mirror_only", "swap_only"
- claim_ast can be null for symmetry (handled specially)

For local_transition template (MICRO-level per-cell rules):
- Include fields: trigger_symbol, result_op, result_symbol
- Example: "Each X cell resolves in one step"
  {
    "law_id": "x_cell_resolves_one_step",
    "template": "local_transition",
    "quantifiers": {"T": 50},
    "trigger_symbol": "X",
    "result_op": "!=",
    "result_symbol": "X",
    "claim": "For each position i: if state[i]=='X' at t, then state[i]!='X' at t+1",
    "forbidden": "An X cell that remains X at the same position in the next timestep"
  }

OUTPUT FORMAT:
[
  {
    "law_id": "particle_count_conserved",
    "template": "invariant",
    "quantifiers": {"T": 50},
    "observables": [{"name": "N", "expr": "count('>') + count('<') + 2*count('X')"}],
    "claim_ast": {"op": "==", "lhs": {"obs": "N", "t": {"var": "t"}}, "rhs": {"obs": "N", "t": {"const": 0}}},
    "forbidden": "exists t where N(t) != N(0)",
    "proposed_tests": [{"family": "random_density_sweep", "params": {"cases": 100}}]
  }
]"""


class PromptBuilder:
    """Builds prompts for law proposal from memory and contract."""

    def __init__(
        self,
        max_token_budget: int = 8000,
        include_counterexamples: bool = True,
    ):
        """Initialize prompt builder.

        Args:
            max_token_budget: Maximum tokens for prompt
            include_counterexamples: Whether to include counterexample gallery
        """
        self.max_token_budget = max_token_budget
        self.include_counterexamples = include_counterexamples

    def build(
        self,
        contract: UniverseContract,
        memory: DiscoveryMemorySnapshot,
        request_count: int = 5,
        target_templates: list[str] | None = None,
        exclude_templates: list[str] | None = None,
    ) -> str:
        """Build a prompt for law proposal.

        The prompt is structured with STATIC content first (for API caching)
        and DYNAMIC content (discoveries) last.

        Args:
            contract: Universe contract
            memory: Discovery memory snapshot
            request_count: Number of laws to request
            target_templates: Templates to focus on (optional)
            exclude_templates: Templates to avoid (optional)

        Returns:
            Formatted prompt string
        """
        sections = []

        # === STATIC SECTION (for caching) ===
        # Universe capabilities - this rarely changes
        sections.append(self._build_capabilities_section(contract))

        # Expression language reminder
        sections.append(self._build_expression_language_section())

        # Request section - mostly static
        sections.append(self._build_request_section(
            request_count, target_templates, exclude_templates
        ))

        # === DYNAMIC SECTION (changes each iteration) ===
        # Accepted laws
        if memory.accepted_laws:
            sections.append(self._build_accepted_section(memory.accepted_laws))

        # Falsified laws
        if memory.falsified_laws:
            sections.append(self._build_falsified_section(memory.falsified_laws))

        # Unknown laws
        if memory.unknown_laws:
            sections.append(self._build_unknown_section(memory.unknown_laws))

        # Counterexamples
        if self.include_counterexamples and memory.counterexamples:
            sections.append(self._build_counterexamples_section(memory.counterexamples))

        return "\n\n".join(sections)

    def _build_expression_language_section(self) -> str:
        """Build expression language and AST format reminder section."""
        return """=== CLAIM AST FORMAT (REQUIRED) ===
Claims must be structured JSON ASTs. Observable expressions are defined separately.

OBSERVABLE EXPRESSIONS (in "observables" field):
  Primitives: count('>'), count('<'), count('X'), count('.'), grid_length
  Position: leftmost(sym), rightmost(sym), spread(sym), max_gap(sym), adjacent_pairs(s1, s2)
  Operators: +, -, *, /

CANONICAL OBSERVABLE NAMES (use these for clarity):
  CONSERVED:
    TotalParticles:  count('>') + count('<') + 2*count('X')   -- use for conservation laws
    RightComponent:  count('>') + count('X')                  -- use for conservation laws
    LeftComponent:   count('<') + count('X')                  -- use for conservation laws
    Momentum:        count('>') - count('<')                  -- use for conservation laws
  NOT CONSERVED:
    FreeMovers:      count('>') + count('<')                  -- changes during collisions!
    OccupiedCells:   count('>') + count('<') + count('X')     -- changes during collisions!
    CollisionCells:  count('X')                               -- changes each step!

CLAIM AST NODES (in "claim_ast" field):
  Constant:     {"const": 5}
  Time var:     {"var": "t"}
  Time t+1:     {"t_plus_1": true}
  Observable:   {"obs": "<name>", "t": <time_node>}
  Binary op:    {"op": "<op>", "lhs": <node>, "rhs": <node>}
  Unary not:    {"op": "not", "arg": <node>}

  Ops: +, -, *, /, ==, !=, <, <=, >, >=, =>, and, or, not

AST EXAMPLES:
  TotalParticles(t) == TotalParticles(0) [CONSERVED - correct]:
    {"op": "==", "lhs": {"obs": "TotalParticles", "t": {"var": "t"}}, "rhs": {"obs": "TotalParticles", "t": {"const": 0}}}

  CollisionCells(t+1) <= CollisionCells(t) [monotone]:
    {"op": "<=", "lhs": {"obs": "CollisionCells", "t": {"t_plus_1": true}}, "rhs": {"obs": "CollisionCells", "t": {"var": "t"}}}

  P(t) > 0 => Q(t+1) [implication]:
    {"op": "=>", "lhs": {"op": ">", ...}, "rhs": {..., "t": {"t_plus_1": true}}}

For symmetry_commutation: include "transform" field, claim_ast can be null."""

    def get_system_instruction(self) -> str:
        """Get the system instruction for the LLM."""
        return SYSTEM_INSTRUCTION

    def _build_capabilities_section(self, contract: UniverseContract) -> str:
        """Build universe capabilities section."""
        caps = contract.capabilities
        lines = [
            "=== UNIVERSE CAPABILITIES ===",
            f"Symbols: {', '.join(contract.symbols)}",
            f"State: {contract.state_representation}",
            "",
            "Primitive observables:",
        ]
        for obs in caps.get("primitive_observables", []):
            lines.append(f"  - {obs}")

        lines.append("")
        lines.append("Available transforms:")
        for t in caps.get("transforms", []):
            lines.append(f"  - {t}")

        lines.append("")
        lines.append("Test families:")
        for g in caps.get("generator_families", []):
            lines.append(f"  - {g}")

        lines.append("")
        lines.append("Config knobs:")
        for k, v in contract.config_knobs.items():
            lines.append(f"  - {k}: {v}")

        return "\n".join(lines)

    def _build_accepted_section(self, laws: list[dict[str, Any]]) -> str:
        """Build accepted laws section."""
        lines = ["=== ACCEPTED LAWS (do not repeat these or equivalent variations) ==="]
        for law in laws:
            lines.append(f"- [{law['template']}] {law['law_id']}: {law['claim']}")
            # Show observable definitions to prevent equivalent re-proposals
            if law.get("observables"):
                obs_strs = [f"{o['name']}={o['expr']}" for o in law["observables"]]
                lines.append(f"    observables: {', '.join(obs_strs)}")
        return "\n".join(lines)

    def _build_falsified_section(self, laws: list[dict[str, Any]]) -> str:
        """Build falsified laws section."""
        lines = ["=== FALSIFIED LAWS (do not repeat these or equivalent variations) ==="]
        for law in laws:
            cx = law.get("counterexample", {})
            lines.append(
                f"- [{law['template']}] {law['law_id']}: {law['claim']}"
            )
            # Show observable definitions
            if law.get("observables"):
                obs_strs = [f"{o['name']}={o['expr']}" for o in law.get("observables", [])]
                if obs_strs:
                    lines.append(f"    observables: {', '.join(obs_strs)}")
            if cx:
                lines.append(f"    counterexample: state='{cx.get('initial_state', '?')}', t_fail={cx.get('t_fail', '?')}")
        return "\n".join(lines)

    def _build_unknown_section(self, laws: list[dict[str, Any]]) -> str:
        """Build unknown laws section."""
        lines = ["=== UNKNOWN LAWS (capability gaps) ==="]
        for law in laws:
            reason = law.get("reason_code", "unknown")
            lines.append(f"- [{law['template']}] {law['law_id']}: {reason}")
        return "\n".join(lines)

    def _build_counterexamples_section(self, examples: list[dict[str, Any]]) -> str:
        """Build counterexamples gallery section."""
        lines = ["=== COUNTEREXAMPLE GALLERY ==="]
        for i, cx in enumerate(examples[:10]):  # Limit to 10
            lines.append(f"{i+1}. state='{cx.get('initial_state', '?')}', t_fail={cx.get('t_fail', '?')}")
            if cx.get("trajectory_excerpt"):
                excerpt = cx["trajectory_excerpt"][:5]
                lines.append(f"   trajectory: {' -> '.join(excerpt)}")
        return "\n".join(lines)

    def _build_request_section(
        self,
        count: int,
        target_templates: list[str] | None,
        exclude_templates: list[str] | None,
    ) -> str:
        """Build request section."""
        lines = [
            "=== YOUR TASK ===",
            f"Propose {count} NEW candidate laws.",
            "",
            "Requirements:",
            "- Each law must be testable and falsifiable",
            "- Include a forbidden condition describing what would disprove it",
            "- Do not repeat accepted or falsified laws",
            "- Prioritize risky, discriminating laws over safe ones",
        ]

        if target_templates:
            lines.append(f"- Focus on templates: {', '.join(target_templates)}")

        if exclude_templates:
            lines.append(f"- Avoid templates: {', '.join(exclude_templates)}")

        lines.append("")
        lines.append("Output ONLY a JSON array of CandidateLaw objects.")

        return "\n".join(lines)

    def estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for a prompt."""
        # Rough estimate: 1 token per 4 characters
        return len(prompt) // 4
