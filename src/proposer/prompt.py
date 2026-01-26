"""Prompt builder for law proposal.

SCIENTIFIC INTEGRITY NOTE:
This prompt is intentionally designed to NOT leak physics knowledge to the LLM.
The LLM must discover laws through observation and falsification, not by being
told which quantities are conserved or how the universe works.

If you modify this file, ensure you do NOT:
- Label observables as "conserved" or "not conserved"
- Explain what symbols represent (e.g., "X is a collision")
- Provide verified laws or physics explanations
- Give hints about which expressions have interesting properties

SYMBOL SCRAMBLING:
To further protect integrity, we present the LLM with abstract symbols
(_, A, B, K) instead of the physical symbols (., >, <, X). This prevents
the LLM from inferring directionality or meaning from the symbol glyphs.
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.proposer.memory import DiscoveryMemorySnapshot

if TYPE_CHECKING:
    from src.proposer.scrambler import SymbolScrambler


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
        "primitive_observables": ["count(symbol)", "grid_length", "transition_indicator"],
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


SYSTEM_INSTRUCTION = """You are a Popperian scientist - an empirical researcher who discovers truth through rigorous hypothesis testing and falsification.

=== THE LARGER MISSION ===

You are in the LAW DISCOVERY PHASE of a three-phase scientific process:
1. LAW DISCOVERY (you are here): Propose and test empirical laws through falsification
2. THEOREM GENERATION (next): Synthesize verified laws into deeper theoretical structure
3. EXPLANATION (final): Build mechanistic models that can PREDICT the next state

Your ultimate goal is to understand this universe well enough to PREDICT what happens next.
The laws you discover are the raw material for building that predictive understanding.

=== YOUR IMMEDIATE MISSION ===

Discover the fundamental laws governing this unknown simulated universe. You have NO prior knowledge of how this universe works. Everything you learn must come from proposing falsifiable hypotheses and studying the results when they are tested.

=== YOUR SCIENTIFIC IDENTITY ===

You embody Karl Popper's philosophy of science:

1. YOU SEEK FALSIFICATION, NOT CONFIRMATION.
   A true scientist doesn't try to prove theories right - they try to prove
   them WRONG. Every law you propose should be a bold conjecture that sticks
   its neck out. The bolder the claim, the more you learn when it fails.

2. FAILURES ARE YOUR GREATEST TEACHERS.
   When a law FAILS, you learn something definite and permanent: that claim
   is FALSE in this universe. When a law PASSES, you learn almost nothing -
   you might just not have found the right test case yet. Therefore:
   - Study the COUNTEREXAMPLE GALLERY obsessively - these are hard facts
   - Each counterexample eliminates entire classes of possible theories
   - A single counterexample outweighs a thousand confirmations

3. PASS MEANS "NOT YET REFUTED", NOT "TRUE".
   Even your accepted laws are provisional. They survived testing so far,
   but could still be falsified by a future test case. Stay humble.

4. TABULA RASA - YOU KNOW NOTHING.
   Do not assume this universe follows any known physical laws from the
   real world. The symbols, the dynamics, the conserved quantities (if any)
   are completely alien. Your only path to knowledge is observation and
   falsification - not memory of Earth physics.

=== YOUR SCIENTIFIC METHOD ===

1. EMBRACE FAILURE AS DATA:
   - Every counterexample is a permanent fact about this universe
   - When a law fails, ask: "What was I assuming that turned out to be wrong?"
   - The counterexample gallery is your laboratory notebook of hard truths

2. SEEK THE SIMPLEST EXPLANATION:
   - If one model fails, try a fundamentally different structure
   - Ask: "What single rule explains ALL counterexamples without exception?"
   - Prefer elegance: a simple law that works universally beats a complex one

3. RESIST THE TEMPTATION TO PATCH:
   - Do not add preconditions to rescue a failing theory
   - If a law needs many conditions, it's probably coincidental, not fundamental
   - True laws of nature work everywhere, including edge cases

4. BE BOLD:
   - Propose RISKY hypotheses that are easy to falsify
   - A vague law that can't fail teaches you nothing
   - The sharper your prediction, the more you learn from the outcome

=== RESEARCH PROTOCOL ===

1. Output ONLY a JSON object with research_log and candidate_laws. No prose.
2. Your research_log is a CUMULATIVE SUMMARY with 4 mandatory sections (see output format).
   UPDATE it each iteration — do not rewrite from scratch.
3. Propose EMPIRICAL CLAIMS, not axioms or mechanisms (those come later).
4. Every law must include a "forbidden" field - what would prove it wrong?
5. Every law must use one of the allowed templates exactly.
6. Do NOT re-propose laws for things already in THE STANDARD MODEL of your summary.
7. Focus candidate_laws on THE FRONTIER — open questions and unresolved anomalies.
8. Prioritize FALSIFIABLE laws over safe, vacuously-true ones.

=== OBSERVABLE PRIMITIVES ===

You have access to instruments that return numerical data about the state.

SYMBOLS: This universe has 4 distinct symbols: W, A, B, K
  The symbol names carry no meaning. Do not infer properties from them.
  All properties must be discovered through experimentation.

Symbol counts:
- count(symbol): Current count of 'W', 'A', 'B', or 'K' in the state

Grid-Phase probes (for detecting parity-dependent patterns):
- count_even(symbol): Count of symbol at EVEN indices (0, 2, 4, ...)
- count_odd(symbol): Count of symbol at ODD indices (1, 3, 5, ...)
- count_at_parity(symbol, parity): Count at indices where index % 2 == parity
  (parity=0 for even, parity=1 for odd)

  Example: If interactions depend on whether a cell is at an even or odd position,
  these instruments will reveal it. Test if count_even('A') behaves differently
  from count_odd('A').

Spatial measurements:
- grid_length: The total size of the grid
- leftmost(symbol): Index of first occurrence (-1 if none)
- rightmost(symbol): Index of last occurrence (-1 if none)
- max_gap(symbol): Length of longest contiguous run of a symbol
- spread(symbol): Distance between first and last occurrence

Pattern detection:
- adjacent_pairs(s1, s2): Count of instances where s1 is immediately followed by s2
- transition_indicator: A count related to future state transitions

Neighborhood Window (local context microscope):
- count_pattern(pattern): Count cells whose 3-cell neighborhood matches 'pattern'
  Pattern is a 3-character string: [left_neighbor, center, right_neighbor]
  Examples:
    count_pattern('AWB')  // Count positions with 'A' to left, 'W' at center, 'B' to right
    count_pattern('WKW')  // Count K cells with W neighbors on both sides
    count_pattern('AWA')  // Count 'W' cells between two A states

  Use this to discover how local configurations predict future events.
  For example, if count_pattern('AWB') at time t correlates with count('K') at t+1,
  you've discovered something about when K states form.

=== TEMPLATES (claim structure types) ===

- invariant: ∀t∈[0..T]: claim holds at every timestep
- monotone: ∀t∈[0..T-1]: comparing f(t+1) to f(t) (increasing or decreasing)
- implication_step: ∀t∈[0..T-1]: P(t) → Q(t+1)
- implication_state: ∀t∈[0..T]: P(t) → Q(t)
- eventually: P(t0) → ∃t∈[t0..t0+H]: Q(t) within horizon H
- symmetry_commutation: evolve(Transform(S), T) == Transform(evolve(S, T))
- bound: ∀t∈[0..T]: f(t) op k
- local_transition: ∀t,i: cell[i]==trigger → cell[i] satisfies result at t+1

=== AVAILABLE TRANSFORMS ===

Transforms can be applied to states. Their effects must be discovered empirically:
- mirror_swap
- shift_k (parameterized by integer k)
- mirror_only
- swap_only

=== LOCAL_TRANSITION TEMPLATE (MICRO-LEVEL RULES) ===

For per-cell rules about what happens at individual positions:
- Expresses: ∀t,i: if state[i] == trigger_symbol at t, then state[i] result_op result_symbol at t+1
- You do NOT need count() functions or claim_ast for this template
- The harness evaluates every cell index i automatically

REQUIRED FIELDS for local_transition:
  "template": "local_transition",
  "trigger_symbol": "<symbol>",    // The symbol at cell i at time t (one of: "W", "A", "B", "K")
  "result_op": "==" or "!=",       // How cell i compares at time t+1
  "result_symbol": "<symbol>",     // The expected symbol at cell i at time t+1
  "observables": [],               // Leave empty - not needed
  "claim_ast": null,               // Leave null - the harness uses trigger/result fields directly

OPTIONAL: neighbor_pattern for CONTEXT-DEPENDENT rules:
  "neighbor_pattern": "<3-char>",  // The required neighborhood [left,center,right] (e.g., "AWB")

OPTIONAL: required_parity for INDEX-PARITY rules:
  "required_parity": 0 or 1,       // 0 = even indices only, 1 = odd indices only

When neighbor_pattern is specified, the rule only applies to positions where
neighbor_config(i) == neighbor_pattern.

When required_parity is specified, the rule only applies to positions where
i % 2 == required_parity. This enables parity-dependent rules like:
"A-states at even indices become B" rather than "all A-states".

You can combine both: neighbor_pattern AND required_parity for highly specific rules.

EXAMPLES:
  "Every K disappears immediately" → trigger_symbol="K", result_op="!=", result_symbol="K"
  "W cells remain W" → trigger_symbol="W", result_op="==", result_symbol="W"
  "A-states persist" → trigger_symbol="A", result_op="==", result_symbol="A"

  CONTEXT-DEPENDENT (with neighbor_pattern):
  "W cells with AWB neighborhood become K" → trigger_symbol="W", neighbor_pattern="AWB", result_op="==", result_symbol="K"
  "A-states in AWW neighborhood stay A" → trigger_symbol="A", neighbor_pattern="AWW", result_op="==", result_symbol="A"

  PARITY-DEPENDENT (with required_parity):
  "A-states at even indices become B" → trigger_symbol="A", required_parity=0, result_op="==", result_symbol="B"
  "B-states at odd indices become A" → trigger_symbol="B", required_parity=1, result_op="==", result_symbol="A"

=== CLAIM AST FORMAT (REQUIRED) ===

Claims must be structured JSON ASTs:

AST Node Types:
- Constant: {"const": 5}
- Time variable: {"var": "t"}
- Time t+1: {"t_plus_1": true}
- Observable at time: {"obs": "<name>", "t": <time_node>}
- Binary operation: {"op": "<op>", "lhs": <node>, "rhs": <node>}
- Unary not: {"op": "not", "arg": <node>}

Operators: +, -, *, /, ==, !=, <, <=, >, >=, =>, and, or, not

=== DERIVED EXPRESSIONS ===

You can combine primitives to create derived observables using: +, -, *
Example: {"name": "combined", "expr": "count('>') + count('<')"}

Experiment with different combinations to discover which quantities
follow simple rules across state transitions.

=== AST EXAMPLES ===

Invariant "Q(t) == Q(0)":
  {"op": "==", "lhs": {"obs": "Q", "t": {"var": "t"}}, "rhs": {"obs": "Q", "t": {"const": 0}}}

Monotone "M(t+1) <= M(t)":
  {"op": "<=", "lhs": {"obs": "M", "t": {"t_plus_1": true}}, "rhs": {"obs": "M", "t": {"var": "t"}}}

Implication "P(t) > 0 => R(t+1) == 0":
  {"op": "=>",
   "lhs": {"op": ">", "lhs": {"obs": "P", "t": {"var": "t"}}, "rhs": {"const": 0}},
   "rhs": {"op": "==", "lhs": {"obs": "R", "t": {"t_plus_1": true}}, "rhs": {"const": 0}}}

For symmetry_commutation: include "transform" field, claim_ast can be null

=== OUTPUT FORMAT ===

Your response MUST be a JSON object with two fields:

{
  "research_log": "Your CUMULATIVE SUMMARY — a structured, living document you UPDATE each iteration (see below).",
  "candidate_laws": [
    {
      "law_id": "unique_identifier",
      "template": "invariant",
      "quantifiers": {"T": 50},
      "observables": [{"name": "Q", "expr": "count('A') + count('B')"}],
      "claim_ast": {"op": "==", "lhs": {"obs": "Q", "t": {"var": "t"}}, "rhs": {"obs": "Q", "t": {"const": 0}}},
      "forbidden": "exists t where Q(t) != Q(0)",
      "proposed_tests": [{"family": "random_density_sweep", "params": {"cases": 100}}]
    }
  ]
}

=== CUMULATIVE SUMMARY (research_log) ===

The research_log is NOT a diary. It is a CUMULATIVE UNIFIED PICTURE of everything you
know about this universe, updated each iteration. You are a Principal Investigator
maintaining a running state of the field — not a lab technician recording daily notes.

Your research_log MUST use exactly these four sections (use the headers as shown, max 500 words total).
Every section MUST be present every iteration, even if empty (write "None yet." if empty).

THE STANDARD MODEL (Fixed Laws):
Invariants, symmetries, and conservation laws with strong support (high-power PASS results).
Once a result appears here, STOP re-proposing laws that test it. Build on it instead.
Only remove an entry if new evidence actively contradicts it — and if so, move it to
THE FALSIFICATION GRAVEYARD with a note.
Format: "- [name]: [statement] (supported by: law_xxx, law_yyy)"

THE REACTION MANUAL (Local Rules):
Confirmed local transition rules — what happens when specific spatial configurations
are encountered. This is the operational handbook of the universe's dynamics.
Format: "- [input pattern] -> [output pattern] (supported by: law_xxx)"

THE FALSIFICATION GRAVEYARD:
Hypotheses that have been CONCLUSIVELY DISPROVEN by counterexamples. This section
exists to prevent you from re-proposing dead ideas. Every entry records: what was
believed, what killed it, and why it cannot be revived.
DO NOT remove entries from this section. It is append-only.
Format: "- [dead hypothesis]: DISPROVEN by [law_xxx/counterexample]. Reason: [why]"

THE FRONTIER (Active Anomalies):
Current contradictions, open questions, and hypotheses you are actively testing.
This is where your candidate_laws should come from — not from re-proving settled science.
Format: "- [question/anomaly]: [current evidence for/against]"

IMPORTANT: If a previous cumulative summary is provided, UPDATE it — do not rewrite from
scratch. Promote results from THE FRONTIER to THE STANDARD MODEL or REACTION MANUAL when
evidence is strong. Move disproven ideas to THE GRAVEYARD. The summary should show clear
iteration-over-iteration progress."""


class PromptBuilder:
    """Builds prompts for law proposal from memory and contract."""

    def __init__(
        self,
        max_token_budget: int = 8000,
        include_counterexamples: bool = True,
        scrambler: "SymbolScrambler | None" = None,
    ):
        """Initialize prompt builder.

        Args:
            max_token_budget: Maximum tokens for prompt
            include_counterexamples: Whether to include counterexample gallery
            scrambler: Symbol scrambler for integrity shielding (optional)
        """
        self.max_token_budget = max_token_budget
        self.include_counterexamples = include_counterexamples

        # Initialize scrambler (use default if not provided)
        if scrambler is None:
            from src.proposer.scrambler import get_default_scrambler
            self._scrambler = get_default_scrambler()
        else:
            self._scrambler = scrambler

    def _scramble_state(self, state: str) -> str:
        """Convert physical state to abstract representation."""
        return self._scrambler.to_abstract(state)

    def _scramble_expr(self, expr: str) -> str:
        """Convert physical observable expression to abstract."""
        return self._scrambler.translate_observable_expr(expr, to_physical=False)

    def build(
        self,
        contract: UniverseContract,
        memory: DiscoveryMemorySnapshot,
        request_count: int | None = None,
        target_templates: list[str] | None = None,
        exclude_templates: list[str] | None = None,
        use_adaptive_count: bool = True,
    ) -> str:
        """Build a prompt for law proposal.

        The prompt is structured with STATIC content first (for API caching)
        and DYNAMIC content (discoveries) last.

        Args:
            contract: Universe contract
            memory: Discovery memory snapshot
            request_count: Number of laws to request (if None and use_adaptive_count=True,
                          computed adaptively based on discovery state)
            target_templates: Templates to focus on (optional)
            exclude_templates: Templates to avoid (optional)
            use_adaptive_count: Whether to adapt request count based on failure rate

        Returns:
            Formatted prompt string
        """
        sections = []

        # Compute request count - either explicit, adaptive, or default
        adaptation_reason = None
        if request_count is None and use_adaptive_count:
            request_count, adaptation_reason = self.compute_adaptive_request_count(memory)
        elif request_count is None:
            request_count = 5  # Default

        # === RESEARCH LOG (your previous notes) ===
        # Inject the LLM's previous research log for continuity
        if memory.previous_research_log:
            sections.append(self._build_research_log_section(memory.previous_research_log))

        # === STATIC SECTION (for caching) ===
        # Universe capabilities - this rarely changes
        sections.append(self._build_capabilities_section(contract))

        # Expression language reminder
        sections.append(self._build_expression_language_section())

        # Request section - includes phase-specific guidance
        sections.append(self._build_request_section(
            request_count, target_templates, exclude_templates, adaptation_reason
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

        # Micro-scale guidance (when global laws are struggling)
        if self._needs_microscale_guidance(memory):
            sections.append(self._build_microscale_guidance_section())

        return "\n\n".join(sections)

    def _build_expression_language_section(self) -> str:
        """Build expression language and AST format reminder section.

        INTEGRITY NOTE: This section intentionally does NOT label any
        expressions as "conserved" or suggest which combinations might
        be invariant. The LLM must discover these through falsification.

        Uses abstract symbols (_, A, B, K) to prevent inferring physics from glyphs.
        """
        # Use abstract symbols from scrambler
        symbols = self._scrambler.abstract_symbols
        return f"""=== CLAIM AST FORMAT ===

Claims must be structured JSON ASTs. Observable expressions are defined separately.

EXPRESSION PRIMITIVES:
  count('{symbols[0]}'), count('{symbols[1]}'), count('{symbols[2]}'), count('{symbols[3]}')
  count_even(sym), count_odd(sym)          // Grid-phase: count at even/odd indices
  count_at_parity(sym, 0), count_at_parity(sym, 1)  // Explicit parity
  count_pattern('{symbols[1]}{symbols[0]}{symbols[2]}'), count_pattern('{symbols[0]}{symbols[3]}{symbols[0]}')  // Neighborhood patterns (3-char)
  grid_length
  transition_indicator
  leftmost(sym), rightmost(sym), spread(sym), max_gap(sym)
  adjacent_pairs(s1, s2)

ARITHMETIC: +, -, * (division / is NOT supported - rewrite algebraically)

CLAIM AST NODES:
  Constant:     {{"const": 5}}
  Time var:     {{"var": "t"}}
  Time t+1:     {{"t_plus_1": true}}
  Observable:   {{"obs": "<name>", "t": <time_node>}}
  Binary op:    {{"op": "<op>", "lhs": <node>, "rhs": <node>}}
  Unary not:    {{"op": "not", "arg": <node>}}

  Operators: +, -, *, /, ==, !=, <, <=, >, >=, =>, and, or, not

AST EXAMPLES:

  Q(t) == Q(0) [compare Q at time t to Q at time 0]:
    {{"op": "==", "lhs": {{"obs": "Q", "t": {{"var": "t"}}}}, "rhs": {{"obs": "Q", "t": {{"const": 0}}}}}}

  R(t) > 0 => S(t+1) == 0 [implication]:
    {{"op": "=>",
     "lhs": {{"op": ">", "lhs": {{"obs": "R", "t": {{"var": "t"}}}}, "rhs": {{"const": 0}}}},
     "rhs": {{"op": "==", "lhs": {{"obs": "S", "t": {{"t_plus_1": true}}}}, "rhs": {{"const": 0}}}}}}

For symmetry_commutation: include "transform" field, claim_ast can be null."""

    def get_system_instruction(self) -> str:
        """Get the system instruction for the LLM."""
        return SYSTEM_INSTRUCTION

    def _build_capabilities_section(self, contract: UniverseContract) -> str:
        """Build universe capabilities section.

        Uses abstract symbols from scrambler to prevent physics inference.
        """
        caps = contract.capabilities
        # Use abstract symbols instead of physical ones
        abstract_symbols = self._scrambler.abstract_symbols
        lines = [
            "=== UNIVERSE CAPABILITIES ===",
            f"Symbols: {', '.join(abstract_symbols)}",
            f"State: {contract.state_representation}",
            "",
            "Primitive observables:",
        ]
        for obs in caps.get("primitive_observables", []):
            # Scramble any symbols in observable descriptions
            scrambled_obs = self._scramble_expr(obs)
            lines.append(f"  - {scrambled_obs}")

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

    def _build_research_log_section(self, previous_log: str) -> str:
        """Build section containing the LLM's previous cumulative summary.

        This enables the LLM to maintain continuity across iterations,
        building on settled science rather than re-discovering it.

        INTEGRITY NOTE: The content of the log is written BY the LLM itself
        in the previous iteration. We are providing a storage buffer, not
        injecting external knowledge.
        """
        return f"""=== YOUR PREVIOUS CUMULATIVE SUMMARY ===

Below is your cumulative summary from the last iteration. UPDATE this document
rather than rewriting from scratch:
- Promote well-supported hypotheses from THE FRONTIER to THE STANDARD MODEL or THE REACTION MANUAL
- Add newly discovered transitions to THE REACTION MANUAL
- Move decisively falsified ideas from THE FRONTIER to THE FALSIFICATION GRAVEYARD
- Add new contradictions or surprises to THE FRONTIER
- Do NOT re-propose laws for items already in THE STANDARD MODEL
- NEVER remove entries from THE FALSIFICATION GRAVEYARD — it is append-only

{previous_log}

=== END OF PREVIOUS SUMMARY ==="""

    def _build_accepted_section(self, laws: list[dict[str, Any]]) -> str:
        """Build accepted laws section."""
        lines = [
            "=== ACCEPTED LAWS (not yet falsified) ===",
            "These laws survived our tests so far. Do not repeat these.",
            "",
        ]
        for law in laws:
            if law is None:
                continue
            template = law.get('template', '?')
            law_id = law.get('law_id', '?')
            claim = law.get('claim', '?')
            # Scramble the claim text
            scrambled_claim = self._scramble_expr(claim) if claim else claim
            lines.append(f"- [{template}] {law_id}: {scrambled_claim}")
            # Show observable definitions (scrambled) to prevent equivalent re-proposals
            if law.get("observables"):
                obs_strs = [
                    f"{o['name']}={self._scramble_expr(o['expr'])}"
                    for o in law["observables"] if o
                ]
                lines.append(f"    observables: {', '.join(obs_strs)}")
        return "\n".join(lines)

    def _build_falsified_section(self, laws: list[dict[str, Any]]) -> str:
        """Build falsified laws section.

        NOTE: This section teaches the scientific method (learn from failures),
        not the physics. Telling the LLM to study WHY laws fail is methodology.
        """
        lines = [
            "=== FALSIFIED LAWS (STUDY THESE CAREFULLY) ===",
            "These laws are DEFINITELY FALSE. Each failure is valuable data.",
            "",
            "For each failed law, ask yourself:",
            "- WHY did this fail? What does the counterexample reveal?",
            "- What assumption was I making that turned out to be wrong?",
            "- Is there a simpler, more universal law that would avoid this failure?",
            "",
        ]
        for law in laws:
            if law is None:
                continue
            cx = law.get("counterexample", {}) or {}
            template = law.get('template', '?')
            law_id = law.get('law_id', '?')
            claim = law.get('claim', '?')
            # Scramble the claim text
            scrambled_claim = self._scramble_expr(claim) if claim else claim
            lines.append(f"- [{template}] {law_id}: {scrambled_claim}")
            # Show observable definitions (scrambled)
            if law.get("observables"):
                obs_strs = [
                    f"{o['name']}={self._scramble_expr(o['expr'])}"
                    for o in law.get("observables", []) if o
                ]
                if obs_strs:
                    lines.append(f"    observables: {', '.join(obs_strs)}")
            if cx:
                initial_state = cx.get('initial_state', '?')
                scrambled_state = self._scramble_state(initial_state) if initial_state != '?' else initial_state
                lines.append(f"    COUNTEREXAMPLE: state='{scrambled_state}', failed at t={cx.get('t_fail', '?')}")
        return "\n".join(lines)

    def _build_unknown_section(self, laws: list[dict[str, Any]]) -> str:
        """Build unknown laws section."""
        lines = ["=== UNKNOWN LAWS (capability gaps) ==="]
        for law in laws:
            if law is None:
                continue
            template = law.get('template', '?')
            law_id = law.get('law_id', '?')
            reason = law.get("reason_code", "unknown")
            lines.append(f"- [{template}] {law_id}: {reason}")
        return "\n".join(lines)

    def _build_counterexamples_section(self, examples: list[dict[str, Any]]) -> str:
        """Build counterexamples gallery section.

        Shows falsified claims with their counterexamples so the LLM can
        learn what kinds of claims fail and why.

        NOTE: This section teaches METHODOLOGY (how to learn from failures),
        not PHYSICS (what the laws actually are). This is appropriate -
        a Popperian scientist should understand the scientific method.
        """
        lines = [
            "=== COUNTEREXAMPLE GALLERY (YOUR MOST VALUABLE DATA) ===",
            "",
            "IMPORTANT: Each counterexample is a FACT about this universe.",
            "- The claim is DEFINITELY FALSE (we have proof)",
            "- The counterexample state and trajectory reveal true patterns",
            "- Study the trajectory to understand WHY the law failed",
            "- Ask: What assumption was wrong? What's the real pattern?",
            "",
            "HOW TO LEARN FROM FAILURES:",
            "- DO NOT just avoid these exact laws - learn the DEEPER LESSON",
            "- If many laws about a quantity fail, your model of it is probably wrong",
            "- Look for patterns in what kinds of states cause failures",
            "- The simplest explanation that fits ALL counterexamples is best",
            "",
        ]
        for i, cx in enumerate(examples[:10]):  # Limit to 10
            if cx is None:
                continue
            template = cx.get("template", "?")
            claim = cx.get("claim", "?")
            forbidden = cx.get("forbidden", "")
            initial = cx.get("initial_state", "?")
            t_fail = cx.get("t_fail", "?")

            # Scramble the claim and forbidden text (which may reference symbols)
            scrambled_claim = self._scramble_expr(claim) if claim else claim
            scrambled_forbidden = self._scramble_expr(forbidden) if forbidden else forbidden

            # Scramble the initial state
            scrambled_initial = self._scramble_state(initial) if initial and initial != "?" else initial

            lines.append(f"{i+1}. [{template}] CLAIM: {scrambled_claim}")
            if scrambled_forbidden:
                lines.append(f"   FORBIDDEN: {scrambled_forbidden}")
            lines.append(f"   FALSIFIED BY: state='{scrambled_initial}' at t={t_fail}")

            # Show trajectory if available (scrambled)
            trajectory = cx.get("trajectory_excerpt")
            if trajectory:
                if isinstance(trajectory, str):
                    import json as json_lib
                    try:
                        trajectory = json_lib.loads(trajectory)
                    except Exception:
                        trajectory = None
                if trajectory and isinstance(trajectory, list):
                    # Scramble each state in the trajectory
                    scrambled_excerpt = [self._scramble_state(s) for s in trajectory[:5]]
                    lines.append(f"   TRAJECTORY: {' → '.join(scrambled_excerpt)}")
            lines.append("")
        return "\n".join(lines)

    def compute_adaptive_request_count(
        self,
        memory: DiscoveryMemorySnapshot,
        default_count: int = 5,
    ) -> tuple[int, str]:
        """Compute adaptive law request count based on discovery state.

        POPPERIAN PRINCIPLE: Failed laws are data. If the AI generates many laws
        based on a flawed assumption, all will fail before the AI can learn from
        the first counterexample. This "Theory Over-fitting" wastes iterations.

        By adapting the count, we ensure:
        1. High failure rates → slower proposals → more learning from counterexamples
        2. Ambiguous claims → focused proposals → fix syntax before scaling
        3. Stable discovery → higher throughput → explore more of hypothesis space

        Args:
            memory: Current discovery memory snapshot
            default_count: Default count when no adaptation needed

        Returns:
            Tuple of (request_count, reason_for_adaptation)
        """
        # Count ambiguous claims (syntax/format errors)
        # Defensive: filter out None values that may be in the list
        ambiguous_count = sum(
            1 for law in memory.unknown_laws
            if law is not None and law.get("reason_code", "").lower() in ("ambiguous_claim", "ambiguous", "compilation_error")
        )

        # Count recent failures (last batch)
        recent_failures = len(memory.falsified_laws)
        recent_passes = len(memory.accepted_laws)
        total_evaluated = recent_failures + recent_passes

        # Calculate failure rate (avoid division by zero)
        failure_rate = recent_failures / max(total_evaluated, 1)

        # PHASE 1: Syntax Fix Mode - Many ambiguous claims
        # The AI is struggling with the format, not the physics
        if ambiguous_count > 5:
            return 3, "syntax_focus"

        # PHASE 2: Learning Mode - High failure rate
        # The AI has a flawed model and needs to digest counterexamples
        if failure_rate > 0.7 and total_evaluated > 5:
            return 4, "high_failure_learning"

        # PHASE 3: Refinement Mode - Some failures, some successes
        # The AI is making progress but still refining
        if failure_rate > 0.4 and total_evaluated > 3:
            return 5, "refinement"

        # PHASE 4: Exploration Mode - Low failure rate, stable discovery
        # The AI has a good model, can explore more hypotheses
        if failure_rate < 0.3 and recent_passes > 3:
            return 8, "exploration"

        # Default: Balanced mode
        return default_count, "balanced"

    def _needs_microscale_guidance(self, memory: DiscoveryMemorySnapshot) -> bool:
        """Determine if the AI needs a nudge toward micro-scale analysis.

        This is triggered when:
        - There are ambiguous claims (global counts can't express the pattern)
        - Many global laws are failing but local_transition hasn't been tried

        NOTE: This is METHODOLOGICAL guidance (suggesting a scale of inquiry),
        not PHYSICS guidance (telling what to find at that scale).
        """
        # Check for ambiguous claims in unknown laws
        # Defensive: filter out None values
        ambiguous_count = sum(
            1 for law in memory.unknown_laws
            if law is not None and law.get("reason_code", "").lower() in ("ambiguous_claim", "ambiguous")
        )

        # Check if local_transition template is underexplored
        # Defensive: filter out None values
        all_laws = memory.accepted_laws + memory.falsified_laws + memory.unknown_laws
        local_attempted = sum(
            1 for law in all_laws
            if law is not None and law.get("template") == "local_transition"
        )

        # Trigger guidance if ambiguous claims exist OR global templates dominate
        has_ambiguous = ambiguous_count > 0
        local_underexplored = local_attempted < 3 and len(memory.falsified_laws) > 5

        return has_ambiguous or local_underexplored

    def _build_microscale_guidance_section(self) -> str:
        """Build guidance suggesting micro-scale analysis.

        INTEGRITY NOTE: This suggests the SCALE of inquiry (cells vs aggregates),
        not the CONTENT of discovery. It's like saying "try the microscope"
        without saying what you'll see through it.

        Uses abstract symbols from scrambler.
        """
        # Get abstract symbols for examples
        symbols = self._scrambler.abstract_symbols  # [_, A, B, K]
        bg, c1, c2, kin = symbols[0], symbols[1], symbols[2], symbols[3]

        return f"""=== METHODOLOGICAL SUGGESTION: MICRO-SCALE ANALYSIS ===

Your global observables (aggregate counts) may be missing important patterns.
Consider using the local_transition template to observe individual cell behavior.

KEY INSIGHT: Global counts cannot track IDENTITY. Even if a total count stays
the same, individual cells may be changing in ways that matter.

HOW TO USE local_transition:
The harness automatically evaluates every cell index i across all timesteps t.
You define WHAT to look for, the harness tests WHERE and WHEN.

REQUIRED JSON FORMAT:
{{
  "law_id": "your_id",
  "template": "local_transition",
  "trigger_symbol": "{kin}",        // The symbol at cell i at time t
  "result_op": "!=",            // Either "==" or "!="
  "result_symbol": "{kin}",         // The expected symbol at cell i at time t+1
  "neighbor_pattern": null,     // OPTIONAL: 3-char pattern like "{c1}{bg}{c2}" for context-dependent rules
  "quantifiers": {{"T": 50}},
  "preconditions": [],
  "observables": [],            // Leave empty - not needed for this template
  "claim_ast": null,            // Leave null - harness uses trigger/result fields
  "claim": "Human-readable description",
  "forbidden": "What would falsify this"
}}

OPTIONAL: For CONTEXT-DEPENDENT rules, add neighbor_pattern:
  "neighbor_pattern": "{c1}{bg}{c2}"     // Rule only applies when neighbor_config(i)=="{c1}{bg}{c2}"

OPTIONAL: For PARITY-DEPENDENT rules, add required_parity:
  "required_parity": 0              // Rule only applies at even indices (i % 2 == 0)
  "required_parity": 1              // Rule only applies at odd indices (i % 2 == 1)

WHAT TO EXPLORE:
- What happens to each symbol type from one timestep to the next?
- Does a symbol at position i "become" a specific other symbol at position i?
- Do neighborhood patterns (3-cell windows) determine what happens next?
- Does index parity (even vs odd) affect transitions differently?
- The simplest per-cell rule that survives falsification may reveal deeper structure."""

    def _build_request_section(
        self,
        count: int,
        target_templates: list[str] | None,
        exclude_templates: list[str] | None,
        adaptation_reason: str | None = None,
    ) -> str:
        """Build request section.

        Args:
            count: Number of laws to request
            target_templates: Templates to focus on
            exclude_templates: Templates to avoid
            adaptation_reason: Why the count was adapted (helps AI understand phase)
        """
        lines = [
            "=== YOUR TASK ===",
            f"Propose {count} NEW candidate laws.",
        ]

        # Add phase-specific guidance based on adaptation reason
        if adaptation_reason == "syntax_focus":
            lines.extend([
                "",
                "NOTE: Many recent proposals had syntax errors. Focus on:",
                "- Using the EXACT JSON format shown in examples",
                "- Including all required fields for your chosen template",
                "- Fewer, correctly-formatted laws > many malformed ones",
            ])
        elif adaptation_reason == "high_failure_learning":
            lines.extend([
                "",
                "NOTE: Many recent laws were falsified. Before proposing more:",
                "- Study the COUNTEREXAMPLE GALLERY carefully",
                "- Ask: What assumption did ALL the failed laws share?",
                "- Propose laws that account for the counterexamples you've seen",
                "- Quality over quantity - each law should reflect a new insight",
            ])
        elif adaptation_reason == "refinement":
            lines.extend([
                "",
                "NOTE: Some laws passed, some failed. You're making progress.",
                "- Build on what's working (accepted laws)",
                "- Learn from what's not (study counterexamples)",
                "- Look for patterns that unify your successes",
            ])
        elif adaptation_reason == "exploration":
            lines.extend([
                "",
                "NOTE: Your model is performing well. Time to explore.",
                "- Test the boundaries of your accepted laws",
                "- Look for deeper patterns that could unify multiple laws",
                "- Consider: Are there simpler formulations that capture the same truth?",
            ])

        lines.extend([
            "",
            "Requirements:",
            "- Each law must be testable and falsifiable",
            "- Include a forbidden condition describing what would disprove it",
            "- Do not repeat accepted or falsified laws",
            "- Prioritize RISKY, easily-falsifiable laws over safe ones",
        ])

        if target_templates:
            lines.append(f"- Focus on templates: {', '.join(target_templates)}")

        if exclude_templates:
            lines.append(f"- Avoid templates: {', '.join(exclude_templates)}")

        lines.extend([
            "",
            "Output a JSON object with:",
            '  "research_log": your UPDATED cumulative summary (4 sections, max 500 words)',
            '  "candidate_laws": array of CandidateLaw objects',
            "",
            "Your research_log MUST contain these four sections:",
            "- THE STANDARD MODEL (Fixed Laws): Settled invariants and conservation laws",
            "- THE REACTION MANUAL (Local Rules): Confirmed local transitions",
            "- THE FALSIFICATION GRAVEYARD: Dead hypotheses (append-only, never re-propose these)",
            "- THE FRONTIER (Active Anomalies): Open questions driving your new candidate_laws",
            "",
            "Focus your candidate_laws on THE FRONTIER, not on re-proving THE STANDARD MODEL.",
        ])

        return "\n".join(lines)

    def estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for a prompt."""
        # Rough estimate: 1 token per 4 characters
        return len(prompt) // 4
