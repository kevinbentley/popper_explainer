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

=== TEMPLATES (claim structure types) ===

- invariant: ∀t∈[0..T]: claim holds at every timestep
- monotone: ∀t∈[0..T-1]: comparing f(t+1) to f(t) (increasing or decreasing)
- implication_step: ∀t∈[0..T-1]: P(t) → Q(t+1)
- implication_state: ∀t∈[0..T]: P(t) → Q(t)
- eventually: P(t0) → ∃t∈[t0..t0+H]: Q(t) within horizon H
- bound: ∀t∈[0..T]: f(t) op k
- local_transition: ∀t,i: cell[i]==trigger → cell[i] satisfies result at t+1

=== LOCAL_TRANSITION TEMPLATE (MICRO-LEVEL RULES) ===

For per-cell rules about what happens at individual positions:
- Expresses: ∀t,i: if state[i] == trigger_symbol at t, then state[i] result_op result_symbol at t+1
- The harness evaluates every cell index i automatically

REQUIRED FIELDS for local_transition:
  "template": "local_transition",
  "trigger_symbol": "<symbol>",    // The symbol at cell i at time t
  "result_op": "==" or "!=",       // How cell i compares at time t+1
  "result_symbol": "<symbol>",     // The expected symbol at cell i at time t+1
  "observables": [],               // Leave empty - not needed
  "claim_ast": null,               // Leave null - harness uses trigger/result fields

=== OUTPUT FORMAT ===

Your response MUST be a JSON object with these fields:

{
  "probes": [
    {
      "probe_id": "my_measurement",
      "code": "def probe(S):\n    return sum(1 for c in S if c != 'W')",
      "hypothesis": "Counts all non-background cells"
    }
  ],
  "research_log": "Your CUMULATIVE SUMMARY — a structured, living document you UPDATE each iteration (see below).",
  "candidate_laws": [
    {
      "law_id": "unique_identifier",
      "template": "invariant",
      "quantifiers": {"T": 50},
      "observables": [{"name": "Q", "probe_id": "my_measurement"}],
      "claim": "Q is constant over time",
      "forbidden": "exists t where Q(t) != Q(0)",
      "proposed_tests": [{"family": "random_density_sweep", "params": {"cases": 100}}]
    }
  ],
  "simulation_requests": [
    {"state": "WWABWWW", "T": 5}
  ]
}

PROBES are your instruments. You MUST define probes to measure the universe and
reference them by probe_id in your observables. Each observable needs a "name"
and a "probe_id" pointing to a probe you defined (either in this response or a
previous iteration).

=== CLAIM AST (for templates that need it) ===

For invariant, monotone, and bound templates, the claim semantics are determined by
the template type itself — you do not need to write a claim_ast.

For implication_step, implication_state, and eventually templates, you need a claim_ast
to express the logical relationship between your observables. The AST uses JSON nodes:

  Constant:     {"const": 5}
  Time var:     {"var": "t"}
  Time t+1:     {"t_plus_1": true}
  Observable:   {"obs": "<name>", "t": <time_node>}
  Comparison:   {"op": "<op>", "lhs": <node>, "rhs": <node>}
  Logical:      {"op": "=>", "lhs": <condition>, "rhs": <conclusion>}
  Not:          {"op": "not", "arg": <node>}

  Comparison operators: ==, !=, <, <=, >, >=
  Logical operators: =>, and, or, not
  Arithmetic: +, -, *

Example — "P(t) > 0 implies Q(t+1) == 0":
  {"op": "=>",
   "lhs": {"op": ">", "lhs": {"obs": "P", "t": {"var": "t"}}, "rhs": {"const": 0}},
   "rhs": {"op": "==", "lhs": {"obs": "Q", "t": {"t_plus_1": true}}, "rhs": {"const": 0}}}

=== SIMULATION REQUESTS (OPTIONAL) ===

You may include a "simulation_requests" field to observe how specific configurations evolve.
Each request specifies an initial state and a number of time steps. Results will be provided
in the next iteration under "SIMULATION RESULTS".

Rules:
- "state": A string using valid symbols (W, A, B, K). Must be non-empty.
- "T": A non-negative integer (number of time steps to simulate). Maximum 500.
- Configurations that are invalid at t=0 will return an error message rather than a state sequence.
  You cannot predict which configurations are invalid — you must discover this empirically.
- Use simulations to CHECK your hypotheses about how the universe evolves, not as a substitute
  for proposing falsifiable laws. Simulations are observations; laws are explanations.

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
        probe_registry=None,
    ):
        """Initialize prompt builder.

        Args:
            max_token_budget: Maximum tokens for prompt
            include_counterexamples: Whether to include counterexample gallery
            scrambler: Symbol scrambler for integrity shielding (optional)
            probe_registry: Optional ProbeRegistry for probe-based discovery mode
        """
        self.max_token_budget = max_token_budget
        self.include_counterexamples = include_counterexamples
        self._probe_registry = probe_registry

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

        # === SIMULATION RESULTS (from previous iteration) ===
        if memory.previous_simulation_results:
            sections.append(self._build_simulation_results_section(
                memory.previous_simulation_results
            ))

        # === STATIC SECTION (for caching) ===
        # Universe capabilities - this rarely changes
        sections.append(self._build_capabilities_section(contract))

        # Probe system section (when probe registry is available)
        if self._probe_registry is not None:
            sections.append(self._build_probe_system_section())
            probe_library = self._probe_registry.to_prompt_summary()
            if probe_library:
                sections.append(f"=== PROBE LIBRARY ===\n\n{probe_library}")

        # Priority research directions from reflection engine
        if memory.priority_research_directions:
            sections.append(self._build_research_directions_section(
                memory.priority_research_directions
            ))

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

    def _build_probe_system_section(self) -> str:
        """Build the PROBE SYSTEM section explaining probe-based measurement.

        This section tells the LLM how to write custom Python measurement functions
        (probes) that run in a sandbox. Probes replace the fixed expression primitives
        with arbitrary measurement logic.
        """
        symbols = self._scrambler.abstract_symbols
        bg, c1, c2, kin = symbols[0], symbols[1], symbols[2], symbols[3]
        return f"""=== PROBE SYSTEM (Your Measurement Instruments) ===

You can define CUSTOM MEASUREMENT FUNCTIONS (probes) that measure anything about a state.
Probes are Python functions that take a state (list of single-character strings) and return a number.

HOW TO DEFINE A PROBE:
Include a "probes" array in your JSON response:
{{
  "probes": [
    {{
      "probe_id": "total_particles",
      "code": "def probe(S):\\n    return sum(1 for c in S if c != '{bg}')",
      "hypothesis": "Counts all non-background cells"
    }}
  ],
  "candidate_laws": [...]
}}

HOW TO USE A PROBE IN A LAW:
Reference the probe_id in your observable definition:
{{
  "observables": [
    {{"name": "P", "probe_id": "total_particles"}}
  ]
}}

SANDBOX RULES:
- Your function MUST be named 'probe' and take 1 or 2 arguments
- It MUST return an int or float
- Allowed builtins: range, len, abs, min, max, float, int, sum, list, dict, tuple,
  set, enumerate, zip, sorted, reversed, bool, str, round, any, all, map, filter, isinstance
- NOT allowed: import, exec, eval, open, print, or any dunder (__) access
- Maximum execution time: 100ms

THE STATE (S):
- S is a list of single-character strings: ['{bg}', '{c1}', '{bg}', '{c2}', ...]
- The symbols are: {bg}, {c1}, {c2}, {kin}
- len(S) gives the grid length
- S[i] gives the symbol at position i

SINGLE-STATE PROBES (measuring one state):
  "def probe(S):\\n    return sum(1 for c in S if c == '{c1}')"
  → Counts {c1} symbols

  "def probe(S):\\n    return sum(1 for i, c in enumerate(S) if c == '{c1}' and i % 2 == 0)"
  → Counts {c1} at even indices

  "def probe(S):\\n    pairs = 0\\n    for i in range(len(S)-1):\\n        if S[i] == '{c1}' and S[i+1] == '{c2}':\\n            pairs += 1\\n    return pairs"
  → Counts adjacent {c1}{c2} pairs

  "def probe(S):\\n    n = len(S)\\n    return sum(1 for i in range(n) if S[i] == '{bg}' and S[(i-1) % n] == '{c1}' and S[(i+1) % n] == '{c2}')"
  → Counts {bg} cells with {c1} on left and {c2} on right (periodic boundary)

TEMPORAL PROBES (measuring transitions):
You can define probes that see BOTH the current and next state:
  "def probe(S_current, S_next):\\n    return sum(1 for a, b in zip(S_current, S_next) if a != b)"
  → Counts how many cells changed between timesteps

  "def probe(S_current, S_next):\\n    return sum(1 for a, b in zip(S_current, S_next) if a == '{bg}' and b != '{bg}')"
  → Counts cells that went from background to non-background

The system auto-detects 1-param vs 2-param probes from the function signature.
Temporal probes measure how the universe CHANGES between timesteps — they cannot
evaluate the last timestep (no "next" state available), so the harness skips it.

IMPORTANT:
- Probes are your ONLY instruments for measuring the universe
- Each observable in a law MUST reference a probe_id
- Probes you define persist across iterations — reuse them by probe_id
- If a probe errors during registration, the law using it gets UNKNOWN verdict"""

    def get_system_instruction(self) -> str:
        """Get the system instruction for the LLM."""
        return SYSTEM_INSTRUCTION

    def _build_capabilities_section(self, contract: UniverseContract) -> str:
        """Build universe capabilities section.

        Uses abstract symbols from scrambler to prevent physics inference.
        Does NOT expose transforms, built-in observables, or test families
        — the LLM must discover structure through probes.
        """
        # Use abstract symbols instead of physical ones
        abstract_symbols = self._scrambler.abstract_symbols
        lines = [
            "=== UNIVERSE CAPABILITIES ===",
            f"Symbols: {', '.join(abstract_symbols)}",
            f"State: {contract.state_representation}",
            f"Boundary: {contract.config_knobs.get('boundary', 'unknown')}",
            f"Grid length range: {contract.config_knobs.get('grid_length_range', 'unknown')}",
            "",
            "You measure the universe by writing Python probe functions (see PROBE SYSTEM).",
        ]

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

    def _build_simulation_results_section(
        self, results: list[dict[str, Any]]
    ) -> str:
        """Build section showing simulation results from the previous iteration.

        These are responses to the LLM's simulation_requests from last iteration.
        Each result shows the initial state, T, and the resulting state sequence
        (or an error if the configuration was invalid).
        """
        lines = [
            "=== SIMULATION RESULTS (from your previous requests) ===",
            "",
        ]
        for i, result in enumerate(results, 1):
            state = result.get("state", "?")
            t_val = result.get("T", "?")
            if "error" in result:
                lines.append(f"{i}. state=\"{state}\", T={t_val}")
                lines.append(f"   ERROR: {result['error']}")
            else:
                sequence = result.get("state_sequence", [])
                lines.append(f"{i}. state=\"{state}\", T={t_val}")
                if len(sequence) <= 20:
                    lines.append(f"   state_sequence: {' -> '.join(sequence)}")
                else:
                    # Show first 10 and last 5 for long sequences
                    head = " -> ".join(sequence[:10])
                    tail = " -> ".join(sequence[-5:])
                    lines.append(f"   state_sequence: {head} -> ... -> {tail}")

                # Include probe output table if available
                probe_table = result.get("probe_outputs")
                if probe_table and isinstance(probe_table, dict):
                    lines.append("   probe_outputs:")
                    for probe_id, values in probe_table.items():
                        if isinstance(values, list) and len(values) <= 20:
                            lines.append(f"     {probe_id}: {values}")
                        elif isinstance(values, list):
                            lines.append(
                                f"     {probe_id}: [{', '.join(str(v) for v in values[:10])}, "
                                f"..., {', '.join(str(v) for v in values[-3:])}]"
                            )
            lines.append("")
        return "\n".join(lines)

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
            # Show observable definitions to prevent equivalent re-proposals
            if law.get("observables"):
                obs_strs = []
                for o in law["observables"]:
                    if not o:
                        continue
                    if o.get("probe_id"):
                        obs_strs.append(f"{o['name']}=probe:{o['probe_id']}")
                    elif o.get("expr"):
                        obs_strs.append(f"{o['name']}={self._scramble_expr(o['expr'])}")
                if obs_strs:
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

        Suggests the SCALE of inquiry (cells vs aggregates) without
        hinting at what the LLM will find at that scale.
        """
        return """=== METHODOLOGICAL SUGGESTION: MICRO-SCALE ANALYSIS ===

Your aggregate probe measurements may be missing important patterns.
Consider using the local_transition template to observe individual cell behavior.

The local_transition template lets you test per-cell rules:
  ∀t,i: if state[i] == trigger_symbol at t, then state[i] result_op result_symbol at t+1

The harness automatically evaluates every cell index i across all timesteps t.
You define WHAT to look for, the harness tests WHERE and WHEN."""

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

    def _build_research_directions_section(
        self,
        directions: list[dict[str, Any]],
    ) -> str:
        """Build section for priority research directions from reflection engine.

        These directions guide the LLM toward specific experimental conditions
        identified by the periodic auditor/theorist analysis.
        """
        lines = [
            "=== PRIORITY RESEARCH DIRECTIONS ===",
            "The following research directions were identified by systematic analysis",
            "of your accepted laws, falsification graveyard, and anomalies.",
            "Prioritize proposals that address these directions.",
            "",
        ]
        for i, d in enumerate(directions, 1):
            priority = d.get("priority", "medium")
            description = d.get("description", "")
            cmd_type = d.get("command_type", "")
            target = d.get("target_law_id")
            lines.append(f"{i}. [{priority.upper()}] {description}")
            if target:
                lines.append(f"   Target law: {target}")
            if cmd_type:
                lines.append(f"   Type: {cmd_type}")
            lines.append("")
        return "\n".join(lines)

    def estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for a prompt."""
        # Rough estimate: 1 token per 4 characters
        return len(prompt) // 4
