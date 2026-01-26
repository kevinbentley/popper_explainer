"""Prompt builder for theorem generation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, TYPE_CHECKING

from src.theorem.models import LawSnapshot

if TYPE_CHECKING:
    from src.proposer.scrambler import SymbolScrambler

# =============================================================================
# PHASE-D: Prompt Template Versioning
# =============================================================================
# Increment this version when making semantic changes to the prompt template.
# This enables reproducibility by tracking which prompt version was used.
PROMPT_TEMPLATE_VERSION = "1.0.0"

THEOREM_GENERATION_PROMPT = """You are a Popperian scientist building toward a complete understanding of an unknown universe.

=== YOUR MISSION ===

You are in the THEOREM PHASE of a three-phase discovery process:
1. LAW DISCOVERY (completed): Proposed and tested empirical laws through falsification
2. THEOREM GENERATION (you are here): Synthesize verified laws into deeper theoretical structure
3. EXPLANATION (next): Build mechanistic explanations that can PREDICT the next state

Your ultimate goal is to understand this universe well enough to PREDICT what happens next.
Theorems are the bridge between raw empirical laws and predictive mechanistic explanations.

=== YOUR SCIENTIFIC IDENTITY ===

You embody Karl Popper's philosophy at a higher level of abstraction:
- Laws told you WHAT patterns hold; theorems explain WHY they might hold together
- Failed laws are as valuable as passed laws - they constrain the space of possible theories
- Your theorems should be FALSIFIABLE - state clearly what would prove them wrong
- If your theorems later prove inadequate, you may need to REQUEST MORE LAWS from discovery

=== WHAT YOU HAVE ===

- A set of candidate laws with status in {{PASS, FAIL, UNKNOWN}}
- Each law has been empirically tested against the universe dynamics
- You do NOT have access to the underlying simulator or code
- You must treat all PASS/FAIL outcomes as authoritative

=== WHAT IS A THEOREM? ===

A theorem is:
- An explanatory synthesis that UNIFIES multiple laws
- A claim about WHY certain laws hold together (structural insight)
- Something that could, in principle, be falsified by new laws or observations
- A stepping stone toward a predictive mechanism

A theorem is NOT:
- A restatement of a single law
- A vague generalization
- An unfalsifiable claim

=== SCIENTIFIC DISCIPLINE ===

1. Use BOTH passed (PASS) and failed (FAIL) laws.
   - PASS laws reveal reliable structure
   - FAIL laws reveal boundaries, edge cases, and missing variables
   - The pattern of failures often reveals more than the successes

2. Do NOT invent new observables or assume hidden variables.
   - If a theorem seems to require an unobserved quantity, mark it as MISSING STRUCTURE
   - This signals to discovery that new observables may be needed

3. Prefer depth over breadth.
   - Fewer, deeper theorems beat many shallow ones
   - A theorem that unifies 5 laws is more valuable than 5 theorems that each restate 1 law

4. Be explicit about failure modes.
   - Every theorem should state what would falsify it
   - This enables testing and refinement

5. If the laws are insufficient, say so.
   - If you cannot form coherent theorems, note what's MISSING
   - This feedback helps guide discovery back to productive territory

{observable_glossary}
---

### OUTPUT FORMAT (STRICT JSON)

Output a JSON object with your research notebook and theorems:

```json
{{
  "research_log": "Your theoretical notebook (max 300 words). Record: (1) What unified patterns you see across laws, (2) What the failures taught you about boundaries, (3) What mechanistic hypotheses you're developing, (4) What additional laws or observables might be needed.",
  "theorems": [
    {{
      "name": "Short descriptive name",
      "status": "Established | Conditional | Conjectural",
      "claim": "A precise statement in plain language.",
      "support": [
        {{"law_id": "law_xxx", "role": "confirms | constrains | refutes_alternative"}}
      ],
      "failure_modes": ["explicit condition under which theorem could be false"],
      "missing_structure": [
        {{"type": "DEFINITION_MISSING | LOCAL_STRUCTURE_MISSING | TEMPORAL_STRUCTURE_MISSING | MECHANISM_MISSING", "target": "what is missing", "note": "optional clarification"}}
      ],
      "discovery_requests": ["Optional: specific laws or observables you need from discovery phase"]
    }}
  ]
}}
```

The research_log is YOUR THEORETICAL NOTEBOOK - use it to maintain continuity across iterations.
Record your evolving understanding of the universe's deep structure.

Support roles:
- "confirms": Law directly supports the theorem's claim
- "constrains": Law limits the theorem's scope or adds boundary conditions
- "refutes_alternative": Law rules out an alternative explanation

Missing structure types:
- DEFINITION_MISSING: Need a clearer definition or observable
- LOCAL_STRUCTURE_MISSING: Need spatial/local pattern information
- TEMPORAL_STRUCTURE_MISSING: Need temporal/sequential information
- MECHANISM_MISSING: Need understanding of underlying cause

---

### META-CONSTRAINTS

- Do NOT infer semantic meaning from law names; only infer from claims and test outcomes.
- Do NOT claim eventual or asymptotic behavior unless directly supported by PASS laws.
- Do NOT assume monotonicity unless it is explicitly supported.
- Treat UNKNOWN laws as unresolved; they may support or undermine a theorem, but must be cited cautiously.
- Prefer local explanations when global-count-based explanations repeatedly fail.

### BACKTRACKING TO DISCOVERY

If you cannot form coherent theorems because the laws are:
- Too sparse (not enough PASS laws to synthesize)
- Too contradictory (failures suggest missing variables)
- Missing key observables (e.g., need local structure but only have global counts)

Then use the "discovery_requests" field to specify what additional laws or observables would help.
This feedback will guide the discovery phase to explore productive territory.

Produce {target_count} theorems.

{previous_research_log_section}
---

### LAWS

{laws_section}
"""


def compute_snapshot_hash(law_snapshots: list[LawSnapshot]) -> str:
    """Compute a deterministic hash of law snapshots for reproducibility.

    Args:
        law_snapshots: List of law snapshots

    Returns:
        SHA256 hash (32 hex chars) of the canonicalized snapshot data
    """
    # Canonicalize: sort by law_id, stable JSON encoding
    sorted_snapshots = sorted(law_snapshots, key=lambda ls: ls.law_id)
    canonical_data = json.dumps(
        [ls.to_dict() for ls in sorted_snapshots],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_data.encode()).hexdigest()[:32]


def build_observable_glossary_section(
    scrambler: SymbolScrambler | None = None,
) -> str:
    """Build the observable glossary section for prompt injection.

    Generates a markdown table from CANONICAL_OBSERVABLES registry.
    All physical symbols in expressions are translated to abstract
    symbols to prevent the LLM from inferring physics.

    Args:
        scrambler: Symbol scrambler for translating expressions.
            If None, expressions are included as-is (NOT recommended).

    Returns:
        Formatted markdown section with observable glossary
    """
    try:
        from src.universe.observables import CANONICAL_OBSERVABLES, ConservationStatus
    except ImportError:
        return ""

    lines = [
        "---",
        "",
        "### OBSERVABLE GLOSSARY",
        "",
        "The following observables are available. Use canonical names in theorems.",
        "",
        "| Name | Definition | Status |",
        "|------|------------|--------|",
    ]

    # Sort by conservation status (conserved first), then by name
    def sort_key(item: tuple) -> tuple:
        name, obs = item
        status_order = {
            ConservationStatus.CONSERVED: 0,
            ConservationStatus.CONDITIONAL: 1,
            ConservationStatus.NOT_CONSERVED: 2,
        }
        return (status_order.get(obs.conservation, 3), name)

    sorted_observables = sorted(CANONICAL_OBSERVABLES.items(), key=sort_key)

    for name, obs in sorted_observables:
        # Format conservation status as compact indicator
        status_map = {
            ConservationStatus.CONSERVED: "[C] Conserved",
            ConservationStatus.CONDITIONAL: "[?] Conditional",
            ConservationStatus.NOT_CONSERVED: "[~] Not conserved",
        }
        status_str = status_map.get(obs.conservation, "[?]")

        # Scramble physical symbols in the expression
        expr = obs.expression
        if scrambler:
            expr = scrambler.translate_observable_expr(expr, to_physical=False)

        # Escape pipes in expression
        expr = expr.replace("|", "\\|")

        lines.append(f"| {name} | `{expr}` | {status_str} |")

    lines.append("")
    return "\n".join(lines)


def build_laws_section(law_snapshots: list[LawSnapshot]) -> str:
    """Build the laws section of the prompt."""
    lines = []

    # Group by status
    pass_laws = [ls for ls in law_snapshots if ls.status == "PASS"]
    fail_laws = [ls for ls in law_snapshots if ls.status == "FAIL"]
    unknown_laws = [ls for ls in law_snapshots if ls.status == "UNKNOWN"]

    if pass_laws:
        lines.append("## PASS Laws (verified to hold)")
        lines.append("")
        for ls in pass_laws:
            lines.append(f"- **{ls.law_id}** [{ls.template}]")
            lines.append(f"  Claim: {ls.claim}")
            if ls.power_metrics:
                power_str = ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in ls.power_metrics.items()
                )
                lines.append(f"  Power: {power_str}")
            lines.append("")

    if fail_laws:
        lines.append("## FAIL Laws (falsified with counterexamples)")
        lines.append("")
        for ls in fail_laws:
            lines.append(f"- **{ls.law_id}** [{ls.template}]")
            lines.append(f"  Claim: {ls.claim}")
            if ls.counterexample:
                cx = ls.counterexample
                lines.append(f"  Counterexample: initial='{cx.get('initial_state', '?')}', t_fail={cx.get('t_fail', '?')}")
            lines.append("")

    if unknown_laws:
        lines.append("## UNKNOWN Laws (unresolved)")
        lines.append("")
        for ls in unknown_laws:
            lines.append(f"- **{ls.law_id}** [{ls.template}]")
            lines.append(f"  Claim: {ls.claim}")
            lines.append("")

    return "\n".join(lines)


def build_previous_research_log_section(previous_log: str | None) -> str:
    """Build the previous research log section for continuity.

    Args:
        previous_log: Previous research log content, or None

    Returns:
        Formatted section string
    """
    if not previous_log:
        return ""

    return f"""---

### YOUR PREVIOUS RESEARCH LOG

Below are your theoretical notes from the last iteration. Build on these insights
rather than starting fresh. Your understanding should deepen over time.

{previous_log}

### END OF PREVIOUS LOG"""


def build_prompt(
    law_snapshots: list[LawSnapshot],
    target_count: int = 10,
    include_glossary: bool = True,
    previous_research_log: str | None = None,
    scrambler: SymbolScrambler | None = None,
) -> str:
    """Build the full theorem generation prompt.

    Args:
        law_snapshots: List of law snapshots to include
        target_count: Target number of theorems to generate
        include_glossary: Whether to include the observable glossary section
        previous_research_log: Previous research log for continuity
        scrambler: Symbol scrambler for translating observable expressions

    Returns:
        Complete prompt string
    """
    laws_section = build_laws_section(law_snapshots)
    observable_glossary = (
        build_observable_glossary_section(scrambler=scrambler)
        if include_glossary
        else ""
    )
    previous_research_log_section = build_previous_research_log_section(previous_research_log)
    return THEOREM_GENERATION_PROMPT.format(
        target_count=target_count,
        laws_section=laws_section,
        observable_glossary=observable_glossary,
        previous_research_log_section=previous_research_log_section,
    )


def compute_prompt_hash(prompt: str) -> str:
    """Compute a hash of the prompt for audit tracking."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def format_law_snapshots_for_context(law_snapshots: list[LawSnapshot]) -> str:
    """Format law snapshots as a compact JSON string for storage."""
    return json.dumps(
        [ls.to_dict() for ls in law_snapshots],
        separators=(",", ":"),
    )
