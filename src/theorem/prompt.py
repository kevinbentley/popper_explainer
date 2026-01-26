"""Prompt builder for theorem generation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, TYPE_CHECKING

from src.theorem.models import LawSnapshot

if TYPE_CHECKING:
    from src.db.models import TheoremRecord
    from src.proposer.scrambler import SymbolScrambler

# =============================================================================
# PHASE-D: Prompt Template Versioning
# =============================================================================
# Increment this version when making semantic changes to the prompt template.
# This enables reproducibility by tracking which prompt version was used.
PROMPT_TEMPLATE_VERSION = "1.4.0"

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
- You should treat FAIL outcomes as strong evidence, and PASS outcomes as provisional support (see EPISTEMIC CAUTION below)

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

=== EPISTEMIC CAUTION: BE YOUR OWN DEVIL'S ADVOCATE ===

A core Popperian principle: corroboration is NOT confirmation. Apply these rules rigorously:

1. A PASS law is not a proven law.
   - PASS means the harness failed to falsify it within a finite test budget.
   - The law may hold only in tested regions of the state space, and break elsewhere.
   - Do NOT treat a cluster of PASS results as certainty. Ask: "What test HASN'T been run that could break this?"

2. A FAIL result is stronger evidence — but not infallible.
   - FAIL means a counterexample was found, which is concrete and reproducible.
   - However, measurement or encoding errors can produce spurious failures.
   - When a FAIL contradicts multiple PASS laws, consider whether the failure might stem from a boundary condition, an edge case in the test harness, or a misspecified observable — not necessarily a wrong theorem.

3. Actively challenge your own theorems.
   - For each theorem you propose or carry forward, ask: "What would make me doubt this?"
   - If new laws (PASS or FAIL) have appeared since the theorem was last affirmed, re-evaluate it honestly. Do NOT preserve a theorem out of inertia.
   - If two theorems conflict, do not quietly drop one. Explicitly state the contradiction and reason about which has stronger support, or whether both need revision.

4. When contradictions arise, diagnose before discarding.
   - A contradiction between laws may indicate a missing variable, a scope error, or a flawed observable definition — not necessarily that one law is wrong.
   - Before retracting a theorem, consider: Is the problem in the theorem itself, in the laws that support it, or in how the observables are defined?
   - Use the "failure_modes" and "missing_structure" fields to record your diagnostic reasoning.

{observable_glossary}
---

### OUTPUT FORMAT (STRICT JSON)

Output a JSON object with your cumulative summary and theorems:

```json
{{
  "research_log": "Your CUMULATIVE SUMMARY (see structure below). This is a living document that you UPDATE each iteration — not a fresh notebook.",
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

=== CUMULATIVE SUMMARY (research_log) ===

The research_log is NOT a diary of today's observations. It is a CUMULATIVE UNIFIED PICTURE
of everything you know, updated each iteration. Think of yourself as a Principal Investigator
maintaining a running state of the field — not a lab technician recording daily notes.

Your research_log MUST contain exactly these four sections (use the headers as shown, max 500 words total).
Every section MUST be present in every iteration, even if empty (write "None yet." if empty).

**THE STANDARD MODEL (Fixed Laws):**
Invariants, symmetries, and conservation laws that have survived extensive testing
(high statistical power, ideally >1000 test cases — use the Power metrics on each law
to judge). These are the bedrock of your understanding.
Once a result appears here, STOP re-verifying it. Build on it. Only remove an entry
if new evidence actively contradicts it — and if you do remove it, move it to
THE FALSIFICATION GRAVEYARD with a note explaining why.
Format: "- [name]: [statement] (supported by: law_xxx, law_yyy)"

**THE REACTION MANUAL (Local Rules):**
Confirmed local transition rules — what happens when specific spatial configurations
are encountered. This is the operational handbook of the universe's dynamics.
Each entry should describe a local pattern and its deterministic outcome.
Format: "- [input pattern] -> [output pattern] (supported by: law_xxx)"

**THE FALSIFICATION GRAVEYARD:**
Theories and hypotheses that have been CONCLUSIVELY DISPROVEN. This section exists
to prevent you from re-proposing or re-testing dead ideas. Every entry records:
what was believed, what evidence killed it, and why it cannot be revived.
DO NOT remove entries from this section. It is append-only.
If a previously dead theory is resurrected by new evidence, move it to THE FRONTIER
with an explicit note explaining what changed.
Format: "- [dead theory]: DISPROVEN by [law_xxx/counterexample]. Reason: [why it failed]"

**THE FRONTIER (Active Anomalies):**
Current contradictions, open questions, and hypotheses under active investigation.
For each entry, note: (1) what the anomaly or question is, (2) what evidence exists
for and against, and (3) your current diagnostic hypothesis about root cause
(missing variable? scope error? encoding issue? boundary condition?).
These are where your theorem slots and discovery_requests should focus — not on
re-proving items already in THE STANDARD MODEL.
Format: "- [anomaly/question]: [evidence summary] | Diagnosis: [current hypothesis]"

IMPORTANT RULES FOR UPDATING:
- If a previous cumulative summary is provided, UPDATE it — do not rewrite from scratch.
- Promote results: FRONTIER -> STANDARD MODEL or REACTION MANUAL when evidence is strong.
- Demote results: STANDARD MODEL -> FRONTIER if new evidence raises doubt.
- Kill results: FRONTIER -> GRAVEYARD when decisively falsified.
- Resurrect (rare): GRAVEYARD -> FRONTIER only if genuinely new evidence warrants it.
- The summary should show clear iteration-over-iteration progress. If the same items
  appear in THE FRONTIER for multiple iterations with no new evidence, note the stall
  and request specific laws or observables that would break the deadlock.

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

{existing_theorems_section}
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
        return """---

### YOUR PREVIOUS CUMULATIVE SUMMARY

No previous summary exists. This is your first iteration. Create the initial
cumulative summary with all four sections:
- THE STANDARD MODEL: Populate from the strongest PASS laws (high power metrics)
- THE REACTION MANUAL: Any confirmed local transitions
- THE FALSIFICATION GRAVEYARD: Any FAIL laws that kill obvious hypotheses
- THE FRONTIER: Open questions and contradictions to investigate next"""

    return f"""---

### YOUR PREVIOUS CUMULATIVE SUMMARY

Below is your cumulative summary from the last iteration. UPDATE this document
rather than rewriting from scratch:
- Promote well-supported hypotheses from THE FRONTIER to THE STANDARD MODEL or THE REACTION MANUAL
- Add newly discovered transitions to THE REACTION MANUAL
- Move decisively falsified theories from THE FRONTIER to THE FALSIFICATION GRAVEYARD
- Add new contradictions or surprises to THE FRONTIER
- Do NOT re-verify items already in THE STANDARD MODEL unless new evidence contradicts them
- NEVER remove entries from THE FALSIFICATION GRAVEYARD — it is append-only

{previous_log}

### END OF PREVIOUS SUMMARY"""


def build_existing_theorems_section(
    theorems: list[TheoremRecord],
    scrambler: SymbolScrambler | None = None,
) -> str:
    """Build a section presenting existing theorems for continuity.

    Formats previously generated theorems so the LLM can confirm,
    refine, supersede, or retract them rather than starting from scratch.

    Args:
        theorems: List of TheoremRecord objects from the database
        scrambler: Symbol scrambler for translating physical symbols

    Returns:
        Formatted markdown section, or empty string if no theorems
    """
    if not theorems:
        return ""

    lines = [
        "---",
        "",
        "### EXISTING THEOREMS",
        "",
        "These theorems were generated in previous iterations. You should:",
        "- CONFIRM theorems that remain well-supported by the laws",
        "- REFINE theorems where new laws suggest modifications",
        "- SUPERSEDE theorems with better formulations",
        "- RETRACT theorems contradicted by new evidence",
        "- BUILD ON existing insights rather than starting from scratch",
        "",
        "Include your existing theorems (updated as needed) in your output along with any new ones.",
        "",
    ]

    for theorem in theorems:
        # Scramble claim text if scrambler available
        claim = theorem.claim
        if scrambler:
            claim = scrambler.translate_observable_expr(claim, to_physical=False)

        lines.append(f"- **{theorem.theorem_id}** (Status: {theorem.status})")
        lines.append(f"  Name: {theorem.name}")
        lines.append(f"  Claim: {claim}")

        # Parse and format support references
        if theorem.support_json:
            try:
                support = json.loads(theorem.support_json)
                if support:
                    support_parts = []
                    for s in support:
                        law_id = s.get("law_id", "?")
                        role = s.get("role", "?")
                        support_parts.append(f"{law_id} ({role})")
                    lines.append(f"  Supported by: {', '.join(support_parts)}")
            except json.JSONDecodeError:
                pass

        # Parse and format failure modes
        if theorem.failure_modes_json:
            try:
                failure_modes = json.loads(theorem.failure_modes_json)
                if failure_modes:
                    modes_str = "; ".join(str(fm) for fm in failure_modes)
                    lines.append(f"  Failure modes: {modes_str}")
            except json.JSONDecodeError:
                pass

        lines.append("")

    return "\n".join(lines)


def build_prompt(
    law_snapshots: list[LawSnapshot],
    target_count: int = 10,
    include_glossary: bool = True,
    previous_research_log: str | None = None,
    scrambler: SymbolScrambler | None = None,
    existing_theorems: list[TheoremRecord] | None = None,
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
    existing_theorems_section = build_existing_theorems_section(
        existing_theorems or [], scrambler=scrambler,
    )
    return THEOREM_GENERATION_PROMPT.format(
        target_count=target_count,
        laws_section=laws_section,
        observable_glossary=observable_glossary,
        previous_research_log_section=previous_research_log_section,
        existing_theorems_section=existing_theorems_section,
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
