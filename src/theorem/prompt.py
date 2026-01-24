"""Prompt builder for theorem generation."""

import hashlib
import json
from typing import Any

from src.theorem.models import LawSnapshot

# =============================================================================
# PHASE-D: Prompt Template Versioning
# =============================================================================
# Increment this version when making semantic changes to the prompt template.
# This enables reproducibility by tracking which prompt version was used.
PROMPT_TEMPLATE_VERSION = "1.0.0"

THEOREM_GENERATION_PROMPT = """You are a Popperian theory-construction agent operating over a simulated universe.

You are given:
- A set of candidate laws with status in {{PASS, FAIL, UNKNOWN}}
- Each law has been empirically tested against the universe dynamics
- You do NOT have access to the underlying simulator or code
- You must treat all PASS/FAIL outcomes as authoritative

Your task is to propose THEOREMS.

A theorem is:
- An explanatory synthesis of one or more laws
- A claim that goes beyond restating individual laws
- Something that could, in principle, be falsified by new observations

You MUST follow these rules:

1. Use BOTH promoted (PASS) laws and failed (FAIL) laws.
   - PASS laws indicate reliable structure
   - FAIL laws indicate boundaries and missing variables

2. You must NOT invent new observables or assume access to hidden variables.
   - If a theorem seems to require an unobserved quantity, you must state this explicitly.

3. Prefer explanatory structure over completeness.
   - Fewer, deeper theorems are better than many shallow ones.

4. Do NOT restate laws verbatim.
   - Every theorem must combine, constrain, or reinterpret multiple laws.

{observable_glossary}
---

### OUTPUT FORMAT (STRICT JSON)

Output ONLY a JSON array of theorems. Each theorem object must have these fields:

```json
[
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
    ]
  }}
]
```

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

Produce {target_count} theorems.

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


def build_observable_glossary_section() -> str:
    """Build the observable glossary section for prompt injection.

    Generates a markdown table from CANONICAL_OBSERVABLES registry.

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

        # Escape pipes in expression
        expr = obs.expression.replace("|", "\\|")

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


def build_prompt(
    law_snapshots: list[LawSnapshot],
    target_count: int = 10,
    include_glossary: bool = True,
) -> str:
    """Build the full theorem generation prompt.

    Args:
        law_snapshots: List of law snapshots to include
        target_count: Target number of theorems to generate
        include_glossary: Whether to include the observable glossary section

    Returns:
        Complete prompt string
    """
    laws_section = build_laws_section(law_snapshots)
    observable_glossary = build_observable_glossary_section() if include_glossary else ""
    return THEOREM_GENERATION_PROMPT.format(
        target_count=target_count,
        laws_section=laws_section,
        observable_glossary=observable_glossary,
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
