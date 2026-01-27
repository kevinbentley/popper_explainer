"""LLM prompt templates for the Reflection Engine.

Two separate calls:
1. Auditor: conflict detection, tautology/redundancy pruning (low temperature)
2. Theorist: derived observables, hidden variables, causal narrative (higher temperature)

SCIENTIFIC INTEGRITY NOTE:
These prompts must NOT leak physics knowledge to the LLM. The LLM must
synthesize theory from empirical evidence alone, not from hints embedded
in the prompt about which quantities are conserved, what symbols represent,
or how the universe works.

If you modify this file, ensure you do NOT:
- Label observables as "conserved" or "not conserved"
- Explain what symbols represent (e.g., "A is a mover", "K is a collision")
- Provide verified laws, physics explanations, or known conservation quantities
- Give examples that reference specific symbol expressions (count(A), count(B), etc.)
- Use physics-laden terms (particle, mover, collision, momentum, boundary) in examples

SYMBOL SCRAMBLING:
All inputs to the LLM use abstract symbols (W, A, B, K). The prompts
must not contain physical symbol references.
"""

AUDITOR_SYSTEM_INSTRUCTION = """You are a rigorous scientific AUDITOR reviewing a body of accepted empirical laws.

Your task is to find logical problems in the current set of accepted laws:

1. CONFLICTS: Does any accepted (PASS) law contradict evidence from the Falsification Graveyard?
   - A law claims "X is always true" but a graveyard entry shows X was false in a specific case
   - Two accepted laws make contradictory claims about the same observable

2. TAUTOLOGIES: Are any accepted laws vacuously true or unfalsifiable in practice?
   - A law with preconditions so restrictive that no test case ever satisfies them
   - A law that asserts something trivially true by definition

3. REDUNDANCY: Are any accepted laws logically subsumed by other accepted laws?
   - Law A is a special case of Law B (B is strictly more general)
   - Two laws say the same thing in different notation

4. DEDUCTIVE ISSUES: Are there logical gaps or inconsistencies in the overall picture?

Be conservative: only flag clear issues, not speculative ones.
Do NOT propose new laws or hypotheses — that is not your role.

Respond with a JSON object matching this schema:
{
  "conflicts": [
    {
      "law_id": "string — the accepted law with a problem",
      "conflicting_law_id": "string | null — another law it conflicts with",
      "counterexample_law_id": "string | null — graveyard law whose counterexample contradicts this",
      "description": "string — clear explanation of the conflict",
      "severity": "low | medium | high"
    }
  ],
  "archives": [
    {
      "law_id": "string — the law to archive/demote",
      "reason": "tautology | redundant | subsumed | conflict",
      "subsumed_by": "string | null — law_id that subsumes this one"
    }
  ],
  "deductive_issues": ["string — description of logical gaps"],
  "summary": "string — 1-2 sentence summary of audit findings"
}
"""

THEORIST_SYSTEM_INSTRUCTION = """You are a creative scientific THEORIST synthesizing a coherent picture from empirical evidence.

You have access to:
- The current set of accepted (PASS) laws (post-audit, with any flagged issues noted)
- The complete Falsification Graveyard (all FAIL laws with counterexamples)
- Active anomalies (UNKNOWN laws with reason codes)
- All research log entries from previous iterations
- The current Standard Model (if one exists from a prior reflection)

Your tasks:

1. DERIVED OBSERVABLES: Identify combinations of existing observables that might reveal
   deeper structure. Look at accepted laws and ask: are there arithmetic combinations
   (sums, differences, products, ratios) of known observables that might also be invariant
   or informative? Let the data guide you — do not guess from symbol names.

2. HIDDEN VARIABLES: Postulate variables that are NOT directly observable but whose existence
   would explain anomalies or unify multiple laws.
   - Look at patterns of failure in the graveyard: do falsified laws share common features
     in their counterexamples? What unobserved quantity, if it existed, would explain why
     some laws hold only under certain conditions?
   - Each hidden variable MUST include a testable prediction — otherwise it is not scientific.

3. CAUSAL NARRATIVE: Write a coherent 2-5 paragraph narrative explaining how the universe works
   based on current evidence. This should connect the dots between accepted laws.
   Base your narrative ONLY on what the evidence shows. Do not speculate about what symbols
   "represent" or assign physical interpretations beyond what the data supports.

4. KNOWLEDGE DECOMPOSITION: Categorize what we know:
   - Firmly established (high-power PASS, survived many tests)
   - Conditionally established (PASS but limited test conditions)
   - Active anomalies (things we can't yet explain)
   - Open questions (things we haven't tested)

5. SEVERE TEST SUGGESTIONS: Propose specific experimental conditions that would maximally
   discriminate between competing hypotheses or stress-test accepted laws.

Respond with a JSON object matching this schema:
{
  "derived_observables": [
    {
      "name": "string — descriptive name (use canonical observable names where possible)",
      "expression": "string — mathematical expression using existing observables",
      "rationale": "string — why this might be informative",
      "source_laws": ["string — law_ids that motivated this"]
    }
  ],
  "hidden_variables": [
    {
      "name": "string — descriptive name",
      "description": "string — what this variable represents",
      "evidence": "string — what anomalies suggest its existence",
      "testable_prediction": "string — how we could detect it indirectly"
    }
  ],
  "causal_narrative": "string — 2-5 paragraph narrative",
  "k_decomposition": "string — structured knowledge decomposition",
  "confidence": 0.0 to 1.0,
  "severe_test_suggestions": [
    {
      "command_type": "initial_condition | topology_test | parity_challenge",
      "description": "string — what to test and why",
      "target_law_id": "string | null — specific law to stress-test",
      "initial_conditions": ["string — specific state strings to test"],
      "grid_lengths": [4, 8, 16, 32, 64]
    }
  ]
}
"""


def build_auditor_prompt(
    fixed_laws: list[dict],
    graveyard: list[dict],
    anomalies: list[dict],
    research_log_entries: list[str],
) -> str:
    """Build the prompt for the auditor LLM call.

    Args:
        fixed_laws: Accepted (PASS) laws with their details
        graveyard: Falsified (FAIL) laws with counterexamples
        anomalies: UNKNOWN laws with reason codes
        research_log_entries: Previous research log entries

    Returns:
        Complete prompt string
    """
    sections = []

    # Fixed laws section
    sections.append("=== ACCEPTED LAWS (PASS) ===")
    sections.append(f"Total: {len(fixed_laws)} laws")
    sections.append("")
    for law in fixed_laws:
        law_id = law.get("law_id", "?")
        template = law.get("template", "?")
        claim = law.get("claim", "?")
        sections.append(f"- [{template}] {law_id}: {claim}")
        if law.get("observables"):
            obs_strs = [
                f"{o.get('name', '?')}={o.get('expr', '?')}"
                for o in law["observables"] if o
            ]
            sections.append(f"    observables: {', '.join(obs_strs)}")
    sections.append("")

    # Graveyard section
    sections.append("=== FALSIFICATION GRAVEYARD (FAIL) ===")
    sections.append(f"Total: {len(graveyard)} falsified laws")
    sections.append("")
    for law in graveyard:
        law_id = law.get("law_id", "?")
        template = law.get("template", "?")
        claim = law.get("claim", "?")
        sections.append(f"- [{template}] {law_id}: {claim}")
        cx = law.get("counterexample", {})
        if cx:
            sections.append(f"    counterexample: state={cx.get('initial_state', '?')}, t_fail={cx.get('t_fail', '?')}")
            if cx.get("trajectory_excerpt"):
                excerpt = cx["trajectory_excerpt"]
                if isinstance(excerpt, str):
                    sections.append(f"    trajectory: {excerpt[:200]}")
                elif isinstance(excerpt, list):
                    sections.append(f"    trajectory: {' -> '.join(str(s) for s in excerpt[:5])}")
    sections.append("")

    # Anomalies section
    if anomalies:
        sections.append("=== ACTIVE ANOMALIES (UNKNOWN) ===")
        sections.append(f"Total: {len(anomalies)} laws")
        sections.append("")
        for law in anomalies:
            law_id = law.get("law_id", "?")
            reason = law.get("reason_code", "?")
            sections.append(f"- {law_id}: {law.get('claim', '?')} (reason: {reason})")
        sections.append("")

    # Research log
    if research_log_entries:
        sections.append("=== RESEARCH LOG ENTRIES ===")
        for entry in research_log_entries:
            sections.append(entry)
        sections.append("")

    sections.append("=== TASK ===")
    sections.append("Audit the accepted laws above. Identify conflicts with the graveyard,")
    sections.append("tautologies, redundancies, and deductive issues.")
    sections.append("Respond with a JSON object as specified in your instructions.")

    return "\n".join(sections)


def build_theorist_prompt(
    fixed_laws: list[dict],
    graveyard: list[dict],
    anomalies: list[dict],
    research_log_entries: list[str],
    current_standard_model: dict | None = None,
) -> str:
    """Build the prompt for the theorist LLM call.

    Args:
        fixed_laws: Accepted (PASS) laws post-audit
        graveyard: Falsified (FAIL) laws with counterexamples
        anomalies: UNKNOWN laws with reason codes
        research_log_entries: Previous research log entries
        current_standard_model: Previous standard model summary (if exists)

    Returns:
        Complete prompt string
    """
    sections = []

    # Current standard model (if exists)
    if current_standard_model:
        sections.append("=== CURRENT STANDARD MODEL (version {}) ===".format(
            current_standard_model.get("version", "?")
        ))
        if current_standard_model.get("causal_narrative_excerpt"):
            sections.append(current_standard_model["causal_narrative_excerpt"])
        if current_standard_model.get("derived_observables"):
            sections.append("\nDerived observables:")
            for d in current_standard_model["derived_observables"]:
                sections.append(f"  - {d.get('name', '?')}: {d.get('expression', '?')}")
        if current_standard_model.get("hidden_variables"):
            sections.append("\nPostulated hidden variables:")
            for h in current_standard_model["hidden_variables"]:
                sections.append(f"  - {h.get('name', '?')}: {h.get('testable_prediction', '?')}")
        sections.append("")

    # Fixed laws section
    sections.append("=== ACCEPTED LAWS (PASS, post-audit) ===")
    sections.append(f"Total: {len(fixed_laws)} laws")
    sections.append("")
    for law in fixed_laws:
        law_id = law.get("law_id", "?")
        template = law.get("template", "?")
        claim = law.get("claim", "?")
        sections.append(f"- [{template}] {law_id}: {claim}")
        if law.get("observables"):
            obs_strs = [
                f"{o.get('name', '?')}={o.get('expr', '?')}"
                for o in law["observables"] if o
            ]
            sections.append(f"    observables: {', '.join(obs_strs)}")
    sections.append("")

    # Graveyard section
    sections.append("=== COMPLETE FALSIFICATION GRAVEYARD ===")
    sections.append(f"Total: {len(graveyard)} falsified laws")
    sections.append("")
    for law in graveyard:
        law_id = law.get("law_id", "?")
        template = law.get("template", "?")
        claim = law.get("claim", "?")
        sections.append(f"- [{template}] {law_id}: {claim}")
        cx = law.get("counterexample", {})
        if cx:
            sections.append(f"    counterexample: state={cx.get('initial_state', '?')}, t_fail={cx.get('t_fail', '?')}")
    sections.append("")

    # Anomalies section
    if anomalies:
        sections.append("=== ACTIVE ANOMALIES (UNKNOWN) ===")
        for law in anomalies:
            law_id = law.get("law_id", "?")
            reason = law.get("reason_code", "?")
            claim = law.get("claim", "?")
            sections.append(f"- {law_id}: {claim} (reason: {reason})")
        sections.append("")

    # Research log
    if research_log_entries:
        sections.append("=== RESEARCH LOG ENTRIES ===")
        for entry in research_log_entries:
            sections.append(entry)
        sections.append("")

    sections.append("=== TASK ===")
    sections.append("Synthesize the evidence above into a coherent theoretical picture.")
    sections.append("Identify derived observables, postulate hidden variables,")
    sections.append("write a causal narrative, and suggest severe tests.")
    sections.append("Respond with a JSON object as specified in your instructions.")

    return "\n".join(sections)
