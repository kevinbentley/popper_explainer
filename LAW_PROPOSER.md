Proposed law requirements:

Law schema requirement: quantifiers, horizon, observables, preconditions

A law is rejected at the front door unless it specifies:
Template type (one of a small finite set)
Quantifiers (forall, exists, and over what index set)
Horizon (T, or H for eventuality)
Preconditions (as predicates on the initial state)
Observables (what must be measured)
Falsifier format (what counts as a counterexample)

If it cannot fit, it is not a law yet; it is a vague hypothesis.

Minimal template set (covers most of the KG universe)

invariant: ∀t≤T: f(t) = f(0)
implication_step: ∀t<T: P(t) → Q(t+1)
implication_state: ∀t≤T: P(t) → Q(t)
monotone: ∀t<T: f(t+1) ≤ f(t) or ≥
eventually: ∀t0≤T: P(t0) → ∃t≤t0+H: Q(t)
symmetry_commutation: evolve(Transform(S), T) = Transform(evolve(S,T))
bound: ∀t≤T: f(t) ≤ k (or ≥)

Every law must include an explicit “forbidden pattern” description: a compact statement of what would refute it.

Example:

invariant law: “forbidden = any t where f(t) != f(0)”
symmetry law: “forbidden = any state S where commutation fails at some t≤T”
implication: “forbidden = any t where P(t) true and Q(t+1) false”

A huge source of garbage laws comes from the model inferring universality from a small number of traces.

So enforce:

Observation objects: concrete trajectories, metrics, event logs
Law objects: quantified claims that generalize beyond observations
Support links: a law must cite which observations motivate it, but it may not quote them as proof

This is also where you exploit long context properly: not by dumping everything, but by giving structured, queryable memory.

Before the LLM is allowed to output a law, it must answer (in structured form):

Is this law logically implied by the current accepted set?
    If yes → do not propose (or mark as “redundant”)

Does it distinguish between at least two competing mechanism hypotheses?
    If no → low value
What new experiment family would most likely falsify it?
    If none → reject as non-Popperian

This makes the discovery step purposeful: each new law should force a new test.

Treat “untestable” as a first-class output with a reason code

Right now “untestable” is a bucket. Make it actionable.
When the tester returns untestable, it must provide:
 - reason_code: one of:
 - missing_observable
 - missing_generator_knob
 - missing_transform
 - ambiguous_claim
 - requires_preimage_search
 - resource_limit (needs bigger N/T)

minimal_capability_needed: e.g. “add transform mirror+swap” or “add metric: collision_events”
rewrite_suggestion: how to restate the law into a supported template

Then the discovery LLM sees this and can:

- propose a rewritten law (now testable), or
- propose an instrumentation upgrade request (if you allow that)

This turns “untestable” into a roadmap.

A tighter “law discovery loop” (recommended)
Loop state

Maintain three evolving sets:

L_accepted (laws that survived strong tests)

L_rejected (falsified with counterexamples)

L_unknown (untestable or inconclusive with reason codes)

And a fourth:

CapabilitySet (observables + transforms + generators the tester supports)

Iteration steps

Propose: discovery LLM proposes K candidate laws (compileable templates only)

Prioritize: rank by expected falsifiability / novelty / discrimination

Test: harness tests top M laws with adversarial case generation

Update: store counterexamples and “why failed”

Reflect: discovery LLM gets only summaries plus key counterexample traces, not the entire history

Then occasionally:
6) Mechanism synthesis: separate step tries to compress accepted laws into small mechanism axioms (local rules), and predicts new laws to test.

This avoids your previous “generate axioms → then theorems → then predictions” being too detached from falsification.

Instead: falsification continuously drives mechanism learning.

Bigger context helps only if it is structured and salient.

Do NOT

Dump every experiment trace raw

Provide 1000 laws as plain text

Give the model a wall of logs and hope it “figures it out”

DO

Provide:

A compact universe spec (symbols, step API, what’s observable)

A rolling “frontier summary”:

top accepted laws (maybe 20–40), each in strict schema

top falsified laws + minimal counterexample

top unknown laws + reason_code

A small “counterexample gallery”:

10–30 canonical traces that killed big classes of claims

A “redundancy map”:

derived/duplicate law patterns you do not want again

This gives the model the benefit of long context (it learns what not to repeat, and what failure looks like), without drowning it.

Bonus: retrieval inside your own system

Instead of putting everything into the prompt, store all laws/traces in a local DB and provide the LLM:

the short summary

plus on-demand retrieval (give it an API tool call in your agent framework)

That’s “long context,” but controllable.

```
{
  "law_id": "right_component_conservation",
  "template": "invariant",
  "scope": {
    "quantifier": "forall",
    "time_horizon": 100
  },
  "preconditions": [
    {"pred": "grid_length", "op": ">=", "value": 5}
  ],
  "observables": [
    {"name": "R_total", "definition": "count('>') + count('X')"}
  ],
  "claim": "R_total(t) == R_total(0) for all t in [0..T]",
  "forbidden": "exists t in [0..T] where R_total(t) != R_total(0)",
  "discriminates": [
    {"against": "particle_count_conservation_only", "why": "component vs particle bookkeeping"}
  ],
  "proposed_tests": [
    {
      "family": "random_density_sweep",
      "params": {"densities": [0.1,0.3,0.6], "cases": 200}
    },
    {
      "family": "adversarial_mutation",
      "params": {"budget": 2000, "mutation_ops": ["swap_cells","flip_symbol","shift"]}
    }
  ]
}
```

extra constraints that will improve quality immediately
Constraint A: ban “axioms” during law discovery

During discovery, the model may only output “claims.” No axioms, no theories.

Later, mechanism synthesis tries to explain accepted claims via small axioms.

Constraint B: require each new law to come with a likely counterexample strategy

Even if the law is true, the model must say how it would try to break it.
This is Popper pressure in prompt form.

