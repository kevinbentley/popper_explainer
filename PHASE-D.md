The fastest way to avoid overfitting is to make the **pipeline more generic and more self-auditing**, not to hand-tune for these particular laws.

If you only make a few code upgrades, make them these. They all pay off regardless of universe details.

---

## 1) Make “theorem generation” a first-class, versioned artifact

**Why:** If you cannot reproduce outputs exactly (same inputs, same prompt, same model), you will chase phantom regressions.

**Code updates**

* Add a `law_snapshot` artifact that is immutable and hash-addressed (content hash).
* Store *everything* required to reproduce theorem output:

  * snapshot hash
  * prompt template version
  * model name + parameters
  * random seed if available
  * raw response
  * parsed response + schema validation results

**Success criteria**

* You can rerun theorem generation from DB state only.
* You can diff two theorem runs and say “only the snapshot changed” or “only the prompt changed”.

---

## 2) Add a canonical “observable glossary” layer (schema, not content)

**Why:** Half your failure clustering is “definition missing.” That’s not physics—it’s interface ambiguity.

**Code updates**

* Create a single registry for observables:

  * `name`, `type`, `definition`, `units`, `dependencies`
* Generate a “glossary block” automatically from the registry.
* Feed this glossary into theorem generation (and optionally into law discovery prompts).

**Success criteria**

* Theorems stop asking “define free movers” and start asking “I need adjacency/run structure.”

This is general infrastructure, not overfitting.

---

## 3) Standardize support roles and enforce them in code

**Why:** Your new roles (`confirms`, `constrains`, `refutes_alternative`) are gold. Make them machine-meaningful.

**Code updates**

* Define an enum:

  * `CONFIRMS`, `CONSTRAINS`, `REFUTES_ALTERNATIVE`, `UNKNOWN_SUPPORT`
* Validate theorem JSON so only these appear.
* Add convenience accessors:

  * `confirmed_laws(theorem)`, `refuted_alternatives(theorem)`, etc.

**Success criteria**

* Clustering and “what to do next” is largely symbolic (roles + IDs), not NLP.

---

## 4) Build a “failure signature” pipeline that is independent of prose

**Why:** You do not want clustering to depend on style variations or wording.

**Code updates**

* Generate `signature_tokens` from:

  * role-coded law IDs (`C:...`, `R:...`)
  * typed missing structure (`DEFINITION_MISSING:...`, etc.)
  * optional: n-grams from failure modes (low weight)

**Success criteria**

* Clusters are stable across runs even if theorem wording changes.

This prevents overfitting because it reduces sensitivity to the exact theorems.

---

## 5) Add typed “missing structure” fields (this is the bridge to looping later)

**Why:** This is your future secondary loop interface. It’s not about these laws; it’s about making the system able to request missing measurements generically.

**Code updates**

* Change theorem schema:

  * `missing_structure: list[{type, target, note}]`
* Enforce `type ∈ {DEFINITION_MISSING, LOCAL_STRUCTURE_MISSING, TEMPORAL_STRUCTURE_MISSING, MECHANISM_MISSING}`
* Store these as structured rows (or JSONB) so you can count and cluster by type.

**Success criteria**

* You can answer: “top missing structure types across runs” without LLM help.

---

## 6) Implement clustering as a reusable, testable service (not a one-off script)

**Why:** You want to be able to run it every time, compare outputs, and unit test behavior.

**Code updates**

* `cluster_failures(run_id) -> clusters.json` where clusters include:

  * bucket label(s)
  * theorem IDs
  * top tokens
  * representative theorem
  * confidence/compactness score
* Use deterministic pass A buckets + simple pass B clustering.
* Add golden tests: given a fixed theorem set, clustering output should be stable.

**Success criteria**

* Clustering is deterministic given fixed inputs.

---

## 7) Add “regression metrics” to prevent silent degradation

**Why:** You will change prompts and code constantly. You need objective guardrails.

**Code updates**
Compute and log per run:

* theorem parse success rate
* avg # supports per theorem
* % theorems with ≥1 refuted alternative
* distribution of missing_structure types
* cluster count per bucket
* cluster stability vs previous run (Adjusted Rand Index if you want, or simple overlap)

**Success criteria**

* You can tell if a prompt change made theorems less falsifiable or less structured.

---

## 8) Add a “no-loop mode” that still prepares for looping

Even if you are not looping yet, you can implement the interfaces now:

**Code updates**

* After theorem generation, run clustering and output:

  * `cluster_summaries`
  * `observable_requests` (not implemented, just proposed)
  * `template_gating_suggestions` (e.g., “avoid eventuality templates”)
* Store those outputs, but do not act on them automatically.

**Success criteria**

* When you decide to loop, you will just “turn on” the executor, not rewrite plumbing.

---

## 9) Most important: tighten the law/theorem snapshot boundary

**Why:** Avoid overfitting by ensuring theorem generation sees a stable, complete view of what the agent “knows,” and nothing else.

**Code updates**

* The snapshot provided to theorem generation should include:

  * laws + statuses
  * observables glossary
  * (optional) a small set of counterexample traces for failed laws (see below)
* Do *not* include extra narrative or “hints”.

**Success criteria**

* You can swap universes and the pipeline still works.

---

## 10) Bonus (high leverage): store counterexample traces for FAIL laws

This one is huge and still generic.

**Why:** Failed laws are informative only if you can show *how* they fail. It also makes theorem-writing more grounded, and it makes clustering cleaner.

**Code updates**

* When a law fails, store:

  * `state_t`, `state_t+1` (or minimal slice)
  * observable values used in the law at t and t+1
  * the violated inequality/equality with numbers
* Expose up to N counterexamples to theorem generation.

**Success criteria**

* Theorems stop saying “complex configurations exist” and start saying “fails on pattern like XX><..”.

This reduces overfitting because the system learns *real boundary cases* instead of verbal handwaving.

---

# If you only do 3 updates next

1. **Versioned snapshots + reproducibility**
2. **Observable glossary registry + injection**
3. **Typed missing structure + role-coded signatures**

Those three set you up for looping later, and they are universe-agnostic.
