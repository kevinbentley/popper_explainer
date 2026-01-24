The next code updates are both (a) high leverage and (b) hard to “overfit” to one universe, I’d focus on turning what you’ve built into **reliable evaluation, stable clustering, and a safe loop interface**.

Here are the most important next updates, in priority order, with very concrete “what to implement” and “done when” checks.

---

## Priority 1: Deterministic clustering + cluster artifacts (reproducible, testable)

Right now you have signature generation and similarity. The next step is to make clustering a **first-class artifact** the same way theorem generation is.

### What to add

* `src/cluster/clusterer.py`

  * `cluster_theorems(run_id, method, params) -> ClusterArtifact`
* `src/cluster/models.py`

  * `ClusterArtifact` with:

    * `artifact_hash`
    * `theorem_run_id`
    * `snapshot_hash`
    * `signature_version`
    * `method` (e.g., `bucket+agglom`, `graph_louvain`)
    * `params`
    * `assignments` (theorem_id → cluster_id)
    * `cluster_summaries` (keywords/tokens + representative theorem)
    * `created_at`
* `src/db/schema.sql`

  * `cluster_artifacts` table and `cluster_runs` table (or `cluster_artifacts` only, keyed by run_id)
* **Determinism**:

  * same inputs ⇒ same clusters (seed everything if needed)

### Why it matters

It prevents endless “my clusters changed” confusion and gives you a stable basis for comparing prompt versions.

### Done when

* You can rerun clustering on a historical theorem run and get the exact same `artifact_hash`.
* You can diff two cluster artifacts and see which theorems moved.

---

## Priority 2: Counterexample trace capture for failed laws (turn failures into usable data)

This is the biggest unlock for later looping and it improves theorems immediately, without tuning.

### What to add

When a law fails in discovery, store a compact “witness”:

* `law_id`
* `t`
* `state_t` (or minimal neighborhood slice if huge)
* `state_t1`
* observable values referenced by the law at `t` and `t+1`
* a human-readable evaluation: `LHS=..., RHS=..., violated by ...`

Schema:

* `law_counterexamples` table (indexed by `law_id`, `snapshot_hash`, `created_at`)
* cap at N examples per law per snapshot (e.g., 20), keep the “most diverse” (see below)

“Diversity” heuristic (simple):

* hash state neighborhoods involved in violation
* keep unique hashes first

### Why it matters

This makes failures concrete, makes theorem generation less hand-wavy, and gives your future loop something objective to target.

### Done when

* Every FAIL law has at least 1 stored witness, unless it failed for “timeout/unknown”.
* Theorem prompt can include 1–3 counterexamples for the top constraining/refuted laws.

---

## Priority 3: Regression dashboard metrics (stop silent degradation)

You’re now versioning theorems and prompts. Next you need **objective health metrics**.

### What to add

`src/metrics/run_metrics.py` that computes per snapshot:

For discovery:

* # laws proposed
* pass/fail/unknown rates by template type
* median tests per law
* top 20 “frequently failing” law templates

For theorems:

* parse success rate
* distribution of support roles per theorem
* distribution of `MissingStructureType`
* avg signature length / token counts
* % theorems with at least one `REFUTES_ALTERNATIVE`

For clustering:

* # clusters
* average cluster size
* fraction of singleton clusters
* stability vs prior run (simple overlap score is fine)

Store:

* `run_metrics` table keyed by `snapshot_hash` and artifact hashes.

### Done when

* You can answer: “did prompt 1.0.1 make theorems worse?” without reading text.

---

## Priority 4: Make the “secondary loop” an interface, not automation

You want to avoid overfitting, so do *not* auto-act on clusters yet. But you should create the code boundary.

### What to add

* `src/loop/proposals.py`

  * `generate_next_step_proposals(cluster_artifact) -> ProposalArtifact`
* Proposal types:

  * `PROMPT_CHANGE` (e.g., forbid eventuality claims unless supported)
  * `TEMPLATE_GATING` (deprioritize monotonicity templates)
  * `OBSERVABLE_REQUEST` (typed: LOCAL/TEMPORAL/etc.)
  * `TEST_BUDGET_CHANGE` (increase tests for specific law families)
* Store proposals as an artifact, not executed automatically.

### Done when

* After one run you can produce a “next steps” JSON, but nothing changes unless you explicitly apply it.

---

## Priority 5: Hardening: schema validation and “strict mode”

You already parse old/new formats. Next: make it impossible for output drift to creep in silently.

### What to add

* JSON Schema for theorem output (and cluster output)
* “strict parse mode” option:

  * if the LLM output violates schema, store raw response but mark run as invalid
  * optionally auto-retry with a repair prompt (but store both artifacts)

### Done when

* Your pipeline never writes partially-parsed garbage into theorems table.

---

## Priority 6: Signature versioning (future-proof your clustering)

You’ve introduced role-coded signatures. You’ll evolve them. Version them now.

### What to add

* `ROLE_CODED_SIGNATURE_VERSION = "1.0.0"` in `src/theorem/signature.py`
* Store it with each theorem row and in `ClusterArtifact`
* If version mismatches, clustering refuses or re-computes

### Done when

* You can compare clusters across time knowing signatures were computed the same way.

---

# The most important next steps

1. **Counterexample witness capture** (turn FAIL into data)
2. **Deterministic clustering artifacts + metrics** (turn runs into science you can measure)

