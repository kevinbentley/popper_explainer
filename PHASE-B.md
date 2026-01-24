Here's a concrete build plan for the failure clustering + secondary loop, assuming:

laws are already in a DB (with PASS/FAIL/UNKNOWN)

discovery loop already runs and writes new law test results

you can call an LLM to generate theorems

I'll write this as an implementation-oriented checklist with data structures and interfaces.

## 0) Add two tables (or collections)

`theorem_runs`

Tracks one theorem-generation batch.

Fields:

* `run_id` (fk)
* `index_in_run` (1..N)
* `name`
* `status` (Established/Conditional/Conjectural)
* `claim_text`
* `support_law_ids` (array)
* `failure_modes` (array of strings)
* `missing_structure` (array of strings)
* `full_json` (if you store the parsed structure verbatim)
* `failure_signature_text` (denormalized string used for clustering)

---

## 1) Define your theorem output schema (strict)

Pick JSON. Make the LLM output **only JSON**.

Example (per theorem):

* `name`
* `status`
* `claim`
* `support`: list of `{law_id, status, role}` (role: supports/constrains)
* `failure_modes`: list[str]
* `missing_structure`: list[str]

Make your parser reject anything not valid JSON.

---

## 2) Theorem generation pipeline

### Step 2.1: Build law

Function: `get_law_snapshot(experiment_id) -> LawSnapshot`

* Pull all laws you want to expose (probably promoted + a curated set of failed/unknown)
* Include:

  * `law_id`
    * `status`
      * `claim` (or template + rendered claim)
        * optional: a short natural-language gloss you generate

Store snapshot content hash as `law_snapshot_id`.

### Step 2.2: Call LLM

Function: `generate_theorems(law_snapshot, prompt_version) -> raw_text`

* Persist in `theorem_runs.raw_response`
* Keep the prompt versioned (you will iterate)

### Step 2.3: Parse + validate

Function: `parse_theorems(raw_text) -> list[Theorem]`

Validation rules:

* support_law_ids must exist in snapshot
* failure_modes and missing_structure must be lists (can be empty, but never missing)
* status in {Established, Conditional, Conjectural}

Persist to `theorems`.

---

## 3) Build the failure signature for clustering

Function: `build_failure_signature(theorem) -> str`

Concatenate:

* all `failure_modes` lines
* all `missing_structure` lines
* (optional) ids of cited FAIL laws (as tokens)

Normalize lightly:

* lowercase
* strip punctuation
* collapse whitespace

Store in `theorems.failure_signature_text`.

This is the thing you vectorize and cluster.

---

## 4) Failure clustering (two-pass)

### Pass A: deterministic bucket labels

Function: `assign_buckets(signature_text) -> set[bucket]`

Implement a small keyword dictionary (config file) like:

* LOCAL_PATTERN: adjacency, neighbor, gap, spacing, arrangement, configuration, alternat, pair, bracket
* TEMPORAL_EVENTUAL: eventually, converge, cease, resolve, attractor, steady, asymptotic
* MONOTONICITY: monotone, nondecreasing, never increase, always decrease
* COUNT_ONLY_TRAP: global count, counts insufficient, only depends on, marginals
* FEATURE_MISMATCH: definition mismatch, observable mismatch, naming, schema, ambiguous
* SYMMETRY_MISAPPLIED: time reversal, mirror only, swap only, reversal

Store bucket tags in memory or a `theorem_bucket_map` table if you want.

### Pass B: semantic clustering within a bucket

Pick one approach:

**Simple start:** TF-IDF + agglomerative

* Vectorize signatures within each bucket
* Agglomerative clustering with cosine distance
* Threshold-based stopping (distance threshold)

**Better:** embeddings + HDBSCAN

* Embed signatures
* HDBSCAN cluster (auto cluster count, isolates noise)

Outputs:

* `cluster_id`
* list of theorem_ids
* cluster label (auto-generated summary terms)

Store in a table:

### `failure_clusters`

* `cluster_id`
* `run_id` (or cluster_run_)
* `bucket`
* `theorem_ids` (array) or link table
* `summary_keywords`
* `representative_theorem_id`

---

## 5) Turn clusters observable

This is the secondary  bridge.

### Step 5.1: Summarize each cluster

Function: `summarize_cluster(cluster) -> ClusterSummary`

Generate:

* top terms (TF-IDF top n-grams)
* representative failure_signature
* list of recurring phrases

### Step 5.2: Propose observables (LLM or rules)

Function: `propose_observables(cluster_summary) -> list[ObservableProposal]`

You can do this two ways:

* **Rule-based mapping** (fast, reliable)
* **LLM-based** constrained by your existing observable schema

d do both: rules first, LLM second if rules yield nothing.

ObservableProposal fields:

* `name` (schema-valid identifier)
* `type` (int, float, vector, histogram)
* `definition` (how to compute from state)
* `motivation` (ties back to cluster)
* `suggested_laws` (
Persist proposals to:

### `observable_proposals`

* `proposal_id`
* `cluster_id`
* `name`
* `definition`
* `priority_score`
* `status` (proposed/implemented/rejected)

---

## 6) Implement selected observables and re-run discovery

### Step 6.1: Observable registry

You likely already have this for discovery. Add:

* versioned observable definitions
* unit tests (important)

### Step 6.2: Enable new observable subset

For the next discovery run, enable:

* existing observables
* * the new ones from proposals

### Step 6.3: Run discovery as usual

Now you should see:

* previously-count- laws remain false (good)
* new *pattern-aware* laws start passing (progress)

---

## 7) Stopping and governance (so this t explode)

Add simple controls:

### Feature explosion guardrails

* max new observables per cycle: * require each new observable to ### Cluster closure condition

Mark a  if:

* 2 consecutive cycles propose no new observables, or
* proposed observables t improve pass-rate in that bucket

Persist this as cluster status.

---

## 8) Metrics to log (ll want these)

Per cycle:

* # theorems generated
* # clusters found per bucket
* # new observable proposals
* # accepted/implemented
* PASS/FAIL delta in laws touching the bucket (before vs : count of new PASS laws that mention new observables

These metrics tell you whether the secondary loop is actually driving explanatory growth.

---

## 9) Recommended first milestone sequence

1. Make theorem JSON output + parsing rock-solid
2. Implement failure_signature and bucket tagging
3. Implement TF-IDF + agglomerative clustering
4. Implement rule-based observable proposals for LOCAL_PATTERN bucket
5. Add 1noveltyafter)
* youdonclosedcluster unlock 1doesnonlyfailing 26. Re-run discovery and confirm new PASS laws appear

Only after that should you bother with embeddings/HDBSCAN.

---
# Theorem Generator Prompt
This is an idea of the prompt we will use for the AI to generate theorems. Adjust it to output structured JSON.
```
You are a Popperian theory-construction agent operating over a simulated universe.

You are given:
- A set of candidate laws with status ∈ {PASS, FAIL, UNKNOWN}
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

---

### OUTPUT FORMAT (STRICT)

For each theorem, output the following sections:

THEOREM <N>: <Short descriptive name>

STATUS:
- {Established | Conditional | Conjectural}

CLAIM:
- A precise statement in plain language.
- If helpful, include symbolic expressions using existing observables.

SUPPORT:
- List of specific law IDs (PASS or FAIL) that support or constrain this theorem.
- Brief explanation of how each law contributes.

EXPLANATORY ROLE:
- What this theorem explains that individual laws do not.
- What intuition or mechanism it suggests about the universe.

FAILURE MODES:
- Explicit conditions or scenarios under which this theorem could be false.
- Reference failed laws that delimit its scope.

MISSING STRUCTURE:
- If this theorem appears to require additional observables to be sharpened or tested,
  list what kind of structure is missing (e.g., local adjacency, gap distribution),
  WITHOUT inventing names or formulas for them.

---

### META-CONSTRAINTS

- Do NOT claim eventual or asymptotic behavior unless directly supported by PASS laws.
- Do NOT assume monotonicity unless it is explicitly supported.
- Treat UNKNOWN laws as unresolved; they may support or undermine a theorem, but must be cited cautiously.
- Prefer local explanations when global-count-based explanations repeatedly fail.

Produce 5-10 theorems.

``` 
