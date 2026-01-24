
## 1) Build the failure signature you actually cluster

Even though you already store `failure_signature_hash`, you still want the **text signature** for clustering.

For each theorem:

**signature_text =**

* `missing_structure` bullets
* `failure_modes` bullets
* * the `law_id`s where `role == "constrains"` (as tokens)

Why constraining law IDs: they’re stable and differentiate clusters like “eventuality” vs “collision trigger.”

**Python-ish:**

```python
def make_signature(thm):
    parts = []
    parts += thm["missing_structure"]
    parts += thm["failure_modes"]
    parts += [s["law_id"] for s in thm["support"] if s["role"] == "constrains"]
    text = " ".join(parts).lower()
    return normalize(text)

def normalize(text: str) -> str:
    # keep underscores (law IDs)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

Store this as `failure_signature_text` in your DB. Keep your hash too.

---

## 2) Pass A: multi-label “bucket tagging” (deterministic)

Use bucket tags as the first clustering pass. Each theorem can belong to multiple buckets.

Here’s a bucket set that fits your run:

### Buckets + keywords

* **DEFINITION_GAP**: `definition`, `precise`, `clear definitions`, `constitutes`, `what is`, `context`
* **COLLISION_TRIGGERS**: `incoming collision`, `trigger`, `prevent`, `guarantee`, `bound`, `insufficient`
* **LOCAL_PATTERN**: `configuration`, `arrangement`, `adjacent`, `pairs`, `spread`, `gap`, `neighbors`
* **EVENTUALITY**: `eventually`, `long-term`, `asymptotic`, `cease`, `resolve`, `attractor`
* **MONOTONICITY**: `monotonic`, `non-increasing`, `non-decreasing`, `never increase`, `strictly`
* **SYMMETRY**: `symmetric`, `mirror`, `swap`, `reflection`, `translational`, `shift`

Implementation:

```python
BUCKETS = {
  "DEFINITION_GAP": ["definition", "precise", "clear definitions", "constitutes", "context"],
  "COLLISION_TRIGGERS": ["incoming collision", "incoming collisions", "trigger", "prevent",
                         "guarantee", "insufficient", "bound", "bounded"],
  "LOCAL_PATTERN": ["configuration", "arrangement", "adjacent", "pairs", "spread",
                    "gap", "neighbors"],
  "EVENTUALITY": ["eventually", "long term", "asymptotic", "cease", "resolve", "attractor"],
  "MONOTONICITY": ["monotonic", "non increasing", "non decreasing", "never increase", "strictly"],
  "SYMMETRY": ["symmetric", "mirror", "swap", "reflection", "translation", "shift"],
}

def tag_buckets(signature_text):
    tags = set()
    for bucket, kws in BUCKETS.items():
        if any(kw in signature_text for kw in kws):
            tags.add(bucket)
    return tags
```

Store these tags (array) per theorem.

---

## 3) Pass B: cluster *within each bucket* (TF-IDF + agglomerative)

For each bucket:

* collect theorems with that bucket tag
* vectorize `failure_signature_text` via TF-IDF
* cluster using agglomerative (cosine distance) with a threshold

This avoids picking K and works on small batches.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

def cluster_bucket(texts, theorem_ids, distance_threshold=0.6):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(texts)

    # Agglomerative wants a distance matrix if metric='precomputed'
    D = cosine_distances(X)

    model = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None
    )
    labels = model.fit_predict(D)

    clusters = {}
    for thm_id, lab in zip(theorem_ids, labels):
        clusters.setdefault(int(lab), []).append(thm_id)

    # keywords per cluster
    feature_names = vec.get_feature_names_out()
    X_dense = X.toarray()
    cluster_info = []
    for lab, ids in clusters.items():
        idxs = [theorem_ids.index(i) for i in ids]
        centroid = X_dense[idxs].mean(axis=0)
        top = centroid.argsort()[-8:][::-1]
        keywords = [feature_names[i] for i in top if centroid[i] > 0]
        cluster_info.append({"label": lab, "theorem_ids": ids, "keywords": keywords[:8]})
    return cluster_info
```

Tuning:

* start with `distance_threshold` between **0.55–0.7**
* smaller threshold → more clusters

---

## 4) What you’ll see on *this* dataset

Even before semantic clustering, Pass A will already give you clear groups.

### Expected bucket memberships

* **DEFINITION_GAP**: 1–8,10 (almost all)
* **COLLISION_TRIGGERS**: 3,7,8,10
* **LOCAL_PATTERN**: 1,4,5,7,9,10 (because of gaps/spread/pairs/configuration)
* **EVENTUALITY**: 9 only (and possibly 5 if “trend” wording matches)
* **MONOTONICITY**: 5 and 9
* **SYMMETRY**: 2 and 6

Then Pass B will split buckets into meaningful subclusters. For example:

**COLLISION_TRIGGERS** likely yields two clusters:

* “zero-count conditions don’t simplify” → theorem 8 (+ parts of 3)
* “presence/absence logic for collisions” → theorem 10 + theorem 3
  and theorem 7 sits between (“bounds are complex, configuration matters”)

**LOCAL_PATTERN** likely yields:

* “gaps / max_empty_gap / empty-space bounds” → theorem 1
* “spread/pairs invariants fail” → theorem 4 (+ parts of 5)
* “collision triggers depend on configuration” → theorem 10 (+ parts of 7)

This is exactly what you want, because each subcluster maps to a small set of observables.

---

## 5) Turn cluster → next action (this is the secondary loop)

After clustering, you don’t just want labels; you want an **action decision**.

Use this deterministic mapping:

### If cluster keywords contain mostly “definition/precise/clear”

→ **Action: schema/prompt fix**, not new observables.

Do this by injecting an “observable glossary” into theorem prompting (at theorem-time only), e.g.:

* FreeMovers = count('>') + count('<')
* CollisionCells = count('X')
* IncomingCollisions definition (or at least: “depends on neighbors, not global counts”)

### If cluster keywords contain “adjacent/pairs/spread/gap/configuration”

→ **Action: propose local-structure observables**
Minimal set that will pay off immediately:

1. Adjacent pair counts:

* `count('><')`, `count('<>')`
* `count('X>')`, `count('<X')`, `count('XX')`, etc.

2. Alternation index:

* number of i where `state[i] != state[i+1]`

3. Bracketing count (cell-centered):

* number of j where left neighbor ∈ {>,X} and right neighbor ∈ {<,X}

  * this one should basically coincide with IncomingCollisions(t)

### If cluster keywords contain “eventually/long-term/cease/resolve”

→ **Action: policy / template gating**
Do *not* add observables unless you explicitly want long-term phenomenology.
Instead:

* ban “eventually” theorem claims unless the support set includes a PASS eventuality law
* or require a Lyapunov-like monotone quantity before allowing eventuality conjectures

---

## 6) One practical improvement: add “typed missing structure”

Right now missing structure is generic. If you slightly change theorem prompting, clustering becomes almost trivial.

Ask the LLM to output missing structure as objects like:

```json
{"type":"DEFINITION_MISSING","target":"FreeMovers"}
{"type":"LOCAL_STRUCTURE_MISSING","target":"adjacency/pairs"}
{"type":"TEMPORAL_STRUCTURE_MISSING","target":"eventuality conditions"}
{"type":"MECHANISM_MISSING","target":"update rule / causal radius"}
```

Then clustering is just grouping by `type` + `target`.

---

## 7) Minimal deliverable checklist (what to code first)

1. `signature_text` builder and storage
2. Pass A bucket tagging
3. Pass B TF-IDF + agglomerative within each bucket
4. Cluster summary (keywords + representative theorem)
5. Mapping: cluster → action (schema fix vs new observables vs template gating)
6. Emit “observable proposals” records for LOCAL_PATTERN clusters

That’s enough to run your first real secondary loop.

