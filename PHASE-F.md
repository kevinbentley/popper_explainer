I'd like to focus on the orchestration right now.
Right now it proposes laws and we decide when the rules are good enough to move to theorem generation. Now, we need to let an orchestration agent decide the flow. 

When we've received enough consistently high readiness metrics from the discovery loop (the quality of the laws being discovered is diminishing), it moves on to theorem generation. 

The theorem generator can then decide if more refinement is needed. For instance, it would make a list of theorems and emit some signal in the LLM output to return to law discovery. 

The current theorem list would get pushed to the discovery loop, so proposals could be centered around falsifying the theories. 

Once it reaches the discovery metric again, it pushes those laws back through the theorem generator. 

The explainer would be similar. Once the theorem generator has a certain level of quality, it is sent to the explainer, which creates some explanations, but can also generate a list of open questions, or criticisms of the theorems. 

Then that is sent back to the theorem generator. How do we know when the explainer is done? We have it make predictions. It can say "given state x at t=0, we will see y at t=1". 
Then we run that through the simulator and give it what really happened. Once the scores are very high, we have a valid explanation. 

I think we have much of the plumbing done for this, but we need a good orchestration engine and the prompts to make the LLM participate in the orchestration.


---
UPDATE: 
I think the core idea is solid: you are turning the LLM from a “proposal factory” into a **self-governing scientist** that can decide when to change phases, but you will only get reliable behavior if you (a) make the LLM’s scores *auditable* and (b) anchor every phase gate to *objective* signals from the harness (not just self-reported confidence).

Here is a version of your loop that is tight, debuggable, and hard for the model to bullshit.

---

## 1) Make orchestration a state machine with explicit, machine-checked signals

Do not let “the agent decides” mean “free-form narrative.” Instead:

* Every phase outputs a **JSON control block** with:

  * `readiness_score` (0–100)
  * `stop_reason` (enum)
  * `evidence` (structured, referential)
  * `requests` (what it needs next)
  * `proposed_transitions` (what to do next)

Then your orchestrator is deterministic:

* It aggregates the last N control blocks and harness metrics.
* It applies a transition rule (simple thresholds + hysteresis).
* It calls the next phase prompt.

This keeps you from building a “vibes-based” pipeline.

### Recommended phases

1. **Law Discovery**
2. **Theorem Synthesis** (compress/organize, propose derivations, identify gaps)
3. **Explanation Synthesis** (propose mechanism + predictions)
4. **Prediction/Refutation Loop** (harness scores explanations)
5. **Finalize** (freeze artifacts, generate report)

---

## 2) Readiness should be a *calibrated* score, not “confidence”

Your readiness score should be computed from *observable* metrics, with the LLM only supplying the interpretation.

Example: define readiness as a weighted function:

* `S_pass`: fraction of “core law suite” passing
* `S_stability`: promoted-law set stability across last K iterations
* `S_novel_cex`: rate of new counterexample *classes* discovered (lower is better)
* `S_harness_health`: no harness errors / no parse gaps / no invalid initial states
* `S_redundancy`: % of new laws semantically redundant with existing ones (higher is better, indicates saturation)

Then have the LLM propose:

* the weights (within bounds),
* whether the current suite is missing any *obvious* categories,
* and whether failures look like “law wrong” vs “observable/measurement mismatch” vs “harness bug.”

But the orchestrator computes the numeric score, or at least checks the LLM’s score against these metrics.

**Key point:** the LLM can *advise* readiness; it should not be the only input.

---

## 3) Use hysteresis and two-way gates so you do not oscillate

A common failure mode: the LLM flips phases too easily.

Use hysteresis:

* Move **Discovery → Theorems** only if readiness ≥ 85 for 3 consecutive rounds *and* `S_harness_health=1` *and* `S_novel_cex < threshold`.
* Move **Theorems → Discovery** only if theorem phase reports a **specific missing lemma** or **contradiction** and provides at least one targeted falsification objective.

Same for explainer.

---

## 4) Theorem generator as “critic + compressor,” not just a list-maker

The theorem stage should produce:

* A *basis* of laws (minimal-ish set)
* Candidate theorem statements grouped by:

  * invariants
  * symmetries
  * monotonicity/termination-like behavior (if any)
  * local-causality rules (incoming collisions etc.)

And it should emit:

* `needs_refinement: true/false`
* `refinement_targets`: a list of *very specific* things to test next

Example refinement targets:

* “We need an observable for IncomingCollisions(t) because multiple ‘adjacent’ laws are ill-posed under the actual rules.”
* “We need to disambiguate ‘ApproachingPairs’ (adjacent `><` vs `>.<`)—current failures suggest we are measuring the wrong pattern.”

This creates a clean handshake: theorem generator tells discovery what to do, not “go discover more.”

---

## 5) Explanation is “model + predictions,” and predictions are scored by the harness

Your “explainer is done when predictions are very high” is exactly right, but tighten it:

### Prediction types (start simple)

* **One-step predictions:** given `s_t`, predict `s_{t+1}`
* **k-step predictions:** predict `s_{t+k}` for small k
* **Observable predictions:** predict `CollisionCells(t+1)`, `FreeMovers(t+1)`, etc.

### Scoring

* Exact match on next state for one-step (strict)
* Hamming distance as a softer metric
* Per-observable MAE / exact for integer observables

### Stopping rule

Explainer “done” when:

* One-step state accuracy ≥ X% on a held-out probe set
* AND it does not collapse on adversarial/minimized cases
* AND it can explain *why* the edge cases behave as they do (textual, but anchored)

Also: maintain a **held-out set** that the explainer never sees during its iterative refinement. Otherwise it will overfit to your feedback loop.

---

## 6) Orchestration engine: what you need (minimal but sufficient)

You do not need a huge framework. You need:

1. **Typed phase interface**

   * input payload
   * output payload + control block
2. **Artifact store**

   * laws, counterexamples, theorem drafts, explanation drafts, prediction sets
3. **Metrics computation**

   * objective readiness components
4. **Transition policy**

   * hysteresis rules
5. **Traceability**

   * every LLM claim references specific artifacts (law IDs, cex IDs, metric snapshots)

If you build those 5, the system becomes “explainable by construction.”

---

## 7) Prompts: the pattern that works

Use **role-separated prompts** and force structure.

### A) Discovery agent prompt skeleton

* Inputs: current law set, last N counterexamples (minimized), current metrics snapshot, “open targets” from theorem/explainer
* Outputs:

  * proposed new laws (small number)
  * proposed observable tweaks (if needed)
  * a control block with:

    * `readiness_score_suggestion`
    * `phase_recommendation`: stay/advance
    * `why` referencing metrics + artifacts

### B) Theorem generator prompt skeleton

* Inputs: promoted laws, failure classes summary, symmetries, observables definitions
* Outputs:

  * grouped theorems
  * “derivation sketch” graph (even informal)
  * missing lemmas / ambiguous observables
  * control block with `needs_refinement`

### C) Explainer prompt skeleton

* Inputs: theorem list + key counterexamples
* Outputs:

  * mechanistic explanation hypothesis
  * prediction batch spec (states + horizons + what to predict)
  * control block with “confidence” *and* “what would falsify this”

---

## 8) The big risk and how to avoid it

The risk is the LLM learning to **game the score** or “declare done” prematurely.

Mitigations:

* Readiness computed from harness metrics (LLM only advises)
* Held-out evaluation sets
* Adversarial generation that the LLM does not control
* Require artifact references for every transition claim
* Keep a small “core law suite” and do not let it drift

---

## Where I think you should start

Given your current plumbing, the highest ROI next step is:

1. Implement the **control block schema** and phase state machine.
2. Define 5–7 objective readiness metrics and compute them every loop.
3. Make “advance phase” require both:

   * objective thresholds, and
   * LLM recommendation with citations to artifacts.

If you want, paste (or describe) what you currently persist per iteration (laws, cex, metrics). I can propose an exact JSON schema for the control block + a concrete transition policy (with hysteresis) that fits what you already have.
---
If the LLM participates in the decision making process, the nature of the loop (reliable predictions == success), the cycle should eventually stabilize. But the danger is that we end up in a local minima and can never break out of a cycle.

---
“Prediction reliability” is a great *global* objective, but it does not guarantee you will escape local minima, because you can get a stable attractor that is **self-consistent but wrong** (or “right” only on the slice of states you keep sampling).

The way out is to make the system optimize *two* things at once:

1. **Accuracy on predictions**
2. **Pressure to search for falsifiers / novelty** (so it cannot settle too early)

Think of it like simulated annealing: you need an exploration signal that does not vanish just because the model found a comfortable explanation.

Here are the practical, non-hand-wavy ways to do that.

---

## 1) Define “done” as accuracy on a held-out + adversarial battery

If the explainer is only evaluated on states it (directly or indirectly) influenced, it can lock into a local minimum.

So you keep three evaluation pools:

* **Held-out random**: generated by a generator the LLM cannot steer.
* **Adversarial**: generated by a search procedure whose objective is “maximize disagreement with explainer predictions.”
* **Regression set**: all previously found minimized counterexamples + near-neighbors.

A stable local minimum will usually fail on at least one of those.

Concrete stop criterion:

* `Acc_heldout ≥ 0.98` and `Acc_adv ≥ 0.90` for K rounds
* and no new counterexample classes found for M rounds

---

## 2) Add an explicit “novelty budget” so the system must keep looking for trouble

Even if you are outcome-based, you can require every cycle to “pay rent” by attempting to falsify.

Make the orchestrator enforce:

* each iteration must include `N_falsification_attempts`
* and measure `novelty_rate` = new counterexample clusters / attempts

Then you treat a drop in novelty as *evidence of saturation*, not as permission to stop exploring.

This is how you avoid “we got 95% accurate and stopped” when the remaining 5% is systematic.

---

## 3) Detect local minima with plateau triggers and forced perturbations

You can formalize “we are stuck” as a plateau:

Plateau conditions (example):

* readiness score (or accuracy) improves by < ε over the last P iterations
* and the set of proposed laws/theorems has high redundancy
* and counterexamples repeat the same few patterns

When plateau triggers, the orchestrator forces one of these perturbations:

### A) Generator shift (“change the world you test on”)

* switch distribution: longer grids, higher density, near-wrap interactions, etc.
* or stratify by regime: sparse vs dense vs periodic-crowded

### B) Objective shift (“change what you reward”)

* temporarily overweight the adversarial set
* or overweight rare observables (spread, max_gap, boundary interactions)

### C) Representation shift (“change the language of laws”)

* ban certain templates for a while (e.g., monotone/eventually)
* require local-causality laws only (“if pattern P in neighborhood, then outcome Q”)

### D) Seed theorem forcing

* ask theorem phase to produce *one* theorem that would be very useful **if true**
* then discovery must attempt to falsify it aggressively

These are controlled “kicks” to escape attractors.

---

## 4) Keep the system honest with calibration checks

Self-governing loops fail when the model learns “what the orchestrator wants to hear.”

Two simple calibration tests:

### A) Abstention / uncertainty penalty

Require the explainer to output:

* prediction + confidence
  Score with a proper scoring rule (Brier/log loss), not just accuracy.

Overconfident wrong predictions get punished harder, which discourages premature stabilization.

### B) Contradiction audit

Have a separate “auditor” pass that only checks:

* are any claimed laws/theorems mutually inconsistent?
* do they contradict known counterexamples?

If auditor finds contradiction, you force a return loop.

---

## 5) Make “progress” multi-dimensional, not single-metric

A single metric like one-step accuracy can still local-min you (especially if the CA has symmetries and easy cases).

Track:

* one-step state accuracy
* k-step rollouts (error compounding)
* per-observable accuracy
* adversarial accuracy
* novelty rate
* theorem compression ratio (how many promoted laws explain how many passes)

You escape local minima when you can improve on *any* axis by changing strategy.

---

## 6) The “local minimum” you should expect in your universe

Given your update rules, a classic trap is:

* the system learns local rules that predict most steps
* but fails specifically around:

  * **X resolution interactions** (X emits both directions)
  * **wrap-around collisions**
  * **multi-collision simultaneous arrivals**

So your adversarial generator should explicitly hunt those regimes:

* high density near boundaries
* patterns with X flanked by movers at distance 2
* alternating blocks that create synchronized arrivals

You do not need the LLM to invent that; you can bake it into adversarial search heuristics.

---

## 7) A simple “no-local-minimum” orchestration policy that works

Use a two-mode controller:

### Mode 1: Exploit (improve explanation)

* run explanation → predictions → update

### Mode 2: Explore (try to break it)

* run adversarial search against the explainer
* cluster failures
* feed clusters to discovery/theorem generator

Switching rule:

* If `Acc_adv` is below threshold OR novelty rate is still high → stay in Explore
* If `Acc_adv` is high AND novelty is near zero → go back to Exploit and attempt finalize

This prevents “stuck polishing” and prevents “stuck exploring.”

---

## Bottom line

An outcome-based loop *can* stabilize, but only if:

* evaluation is partially independent of the model,
* you maintain adversarial pressure,
* and you have plateau-triggered perturbations.

If you tell me what your current test generator controls are (grid length range, density controls, whether X is allowed at t=0, etc.), I can propose a concrete plateau trigger + perturbation schedule that fits your harness exactly.

---

So we need a plan for:
1. The orchestration engine
2. Quality metrics for laws
3. Metrics for when to go from theorem generation -> Explanation generation
4. A way for the explanation generation to kick things back to the theorem generator (which then might kick back to discovery)
5. A Prediction verifier (given state x at T=0, we should see y at T=1) The verifier can return pass/fail on the predictions, but also the actual outcome
6. When a certain threshold of predictions are correct (all of the time? 90% of the time?), the explanation is sent to an LLM to generate the most concise explanation of the system as possible.

