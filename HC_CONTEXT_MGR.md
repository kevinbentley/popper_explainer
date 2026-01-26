This document outlines the architecture for the **Agentic High-Context Rolling Cache (AHC-RC)** system. This design enables the discovery agent to handle extremely long experimental sessions without hitting the 1M+ token limit or suffering from "contextual noise" while maintaining a persistent record of all findings in **SQLite**.

---

## 1. Context Architecture: The Three-Tier Memory

To keep tokens low and reasoning high, the prompt must be dynamically constructed for every turn.

### Tier 1: The Permanent Core (Fixed)

This is the "Axiomatic" layer. It is sent at the top of every prompt and never changes.

* 
**System Identity:** The Popperian Scientist persona.


* 
**Tool Definitions:** JSON schemas for `evaluate_laws`, `run_prediction`, `query_log`, etc.


* 
**Grammar:** The symbols (`_, A, B, K`) and instrument primitives (`neighbor_config`, `coordinate_class`).



### Tier 2: The Evolving Theory (Dynamic)

This represents the agent's current "World Model."

* 
**Established Theorems:** The latest output from `retrieve_theorems`.


* 
**Negative Knowledge Summary:** A concise list of previously falsified "high-level" concepts (e.g., "Active symbols are not conserved," "A and B symmetry is broken").



### Tier 3: The Rolling Journal (Last  Turns)

A sliding window of the most recent raw interactions to allow for immediate local troubleshooting.

* 
**Size:** The last 10–15 turns (Journal entries and Tool Results).


* 
**Function:** Provides the specific counterexample trajectories for the agent's current active hypotheses.



---

## 2. The Compaction Workflow (Maintenance)

When the rolling window (Tier 3) exceeds a set token threshold (e.g., 50k tokens), the system triggers a **Compaction Event**.

1. 
**Summarization:** The system makes a hidden API call to the LLM: *"Analyze the following 50 turns. Update the 'Negative Knowledge' summary and identify any laws that have consistently passed but haven't been stored as theorems yet"*.


2. **Snapshotting:** The current confirmed theorems and the updated summary are written to the SQLite `meta_knowledge` table.
3. **Buffer Reset:** The raw text of those 50 turns is removed from the active prompt. The new summary is injected at the top of Tier 2.

---

## 3. SQLite Database Schema for the Coder

To support this, the database must store the "Turn History" and the "Knowledge State" separately.

```sql
-- Records every raw thought and tool exchange
CREATE TABLE interaction_log (
    turn_id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    journal_entry TEXT,      -- The agent's reasoning (LAB NOTE)
    tool_call_json TEXT,     -- The raw JSON emitted
    tool_result_json TEXT,   -- The raw result/counterexample from the harness
    token_count INTEGER      -- Useful for triggering compaction
);

-- Stores the state of synthesized knowledge
CREATE TABLE meta_knowledge (
    version_id INTEGER PRIMARY KEY,
    theorems_json TEXT,      -- Current valid theorems
    negative_knowledge TEXT, -- Summarized falsifications
    last_compacted_turn INTEGER
);

```

---

## 4. The Agent's Internal "Memory Query" Tool

The agent must be taught to use the `query_log` tool as its "Long-term Memory Access."

* 
**Logic:** If the agent sees a failure and thinks, *"I've seen this trajectory before,"* it shouldn't try to remember—it should call `query_log(pattern="AA_BB")`.


* **Harness Response:** The system searches the `interaction_log` and returns only the relevant previous turns. This keeps the prompt lean while allowing "flashbacks" to critical data.



---

## 5. Implementation Notes for the Coder

* **Rolling Buffer Logic:** In the Python script, use a simple `tail` on the `interaction_log` table to build the message history.
* 
**Opaque Prediction:** Ensure `run_prediction` only returns `PASS/FAIL` or a trajectory if it fails; never return the ground-truth outcome directly.


* 
**Periodic Boundaries:** The `neighbor_config` tool must use modulo math (e.g., `(i - 1) % grid_len`) to handle the circular grid.



Would you like me to generate the **Compaction Prompt** that the agent will use to self-summarize its progress during these maintenance cycles?