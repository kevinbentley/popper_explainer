## Design Document: Agentic High-Context Discovery System (AHC-DS)

### 1. Executive Summary

The **Agentic High-Context Discovery System (AHC-DS)** is an autonomous research framework designed to reverse-engineer the hidden physics of a discrete 1D universe. Unlike batch-processing models, this system utilizes a high-context "Live Journal" architecture, where an LLM agent manages its own discovery loop, performs experiments via tool calls, and maintains a persistent memory of all findings within a **SQLite-backed** environment.

---

### 2. Core Workflow: The "Think-Act-Learn" Loop

The system operates in a stateful, continuous loop until the universe is mathematically solved.

1. **Reset (Tabula Rasa)**: The SQLite database is purged, and the Agent is seeded with the "You Know Nothing" prompt.
2. **Reasoning (Journaling)**: The Agent writes a natural language entry in its journal (Chain-of-Thought) to synthesize past observations and define a current objective.
3. **Actuation (Function Call)**: The Agent emits a structured function call (e.g., `evaluate_laws`) to the Physics Harness.
4. **Observation (Logging)**: The Harness executes the request against the simulator and appends the raw results (successes/failures/counterexamples) to the SQLite log and conversation history.
5. **Formalization**: Upon repeated success, the Agent calls `store_theorem` to lock in established physical principles.

---

### 3. Functional Interface (The Toolset)

The Agent interacts with the environment through the following specific function calls:

#### **A. Experimental Tools**

* **`evaluate_laws(list[AST])`**:
* **Input**: A list of candidate laws in Abstract Syntax Tree format.
* **Output**: Pass/Fail results per law, including counterexample trajectories for failures.


* **`run_prediction(state_t0)`**:
* **Input**: A specific 1D grid configuration.
* **Output**: The ground-truth `state_t1` from the simulator.


* **`request_samples(pattern_filter, count)`**:
* **Input**: A string pattern (e.g., "A_B") and the number of samples.
* **Output**: Trajectory slices containing those specific configurations.



#### **B. Memory & Knowledge Management**

* **`store_theorem(name, law_ids, description)`**: Saves a validated set of laws as a permanent theorem in the SQLite DB.
* **`retrieve_theorems(filter)`**: Fetches established theorems to ensure new hypotheses do not contradict known physics.
* **`edit_theorem(theorem_id, updates)`**: Refines or restricts existing theorems based on new counterexamples.
* **`query_log(query)`**: Allows the Agent to search its own SQLite history for specific data points or previous failures.

---

### 4. Instrumentation (The Microscope)

The Agent has access to the following opaque primitives within its `evaluate_laws` calls:

* **`neighbor_config(i)`**: Returns the local 3-cell state context (Left, Center, Right).
* **`coordinate_class(i)`**: Returns , enabling discovery of grid-phase/parity effects.
* **`transition_indicator`**: A global non-negative scalar indicating system-wide activity.

---

### 5. Termination: The "Unified Theory"

The experiment concludes when the Agent demonstrates a **Theoretical Plateau**:

* **Predictive Success**: The Agent achieves 100% accuracy on `run_prediction` for 5,000 randomized states.
* **Symbolic Completion**: All symbols (`A, B, _, K`) have a complete local transition function mapping any possible `neighbor_config` and `coordinate_class` to a deterministic next state.

---
