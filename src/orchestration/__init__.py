"""Orchestration engine for Popperian discovery loop.

This module implements a state-machine orchestration engine that governs
the scientific discovery loop through phases:

1. DISCOVERY - Law proposal and testing
2. THEOREM - Theorem synthesis from validated laws
3. EXPLANATION - Mechanistic explanation generation
4. PREDICTION - Prediction verification against held-out sets
5. FINALIZE - Report generation and artifact freeze

Key components:
- OrchestratorEngine: Main state machine (engine.py)
- ControlBlock: Structured phase output (control_block.py)
- PhaseHandler: Protocol for phase implementations (phases.py)
- ReadinessMetrics: Objective metrics from harness data (readiness.py)
- TransitionPolicy: Hysteresis-based transition rules (transitions.py)
"""

# Use lazy imports to avoid circular dependencies
# Import directly from submodules when needed:
#   from src.orchestration.control_block import ControlBlock
#   from src.orchestration.phases import Phase
#   from src.orchestration.readiness import ReadinessMetrics

__all__ = [
    # Control block
    "ControlBlock",
    "EvidenceReference",
    "PhaseRecommendation",
    "PhaseRequest",
    "ProposedTransition",
    "StopReason",
    # Phases
    "Phase",
    "PhaseContext",
    "PhaseHandler",
    "OrchestratorConfig",
    "IterationResult",
    "RunState",
    # Readiness
    "ReadinessComputer",
    "ReadinessMetrics",
]


def __getattr__(name: str):
    """Lazy import handler to avoid circular dependencies."""
    if name in (
        "ControlBlock",
        "EvidenceReference",
        "PhaseRecommendation",
        "PhaseRequest",
        "ProposedTransition",
        "StopReason",
        "create_control_block_from_llm_output",
    ):
        from src.orchestration import control_block
        return getattr(control_block, name)

    if name in (
        "Phase",
        "PhaseContext",
        "PhaseHandler",
        "OrchestratorConfig",
        "IterationResult",
        "RunState",
    ):
        from src.orchestration import phases
        return getattr(phases, name)

    if name in ("ReadinessComputer", "ReadinessMetrics"):
        from src.orchestration import readiness
        return getattr(readiness, name)

    raise AttributeError(f"module 'src.orchestration' has no attribute '{name}'")
