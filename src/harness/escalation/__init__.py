"""Power escalation module for re-testing accepted laws.

This module provides the infrastructure for periodically re-testing
accepted laws under progressively stronger harness configurations
to identify false positives and achieve reliable convergence.

Key concepts:
- EscalationLevel: Named presets (baseline, escalation_1, escalation_2, escalation_3)
- FlipType: Classification of verdict changes (stable, revoked, downgraded)
- run_escalation(): Main function to run escalation testing
- EscalationPolicyConfig: Configuration for when to run escalation
- EscalationState: Tracks state for policy decisions

Usage:
    from src.harness.escalation import EscalationLevel, run_escalation

    with Repository("results/discovery.db") as repo:
        result = run_escalation(EscalationLevel.ESCALATION_1, repo)
        print(f"Tested {result.laws_tested} laws")
        print(f"Revoked: {result.revoked_count}")

Integration with discovery loop:
    from src.harness.escalation import (
        EscalationPolicyConfig, EscalationState, get_escalation_decisions
    )

    config = EscalationPolicyConfig()
    state = EscalationState()

    # In discovery loop:
    run_1, run_2 = get_escalation_decisions(iteration, state, config, accepted_laws)
"""

from src.harness.escalation.flip import FlipType, RetestResult, classify_flip
from src.harness.escalation.levels import (
    EscalationLevel,
    EscalationPreset,
    get_config,
    get_preset,
    list_levels,
)
from src.harness.escalation.policy import (
    EscalationPolicyConfig,
    EscalationState,
    check_trigger_conditions,
    get_escalation_decisions,
    get_laws_for_escalation_1,
    get_promotion_status,
    is_law_promoted,
    should_run_escalation_1,
    should_run_escalation_2,
)
from src.harness.escalation.runner import (
    EscalationRunResult,
    get_escalation_summary,
    run_escalation,
)

__all__ = [
    # Levels
    "EscalationLevel",
    "EscalationPreset",
    "get_preset",
    "get_config",
    "list_levels",
    # Flip types
    "FlipType",
    "RetestResult",
    "classify_flip",
    # Runner
    "run_escalation",
    "EscalationRunResult",
    "get_escalation_summary",
    # Policy
    "EscalationPolicyConfig",
    "EscalationState",
    "check_trigger_conditions",
    "get_escalation_decisions",
    "get_laws_for_escalation_1",
    "get_promotion_status",
    "is_law_promoted",
    "should_run_escalation_1",
    "should_run_escalation_2",
]
