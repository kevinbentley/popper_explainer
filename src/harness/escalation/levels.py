"""Escalation level definitions and preset configurations.

Escalation levels define progressively stronger harness configurations
for re-testing accepted laws to find false positives.
"""

from dataclasses import dataclass
from enum import Enum

from src.harness.config import HarnessConfig


class EscalationLevel(str, Enum):
    """Named escalation levels with progressively stronger testing."""

    BASELINE = "baseline"
    ESCALATION_1 = "escalation_1"
    ESCALATION_2 = "escalation_2"
    ESCALATION_3 = "escalation_3"


@dataclass
class EscalationPreset:
    """A preset configuration for an escalation level.

    Attributes:
        level: The escalation level this preset corresponds to
        config: The HarnessConfig with escalated parameters
        description: Human-readable description of this level
        cost_factor: Relative cost vs baseline (for budgeting)
    """

    level: EscalationLevel
    config: HarnessConfig
    description: str
    cost_factor: float


# Baseline generator weights (same as default HarnessConfig)
_BASELINE_WEIGHTS = {
    "random_density_sweep": 0.4,
    "constrained_pair_interactions": 0.3,
    "edge_wrapping_cases": 0.15,
    "symmetry_metamorphic_suite": 0.15,
}

# Level 2+ emphasizes symmetry testing
_ESCALATION_2_WEIGHTS = {
    "random_density_sweep": 0.35,
    "constrained_pair_interactions": 0.25,
    "edge_wrapping_cases": 0.15,
    "symmetry_metamorphic_suite": 0.25,
}

# Level 3 adds more adversarial focus
_ESCALATION_3_WEIGHTS = {
    "random_density_sweep": 0.3,
    "constrained_pair_interactions": 0.25,
    "edge_wrapping_cases": 0.15,
    "symmetry_metamorphic_suite": 0.30,
}


_PRESETS: dict[EscalationLevel, EscalationPreset] = {
    EscalationLevel.BASELINE: EscalationPreset(
        level=EscalationLevel.BASELINE,
        config=HarnessConfig(
            seed=42,
            max_runtime_ms_per_law=5000,
            max_cases=300,
            default_T=50,
            max_T=200,
            min_cases_used_for_pass=50,
            enable_adversarial_search=True,
            adversarial_budget=1500,
            enable_counterexample_minimization=True,
            minimization_budget=300,
            require_non_vacuous=True,
            generator_weights=_BASELINE_WEIGHTS,
        ),
        description="Fast/cheap baseline testing (default discovery config)",
        cost_factor=1.0,
    ),
    EscalationLevel.ESCALATION_1: EscalationPreset(
        level=EscalationLevel.ESCALATION_1,
        config=HarnessConfig(
            seed=42,
            max_runtime_ms_per_law=10000,
            max_cases=600,
            default_T=75,
            max_T=300,
            min_cases_used_for_pass=100,
            enable_adversarial_search=True,
            adversarial_budget=2500,
            enable_counterexample_minimization=True,
            minimization_budget=400,
            require_non_vacuous=True,
            generator_weights=_BASELINE_WEIGHTS,
        ),
        description="2x cases, 1.5x horizon, stronger pass threshold",
        cost_factor=2.5,
    ),
    EscalationLevel.ESCALATION_2: EscalationPreset(
        level=EscalationLevel.ESCALATION_2,
        config=HarnessConfig(
            seed=42,
            max_runtime_ms_per_law=20000,
            max_cases=900,
            default_T=100,
            max_T=400,
            min_cases_used_for_pass=150,
            enable_adversarial_search=True,
            adversarial_budget=4000,
            enable_counterexample_minimization=True,
            minimization_budget=500,
            require_non_vacuous=True,
            generator_weights=_ESCALATION_2_WEIGHTS,
        ),
        description="3x cases, 2x horizon, emphasis on symmetry tests",
        cost_factor=4.0,
    ),
    EscalationLevel.ESCALATION_3: EscalationPreset(
        level=EscalationLevel.ESCALATION_3,
        config=HarnessConfig(
            seed=42,
            max_runtime_ms_per_law=30000,
            max_cases=1500,
            default_T=150,
            max_T=500,
            min_cases_used_for_pass=200,
            enable_adversarial_search=True,
            adversarial_budget=6000,
            enable_counterexample_minimization=True,
            minimization_budget=600,
            require_non_vacuous=True,
            generator_weights=_ESCALATION_3_WEIGHTS,
        ),
        description="Maximal testing: 5x cases, 3x horizon, heavy adversarial",
        cost_factor=8.0,
    ),
}


def get_preset(level: EscalationLevel) -> EscalationPreset:
    """Get the preset configuration for an escalation level.

    Args:
        level: The escalation level

    Returns:
        EscalationPreset with config and metadata
    """
    return _PRESETS[level]


def get_config(level: EscalationLevel, seed: int | None = None) -> HarnessConfig:
    """Get the harness config for an escalation level.

    Args:
        level: The escalation level
        seed: Optional seed override (defaults to preset seed)

    Returns:
        HarnessConfig configured for the escalation level
    """
    import dataclasses

    preset = get_preset(level)
    if seed is not None:
        return dataclasses.replace(preset.config, seed=seed)
    return preset.config


def list_levels() -> list[EscalationPreset]:
    """List all available escalation levels with their presets.

    Returns:
        List of EscalationPreset in order from baseline to highest
    """
    return [
        _PRESETS[EscalationLevel.BASELINE],
        _PRESETS[EscalationLevel.ESCALATION_1],
        _PRESETS[EscalationLevel.ESCALATION_2],
        _PRESETS[EscalationLevel.ESCALATION_3],
    ]
