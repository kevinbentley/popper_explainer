"""Configuration for the test harness."""

from dataclasses import dataclass, field


@dataclass
class HarnessConfig:
    """Configuration for law evaluation.

    Attributes:
        seed: Master random seed for reproducibility
        max_runtime_ms_per_law: Maximum time to spend evaluating one law
        max_cases: Maximum number of test cases to generate
        default_T: Default time horizon for simulations
        max_T: Maximum allowed time horizon
        min_cases_used_for_pass: Minimum cases needed for PASS verdict
        enable_adversarial_search: Whether to use adversarial case generation
        adversarial_budget: Number of adversarial attempts
        enable_counterexample_minimization: Whether to minimize counterexamples
        minimization_budget: Max attempts for minimization
        store_full_trajectories: Whether to store complete trajectories
        require_non_vacuous: Require non-vacuous tests for implications
    """

    seed: int = 42
    max_runtime_ms_per_law: int = 5000
    max_cases: int = 300
    default_T: int = 50
    max_T: int = 200
    min_cases_used_for_pass: int = 50
    enable_adversarial_search: bool = True
    adversarial_budget: int = 1500
    enable_counterexample_minimization: bool = True
    minimization_budget: int = 300
    store_full_trajectories: bool = False
    require_non_vacuous: bool = True

    # Generator weights for default strategy
    # pathological_cases is critical for catching false positives from uniform grids
    generator_weights: dict[str, float] = field(default_factory=lambda: {
        "random_density_sweep": 0.35,
        "constrained_pair_interactions": 0.25,
        "edge_wrapping_cases": 0.15,
        "symmetry_metamorphic_suite": 0.15,
        "pathological_cases": 0.10,  # Uniform grids, alternating patterns, edge cases
    })

    def content_hash(self) -> str:
        """Compute a hash of the configuration for reproducibility."""
        import hashlib
        import json

        content = {
            "seed": self.seed,
            "max_cases": self.max_cases,
            "default_T": self.default_T,
            "max_T": self.max_T,
            "min_cases_used_for_pass": self.min_cases_used_for_pass,
            "enable_adversarial_search": self.enable_adversarial_search,
            "adversarial_budget": self.adversarial_budget,
            "require_non_vacuous": self.require_non_vacuous,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
