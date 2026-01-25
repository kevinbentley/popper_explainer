from src.harness.generators.base import Generator, GeneratorRegistry
from src.harness.generators.constrained_pairs import ConstrainedPairGenerator
from src.harness.generators.edge_wrapping import EdgeWrappingGenerator
from src.harness.generators.extreme_states import ExtremeStatesGenerator
from src.harness.generators.pathological import PathologicalGenerator
from src.harness.generators.random_density import RandomDensityGenerator
from src.harness.generators.symmetry_suite import SymmetryMetamorphicGenerator
from src.harness.generators.adversarial import (
    AdversarialMutationGenerator,
    GuidedAdversarialGenerator,
)
from src.harness.generators.precondition_breaker import (
    PreconditionBreakingGenerator,
    MultiplicityGenerator,
    PeriodicBoundaryGenerator,
    UniversalStressGenerator,
)

# Register all generators
GeneratorRegistry.register("random_density_sweep", RandomDensityGenerator)
GeneratorRegistry.register("constrained_pair_interactions", ConstrainedPairGenerator)
GeneratorRegistry.register("edge_wrapping_cases", EdgeWrappingGenerator)
GeneratorRegistry.register("symmetry_metamorphic_suite", SymmetryMetamorphicGenerator)
GeneratorRegistry.register("adversarial_mutation_search", AdversarialMutationGenerator)
GeneratorRegistry.register("guided_adversarial_search", GuidedAdversarialGenerator)
GeneratorRegistry.register("pathological_cases", PathologicalGenerator)
GeneratorRegistry.register("extreme_states", ExtremeStatesGenerator)
# Adversarial generators for breaking narrow laws
GeneratorRegistry.register("precondition_breaking", PreconditionBreakingGenerator)
GeneratorRegistry.register("multiplicity_crowding", MultiplicityGenerator)
GeneratorRegistry.register("periodic_boundary_stress", PeriodicBoundaryGenerator)
# Universal stress states - mandatory tests run FIRST for every law
GeneratorRegistry.register("universal_stress", UniversalStressGenerator)

__all__ = [
    "Generator",
    "GeneratorRegistry",
    "RandomDensityGenerator",
    "ConstrainedPairGenerator",
    "EdgeWrappingGenerator",
    "ExtremeStatesGenerator",
    "PathologicalGenerator",
    "SymmetryMetamorphicGenerator",
    "AdversarialMutationGenerator",
    "GuidedAdversarialGenerator",
    "PreconditionBreakingGenerator",
    "MultiplicityGenerator",
    "PeriodicBoundaryGenerator",
    "UniversalStressGenerator",
]
