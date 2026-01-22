from src.harness.generators.base import Generator, GeneratorRegistry
from src.harness.generators.constrained_pairs import ConstrainedPairGenerator
from src.harness.generators.edge_wrapping import EdgeWrappingGenerator
from src.harness.generators.random_density import RandomDensityGenerator
from src.harness.generators.symmetry_suite import SymmetryMetamorphicGenerator
from src.harness.generators.adversarial import (
    AdversarialMutationGenerator,
    GuidedAdversarialGenerator,
)

# Register all generators
GeneratorRegistry.register("random_density_sweep", RandomDensityGenerator)
GeneratorRegistry.register("constrained_pair_interactions", ConstrainedPairGenerator)
GeneratorRegistry.register("edge_wrapping_cases", EdgeWrappingGenerator)
GeneratorRegistry.register("symmetry_metamorphic_suite", SymmetryMetamorphicGenerator)
GeneratorRegistry.register("adversarial_mutation_search", AdversarialMutationGenerator)
GeneratorRegistry.register("guided_adversarial_search", GuidedAdversarialGenerator)

__all__ = [
    "Generator",
    "GeneratorRegistry",
    "RandomDensityGenerator",
    "ConstrainedPairGenerator",
    "EdgeWrappingGenerator",
    "SymmetryMetamorphicGenerator",
    "AdversarialMutationGenerator",
    "GuidedAdversarialGenerator",
]
