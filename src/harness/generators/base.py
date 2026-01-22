"""Base class and registry for case generators."""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Type

from src.harness.case import Case


class Generator(ABC):
    """Abstract base class for test case generators.

    Generators produce test cases (initial states + configs) for
    evaluating laws against the simulator.
    """

    @abstractmethod
    def family_name(self) -> str:
        """Return the generator family name."""
        pass

    @abstractmethod
    def generate(self, params: dict[str, Any], seed: int, count: int) -> list[Case]:
        """Generate test cases.

        Args:
            params: Generator-specific parameters
            seed: Random seed for reproducibility
            count: Number of cases to generate

        Returns:
            List of generated test cases
        """
        pass

    def params_hash(self, params: dict[str, Any]) -> str:
        """Compute a hash of the parameters for caching."""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]


class GeneratorRegistry:
    """Registry of available generators."""

    _generators: dict[str, Type[Generator]] = {}

    @classmethod
    def register(cls, name: str, generator_class: Type[Generator]) -> None:
        """Register a generator class."""
        cls._generators[name] = generator_class

    @classmethod
    def get(cls, name: str) -> Type[Generator] | None:
        """Get a generator class by name."""
        return cls._generators.get(name)

    @classmethod
    def create(cls, name: str) -> Generator | None:
        """Create a generator instance by name."""
        gen_class = cls._generators.get(name)
        if gen_class is None:
            return None
        return gen_class()

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available generator names."""
        return list(cls._generators.keys())
