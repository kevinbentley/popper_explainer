"""Theorist/Auditor Reflection Engine.

Periodic sub-routine within the Discovery phase that:
1. Audits fixed laws for conflicts, tautologies, and redundancy
2. Synthesizes a Standard Model with derived observables and causal narrative
3. Generates severe test commands to guide next discovery iterations
"""

from src.reflection.models import (
    AuditorResult,
    ReflectionResult,
    SevereTestCommand,
    StandardModel,
    TheoristResult,
)

__all__ = [
    "AuditorResult",
    "ReflectionResult",
    "SevereTestCommand",
    "StandardModel",
    "TheoristResult",
]
