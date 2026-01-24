"""Explanation synthesis subsystem.

Components for generating mechanistic explanations from validated theorems:
- Models: Explanation and Mechanism dataclasses
- Generator: LLM-based explanation synthesis
- Predictor: State prediction from explanations
"""

from src.orchestration.explanation.models import (
    Criticism,
    Explanation,
    ExplanationBatch,
    ExplanationStatus,
    Mechanism,
    MechanismRule,
    MechanismType,
    OpenQuestion,
)
from src.orchestration.explanation.generator import (
    ExplanationGenerator,
    ExplanationGeneratorConfig,
)
from src.orchestration.explanation.predictor import (
    MechanismBasedPredictor,
    LLMBasedPredictor,
    HybridPredictor,
    create_predictor,
)

__all__ = [
    # Models
    "Criticism",
    "Explanation",
    "ExplanationBatch",
    "ExplanationStatus",
    "Mechanism",
    "MechanismRule",
    "MechanismType",
    "OpenQuestion",
    # Generator
    "ExplanationGenerator",
    "ExplanationGeneratorConfig",
    # Predictor
    "MechanismBasedPredictor",
    "LLMBasedPredictor",
    "HybridPredictor",
    "create_predictor",
]
