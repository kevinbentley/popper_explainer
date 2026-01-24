"""Prediction verification subsystem.

Components for generating and verifying predictions from explanations:
- PredictionTestSets: Held-out and adversarial test set managers
- Adversarial: Adversarial test case generation
- Verifier: Prediction verification and scoring
"""

from src.orchestration.prediction.adversarial import (
    AdversarialPredictionGenerator,
    AdversarialSearchConfig,
)
from src.orchestration.prediction.test_sets import (
    HeldOutSetManager,
    PredictionTestCase,
    PredictionTestSet,
)
from src.orchestration.prediction.verifier import (
    AccuracyThresholds,
    PredictionResult,
    PredictionVerifier,
    VerificationReport,
)

__all__ = [
    # Test sets
    "HeldOutSetManager",
    "PredictionTestCase",
    "PredictionTestSet",
    # Adversarial
    "AdversarialPredictionGenerator",
    "AdversarialSearchConfig",
    # Verifier
    "PredictionVerifier",
    "PredictionResult",
    "VerificationReport",
    "AccuracyThresholds",
]
