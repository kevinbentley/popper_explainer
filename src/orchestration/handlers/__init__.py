"""Phase handlers for the orchestration engine.

Each handler implements the PhaseHandler protocol for a specific phase
of the discovery loop.

Handlers:
- DiscoveryPhaseHandler: Law proposal and testing
- TheoremPhaseHandler: Theorem synthesis from validated laws
- ExplanationPhaseHandler: Mechanistic explanation synthesis
- PredictionPhaseHandler: Prediction verification
- FinalizePhaseHandler: Report generation
"""

from src.orchestration.handlers.discovery_handler import DiscoveryPhaseHandler
from src.orchestration.handlers.explanation_handler import ExplanationPhaseHandler
from src.orchestration.handlers.finalize_handler import FinalizePhaseHandler
from src.orchestration.handlers.prediction_handler import PredictionPhaseHandler
from src.orchestration.handlers.theorem_handler import TheoremPhaseHandler

__all__ = [
    "DiscoveryPhaseHandler",
    "TheoremPhaseHandler",
    "ExplanationPhaseHandler",
    "PredictionPhaseHandler",
    "FinalizePhaseHandler",
]
