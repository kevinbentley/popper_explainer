"""Law discovery module.

This module handles candidate law generation and novelty tracking.
"""

from src.discovery.novelty import (
    NoveltyResult,
    NoveltyStats,
    NoveltyTracker,
    ProbeSuite,
    SemanticEvaluator,
    SemanticSignature,
    check_law_novelty,
    get_novelty_stats,
    get_novelty_tracker,
    is_discovery_saturated,
    reset_novelty_tracker,
)

__all__ = [
    "NoveltyResult",
    "NoveltyStats",
    "NoveltyTracker",
    "ProbeSuite",
    "SemanticEvaluator",
    "SemanticSignature",
    "check_law_novelty",
    "get_novelty_stats",
    "get_novelty_tracker",
    "is_discovery_saturated",
    "reset_novelty_tracker",
]
