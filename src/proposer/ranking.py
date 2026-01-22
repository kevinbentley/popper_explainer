"""Ranking model for law proposals."""

from dataclasses import dataclass, field
from typing import Any

from src.claims.schema import CandidateLaw, Template
from src.proposer.memory import DiscoveryMemorySnapshot
from src.proposer.redundancy import RedundancyDetector


@dataclass
class RankingFeatures:
    """Features used for ranking a candidate law.

    Attributes:
        risk: How easily falsifiable (strong prohibitions, broad preconditions)
        novelty: Distance from known laws
        discrimination: Likelihood to separate rival mechanisms
        testability: Based on current tester capabilities
        redundancy: Similarity to existing items
        overall_score: Computed overall score
    """

    risk: float = 0.0
    novelty: float = 0.0
    discrimination: float = 0.0
    testability: float = 0.0
    redundancy: float = 0.0
    overall_score: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "risk": self.risk,
            "novelty": self.novelty,
            "discrimination": self.discrimination,
            "testability": self.testability,
            "redundancy": self.redundancy,
            "overall_score": self.overall_score,
        }


@dataclass
class RankingWeights:
    """Weights for ranking formula.

    Formula: score = w_risk*risk + w_novelty*novelty + w_discrimination*discrimination
                   + w_testability*testability - w_redundancy*redundancy
    """

    risk: float = 0.25
    novelty: float = 0.20
    discrimination: float = 0.20
    testability: float = 0.25
    redundancy: float = 0.10


class RankingModel:
    """Ranks candidate laws by expected value for testing.

    Higher scores indicate laws that are more worth testing:
    - More falsifiable (risky)
    - More novel (different from known laws)
    - More discriminating (separates rival theories)
    - More testable (capabilities available)
    - Less redundant (not duplicates)
    """

    def __init__(
        self,
        weights: RankingWeights | None = None,
        redundancy_detector: RedundancyDetector | None = None,
    ):
        """Initialize ranking model.

        Args:
            weights: Ranking weights
            redundancy_detector: Detector for redundancy scoring
        """
        self.weights = weights or RankingWeights()
        self.redundancy_detector = redundancy_detector

    def compute_features(
        self,
        law: CandidateLaw,
        memory: DiscoveryMemorySnapshot | None = None,
    ) -> RankingFeatures:
        """Compute ranking features for a law.

        Args:
            law: Law to rank
            memory: Discovery memory for context

        Returns:
            RankingFeatures with scores
        """
        features = RankingFeatures()

        features.risk = self._compute_risk(law)
        features.novelty = self._compute_novelty(law, memory)
        features.discrimination = self._compute_discrimination(law)
        features.testability = self._compute_testability(law, memory)
        features.redundancy = self._compute_redundancy(law)

        # Compute overall score
        features.overall_score = (
            self.weights.risk * features.risk
            + self.weights.novelty * features.novelty
            + self.weights.discrimination * features.discrimination
            + self.weights.testability * features.testability
            - self.weights.redundancy * features.redundancy
        )

        return features

    def rank(
        self,
        laws: list[CandidateLaw],
        memory: DiscoveryMemorySnapshot | None = None,
    ) -> list[tuple[CandidateLaw, RankingFeatures]]:
        """Rank a list of laws by overall score.

        Args:
            laws: Laws to rank
            memory: Discovery memory for context

        Returns:
            List of (law, features) tuples sorted by score descending
        """
        ranked = []
        for law in laws:
            features = self.compute_features(law, memory)
            ranked.append((law, features))

        # Sort by overall score descending
        ranked.sort(key=lambda x: x[1].overall_score, reverse=True)

        return ranked

    def _compute_risk(self, law: CandidateLaw) -> float:
        """Compute risk score (how falsifiable).

        Higher risk = easier to falsify = better.
        """
        score = 0.5  # Base score

        # Fewer preconditions = broader applicability = riskier
        if not law.preconditions:
            score += 0.2
        elif len(law.preconditions) <= 2:
            score += 0.1

        # Certain templates are riskier
        risky_templates = {
            Template.INVARIANT,  # Easy to find counterexamples
            Template.SYMMETRY_COMMUTATION,  # Clear violation criterion
        }
        if law.template in risky_templates:
            score += 0.2

        # Strong claims (equality rather than bounds) are riskier
        if "==" in law.forbidden or "!=" in law.forbidden:
            score += 0.1

        return min(score, 1.0)

    def _compute_novelty(
        self, law: CandidateLaw, memory: DiscoveryMemorySnapshot | None
    ) -> float:
        """Compute novelty score (distance from known laws).

        Higher = more novel.
        """
        if not memory or not memory.accepted_laws:
            return 0.8  # High novelty if no known laws

        score = 0.8  # Start high

        # Check against accepted laws
        for known in memory.accepted_laws:
            # Same template = less novel
            if known.get("template") == law.template.value:
                score -= 0.1

            # Same observables = less novel
            known_obs = {o.get("name") for o in known.get("observables", [])}
            law_obs = {o.name for o in law.observables}
            if known_obs & law_obs:
                score -= 0.1

        return max(score, 0.0)

    def _compute_discrimination(self, law: CandidateLaw) -> float:
        """Compute discrimination score (separates rival theories).

        Higher = more discriminating.
        """
        score = 0.5  # Base score

        # Laws with explicit distinguishes_from are more discriminating
        # (We don't have this in our schema yet, but could be added)

        # Symmetry laws are often discriminating
        if law.template == Template.SYMMETRY_COMMUTATION:
            score += 0.3

        # Laws with specific transforms are discriminating
        if law.transform:
            score += 0.1

        # Implication laws can discriminate between mechanisms
        if law.template in (Template.IMPLICATION_STEP, Template.IMPLICATION_STATE):
            score += 0.2

        return min(score, 1.0)

    def _compute_testability(
        self, law: CandidateLaw, memory: DiscoveryMemorySnapshot | None
    ) -> float:
        """Compute testability score (capabilities available).

        Higher = more testable.
        """
        # Start with full testability
        score = 1.0

        # Missing capabilities reduce testability
        caps = law.capability_requirements
        if caps.missing_observables:
            score -= 0.3 * len(caps.missing_observables)
        if caps.missing_transforms:
            score -= 0.3 * len(caps.missing_transforms)
        if caps.missing_generators:
            score -= 0.2 * len(caps.missing_generators)

        # Check if proposed tests are available
        if memory and memory.capabilities:
            available_generators = memory.capabilities.get("generator_families", [])
            for test in law.proposed_tests:
                if test.family not in available_generators:
                    score -= 0.1

        return max(score, 0.0)

    def _compute_redundancy(self, law: CandidateLaw) -> float:
        """Compute redundancy score.

        Higher = more redundant = worse.
        """
        if not self.redundancy_detector:
            return 0.0

        match = self.redundancy_detector.check(law)
        if match:
            return match.similarity

        return 0.0
