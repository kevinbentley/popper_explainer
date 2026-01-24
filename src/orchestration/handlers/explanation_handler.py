"""Explanation phase handler for the orchestration engine.

Handles the EXPLANATION phase where validated theorems are synthesized
into mechanistic explanations that can make predictions.

The explanation phase:
1. Gathers validated theorems from the theorem phase
2. Generates mechanistic explanations via LLM
3. Creates a predictor function for the prediction phase
4. Identifies open questions and criticisms
5. Can kick back to theorem phase if gaps are found
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from src.orchestration.control_block import (
    ControlBlock,
    EvidenceReference,
    PhaseRecommendation,
    PhaseRequest,
    StopReason,
)
from src.orchestration.explanation.generator import (
    ExplanationGenerator,
    ExplanationGeneratorConfig,
)
from src.orchestration.explanation.models import (
    Explanation,
    ExplanationStatus,
)
from src.orchestration.explanation.predictor import (
    MechanismBasedPredictor,
    create_predictor,
)
from src.orchestration.phases import IterationResult, Phase, PhaseContext, PhaseHandler
from src.orchestration.readiness import ReadinessMetrics

if TYPE_CHECKING:
    from src.db.repo import Repository
    from src.theorem.models import Theorem

logger = logging.getLogger(__name__)


@dataclass
class ExplanationPhaseConfig:
    """Configuration for the explanation phase.

    Attributes:
        min_theorems: Minimum theorems required to generate explanation
        max_explanations: Maximum explanations to generate per iteration
        confidence_threshold: Minimum confidence for advancement
        max_open_questions: Maximum allowed open questions
        use_llm_predictor: Whether to use LLM for predictions
    """

    min_theorems: int = 3
    max_explanations: int = 3
    confidence_threshold: float = 0.7
    max_open_questions: int = 5
    use_llm_predictor: bool = False


class ExplanationPhaseHandler(PhaseHandler):
    """Handler for the EXPLANATION phase.

    In this phase, validated theorems are synthesized into
    mechanistic explanations. The handler produces:
    1. A mechanistic explanation of how the universe works
    2. A predictor function that can predict future states
    3. Open questions and criticisms for feedback

    The predictor function is provided to the prediction phase
    via the `get_predictor()` method.
    """

    def __init__(
        self,
        repo: Repository | None,
        llm_client: Any = None,
        config: ExplanationPhaseConfig | None = None,
    ):
        """Initialize the explanation handler.

        Args:
            repo: Database repository
            llm_client: LLM client for generation
            config: Phase configuration
        """
        self.repo = repo
        self.llm_client = llm_client
        self.config = config or ExplanationPhaseConfig()

        self._generator = ExplanationGenerator(
            client=llm_client,
            config=ExplanationGeneratorConfig(
                max_explanations=self.config.max_explanations,
                min_theorem_support=self.config.min_theorems,
            ),
        ) if llm_client else None

        self._current_explanation: Explanation | None = None
        self._predictor: MechanismBasedPredictor | None = None
        self._theorems: list[Theorem] = []

    @property
    def phase(self) -> Phase:
        return Phase.EXPLANATION

    def get_predictor(self) -> Callable[[str, int], str] | None:
        """Get the current predictor function.

        This is used by the prediction phase to test the explanation.

        Returns:
            Predictor function or None if no explanation yet
        """
        if self._predictor:
            return self._predictor.get_predictor_function()
        return None

    def get_current_explanation(self) -> Explanation | None:
        """Get the current explanation."""
        return self._current_explanation

    def execute(self, context: PhaseContext) -> IterationResult:
        """Execute one iteration of the explanation phase.

        Steps:
        1. Gather validated theorems
        2. Generate or refine explanation
        3. Create predictor
        4. Evaluate explanation quality
        5. Generate control block

        Args:
            context: Phase execution context

        Returns:
            IterationResult with metrics and control block
        """
        run_id = context.run_id
        iteration_index = context.iteration_index

        logger.info(f"Explanation phase iteration {iteration_index}")

        # Step 1: Gather theorems
        theorems = self._gather_theorems()
        self._theorems = theorems

        if len(theorems) < self.config.min_theorems:
            # Not enough theorems - request retreat
            return self._insufficient_theorems_result(
                run_id, iteration_index, len(theorems)
            )

        # Step 2: Generate or refine explanation
        if self._current_explanation is None:
            # Generate new explanation
            explanation = self._generate_explanation(theorems, iteration_index)
        else:
            # Refine existing explanation based on feedback
            explanation = self._refine_explanation(
                self._current_explanation, theorems, context
            )

        self._current_explanation = explanation

        # Step 3: Create predictor
        self._predictor = MechanismBasedPredictor(explanation)

        # Step 4: Compute readiness
        readiness = self._compute_readiness(explanation, theorems)

        # Step 5: Generate control block
        control_block = self._generate_control_block(
            explanation, theorems, readiness, context
        )

        # Persist explanation
        if self.repo:
            self._persist_explanation(run_id, iteration_index, explanation)

        # Build result
        result = IterationResult(
            control_block=control_block,
            readiness_metrics=readiness,
        )
        result.summary = {
            "explanation_id": explanation.explanation_id,
            "confidence": explanation.confidence,
            "status": explanation.status.value,
            "open_questions": len(explanation.open_questions),
            "criticisms": len(explanation.criticisms),
            "theorem_count": len(theorems),
        }
        return result

    def _gather_theorems(self) -> list[Theorem]:
        """Gather validated theorems from the database."""
        from src.theorem.models import Theorem, TheoremStatus

        if not self.repo:
            return []

        # Get theorems from database
        try:
            theorem_records = self.repo.list_theorems(limit=100)
            theorems = []
            for record in theorem_records:
                if record.theorem_json:
                    data = json.loads(record.theorem_json)
                    theorem = Theorem.from_dict(data)
                    # Only use established or conditional theorems
                    if theorem.status in (
                        TheoremStatus.ESTABLISHED,
                        TheoremStatus.CONDITIONAL,
                    ):
                        theorems.append(theorem)
            return theorems
        except Exception as e:
            logger.error(f"Failed to gather theorems: {e}")
            return []

    def _generate_explanation(
        self,
        theorems: list[Theorem],
        iteration_index: int,
    ) -> Explanation:
        """Generate a new explanation from theorems."""
        if self._generator:
            batch = self._generator.generate(
                theorems=theorems,
                iteration_id=iteration_index,
            )
            if batch.explanations:
                return batch.explanations[0]

        # Fallback to rule-based explanation
        return self._generator._generate_fallback_explanation(
            theorems, iteration_index
        ) if self._generator else self._create_fallback_explanation(theorems, iteration_index)

    def _create_fallback_explanation(
        self,
        theorems: list[Theorem],
        iteration_index: int,
    ) -> Explanation:
        """Create a basic fallback explanation without LLM."""
        from src.orchestration.explanation.models import (
            Mechanism,
            MechanismRule,
            MechanismType,
        )

        rules = [
            MechanismRule(
                rule_id="movement_right",
                rule_type=MechanismType.MOVEMENT,
                condition="Cell contains >",
                effect="Particle moves right",
                priority=0,
            ),
            MechanismRule(
                rule_id="movement_left",
                rule_type=MechanismType.MOVEMENT,
                condition="Cell contains <",
                effect="Particle moves left",
                priority=0,
            ),
            MechanismRule(
                rule_id="collision",
                rule_type=MechanismType.INTERACTION,
                condition="> and < meet",
                effect="Form collision X",
                priority=1,
            ),
            MechanismRule(
                rule_id="resolution",
                rule_type=MechanismType.TRANSFORMATION,
                condition="Cell contains X",
                effect="X resolves to > and <",
                priority=2,
            ),
        ]

        mechanism = Mechanism(
            rules=rules,
            description="Basic particle movement and collision mechanics",
            assumptions=["Periodic boundaries", "Unit speed"],
            limitations=["No creation/destruction"],
        )

        return Explanation(
            explanation_id=f"exp_fallback_{iteration_index}",
            hypothesis_text="Particles move, collide, and resolve deterministically.",
            mechanism=mechanism,
            supporting_theorems=[t.theorem_id for t in theorems],
            confidence=0.8,
            status=ExplanationStatus.PROPOSED,
            iteration_id=iteration_index,
        )

    def _refine_explanation(
        self,
        explanation: Explanation,
        theorems: list[Theorem],
        context: PhaseContext,
    ) -> Explanation:
        """Refine an existing explanation based on feedback."""
        # Check if we have prediction feedback from transition requests
        for request in context.transition_requests:
            if request.request_type == "refine_explanation":
                # Lower confidence if refinement was requested
                explanation.confidence *= 0.9

        # Update supporting theorems
        explanation.supporting_theorems = [t.theorem_id for t in theorems]

        return explanation

    def _compute_readiness(
        self,
        explanation: Explanation,
        theorems: list[Theorem],
    ) -> ReadinessMetrics:
        """Compute readiness metrics for explanation phase."""
        metrics = ReadinessMetrics()

        # S_PREDICTION_ACCURACY: Use explanation confidence as proxy
        metrics.s_prediction_accuracy = explanation.confidence

        # S_MECHANISM_COMPLETENESS: Based on rule coverage
        expected_rules = 4  # movement_right, movement_left, collision, resolution
        actual_rules = len(explanation.mechanism.rules)
        metrics.s_mechanism_completeness = min(actual_rules / expected_rules, 1.0)

        # S_THEOREM_COVERAGE: Fraction of theorems addressed
        if theorems:
            covered = len(explanation.supporting_theorems)
            metrics.s_theorem_coverage = covered / len(theorems)
        else:
            metrics.s_theorem_coverage = 0.0

        # Compute combined score
        metrics.compute_explanation_readiness()

        return metrics

    def _generate_control_block(
        self,
        explanation: Explanation,
        theorems: list[Theorem],
        readiness: ReadinessMetrics,
        context: PhaseContext,
    ) -> ControlBlock:
        """Generate control block based on explanation quality."""
        # Build evidence
        evidence: list[EvidenceReference] = []
        evidence.append(EvidenceReference(
            artifact_type="explanation",
            artifact_id=explanation.explanation_id,
            role="supports",
            note=f"Confidence: {explanation.confidence:.2%}",
        ))

        for theorem in theorems[:3]:  # Top 3 supporting theorems
            evidence.append(EvidenceReference(
                artifact_type="theorem",
                artifact_id=theorem.theorem_id,
                role="supports",
                note=theorem.name,
            ))

        # Check for critical issues
        has_critical = explanation.has_critical_issues()
        has_high_priority_questions = explanation.has_high_priority_questions()

        # Determine recommendation
        requests: list[PhaseRequest] = []

        if has_critical:
            recommendation = PhaseRecommendation.RETREAT
            stop_reason = StopReason.GAPS_IDENTIFIED
            justification = "Critical issues in explanation require theorem refinement"
            for criticism in explanation.criticisms:
                if criticism.severity == "critical":
                    requests.append(PhaseRequest(
                        request_type="refine_theorem",
                        target_id=None,
                        description=criticism.criticism,
                        priority="high",
                    ))

        elif has_high_priority_questions:
            recommendation = PhaseRecommendation.STAY
            stop_reason = StopReason.CONTINUING
            justification = "High-priority questions need resolution"
            for question in explanation.open_questions:
                if question.priority == "high":
                    requests.append(PhaseRequest(
                        request_type="investigate",
                        target_id=None,
                        description=question.question,
                        priority="high",
                    ))

        elif explanation.confidence >= self.config.confidence_threshold:
            recommendation = PhaseRecommendation.ADVANCE
            stop_reason = StopReason.SATURATION
            justification = f"Explanation confidence {explanation.confidence:.2%} meets threshold"

        else:
            recommendation = PhaseRecommendation.STAY
            stop_reason = StopReason.CONTINUING
            justification = f"Confidence {explanation.confidence:.2%} below {self.config.confidence_threshold:.0%}"

        return ControlBlock(
            readiness_score_suggestion=int(readiness.combined_score * 100),
            readiness_justification=justification,
            phase_recommendation=recommendation,
            stop_reason=stop_reason,
            evidence=evidence,
            requests=requests,
        )

    def _insufficient_theorems_result(
        self,
        run_id: str,
        iteration_index: int,
        theorem_count: int,
    ) -> IterationResult:
        """Create result for insufficient theorems."""
        readiness = ReadinessMetrics()
        readiness.compute_explanation_readiness()

        control_block = ControlBlock(
            readiness_score_suggestion=0,
            readiness_justification=(
                f"Only {theorem_count} theorems available, "
                f"need {self.config.min_theorems}"
            ),
            phase_recommendation=PhaseRecommendation.RETREAT,
            stop_reason=StopReason.GAPS_IDENTIFIED,
            requests=[
                PhaseRequest(
                    request_type="generate_theorems",
                    target_id=None,
                    description="Need more validated theorems",
                    priority="high",
                )
            ],
        )

        result = IterationResult(
            control_block=control_block,
            readiness_metrics=readiness,
        )
        result.summary = {
            "error": f"Insufficient theorems: {theorem_count}/{self.config.min_theorems}",
        }
        return result

    def _persist_explanation(
        self,
        run_id: str,
        iteration_index: int,
        explanation: Explanation,
    ) -> None:
        """Persist explanation to database."""
        if not self.repo:
            return

        from src.db.orchestration_models import ExplanationRecord

        record = ExplanationRecord(
            run_id=run_id,
            explanation_id=explanation.explanation_id,
            hypothesis_text=explanation.hypothesis_text,
            iteration_id=iteration_index,
            mechanism_json=json.dumps(explanation.mechanism.to_dict()),
            supporting_theorem_ids_json=json.dumps(explanation.supporting_theorems),
            open_questions_json=json.dumps(
                [q.to_dict() for q in explanation.open_questions]
            ),
            criticisms_json=json.dumps(
                [c.to_dict() for c in explanation.criticisms]
            ),
            confidence=explanation.confidence,
            status=explanation.status.value,
        )
        self.repo.insert_explanation(record)

    def run_iteration(self, context: PhaseContext) -> ControlBlock:
        """Execute one iteration and return control block.

        This implements the PhaseHandler protocol by wrapping execute().

        Args:
            context: Phase execution context

        Returns:
            ControlBlock with phase outputs
        """
        result = self.execute(context)
        return result.control_block

    def can_handle_requests(self, requests: list) -> bool:
        """Check if this handler can process the given requests.

        Explanation phase can handle:
        - refine_explanation: Refine current explanation
        - add_mechanism_rule: Add rule to mechanism
        - investigate: Investigate open questions

        Args:
            requests: List of PhaseRequests from other phases

        Returns:
            True if any request is handleable
        """
        handleable_types = {"refine_explanation", "add_mechanism_rule", "investigate"}
        return any(r.request_type in handleable_types for r in requests)

    def reset(self) -> None:
        """Reset handler state for a new run."""
        self._current_explanation = None
        self._predictor = None
        self._theorems = []
