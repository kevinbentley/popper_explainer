"""Theorem synthesis phase handler.

Wraps the existing TheoremGenerator to produce ControlBlock outputs
compatible with the orchestration engine.

Key responsibilities:
- Synthesize theorems from validated laws
- Identify gaps and refinement targets
- Produce kick-back requests to discovery phase
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.orchestration.control_block import (
    ControlBlock,
    EvidenceReference,
    PhaseRecommendation,
    PhaseRequest,
    ProposedTransition,
    StopReason,
)
from src.orchestration.phases import Phase, PhaseContext, PhaseHandler
from src.theorem.generator import TheoremGenerator
from src.theorem.models import (
    MissingStructureType,
    Theorem,
    TheoremBatch,
    TheoremStatus,
    TypedMissingStructure,
)

if TYPE_CHECKING:
    from src.db.repo import Repository


@dataclass
class RefinementTarget:
    """A specific target for discovery phase refinement.

    Created when theorem synthesis identifies gaps that need
    more laws or targeted falsification.
    """

    target_type: str  # 'missing_observable', 'ambiguous_definition', 'falsify_conjecture', 'explore_template'
    theorem_id: str | None
    description: str
    priority: str = "medium"  # 'high', 'medium', 'low'
    suggested_template: str | None = None  # Template to explore

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type,
            "theorem_id": self.theorem_id,
            "description": self.description,
            "priority": self.priority,
            "suggested_template": self.suggested_template,
        }


@dataclass
class TheoremIterationResult:
    """Result of a theorem synthesis iteration."""

    theorems_generated: int
    theorems_established: int
    theorems_conditional: int
    theorems_conjectural: int
    refinement_targets: list[RefinementTarget]
    needs_refinement: bool
    batch: TheoremBatch
    law_count: int  # Number of laws used as input


class TheoremPhaseHandler:
    """Handler for the theorem synthesis phase.

    This handler:
    1. Builds law snapshots from database
    2. Generates theorems using the LLM
    3. Analyzes theorems for gaps and missing structure
    4. Produces refinement targets for discovery phase
    5. Determines if we should advance or retreat
    """

    def __init__(
        self,
        generator: TheoremGenerator,
        repo: Repository | None = None,
    ):
        """Initialize theorem handler.

        Args:
            generator: TheoremGenerator for synthesizing theorems
            repo: Optional repository for persistence
        """
        self.generator = generator
        self.repo = repo

    @property
    def phase(self) -> Phase:
        return Phase.THEOREM

    def run_iteration(self, context: PhaseContext) -> ControlBlock:
        """Execute one theorem synthesis iteration.

        Args:
            context: PhaseContext with run state

        Returns:
            ControlBlock with phase outputs and recommendations
        """
        repo = context.repo or self.repo
        if not repo:
            return self._error_control_block("No repository available")

        # Build law snapshots from database
        law_snapshots = self.generator.build_law_snapshot(repo)

        if not law_snapshots:
            return self._insufficient_laws_control_block(context)

        # Count PASS laws (needed for theorem generation)
        pass_count = sum(1 for s in law_snapshots if s.status == "PASS")
        if pass_count < 3:
            return self._insufficient_laws_control_block(context, pass_count)

        # Generate theorems
        batch, artifact = self.generator.generate_with_artifact(
            law_snapshots,
            model_name="gemini",
        )

        # Analyze for refinement targets
        refinement_targets = self._identify_refinement_targets(batch.theorems)
        needs_refinement = len(refinement_targets) > 0

        # Persist if repo available
        if repo:
            self._persist_theorems(context.run_id, batch, repo)

        # Build result
        result = self._build_result(batch, refinement_targets, len(law_snapshots))

        # Build control block
        return self._build_control_block(result, context)

    def can_handle_requests(self, requests: list[PhaseRequest]) -> bool:
        """Check if this handler can process requests.

        Theorem phase can handle:
        - clarify_theorem: Request for theorem clarification
        - compress_laws: Request to compress laws into theorems

        Args:
            requests: Requests from other phases

        Returns:
            True if any request is handleable
        """
        handleable_types = {"clarify_theorem", "compress_laws"}
        return any(r.request_type in handleable_types for r in requests)

    def _identify_refinement_targets(
        self,
        theorems: list[Theorem],
    ) -> list[RefinementTarget]:
        """Identify gaps and refinement targets from theorems.

        Analyzes:
        - Missing structure (observables, definitions)
        - Failure modes that suggest needed laws
        - Conjectural theorems that need testing

        Args:
            theorems: Generated theorems

        Returns:
            List of refinement targets for discovery
        """
        targets: list[RefinementTarget] = []

        for theorem in theorems:
            # Check typed missing structure
            for tms in theorem.typed_missing_structure:
                target = self._missing_structure_to_target(theorem, tms)
                if target:
                    targets.append(target)

            # Check failure modes for patterns
            for failure_mode in theorem.failure_modes:
                target = self._failure_mode_to_target(theorem, failure_mode)
                if target:
                    targets.append(target)

            # Conjectural theorems need falsification attempts
            if theorem.status == TheoremStatus.CONJECTURAL:
                targets.append(RefinementTarget(
                    target_type="falsify_conjecture",
                    theorem_id=theorem.theorem_id,
                    description=f"Attempt to falsify conjectural theorem: {theorem.name}",
                    priority="medium",
                ))

        # Dedupe and prioritize
        targets = self._dedupe_targets(targets)

        return targets

    def _missing_structure_to_target(
        self,
        theorem: Theorem,
        tms: TypedMissingStructure,
    ) -> RefinementTarget | None:
        """Convert a typed missing structure to a refinement target.

        Args:
            theorem: Source theorem
            tms: Typed missing structure

        Returns:
            RefinementTarget or None
        """
        if tms.type == MissingStructureType.DEFINITION_MISSING:
            return RefinementTarget(
                target_type="missing_observable",
                theorem_id=theorem.theorem_id,
                description=f"Observable needed: {tms.target}",
                priority="high",
            )

        elif tms.type == MissingStructureType.LOCAL_STRUCTURE_MISSING:
            return RefinementTarget(
                target_type="explore_template",
                theorem_id=theorem.theorem_id,
                description=f"Local structure needed: {tms.target}",
                priority="medium",
                suggested_template="implication_step",
            )

        elif tms.type == MissingStructureType.TEMPORAL_STRUCTURE_MISSING:
            return RefinementTarget(
                target_type="explore_template",
                theorem_id=theorem.theorem_id,
                description=f"Temporal structure needed: {tms.target}",
                priority="medium",
                suggested_template="eventually",
            )

        elif tms.type == MissingStructureType.MECHANISM_MISSING:
            return RefinementTarget(
                target_type="explore_template",
                theorem_id=theorem.theorem_id,
                description=f"Mechanism needed: {tms.target}",
                priority="low",
                suggested_template="implication_state",
            )

        return None

    def _failure_mode_to_target(
        self,
        theorem: Theorem,
        failure_mode: str,
    ) -> RefinementTarget | None:
        """Convert a failure mode to a refinement target.

        Args:
            theorem: Source theorem
            failure_mode: Failure mode description

        Returns:
            RefinementTarget or None
        """
        lower = failure_mode.lower()

        # Collision-related failures
        if any(kw in lower for kw in ["collision", "collide", "impact", "merge"]):
            return RefinementTarget(
                target_type="explore_template",
                theorem_id=theorem.theorem_id,
                description=f"Collision-related failure: {failure_mode}",
                priority="high",
                suggested_template="implication_step",
            )

        # Boundary-related failures
        if any(kw in lower for kw in ["boundary", "wrap", "edge", "border"]):
            return RefinementTarget(
                target_type="explore_template",
                theorem_id=theorem.theorem_id,
                description=f"Boundary-related failure: {failure_mode}",
                priority="medium",
                suggested_template="implication_state",
            )

        # Counting-related failures
        if any(kw in lower for kw in ["count", "number", "total", "sum"]):
            return RefinementTarget(
                target_type="ambiguous_definition",
                theorem_id=theorem.theorem_id,
                description=f"Count-based failure (may need local structure): {failure_mode}",
                priority="medium",
            )

        return None

    def _dedupe_targets(
        self,
        targets: list[RefinementTarget],
    ) -> list[RefinementTarget]:
        """Deduplicate and limit refinement targets.

        Args:
            targets: Raw list of targets

        Returns:
            Deduplicated and prioritized list
        """
        seen: set[str] = set()
        unique: list[RefinementTarget] = []

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        targets.sort(key=lambda t: priority_order.get(t.priority, 2))

        for target in targets:
            # Create a key for deduplication
            key = f"{target.target_type}:{target.description[:50]}"
            if key not in seen:
                seen.add(key)
                unique.append(target)

        # Limit to top 10
        return unique[:10]

    def _build_result(
        self,
        batch: TheoremBatch,
        refinement_targets: list[RefinementTarget],
        law_count: int,
    ) -> TheoremIterationResult:
        """Build iteration result from batch and analysis.

        Args:
            batch: TheoremBatch from generator
            refinement_targets: Identified refinement targets
            law_count: Number of input laws

        Returns:
            TheoremIterationResult
        """
        established = sum(1 for t in batch.theorems if t.status == TheoremStatus.ESTABLISHED)
        conditional = sum(1 for t in batch.theorems if t.status == TheoremStatus.CONDITIONAL)
        conjectural = sum(1 for t in batch.theorems if t.status == TheoremStatus.CONJECTURAL)

        return TheoremIterationResult(
            theorems_generated=len(batch.theorems),
            theorems_established=established,
            theorems_conditional=conditional,
            theorems_conjectural=conjectural,
            refinement_targets=refinement_targets,
            needs_refinement=len(refinement_targets) > 0,
            batch=batch,
            law_count=law_count,
        )

    def _build_control_block(
        self,
        result: TheoremIterationResult,
        context: PhaseContext,
    ) -> ControlBlock:
        """Build control block from iteration result.

        Args:
            result: Theorem iteration result
            context: Phase context

        Returns:
            ControlBlock with recommendations
        """
        # Compute readiness suggestion
        readiness = self._compute_readiness_suggestion(result)

        # Determine recommendation
        if result.needs_refinement and len(result.refinement_targets) >= 3:
            # Many gaps - retreat to discovery
            recommendation = PhaseRecommendation.RETREAT
            stop_reason = StopReason.GAPS_IDENTIFIED
        elif result.theorems_established >= 5 and not result.needs_refinement:
            # Good theorems, no major gaps - advance
            recommendation = PhaseRecommendation.ADVANCE
            stop_reason = StopReason.THEOREMS_STABLE
        elif result.theorems_generated == 0:
            # No theorems generated - need more laws
            recommendation = PhaseRecommendation.RETREAT
            stop_reason = StopReason.NEEDS_MORE_LAWS
        else:
            # Continue refining
            recommendation = PhaseRecommendation.STAY
            stop_reason = StopReason.CONTINUING

        # Build evidence references
        evidence: list[EvidenceReference] = []
        for theorem in result.batch.theorems[:5]:
            evidence.append(EvidenceReference(
                artifact_type="theorem",
                artifact_id=theorem.theorem_id,
                role="supports" if theorem.status == TheoremStatus.ESTABLISHED else "requires",
                note=f"{theorem.status.value}: {theorem.name}",
            ))

        # Build requests for discovery (if retreating)
        requests: list[PhaseRequest] = []
        if recommendation == PhaseRecommendation.RETREAT:
            for target in result.refinement_targets[:5]:
                requests.append(PhaseRequest(
                    request_type=target.target_type,
                    target_id=target.theorem_id,
                    description=target.description,
                    priority=target.priority,
                ))

        # Build justification
        justification = (
            f"Generated {result.theorems_generated} theorems "
            f"({result.theorems_established} established, "
            f"{result.theorems_conditional} conditional, "
            f"{result.theorems_conjectural} conjectural) "
            f"from {result.law_count} laws. "
            f"Refinement targets: {len(result.refinement_targets)}."
        )

        # Build proposed transitions
        proposed_transitions: list[ProposedTransition] = []
        if recommendation == PhaseRecommendation.RETREAT:
            proposed_transitions.append(ProposedTransition(
                target_phase=Phase.DISCOVERY.value,
                reason=f"Need targeted laws for {len(result.refinement_targets)} gaps",
                confidence=0.8,
            ))
        elif recommendation == PhaseRecommendation.ADVANCE:
            proposed_transitions.append(ProposedTransition(
                target_phase=Phase.EXPLANATION.value,
                reason="Theorems stable, ready for mechanistic explanation",
                confidence=0.85,
            ))

        return ControlBlock(
            readiness_score_suggestion=readiness,
            readiness_justification=justification,
            phase_recommendation=recommendation,
            stop_reason=stop_reason,
            evidence=evidence,
            requests=requests,
            proposed_transitions=proposed_transitions,
            phase_outputs={
                "theorems_generated": result.theorems_generated,
                "theorems_established": result.theorems_established,
                "theorems_conditional": result.theorems_conditional,
                "theorems_conjectural": result.theorems_conjectural,
                "refinement_targets_count": len(result.refinement_targets),
                "refinement_targets": [t.to_dict() for t in result.refinement_targets],
                "needs_refinement": result.needs_refinement,
                "law_count": result.law_count,
            },
        )

    def _compute_readiness_suggestion(
        self,
        result: TheoremIterationResult,
    ) -> int:
        """Compute LLM's readiness score suggestion.

        Args:
            result: Iteration result

        Returns:
            Readiness score 0-100
        """
        base = 50

        # Boost for established theorems
        if result.theorems_established >= 5:
            base = 80
        elif result.theorems_established >= 3:
            base = 70

        # Penalty for refinement targets
        if result.needs_refinement:
            penalty = min(len(result.refinement_targets) * 5, 30)
            base = max(base - penalty, 30)

        # Penalty for no theorems
        if result.theorems_generated == 0:
            base = 20

        # Boost for good established/total ratio
        if result.theorems_generated > 0:
            established_ratio = result.theorems_established / result.theorems_generated
            if established_ratio > 0.5:
                base = min(base + 10, 90)

        return base

    def _insufficient_laws_control_block(
        self,
        context: PhaseContext,
        pass_count: int = 0,
    ) -> ControlBlock:
        """Create control block for insufficient laws case.

        Args:
            context: Phase context
            pass_count: Number of PASS laws

        Returns:
            ControlBlock requesting retreat to discovery
        """
        return ControlBlock(
            readiness_score_suggestion=20,
            readiness_justification=f"Only {pass_count} PASS laws - need at least 3 for theorem generation",
            phase_recommendation=PhaseRecommendation.RETREAT,
            stop_reason=StopReason.NEEDS_MORE_LAWS,
            evidence=[],
            requests=[PhaseRequest(
                request_type="explore_template",
                target_id=None,
                description="Need more validated laws for theorem synthesis",
                priority="high",
            )],
            proposed_transitions=[ProposedTransition(
                target_phase=Phase.DISCOVERY.value,
                reason="Insufficient laws for theorem generation",
                confidence=0.95,
            )],
            phase_outputs={
                "theorems_generated": 0,
                "pass_laws_available": pass_count,
                "needs_refinement": True,
            },
        )

    def _error_control_block(self, error: str) -> ControlBlock:
        """Create control block for error case.

        Args:
            error: Error message

        Returns:
            ControlBlock with error
        """
        return ControlBlock(
            readiness_score_suggestion=0,
            readiness_justification=f"Error: {error}",
            phase_recommendation=PhaseRecommendation.STAY,
            stop_reason=StopReason.CONTINUING,
            phase_outputs={"error": error},
        )

    def _persist_theorems(
        self,
        run_id: str,
        batch: TheoremBatch,
        repo: Repository,
    ) -> None:
        """Persist theorems to database.

        Args:
            run_id: Orchestration run ID
            batch: TheoremBatch to persist
            repo: Database repository
        """
        from src.db.models import TheoremRunRecord, TheoremRecord

        # Create theorem run record
        run_record = TheoremRunRecord(
            run_id=f"orch_{run_id}_{batch.prompt_hash[:8]}",
            status="completed",
            config_json=json.dumps({}),
            pass_laws_count=0,  # Could track this
            fail_laws_count=0,
            theorems_generated=len(batch.theorems),
            prompt_hash=batch.prompt_hash,
        )

        try:
            run_db_id = repo.insert_theorem_run(run_record)
        except Exception:
            return  # Silently fail if already exists

        # Persist each theorem
        for theorem in batch.theorems:
            theorem_record = TheoremRecord(
                theorem_run_id=run_db_id,
                theorem_id=theorem.theorem_id,
                name=theorem.name,
                status=theorem.status.value,
                claim=theorem.claim,
                support_json=json.dumps([s.to_dict() for s in theorem.support]),
                failure_modes_json=json.dumps(theorem.failure_modes),
                missing_structure_json=json.dumps(theorem.missing_structure),
                typed_missing_structure_json=json.dumps(
                    [tms.to_dict() for tms in theorem.typed_missing_structure]
                ),
                bucket_tags_json=json.dumps(theorem.bucket_tags),
            )
            try:
                repo.insert_theorem(theorem_record)
            except Exception:
                pass  # Silently fail if already exists
