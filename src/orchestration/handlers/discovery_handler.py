"""Discovery phase handler.

Wraps the existing LawProposer and Harness to produce
ControlBlock outputs compatible with the orchestration engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.claims.schema import CandidateLaw
from src.discovery.novelty import NoveltyTracker
from src.harness.harness import Harness
from src.harness.verdict import LawVerdict
from src.orchestration.control_block import (
    ControlBlock,
    EvidenceReference,
    PhaseRecommendation,
    PhaseRequest,
    StopReason,
)
from src.orchestration.phases import Phase, PhaseContext, PhaseHandler
from src.proposer.memory import DiscoveryMemory, DiscoveryMemorySnapshot
from src.proposer.proposer import LawProposer, ProposalRequest

if TYPE_CHECKING:
    from src.db.repo import Repository


@dataclass
class DiscoveryIterationResult:
    """Result of a discovery iteration."""

    laws_proposed: int
    laws_passed: int
    laws_failed: int
    laws_unknown: int
    laws_rejected: int
    laws_redundant: int
    novelty_stats: dict[str, Any]
    is_saturated: bool
    verdicts: list[tuple[CandidateLaw, LawVerdict]]


class DiscoveryPhaseHandler:
    """Handler for the law discovery phase.

    This handler:
    1. Builds a memory snapshot from database state
    2. Calls LawProposer to generate candidate laws
    3. Evaluates laws via Harness
    4. Tracks novelty and saturation
    5. Produces a ControlBlock with readiness assessment
    """

    def __init__(
        self,
        proposer: LawProposer,
        harness: Harness,
        novelty_tracker: NoveltyTracker | None = None,
        repo: Repository | None = None,
    ):
        """Initialize discovery handler.

        Args:
            proposer: LawProposer for generating candidate laws
            harness: Harness for evaluating laws
            novelty_tracker: Optional NoveltyTracker for saturation detection
            repo: Optional repository for persistence
        """
        self.proposer = proposer
        self.harness = harness
        self.novelty_tracker = novelty_tracker
        self.repo = repo

    @property
    def phase(self) -> Phase:
        return Phase.DISCOVERY

    def run_iteration(self, context: PhaseContext) -> ControlBlock:
        """Execute one discovery iteration.

        Args:
            context: PhaseContext with run state

        Returns:
            ControlBlock with phase outputs and recommendations
        """
        # Build memory snapshot
        memory = self._build_memory(context)

        # Check for targeted requests from theorem phase
        request = self._build_request(context)

        # Propose laws
        batch = self.proposer.propose(memory, request)

        # Evaluate proposed laws
        verdicts: list[tuple[CandidateLaw, LawVerdict]] = []
        for law in batch.laws:
            verdict = self.harness.evaluate(law)
            verdicts.append((law, verdict))

            # Track novelty if tracker available
            if self.novelty_tracker:
                self.novelty_tracker.check_novelty(law)

            # Persist if repo available
            if self.repo:
                self._persist_evaluation(context.run_id, law, verdict)

        # Build result
        result = self._build_result(batch, verdicts)

        # Build control block
        return self._build_control_block(result, context)

    def can_handle_requests(self, requests: list[PhaseRequest]) -> bool:
        """Check if this handler can process requests.

        Discovery can handle:
        - test_law: Test a specific law
        - falsify_theorem: Target falsification of a theorem
        - explore_template: Focus on specific templates

        Args:
            requests: Requests from other phases

        Returns:
            True if any request is handleable
        """
        handleable_types = {"test_law", "falsify_theorem", "explore_template"}
        return any(r.request_type in handleable_types for r in requests)

    def _build_memory(self, context: PhaseContext) -> DiscoveryMemorySnapshot:
        """Build memory snapshot from context and database.

        Args:
            context: Phase context

        Returns:
            DiscoveryMemorySnapshot for proposer
        """
        # Get recent laws from database
        accepted_laws: list[dict[str, Any]] = []
        falsified_laws: list[dict[str, Any]] = []
        counterexample_gallery: list[dict[str, Any]] = []

        if context.repo:
            # Get recent PASS laws
            pass_evals = context.repo.list_evaluations(status="PASS", limit=50)
            for ev in pass_evals:
                law_record = context.repo.get_law(ev.law_id)
                if law_record:
                    law_dict = json.loads(law_record.law_json)
                    accepted_laws.append({
                        "law_id": ev.law_id,
                        "template": law_record.template,
                        "claim": law_dict.get("claim", ""),
                    })

            # Get recent FAIL laws with counterexamples
            fail_evals = context.repo.list_evaluations(status="FAIL", limit=30)
            for ev in fail_evals:
                law_record = context.repo.get_law(ev.law_id)
                if law_record:
                    law_dict = json.loads(law_record.law_json)
                    falsified_laws.append({
                        "law_id": ev.law_id,
                        "template": law_record.template,
                        "claim": law_dict.get("claim", ""),
                    })

                    # Get counterexample
                    cex = context.repo.get_counterexample_for_evaluation(ev.id)
                    if cex:
                        counterexample_gallery.append({
                            "law_id": ev.law_id,
                            "initial_state": cex.initial_state,
                            "t_fail": cex.t_fail,
                        })

        return DiscoveryMemorySnapshot(
            accepted_laws=accepted_laws,
            falsified_laws=falsified_laws,
            counterexamples=counterexample_gallery[:20],  # Limit size
            unknown_laws=[],
        )

    def _build_request(self, context: PhaseContext) -> ProposalRequest:
        """Build proposal request from context.

        Args:
            context: Phase context with transition requests

        Returns:
            ProposalRequest for proposer
        """
        target_templates = None
        exclude_templates = None

        # Check for requests from theorem phase
        for req in context.transition_requests:
            if req.request_type == "explore_template":
                target_templates = target_templates or []
                if req.target_id:
                    target_templates.append(req.target_id)

        return ProposalRequest(
            count=context.config.laws_per_iteration,
            target_templates=target_templates,
            exclude_templates=exclude_templates,
        )

    def _build_result(
        self,
        batch,
        verdicts: list[tuple[CandidateLaw, LawVerdict]],
    ) -> DiscoveryIterationResult:
        """Build iteration result from batch and verdicts.

        Args:
            batch: ProposalBatch from proposer
            verdicts: Law/verdict pairs

        Returns:
            DiscoveryIterationResult
        """
        # Count verdicts by status
        passed = sum(1 for _, v in verdicts if v.status == "PASS")
        failed = sum(1 for _, v in verdicts if v.status == "FAIL")
        unknown = sum(1 for _, v in verdicts if v.status == "UNKNOWN")

        # Get novelty stats
        novelty_stats = {}
        is_saturated = False
        if self.novelty_tracker:
            stats = self.novelty_tracker.get_window_stats()
            novelty_stats = {
                "combined_novelty_rate": stats.combined_novelty_rate,
                "syntactic_novelty_rate": stats.syntactic_novelty_rate,
                "semantic_novelty_rate": stats.semantic_novelty_rate,
                "total_laws": stats.total_laws,
            }
            is_saturated = self.novelty_tracker.is_saturated()

        return DiscoveryIterationResult(
            laws_proposed=len(batch.laws),
            laws_passed=passed,
            laws_failed=failed,
            laws_unknown=unknown,
            laws_rejected=len(batch.rejections),
            laws_redundant=len(batch.redundant),
            novelty_stats=novelty_stats,
            is_saturated=is_saturated,
            verdicts=verdicts,
        )

    def _build_control_block(
        self,
        result: DiscoveryIterationResult,
        context: PhaseContext,
    ) -> ControlBlock:
        """Build control block from iteration result.

        Args:
            result: Discovery iteration result
            context: Phase context

        Returns:
            ControlBlock with recommendations
        """
        # Compute readiness suggestion
        readiness = self._compute_readiness_suggestion(result)

        # Determine recommendation
        if result.is_saturated:
            recommendation = PhaseRecommendation.ADVANCE
            stop_reason = StopReason.SATURATION
        elif result.novelty_stats.get("combined_novelty_rate", 1.0) < 0.2:
            recommendation = PhaseRecommendation.ADVANCE
            stop_reason = StopReason.HIGH_REDUNDANCY
        else:
            recommendation = PhaseRecommendation.STAY
            stop_reason = StopReason.CONTINUING

        # Build evidence references
        evidence: list[EvidenceReference] = []
        for law, verdict in result.verdicts:
            if verdict.status == "PASS":
                evidence.append(EvidenceReference(
                    artifact_type="law",
                    artifact_id=law.law_id,
                    role="supports",
                    note=f"PASS with coverage {verdict.power_metrics.coverage_score:.2f}" if verdict.power_metrics else "PASS",
                ))
            elif verdict.status == "FAIL":
                evidence.append(EvidenceReference(
                    artifact_type="law",
                    artifact_id=law.law_id,
                    role="refutes",
                    note=f"FAIL at t={verdict.counterexample.t_fail}" if verdict.counterexample else "FAIL",
                ))

        # Build justification
        novelty_rate = result.novelty_stats.get("combined_novelty_rate", 1.0)
        justification = (
            f"Proposed {result.laws_proposed} laws: "
            f"{result.laws_passed} PASS, {result.laws_failed} FAIL, {result.laws_unknown} UNKNOWN. "
            f"Novelty rate: {novelty_rate:.1%}. "
            f"{'Saturated - ready to advance.' if result.is_saturated else 'Still discovering novel laws.'}"
        )

        return ControlBlock(
            readiness_score_suggestion=readiness,
            readiness_justification=justification,
            phase_recommendation=recommendation,
            stop_reason=stop_reason,
            evidence=evidence[:10],  # Limit evidence
            requests=[],  # Discovery doesn't generate requests
            proposed_transitions=[],
            phase_outputs={
                "laws_proposed": result.laws_proposed,
                "laws_passed": result.laws_passed,
                "laws_failed": result.laws_failed,
                "laws_unknown": result.laws_unknown,
                "laws_rejected": result.laws_rejected,
                "laws_redundant": result.laws_redundant,
                "novelty_stats": result.novelty_stats,
                "is_saturated": result.is_saturated,
            },
        )

    def _compute_readiness_suggestion(
        self,
        result: DiscoveryIterationResult,
    ) -> int:
        """Compute LLM's readiness score suggestion.

        This is advisory - the orchestrator computes objective readiness.

        Args:
            result: Iteration result

        Returns:
            Readiness score 0-100
        """
        base = 50

        # Boost for saturation
        if result.is_saturated:
            base = 90

        # Boost for low novelty rate
        novelty_rate = result.novelty_stats.get("combined_novelty_rate", 1.0)
        if novelty_rate < 0.1:
            base = max(base, 85)
        elif novelty_rate < 0.3:
            base = max(base, 70)

        # Boost for high pass rate
        if result.laws_proposed > 0:
            pass_rate = result.laws_passed / result.laws_proposed
            if pass_rate > 0.5:
                base = min(base + 10, 95)

        # Penalty for many unknown
        if result.laws_proposed > 0:
            unknown_rate = result.laws_unknown / result.laws_proposed
            if unknown_rate > 0.5:
                base = max(base - 15, 30)

        return base

    def _persist_evaluation(
        self,
        run_id: str,
        law: CandidateLaw,
        verdict: LawVerdict,
    ) -> None:
        """Persist law and evaluation to database.

        Args:
            run_id: Orchestration run ID
            law: Candidate law
            verdict: Evaluation verdict
        """
        if not self.repo:
            return

        # Persist law if not exists
        existing = self.repo.get_law(law.law_id)
        if not existing:
            from src.db.models import LawRecord
            import hashlib

            law_json = json.dumps(law.model_dump())
            law_hash = hashlib.sha256(law_json.encode()).hexdigest()[:16]

            law_record = LawRecord(
                law_id=law.law_id,
                law_hash=law_hash,
                template=law.template.value if hasattr(law.template, 'value') else str(law.template),
                law_json=law_json,
            )
            self.repo.insert_law(law_record)

        # Persist evaluation
        from src.db.models import EvaluationRecord

        eval_record = EvaluationRecord(
            law_id=law.law_id,
            law_hash=existing.law_hash if existing else "",
            status=verdict.status,
            harness_config_hash="",  # TODO: compute from harness config
            sim_hash="",  # TODO: compute from simulator
            cases_attempted=verdict.power_metrics.cases_attempted if verdict.power_metrics else 0,
            cases_used=verdict.power_metrics.cases_used if verdict.power_metrics else 0,
            reason_code=verdict.reason_code.value if verdict.reason_code else None,
            power_metrics_json=json.dumps(verdict.power_metrics.to_dict()) if verdict.power_metrics else None,
            runtime_ms=verdict.runtime_ms,
        )
        self.repo.insert_evaluation(eval_record)
