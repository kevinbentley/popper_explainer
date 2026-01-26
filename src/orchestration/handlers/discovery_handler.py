"""Discovery phase handler.

Wraps the existing LawProposer and Harness to produce
ControlBlock outputs compatible with the orchestration engine.

Supports parallel workers for faster law discovery - multiple
LLM calls run concurrently and results are deduplicated.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.claims.schema import CandidateLaw
from src.db.models import NoveltySnapshotRecord
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
from src.reflection.engine import ReflectionEngine
from src.reflection.persistence import (
    build_standard_model_summary,
    load_latest_standard_model,
    save_reflection_result,
)

if TYPE_CHECKING:
    from src.db.repo import Repository
    from src.proposer.client import GeminiClient, MockGeminiClient

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryPhaseConfig:
    """Configuration for discovery phase handler.

    Attributes:
        num_workers: Number of parallel LLM workers (default 1 = sequential)
        dedupe_by_fingerprint: Whether to deduplicate by law fingerprint
    """

    num_workers: int = 1
    dedupe_by_fingerprint: bool = True


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
    research_log: str | None = None  # LLM's research notes for this iteration


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
        config: DiscoveryPhaseConfig | None = None,
        reflection_engine: ReflectionEngine | None = None,
    ):
        """Initialize discovery handler.

        Args:
            proposer: LawProposer for generating candidate laws
            harness: Harness for evaluating laws
            novelty_tracker: Optional NoveltyTracker for saturation detection
            repo: Optional repository for persistence
            config: Phase configuration (num_workers, etc.)
            reflection_engine: Optional reflection engine for periodic analysis
        """
        self.proposer = proposer
        self.harness = harness
        self.novelty_tracker = novelty_tracker
        self.repo = repo
        self.config = config or DiscoveryPhaseConfig()
        self.reflection_engine = reflection_engine
        self._discovery_iteration_count: int = 0

    @property
    def phase(self) -> Phase:
        return Phase.DISCOVERY

    def run_iteration(self, context: PhaseContext) -> ControlBlock:
        """Execute one discovery iteration.

        If num_workers > 1, runs multiple proposer calls in parallel
        and deduplicates the results before evaluation.

        Args:
            context: PhaseContext with run state

        Returns:
            ControlBlock with phase outputs and recommendations
        """
        # Build memory snapshot (shared across all workers)
        memory = self._build_memory(context)

        # Check for targeted requests from theorem phase
        request = self._build_request(context)

        # Propose laws (parallel if num_workers > 1)
        if self.config.num_workers > 1:
            all_laws, total_rejections, total_redundant, warnings = self._propose_parallel(
                memory, request
            )
        else:
            batch = self.proposer.propose(memory, request)
            all_laws = batch.laws
            total_rejections = len(batch.rejections)
            total_redundant = len(batch.redundant)
            warnings = batch.warnings

        # Evaluate proposed laws
        verdicts: list[tuple[CandidateLaw, LawVerdict]] = []
        for law in all_laws:
            verdict = self.harness.evaluate(law)
            verdicts.append((law, verdict))

            # Track novelty if tracker available
            if self.novelty_tracker:
                self.novelty_tracker.check_novelty(law)

            # Persist if repo available
            if self.repo:
                self._persist_evaluation(context.run_id, law, verdict)

        # Build result
        result = self._build_result_from_parallel(
            all_laws, verdicts, total_rejections, total_redundant, warnings
        )

        # Persist novelty snapshot for ReadinessComputer to pick up
        if self.repo:
            self._persist_novelty_snapshot(result)

        # Reflection engine: periodic auditor/theorist analysis
        self._discovery_iteration_count += 1
        if self._should_trigger_reflection(context):
            reflection_addendum = self._run_reflection(context, result)
            if reflection_addendum and result.research_log:
                result.research_log = result.research_log + "\n" + reflection_addendum
            elif reflection_addendum:
                result.research_log = reflection_addendum

        # Build control block
        return self._build_control_block(result, context)

    def _propose_parallel(
        self,
        memory: DiscoveryMemorySnapshot,
        request: ProposalRequest,
    ) -> tuple[list[CandidateLaw], int, int, list[str]]:
        """Run multiple proposer calls in parallel and deduplicate results.

        Note: LLM logging is disabled during parallel execution because:
        1. SQLite connections can't cross thread boundaries
        2. The proposer's last_exchange state has race conditions

        For detailed LLM logs, use single-worker mode (--num-workers 1).

        Args:
            memory: Discovery memory snapshot
            request: Proposal request

        Returns:
            Tuple of (deduplicated_laws, total_rejections, total_redundant, warnings)
        """
        all_laws: list[CandidateLaw] = []
        total_rejections = 0
        total_redundant = 0
        warnings: list[str] = []
        seen_law_ids: set[str] = set()
        duplicates_removed = 0

        # Save and temporarily disable LLM logger to avoid threading issues
        saved_logger = getattr(self.proposer, '_llm_logger', None)
        if saved_logger:
            self.proposer.set_llm_logger(None)

        def worker_task(worker_id: int):
            """Single worker task."""
            logger.debug(f"Worker {worker_id} starting proposal")
            return self.proposer.propose(memory, request)

        try:
            # Run workers in parallel
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = {
                    executor.submit(worker_task, i): i
                    for i in range(self.config.num_workers)
                }

                for future in as_completed(futures):
                    worker_id = futures[future]
                    try:
                        batch = future.result()
                        logger.debug(
                            f"Worker {worker_id} proposed {len(batch.laws)} laws"
                        )

                        # Add laws, deduplicating by law_id
                        for law in batch.laws:
                            if law.law_id not in seen_law_ids:
                                seen_law_ids.add(law.law_id)
                                all_laws.append(law)
                            else:
                                duplicates_removed += 1

                        total_rejections += len(batch.rejections)
                        total_redundant += len(batch.redundant)
                        warnings.extend(batch.warnings)

                    except Exception as e:
                        logger.error(f"Worker {worker_id} failed: {e}")
                        warnings.append(f"Worker {worker_id} failed: {e}")

        finally:
            # Restore LLM logger
            if saved_logger:
                self.proposer.set_llm_logger(saved_logger)

        if duplicates_removed > 0:
            logger.info(
                f"Parallel discovery: {self.config.num_workers} workers, "
                f"{len(all_laws)} unique laws, {duplicates_removed} duplicates removed"
            )

        return all_laws, total_rejections, total_redundant, warnings

    def _build_result_from_parallel(
        self,
        laws: list[CandidateLaw],
        verdicts: list[tuple[CandidateLaw, LawVerdict]],
        total_rejections: int,
        total_redundant: int,
        warnings: list[str],
    ) -> DiscoveryIterationResult:
        """Build iteration result from parallel worker output.

        Args:
            laws: All proposed laws (deduplicated)
            verdicts: Law/verdict pairs
            total_rejections: Total rejections across all workers
            total_redundant: Total redundant across all workers
            warnings: All warnings

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

        # For parallel execution, get the research log from the proposer
        # (it stores the most recent one from any worker)
        research_log = self.proposer.get_research_log()

        return DiscoveryIterationResult(
            laws_proposed=len(laws),
            laws_passed=passed,
            laws_failed=failed,
            laws_unknown=unknown,
            laws_rejected=total_rejections,
            laws_redundant=total_redundant,
            novelty_stats=novelty_stats,
            is_saturated=is_saturated,
            verdicts=verdicts,
            research_log=research_log,
        )

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

        # Extract previous research_log from the most recent control block.
        # This is the primary persistence mechanism — it survives crashes and
        # resumes because control blocks are stored in the database.
        previous_research_log: str | None = None
        if context.previous_control_blocks:
            # Control blocks are ordered most-recent-first
            for cb in context.previous_control_blocks:
                outputs = cb.phase_outputs or {}
                log = outputs.get("research_log")
                if log and isinstance(log, str):
                    previous_research_log = log
                    break

        if context.repo:
            # Track seen law_ids to avoid duplicates
            seen_accepted: set[str] = set()
            seen_falsified: set[str] = set()

            # Helper to render claim as readable text
            def render_claim(law_dict: dict) -> str:
                claim_text = law_dict.get("claim", "")
                if claim_text == "(see claim_ast)" and law_dict.get("claim_ast"):
                    from src.claims.ast_schema import ast_to_string
                    try:
                        claim_text = ast_to_string(law_dict["claim_ast"])
                    except Exception:
                        claim_text = str(law_dict["claim_ast"])
                return claim_text

            # Get recent PASS laws
            pass_evals = context.repo.list_evaluations(status="PASS", limit=50)
            for ev in pass_evals:
                if ev.law_id in seen_accepted:
                    continue
                seen_accepted.add(ev.law_id)

                law_record = context.repo.get_law(ev.law_id)
                if law_record:
                    law_dict = json.loads(law_record.law_json)
                    accepted_laws.append({
                        "law_id": ev.law_id,
                        "template": law_record.template,
                        "claim": render_claim(law_dict),
                        "observables": law_dict.get("observables", []),
                    })

            # Get recent FAIL laws with counterexamples
            fail_evals = context.repo.list_evaluations(status="FAIL", limit=30)
            for ev in fail_evals:
                if ev.law_id in seen_falsified:
                    continue
                seen_falsified.add(ev.law_id)

                law_record = context.repo.get_law(ev.law_id)
                if law_record:
                    law_dict = json.loads(law_record.law_json)
                    falsified_laws.append({
                        "law_id": ev.law_id,
                        "template": law_record.template,
                        "claim": render_claim(law_dict),
                        "observables": law_dict.get("observables", []),
                    })

                    # Get counterexample with claim context
                    cex = context.repo.get_counterexample_for_evaluation(ev.id)
                    if cex:
                        counterexample_gallery.append({
                            "law_id": ev.law_id,
                            "template": law_record.template,
                            "claim": render_claim(law_dict),
                            "forbidden": law_dict.get("forbidden", ""),
                            "initial_state": cex.initial_state,
                            "t_fail": cex.t_fail,
                            "trajectory_excerpt": cex.trajectory_excerpt_json,
                        })

        # Load standard model for filtering and summary
        standard_model_summary = None
        archived_law_ids: set[str] = set()
        if context.repo:
            sm = load_latest_standard_model(context.repo, context.run_id)
            if sm:
                standard_model_summary = build_standard_model_summary(sm)
                archived_law_ids = set(sm.archived_laws)

        # Filter out archived laws from accepted_laws
        if archived_law_ids:
            accepted_laws = [
                law for law in accepted_laws
                if law.get("law_id") not in archived_law_ids
            ]

        # Load unconsumed severe test commands as priority research directions
        priority_research_directions = None
        if context.repo:
            commands = context.repo.list_unconsumed_severe_test_commands(
                context.run_id, limit=10
            )
            if commands:
                priority_research_directions = [
                    {
                        "command_type": cmd.command_type,
                        "description": cmd.description,
                        "target_law_id": cmd.target_law_id,
                        "priority": cmd.priority,
                    }
                    for cmd in commands
                ]
                # Mark as consumed
                for cmd in commands:
                    if cmd.id is not None:
                        context.repo.mark_severe_test_consumed(cmd.id)

        return DiscoveryMemorySnapshot(
            accepted_laws=accepted_laws,
            falsified_laws=falsified_laws,
            counterexamples=counterexample_gallery[:20],  # Limit size
            unknown_laws=[],
            previous_research_log=previous_research_log,
            standard_model_summary=standard_model_summary,
            priority_research_directions=priority_research_directions,
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
            research_log=batch.research_log,
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

        # Calculate redundancy rate
        total_from_llm = result.laws_proposed + result.laws_redundant
        redundancy_rate = result.laws_redundant / total_from_llm if total_from_llm > 0 else 0.0

        # Determine recommendation - high redundancy is a key saturation signal
        if result.is_saturated:
            recommendation = PhaseRecommendation.ADVANCE
            stop_reason = StopReason.SATURATION
        elif redundancy_rate > 0.7:
            # Most LLM proposals are duplicates - time to move on
            recommendation = PhaseRecommendation.ADVANCE
            stop_reason = StopReason.HIGH_REDUNDANCY
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

        # Build justification - include redundancy in novelty calculation
        # Effective novelty = proposed / (proposed + redundant)
        total_from_llm = result.laws_proposed + result.laws_redundant
        effective_novelty = result.laws_proposed / total_from_llm if total_from_llm > 0 else 1.0
        redundancy_rate = result.laws_redundant / total_from_llm if total_from_llm > 0 else 0.0

        if redundancy_rate > 0.7:
            saturation_msg = "High redundancy - approaching saturation."
        elif redundancy_rate > 0.5:
            saturation_msg = "Moderate redundancy - consider advancing soon."
        elif result.is_saturated:
            saturation_msg = "Saturated - ready to advance."
        else:
            saturation_msg = "Still discovering novel laws."

        justification = (
            f"Proposed {result.laws_proposed} laws ({result.laws_redundant} redundant filtered): "
            f"{result.laws_passed} PASS, {result.laws_failed} FAIL, {result.laws_unknown} UNKNOWN. "
            f"Effective novelty: {effective_novelty:.1%}. "
            f"{saturation_msg}"
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
                "redundancy_rate": redundancy_rate,
                "effective_novelty": effective_novelty,
                "novelty_stats": result.novelty_stats,
                "is_saturated": result.is_saturated,
                "research_log": result.research_log,
            },
        )

    def _compute_readiness_suggestion(
        self,
        result: DiscoveryIterationResult,
    ) -> int:
        """Compute LLM's readiness score suggestion.

        This is advisory - the orchestrator computes objective readiness.

        Factors in:
        - Redundancy rate (high = saturation signal)
        - Pass rate of proposed laws
        - Unknown rate (too many = harness issues)

        Args:
            result: Iteration result

        Returns:
            Readiness score 0-100
        """
        base = 50

        # Boost for saturation
        if result.is_saturated:
            base = 90

        # Calculate effective novelty including redundancy
        total_from_llm = result.laws_proposed + result.laws_redundant
        redundancy_rate = result.laws_redundant / total_from_llm if total_from_llm > 0 else 0.0

        # High redundancy is a strong saturation signal
        if redundancy_rate > 0.8:
            base = max(base, 90)  # Very high - definitely saturated
        elif redundancy_rate > 0.6:
            base = max(base, 80)  # High - approaching saturation
        elif redundancy_rate > 0.4:
            base = max(base, 70)  # Moderate - consider advancing

        # Boost for low novelty rate (from novelty tracker)
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

    def _should_trigger_reflection(self, context: PhaseContext) -> bool:
        """Determine if reflection should run this iteration.

        Conditions:
        - Reflection engine is available
        - Reflection is enabled in config
        - We've hit the interval (every N discovery iterations)
        - No reflection session already exists for this iteration (resume safety)
        """
        if self.reflection_engine is None:
            return False
        if not context.config.enable_reflection:
            return False
        if self._discovery_iteration_count % context.config.reflection_interval != 0:
            return False
        if self._discovery_iteration_count == 0:
            return False

        # Resume safety: check if session already exists for this iteration
        if context.repo:
            existing = context.repo.get_reflection_session(
                context.run_id, context.iteration_index
            )
            if existing is not None:
                logger.info(
                    f"Reflection session already exists for iteration {context.iteration_index}, skipping"
                )
                return False

        return True

    def _run_reflection(
        self,
        context: PhaseContext,
        result: DiscoveryIterationResult,
    ) -> str | None:
        """Run the reflection engine and persist results.

        Args:
            context: Phase context
            result: Current iteration result

        Returns:
            Research log addendum string, or None if reflection failed
        """
        if self.reflection_engine is None or not context.repo:
            return None

        logger.info(
            f"Triggering reflection at discovery iteration {self._discovery_iteration_count}"
        )

        try:
            # Gather inputs from database
            fixed_laws = self._gather_fixed_laws(context)
            graveyard = self._gather_graveyard(context)
            anomalies = self._gather_anomalies(context)
            research_log_entries = self._gather_research_log_entries(context)

            # Load current standard model if exists
            current_model = load_latest_standard_model(
                context.repo, context.run_id
            )

            # Set logger context so LLM transcripts are tagged correctly
            self.reflection_engine.set_logger_context(
                run_id=context.run_id,
                iteration_id=context.iteration_index,
                phase="discovery",
            )

            # Run reflection
            reflection_result = self.reflection_engine.run(
                fixed_laws=fixed_laws,
                graveyard=graveyard,
                anomalies=anomalies,
                research_log_entries=research_log_entries,
                current_standard_model=current_model,
            )

            # Persist results
            save_reflection_result(
                repo=context.repo,
                run_id=context.run_id,
                iteration_index=context.iteration_index,
                result=reflection_result,
                trigger_reason="periodic",
            )

            logger.info(
                f"Reflection complete: {len(reflection_result.auditor_result.conflicts)} conflicts, "
                f"{len(reflection_result.auditor_result.archives)} archives, "
                f"{len(reflection_result.severe_test_commands)} severe tests"
            )

            return reflection_result.research_log_addendum

        except Exception as e:
            logger.error(f"Reflection engine failed: {e}", exc_info=True)
            return None

    def _gather_fixed_laws(self, context: PhaseContext) -> list[dict[str, Any]]:
        """Gather all PASS laws from the database for reflection input."""
        if not context.repo:
            return []

        result = []
        pass_evals = context.repo.list_evaluations(status="PASS", limit=500)
        seen: set[str] = set()
        for ev in pass_evals:
            if ev.law_id in seen:
                continue
            seen.add(ev.law_id)
            law_record = context.repo.get_law(ev.law_id)
            if law_record:
                law_dict = json.loads(law_record.law_json)
                result.append({
                    "law_id": ev.law_id,
                    "template": law_record.template,
                    "claim": law_dict.get("claim", ""),
                    "observables": law_dict.get("observables", []),
                    "forbidden": law_dict.get("forbidden", ""),
                })
        return result

    def _gather_graveyard(self, context: PhaseContext) -> list[dict[str, Any]]:
        """Gather all FAIL laws with counterexamples for reflection input."""
        if not context.repo:
            return []

        result = []
        fail_evals = context.repo.list_evaluations(status="FAIL", limit=500)
        seen: set[str] = set()
        for ev in fail_evals:
            if ev.law_id in seen:
                continue
            seen.add(ev.law_id)
            law_record = context.repo.get_law(ev.law_id)
            if law_record:
                law_dict = json.loads(law_record.law_json)
                entry: dict[str, Any] = {
                    "law_id": ev.law_id,
                    "template": law_record.template,
                    "claim": law_dict.get("claim", ""),
                    "observables": law_dict.get("observables", []),
                    "forbidden": law_dict.get("forbidden", ""),
                }
                cex = context.repo.get_counterexample_for_evaluation(ev.id)
                if cex:
                    entry["counterexample"] = {
                        "initial_state": cex.initial_state,
                        "t_fail": cex.t_fail,
                        "trajectory_excerpt": cex.trajectory_excerpt_json,
                    }
                result.append(entry)
        return result

    def _gather_anomalies(self, context: PhaseContext) -> list[dict[str, Any]]:
        """Gather all UNKNOWN laws for reflection input."""
        if not context.repo:
            return []

        result = []
        unknown_evals = context.repo.list_evaluations(status="UNKNOWN", limit=200)
        seen: set[str] = set()
        for ev in unknown_evals:
            if ev.law_id in seen:
                continue
            seen.add(ev.law_id)
            law_record = context.repo.get_law(ev.law_id)
            if law_record:
                law_dict = json.loads(law_record.law_json)
                result.append({
                    "law_id": ev.law_id,
                    "template": law_record.template,
                    "claim": law_dict.get("claim", ""),
                    "reason_code": ev.reason_code or "",
                })
        return result

    def _gather_research_log_entries(self, context: PhaseContext) -> list[str]:
        """Gather research log entries from all prior control blocks."""
        entries = []
        for cb in context.previous_control_blocks:
            outputs = cb.phase_outputs or {}
            log = outputs.get("research_log")
            if log and isinstance(log, str):
                entries.append(log)
        return entries

    def _persist_novelty_snapshot(self, result: DiscoveryIterationResult) -> None:
        """Persist novelty snapshot for ReadinessComputer to use.

        This bridges the gap between the proposer's redundancy tracking
        and the objective readiness computation.

        Args:
            result: Discovery iteration result with redundancy data
        """
        if not self.repo:
            logger.debug("No repo, skipping novelty snapshot")
            return

        # Calculate metrics from result
        total_from_llm = result.laws_proposed + result.laws_redundant
        logger.debug(f"Novelty snapshot: proposed={result.laws_proposed}, redundant={result.laws_redundant}, total={total_from_llm}")
        if total_from_llm == 0:
            logger.debug("Total from LLM is 0, skipping novelty snapshot")
            return  # Nothing to record

        syntactic_novelty_rate = result.laws_proposed / total_from_llm
        semantic_novelty_rate = syntactic_novelty_rate  # Same for now
        combined_novelty_rate = syntactic_novelty_rate

        # Get novelty tracker stats if available
        total_laws_seen = 0
        unique_syntactic = 0
        unique_semantic = 0
        if self.novelty_tracker:
            stats = self.novelty_tracker.get_window_stats()
            total_laws_seen = stats.total_laws
            unique_syntactic = stats.syntactically_novel
            unique_semantic = stats.semantically_novel

        snapshot = NoveltySnapshotRecord(
            window_size=total_from_llm,
            total_laws_in_window=total_from_llm,
            syntactically_novel_count=result.laws_proposed,
            semantically_novel_count=result.laws_proposed,
            fully_novel_count=result.laws_proposed,
            syntactic_novelty_rate=syntactic_novelty_rate,
            semantic_novelty_rate=semantic_novelty_rate,
            combined_novelty_rate=combined_novelty_rate,
            is_saturated=result.is_saturated or combined_novelty_rate < 0.3,
            total_laws_seen=total_laws_seen,
            unique_syntactic_fingerprints=unique_syntactic,
            unique_semantic_signatures=unique_semantic,
        )
        self.repo.insert_novelty_snapshot(snapshot)

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
