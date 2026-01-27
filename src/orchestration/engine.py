"""Main orchestration engine.

The OrchestratorEngine is the central state machine that governs
the Popperian discovery loop through its phases.

Key responsibilities:
- Initialize or resume runs from database
- Execute phase handlers and collect control blocks
- Compute objective readiness metrics
- Apply transition policy with hysteresis
- Persist all state for resumability
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from src.db.orchestration_models import (
    OrchestrationIterationRecord,
    OrchestrationRunRecord,
    PhaseTransitionRecord,
    ReadinessSnapshotRecord,
)
from src.db.repo import Repository
from src.orchestration.control_block import ControlBlock, PhaseRecommendation, StopReason
from src.orchestration.phases import (
    IterationResult,
    OrchestratorConfig,
    Phase,
    PhaseContext,
    PhaseHandler,
    RunState,
)
from src.orchestration.readiness import ReadinessComputer, ReadinessMetrics
from src.orchestration.transitions import TransitionPolicy

if TYPE_CHECKING:
    pass


@dataclass
class OrchestrationResult:
    """Final result of an orchestration run."""

    run_id: str
    status: str  # 'completed', 'aborted', 'error'
    final_phase: Phase
    total_iterations: int
    phase_iterations: dict[Phase, int] = field(default_factory=dict)
    final_readiness: ReadinessMetrics | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class OrchestratorEngine:
    """Main state machine for scientific discovery loop.

    The engine coordinates:
    1. Phase handlers (discovery, theorem, explanation, prediction, finalize)
    2. Transition policy (hysteresis-based phase changes)
    3. Readiness computation (objective metrics from harness data)
    4. Database persistence (for crash recovery and resumability)

    Usage:
        engine = OrchestratorEngine(repo, config)
        result = engine.run()  # New run
        result = engine.run(run_id="existing_id", resume=True)  # Resume
    """

    def __init__(
        self,
        repo: Repository,
        config: OrchestratorConfig | None = None,
        handlers: dict[Phase, PhaseHandler] | None = None,
    ):
        """Initialize the orchestration engine.

        Args:
            repo: Database repository for persistence
            config: Orchestrator configuration
            handlers: Phase handlers (will use defaults if not provided)
        """
        self.repo = repo
        self.config = config or OrchestratorConfig()
        self.handlers: dict[Phase, PhaseHandler] = handlers or {}

        self.readiness_computer = ReadinessComputer(repo)
        self.transition_policy = TransitionPolicy(config)

        # Current run state
        self._state: RunState | None = None

        # Optional callback for wiring handlers between phases
        self._wire_predictor: callable | None = None

    def register_handler(self, handler: PhaseHandler) -> None:
        """Register a phase handler.

        Args:
            handler: Handler implementing PhaseHandler protocol
        """
        self.handlers[handler.phase] = handler

    def run(
        self,
        run_id: str | None = None,
        resume: bool = False,
        max_iterations: int | None = None,
    ) -> OrchestrationResult:
        """Run the orchestration loop.

        Args:
            run_id: Existing run ID to resume, or None to start new
            resume: If True, resume from last checkpoint
            max_iterations: Override max iterations (for testing)

        Returns:
            OrchestrationResult with final state
        """
        try:
            # Initialize or resume
            if resume and run_id:
                self._state = self._resume_run(run_id)
            else:
                self._state = self._initialize_run(run_id)

            # Override max iterations if specified
            if max_iterations is not None:
                self.config.max_total_iterations = max_iterations

            # Main loop
            while not self._should_stop():
                iteration_result = self._run_iteration()

                # Handle transition
                if iteration_result.transition_triggered:
                    self._execute_transition(
                        iteration_result.target_phase,
                        iteration_result.transition_reason,
                        iteration_result.readiness_metrics,
                    )

            # Finalize
            return self._finalize()

        except Exception as e:
            # Handle errors gracefully
            if self._state:
                self._update_run_status("aborted")
            return OrchestrationResult(
                run_id=self._state.run_id if self._state else "unknown",
                status="error",
                final_phase=self._state.current_phase if self._state else Phase.DISCOVERY,
                total_iterations=self._state.iteration_index if self._state else 0,
                error=str(e),
            )

    def _initialize_run(self, run_id: str | None = None) -> RunState:
        """Initialize a new orchestration run.

        Args:
            run_id: Optional run ID (generates if not provided)

        Returns:
            Initial RunState
        """
        run_id = run_id or f"orch_{uuid.uuid4().hex[:12]}"

        # Create run record
        run_record = OrchestrationRunRecord(
            run_id=run_id,
            status="running",
            current_phase="discovery",
            config_json=json.dumps(self.config.to_dict()),
            total_iterations=0,
        )
        self.repo.insert_orchestration_run(run_record)

        return RunState(
            run_id=run_id,
            current_phase=Phase.DISCOVERY,
            iteration_index=0,
            phase_iteration_counts={Phase.DISCOVERY: 0},
            status="running",
            config=self.config,
        )

    def _resume_run(self, run_id: str) -> RunState:
        """Resume an existing run from database.

        Args:
            run_id: Run ID to resume

        Returns:
            Restored RunState
        """
        run_record = self.repo.get_orchestration_run(run_id)
        if not run_record:
            raise ValueError(f"Run {run_id} not found in database")

        if run_record.status == "completed":
            raise ValueError(f"Run {run_id} is already completed")

        # Get latest iteration
        latest_iter = self.repo.get_latest_iteration(run_id)
        iteration_index = (latest_iter.iteration_index + 1) if latest_iter else 0

        # Restore config
        config_dict = json.loads(run_record.config_json)
        config = OrchestratorConfig.from_dict(config_dict)

        # Build phase iteration counts from history
        phase_counts: dict[Phase, int] = {}
        iterations = self.repo.list_iterations_for_run(run_id, limit=1000)
        for it in iterations:
            phase = Phase.from_string(it.phase)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        # Restore readiness history
        for it in iterations:
            if it.readiness_metrics_json:
                metrics = json.loads(it.readiness_metrics_json)
                phase = Phase.from_string(it.phase)
                score = metrics.get("combined_score", 0.0) * 100
                self.transition_policy.record_readiness(phase, score)

        return RunState(
            run_id=run_id,
            current_phase=Phase.from_string(run_record.current_phase),
            iteration_index=iteration_index,
            phase_iteration_counts=phase_counts,
            status="running",
            config=config,
        )

    def _should_stop(self) -> bool:
        """Check if orchestration should stop.

        Returns:
            True if should stop
        """
        if not self._state:
            return True

        # Check total iterations
        if self._state.iteration_index >= self.config.max_total_iterations:
            return True

        # Check if finalized
        if self._state.current_phase == Phase.FINALIZE:
            # Run one finalize iteration then stop
            if self._state.get_phase_iterations(Phase.FINALIZE) > 0:
                return True

        # Check status
        if self._state.status != "running":
            return True

        return False

    def _run_iteration(self) -> IterationResult:
        """Execute one iteration of the current phase.

        Returns:
            IterationResult with control block and transition decision
        """
        state = self._state
        assert state is not None

        # Create iteration record (status=running)
        iteration_record = OrchestrationIterationRecord(
            run_id=state.run_id,
            iteration_index=state.iteration_index,
            phase=state.current_phase.value,
            status="running",
        )
        iter_id = self.repo.insert_orchestration_iteration(iteration_record)

        try:
            # Build context
            context = self._build_context()

            # Get handler
            handler = self.handlers.get(state.current_phase)
            if not handler:
                # No handler - create a stub control block
                control_block = ControlBlock(
                    readiness_score_suggestion=50,
                    readiness_justification="No handler registered for this phase",
                    phase_recommendation=PhaseRecommendation.STAY,
                    stop_reason=StopReason.CONTINUING,
                    phase_name=state.current_phase.value,
                    iteration_number=state.iteration_index,
                )
            else:
                # Run phase handler
                control_block = handler.run_iteration(context)
                control_block.phase_name = state.current_phase.value
                control_block.iteration_number = state.iteration_index

            # Compute objective readiness
            readiness = self.readiness_computer.compute_for_phase(
                state.run_id,
                state.current_phase,
            )

            # Record readiness for hysteresis
            self.transition_policy.record_readiness(
                state.current_phase,
                readiness.combined_score * 100,
            )
            self.transition_policy.record_iteration(state.current_phase)

            # Check transition
            decision = self.transition_policy.should_transition(
                state.current_phase,
                readiness,
                control_block,
            )

            # Persist iteration
            self._persist_iteration(
                iter_id,
                control_block,
                readiness,
                "completed",
            )

            # Update state
            state.iteration_index += 1
            state.increment_phase_iteration()
            self.repo.update_orchestration_run(
                state.run_id,
                total_iterations=state.iteration_index,
            )

            return IterationResult(
                control_block=control_block,
                readiness_metrics=readiness,
                transition_triggered=decision.should_transition,
                target_phase=decision.target_phase,
                transition_reason=decision.reason,
            )

        except Exception as e:
            # Mark iteration as aborted
            self.repo.update_orchestration_iteration(
                iter_id,
                status="aborted",
                completed_at=datetime.utcnow().isoformat(),
            )
            raise

    def _build_context(self) -> PhaseContext:
        """Build context for phase handler.

        Returns:
            PhaseContext with current state
        """
        state = self._state
        assert state is not None

        # Get recent control blocks
        recent_iterations = self.repo.list_iterations_for_run(
            state.run_id,
            limit=10,
        )
        previous_control_blocks = []
        for it in recent_iterations:
            if it.control_block_json:
                try:
                    cb = ControlBlock.from_json(it.control_block_json)
                    previous_control_blocks.append(cb)
                except Exception as e:
                    logger.warning(
                        "_build_context: failed to parse control block for iteration %s: %s",
                        it.iteration_index, e,
                    )
        logger.debug(
            "_build_context: %d iterations queried, %d control blocks parsed, passing last 5 to handler",
            len(recent_iterations),
            len(previous_control_blocks),
        )

        # Get current readiness
        readiness = self.readiness_computer.compute_for_phase(
            state.run_id,
            state.current_phase,
        )

        # Get pending requests from previous phases
        transition_requests = []
        for cb in previous_control_blocks:
            for req in cb.requests:
                transition_requests.append(req)

        return PhaseContext(
            run_id=state.run_id,
            iteration_index=state.iteration_index,
            repo=self.repo,
            config=self.config,
            current_phase=state.current_phase,
            memory_snapshot={},  # Will be populated by handler
            previous_control_blocks=previous_control_blocks[-5:],
            readiness_metrics=readiness,
            transition_requests=transition_requests,
        )

    def _execute_transition(
        self,
        target_phase: Phase | None,
        reason: str,
        readiness: ReadinessMetrics,
    ) -> None:
        """Execute a phase transition.

        Args:
            target_phase: Target phase
            reason: Reason for transition
            readiness: Current readiness metrics
        """
        if not target_phase or not self._state:
            return

        from_phase = self._state.current_phase

        # Insert transition record
        transition_record = PhaseTransitionRecord(
            run_id=self._state.run_id,
            iteration_id=self._state.iteration_index - 1,  # Previous iteration
            from_phase=from_phase.value,
            to_phase=target_phase.value,
            trigger="readiness_threshold",  # Or extract from decision
            readiness_score=readiness.combined_score * 100,
            evidence_json=json.dumps(readiness.to_dict()),
        )
        self.repo.insert_phase_transition(transition_record)

        # Update state
        self._state.current_phase = target_phase

        # Update run record
        self.repo.update_orchestration_run(
            self._state.run_id,
            current_phase=target_phase.value,
        )

        # Reset hysteresis for source phase
        self.transition_policy.reset_phase_history(from_phase)

        # Wire predictor when transitioning to prediction phase
        if target_phase == Phase.PREDICTION and self._wire_predictor:
            try:
                self._wire_predictor()
            except Exception as e:
                logger.warning(f"Failed to wire predictor: {e}")

    def _persist_iteration(
        self,
        iteration_id: int,
        control_block: ControlBlock,
        readiness: ReadinessMetrics,
        status: str,
    ) -> None:
        """Persist iteration results.

        Args:
            iteration_id: Database ID of iteration
            control_block: Phase output
            readiness: Objective metrics
            status: Iteration status
        """
        state = self._state
        assert state is not None

        # Update iteration record
        self.repo.update_orchestration_iteration(
            iteration_id,
            status=status,
            control_block_json=control_block.to_json(),
            readiness_metrics_json=readiness.to_json(),
            completed_at=datetime.utcnow().isoformat(),
        )

        # Insert readiness snapshot
        snapshot = ReadinessSnapshotRecord(
            run_id=state.run_id,
            iteration_id=iteration_id,
            phase=state.current_phase.value,
            s_pass=readiness.s_pass,
            s_stability=readiness.s_stability,
            s_novel_cex=readiness.s_novel_cex,
            s_harness_health=readiness.s_harness_health,
            s_redundancy=readiness.s_redundancy,
            s_coverage=readiness.s_coverage,
            s_prediction_accuracy=readiness.s_prediction_accuracy,
            s_adversarial_accuracy=readiness.s_adversarial_accuracy,
            s_held_out_accuracy=readiness.s_held_out_accuracy,
            combined_score=readiness.combined_score,
            weights_json=json.dumps(readiness.weights),
            source_counts_json=json.dumps(readiness.source_counts),
        )
        self.repo.insert_readiness_snapshot(snapshot)

    def _update_run_status(self, status: str) -> None:
        """Update run status.

        Args:
            status: New status
        """
        if self._state:
            self._state.status = status
            self.repo.update_orchestration_run(
                self._state.run_id,
                status=status,
                completed_at=datetime.utcnow().isoformat() if status in ("completed", "aborted") else None,
            )

    def _finalize(self) -> OrchestrationResult:
        """Finalize the run.

        Returns:
            OrchestrationResult
        """
        state = self._state
        assert state is not None

        # Update status
        self._update_run_status("completed")

        # Get final readiness
        final_readiness = self.readiness_computer.compute_for_phase(
            state.run_id,
            state.current_phase,
        )

        return OrchestrationResult(
            run_id=state.run_id,
            status="completed",
            final_phase=state.current_phase,
            total_iterations=state.iteration_index,
            phase_iterations=dict(state.phase_iteration_counts),
            final_readiness=final_readiness,
        )
