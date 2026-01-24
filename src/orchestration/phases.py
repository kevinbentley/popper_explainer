"""Phase definitions and handler protocol.

Defines the phases of the Popperian discovery loop and the protocol
that phase handlers must implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.db.repo import Repository
    from src.orchestration.control_block import ControlBlock, PhaseRequest
    from src.orchestration.readiness import ReadinessMetrics


class Phase(str, Enum):
    """Phases of the Popperian discovery loop.

    The phases form a directed graph with the following transitions:

    DISCOVERY <-> THEOREM <-> EXPLANATION <-> PREDICTION -> FINALIZE

    - Forward transitions (ADVANCE) require sustained high readiness
    - Backward transitions (RETREAT) occur when refinement is needed
    - FINALIZE is terminal
    """

    DISCOVERY = "discovery"
    THEOREM = "theorem"
    EXPLANATION = "explanation"
    PREDICTION = "prediction"
    FINALIZE = "finalize"

    @classmethod
    def from_string(cls, value: str) -> Phase:
        """Parse phase from string value."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Unknown phase: {value}")

    def next_phase(self) -> Phase | None:
        """Get the next phase in the forward direction."""
        order = [
            Phase.DISCOVERY,
            Phase.THEOREM,
            Phase.EXPLANATION,
            Phase.PREDICTION,
            Phase.FINALIZE,
        ]
        try:
            idx = order.index(self)
            if idx < len(order) - 1:
                return order[idx + 1]
            return None
        except ValueError:
            return None

    def previous_phase(self) -> Phase | None:
        """Get the previous phase for retreat."""
        order = [
            Phase.DISCOVERY,
            Phase.THEOREM,
            Phase.EXPLANATION,
            Phase.PREDICTION,
            Phase.FINALIZE,
        ]
        try:
            idx = order.index(self)
            if idx > 0:
                return order[idx - 1]
            return None
        except ValueError:
            return None

    def can_transition_to(self, target: Phase) -> bool:
        """Check if direct transition to target phase is valid.

        Valid transitions:
        - Forward to next phase
        - Backward to previous phase (for refinement)
        - Stay in current phase
        """
        if target == self:
            return True  # Stay is always valid

        next_p = self.next_phase()
        prev_p = self.previous_phase()

        return target == next_p or target == prev_p


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestration engine."""

    # Phase transition thresholds (objective readiness must exceed)
    discovery_to_theorem_threshold: float = 85.0
    theorem_to_explanation_threshold: float = 80.0
    explanation_to_prediction_threshold: float = 75.0
    prediction_to_finalize_threshold: float = 95.0

    # Hysteresis: require N consecutive rounds above threshold
    consecutive_rounds_for_advance: int = 3

    # Maximum iterations per phase before forced transition
    max_phase_iterations: dict[Phase, int] = field(
        default_factory=lambda: {
            Phase.DISCOVERY: 100,
            Phase.THEOREM: 50,
            Phase.EXPLANATION: 50,
            Phase.PREDICTION: 100,
            Phase.FINALIZE: 1,
        }
    )

    # Total iteration limit
    max_total_iterations: int = 500

    # Prediction accuracy targets for finalization
    held_out_accuracy_target: float = 0.98
    adversarial_accuracy_target: float = 0.90

    # Plateau detection
    plateau_window: int = 20
    plateau_improvement_threshold: float = 0.01

    # Laws to propose/test per discovery iteration
    laws_per_iteration: int = 10
    tests_per_law: int = 200

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "discovery_to_theorem_threshold": self.discovery_to_theorem_threshold,
            "theorem_to_explanation_threshold": self.theorem_to_explanation_threshold,
            "explanation_to_prediction_threshold": self.explanation_to_prediction_threshold,
            "prediction_to_finalize_threshold": self.prediction_to_finalize_threshold,
            "consecutive_rounds_for_advance": self.consecutive_rounds_for_advance,
            "max_phase_iterations": {p.value: v for p, v in self.max_phase_iterations.items()},
            "max_total_iterations": self.max_total_iterations,
            "held_out_accuracy_target": self.held_out_accuracy_target,
            "adversarial_accuracy_target": self.adversarial_accuracy_target,
            "plateau_window": self.plateau_window,
            "plateau_improvement_threshold": self.plateau_improvement_threshold,
            "laws_per_iteration": self.laws_per_iteration,
            "tests_per_law": self.tests_per_law,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrchestratorConfig:
        """Deserialize from dictionary."""
        max_phase_iter = data.get("max_phase_iterations", {})
        return cls(
            discovery_to_theorem_threshold=data.get(
                "discovery_to_theorem_threshold", 85.0
            ),
            theorem_to_explanation_threshold=data.get(
                "theorem_to_explanation_threshold", 80.0
            ),
            explanation_to_prediction_threshold=data.get(
                "explanation_to_prediction_threshold", 75.0
            ),
            prediction_to_finalize_threshold=data.get(
                "prediction_to_finalize_threshold", 95.0
            ),
            consecutive_rounds_for_advance=data.get("consecutive_rounds_for_advance", 3),
            max_phase_iterations={
                Phase(k): v for k, v in max_phase_iter.items()
            } if max_phase_iter else cls().max_phase_iterations,
            max_total_iterations=data.get("max_total_iterations", 500),
            held_out_accuracy_target=data.get("held_out_accuracy_target", 0.98),
            adversarial_accuracy_target=data.get("adversarial_accuracy_target", 0.90),
            plateau_window=data.get("plateau_window", 20),
            plateau_improvement_threshold=data.get("plateau_improvement_threshold", 0.01),
            laws_per_iteration=data.get("laws_per_iteration", 10),
            tests_per_law=data.get("tests_per_law", 200),
        )


@dataclass
class PhaseContext:
    """Context passed to phase handlers.

    Contains all information a phase handler needs to execute:
    - Run state and configuration
    - Memory snapshots from previous iterations
    - Transition requests from other phases
    - Current readiness metrics
    """

    # Run identification
    run_id: str
    iteration_index: int

    # Database access
    repo: Repository

    # Configuration
    config: OrchestratorConfig

    # Current phase
    current_phase: Phase

    # Memory snapshot for LLM context
    memory_snapshot: dict[str, Any] = field(default_factory=dict)

    # Control blocks from recent iterations
    previous_control_blocks: list[ControlBlock] = field(default_factory=list)

    # Current objective readiness metrics
    readiness_metrics: ReadinessMetrics | None = None

    # Requests from other phases (e.g., targeted falsification)
    transition_requests: list[PhaseRequest] = field(default_factory=list)

    # Phase-specific context (set by orchestrator)
    extra: dict[str, Any] = field(default_factory=dict)

    def get_recent_control_blocks(self, n: int = 5) -> list[ControlBlock]:
        """Get the N most recent control blocks."""
        return self.previous_control_blocks[-n:]


@runtime_checkable
class PhaseHandler(Protocol):
    """Protocol that all phase handlers must implement.

    Each phase handler:
    1. Receives a PhaseContext with run state and configuration
    2. Executes its phase-specific logic (LLM calls, harness evaluation, etc.)
    3. Returns a ControlBlock with outputs and transition recommendations
    """

    @property
    def phase(self) -> Phase:
        """Which phase this handler implements."""
        ...

    def run_iteration(self, context: PhaseContext) -> ControlBlock:
        """Execute one iteration of this phase.

        Args:
            context: PhaseContext with run state, memory, and configuration

        Returns:
            ControlBlock with phase outputs and transition recommendation
        """
        ...

    def can_handle_requests(self, requests: list[PhaseRequest]) -> bool:
        """Check if this handler can process the given requests.

        Args:
            requests: List of PhaseRequests from other phases

        Returns:
            True if this handler can process at least some of the requests
        """
        ...


@dataclass
class IterationResult:
    """Result of a single orchestration iteration."""

    # The control block emitted by the phase handler
    control_block: ControlBlock

    # Objective readiness metrics (computed by orchestrator)
    readiness_metrics: ReadinessMetrics

    # Whether a transition was triggered
    transition_triggered: bool = False

    # Target phase if transitioning
    target_phase: Phase | None = None

    # Reason for transition
    transition_reason: str = ""

    # Whether this is a plateau escape
    is_plateau_escape: bool = False

    # Perturbation applied (if any)
    perturbation_applied: str | None = None


@dataclass
class RunState:
    """Current state of an orchestration run.

    Persisted to database for resumability.
    """

    run_id: str
    current_phase: Phase
    iteration_index: int
    phase_iteration_counts: dict[Phase, int] = field(default_factory=dict)
    status: str = "running"  # 'running', 'completed', 'aborted'

    # Readiness history for hysteresis
    readiness_history: list[float] = field(default_factory=list)

    # Configuration snapshot
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def increment_phase_iteration(self) -> None:
        """Increment iteration count for current phase."""
        current = self.phase_iteration_counts.get(self.current_phase, 0)
        self.phase_iteration_counts[self.current_phase] = current + 1

    def get_phase_iterations(self, phase: Phase | None = None) -> int:
        """Get iteration count for a phase (defaults to current)."""
        phase = phase or self.current_phase
        return self.phase_iteration_counts.get(phase, 0)
