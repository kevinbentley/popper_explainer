"""Escalation trigger policies for discovery loop integration.

Defines when and how to run escalation during discovery:
- Trigger conditions (minimum laws, template diversity)
- Cadence rules (frequent cheap vs occasional expensive)
- Scope selection (recent laws vs all laws)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.claims.schema import CandidateLaw, Template
from src.harness.escalation.levels import EscalationLevel

if TYPE_CHECKING:
    from src.db.repo import Repository


@dataclass
class EscalationPolicyConfig:
    """Configuration for escalation policy during discovery.

    Status model:
    - Baseline PASS = PROVISIONAL (for most templates)
    - Baseline PASS = ACCEPTED (for quick_promotion_templates only)
    - Escalation_1 PASS = ACCEPTED (promoted)
    - Any FAIL = REVOKED

    Template-based promotion:
    - `invariant` laws get quick promotion (baseline PASS is enough)
    - All other templates stay PROVISIONAL until escalation_1 passes
    - This prevents fragile implications/bounds from being trusted too early

    Attributes:
        min_accepted_laws: Minimum baseline-passed laws before escalation starts (M)
        min_template_families: Minimum distinct template families required
        quick_promotion_templates: Templates that get accepted on baseline PASS alone
        escalation_1_interval: Run escalation_1 every K iterations (3-5)
        escalation_1_window: Only test laws accepted in last W iterations
        escalation_1_priority_templates: Templates to prioritize for escalation_1
        escalation_2_interval: Run escalation_2 every N iterations (10-15)
        novelty_threshold: Trigger escalation_2 if new laws per window below this
        novelty_window: Window size for measuring novelty rate
    """

    # Trigger conditions
    min_accepted_laws: int = 5
    min_template_families: int = 2

    # Template-based promotion rules
    # Only these templates get quick promotion (baseline PASS = accepted)
    # All other templates stay provisional until escalation_1 passes
    quick_promotion_templates: list[str] = None  # type: ignore

    # Escalation_1: cheap, frequent - promotes provisional to accepted
    escalation_1_interval: int = 4  # Every 3-5 iterations
    escalation_1_window: int = 8  # Test laws from last W iterations
    escalation_1_priority_templates: list[str] = None  # type: ignore

    # Escalation_2: expensive, occasional - validates all accepted
    escalation_2_interval: int = 12  # Every 10-15 iterations
    novelty_threshold: float = 0.5  # New accepted laws per iteration
    novelty_window: int = 5  # Window for measuring novelty

    def __post_init__(self):
        # Default quick promotion templates: only invariant
        # Invariants are the most robust - they make a claim about ALL timesteps
        # and are hard to accidentally satisfy with special initial conditions
        if self.quick_promotion_templates is None:
            self.quick_promotion_templates = [
                "invariant",
            ]

        # Default priority templates for escalation_1: implications and bounds
        # These are more likely to be fragile and need early validation
        if self.escalation_1_priority_templates is None:
            self.escalation_1_priority_templates = [
                "implication_step",
                "implication_state",
                "bound",
                "eventually",
                "symmetry_commutation",
                "monotone",
            ]


@dataclass
class EscalationState:
    """Tracks state needed for escalation policy decisions.

    Attributes:
        last_escalation_1_iteration: Last iteration when escalation_1 ran
        last_escalation_2_iteration: Last iteration when escalation_2 ran
        accepted_by_iteration: Maps iteration -> list of law_ids accepted
        trigger_met_iteration: First iteration when trigger conditions were met
    """

    last_escalation_1_iteration: int = 0
    last_escalation_2_iteration: int = 0
    accepted_by_iteration: dict[int, list[str]] = field(default_factory=dict)
    trigger_met_iteration: int | None = None

    def record_accepted(self, iteration: int, law_id: str) -> None:
        """Record that a law was accepted at this iteration."""
        if iteration not in self.accepted_by_iteration:
            self.accepted_by_iteration[iteration] = []
        self.accepted_by_iteration[iteration].append(law_id)

    def get_recent_accepted_law_ids(self, current_iteration: int, window: int) -> list[str]:
        """Get law IDs accepted in the last W iterations."""
        law_ids = []
        for it in range(max(1, current_iteration - window + 1), current_iteration + 1):
            law_ids.extend(self.accepted_by_iteration.get(it, []))
        return law_ids

    def compute_novelty_rate(self, current_iteration: int, window: int) -> float:
        """Compute average new accepted laws per iteration over window."""
        if current_iteration < window:
            return float('inf')  # Not enough data

        total = 0
        for it in range(current_iteration - window + 1, current_iteration + 1):
            total += len(self.accepted_by_iteration.get(it, []))

        return total / window


def check_trigger_conditions(
    accepted_laws: list[CandidateLaw],
    config: EscalationPolicyConfig,
) -> bool:
    """Check if escalation trigger conditions are met.

    Returns True if:
    - At least min_accepted_laws laws are accepted
    - At least min_template_families different templates are represented
    """
    if len(accepted_laws) < config.min_accepted_laws:
        return False

    # Count distinct template families
    templates = {law.template for law in accepted_laws}
    if len(templates) < config.min_template_families:
        return False

    return True


def should_run_escalation_1(
    iteration: int,
    state: EscalationState,
    config: EscalationPolicyConfig,
    accepted_laws: list[CandidateLaw],
) -> bool:
    """Check if escalation_1 should run this iteration.

    Conditions:
    - Trigger conditions are met
    - At least K iterations since last escalation_1
    - There are recent laws to test
    """
    # Check trigger
    if not check_trigger_conditions(accepted_laws, config):
        return False

    # Record when trigger was first met
    if state.trigger_met_iteration is None:
        state.trigger_met_iteration = iteration

    # Check interval
    iterations_since = iteration - state.last_escalation_1_iteration
    if iterations_since < config.escalation_1_interval:
        return False

    # Check if there are recent laws to test
    recent_ids = state.get_recent_accepted_law_ids(iteration, config.escalation_1_window)
    if not recent_ids:
        return False

    return True


def should_run_escalation_2(
    iteration: int,
    state: EscalationState,
    config: EscalationPolicyConfig,
    accepted_laws: list[CandidateLaw],
    force_before_theorem: bool = False,
) -> bool:
    """Check if escalation_2 should run this iteration.

    Conditions (any of):
    - Novelty rate dropped below threshold
    - Scheduled interval reached
    - Force flag set (before theorem mode)
    """
    # Check trigger
    if not check_trigger_conditions(accepted_laws, config):
        return False

    # Force before theorem/explanation mode
    if force_before_theorem:
        return True

    # Check scheduled interval
    iterations_since = iteration - state.last_escalation_2_iteration
    if iterations_since >= config.escalation_2_interval:
        return True

    # Check novelty slowdown
    novelty_rate = state.compute_novelty_rate(iteration, config.novelty_window)
    if novelty_rate < config.novelty_threshold:
        return True

    return False


def get_laws_for_escalation_1(
    iteration: int,
    state: EscalationState,
    config: EscalationPolicyConfig,
    repo: "Repository",
) -> list[str]:
    """Get law IDs to test for escalation_1 (recent provisional laws).

    Returns law IDs that:
    - Were accepted (provisional) in the last W iterations
    - Haven't been tested at escalation_1 level yet
    - Prioritizes implication/bound templates (more fragile)
    """
    recent_ids = state.get_recent_accepted_law_ids(iteration, config.escalation_1_window)

    # Filter to those needing escalation_1 testing
    needs_testing = repo.get_laws_needing_escalation(EscalationLevel.ESCALATION_1.value)
    needs_testing_map = {law.law_id: law for law, _ in needs_testing}

    # Filter to recent laws that need testing
    recent_needing = [lid for lid in recent_ids if lid in needs_testing_map]

    # Sort to prioritize implication/bound templates
    priority_templates = set(config.escalation_1_priority_templates or [])

    def priority_key(law_id: str) -> int:
        law = needs_testing_map.get(law_id)
        if law and law.template in priority_templates:
            return 0  # High priority
        return 1  # Normal priority

    return sorted(recent_needing, key=priority_key)


def is_law_promoted(
    law_id: str,
    repo: "Repository",
    config: EscalationPolicyConfig | None = None,
) -> bool:
    """Check if a law has been promoted to ACCEPTED.

    Promotion rules depend on template:
    - Quick promotion templates (default: invariant): baseline PASS is enough
    - All other templates: require escalation_1 PASS

    A law is promoted if:
    - It's a quick_promotion template AND passed baseline evaluation
    - OR it has a law_retest at escalation_1+ level with flip_type='stable'

    Args:
        law_id: The law to check
        repo: Database repository
        config: Policy config (uses defaults if None)

    Returns:
        True if law is promoted to ACCEPTED status
    """
    if config is None:
        config = EscalationPolicyConfig()

    # Fetch the law to check its template
    law_record = repo.get_law(law_id)
    if not law_record:
        return False

    # Parse the law to get template
    import json
    law_json = json.loads(law_record.law_json)
    template = law_json.get("template", "")

    # Quick promotion templates: baseline PASS is enough
    if template in config.quick_promotion_templates:
        # Check if law has a passing evaluation (latest)
        latest_eval = repo.get_latest_evaluation(law_id)
        if latest_eval and latest_eval.status == "PASS":
            return True
        return False

    # Other templates: require escalation_1 PASS
    retests = repo.get_retests_by_flip_type("stable", limit=1000)
    for retest, law in retests:
        if law.law_id == law_id:
            # Check if this retest was at escalation_1+ level
            run = repo.get_escalation_run(retest.escalation_run_id)
            if run and run.level in (
                EscalationLevel.ESCALATION_1.value,
                EscalationLevel.ESCALATION_2.value,
                EscalationLevel.ESCALATION_3.value,
            ):
                return True
    return False


def get_promotion_status(
    law_ids: list[str],
    repo: "Repository",
    config: EscalationPolicyConfig | None = None,
) -> dict[str, str]:
    """Get promotion status for multiple laws.

    Promotion rules depend on template:
    - Quick promotion templates (default: invariant): baseline PASS → 'accepted'
    - Other templates: require escalation_1 PASS for 'accepted'

    Returns dict mapping law_id to status:
    - 'accepted': Promoted (quick template + baseline OR escalation_1 passed)
    - 'provisional': Passed baseline only but requires escalation
    - 'revoked': Failed at some escalation level

    Args:
        law_ids: Laws to check
        repo: Database repository
        config: Policy config (uses defaults if None)

    Returns:
        Dict mapping law_id to status string
    """
    import json

    if config is None:
        config = EscalationPolicyConfig()

    status_map: dict[str, str] = {}

    # Get all retests
    stable_retests = repo.get_retests_by_flip_type("stable", limit=10000)
    revoked_retests = repo.get_retests_by_flip_type("revoked", limit=10000)

    # Build lookup for escalation_1+ stable retests
    escalation_promoted_ids: set[str] = set()
    for retest, law in stable_retests:
        run = repo.get_escalation_run(retest.escalation_run_id)
        if run and run.level in (
            EscalationLevel.ESCALATION_1.value,
            EscalationLevel.ESCALATION_2.value,
            EscalationLevel.ESCALATION_3.value,
        ):
            escalation_promoted_ids.add(law.law_id)

    # Build lookup for revoked laws
    revoked_ids: set[str] = set()
    for retest, law in revoked_retests:
        revoked_ids.add(law.law_id)

    # Pre-fetch law records for template checking
    law_templates: dict[str, str] = {}
    law_has_baseline_pass: dict[str, bool] = {}

    for law_id in law_ids:
        law_record = repo.get_law(law_id)
        if law_record:
            law_json = json.loads(law_record.law_json)
            law_templates[law_id] = law_json.get("template", "")

            # Check for baseline pass (latest evaluation)
            latest_eval = repo.get_latest_evaluation(law_id)
            law_has_baseline_pass[law_id] = (
                latest_eval is not None and latest_eval.status == "PASS"
            )

    # Classify each law
    for law_id in law_ids:
        if law_id in revoked_ids:
            status_map[law_id] = "revoked"
        elif law_id in escalation_promoted_ids:
            # Explicitly passed escalation → accepted regardless of template
            status_map[law_id] = "accepted"
        else:
            # Check template-based promotion
            template = law_templates.get(law_id, "")
            has_pass = law_has_baseline_pass.get(law_id, False)

            if template in config.quick_promotion_templates and has_pass:
                # Quick promotion template with baseline pass → accepted
                status_map[law_id] = "accepted"
            elif has_pass:
                # Other template with baseline pass → provisional
                status_map[law_id] = "provisional"
            else:
                # No baseline pass → not even provisional
                status_map[law_id] = "unknown"

    return status_map


def get_escalation_decisions(
    iteration: int,
    state: EscalationState,
    config: EscalationPolicyConfig,
    accepted_laws: list[CandidateLaw],
    force_escalation_2: bool = False,
) -> tuple[bool, bool]:
    """Determine which escalations should run this iteration.

    Args:
        iteration: Current iteration number
        state: Escalation tracking state
        config: Policy configuration
        accepted_laws: Currently accepted laws
        force_escalation_2: Force escalation_2 (e.g., before theorem mode)

    Returns:
        Tuple of (run_escalation_1, run_escalation_2)
    """
    run_1 = should_run_escalation_1(iteration, state, config, accepted_laws)
    run_2 = should_run_escalation_2(
        iteration, state, config, accepted_laws, force_before_theorem=force_escalation_2
    )

    return run_1, run_2
