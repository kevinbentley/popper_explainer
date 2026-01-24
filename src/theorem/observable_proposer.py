"""Observable proposal generation from failure clusters.

Uses rule-based mapping from failure buckets to suggested observables.
PHASE-C: Action-based filtering and enhanced LOCAL_PATTERN observables.
"""

import uuid
from dataclasses import dataclass
from typing import Any

from src.theorem.models import FailureCluster, ObservableProposal


@dataclass
class ObservableTemplate:
    """Template for a proposed observable."""

    name: str
    expr: str
    description: str
    priority: str  # 'high', 'medium', 'low'


# PHASE-C observable rules by bucket
# Note: DEFINITION_GAP and EVENTUALITY typically have no observables
# (their actions are SCHEMA_FIX and GATING respectively)
BUCKET_OBSERVABLE_RULES: dict[str, list[ObservableTemplate]] = {
    "LOCAL_PATTERN": [
        ObservableTemplate(
            name="adjacent_pair_><",
            expr="count('><')",
            description="Count of converging pairs (>< pattern)",
            priority="high",
        ),
        ObservableTemplate(
            name="adjacent_pair_<>",
            expr="count('<>')",
            description="Count of diverging pairs (<> pattern)",
            priority="high",
        ),
        ObservableTemplate(
            name="adjacent_pair_X>",
            expr="count('X>')",
            description="Count of X followed by right-mover",
            priority="high",
        ),
        ObservableTemplate(
            name="adjacent_pair_<X",
            expr="count('<X')",
            description="Count of left-mover followed by X",
            priority="high",
        ),
        ObservableTemplate(
            name="alternation_index",
            expr="sum(1 for i in range(len(state)-1) if state[i] != state[i+1])",
            description="Number of direction changes in state string",
            priority="high",
        ),
        ObservableTemplate(
            name="bracketing_count",
            expr="cells where left in {>,X} and right in {<,X}",
            description="Cells bracketed by converging neighbors",
            priority="high",
        ),
        ObservableTemplate(
            name="gap_sizes",
            expr="[len(gap) for gap in re.findall(r'\\.+', state)]",
            description="Distribution of gap sizes between particles",
            priority="medium",
        ),
        ObservableTemplate(
            name="local_density_variance",
            expr="var([count_in_window(state, i, 3) for i in range(len(state))])",
            description="Variance of local density in sliding windows",
            priority="medium",
        ),
    ],
    "COLLISION_TRIGGERS": [
        ObservableTemplate(
            name="collision_cell_neighbors",
            expr="cells adjacent to X",
            description="Cells immediately adjacent to collision sites",
            priority="high",
        ),
        ObservableTemplate(
            name="converging_pair_count",
            expr="count('><')",
            description="Number of converging pairs (will collide)",
            priority="high",
        ),
        ObservableTemplate(
            name="time_to_collision",
            expr="min(distance(p1, p2) for p1, p2 in converging_pairs)",
            description="Minimum time until next collision event",
            priority="high",
        ),
        ObservableTemplate(
            name="collision_precursor_density",
            expr="count('>') * count('<') / len(state)",
            description="Collision potential based on opposing movers",
            priority="medium",
        ),
    ],
    "EVENTUALITY": [
        # EVENTUALITY maps to GATING action, but may still have some observables
        ObservableTemplate(
            name="time_since_last_change",
            expr="t - last_change_time",
            description="Timesteps since state last changed",
            priority="medium",
        ),
        ObservableTemplate(
            name="stability_duration",
            expr="consecutive_timesteps_unchanged(observable_X)",
            description="How long an observable has been stable",
            priority="medium",
        ),
    ],
    "MONOTONICITY": [
        ObservableTemplate(
            name="delta_observable",
            expr="observable(t) - observable(t-1)",
            description="Change in observable between timesteps",
            priority="high",
        ),
        ObservableTemplate(
            name="sign_changes",
            expr="count(sign(delta(t)) != sign(delta(t-1)))",
            description="Number of times derivative sign changed",
            priority="medium",
        ),
        ObservableTemplate(
            name="cumulative_change",
            expr="sum(abs(delta(t)) for t in range(T))",
            description="Total absolute change over time",
            priority="low",
        ),
    ],
    "SYMMETRY": [
        ObservableTemplate(
            name="reflection_difference",
            expr="hamming(state, reflect(state))",
            description="Difference between state and its reflection",
            priority="high",
        ),
        ObservableTemplate(
            name="time_reversal_difference",
            expr="hamming(state, reverse_velocities(state))",
            description="Difference under time reversal",
            priority="high",
        ),
        ObservableTemplate(
            name="shift_invariant_signature",
            expr="canonical_shift(state)",
            description="Shift-invariant state representation",
            priority="medium",
        ),
    ],
    "DEFINITION_GAP": [],  # No observables - action is SCHEMA_FIX
    "OTHER": [],  # No default rules for OTHER
}

# Action to bucket mapping for filtering
ACTION_BUCKET_MAP = {
    "SCHEMA_FIX": ["DEFINITION_GAP"],
    "OBSERVABLE": ["LOCAL_PATTERN", "COLLISION_TRIGGERS", "MONOTONICITY", "SYMMETRY"],
    "GATING": ["EVENTUALITY"],
}


class ObservableProposer:
    """Proposes observables based on failure cluster analysis."""

    def __init__(
        self,
        rules: dict[str, list[ObservableTemplate]] | None = None,
    ):
        self.rules = rules or BUCKET_OBSERVABLE_RULES

    def propose_from_cluster(
        self,
        cluster: FailureCluster,
        context: dict[str, Any] | None = None,
    ) -> list[ObservableProposal]:
        """Propose observables for a single cluster.

        Args:
            cluster: The failure cluster to generate proposals for
            context: Optional additional context (e.g., existing observables)

        Returns:
            List of ObservableProposal objects
        """
        # Filter based on recommended action
        action = cluster.recommended_action

        # SCHEMA_FIX and GATING actions typically don't need observables
        # but we still check if there are any templates
        relevant_buckets = set(cluster.bucket_tags)

        # Collect templates from all relevant buckets
        templates: list[ObservableTemplate] = []
        for bucket_tag in relevant_buckets:
            bucket_templates = self.rules.get(bucket_tag, [])
            templates.extend(bucket_templates)

        if not templates:
            return []

        # Deduplicate templates by name
        seen_names: set[str] = set()
        unique_templates: list[ObservableTemplate] = []
        for t in templates:
            if t.name not in seen_names:
                unique_templates.append(t)
                seen_names.add(t.name)

        proposals = []
        for template in unique_templates:
            proposal = ObservableProposal(
                proposal_id=f"prop_{cluster.cluster_id}_{uuid.uuid4().hex[:8]}",
                cluster_id=cluster.cluster_id,
                observable_name=template.name,
                observable_expr=template.expr,
                rationale=self._build_rationale(cluster, template),
                priority=template.priority,
                action_type=action,
            )
            proposals.append(proposal)

        return proposals

    def propose_from_all_clusters(
        self,
        clusters: list[FailureCluster],
        dedupe: bool = True,
        filter_by_action: bool = True,
    ) -> list[ObservableProposal]:
        """Propose observables from all clusters.

        Args:
            clusters: List of failure clusters
            dedupe: Whether to deduplicate by observable name
            filter_by_action: If True, skip clusters with SCHEMA_FIX action

        Returns:
            List of all ObservableProposal objects
        """
        all_proposals: list[ObservableProposal] = []
        seen_names: set[str] = set()

        for cluster in clusters:
            # Optionally skip SCHEMA_FIX clusters (they need prompt/schema changes, not observables)
            if filter_by_action and cluster.recommended_action == "SCHEMA_FIX":
                continue

            proposals = self.propose_from_cluster(cluster)
            for proposal in proposals:
                if dedupe and proposal.observable_name in seen_names:
                    continue
                all_proposals.append(proposal)
                seen_names.add(proposal.observable_name)

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        all_proposals.sort(key=lambda p: priority_order.get(p.priority, 3))

        return all_proposals

    def _build_rationale(
        self,
        cluster: FailureCluster,
        template: ObservableTemplate,
    ) -> str:
        """Build a rationale string for a proposal."""
        n_theorems = len(cluster.theorem_ids)
        bucket_str = ", ".join(cluster.bucket_tags)

        # Include top keywords if available
        keywords_str = ""
        if cluster.top_keywords:
            kw_list = [kw for kw, _ in cluster.top_keywords[:5]]
            keywords_str = f" Keywords: {', '.join(kw_list)}."

        rationale = (
            f"{template.description}. "
            f"Proposed to address {bucket_str} failures "
            f"affecting {n_theorems} theorem(s). "
            f"Recommended action: {cluster.recommended_action}."
            f"{keywords_str}"
        )

        return rationale

    def get_high_priority_proposals(
        self,
        clusters: list[FailureCluster],
        max_count: int = 5,
    ) -> list[ObservableProposal]:
        """Get only high-priority proposals.

        Args:
            clusters: List of failure clusters
            max_count: Maximum number of proposals to return

        Returns:
            List of high-priority ObservableProposal objects
        """
        all_proposals = self.propose_from_all_clusters(clusters, dedupe=True)
        high_priority = [p for p in all_proposals if p.priority == "high"]
        return high_priority[:max_count]

    def get_proposals_by_action(
        self,
        clusters: list[FailureCluster],
        action: str,
    ) -> list[ObservableProposal]:
        """Get proposals for clusters with a specific recommended action.

        Args:
            clusters: List of failure clusters
            action: Action type to filter by (SCHEMA_FIX, OBSERVABLE, GATING)

        Returns:
            List of ObservableProposal objects
        """
        filtered_clusters = [c for c in clusters if c.recommended_action == action]
        return self.propose_from_all_clusters(
            filtered_clusters,
            dedupe=True,
            filter_by_action=False,  # Don't filter again
        )
