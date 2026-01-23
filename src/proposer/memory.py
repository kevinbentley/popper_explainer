"""Discovery memory for maintaining state across law proposal iterations."""

from dataclasses import dataclass, field
from typing import Any

from src.claims.schema import CandidateLaw
from src.harness.verdict import LawVerdict, Counterexample


@dataclass
class DiscoveryMemorySnapshot:
    """Snapshot of discovery state for prompting.

    Attributes:
        accepted_laws: Laws that passed testing
        falsified_laws: Laws that failed with counterexamples
        unknown_laws: Laws with unknown status and reason codes
        counterexamples: Gallery of counterexamples
        capabilities: Current tester capabilities
    """

    accepted_laws: list[dict[str, Any]] = field(default_factory=list)
    falsified_laws: list[dict[str, Any]] = field(default_factory=list)
    unknown_laws: list[dict[str, Any]] = field(default_factory=list)
    counterexamples: list[dict[str, Any]] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "accepted_laws": self.accepted_laws,
            "falsified_laws": self.falsified_laws,
            "unknown_laws": self.unknown_laws,
            "counterexamples": self.counterexamples,
            "capabilities": self.capabilities,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveryMemorySnapshot":
        """Create from dictionary."""
        return cls(
            accepted_laws=data.get("accepted_laws", []),
            falsified_laws=data.get("falsified_laws", []),
            unknown_laws=data.get("unknown_laws", []),
            counterexamples=data.get("counterexamples", []),
            capabilities=data.get("capabilities", {}),
        )


class DiscoveryMemory:
    """Working memory for law discovery.

    Maintains history of evaluations and provides snapshots for prompting.
    """

    def __init__(
        self,
        max_accepted: int = 30,
        max_falsified: int = 20,
        max_unknown: int = 20,
        max_counterexamples: int = 20,
    ):
        """Initialize discovery memory.

        Args:
            max_accepted: Maximum accepted laws to keep
            max_falsified: Maximum falsified laws to keep
            max_unknown: Maximum unknown laws to keep
            max_counterexamples: Maximum counterexamples to keep
        """
        self.max_accepted = max_accepted
        self.max_falsified = max_falsified
        self.max_unknown = max_unknown
        self.max_counterexamples = max_counterexamples

        self._accepted: list[tuple[CandidateLaw, LawVerdict]] = []
        self._falsified: list[tuple[CandidateLaw, LawVerdict]] = []
        self._unknown: list[tuple[CandidateLaw, LawVerdict]] = []
        self._counterexamples: list[Counterexample] = []
        self._capabilities: dict[str, Any] = {}

    def record_evaluation(self, law: CandidateLaw, verdict: LawVerdict) -> None:
        """Record a law evaluation result.

        Args:
            law: The evaluated law
            verdict: The verdict
        """
        if verdict.status == "PASS":
            self._accepted.append((law, verdict))
            # Keep most recent, trim oldest
            if len(self._accepted) > self.max_accepted:
                self._accepted = self._accepted[-self.max_accepted :]

        elif verdict.status == "FAIL":
            self._falsified.append((law, verdict))
            if len(self._falsified) > self.max_falsified:
                self._falsified = self._falsified[-self.max_falsified :]

            # Also record counterexample
            if verdict.counterexample:
                self._counterexamples.append(verdict.counterexample)
                if len(self._counterexamples) > self.max_counterexamples:
                    self._counterexamples = self._counterexamples[-self.max_counterexamples :]

        else:  # UNKNOWN
            self._unknown.append((law, verdict))
            if len(self._unknown) > self.max_unknown:
                self._unknown = self._unknown[-self.max_unknown :]

    def set_capabilities(self, capabilities: dict[str, Any]) -> None:
        """Update current tester capabilities.

        Args:
            capabilities: Capabilities dictionary
        """
        self._capabilities = capabilities

    def get_snapshot(self) -> DiscoveryMemorySnapshot:
        """Get a snapshot of current memory state.

        Returns:
            DiscoveryMemorySnapshot for prompting
        """
        return DiscoveryMemorySnapshot(
            accepted_laws=self._format_accepted_laws(),
            falsified_laws=self._format_falsified_laws(),
            unknown_laws=self._format_unknown_laws(),
            counterexamples=self._format_counterexamples(),
            capabilities=self._capabilities,
        )

    def _format_accepted_laws(self) -> list[dict[str, Any]]:
        """Format accepted laws for snapshot."""
        result = []
        for law, verdict in self._accepted:
            result.append({
                "law_id": law.law_id,
                "template": law.template.value,
                "claim": law.claim,
                "preconditions": [
                    {"lhs": p.lhs, "op": p.op.value, "rhs": p.rhs}
                    for p in law.preconditions
                ],
                "observables": [
                    {"name": o.name, "expr": o.expr}
                    for o in law.observables
                ],
                "cases_used": verdict.power_metrics.cases_used,
                "coverage_score": verdict.power_metrics.coverage_score,
            })
        return result

    def _format_falsified_laws(self) -> list[dict[str, Any]]:
        """Format falsified laws for snapshot."""
        result = []
        for law, verdict in self._falsified:
            item = {
                "law_id": law.law_id,
                "template": law.template.value,
                "claim": law.claim,
                "preconditions": [
                    {"lhs": p.lhs, "op": p.op.value, "rhs": p.rhs}
                    for p in law.preconditions
                ],
                "observables": [
                    {"name": o.name, "expr": o.expr}
                    for o in law.observables
                ],
            }
            if verdict.counterexample:
                item["counterexample"] = {
                    "initial_state": verdict.counterexample.initial_state,
                    "t_fail": verdict.counterexample.t_fail,
                }
            result.append(item)
        return result

    def _format_unknown_laws(self) -> list[dict[str, Any]]:
        """Format unknown laws for snapshot."""
        result = []
        for law, verdict in self._unknown:
            result.append({
                "law_id": law.law_id,
                "template": law.template.value,
                "claim": law.claim,
                "reason_code": verdict.reason_code.value if verdict.reason_code else None,
                "notes": verdict.notes,
            })
        return result

    def _format_counterexamples(self) -> list[dict[str, Any]]:
        """Format counterexamples for snapshot."""
        result = []
        for cx in self._counterexamples:
            result.append({
                "initial_state": cx.initial_state,
                "t_fail": cx.t_fail,
                "trajectory_excerpt": cx.trajectory_excerpt,
                "observables_at_fail": cx.observables_at_fail,
            })
        return result

    def clear(self) -> None:
        """Clear all memory."""
        self._accepted = []
        self._falsified = []
        self._unknown = []
        self._counterexamples = []

    @property
    def stats(self) -> dict[str, int]:
        """Get memory statistics."""
        return {
            "accepted": len(self._accepted),
            "falsified": len(self._falsified),
            "unknown": len(self._unknown),
            "counterexamples": len(self._counterexamples),
        }
