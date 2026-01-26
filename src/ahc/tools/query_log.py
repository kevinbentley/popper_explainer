"""Tool for querying the session log and history."""

import json
import logging
from typing import Any

from src.ahc.db.models import JournalEntryType
from src.ahc.db.repo import AHCRepository
from src.ahc.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class QueryLogTool(BaseTool):
    """Tool for searching the session log and history.

    Provides access to journal entries, tool calls, and other
    historical data from the current session.
    """

    def __init__(self, repo: AHCRepository | None = None):
        """Initialize the tool.

        Args:
            repo: Database repository
        """
        self._repo = repo
        self._session_id: int | None = None

    def set_context(self, session_id: int) -> None:
        """Set the current session context.

        Args:
            session_id: Database session ID
        """
        self._session_id = session_id

    @property
    def name(self) -> str:
        return "query_log"

    @property
    def description(self) -> str:
        return """Search your session history and logs.

Use this to:
- Recall previous observations and thoughts
- Review past experiments and their results
- Find specific tool calls and their outcomes
- Track your reasoning process

Query types:
- "journal": Search journal entries (thoughts, observations, hypotheses)
- "tool_calls": Search past tool invocations
- "predictions": Search prediction results
- "evaluations": Search law evaluation results
- "summary": Get a summary of session activity"""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query_type",
                type="string",
                description="Type of log to search",
                required=True,
                enum=["journal", "tool_calls", "predictions", "evaluations", "summary"],
            ),
            ToolParameter(
                name="filter",
                type="object",
                description="Filter criteria (type-specific)",
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum number of results (default: 20)",
                required=False,
                default=20,
            ),
        ]

    def execute(
        self,
        query_type: str,
        filter: dict[str, Any] | None = None,
        limit: int = 20,
        **kwargs,
    ) -> ToolResult:
        """Execute a log query.

        Args:
            query_type: Type of log to search
            filter: Optional filter criteria
            limit: Maximum results

        Returns:
            ToolResult with matching log entries
        """
        if self._repo is None or self._session_id is None:
            return ToolResult.fail("Repository not configured")

        try:
            # Coerce types
            limit = int(limit)

            filter = filter or {}
            limit = min(max(1, limit), 100)

            if query_type == "journal":
                return self._query_journal(filter, limit)
            elif query_type == "tool_calls":
                return self._query_tool_calls(filter, limit)
            elif query_type == "predictions":
                return self._query_predictions(filter, limit)
            elif query_type == "evaluations":
                return self._query_evaluations(filter, limit)
            elif query_type == "summary":
                return self._get_summary()
            else:
                return ToolResult.fail(f"Unknown query type: {query_type}")

        except Exception as e:
            logger.exception(f"Query failed: {query_type}")
            return ToolResult.fail(f"Query failed: {str(e)}")

    def _query_journal(self, filter: dict, limit: int) -> ToolResult:
        """Query journal entries."""
        entry_type = None
        if "type" in filter:
            try:
                entry_type = JournalEntryType(filter["type"])
            except ValueError:
                pass

        entries = self._repo.get_journal_entries(
            self._session_id,
            entry_type=entry_type,
            limit=limit,
        )

        results = []
        for entry in entries:
            results.append({
                "turn": entry.turn_number,
                "type": entry.entry_type.value,
                "content": entry.content,
                "metadata": json.loads(entry.metadata_json) if entry.metadata_json else None,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
            })

        return ToolResult.ok(
            data={
                "count": len(results),
                "entries": results,
            }
        )

    def _query_tool_calls(self, filter: dict, limit: int) -> ToolResult:
        """Query tool call history."""
        tool_name = filter.get("tool_name")

        calls = self._repo.get_tool_calls(
            self._session_id,
            tool_name=tool_name,
            limit=limit,
        )

        results = []
        for call in calls:
            results.append({
                "turn": call.turn_number,
                "tool": call.tool_name,
                "arguments": json.loads(call.arguments_json) if call.arguments_json else None,
                "result": json.loads(call.result_json) if call.result_json else None,
                "error": call.error,
                "duration_ms": call.duration_ms,
            })

        return ToolResult.ok(
            data={
                "count": len(results),
                "calls": results,
            }
        )

    def _query_predictions(self, filter: dict, limit: int) -> ToolResult:
        """Query prediction history."""
        stats = self._repo.get_accuracy_stats(self._session_id)

        # Get recent predictions from tool calls
        calls = self._repo.get_tool_calls(
            self._session_id,
            tool_name="run_prediction",
            limit=limit,
        )

        predictions = []
        for call in calls:
            result = json.loads(call.result_json) if call.result_json else {}
            if result.get("success") and result.get("data", {}).get("prediction"):
                pred = result["data"]["prediction"]
                predictions.append({
                    "turn": call.turn_number,
                    "initial_state": result["data"].get("initial_state"),
                    "predicted": pred.get("predicted"),
                    "actual": pred.get("actual"),
                    "correct": pred.get("correct"),
                })

        return ToolResult.ok(
            data={
                "stats": stats,
                "recent_predictions": predictions,
            }
        )

    def _query_evaluations(self, filter: dict, limit: int) -> ToolResult:
        """Query law evaluation history."""
        status = filter.get("status")

        evaluations = self._repo.get_law_evaluations(
            self._session_id,
            status=status,
            limit=limit,
        )

        results = []
        for eval in evaluations:
            results.append({
                "turn": eval.turn_number,
                "law_id": eval.law_id,
                "status": eval.status,
                "reason_code": eval.reason_code,
                "has_counterexample": eval.counterexample_json is not None,
                "runtime_ms": eval.runtime_ms,
            })

        # Compute summary stats
        summary = {
            "total": len(evaluations),
            "passed": sum(1 for e in evaluations if e.status == "PASS"),
            "failed": sum(1 for e in evaluations if e.status == "FAIL"),
            "unknown": sum(1 for e in evaluations if e.status == "UNKNOWN"),
        }

        return ToolResult.ok(
            data={
                "summary": summary,
                "evaluations": results,
            }
        )

    def _get_summary(self) -> ToolResult:
        """Get overall session summary."""
        # Prediction accuracy
        accuracy_stats = self._repo.get_accuracy_stats(self._session_id)

        # Transition rule completeness
        transition_completeness = self._repo.get_transition_completeness(self._session_id)

        # Theorem counts
        theorems = self._repo.get_theorems(self._session_id)
        theorem_counts = {
            "total": len(theorems),
            "proposed": sum(1 for t in theorems if t.status.value == "proposed"),
            "validated": sum(1 for t in theorems if t.status.value == "validated"),
            "refuted": sum(1 for t in theorems if t.status.value == "refuted"),
        }

        # Law evaluation counts
        evaluations = self._repo.get_law_evaluations(self._session_id)
        eval_counts = {
            "total": len(evaluations),
            "passed": sum(1 for e in evaluations if e.status == "PASS"),
            "failed": sum(1 for e in evaluations if e.status == "FAIL"),
            "unknown": sum(1 for e in evaluations if e.status == "UNKNOWN"),
        }

        # Last turn number
        last_turn = self._repo.get_last_turn_number(self._session_id)

        return ToolResult.ok(
            data={
                "turns_completed": last_turn,
                "prediction_accuracy": accuracy_stats,
                "transition_rules": transition_completeness,
                "theorems": theorem_counts,
                "law_evaluations": eval_counts,
            }
        )
