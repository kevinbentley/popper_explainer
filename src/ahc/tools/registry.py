"""Tool registry for AHC-DS."""

import json
import logging
import time
from datetime import datetime
from typing import Any

from src.ahc.db.models import ToolCallRecord
from src.ahc.db.repo import AHCRepository
from src.ahc.tools.base import BaseTool, ToolResult, ToolError

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry and dispatcher for AHC-DS tools.

    Manages tool registration and dispatches calls with logging.
    """

    def __init__(self, repo: AHCRepository | None = None):
        """Initialize the registry.

        Args:
            repo: Optional database repository for logging tool calls
        """
        self._tools: dict[str, BaseTool] = {}
        self._repo = repo

    def register(self, tool: BaseTool) -> None:
        """Register a tool.

        Args:
            tool: The tool to register

        Raises:
            ValueError: If a tool with this name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool.

        Args:
            name: Name of the tool to unregister
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            The tool, or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all registered tools.

        Returns:
            List of tool schemas compatible with function calling APIs
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def execute(
        self,
        tool_name: str,
        session_id: int | None = None,
        turn_number: int = 0,
        **kwargs,
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Tool name
            session_id: Optional session ID for logging
            turn_number: Turn number for logging
            **kwargs: Tool arguments

        Returns:
            ToolResult from the tool execution

        Raises:
            ToolError: If the tool is not found
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.fail(f"Unknown tool: {tool_name}")

        # Validate arguments
        errors = tool.validate_args(**kwargs)
        if errors:
            return ToolResult.fail(f"Invalid arguments: {'; '.join(errors)}")

        # Execute with timing
        start_time = time.time()
        start_datetime = datetime.now()

        try:
            result = tool.execute(**kwargs)
        except Exception as e:
            logger.exception(f"Tool execution failed: {name}")
            result = ToolResult.fail(str(e))

        duration_ms = int((time.time() - start_time) * 1000)
        result.metadata["duration_ms"] = duration_ms

        # Log to database if repo available
        if self._repo and session_id is not None:
            self._log_tool_call(
                session_id=session_id,
                turn_number=turn_number,
                tool_name=tool_name,
                arguments=kwargs,
                result=result,
                started_at=start_datetime,
                duration_ms=duration_ms,
            )

        return result

    def _log_tool_call(
        self,
        session_id: int,
        turn_number: int,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        started_at: datetime,
        duration_ms: int,
    ) -> None:
        """Log a tool call to the database."""
        try:
            record = ToolCallRecord(
                session_id=session_id,
                turn_number=turn_number,
                tool_name=tool_name,
                arguments_json=json.dumps(arguments),
                result_json=json.dumps(result.to_dict()),
                error=result.error,
                started_at=started_at,
                completed_at=datetime.now(),
                duration_ms=duration_ms,
            )
            self._repo.insert_tool_call(record)
        except Exception as e:
            logger.warning(f"Failed to log tool call: {e}")


def create_default_registry(repo: AHCRepository | None = None) -> ToolRegistry:
    """Create a registry with all default tools registered.

    Args:
        repo: Optional database repository

    Returns:
        Configured ToolRegistry
    """
    from src.ahc.tools.evaluate_laws import EvaluateLawsTool
    from src.ahc.tools.run_prediction import RunPredictionTool
    from src.ahc.tools.request_samples import RequestSamplesTool
    from src.ahc.tools.simulate_states import SimulateStatesTool
    from src.ahc.tools.theorem_tools import (
        StoreTheoremTool,
        RetrieveTheoremsTool,
        EditTheoremTool,
    )
    from src.ahc.tools.query_log import QueryLogTool

    registry = ToolRegistry(repo=repo)

    # Core experimental tools
    registry.register(EvaluateLawsTool(repo=repo))
    registry.register(RunPredictionTool())
    registry.register(RequestSamplesTool())
    registry.register(SimulateStatesTool())

    # Memory tools
    registry.register(StoreTheoremTool(repo=repo))
    registry.register(RetrieveTheoremsTool(repo=repo))
    registry.register(EditTheoremTool(repo=repo))
    registry.register(QueryLogTool(repo=repo))

    return registry


def create_popperian_registry(repo: AHCRepository | None = None) -> ToolRegistry:
    """Create a registry with only Popperian discovery tools.

    This registry enforces proper Popperian methodology:
    - NO direct observation of transitions (no run_prediction)
    - NO trajectory samples (no request_samples)
    - ONLY law proposal and falsification (evaluate_laws)

    The LLM must discover through hypothesis and falsification,
    not through brute-force observation.

    Args:
        repo: Optional database repository

    Returns:
        Configured ToolRegistry with restricted tools
    """
    from src.ahc.tools.evaluate_laws import EvaluateLawsTool
    from src.ahc.tools.theorem_tools import (
        StoreTheoremTool,
        RetrieveTheoremsTool,
        EditTheoremTool,
    )
    from src.ahc.tools.query_log import QueryLogTool

    registry = ToolRegistry(repo=repo)

    # ONLY law evaluation - the core Popperian tool
    registry.register(EvaluateLawsTool(repo=repo))

    # Memory tools for tracking discoveries
    registry.register(StoreTheoremTool(repo=repo))
    registry.register(RetrieveTheoremsTool(repo=repo))
    registry.register(EditTheoremTool(repo=repo))
    registry.register(QueryLogTool(repo=repo))

    return registry
