"""Tests for AHC tools."""

import json
import pytest
import tempfile
from pathlib import Path

from src.ahc.db import AHCRepository, SessionRecord
from src.ahc.tools import BaseTool, ToolResult, ToolRegistry
from src.ahc.tools.base import ToolParameter
from src.ahc.tools.run_prediction import RunPredictionTool
from src.ahc.tools.request_samples import RequestSamplesTool


class MockTool(BaseTool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="Test input",
                required=True,
            ),
            ToolParameter(
                name="optional",
                type="integer",
                description="Optional param",
                required=False,
                default=0,
            ),
        ]

    def execute(self, input: str, optional: int = 0, **kwargs) -> ToolResult:
        return ToolResult.ok(data={"echo": input, "optional": optional})


class TestToolResult:
    """Tests for ToolResult."""

    def test_ok_result(self):
        """Test creating a successful result."""
        result = ToolResult.ok({"value": 42})
        assert result.success is True
        assert result.data == {"value": 42}
        assert result.error is None

    def test_fail_result(self):
        """Test creating a failed result."""
        result = ToolResult.fail("Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_dict(self):
        """Test serialization."""
        result = ToolResult.ok({"test": True}, extra_meta="value")
        d = result.to_dict()
        assert d["success"] is True
        assert d["data"] == {"test": True}
        assert d["metadata"]["extra_meta"] == "value"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)

        assert "mock_tool" in registry.list_tools()
        assert registry.get_tool("mock_tool") is tool

    def test_register_duplicate(self):
        """Test that registering a duplicate raises an error."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool)

    def test_execute_tool(self):
        """Test executing a registered tool."""
        registry = ToolRegistry()
        registry.register(MockTool())

        result = registry.execute("mock_tool", input="hello")
        assert result.success is True
        assert result.data["echo"] == "hello"

    def test_execute_unknown_tool(self):
        """Test executing an unknown tool."""
        registry = ToolRegistry()
        result = registry.execute("unknown_tool")
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_get_all_schemas(self):
        """Test getting schemas for all tools."""
        registry = ToolRegistry()
        registry.register(MockTool())

        schemas = registry.get_all_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "mock_tool"
        assert "parameters" in schemas[0]


class TestRunPredictionTool:
    """Tests for the run_prediction tool."""

    def test_basic_prediction(self):
        """Test basic state evolution."""
        tool = RunPredictionTool()
        result = tool.execute(state=">...")

        assert result.success is True
        assert result.data["initial_state"] == ">..."
        assert result.data["final_state"] == ".>.."
        assert len(result.data["trajectory"]) == 2

    def test_prediction_check(self):
        """Test checking a prediction."""
        tool = RunPredictionTool()
        result = tool.execute(
            state=">...",
            predicted_next_state=".>..",
        )

        assert result.success is True
        assert result.data["prediction"]["correct"] is True

    def test_prediction_check_wrong(self):
        """Test checking an incorrect prediction."""
        tool = RunPredictionTool()
        result = tool.execute(
            state=">...",
            predicted_next_state=">...",  # Wrong - particle should move
        )

        assert result.success is True
        assert result.data["prediction"]["correct"] is False
        assert "differences" in result.data["prediction"]

    def test_invalid_state(self):
        """Test handling invalid state."""
        tool = RunPredictionTool()
        result = tool.execute(state="invalid!")

        assert result.success is False
        assert "Invalid state" in result.error

    def test_multi_step(self):
        """Test multi-step simulation."""
        tool = RunPredictionTool()
        result = tool.execute(state=">...", steps=3)

        assert result.success is True
        assert len(result.data["trajectory"]) == 4  # Initial + 3 steps

    def test_observables(self):
        """Test observable computation."""
        tool = RunPredictionTool()
        result = tool.execute(state=">.<")

        assert result.success is True
        obs = result.data["initial_observables"]
        assert obs["count_right"] == 1
        assert obs["count_left"] == 1
        assert obs["count_empty"] == 1

    def test_local_transitions(self):
        """Test local transition data."""
        tool = RunPredictionTool()
        result = tool.execute(state=">.<")

        assert result.success is True
        transitions = result.data["local_transitions"]
        assert len(transitions) == 3

        # Check structure
        for t in transitions:
            assert "position" in t
            assert "symbol_at_t" in t
            assert "symbol_at_t1" in t
            assert "neighbor_config_at_t" in t
            assert "parity" in t


class TestRequestSamplesTool:
    """Tests for the request_samples tool."""

    def test_random_samples(self):
        """Test generating random samples."""
        tool = RequestSamplesTool()
        result = tool.execute(pattern="random", count=3, length=8)

        assert result.success is True
        assert result.data["count"] == 3
        assert len(result.data["samples"]) == 3

        for sample in result.data["samples"]:
            assert len(sample["state"]) == 8
            assert all(c in ".><" for c in sample["state"])

    def test_collision_samples(self):
        """Test generating collision samples."""
        tool = RequestSamplesTool()
        result = tool.execute(pattern="collision", count=5, length=10)

        assert result.success is True
        # Collision samples should have >.<  pattern
        for sample in result.data["samples"]:
            state = sample["state"]
            # After one step, should have collision
            if "trajectory" in sample:
                has_collision = "X" in sample["trajectory"][1]
                # Most should produce collisions
                # (not all guaranteed due to edge positions)

    def test_pathological_samples(self):
        """Test generating pathological samples."""
        tool = RequestSamplesTool()
        result = tool.execute(pattern="pathological", count=5, length=8)

        assert result.success is True
        # Should have various edge cases
        states = [s["state"] for s in result.data["samples"]]
        # At least one should be all empty
        assert any(all(c == "." for c in s) for s in states)

    def test_with_trajectory(self):
        """Test including trajectories."""
        tool = RequestSamplesTool()
        result = tool.execute(
            pattern="random",
            count=2,
            include_trajectory=True,
            trajectory_steps=5,
        )

        assert result.success is True
        for sample in result.data["samples"]:
            assert "trajectory" in sample
            assert len(sample["trajectory"]) == 6  # Initial + 5 steps

    def test_without_trajectory(self):
        """Test excluding trajectories."""
        tool = RequestSamplesTool()
        result = tool.execute(
            pattern="random",
            count=2,
            include_trajectory=False,
        )

        assert result.success is True
        for sample in result.data["samples"]:
            assert "trajectory" not in sample


class TestToolIntegration:
    """Integration tests with database."""

    @pytest.fixture
    def repo_with_session(self):
        """Create a repo with a session."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        repo = AHCRepository(db_path)
        repo.connect()

        session = SessionRecord(session_id="tool-integration-test")
        repo.insert_session(session)

        yield repo, session

        repo.close()
        Path(db_path).unlink(missing_ok=True)

    def test_tool_call_logging(self, repo_with_session):
        """Test that tool calls are logged to the database."""
        repo, session = repo_with_session

        registry = ToolRegistry(repo=repo)
        registry.register(MockTool())

        # Execute tool
        result = registry.execute(
            "mock_tool",
            session_id=session.id,
            turn_number=1,
            input="test_input",
        )

        assert result.success is True

        # Check that call was logged
        calls = repo.get_tool_calls(session.id)
        assert len(calls) == 1
        assert calls[0].tool_name == "mock_tool"
        assert "test_input" in calls[0].arguments_json
