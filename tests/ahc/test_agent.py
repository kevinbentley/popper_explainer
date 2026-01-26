"""Tests for AHC agent components."""

import json
import pytest
import tempfile
from pathlib import Path

from src.ahc.db import AHCRepository, SessionRecord, SessionStatus
from src.ahc.agent import AgentLoop, AgentConfig, JournalManager, TerminationChecker
from src.ahc.agent.termination import TerminationStatus
from src.ahc.llm import MockLLMClient


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def repo_with_session(temp_db):
    """Create a repo with a session."""
    repo = AHCRepository(temp_db)
    repo.connect()

    session = SessionRecord(session_id="test-session")
    repo.insert_session(session)

    yield repo, session

    repo.close()


class TestJournalManager:
    """Tests for JournalManager."""

    def test_log_thought(self, repo_with_session):
        """Test logging a thought."""
        repo, session = repo_with_session
        journal = JournalManager(repo, session.id)
        journal.set_turn(1)

        entry_id = journal.log_thought("This is a thought")
        assert entry_id > 0

        entries = journal.get_recent_entries()
        assert len(entries) == 1
        assert entries[0].content == "This is a thought"

    def test_log_different_types(self, repo_with_session):
        """Test logging different entry types."""
        repo, session = repo_with_session
        journal = JournalManager(repo, session.id)
        journal.set_turn(1)

        journal.log_observation("Observed something")
        journal.log_hypothesis("Maybe X causes Y")
        journal.log_experiment("Testing hypothesis")
        journal.log_conclusion("Hypothesis confirmed")

        entries = journal.get_recent_entries()
        assert len(entries) == 4

    def test_get_summary(self, repo_with_session):
        """Test getting journal summary."""
        repo, session = repo_with_session
        journal = JournalManager(repo, session.id)

        journal.set_turn(1)
        journal.log_thought("Thought 1")
        journal.log_observation("Observation 1")

        journal.set_turn(2)
        journal.log_thought("Thought 2")

        summary = journal.get_summary()
        assert summary["total_entries"] == 3
        assert summary["turns_covered"] == 2
        assert summary["by_type"]["thought"] == 2
        assert summary["by_type"]["observation"] == 1

    def test_format_for_context(self, repo_with_session):
        """Test formatting for LLM context."""
        repo, session = repo_with_session
        journal = JournalManager(repo, session.id)
        journal.set_turn(1)

        journal.log_thought("Initial thought")
        journal.log_observation("Saw pattern >.<")

        formatted = journal.format_for_context()
        assert "Initial thought" in formatted
        assert "Saw pattern >.<" in formatted
        assert "Turn 1" in formatted


class TestTerminationChecker:
    """Tests for TerminationChecker."""

    def test_check_not_terminated(self, repo_with_session):
        """Test checking when not terminated."""
        repo, session = repo_with_session
        checker = TerminationChecker(repo, session.id)

        status = checker.check()
        assert not status.terminated
        assert status.predictions_count == 0
        assert status.accuracy == 0.0

    def test_check_with_predictions(self, repo_with_session):
        """Test checking with some predictions."""
        repo, session = repo_with_session

        # Add some predictions
        from src.ahc.db.models import PredictionRecord
        for i in range(10):
            pred = PredictionRecord(
                session_id=session.id,
                turn_number=1,
                state_t0=f"state{i}",
                predicted_state_t1="correct",
                actual_state_t1="correct",
                is_correct=True,
            )
            repo.insert_prediction(pred)

        checker = TerminationChecker(repo, session.id)
        status = checker.check()

        assert not status.terminated  # Not enough predictions
        assert status.predictions_count == 10
        assert status.accuracy == 1.0

    def test_termination_status_dataclass(self):
        """Test TerminationStatus dataclass."""
        status = TerminationStatus(
            terminated=True,
            accuracy=1.0,
            predictions_count=5000,
            predictions_target=5000,
            transition_complete=True,
            reason="All conditions met",
        )

        assert status.terminated
        assert status.accuracy == 1.0
        assert status.predictions_count == 5000


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AgentConfig()
        assert config.model == "gemini-2.5-flash"
        assert config.max_turns == 10000
        assert config.accuracy_target == 1.0
        assert config.predictions_target == 5000

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            model="custom-model",
            max_turns=100,
            seed=123,
        )
        assert config.model == "custom-model"
        assert config.max_turns == 100
        assert config.seed == 123

    def test_to_json(self):
        """Test JSON serialization."""
        config = AgentConfig(seed=42)
        json_str = config.to_json()
        parsed = json.loads(json_str)
        assert parsed["seed"] == 42


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    def test_scripted_responses(self):
        """Test scripted response handling."""
        client = MockLLMClient()
        client.add_response("First response")
        client.add_response("Second response")

        r1 = client.generate_with_tools([], [])
        assert r1["content"] == "First response"

        r2 = client.generate_with_tools([], [])
        assert r2["content"] == "Second response"

    def test_tool_call_response(self):
        """Test response with tool call."""
        client = MockLLMClient()
        client.add_tool_call_response(
            tool_name="run_prediction",
            arguments={"state": ">.."},
            content="Let me test this state",
        )

        response = client.generate_with_tools([], [])
        assert response["content"] == "Let me test this state"
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0]["name"] == "run_prediction"

    def test_call_recording(self):
        """Test that calls are recorded."""
        client = MockLLMClient()
        client.add_response("Test")

        client.generate_with_tools(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"name": "test"}],
            temperature=0.5,
        )

        assert len(client.calls) == 1
        assert client.calls[0]["messages"][0]["content"] == "Hello"
        assert client.calls[0]["temperature"] == 0.5


class TestAgentLoopIntegration:
    """Integration tests for AgentLoop."""

    def test_agent_initialization(self, temp_db):
        """Test that agent initializes correctly."""
        config = AgentConfig(
            db_path=temp_db,
            max_turns=5,
        )

        mock_client = MockLLMClient()
        mock_client.add_response("Initial exploration")

        agent = AgentLoop(config=config, llm_client=mock_client)

        # Initialize session (don't run loop)
        agent._init_session()

        assert agent._session is not None
        assert agent._session.status == SessionStatus.RUNNING
        assert agent._tools is not None
        assert agent._journal is not None

        # Cleanup
        agent._repo.close()

    def test_agent_with_tool_call(self, temp_db):
        """Test agent processing a tool call."""
        config = AgentConfig(
            db_path=temp_db,
            max_turns=2,
        )

        mock_client = MockLLMClient()
        # First turn: make a tool call
        mock_client.add_tool_call_response(
            tool_name="run_prediction",
            arguments={"state": ">.."},
            content="Let me observe this state",
        )
        # Second turn: conclude
        mock_client.add_response("Based on my observation, > moves right.")

        agent = AgentLoop(config=config, llm_client=mock_client)

        # Run the agent (will hit max_turns)
        session_id = agent.run()

        assert session_id is not None

        # Verify tool call was recorded
        repo = AHCRepository(temp_db)
        repo.connect()
        try:
            session = repo.get_session(session_id)
            assert session is not None
            assert session.status == SessionStatus.COMPLETED

            # Check tool calls
            tool_calls = repo.get_tool_calls(session.id)
            assert len(tool_calls) >= 1
            assert any(tc.tool_name == "run_prediction" for tc in tool_calls)
        finally:
            repo.close()
