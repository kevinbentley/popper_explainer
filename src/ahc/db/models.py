"""Data models for AHC-DS database records."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    """Status of an AHC session."""
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class JournalEntryType(str, Enum):
    """Types of journal entries."""
    THOUGHT = "thought"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    CONCLUSION = "conclusion"
    ERROR = "error"


class TheoremStatus(str, Enum):
    """Status of a theorem."""
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REFUTED = "refuted"


@dataclass
class SessionRecord:
    """A top-level AHC session."""
    session_id: str
    status: SessionStatus = SessionStatus.RUNNING
    created_at: datetime | None = None
    updated_at: datetime | None = None
    config_json: str | None = None
    model_id: str | None = None
    seed: int | None = None

    # Termination metrics
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    transition_rules_complete: bool = False
    terminated_at: datetime | None = None
    termination_reason: str | None = None

    # Database ID (set after insert)
    id: int | None = None


@dataclass
class JournalEntry:
    """A chain-of-thought journal entry."""
    session_id: int
    turn_number: int
    entry_type: JournalEntryType
    content: str
    metadata_json: str | None = None
    created_at: datetime | None = None
    id: int | None = None


@dataclass
class ToolCallRecord:
    """Record of a tool invocation."""
    session_id: int
    turn_number: int
    tool_name: str
    arguments_json: str
    result_json: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    id: int | None = None


@dataclass
class PredictionRecord:
    """A prediction made by the agent."""
    session_id: int
    turn_number: int
    state_t0: str
    predicted_state_t1: str
    actual_state_t1: str
    is_correct: bool
    prediction_method: str | None = None
    created_at: datetime | None = None
    id: int | None = None


@dataclass
class TheoremRecord:
    """A theorem stored by the agent."""
    session_id: int
    name: str
    description: str | None
    law_ids_json: str
    status: TheoremStatus = TheoremStatus.PROPOSED
    evidence_json: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    id: int | None = None


@dataclass
class LawEvaluationRecord:
    """Result of evaluating a law."""
    session_id: int
    turn_number: int
    law_json: str
    law_id: str
    law_hash: str
    status: str  # PASS, FAIL, UNKNOWN
    reason_code: str | None = None
    counterexample_json: str | None = None
    power_metrics_json: str | None = None
    runtime_ms: int | None = None
    created_at: datetime | None = None
    id: int | None = None


@dataclass
class TrajectorySampleRecord:
    """Cached trajectory samples."""
    session_id: int
    pattern: str
    count_requested: int
    samples_json: str
    created_at: datetime | None = None
    id: int | None = None


@dataclass
class TransitionRuleRecord:
    """A discovered transition rule."""
    session_id: int
    symbol: str
    neighbor_config: str
    coordinate_class: str  # "even", "odd", or "any"
    result_symbol: str
    confidence: float = 1.0
    evidence_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    id: int | None = None


@dataclass
class ConversationTurnRecord:
    """A turn in the LLM conversation."""
    session_id: int
    turn_number: int
    role: str  # system, user, assistant
    content: str
    tool_calls_json: str | None = None
    token_count: int = 0  # Estimated tokens for context management
    created_at: datetime | None = None
    id: int | None = None


@dataclass
class MetaKnowledgeRecord:
    """Compaction snapshot storing synthesized knowledge state.

    Created during compaction events to preserve the agent's learned
    knowledge while trimming the raw conversation history.
    """
    session_id: int
    version: int  # Increments with each compaction

    # Knowledge state at compaction time
    theorems_snapshot_json: str | None = None  # Current validated theorems
    negative_knowledge: str | None = None  # Summarized falsifications

    # Compaction metadata
    last_compacted_turn: int = 0  # Turn number where compaction occurred
    turns_compacted: int | None = None  # Number of turns summarized
    token_count_before: int | None = None  # Tokens before compaction
    token_count_after: int | None = None  # Tokens after compaction

    # Audit
    compaction_prompt_hash: str | None = None  # Hash of summarization prompt
    created_at: datetime | None = None
    id: int | None = None
