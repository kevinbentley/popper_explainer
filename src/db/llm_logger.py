"""LLM logging utility for capturing all LLM interactions.

This module provides utilities for logging LLM calls to the database
for debugging, auditing, and replay purposes.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

from src.db.models import LLMTranscriptRecord

if TYPE_CHECKING:
    from src.db.repo import Repository


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients that can be wrapped."""

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from a prompt."""
        ...


@dataclass
class LLMLoggerContext:
    """Context for LLM logging.

    Tracks run/iteration/phase context that persists across multiple calls.
    """

    run_id: str | None = None
    iteration_id: int | None = None
    phase: str | None = None


@dataclass
class LLMLogger:
    """Logger for LLM interactions.

    Captures all LLM calls with full prompt/response text, timing,
    and token usage for debugging and auditing.

    Usage:
        logger = LLMLogger(repo, component="law_proposer", model_name="gemini-2.5-flash")
        logger.set_context(run_id="run_123", iteration_id=5, phase="discovery")

        # Log a call manually
        logger.log_call(
            prompt="...",
            system_instruction="...",
            response="...",
            duration_ms=1234,
            success=True,
        )

        # Or use the wrapper to automatically log
        wrapped_client = logger.wrap_client(original_client)
    """

    repo: Repository
    component: str
    model_name: str
    context: LLMLoggerContext = field(default_factory=LLMLoggerContext)

    def set_context(
        self,
        run_id: str | None = None,
        iteration_id: int | None = None,
        phase: str | None = None,
    ) -> None:
        """Update the logging context.

        Args:
            run_id: Orchestration run ID
            iteration_id: Current iteration index
            phase: Current phase name
        """
        if run_id is not None:
            self.context.run_id = run_id
        if iteration_id is not None:
            self.context.iteration_id = iteration_id
        if phase is not None:
            self.context.phase = phase

    def clear_context(self) -> None:
        """Reset the logging context."""
        self.context = LLMLoggerContext()

    def log_call(
        self,
        prompt: str,
        response: str,
        success: bool = True,
        system_instruction: str | None = None,
        research_log: str | None = None,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        thinking_tokens: int = 0,
        duration_ms: int | None = None,
        error_message: str | None = None,
    ) -> int:
        """Log an LLM call to the database.

        Args:
            prompt: The prompt text sent to the LLM
            response: The response received (or empty if failed)
            success: Whether the call succeeded
            system_instruction: Optional system instruction
            research_log: Extracted research_log from law_proposer response
            prompt_tokens: Estimated prompt tokens
            output_tokens: Estimated output tokens
            thinking_tokens: Thinking tokens (for extended thinking models)
            duration_ms: Call duration in milliseconds
            error_message: Error message if call failed

        Returns:
            Database ID of the inserted record
        """
        # Compute prompt hash for deduplication tracking
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        # Estimate tokens if not provided
        if prompt_tokens is None:
            prompt_tokens = len(prompt) // 4  # Rough estimate
        if output_tokens is None:
            output_tokens = len(response) // 4  # Rough estimate

        total_tokens = prompt_tokens + output_tokens + thinking_tokens

        record = LLMTranscriptRecord(
            component=self.component,
            model_name=self.model_name,
            prompt=prompt,
            raw_response=response,
            research_log=research_log,
            success=success,
            run_id=self.context.run_id,
            iteration_id=self.context.iteration_id,
            phase=self.context.phase,
            system_instruction=system_instruction,
            prompt_hash=prompt_hash,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            error_message=error_message,
        )

        return self.repo.insert_llm_transcript(record)

    def wrap_client(self, client: LLMClientProtocol) -> "LoggingClientWrapper":
        """Wrap an LLM client to automatically log all calls.

        Args:
            client: The original LLM client

        Returns:
            A wrapped client that logs all generate() calls
        """
        return LoggingClientWrapper(client, self)


class LoggingClientWrapper:
    """Wrapper that logs all LLM client calls.

    This wrapper intercepts generate() calls, times them,
    and logs them to the database via the provided logger.
    """

    def __init__(self, client: LLMClientProtocol, logger: LLMLogger):
        """Initialize the wrapper.

        Args:
            client: The original LLM client to wrap
            logger: The logger to use for recording calls
        """
        self._client = client
        self._logger = logger

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response, logging the call.

        Args:
            prompt: The prompt text
            system_instruction: Optional system instruction
            temperature: Optional temperature override
            **kwargs: Additional arguments passed to the underlying client

        Returns:
            The response text from the LLM
        """
        start_time = time.time()
        response = ""
        success = True
        error_message = None

        try:
            response = self._client.generate(
                prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                **kwargs,
            )
            return response
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            self._logger.log_call(
                prompt=prompt,
                response=response,
                success=success,
                system_instruction=system_instruction,
                duration_ms=duration_ms,
                error_message=error_message,
            )


def create_logging_client_wrapper(
    client: LLMClientProtocol,
    repo: Repository,
    component: str,
    model_name: str,
    run_id: str | None = None,
    iteration_id: int | None = None,
    phase: str | None = None,
) -> LoggingClientWrapper:
    """Create a logging wrapper for an LLM client.

    This is a convenience function for creating a wrapped client
    with a logger configured in one step.

    Args:
        client: The LLM client to wrap
        repo: Repository for database operations
        component: Component name for logging
        model_name: Model name for logging
        run_id: Optional run ID for context
        iteration_id: Optional iteration ID for context
        phase: Optional phase name for context

    Returns:
        A wrapped client that logs all calls
    """
    logger = LLMLogger(repo=repo, component=component, model_name=model_name)
    logger.set_context(run_id=run_id, iteration_id=iteration_id, phase=phase)
    return logger.wrap_client(client)
