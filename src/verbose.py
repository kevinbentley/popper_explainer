"""Verbose logging for LLM exchanges and probe events.

Two output channels:
- LLM exchanges (system instruction + prompt + response) -> log file
- Probe events (registration, execution) -> console (stderr)
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import TextIO, Union


class VerboseLogger:
    """Two-channel verbose logger.

    Channel 1 — log file: Every LLM call with full system instruction,
    prompt, and response text.

    Channel 2 — console: Probe lifecycle events (registration and
    execution calls with arguments and results).
    """

    def __init__(
        self,
        log_file: str | Path | None = None,
        console: TextIO | None = None,
    ) -> None:
        self._log_file = Path(log_file) if log_file else None
        self._console = console if console is not None else sys.stderr

    @staticmethod
    def _timestamp() -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # LLM exchange logging (to file)
    # ------------------------------------------------------------------

    def log_llm_exchange(
        self,
        component: str,
        system_instruction: str | None,
        prompt: str,
        response: str,
        duration_ms: int | None = None,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Write a full LLM exchange to the log file."""
        if self._log_file is None:
            return

        ts = self._timestamp()
        sep = "=" * 80

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{sep}\n")
            f.write(f"[{ts}] LLM CALL — {component}\n")
            parts = []
            if duration_ms is not None:
                parts.append(f"Duration: {duration_ms}ms")
            parts.append(f"Success: {success}")
            f.write(" | ".join(parts) + "\n")
            if error_message:
                f.write(f"Error: {error_message}\n")
            f.write(f"{sep}\n\n")

            if system_instruction:
                f.write("--- SYSTEM INSTRUCTION ---\n")
                f.write(system_instruction)
                f.write("\n--- END SYSTEM INSTRUCTION ---\n\n")

            f.write("--- PROMPT ---\n")
            f.write(prompt)
            f.write("\n--- END PROMPT ---\n\n")

            f.write("--- RESPONSE ---\n")
            f.write(response)
            f.write("\n--- END RESPONSE ---\n\n")

    # ------------------------------------------------------------------
    # Probe event logging (to console)
    # ------------------------------------------------------------------

    def log_probe_registered(
        self,
        probe_id: str,
        source: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Print probe registration event to console."""
        ts = self._timestamp()
        if status == "active":
            first_line = source.strip().splitlines()[0]
            self._console.write(
                f"[{ts}] PROBE REGISTERED: {probe_id} (active)\n"
                f"  source: {first_line}...\n"
            )
        else:
            self._console.write(
                f"[{ts}] PROBE REGISTERED: {probe_id} (status={status})\n"
            )
            if error_message:
                self._console.write(f"  error: {error_message}\n")
        self._console.flush()

    def log_probe_called(
        self,
        probe_id: str,
        state_repr: str,
        result: Union[float, int, None] = None,
        error: str | None = None,
    ) -> None:
        """Print probe execution event to console."""
        ts = self._timestamp()
        if error:
            self._console.write(
                f"[{ts}] PROBE CALL: {probe_id}('{state_repr}') -> ERROR: {error}\n"
            )
        else:
            self._console.write(
                f"[{ts}] PROBE CALL: {probe_id}('{state_repr}') -> {result}\n"
            )
        self._console.flush()

    def log_probe_dedup(
        self,
        new_probe_id: str,
        existing_probe_id: str,
    ) -> None:
        """Print probe deduplication event to console."""
        ts = self._timestamp()
        self._console.write(
            f"[{ts}] PROBE DEDUP: {new_probe_id} -> existing {existing_probe_id}\n"
        )
        self._console.flush()
