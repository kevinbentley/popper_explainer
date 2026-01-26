"""Severe Test Generator for the Reflection Engine.

Converts theorist suggestions into concrete SevereTestCommand objects
that can be consumed by the discovery handler on the next iteration.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.reflection.models import SevereTestCommand, TheoristResult

logger = logging.getLogger(__name__)

# Valid command types
VALID_COMMAND_TYPES = {"initial_condition", "topology_test", "parity_challenge"}
VALID_PRIORITIES = {"high", "medium", "low"}


class SevereTestGenerator:
    """Generates SevereTestCommands from theorist output.

    Usage:
        generator = SevereTestGenerator()
        commands = generator.generate(theorist_result)
    """

    def generate(
        self,
        theorist_result: TheoristResult,
        max_commands: int = 10,
    ) -> list[SevereTestCommand]:
        """Generate severe test commands from theorist suggestions.

        Args:
            theorist_result: Output from the theorist task
            max_commands: Maximum commands to generate

        Returns:
            List of SevereTestCommand objects
        """
        commands: list[SevereTestCommand] = []

        for suggestion in theorist_result.severe_test_suggestions:
            if len(commands) >= max_commands:
                break

            cmd = self._convert_suggestion(suggestion)
            if cmd is not None:
                commands.append(cmd)

        # Also generate commands from hidden variable testable predictions
        for hv in theorist_result.hidden_variables:
            if len(commands) >= max_commands:
                break

            if hv.testable_prediction:
                commands.append(SevereTestCommand(
                    command_type="initial_condition",
                    description=(
                        f"Test hidden variable '{hv.name}': {hv.testable_prediction}"
                    ),
                    priority="high",
                ))

        return commands

    def _convert_suggestion(
        self,
        suggestion: dict[str, Any],
    ) -> SevereTestCommand | None:
        """Convert a single theorist suggestion to a SevereTestCommand.

        Args:
            suggestion: Raw suggestion dict from theorist

        Returns:
            SevereTestCommand or None if invalid
        """
        command_type = suggestion.get("command_type", "initial_condition")
        if command_type not in VALID_COMMAND_TYPES:
            command_type = "initial_condition"

        description = suggestion.get("description", "")
        if not description:
            return None

        priority = suggestion.get("priority", "medium")
        if priority not in VALID_PRIORITIES:
            priority = "medium"

        # Convert initial conditions list to JSON
        initial_conditions = suggestion.get("initial_conditions")
        initial_conditions_json = None
        if initial_conditions and isinstance(initial_conditions, list):
            initial_conditions_json = json.dumps(initial_conditions)

        # Convert grid lengths list to JSON
        grid_lengths = suggestion.get("grid_lengths")
        grid_lengths_json = None
        if grid_lengths and isinstance(grid_lengths, list):
            grid_lengths_json = json.dumps(grid_lengths)

        return SevereTestCommand(
            command_type=command_type,
            description=description,
            target_law_id=suggestion.get("target_law_id"),
            initial_conditions_json=initial_conditions_json,
            grid_lengths_json=grid_lengths_json,
            priority=priority,
        )
