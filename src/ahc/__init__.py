"""Agentic High-Context Discovery System (AHC-DS).

This module implements a continuous "Think-Act-Learn" loop where an LLM
maintains its own state via a live journal and emits tool calls to interact
with the physics simulator.

Key difference from the existing orchestrator: AHC-DS maintains a persistent
LLM conversation where the agent controls the discovery loop.
"""

__version__ = "0.1.0"
