"""Tools for AHC-DS agent interaction.

Provides a set of tools the agent can invoke to:
- Evaluate laws against the simulator
- Run predictions
- Request samples
- Manage theorems and memory
"""

from src.ahc.tools.base import BaseTool, ToolResult, ToolError
from src.ahc.tools.registry import ToolRegistry

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolError",
    "ToolRegistry",
]
