"""Base tool infrastructure for AHC-DS."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ToolError(Exception):
    """Error raised when a tool execution fails."""
    pass


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        success: Whether the tool executed successfully
        data: The result data (tool-specific)
        error: Error message if success is False
        metadata: Additional metadata about the execution
    """
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def ok(cls, data: Any, **metadata) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata) -> "ToolResult":
        """Create a failed result."""
        return cls(success=False, error=error, metadata=metadata)


@dataclass
class ToolParameter:
    """Specification for a tool parameter.

    Used to generate function schemas for LLM tool calling.
    """
    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None
    items: dict[str, Any] | None = None  # For array types


class BaseTool(ABC):
    """Abstract base class for AHC-DS tools.

    Each tool must implement:
    - name: Tool name for dispatch
    - description: Human-readable description
    - parameters: List of parameter specifications
    - execute: The actual tool logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool description."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Return the parameter specifications."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with the execution outcome
        """
        pass

    def get_schema(self) -> dict[str, Any]:
        """Generate a JSON schema for this tool.

        This schema is compatible with OpenAI/Google function calling format.
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.items:
                prop["items"] = param.items
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def validate_args(self, **kwargs) -> list[str]:
        """Validate arguments against parameter specs.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for param in self.parameters:
            if param.required and param.name not in kwargs:
                errors.append(f"Missing required parameter: {param.name}")

        return errors
