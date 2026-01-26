"""LLM client with function calling support."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GeminiConfig:
    """Configuration for Gemini API client.

    Attributes:
        api_key: Gemini API key
        model: Model to use
        temperature: Sampling temperature
        max_output_tokens: Maximum tokens in response
    """
    api_key: str | None = None
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 16384

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class FunctionCallingClient:
    """LLM client with function calling support.

    Extends the basic Gemini client to support tool/function calling
    for the AHC agent's think-act-learn loop.
    """

    def __init__(self, config: GeminiConfig | None = None):
        """Initialize the client.

        Args:
            config: Client configuration
        """
        self.config = config or GeminiConfig()
        self._client = None
        self._model = None

    def _ensure_initialized(self) -> None:
        """Lazily initialize the Gemini client."""
        if self._client is not None:
            return

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        if not self.config.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment "
                "variable or pass api_key in GeminiConfig."
            )

        genai.configure(api_key=self.config.api_key)
        self._client = genai
        self._model = genai.GenerativeModel(self.config.model)

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate a response with potential tool calls.

        Args:
            messages: Conversation history
            tools: Tool schemas in function calling format
            temperature: Override temperature

        Returns:
            Dict with 'content' (str) and 'tool_calls' (list)
        """
        self._ensure_initialized()

        # Convert tool schemas to Gemini format
        gemini_tools = self._convert_tools_to_gemini_format(tools)

        # Convert messages to Gemini format
        gemini_messages = self._convert_messages_to_gemini_format(messages)

        # Build generation config
        generation_config = {
            "temperature": temperature or self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
        }

        # Extract system instruction from messages
        system_instruction = self._extract_system_instruction(messages)

        try:
            # Create a model instance with tools and system instruction
            model_kwargs: dict[str, Any] = {
                "model_name": self.config.model,
            }
            if gemini_tools:
                model_kwargs["tools"] = gemini_tools
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction

            model_with_tools = self._client.GenerativeModel(**model_kwargs)

            # Generate response
            response = model_with_tools.generate_content(
                gemini_messages,
                generation_config=generation_config,
            )

            return self._parse_response(response)

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return {
                "content": f"Error: {str(e)}",
                "tool_calls": [],
            }

    def _convert_tools_to_gemini_format(
        self,
        tools: list[dict[str, Any]],
    ) -> list[Any]:
        """Convert tool schemas to Gemini function declarations.

        Args:
            tools: Tool schemas

        Returns:
            List of Gemini function declarations
        """
        if not tools:
            return []

        try:
            from google.generativeai import protos

            function_declarations = []

            for tool in tools:
                # Convert parameters to Gemini schema format
                params = tool.get("parameters", {})
                properties = params.get("properties", {})
                required = params.get("required", [])

                # Build Gemini-compatible schema
                schema_props = {}
                for prop_name, prop_spec in properties.items():
                    schema_props[prop_name] = self._convert_property_to_gemini(prop_spec)

                # Create function declaration
                func_decl = protos.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=protos.Schema(
                        type=protos.Type.OBJECT,
                        properties=schema_props,
                        required=required,
                    ) if schema_props else None,
                )
                function_declarations.append(func_decl)

            # Wrap in Tool
            return [protos.Tool(function_declarations=function_declarations)]

        except Exception as e:
            logger.error(
                f"Failed to convert tools to Gemini format: {e}. "
                "The model will have NO function calling capability.",
                exc_info=True,
            )
            raise

    def _convert_property_to_gemini(self, prop_spec: dict[str, Any]) -> Any:
        """Convert a property specification to Gemini Schema.

        Args:
            prop_spec: Property specification dict

        Returns:
            Gemini Schema object
        """
        from google.generativeai import protos

        prop_type = prop_spec.get("type", "string")
        gemini_type = self._map_type_to_gemini(prop_type)

        kwargs = {
            "type": gemini_type,
            "description": prop_spec.get("description", ""),
        }

        # Handle enum (Gemini requires all enum values to be strings)
        if "enum" in prop_spec:
            kwargs["enum"] = [str(v) for v in prop_spec["enum"]]

        # Handle array items
        if prop_type == "array" and "items" in prop_spec:
            items_spec = prop_spec["items"]
            kwargs["items"] = self._convert_property_to_gemini(items_spec)

        # Handle object properties
        if prop_type == "object" and "properties" in prop_spec:
            nested_props = {}
            for nested_name, nested_spec in prop_spec["properties"].items():
                nested_props[nested_name] = self._convert_property_to_gemini(nested_spec)
            kwargs["properties"] = nested_props

        return protos.Schema(**kwargs)

    def _map_type_to_gemini(self, json_type: str) -> Any:
        """Map JSON schema types to Gemini types."""
        from google.generativeai import protos

        type_map = {
            "string": protos.Type.STRING,
            "integer": protos.Type.INTEGER,
            "number": protos.Type.NUMBER,
            "boolean": protos.Type.BOOLEAN,
            "array": protos.Type.ARRAY,
            "object": protos.Type.OBJECT,
        }
        return type_map.get(json_type, protos.Type.STRING)

    def _extract_system_instruction(
        self,
        messages: list[dict[str, Any]],
    ) -> str | None:
        """Extract system messages to use as Gemini system_instruction.

        Args:
            messages: Conversation messages

        Returns:
            Combined system instruction string, or None if no system messages
        """
        system_parts = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    system_parts.append(content)

        if not system_parts:
            return None

        combined = "\n\n".join(system_parts)
        logger.debug(
            f"Extracted system instruction: {len(combined)} chars "
            f"from {len(system_parts)} system message(s)"
        )
        return combined

    def _convert_messages_to_gemini_format(
        self,
        messages: list[dict[str, Any]],
    ) -> list[Any]:
        """Convert messages to Gemini content format.

        Args:
            messages: Conversation messages

        Returns:
            Gemini-compatible content list
        """
        # Gemini uses a different conversation format
        # System messages go in system_instruction (extracted separately),
        # others as Content objects
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Skip system messages (passed via system_instruction parameter)
            if role == "system":
                continue

            # Map roles
            gemini_role = "user" if role == "user" else "model"

            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}],
            })

        return contents

    def _convert_protobuf_to_native(self, obj: Any) -> Any:
        """Recursively convert protobuf objects to native Python types.

        Args:
            obj: Protobuf object or native type

        Returns:
            Native Python type (dict, list, str, int, float, bool, None)
        """
        # Handle None
        if obj is None:
            return None

        # Handle native types
        if isinstance(obj, (str, int, float, bool)):
            return obj

        # Handle dict-like objects (MapComposite, etc.)
        if hasattr(obj, 'items'):
            return {k: self._convert_protobuf_to_native(v) for k, v in obj.items()}

        # Handle list-like objects (RepeatedComposite, etc.)
        if hasattr(obj, '__iter__'):
            return [self._convert_protobuf_to_native(item) for item in obj]

        # Fallback: try to convert to string
        return str(obj)

    def _parse_response(self, response) -> dict[str, Any]:
        """Parse a Gemini response into our standard format.

        Args:
            response: Gemini response object

        Returns:
            Dict with 'content' and 'tool_calls'
        """
        result = {
            "content": "",
            "tool_calls": [],
        }

        try:
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    # Check for text content
                    if hasattr(part, "text") and part.text:
                        result["content"] += part.text

                    # Check for function call
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        # Convert protobuf args to native Python types
                        args = self._convert_protobuf_to_native(fc.args) if fc.args else {}
                        result["tool_calls"].append({
                            "name": fc.name,
                            "arguments": args,
                        })

        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")
            result["content"] = str(response)

        return result


class MockLLMClient:
    """Mock LLM client for testing without API calls.

    Supports scripted responses and tool calls for testing
    the agent loop.
    """

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
    ):
        """Initialize the mock client.

        Args:
            responses: List of scripted responses to return in order
        """
        self._responses = responses or []
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    def add_response(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add a scripted response.

        Args:
            content: Response content
            tool_calls: Optional tool calls to include
        """
        self._responses.append({
            "content": content,
            "tool_calls": tool_calls or [],
        })

    def add_tool_call_response(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        content: str = "",
    ) -> None:
        """Add a response that makes a tool call.

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments
            content: Optional accompanying content
        """
        self._responses.append({
            "content": content,
            "tool_calls": [{
                "name": tool_name,
                "arguments": arguments,
            }],
        })

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Return the next scripted response.

        Args:
            messages: Conversation history (recorded)
            tools: Tool schemas (recorded)
            temperature: Temperature (ignored)

        Returns:
            Next scripted response
        """
        # Record the call
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
        })

        # Return scripted response or default
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = {
                "content": "I've completed my exploration for now.",
                "tool_calls": [],
            }

        self._call_count += 1
        return response

    def reset(self) -> None:
        """Reset the call counter."""
        self._call_count = 0
        self.calls = []
