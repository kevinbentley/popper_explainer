"""Gemini LLM client for law proposal."""

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class GeminiConfig:
    """Configuration for Gemini API client.

    Attributes:
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        model: Model to use (default: gemini-1.5-flash)
        temperature: Sampling temperature (default: 0.7)
        max_output_tokens: Maximum tokens in response (default: 4096)
        top_p: Nucleus sampling parameter (default: 0.95)
        top_k: Top-k sampling parameter (default: 40)
    """

    api_key: str | None = None
    model: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 4096
    top_p: float = 0.95
    top_k: int = 40

    def __post_init__(self):
        if self.api_key is None:
            # Check both common environment variable names
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class GeminiClient:
    """Client for Gemini LLM API.

    Usage:
        client = GeminiClient(config)
        response = client.generate(prompt)
    """

    def __init__(self, config: GeminiConfig | None = None):
        """Initialize the Gemini client.

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

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The prompt to send
            system_instruction: Optional system instruction
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            The generated text response
        """
        self._ensure_initialized()

        generation_config = {
            "temperature": temperature or self.config.temperature,
            "max_output_tokens": max_tokens or self.config.max_output_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
        }

        # Create model with system instruction if provided
        if system_instruction:
            model = self._client.GenerativeModel(
                self.config.model,
                system_instruction=system_instruction,
            )
        else:
            model = self._model

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        return response.text

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Generate a JSON response from the LLM.

        Attempts to parse the response as JSON.

        Args:
            prompt: The prompt to send
            system_instruction: Optional system instruction
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Parsed JSON object

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        text = self.generate(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Try to extract JSON from markdown code blocks
        text = self._extract_json(text)

        return json.loads(text)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()

        # Check for markdown code blocks
        if text.startswith("```"):
            # Find the end of the opening fence
            first_newline = text.find("\n")
            if first_newline == -1:
                return text

            # Find the closing fence
            closing_fence = text.rfind("```")
            if closing_fence > first_newline:
                text = text[first_newline + 1 : closing_fence].strip()

        return text

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate token count
        """
        self._ensure_initialized()

        try:
            result = self._model.count_tokens(text)
            return result.total_tokens
        except Exception:
            # Fallback: rough estimate based on character count
            return len(text) // 4


class MockGeminiClient:
    """Mock client for testing without API calls."""

    def __init__(self, responses: list[str] | None = None):
        """Initialize mock client.

        Args:
            responses: List of responses to return in order
        """
        self._responses = responses or []
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Record call and return mock response."""
        self.calls.append({
            "prompt": prompt,
            "system_instruction": system_instruction,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = "[]"  # Default empty response

        self._call_count += 1
        return response

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Record call and return parsed mock response."""
        text = self.generate(prompt, system_instruction, temperature, max_tokens)
        return json.loads(text)

    def count_tokens(self, text: str) -> int:
        """Return approximate token count."""
        return len(text) // 4
