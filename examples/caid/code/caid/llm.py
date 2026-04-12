"""LLM call abstraction via litellm for multi-provider support.

Provides a unified async interface for making LLM calls with
JSON parsing, retry logic, and token usage tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional, Type, TypeVar

import litellm
from pydantic import BaseModel

from caid.config import LLMConfig

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


class LLMError(Exception):
    """Raised when an LLM call fails after all retries."""

    pass


class LLMClient:
    """Async LLM client with structured output support.

    Wraps litellm.acompletion for multi-provider LLM calls,
    adding JSON extraction, Pydantic validation, retry logic,
    and cumulative token usage tracking.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    async def call(
        self,
        messages: list[dict[str, str]],
        response_model: Optional[Type[T]] = None,
        max_retries: int = 3,
    ) -> str | T:
        """Make an async LLM call.

        Args:
            messages: Chat messages in OpenAI format.
            response_model: If provided, parse response as this Pydantic model.
            max_retries: Number of retry attempts on transient errors.

        Returns:
            Plain text response, or validated Pydantic model if response_model given.

        Raises:
            LLMError: If all retries are exhausted.
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = await litellm.acompletion(
                    model=self.config.model,
                    messages=messages,
                    api_key=self.config.api_key or None,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                content: str = response.choices[0].message.content or ""

                # Track token usage
                usage = getattr(response, "usage", None)
                if usage:
                    self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
                    self.total_completion_tokens += getattr(
                        usage, "completion_tokens", 0
                    )

                if response_model is not None:
                    json_str = extract_json(content)
                    return response_model.model_validate_json(json_str)
                return content

            except (
                litellm.RateLimitError,
                litellm.Timeout,
                litellm.APIConnectionError,
            ) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2**attempt
                    logger.warning(
                        "LLM call attempt %d failed (%s), retrying in %ds",
                        attempt + 1,
                        type(e).__name__,
                        wait,
                    )
                    await asyncio.sleep(wait)
            except Exception as e:
                raise LLMError(f"LLM call failed: {e}") from e

        raise LLMError(f"LLM call failed after {max_retries} retries: {last_error}")

    @property
    def token_usage(self) -> dict[str, int]:
        """Current cumulative token usage."""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }


def extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown code blocks.

    Handles the following formats:
      - ```json ... ```
      - ``` ... ```
      - Raw JSON (starting with { or [)
    """
    # Try ```json block first
    match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try plain ``` block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        # Verify it looks like JSON
        if candidate.startswith(("{", "[")):
            return candidate

    # Try to find raw JSON
    text = text.strip()
    if text.startswith(("{", "[")):
        return text

    # Last resort: find the first { ... } or [ ... ] block
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        return brace_match.group(1)

    bracket_match = re.search(r"(\[.*\])", text, re.DOTALL)
    if bracket_match:
        return bracket_match.group(1)

    return text


if __name__ == "__main__":
    # Test JSON extraction
    test_cases = [
        ('```json\n{"key": "value"}\n```', '{"key": "value"}'),
        ('Here is the result:\n```\n{"a": 1}\n```\nDone.', '{"a": 1}'),
        ('{"direct": true}', '{"direct": true}'),
        ('Some text {"embedded": 1} more text', '{"embedded": 1}'),
    ]
    for text, expected in test_cases:
        result = extract_json(text)
        assert result == expected, f"Failed: {text!r} -> {result!r} != {expected!r}"
    print("JSON extraction tests passed")

    # Test LLMClient initialization
    config = LLMConfig(model="gpt-3.5-turbo", api_key="test-key")
    client = LLMClient(config)
    assert client.total_prompt_tokens == 0
    assert client.total_completion_tokens == 0
    print(f"LLMClient initialized: model={config.model}")
    print(f"Token usage: {client.token_usage}")
    print("LLM module OK")
