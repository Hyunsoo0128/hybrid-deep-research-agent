"""
Claude API Provider
"""

from __future__ import annotations
import os
from typing import AsyncIterator
import anthropic


class ClaudeProvider:
    """
    Anthropic Claude API implementation.
    Only this class needs to change when swapping to OllamaProvider.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ):
        self.model = model
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def stream(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as s:
            async for text in s.text_stream:
                yield text

    async def embed(self, text: str) -> list[float]:
        # Embeddings not used in Phase 1. To be implemented in Phase 3 (local files).
        raise NotImplementedError("Embedding not implemented in Phase 1")
