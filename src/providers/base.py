"""
LLM Provider abstraction layer
— Swappable between Claude API (current) / Ollama (future)
"""

from typing import Protocol, AsyncIterator, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """All agents depend only on this interface. Replace this when swapping LLMs."""

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Single completion (non-streaming)"""
        ...

    async def stream(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Streaming completion"""
        ...

    async def embed(self, text: str) -> list[float]:
        """Text embedding"""
        ...
