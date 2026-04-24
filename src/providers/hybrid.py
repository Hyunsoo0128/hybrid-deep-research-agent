"""
HybridProvider — Routes LLM calls between a cloud provider and a local provider.

Motivation (Privacy-Preserving Speculative RAG):
  Privacy:       original documents and queries stay local; cloud sees only
                 locally-generated abstractions.
  Cost:          evaluation nodes (CRAG scoring, CONSTRUCT, supervisor, …) run
                 locally at $0/call; generation nodes (writer, gap_detector, …)
                 use the cloud model.
  Rate limits:   high-frequency evaluation loops no longer consume cloud API quota.

Environment variables:
  LLM_PROVIDER=hybrid               → activates this provider
  HYBRID_CLOUD_PROVIDER=bedrock     → cloud backend  (bedrock | claude)
  HYBRID_LOCAL_PROVIDER=ollama      → local backend  (ollama)
  HYBRID_CLOUD_MODEL=...            → cloud model override (optional)
  HYBRID_LOCAL_MODEL=qwen3:8b       → local model override (optional)

Usage in nodes:
  Tier B (simple routing) — build_graph passes hybrid.cloud or hybrid.local
  directly to each node via partial(). No node code change required.

  Tier A (Speculative RAG pattern) — node receives the HybridProvider itself
  and accesses .cloud / .local explicitly:
      local_llm = llm.local if hasattr(llm, 'local') else llm
      cloud_llm = llm.cloud if hasattr(llm, 'cloud') else llm

  Graceful degradation: when llm is a plain provider (not HybridProvider),
  hasattr checks fall through and both roles use the same model. This preserves
  backward compatibility for LLM_PROVIDER=bedrock|claude|ollama users.
"""

from __future__ import annotations
from typing import AsyncIterator

from .base import LLMProvider


class HybridProvider:
    """
    Wraps a cloud provider and a local provider.

    Exposes .cloud and .local for nodes that need explicit routing (Tier A).
    The default complete() / stream() delegates to cloud; embed() always uses local.

    Satisfies the LLMProvider Protocol so it can be passed anywhere a plain
    provider is accepted (e.g. graph nodes that only call complete/embed).
    """

    def __init__(self, cloud: LLMProvider, local: LLMProvider) -> None:
        self.cloud: LLMProvider = cloud
        self.local: LLMProvider = local

    # ── LLMProvider Protocol implementation ────────────────────────────────

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Default: delegates to cloud. Tier B nodes that are fully routed to
        cloud or local receive the sub-provider directly from build_graph and
        never call this method."""
        return await self.cloud.complete(
            messages, system=system, max_tokens=max_tokens, temperature=temperature
        )

    async def stream(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Streaming always uses cloud (writer node is cloud-routed)."""
        async for chunk in await self.cloud.stream(
            messages, system=system, max_tokens=max_tokens, temperature=temperature
        ):
            yield chunk

    async def embed(self, text: str) -> list[float]:
        """Embeddings always use local — no document content leaves the machine."""
        return await self.local.embed(text)
