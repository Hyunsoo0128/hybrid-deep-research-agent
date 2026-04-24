"""
Ollama Provider — Local LLM calls (Phase 5)

Identical interface to BedrockProvider / ClaudeProvider.
Environment variables:
  LLM_PROVIDER=ollama       → activates this provider
  OLLAMA_HOST               → Ollama server address (default: http://localhost:11434)
  OLLAMA_MODEL              → main LLM model (default: qwen3:14b)
  OLLAMA_EMBED_MODEL        → embedding model (default: nomic-embed-text)

Recommended models:
  qwen3:14b         — balanced Korean/English, strong reasoning, 14B parameters
  qwen3:32b         — higher accuracy, requires 32GB VRAM
  exaone3.5:7.8b    — LG AI, Korean-specialized
  llama4:scout      — Meta MoE, long context
"""

from __future__ import annotations
import os
from typing import AsyncIterator

import re as _re

# qwen3 thinking-mode control token — appended to system prompt to disable
# extended reasoning (which can 10-50x response time for agentic workloads).
_NO_THINK_TOKEN = "/no_think"
_THINK_TAG_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)


def _build_messages(
    messages: list[dict],
    system: str,
) -> list[dict]:
    """Prepend system prompt to messages list.
    Appends /no_think to system prompt to disable qwen3 extended reasoning."""
    system_with_flag = (system + "\n" + _NO_THINK_TOKEN).strip() if system else _NO_THINK_TOKEN
    return [{"role": "system", "content": system_with_flag}] + messages


def _strip_thinking(content: str) -> str:
    """Remove <think>...</think> blocks from model output if present."""
    return _THINK_TAG_RE.sub("", content).strip()


class OllamaProvider:
    """
    Ollama local LLM call implementation.

    Supports complete(), stream(), and embed().
    Embeddings use embed_model (default: nomic-embed-text).
    """

    def __init__(
        self,
        model: str | None = None,
        embed_model: str | None = None,
        host: str | None = None,
        token_counter=None,
    ):
        try:
            from ollama import AsyncClient
            self._AsyncClient = AsyncClient
        except ImportError as e:
            raise ImportError(
                "ollama package is required: pip install ollama>=0.4.0"
            ) from e

        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:14b")
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        _host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = AsyncClient(host=_host)
        self._counter = token_counter  # TokenCounter | None

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Single completion (non-streaming)."""
        full_messages = _build_messages(messages, system)
        try:
            response = await self._client.chat(
                model=self.model,
                messages=full_messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            )
            # ollama>=0.4: response.message.content
            # ollama<0.4 (dict): response["message"]["content"]
            msg = response.message if hasattr(response, "message") else response["message"]
            content = msg.content if hasattr(msg, "content") else msg["content"]

            # Accumulate token counts if counter is injected.
            # Ollama response fields: prompt_eval_count (input), eval_count (output)
            if self._counter is not None:
                in_tok  = getattr(response, "prompt_eval_count", 0) or 0
                out_tok = getattr(response, "eval_count", 0) or 0
                self._counter.add(in_tok, out_tok)

            return _strip_thinking(content) if content else ""
        except Exception as e:
            err = str(e)
            if "connect" in err.lower() or "refused" in err.lower():
                raise RuntimeError(
                    f"Cannot connect to Ollama server. "
                    f"Please check that 'ollama serve' is running. ({e})"
                ) from e
            raise

    async def stream(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Streaming completion."""
        full_messages = _build_messages(messages, system)
        try:
            async for chunk in await self._client.chat(
                model=self.model,
                messages=full_messages,
                stream=True,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            ):
                msg = chunk.message if hasattr(chunk, "message") else chunk.get("message", {})
                text = msg.content if hasattr(msg, "content") else msg.get("content", "")
                if text:
                    yield text
        except Exception as e:
            err = str(e)
            if "connect" in err.lower() or "refused" in err.lower():
                raise RuntimeError(
                    f"Cannot connect to Ollama server. "
                    f"Please check that 'ollama serve' is running. ({e})"
                ) from e
            raise

    async def embed(self, text: str) -> list[float]:
        """Text embedding (nomic-embed-text by default)."""
        try:
            response = await self._client.embed(
                model=self.embed_model,
                input=text,
            )
            # ollama>=0.4: response.embeddings[0]
            embeddings = (
                response.embeddings
                if hasattr(response, "embeddings")
                else response["embeddings"]
            )
            return embeddings[0]
        except Exception as e:
            raise RuntimeError(
                f"Ollama embedding error (model={self.embed_model}): {e}"
            ) from e
