"""
AWS Bedrock AgentCore Provider — LLMProvider backed by managed agent runtime.

Phase G: Routes llm.complete() calls to invoke_agent_runtime() instead of
direct Bedrock API calls. The deployed agent receives a JSON payload and
returns a text response.

Request payload sent to agent:
    {
        "messages": [...],       # Anthropic messages format
        "system": "...",
        "max_tokens": 4096,
        "temperature": 0.3,
    }

Expected response from agent:
    {"text": "..."}

Backward compatibility: if agent_runtime_arn is not set, falls back to BedrockProvider.

Environment variables:
    AGENTCORE_WRITER_ARN         → writer-agent runtime ARN
    AGENTCORE_CRITIC_ARN         → critic-agent runtime ARN (Stage 2)
    AGENTCORE_GAP_ARN            → gap-agent runtime ARN
    AGENTCORE_PLAN_ARN           → plan-stage2-agent runtime ARN
    AGENTCORE_REGION             → AWS region (default: us-east-1)
"""

from __future__ import annotations
import asyncio
import json
import os
from typing import AsyncIterator

import boto3

from .base import LLMProvider


class AgentCoreProvider(LLMProvider):
    """
    Async LLM calls via AWS Bedrock AgentCore managed runtime.

    Replaces direct Bedrock API calls with invoke_agent_runtime(), enabling:
    - Centralized CloudWatch monitoring (per-agent invocation logs + latency)
    - Canary rollout: swap agent version without changing client code
    - Horizontal scaling managed by AgentCore runtime
    - Privacy enforcement: argument construction happens in LangGraph node code,
      not inside the agent — auditable at code review

    Falls back to BedrockProvider if agent_runtime_arn is None or empty.
    """

    def __init__(
        self,
        agent_runtime_arn: str,
        region: str | None = None,
        fallback_model: str = "us.anthropic.claude-sonnet-4-6",
        token_counter=None,
    ):
        self._arn = agent_runtime_arn
        self._region = region or os.getenv("AGENTCORE_REGION", "us-east-1")
        self._token_counter = token_counter

        if self._arn:
            self._client = boto3.client("bedrock-agentcore", region_name=self._region)
            self._fallback: LLMProvider | None = None
        else:
            # No ARN configured — delegate directly to BedrockProvider
            from .bedrock import BedrockProvider
            self._client = None
            self._fallback = BedrockProvider(
                model=fallback_model,
                region=self._region,
                token_counter=token_counter,
            )

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        if self._fallback is not None:
            return await self._fallback.complete(
                messages, system=system, max_tokens=max_tokens, temperature=temperature
            )

        payload = json.dumps({
            "messages": messages,
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.invoke_agent_runtime(
                agentRuntimeArn=self._arn,
                payload=payload,
                contentType="application/json",
                accept="application/json",
            ),
        )

        raw = response["payload"].read()
        data = json.loads(raw)

        text = data.get("text") or data.get("content") or ""
        if not isinstance(text, str):
            text = str(text)

        return text

    async def stream(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        # AgentCore invoke is request/response — yield the full response as one chunk
        text = await self.complete(
            messages, system=system, max_tokens=max_tokens, temperature=temperature
        )
        yield text

    async def embed(self, text: str) -> list[float]:
        if self._fallback is not None:
            return await self._fallback.embed(text)
        raise NotImplementedError(
            "Embedding not supported via AgentCore. Use OllamaProvider for local embeddings."
        )


def from_env(role: str, fallback_model: str = "us.anthropic.claude-sonnet-4-6") -> LLMProvider:
    """
    Build an AgentCoreProvider (or BedrockProvider fallback) from environment variables.

    role: "writer" | "critic" | "gap" | "plan"

    Reads AGENTCORE_{ROLE}_ARN. If not set, returns BedrockProvider directly.
    """
    env_key = f"AGENTCORE_{role.upper()}_ARN"
    arn = os.getenv(env_key, "").strip()
    if not arn:
        from .bedrock import BedrockProvider
        return BedrockProvider(model=fallback_model)
    return AgentCoreProvider(agent_runtime_arn=arn, fallback_model=fallback_model)
