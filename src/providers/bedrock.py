"""
AWS Bedrock Provider — Claude API calls

Identical interface to ClaudeProvider.
AWS credentials are automatically obtained from environment variables or IAM Role.

Supported authentication methods (in priority order):
  1. IAM Role (EC2, Lambda, ECS and other AWS environments)
  2. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables
  3. ~/.aws/credentials profile
"""

from __future__ import annotations
import os
from typing import AsyncIterator

from anthropic import AsyncAnthropicBedrock


class BedrockProvider:
    """
    Async Claude calls via AWS Bedrock.

    Latest models (claude-sonnet-4-6, etc.) cannot be called on-demand directly.
    Must use inference profile ID:
      us.anthropic.claude-sonnet-4-6   (us-west-2 recommended)
      global.anthropic.claude-sonnet-4-6

    token_counter: optional TokenCounter instance (from eval/pricing.py).
      When set, each complete() call accumulates real usage.input_tokens /
      usage.output_tokens from the Anthropic response object.
    """

    def __init__(
        self,
        model: str = "us.anthropic.claude-sonnet-4-6",
        region: str | None = None,
        token_counter=None,
    ):
        self.model = model
        self._region = region or os.getenv("AWS_REGION", "us-west-2")
        self._client = AsyncAnthropicBedrock(aws_region=self._region)
        self._counter = token_counter  # TokenCounter | None

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

        if self._counter is not None:
            usage = response.usage
            self._counter.add(usage.input_tokens, usage.output_tokens)

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
        # To be implemented with Amazon Titan Embeddings in Phase 3 (local files)
        raise NotImplementedError("Embedding will be implemented in Phase 3")
