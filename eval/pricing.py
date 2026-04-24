"""
LLM provider pricing constants and token counter.

All prices in USD per 1,000 tokens.
Sources: AWS Bedrock pricing page (verified 2026-04-23).
Update this file when prices change — no other file should hard-code prices.
"""

from __future__ import annotations


# ── Bedrock cross-region inference profile pricing ────────────────────────────
# USD per 1,000 tokens

BEDROCK_PRICING: dict[str, dict[str, float]] = {
    # Claude Sonnet 4.6
    "us.anthropic.claude-sonnet-4-6":     {"input": 0.003,  "output": 0.015},
    "global.anthropic.claude-sonnet-4-6": {"input": 0.003,  "output": 0.015},
    # Claude Haiku 4.5
    "us.anthropic.claude-haiku-4-5-20251001":          {"input": 0.0008, "output": 0.004},
    "us.anthropic.claude-haiku-4-5-20251001-v1:0":     {"input": 0.0008, "output": 0.004},
    "global.anthropic.claude-haiku-4-5-20251001":      {"input": 0.0008, "output": 0.004},
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 0.0008, "output": 0.004},
}

# Direct Anthropic API pricing (USD / 1K tokens)
CLAUDE_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 0.003,  "output": 0.015},
    "claude-haiku-4-5":  {"input": 0.0008, "output": 0.004},
}

# Ollama: local inference — always $0
OLLAMA_PRICING: dict[str, dict[str, float]] = {}


def get_price(provider: str, model: str) -> dict[str, float]:
    """Return {"input": $/1K, "output": $/1K} for the given provider/model.
    Falls back to Sonnet 4.6 pricing for unknown Bedrock models.
    """
    if provider == "bedrock":
        return BEDROCK_PRICING.get(model, {"input": 0.003, "output": 0.015})
    if provider == "claude":
        return CLAUDE_PRICING.get(model, {"input": 0.003, "output": 0.015})
    # ollama and unknown providers: free
    return {"input": 0.0, "output": 0.0}


def compute_cost_usd(
    input_tokens: int,
    output_tokens: int,
    price: dict[str, float],
) -> float:
    """Compute cost in USD from token counts and $/1K price."""
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000


# ── Token counter (injected into providers during benchmark) ──────────────────

class TokenCounter:
    """
    Accumulates token usage across multiple LLM calls.
    Injected into provider instances; each complete() call adds to it.
    Call reset() before each pipeline run.
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def cost(self, price: dict[str, float]) -> float:
        return compute_cost_usd(self.input_tokens, self.output_tokens, price)
