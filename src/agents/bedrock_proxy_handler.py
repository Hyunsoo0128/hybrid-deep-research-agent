"""
AgentCore Bedrock Proxy — server-side handler for deployment to AgentCore managed runtime.

This file is deployed to AWS Bedrock AgentCore as the agent runtime code.
It receives requests from AgentCoreProvider.complete() and proxies them to Bedrock.

Deployment:
    1. Package this file + anthropic SDK into a container or S3 zip.
    2. Create agent runtime via bedrock-agentcore-control:
       aws bedrock-agentcore-control create-agent-runtime \
         --agent-runtime-name writer-agent-v1 \
         --agent-runtime-artifact '{"codeConfiguration": {...}}' \
         --role-arn arn:aws:iam::ACCOUNT:role/AgentCoreRole
    3. Create an endpoint and note the agentRuntimeArn.
    4. Set AGENTCORE_WRITER_ARN (or relevant env var) in the client environment.

Environment variables (in the AgentCore runtime):
    BEDROCK_MODEL     → model ID to use (default: us.anthropic.claude-sonnet-4-6)
    AWS_REGION        → region for Bedrock calls (default: us-east-1)

Request format (from AgentCoreProvider.complete()):
    {
        "messages": [...],       # Anthropic messages list
        "system": "...",         # system prompt (may be empty)
        "max_tokens": 4096,
        "temperature": 0.3,
    }

Response format:
    {"text": "..."}              # generated text
"""

from __future__ import annotations
import json
import os

import anthropic

_MODEL = os.getenv("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6")
_REGION = os.getenv("AWS_REGION", "us-east-1")

_client = anthropic.AnthropicBedrock(aws_region=_REGION)


def handler(event: dict, context=None) -> dict:
    """
    AgentCore invocation handler.

    Receives the decoded JSON payload from invoke_agent_runtime() and
    returns a dict that AgentCore serializes as the response payload.
    """
    messages: list[dict] = event.get("messages", [])
    system: str = event.get("system", "")
    max_tokens: int = int(event.get("max_tokens", 4096))
    temperature: float = float(event.get("temperature", 0.3))

    kwargs: dict = dict(
        model=_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if system:
        kwargs["system"] = system

    response = _client.messages.create(**kwargs)
    text = response.content[0].text

    return {"text": text}


def agentcore_entrypoint(payload: bytes) -> bytes:
    """
    Entry point for AgentCore runtime invocation.

    AgentCore calls this function with the raw request payload bytes.
    Returns the raw response payload bytes.
    """
    event = json.loads(payload.decode("utf-8"))
    result = handler(event)
    return json.dumps(result).encode("utf-8")
