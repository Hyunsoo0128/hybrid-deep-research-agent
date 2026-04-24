#!/usr/bin/env python3
"""
Smoke Test — Quick validation before running the full benchmark

Checks:
  1. Ollama connectivity + model response
  2. Bedrock (Sonnet) connectivity
  3. S3 read/write
  4. standalone_benchmark.py T1 (JSON parsing) — single item
  5. standalone_benchmark.py T3 (Critic) — single item

Usage:
  MODEL=qwen3:8b \\
  OLLAMA_HOST=http://localhost:11434 \\
  SONNET_MODEL=us.anthropic.claude-sonnet-4-6 \\
  S3_BUCKET=my-bench-bucket \\
  AWS_REGION=us-east-1 \\
  python eval/smoke_test.py
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("MODEL",        "qwen3:8b")
SONNET_MODEL = os.getenv("SONNET_MODEL", "us.anthropic.claude-sonnet-4-6")
AWS_REGION   = os.getenv("AWS_REGION",   "us-east-1")
S3_BUCKET    = os.getenv("S3_BUCKET",    "")
S3_PREFIX    = os.getenv("S3_PREFIX",    "benchmark-results")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"


async def check_ollama() -> tuple[bool, str]:
    """Check Ollama connectivity and model response."""
    try:
        import httpx
        payload = {
            "model":   OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly one word: ready"}],
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 10},
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            r.raise_for_status()
            content = r.json()["message"]["content"].strip()
        return True, f"response='{content[:30]}'"
    except Exception as e:
        return False, str(e)[:80]


async def check_bedrock() -> tuple[bool, str]:
    """Check Bedrock Sonnet connectivity."""
    try:
        from anthropic import AsyncAnthropicBedrock
        client = AsyncAnthropicBedrock(aws_region=AWS_REGION)
        resp = await client.messages.create(
            model=SONNET_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly one word: ready"}],
            max_tokens=10,
            temperature=0.0,
        )
        content = resp.content[0].text.strip()
        return True, f"response='{content[:30]}'"
    except Exception as e:
        return False, str(e)[:80]


def check_s3() -> tuple[bool, str]:
    """Check S3 write + read + delete."""
    if not S3_BUCKET:
        return None, "S3_BUCKET not set"
    try:
        import boto3
        s3 = boto3.client("s3", region_name=AWS_REGION)
        test_key  = f"{S3_PREFIX}/smoke_test_{uuid.uuid4().hex[:8]}.json"
        test_body = json.dumps({"smoke_test": True}).encode()

        s3.put_object(Bucket=S3_BUCKET, Key=test_key, Body=test_body)
        r = s3.get_object(Bucket=S3_BUCKET, Key=test_key)
        read_back = r["Body"].read()
        s3.delete_object(Bucket=S3_BUCKET, Key=test_key)

        ok = read_back == test_body
        return ok, f"write+read+delete OK  bucket={S3_BUCKET}"
    except Exception as e:
        return False, str(e)[:80]


async def check_t1_smoke(ollama_client) -> tuple[bool, str]:
    """T1: Single JSON parsing check."""
    from eval.standalone_benchmark import llm_json
    prompt = '{"intent":"analytical","sub_queries":[{"id":"sq1","question":"What is quantum computing?","dimension":"Definition/Background"}],"depth":"normal"}'
    data = await llm_json(
        ollama_client,
        [{"role": "user", "content": f'Output ONLY this JSON: {prompt}'}],
        "Output ONLY valid JSON. No explanations.",
        '{"intent":"...","sub_queries":[...],"depth":"..."}',
        max_tokens=256, dsap_enabled=True,
        fallback=None,
    )
    ok = data is not None and "sub_queries" in data
    return ok, f"parsed_keys={list(data.keys()) if data else 'None'}"


async def check_t3_smoke(ollama_client) -> tuple[bool, str]:
    """T3: Single Critic check (AlignRAG error detection)."""
    from eval.standalone_benchmark import llm_json, T3_DRAFT, _T3_CRITIC_SYSTEM, _T3_CRITIC_PROMPT, T3_CITATIONS

    citations_text = "\n".join(f"[{c['id']}] {c['title']}: {c['excerpt']}" for c in T3_CITATIONS)
    data = await llm_json(
        ollama_client,
        [{"role": "user", "content": _T3_CRITIC_PROMPT.format(
            citations=citations_text, draft=T3_DRAFT
        )}],
        _T3_CRITIC_SYSTEM,
        '{"misaligned_claims":[],"uncited_claims":[],"suggestions":[],"passed":false}',
        max_tokens=800, dsap_enabled=True,
        fallback={"misaligned_claims": [], "passed": True},
    )
    detected = len(data.get("misaligned_claims", []))
    ok = data is not None and "misaligned_claims" in data
    return ok, f"detected={detected}/3 injected errors"


async def main() -> None:
    print(f"\n{'═'*60}")
    print(f"  SMOKE TEST")
    print(f"  Model:  {OLLAMA_MODEL}")
    print(f"  Host:   {OLLAMA_HOST}")
    print(f"  Sonnet: {SONNET_MODEL}")
    print(f"{'═'*60}\n")

    results = []

    # 1. Ollama
    print("  [1/5] Ollama connectivity ...", end="", flush=True)
    t0 = time.time()
    ok, msg = await check_ollama()
    elapsed = round(time.time() - t0, 1)
    status = PASS if ok else FAIL
    print(f"  {status}  {msg}  ({elapsed}s)")
    results.append(("Ollama", ok))

    # 2. Bedrock
    print("  [2/5] Bedrock connectivity ...", end="", flush=True)
    t0 = time.time()
    ok, msg = await check_bedrock()
    elapsed = round(time.time() - t0, 1)
    status = PASS if ok else FAIL
    print(f"  {status}  {msg}  ({elapsed}s)")
    results.append(("Bedrock", ok))

    # 3. S3
    print("  [3/5] S3 read/write       ...", end="", flush=True)
    t0 = time.time()
    ok, msg = check_s3()
    elapsed = round(time.time() - t0, 1)
    if ok is None:
        status = SKIP
        results.append(("S3", None))
    else:
        status = PASS if ok else FAIL
        results.append(("S3", ok))
    print(f"  {status}  {msg}  ({elapsed}s)")

    # 4. T1 smoke (requires Ollama to pass)
    if results[0][1]:  # Ollama passed
        from eval.standalone_benchmark import OllamaClient
        ollama_client = OllamaClient(OLLAMA_HOST, OLLAMA_MODEL)

        print("  [4/5] T1 JSON smoke       ...", end="", flush=True)
        t0 = time.time()
        ok, msg = await check_t1_smoke(ollama_client)
        elapsed = round(time.time() - t0, 1)
        status = PASS if ok else FAIL
        print(f"  {status}  {msg}  ({elapsed}s)")
        results.append(("T1 JSON", ok))

        # 5. T3 smoke
        print("  [5/5] T3 Critic smoke     ...", end="", flush=True)
        t0 = time.time()
        ok, msg = await check_t3_smoke(ollama_client)
        elapsed = round(time.time() - t0, 1)
        status = PASS if ok else FAIL
        print(f"  {status}  {msg}  ({elapsed}s)")
        results.append(("T3 Critic", ok))
    else:
        print("  [4/5] T1 JSON smoke       ...  ⏭  SKIP (Ollama failed)")
        print("  [5/5] T3 Critic smoke     ...  ⏭  SKIP (Ollama failed)")
        results.append(("T1 JSON", None))
        results.append(("T3 Critic", None))

    # Summary
    print(f"\n{'─'*60}")
    passed  = sum(1 for _, r in results if r is True)
    failed  = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    print(f"  Result: {passed} passed / {failed} failed / {skipped} skipped")

    if failed > 0:
        print("  ❌ Smoke test FAILED — fix issues before running full benchmark")
        sys.exit(1)
    else:
        print("  ✅ Smoke test PASSED — ready to run standalone_benchmark.py")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
