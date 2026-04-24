"""
Phase E Validation — H2 Re-test

H2 failure (baseline): zero-shot AlignRAG detected 0 misalignments on the
ALIGNRAG_DRAFT_WITH_ERRORS fixture (3 injected errors).

This test verifies whether Phase E Spec RAG (Stage1 local → Stage2 cloud → Stage3 local)
produces misaligned_claims > 0 on the same fixture.

Conditions:
  baseline  — single-prompt critique() with a single provider (reproduces H2 failure)
  spec_rag  — Phase E critique() with HybridProvider (local drafter + cloud verifier)

Usage:
  python eval/critic_h2_retest.py
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.component_benchmark import (
    ALIGNRAG_CITATIONS,
    ALIGNRAG_DRAFT_WITH_ERRORS,
    ALIGNRAG_GOLD_ERROR_COUNT,
)
from src.nodes.critic import critique
from src.state import ResearchState

QUERY = "Current state of quantum computing"
SUB_QUERIES = [
    {"id": "sq1", "question": "Current state of quantum computing performance", "dimension": "Current State/Evidence"},
    {"id": "sq2", "question": "Technical challenges in quantum error correction", "dimension": "Limitations/Challenges"},
]


def _build_state(draft: str, provider: str) -> ResearchState:
    return ResearchState(
        original_query=QUERY,
        plan={"intent": "analytical", "sub_queries": SUB_QUERIES, "depth": "normal"},
        sub_queries=SUB_QUERIES,
        search_results=[],
        citations=ALIGNRAG_CITATIONS,
        evidence_store=[
            {
                "id": c["id"],
                "url": "",
                "title": c["title"],
                "excerpt": c["excerpt"],
                "confidence": 0.9,
                "trust_level": "high",
                "crawled_at": "",
                "verification_level": "verified",
            }
            for c in ALIGNRAG_CITATIONS
        ],
        draft_report=draft,
        final_report="",
        revision_count=0,
        critic_feedback={},
        mass_rag_outputs=[],
        feature_flags={"alignrag": True, "dsap": True},
        checklist=[],
        supervisor_decisions=[],
        remaining_steps=20,
        error_log=[],
    )


def _build_local_llm():
    from src.providers.ollama import OllamaProvider
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaProvider(model=model, host=host)


def _build_cloud_llm():
    provider = os.getenv("CLOUD_PROVIDER", "bedrock")
    if provider == "bedrock":
        from src.providers.bedrock import BedrockProvider
        model = os.getenv("CLOUD_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        return BedrockProvider(model=model)
    from src.providers.ollama import OllamaProvider
    return OllamaProvider(model=os.getenv("OLLAMA_MODEL", "qwen3:8b"))


def _build_hybrid_llm():
    from src.providers.hybrid import HybridProvider
    return HybridProvider(cloud=_build_cloud_llm(), local=_build_local_llm())


def _print_result(label: str, feedback: dict, elapsed: float):
    misaligned = feedback.get("misaligned_claims", [])
    count = len(misaligned)
    recall = min(count / ALIGNRAG_GOLD_ERROR_COUNT, 1.0)
    passed = feedback.get("passed", True)

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  misaligned_claims : {count} / {ALIGNRAG_GOLD_ERROR_COUNT} gold errors")
    print(f"  recall            : {recall:.2f}")
    print(f"  passed            : {passed}")
    print(f"  elapsed           : {elapsed:.1f}s")

    if misaligned:
        print("\n  Detected claims:")
        for i, m in enumerate(misaligned, 1):
            phase = m.get("phase", "?")
            claim = m.get("claim", "")[:80]
            hint = m.get("correction_hint", "")[:60]
            print(f"    [{i}] {phase}: \"{claim}\"")
            if hint:
                print(f"         hint: {hint}")
    else:
        print("\n  (no misaligned claims detected)")

    return {"label": label, "detected": count, "gold": ALIGNRAG_GOLD_ERROR_COUNT,
            "recall": round(recall, 3), "passed": passed, "elapsed_s": round(elapsed, 2),
            "misaligned_claims": misaligned}


async def run_baseline():
    """Single-provider path — reproduces H2 failure."""
    llm = _build_local_llm()
    state = _build_state(ALIGNRAG_DRAFT_WITH_ERRORS, "local")
    t0 = time.time()
    result = await critique(state, llm)
    elapsed = time.time() - t0
    return _print_result("Baseline (single local LLM)", result["critic_feedback"], elapsed)


async def run_spec_rag():
    """Phase E Spec RAG path — HybridProvider."""
    llm = _build_hybrid_llm()
    state = _build_state(ALIGNRAG_DRAFT_WITH_ERRORS, "hybrid")
    t0 = time.time()
    result = await critique(state, llm)
    elapsed = time.time() - t0
    return _print_result("Phase E Spec RAG (local→cloud→local)", result["critic_feedback"], elapsed)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "spec_rag", "both"], default="both")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase E H2 Re-test — AlignRAG Spec RAG Critic")
    print(f"  Gold errors in fixture: {ALIGNRAG_GOLD_ERROR_COUNT}")
    print("=" * 60)

    results = []
    if args.condition in ("baseline", "both"):
        results.append(await run_baseline())
    if args.condition in ("spec_rag", "both"):
        results.append(await run_spec_rag())

    # Summary
    print(f"\n{'═'*60}")
    print("  SUMMARY")
    print(f"{'═'*60}")
    for r in results:
        gate = "PASS ✓" if r["detected"] > 0 else "FAIL ✗"
        print(f"  {r['label'][:40]:<40}  recall={r['recall']:.2f}  [{gate}]")

    if len(results) == 2:
        delta = results[1]["recall"] - results[0]["recall"]
        print(f"\n  Recall delta (Spec RAG − baseline): {delta:+.2f}")
        h2_resolved = results[1]["detected"] > 0
        print(f"  H2 resolved: {'YES' if h2_resolved else 'NO'}")

    # Save results
    out_path = Path("eval/results/critic_h2_retest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"gold_error_count": ALIGNRAG_GOLD_ERROR_COUNT, "results": results}, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
