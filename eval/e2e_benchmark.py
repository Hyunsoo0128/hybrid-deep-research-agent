"""
Layer 2: End-to-End Pipeline Benchmark — Phase Final

BM1b: Uses actual build_graph() LangGraph pipeline (not manual simulation).
      Auto-resumes plan_review interrupt with approved=True.
BM2:  6 conditions: baseline / phase1 / phase1_2 / phase1_2_3 / +2 leave-one-out
BM3b: Bedrock (6 conditions) + Ollama slim (3 conditions)
BM4:  Extended metrics: CRAG / VCM / MASS-RAG / CONSTRUCT / AlignRAG / STRIDE / EAM
BM5c: 5 queries — factual / analytical / comparative / definitional / health (type diversity)

Fixture: eval/fixtures/mock_search_results.json (mocked Tavily results, no live calls)

Conditions (full / bedrock):
  baseline             all flags OFF
  phase1               query_decomp + crag + rhinoinsight
  phase1_2             phase1 + mass_rag + alignrag + dsap
  phase1_2_3           phase1_2 + stride + construct
  phase1_2_3_no_mass_rag  leave-one-out: mass_rag=False (marginal value of MASS-RAG)
  phase1_2_3_no_stride    leave-one-out: stride=False  (marginal value of STRIDE)

Conditions (slim / ollama):
  baseline, phase1_2, phase1_2_3

Conditions (hybrid — Phase A–G validation):
  phase1_2_3_hybrid    phase1_2_3 flags + HybridProvider (local qwen3:8b + cloud Haiku 4.5)
  Key metric: avg_misaligned_claims_count vs phase1_2_3 (single Bedrock) — Phase E delta

Output schema (eval/results/{label}.json):
  {
    "phase_label", "timestamp", "provider", "model", "pricing",
    "conditions": [
      {
        "name", "config", "dataset",
        "aggregate": {
          # quality
          avg_keyword_coverage, avg_citation_density, avg_report_length_score,
          avg_structure_score, avg_overall_score, avg_llm_judge,
          avg_citations_per_report,
          # CRAG
          avg_relevant_doc_ratio, avg_retrieval_confidence, avg_strip_retention,
          # VCM
          avg_checklist_coverage_ratio,
          # MASS-RAG + CONSTRUCT
          avg_mass_rag_synthesis_ratio, avg_trust_score, avg_untrustworthy_count,
          # AlignRAG
          avg_revision_count, avg_misaligned_claims_count,
          # STRIDE
          avg_supervisor_rewrite_ratio,
          # EAM
          avg_corroboration_ratio,
          # cost / latency
          cost_per_query_usd, latency_sec,
          total_input_tokens, total_output_tokens, total_cost_usd
        },
        "per_query": [...]
      }
    ]
  }

Usage:
  python -m eval.e2e_benchmark --provider bedrock --label phase_final
  python -m eval.e2e_benchmark --provider ollama  --label phase_final_ollama --slim
  python -m eval.e2e_benchmark --provider bedrock --label debug --conditions baseline,phase1_2_3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

os.environ.setdefault("TAVILY_API_KEY", "dummy-fixture-key")

from src.tools.search import SearchResult, SearchTool
from src.state import DEFAULT_FEATURE_FLAGS, initial_state
from eval.pricing import TokenCounter, get_price, compute_cost_usd

try:
    from langgraph.errors import GraphInterrupt
except ImportError:
    # Older langgraph versions may not have this; treat as generic Exception
    class GraphInterrupt(Exception):  # type: ignore[no-redef]
        pass


# ── Test conditions (BM2) ─────────────────────────────────────────────────────

_PHASE1: dict = {
    **{k: False for k in DEFAULT_FEATURE_FLAGS},
    "query_decomp": True,
    "crag":         True,
    "rhinoinsight": True,
}

_PHASE1_2: dict = {
    **_PHASE1,
    "mass_rag":  True,
    "alignrag":  True,
    "dsap":      True,
}

_PHASE1_2_3: dict = {
    **_PHASE1_2,
    "stride":    True,
    "construct": True,
}

CONDITIONS_FULL: dict[str, dict] = {
    "baseline":               {k: False for k in DEFAULT_FEATURE_FLAGS},
    "phase1":                 _PHASE1,
    "phase1_2":               _PHASE1_2,
    "phase1_2_3":             _PHASE1_2_3,
    # Leave-one-out: marginal value of MASS-RAG (most expensive technique)
    # construct=False too: CONSTRUCT operates on MASS-RAG outputs; no outputs → no scoring
    "phase1_2_3_no_mass_rag": {**_PHASE1_2_3, "mass_rag": False, "construct": False},
    # Leave-one-out: marginal value of STRIDE (most complex technique)
    "phase1_2_3_no_stride":   {**_PHASE1_2_3, "stride": False},
}

CONDITIONS_SLIM: dict[str, dict] = {
    "baseline":   {k: False for k in DEFAULT_FEATURE_FLAGS},
    "phase1_2":   _PHASE1_2,
    "phase1_2_3": _PHASE1_2_3,
}

# Phase A–G validation: compare single Bedrock vs HybridProvider on same flag set.
# Key signal: avg_misaligned_claims_count delta (Phase E Spec RAG effect).
# Run with: --provider hybrid --conditions phase1_2_3,phase1_2_3_hybrid
CONDITIONS_HYBRID: dict[str, dict] = {
    "phase1_2_3":        _PHASE1_2_3,         # baseline: single Bedrock provider
    "phase1_2_3_hybrid": _PHASE1_2_3,         # Phase A–G: HybridProvider (provider swap only)
}


# ── Test queries (BM5c — 5 queries, type-diverse) ────────────────────────────

TEST_QUERIES = [
    # Q1 — ANALYTICAL: cause/effect + comparison
    {
        "id": "q1",
        "query": "What are the main causes and effects of global warming?",
        "type": "analytical",
        "sub_queries": [
            {"id": "sq1a", "question": "What are the primary causes of global warming?", "dimension": "Cause/Mechanism"},
            {"id": "sq1b", "question": "What are the measurable effects of global warming on ecosystems?", "dimension": "Current State/Evidence"},
            {"id": "sq1c", "question": "How does CO2 compare to other greenhouse gases in warming potential?", "dimension": "Comparison/Alternatives"},
        ],
        "expected_keywords": ["CO2", "carbon dioxide", "greenhouse", "temperature", "fossil fuel", "methane"],
        "has_contradiction": False,
    },
    # Q2 — DEFINITIONAL: mechanism + limitations
    {
        "id": "q2",
        "query": "How does transformer architecture work in large language models?",
        "type": "definitional",
        "sub_queries": [
            {"id": "sq2a", "question": "What is the attention mechanism in transformer models?", "dimension": "Definition/Background"},
            {"id": "sq2b", "question": "What are the limitations of transformer architecture?", "dimension": "Limitations/Challenges"},
            {"id": "sq2c", "question": "How do transformers compare to RNNs for sequence modeling?", "dimension": "Comparison/Alternatives"},
        ],
        "expected_keywords": ["attention", "self-attention", "transformer", "encoder", "decoder", "BERT", "GPT"],
        "has_contradiction": False,
    },
    # Q3 — CURRENT STATE: progress + applications
    {
        "id": "q3",
        "query": "What is the current state of quantum computing and its practical applications?",
        "type": "current_state",
        "sub_queries": [
            {"id": "sq3a", "question": "What progress has been made in quantum computing hardware?", "dimension": "Current State/Evidence"},
            {"id": "sq3b", "question": "What are real-world applications of quantum computing today?", "dimension": "Definition/Background"},
            {"id": "sq3c", "question": "What are the main challenges to practical quantum computing?", "dimension": "Limitations/Challenges"},
        ],
        "expected_keywords": ["qubit", "quantum", "error correction", "IBM", "Google", "decoherence"],
        "has_contradiction": False,
    },
    # Q4 — COMPARATIVE: explicit trade-off (two technologies)
    {
        "id": "q4",
        "query": "How does Python compare to Rust for systems programming?",
        "type": "comparative",
        "sub_queries": [
            {"id": "sq4a", "question": "What are Python's strengths and weaknesses for systems programming?", "dimension": "Limitations/Challenges"},
            {"id": "sq4b", "question": "What makes Rust suitable for systems programming?", "dimension": "Definition/Background"},
            {"id": "sq4c", "question": "In what scenarios should you choose Rust over Python?", "dimension": "Comparison/Alternatives"},
        ],
        "expected_keywords": ["Python", "Rust", "performance", "memory safety", "GIL", "ownership", "borrow checker"],
        "has_contradiction": True,  # opinions on Python-for-systems diverge
    },
    # Q5 — FACTUAL/HEALTH: evidence-based + caveats
    {
        "id": "q5",
        "query": "What are the health effects of intermittent fasting and who should avoid it?",
        "type": "factual",
        "sub_queries": [
            {"id": "sq5a", "question": "What are the proven health benefits of intermittent fasting?", "dimension": "Current State/Evidence"},
            {"id": "sq5b", "question": "What are the risks and side effects of intermittent fasting?", "dimension": "Limitations/Challenges"},
            {"id": "sq5c", "question": "Who should avoid intermittent fasting for health reasons?", "dimension": "Cause/Mechanism"},
        ],
        "expected_keywords": ["fasting", "insulin", "autophagy", "weight loss", "metabolism", "blood sugar", "caloric"],
        "has_contradiction": True,  # some populations benefit, others face risk
    },
]


# ── Fixture loader & mock patch ──────────────────────────────────────────────

def load_fixtures() -> dict:
    fixture_path = ROOT / "eval" / "fixtures" / "mock_search_results.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}\nRun fixture collection first.")
    with open(fixture_path) as f:
        return json.load(f)


def build_mock_search(fixtures: dict, query_id: str) -> Any:
    from src.tools.search import FetchResult

    query_fixture = fixtures.get(query_id, {})
    all_results: list[SearchResult] = []

    for sq_id, sq_data in query_fixture.get("results", {}).items():
        for r in sq_data.get("search_results", []):
            all_results.append(SearchResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                summary=r.get("summary", "")[:200],
                relevance_score=float(r.get("relevance_score", 0.5)),
                raw_content=r.get("raw_content", ""),
            ))

    url_content_map: dict[str, str] = {r.url: r.raw_content or r.summary for r in all_results}

    async def mock_fetch_page_async(url: str, max_chars: int = 3000) -> FetchResult:
        content = url_content_map.get(url, "Fixture content for " + url)
        return FetchResult(url=url, title="", content=content[:max_chars], success=True)

    mock = MagicMock(spec=SearchTool)
    mock.search.return_value = all_results[:5]
    mock.search_async = AsyncMock(return_value=all_results[:5])
    mock.fetch_page.side_effect = lambda url, max_chars=3000: FetchResult(
        url=url, title="",
        content=url_content_map.get(url, "Fixture content")[:max_chars],
        success=True,
    )
    mock.fetch_page_async = mock_fetch_page_async
    return mock


# ── Pipeline runner (BM1b — real build_graph()) ──────────────────────────────

async def run_pipeline_graph(
    tq: dict,
    condition_flags: dict,
    llm: Any,
    mock_search: Any,
    counter: TokenCounter,
) -> dict:
    """
    BM1b: Run the actual compiled LangGraph pipeline.

    Uses build_graph() — tests the real production graph including generate_plan,
    supervisor, evidence_stage2, and all Phase 3 nodes. No manual simulation drift.

    LangGraph 0.6.8 interrupt behavior:
      - ainvoke() returns {'__interrupt__': [...]} dict (does NOT raise GraphInterrupt)
      - aget_state(config).next is non-empty when graph is paused at interrupt
      - ainvoke(Command(resume=...), config) resumes and runs to completion

    Plan review interrupt is auto-resumed with approved=True (accept LLM-generated plan).
    """
    from langgraph.types import Command
    from src.graph import build_graph

    t0 = time.time()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Fresh compiled graph per run — each has an isolated MemorySaver scope
    graph = build_graph(llm=llm, search_tool=mock_search)

    init = initial_state(
        session_id=f"bench_{thread_id[:8]}",
        query=tq["query"],
        report_length="standard",
        feature_flags=condition_flags,
    )

    # ── Phase 1: run until plan_review interrupt ──────────────────────────────
    # In LangGraph 0.6.8, ainvoke returns normally (no exception) with __interrupt__ in result.
    # The snapshot.next check below is the authoritative guard.
    try:
        await graph.ainvoke(init, config)
    except Exception:
        pass  # GraphInterrupt (older LG versions) or unexpected error — proceed to state check

    # ── Phase 2: auto-resume with approval (accept LLM-generated plan) ────────
    # snapshot.next is non-empty when graph is paused at an interrupt
    try:
        snapshot = await graph.aget_state(config)
        if snapshot and snapshot.next:
            try:
                await graph.ainvoke(
                    Command(resume={"approved": True, "report_length": "standard"}),
                    config,
                )
            except Exception:
                pass  # unexpected second interrupt or error
    except Exception:
        pass  # checkpointer issue — proceed to state collection

    # ── Collect final state ───────────────────────────────────────────────────
    state: dict = {}
    try:
        final_snapshot = await graph.aget_state(config)
        if final_snapshot:
            state = final_snapshot.values or {}
    except Exception:
        pass

    elapsed = time.time() - t0

    # ── Report ────────────────────────────────────────────────────────────────
    final_report = state.get("final_report") or state.get("draft_report", "")

    # ── CRAG metrics ─────────────────────────────────────────────────────────
    retrieval_quality: list[dict] = state.get("retrieval_quality", [])
    verdicts = [rq.get("verdict", "") for rq in retrieval_quality]
    relevant_doc_ratio   = verdicts.count("CORRECT") / len(verdicts) if verdicts else 0.0
    avg_retrieval_conf   = (sum(rq.get("max_doc_score", 0.0) for rq in retrieval_quality)
                            / len(retrieval_quality) if retrieval_quality else 0.0)
    avg_strip_retention  = (sum(rq.get("strip_retention_ratio", 1.0) for rq in retrieval_quality)
                            / len(retrieval_quality) if retrieval_quality else 1.0)

    # ── VCM metrics ───────────────────────────────────────────────────────────
    checklist: list[dict] = state.get("checklist", [])
    status_counts = {"pending": 0, "partial": 0, "complete": 0}
    for item in checklist:
        s = item.get("status", "pending")
        status_counts[s] = status_counts.get(s, 0) + 1
    checklist_coverage = (
        (status_counts["complete"] + 0.5 * status_counts["partial"]) / len(checklist)
        if checklist else 0.0
    )

    # ── MASS-RAG + CONSTRUCT metrics ──────────────────────────────────────────
    mass_rag_outs: list[dict] = state.get("mass_rag_outputs", [])
    mass_rag_synthesis_ratio = (
        sum(1 for m in mass_rag_outs if m.get("summary")) / len(mass_rag_outs)
        if mass_rag_outs else 0.0
    )
    trust_scores_list = [m["trust_scores"] for m in mass_rag_outs if m.get("trust_scores")]
    avg_trust_score = (
        sum(ts.get("document_score", 0.0) for ts in trust_scores_list) / len(trust_scores_list)
        if trust_scores_list else 0.0
    )
    avg_untrustworthy_count = (
        sum(len(ts.get("untrustworthy_fields", [])) for ts in trust_scores_list) / len(trust_scores_list)
        if trust_scores_list else 0.0
    )

    # ── AlignRAG metrics ──────────────────────────────────────────────────────
    critic_feedback = state.get("critic_feedback") or {}
    revision_count       = state.get("revision_count", 0)
    misaligned_count     = len(critic_feedback.get("misaligned_claims", []))

    # ── STRIDE Supervisor metrics ─────────────────────────────────────────────
    supervisor_decisions: list[dict] = state.get("supervisor_decisions", [])
    action_counts = {"retrieve": 0, "rewrite": 0, "answer": 0}
    for d in supervisor_decisions:
        a = d.get("action", "retrieve")
        action_counts[a] = action_counts.get(a, 0) + 1
    supervisor_rewrite_ratio = (
        action_counts["rewrite"] / len(supervisor_decisions)
        if supervisor_decisions else 0.0
    )

    # ── EAM verification metrics ──────────────────────────────────────────────
    evidence_store: list[dict] = state.get("evidence_store", [])
    ev_levels = {"corroborated": 0, "single_source": 0, "unverified": 0}
    for ev in evidence_store:
        lvl = ev.get("verification_level", "unverified")
        ev_levels[lvl] = ev_levels.get(lvl, 0) + 1
    corroboration_ratio = ev_levels["corroborated"] / len(evidence_store) if evidence_store else 0.0

    return {
        "report":                  final_report,
        "citations":               state.get("citations", []),
        "elapsed":                 elapsed,
        "input_tokens":            counter.input_tokens,
        "output_tokens":           counter.output_tokens,
        # CRAG
        "relevant_doc_ratio":      round(relevant_doc_ratio, 3),
        "avg_retrieval_confidence": round(avg_retrieval_conf, 3),
        "avg_strip_retention":     round(avg_strip_retention, 3),
        # VCM
        "checklist_count":         len(checklist),
        "checklist_status_counts": status_counts,
        "checklist_coverage_ratio": round(checklist_coverage, 3),
        # MASS-RAG + CONSTRUCT
        "mass_rag_count":               len(mass_rag_outs),
        "mass_rag_synthesis_ratio":     round(mass_rag_synthesis_ratio, 3),
        "mass_rag_has_spans":           any(m.get("key_spans") for m in mass_rag_outs),
        "mass_rag_has_inferences":      any(m.get("inferences") for m in mass_rag_outs),
        "avg_trust_score":              round(avg_trust_score, 3),
        "avg_untrustworthy_count":      round(avg_untrustworthy_count, 2),
        # AlignRAG
        "revision_count":           revision_count,
        "misaligned_claims_count":  misaligned_count,
        # STRIDE
        "supervisor_decisions_count": len(supervisor_decisions),
        "supervisor_action_counts":   action_counts,
        "supervisor_rewrite_ratio":   round(supervisor_rewrite_ratio, 3),
        # EAM
        "evidence_store_count":     len(evidence_store),
        "reranked_count":           len(state.get("reranked_citations", [])),
        "evidence_level_counts":    ev_levels,
        "corroboration_ratio":      round(corroboration_ratio, 3),
    }


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_keyword_coverage(report: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    rl = report.lower()
    return sum(1 for kw in keywords if kw.lower() in rl) / len(keywords)


_CITATION_DENSITY_TARGET = 20  # inline citations needed for full score (1.0)
# v2: score = min(inline_refs / TARGET, 1.0)
# Rationale: original formula (inline/total_available) penalizes systems that retrieve
# more sources (e.g. STRIDE supervisor rewrites), creating a metric artifact where
# better retrieval → lower density score. This version rewards absolute citation
# adequacy independent of how many sources were available.

def score_citation_density(report: str, citation_count: int) -> float:
    refs = re.findall(r"\[Source\s+\d+\]", report)
    return min(len(refs) / _CITATION_DENSITY_TARGET, 1.0)


def score_report_length(report: str) -> float:
    return min(len(report.split()) / 500, 1.0)


def score_structure(report: str) -> float:
    checks = [
        bool(re.search(r"^#{1,3} ", report, re.MULTILINE)),
        bool(re.search(r"^\d+\.", report, re.MULTILINE) or re.search(r"^[-*•]", report, re.MULTILINE)),
        bool(re.search(r"references", report, re.IGNORECASE)),
    ]
    return sum(checks) / len(checks)


def compute_scores(report: str, tq: dict, citation_count: int) -> dict[str, float]:
    return {
        "keyword_coverage": score_keyword_coverage(report, tq["expected_keywords"]),
        "citation_density": score_citation_density(report, citation_count),
        "report_length":    score_report_length(report),
        "structure":        score_structure(report),
    }


async def llm_judge_vs_baseline(
    query: str,
    report_a: str,
    report_b: str,
    llm: Any,
) -> float:
    if not report_b or len(report_b) < 100:
        return 0.0
    if not report_a or len(report_a) < 100:
        return 1.0

    prompt = (
        f'You are evaluating two research reports on the question: "{query}"\n\n'
        f"Report A:\n{report_a[:1500]}\n\nReport B:\n{report_b[:1500]}\n\n"
        "Which report is better in terms of: depth, citation quality, factual accuracy, and structure?\n"
        'Reply with ONLY one of: "A", "B", or "TIE".'
    )
    try:
        resp = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system="You are an objective research quality evaluator.",
            max_tokens=10,
            temperature=0.0,
        )
        resp = resp.strip().upper()
        if "B" in resp and "A" not in resp:
            return 1.0
        if "A" in resp and "B" not in resp:
            return 0.0
        return 0.5
    except Exception:
        return 0.5


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _pct(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (pct / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)


def _stats(lst: list[float]) -> dict:
    return {"mean": round(_mean(lst), 4), "p50": round(_pct(lst, 50), 4), "p95": round(_pct(lst, 95), 4)}


# ── Main benchmark ────────────────────────────────────────────────────────────

async def run_benchmark(
    provider: str,
    model: str | None,
    slim: bool = False,
    label: str | None = None,
    only_conditions: list[str] | None = None,
    query_ids: list[str] | None = None,
) -> None:
    resolved_model = model or {
        "bedrock": "us.anthropic.claude-sonnet-4-6",
        "claude":  "claude-sonnet-4-6",
        "ollama":  "qwen3:8b",
        "hybrid":  "us.anthropic.claude-haiku-4-5-20251001-v1:0",  # cloud model for pricing
    }.get(provider, "unknown")

    print(f"\nRunning E2E benchmark — {provider}/{resolved_model}  [BM1b: real build_graph()]")
    if provider == "hybrid":
        print(f"  Hybrid: cloud=Haiku4.5 (billed) + local=qwen3:8b ($0)")
        print(f"  Phase E Spec RAG active — misaligned_claims delta vs phase1_2_3 is key metric")
    if label:
        print(f"  Label: {label}")
    print("─" * 60)

    if provider == "hybrid":
        conditions_base = CONDITIONS_HYBRID
    elif slim:
        conditions_base = CONDITIONS_SLIM
    else:
        conditions_base = CONDITIONS_FULL
    if only_conditions:
        conditions = {k: v for k, v in conditions_base.items() if k in only_conditions}
        if not conditions:
            raise ValueError(f"No matching conditions. Available: {list(conditions_base.keys())}")
    else:
        conditions = conditions_base
    print(f"  Conditions: {list(conditions.keys())}")

    queries = [tq for tq in TEST_QUERIES if not query_ids or tq["id"] in query_ids]
    print(f"  Queries: {[tq['id'] for tq in queries]} ({len(queries)} total)")
    print(f"  Total runs: {len(conditions) * len(queries)}\n")

    price = get_price(provider, resolved_model)
    counter = TokenCounter()
    llm = _load_provider(provider, model, counter)
    fixtures = load_fixtures()

    # {cond_name: [per_query_result, ...]}
    all_condition_results: dict[str, list[dict]] = {c: [] for c in conditions}
    baseline_reports: dict[str, str] = {}  # query_id → report

    for tq in queries:
        print(f"\n  [{tq['id']}] ({tq['type']}) {tq['query'][:60]}...")
        mock_search = build_mock_search(fixtures, tq["id"])

        cond_outputs: dict[str, dict] = {}
        for cond_name, cond_flags in conditions.items():
            print(f"    [{cond_name:<28}] ", end="", flush=True)
            counter.reset()
            try:
                output = await run_pipeline_graph(tq, cond_flags, llm, mock_search, counter)
                cost = compute_cost_usd(output["input_tokens"], output["output_tokens"], price)
                output["cost_usd"] = cost
                cond_outputs[cond_name] = output
                words = len(output["report"].split())
                print(f"{output['elapsed']:5.1f}s  {words:4d}w  "
                      f"in={output['input_tokens']:5d}  out={output['output_tokens']:4d}  ${cost:.4f}")
            except Exception as e:
                cond_outputs[cond_name] = {
                    "report": "", "citations": [], "elapsed": 0.0,
                    "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                    "error": str(e),
                }
                print(f"ERROR: {e}")

        if "baseline" in cond_outputs:
            baseline_reports[tq["id"]] = cond_outputs["baseline"].get("report", "")

        for cond_name, output in cond_outputs.items():
            report    = output.get("report", "")
            citations = output.get("citations", [])
            err       = output.get("error", "")
            scores    = compute_scores(report, tq, len(citations)) if report else {}

            if cond_name == "baseline" or not baseline_reports.get(tq["id"]):
                judge_score = 0.5
            else:
                counter.reset()
                judge_score = await llm_judge_vs_baseline(
                    tq["query"], baseline_reports[tq["id"]], report, llm
                )

            all_condition_results[cond_name].append({
                "query_id":     tq["id"],
                "query":        tq["query"],
                "query_type":   tq.get("type", ""),
                "scores":       scores,
                "llm_judge":    judge_score,
                "elapsed_sec":  round(output.get("elapsed", 0.0), 2),
                "input_tokens": output.get("input_tokens", 0),
                "output_tokens": output.get("output_tokens", 0),
                "cost_usd":     round(output.get("cost_usd", 0.0), 6),
                "citation_count": len(citations),
                # CRAG
                "relevant_doc_ratio":        output.get("relevant_doc_ratio", 0.0),
                "avg_retrieval_confidence":  output.get("avg_retrieval_confidence", 0.0),
                "avg_strip_retention":       output.get("avg_strip_retention", 1.0),
                # VCM
                "checklist_count":           output.get("checklist_count", 0),
                "checklist_status_counts":   output.get("checklist_status_counts", {}),
                "checklist_coverage_ratio":  output.get("checklist_coverage_ratio", 0.0),
                # MASS-RAG + CONSTRUCT
                "mass_rag_count":            output.get("mass_rag_count", 0),
                "mass_rag_synthesis_ratio":  output.get("mass_rag_synthesis_ratio", 0.0),
                "mass_rag_has_spans":        output.get("mass_rag_has_spans", False),
                "mass_rag_has_inferences":   output.get("mass_rag_has_inferences", False),
                "avg_trust_score":           output.get("avg_trust_score", 0.0),
                "avg_untrustworthy_count":   output.get("avg_untrustworthy_count", 0.0),
                # AlignRAG
                "revision_count":            output.get("revision_count", 0),
                "misaligned_claims_count":   output.get("misaligned_claims_count", 0),
                # STRIDE
                "supervisor_decisions_count": output.get("supervisor_decisions_count", 0),
                "supervisor_action_counts":   output.get("supervisor_action_counts", {}),
                "supervisor_rewrite_ratio":   output.get("supervisor_rewrite_ratio", 0.0),
                # EAM
                "evidence_store_count":      output.get("evidence_store_count", 0),
                "reranked_count":            output.get("reranked_count", 0),
                "evidence_level_counts":     output.get("evidence_level_counts", {}),
                "corroboration_ratio":       output.get("corroboration_ratio", 0.0),
                "report_preview": report[:300],
                "error": err,
            })

    # ── Build output document ─────────────────────────────────────────────────
    cond_order = list(conditions.keys())
    conditions_output = []

    for cond_name in cond_order:
        per_query = all_condition_results[cond_name]
        if not per_query:
            continue

        # Quality scores
        kw   = [r["scores"].get("keyword_coverage", 0) for r in per_query]
        cd   = [r["scores"].get("citation_density",  0) for r in per_query]
        rl   = [r["scores"].get("report_length",      0) for r in per_query]
        st   = [r["scores"].get("structure",          0) for r in per_query]
        jd   = [r["llm_judge"]     for r in per_query]
        ela  = [r["elapsed_sec"]   for r in per_query]
        cit  = [r["citation_count"] for r in per_query]
        cost = [r["cost_usd"]      for r in per_query]
        overall = [(a + b + c + d) / 4 for a, b, c, d in zip(kw, cd, rl, st)]

        conditions_output.append({
            "name": cond_name,
            "config": {"feature_flags": conditions[cond_name], "depth": "normal"},
            "dataset": {"name": "e2e_fixtures", "sample_count": len(per_query)},
            "aggregate": {
                # Quality
                "avg_keyword_coverage":      round(_mean(kw),      3),
                "avg_citation_density":      round(_mean(cd),      3),
                "avg_report_length_score":   round(_mean(rl),      3),
                "avg_structure_score":       round(_mean(st),      3),
                "avg_overall_score":         round(_mean(overall), 3),
                "avg_llm_judge":             round(_mean(jd),      3),
                "avg_citations_per_report":  round(_mean(cit),     1),
                # CRAG
                "avg_relevant_doc_ratio":    round(_mean([r["relevant_doc_ratio"]       for r in per_query]), 3),
                "avg_retrieval_confidence":  round(_mean([r["avg_retrieval_confidence"]  for r in per_query]), 3),
                "avg_strip_retention":       round(_mean([r["avg_strip_retention"]        for r in per_query]), 3),
                # VCM
                "avg_checklist_coverage_ratio": round(_mean([r["checklist_coverage_ratio"] for r in per_query]), 3),
                # MASS-RAG + CONSTRUCT
                "avg_mass_rag_synthesis_ratio": round(_mean([r["mass_rag_synthesis_ratio"] for r in per_query]), 3),
                "avg_trust_score":              round(_mean([r["avg_trust_score"]           for r in per_query]), 3),
                "avg_untrustworthy_count":      round(_mean([r["avg_untrustworthy_count"]   for r in per_query]), 2),
                # AlignRAG
                "avg_revision_count":           round(_mean([r["revision_count"]            for r in per_query]), 2),
                "avg_misaligned_claims_count":  round(_mean([r["misaligned_claims_count"]   for r in per_query]), 2),
                # STRIDE
                "avg_supervisor_rewrite_ratio": round(_mean([r["supervisor_rewrite_ratio"]  for r in per_query]), 3),
                # EAM
                "avg_corroboration_ratio":      round(_mean([r["corroboration_ratio"]        for r in per_query]), 3),
                # Cost / latency
                "cost_per_query_usd":  _stats(cost),
                "latency_sec":         _stats(ela),
                "total_input_tokens":  sum(r["input_tokens"]  for r in per_query),
                "total_output_tokens": sum(r["output_tokens"] for r in per_query),
                "total_cost_usd":      round(sum(cost), 6),
            },
            "per_query": per_query,
        })

    output_doc = {
        "phase_label": label or f"{provider}_{resolved_model}",
        "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider":    provider,
        "model":       resolved_model,
        "pricing":     {"input_per_1k": price["input"], "output_per_1k": price["output"]},
        "benchmark_config": {
            "infra": "BM1b_build_graph",
            "conditions": cond_order,
            "query_count": len(queries),
            "total_runs": len(conditions) * len(queries),
        },
        "conditions":  conditions_output,
    }

    _print_summary(output_doc, cond_order)

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_label = label or f"{provider}_{resolved_model.replace(':', '_').replace('/', '_')}"
    out_path = out_dir / f"{file_label}.json"
    with open(out_path, "w") as f:
        json.dump(output_doc, f, indent=2)
    print(f"\nSaved: {out_path.relative_to(ROOT)}")


def _print_summary(doc: dict, cond_order: list[str]) -> None:
    title = f"{doc['phase_label']}  ({doc['provider']}/{doc['model']})"
    W = 120
    print(f"\n{'═'*W}")
    print(f"  {title}")
    print(f"{'═'*W}")

    # ── Table 1: Quality scores ────────────────────────────────────────────────
    print(
        f"  {'Condition':<28} {'KWcov':>6} {'CitD':>5} {'Len':>5} {'Str':>5} "
        f"{'Judge':>6} {'Cites':>6} {'p50s':>6} {'p95s':>6} {'$/q':>8} {'Overall':>8}"
    )
    print(f"  {'─'*28} {'─'*6} {'─'*5} {'─'*5} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*8}")

    cond_map = {c["name"]: c for c in doc["conditions"]}
    for cname in cond_order:
        c = cond_map.get(cname)
        if not c:
            continue
        a = c["aggregate"]
        print(
            f"  {cname:<28} {a['avg_keyword_coverage']:>6.2f} {a['avg_citation_density']:>5.2f} "
            f"{a['avg_report_length_score']:>5.2f} {a['avg_structure_score']:>5.2f} "
            f"{a['avg_llm_judge']:>6.2f} {a['avg_citations_per_report']:>6.1f} "
            f"{a['latency_sec']['p50']:>6.1f} {a['latency_sec']['p95']:>6.1f} "
            f"{a['cost_per_query_usd']['mean']:>8.4f} {a['avg_overall_score']:>8.2f}"
        )

    print(f"  {'─'*28} {'─'*6} {'─'*5} {'─'*5} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*8}")

    # ── Table 2: Technique signal metrics ─────────────────────────────────────
    print(f"\n  Technique Signal Metrics:")
    print(
        f"  {'Condition':<28} {'CRAG%':>6} {'VCM%':>5} {'MRag%':>6} {'Trust':>6} "
        f"{'Revs':>5} {'Misal':>6} {'SupRW':>6} {'Corr%':>6}"
    )
    print(f"  {'─'*28} {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*6}")
    for cname in cond_order:
        c = cond_map.get(cname)
        if not c:
            continue
        a = c["aggregate"]
        print(
            f"  {cname:<28} "
            f"{a['avg_relevant_doc_ratio']:>6.2f} "
            f"{a['avg_checklist_coverage_ratio']:>5.2f} "
            f"{a['avg_mass_rag_synthesis_ratio']:>6.2f} "
            f"{a['avg_trust_score']:>6.2f} "
            f"{a['avg_revision_count']:>5.1f} "
            f"{a['avg_misaligned_claims_count']:>6.1f} "
            f"{a['avg_supervisor_rewrite_ratio']:>6.2f} "
            f"{a['avg_corroboration_ratio']:>6.2f}"
        )
    print(f"  {'─'*28} {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*6}")
    print("  [CRAG%=relevant_doc_ratio VCM%=checklist_coverage MRag%=synthesis_ratio Trust=avg_construct_score")
    print("   Revs=revision_count Misal=misaligned_claims SupRW=supervisor_rewrite_ratio Corr%=corroboration]")

    # ── Token / Cost totals ────────────────────────────────────────────────────
    print("\n  Token / Cost totals:")
    for cname in cond_order:
        c = cond_map.get(cname)
        if not c:
            continue
        a = c["aggregate"]
        print(
            f"    {cname:<28} in={a['total_input_tokens']:>7,}  "
            f"out={a['total_output_tokens']:>6,}  total=${a['total_cost_usd']:.4f}"
        )
    print(f"{'═'*W}")


# ── Provider loader ───────────────────────────────────────────────────────────

def _load_provider(provider: str, model: str | None, counter: TokenCounter):
    if provider == "bedrock":
        from src.providers.bedrock import BedrockProvider
        return BedrockProvider(
            model=model or "us.anthropic.claude-sonnet-4-6",
            token_counter=counter,
        )
    if provider == "claude":
        from src.providers.claude import ClaudeProvider
        return ClaudeProvider(model=model or "claude-sonnet-4-6")
    if provider == "ollama":
        from src.providers.ollama import OllamaProvider
        return OllamaProvider(model=model or "qwen3:8b", token_counter=counter)
    if provider == "hybrid":
        # Phase A–G: cloud = Bedrock Haiku 4.5 (billed, counter injected)
        #            local = Ollama qwen3:8b ($0, no counter)
        # Enables Phase E Spec RAG (local Stage1→cloud Stage2→local Stage3)
        from src.providers.hybrid import HybridProvider
        from src.providers.bedrock import BedrockProvider
        from src.providers.ollama import OllamaProvider
        cloud_model = model or "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return HybridProvider(
            cloud=BedrockProvider(model=cloud_model, token_counter=counter),
            local=OllamaProvider(model=os.getenv("OLLAMA_MODEL", "qwen3:8b"), host=ollama_host),
        )
    raise ValueError(f"Unknown provider: {provider}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E Pipeline Benchmark (Layer 2 — BM1b)")
    parser.add_argument("--provider",   default="bedrock", choices=["bedrock", "claude", "ollama", "hybrid"])
    parser.add_argument("--model",      default=None,  help="Override model ID")
    parser.add_argument("--slim",       action="store_true", help="3 conditions only (baseline/phase1_2/phase1_2_3)")
    parser.add_argument("--label",      default=None,  help="Phase tag in output filename (e.g. phase_final)")
    parser.add_argument("--conditions", default=None,  help="Comma-separated conditions (e.g. baseline,phase1_2_3)")
    parser.add_argument("--queries",    default=None,  help="Comma-separated query IDs (e.g. q1,q3)")
    args = parser.parse_args()

    only_c = [c.strip() for c in args.conditions.split(",")] if args.conditions else None
    only_q = [q.strip() for q in args.queries.split(",")]    if args.queries    else None
    asyncio.run(run_benchmark(
        args.provider, args.model,
        slim=args.slim, label=args.label,
        only_conditions=only_c, query_ids=only_q,
    ))
