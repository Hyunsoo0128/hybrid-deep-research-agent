"""
Search Worker Node — CRAG (arxiv:2401.15884) + MASS-RAG (arxiv:2604.18509)

CRAG implementation (Phase 1):
  Step 1 — Retrieval Evaluator: batch-score all results → float relevance_score per doc
  Step 2 — Query-level verdict: max(scores) → CORRECT / AMBIGUOUS / INCORRECT
            CORRECT   : max_score ≥ 0.5  (threshold CRAG_CORRECT_THRESHOLD)
            INCORRECT : max_score < 0.3  (threshold CRAG_INCORRECT_THRESHOLD)
            AMBIGUOUS : between
  Step 3 — Decompose-then-Recompose (CORRECT docs only):
            fetch full content → split into 2-sentence strips → LLM scores each strip →
            drop strips below 0.5 → reassemble filtered_excerpt  (A2 batch: strip scoring
            combined with extraction in one LLM call)
  Step 4 — INCORRECT path: minimal summary, emit retrieval_quality signal for gap_detector

MASS-RAG (Phase 2, arxiv:2604.18509):
  3-agent parallel analysis on CORRECT+AMBIGUOUS doc pool per sub-query:
    Summarizer  → concise domain summary (max 200 tokens)
    Extractor   → key spans with source citation IDs (max 400 tokens)
    Reasoner    → inferences and conclusions (max 300 tokens)
  Then Synthesis (M7a separate call, max 500 tokens) combines into M3b structured output:
    {summary, key_spans: [{text, source_citation_ids, type}],
     inferences: [{claim, supporting_span_indices}]}
  Design decisions: M1a (per sub-query pool), M4c' (confidence metadata in prompt),
  M5b (DSAP retry + degraded fallback), M6a (asyncio.gather), M7a (separate synthesis).
  fast depth: MASS-RAG disabled entirely.

DSAP (arxiv:2512.20660): JSON guard via llm_json on all structured calls.

Input:  {"sub_query": dict, "original_query": str, "depth": str, "feature_flags": dict}
Output: {"citations": list[dict], "retrieval_quality": list[dict], "mass_rag_outputs": list[dict]}
"""

from __future__ import annotations
import asyncio
import os
import re
import uuid
from datetime import datetime, timezone

from ..state import Citation, SourceType, TrustLevel
from ..tools.search import SearchTool
from ..security.injection_filter import InjectionFilter
from ..providers.base import LLMProvider
from ..utils.llm_json import llm_json
from .quality_scorer import score_mass_rag_output

_FETCH_THRESHOLD = 0.7

_DEPTH_PARAMS = {
    "fast":   {"max_results": 4,  "max_fetch": 1},
    "normal": {"max_results": 7,  "max_fetch": 3},
    "deep":   {"max_results": 12, "max_fetch": 6},
}

# CRAG verdict thresholds (query-level, based on max document score).
# Override via env vars for local LLM recalibration (Phase B-2).
# qwen3:8b tends to produce lower relevance scores than Claude — lower these
# thresholds if local scoring distribution shifts the AMBIGUOUS band too wide.
CRAG_CORRECT_THRESHOLD   = float(os.getenv("CRAG_CORRECT_THRESHOLD",   "0.5"))
CRAG_INCORRECT_THRESHOLD = float(os.getenv("CRAG_INCORRECT_THRESHOLD", "0.3"))


# ── CRAG Step 1: Retrieval Evaluator ─────────────────────────────────────────

_EVAL_SYSTEM = """You are an expert in evaluating information relevance.
Assess how relevant each search result is to the research question."""

_EVAL_PROMPT = """Research question: {question}

Search results:
{results_text}

Rate the relevance of each result to the research question.
relevance_score: 0.0 = completely irrelevant, 1.0 = directly and fully answers the question.

Respond ONLY in this JSON format:
{{"evaluations": [{{"index": 0, "relevance_score": 0.85}}, {{"index": 1, "relevance_score": 0.2}}]}}"""

_EVAL_SCHEMA = '{"evaluations": [{"index": 0, "relevance_score": 0.0}]}'


async def _evaluate_results(
    question: str,
    results: list,
    llm: LLMProvider,
    dsap_enabled: bool = True,
) -> dict[int, float]:
    """
    CRAG Retrieval Evaluator.
    Returns {index: relevance_score (0.0–1.0)} for all results.
    """
    results_text_lines = [
        f"[{i}] {r.title}\n    {r.summary[:150]}"
        for i, r in enumerate(results)
    ]

    data = await llm_json(
        llm=llm,
        messages=[{
            "role": "user",
            "content": _EVAL_PROMPT.format(
                question=question,
                results_text="\n".join(results_text_lines),
            ),
        }],
        system=_EVAL_SYSTEM,
        schema_hint=_EVAL_SCHEMA,
        max_tokens=256,
        temperature=0.0,
        dsap_enabled=dsap_enabled,
        fallback={"evaluations": []},
    )

    score_map: dict[int, float] = {}
    for item in data.get("evaluations", []):
        idx = item.get("index")
        score = item.get("relevance_score", 0.5)
        if isinstance(idx, int) and isinstance(score, (int, float)):
            score_map[idx] = float(max(0.0, min(1.0, score)))

    return score_map


# ── CRAG Step 2: Query-level verdict ─────────────────────────────────────────

def _compute_query_verdict(scores: list[float]) -> str:
    """
    Determines CORRECT / AMBIGUOUS / INCORRECT based on the highest doc score.
    """
    if not scores:
        return "INCORRECT"
    max_score = max(scores)
    if max_score >= CRAG_CORRECT_THRESHOLD:
        return "CORRECT"
    if max_score < CRAG_INCORRECT_THRESHOLD:
        return "INCORRECT"
    return "AMBIGUOUS"


async def _readjudicate_ambiguous(
    question: str,
    results: list,
    local_scores: list[float],
    cloud_llm: LLMProvider,
    dsap_enabled: bool,
) -> tuple[str, list[float]]:
    """
    Phase B-2: Cloud re-adjudication for AMBIGUOUS verdicts.

    When the local LLM (e.g. qwen3:8b) produces scores in the ambiguous band
    (CRAG_INCORRECT_THRESHOLD ≤ max < CRAG_CORRECT_THRESHOLD), the cloud LLM
    makes a second judgment on the same titles/summaries.  If the cloud verdict
    resolves to CORRECT or INCORRECT, that verdict and its scores are returned.
    If the cloud also returns AMBIGUOUS, the original local scores are kept and
    the verdict stays AMBIGUOUS.

    Privacy note: only result titles and 150-char summaries are sent — no
    fetched full-page content, no original user query beyond the sub-question.
    """
    cloud_score_map = await _evaluate_results(
        question, results, cloud_llm, dsap_enabled=dsap_enabled
    )
    cloud_scores = [cloud_score_map.get(i, local_scores[i]) for i in range(len(results))]
    cloud_verdict = _compute_query_verdict(cloud_scores)

    if cloud_verdict != "AMBIGUOUS":
        return cloud_verdict, cloud_scores
    # Both local and cloud say AMBIGUOUS — trust local scores, keep AMBIGUOUS
    return "AMBIGUOUS", local_scores


# ── CRAG Step 3: Decompose-then-Recompose ────────────────────────────────────

_DRC_SYSTEM = "You are an expert at filtering and extracting relevant information from documents."

_DRC_PROMPT = """Research question: {question}

Document content:
{content}

This document is relevant to the research question. Please:
1. Split the content into strips of approximately 2 sentences each
2. Score each strip's relevance to the research question (0.0–1.0)
3. Assemble a filtered_excerpt from strips with relevance ≥ 0.5 only

Respond ONLY in this JSON format:
{{"strips": [{{"text": "First 2 sentences.", "relevance": 0.9}}, {{"text": "Unrelated sentence.", "relevance": 0.1}}], "filtered_excerpt": "Concatenated high-relevance strips here."}}"""

_DRC_SCHEMA = '{"strips": [{"text": "...", "relevance": 0.0}], "filtered_excerpt": "..."}'

_STRIP_RELEVANCE_THRESHOLD = 0.5


def _split_into_strips(text: str) -> list[str]:
    """Split text into 2-sentence strips using regex."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    # Group into 2-sentence units
    strips = []
    for i in range(0, len(sentences), 2):
        chunk = " ".join(sentences[i:i + 2])
        if chunk:
            strips.append(chunk)
    return strips


async def _decompose_recompose(
    question: str,
    content: str,
    llm: LLMProvider,
    dsap_enabled: bool = True,
) -> tuple[str, float]:
    """
    CRAG Decompose-then-Recompose for CORRECT documents (A2 batch approach).
    Returns: (filtered_excerpt, strip_retention_ratio)
    """
    data = await llm_json(
        llm=llm,
        messages=[{
            "role": "user",
            "content": _DRC_PROMPT.format(
                question=question,
                content=content[:2500],
            ),
        }],
        system=_DRC_SYSTEM,
        schema_hint=_DRC_SCHEMA,
        max_tokens=600,
        temperature=0.0,
        dsap_enabled=dsap_enabled,
        fallback={"strips": [], "filtered_excerpt": content[:500]},
    )

    strips = data.get("strips", [])
    filtered_excerpt = data.get("filtered_excerpt", "").strip()

    # Compute retention ratio
    total = len(strips)
    retained = sum(1 for s in strips if s.get("relevance", 0) >= _STRIP_RELEVANCE_THRESHOLD)
    retention_ratio = retained / total if total > 0 else 1.0

    # Fallback: if filtered_excerpt empty, use raw content truncated
    if not filtered_excerpt and strips:
        filtered_excerpt = " ".join(
            s["text"] for s in strips
            if s.get("relevance", 0) >= _STRIP_RELEVANCE_THRESHOLD
        )
    if not filtered_excerpt:
        filtered_excerpt = content[:500]

    return filtered_excerpt, retention_ratio


# ── Standard extraction (AMBIGUOUS / fallback) ────────────────────────────────

_EXTRACT_SYSTEM = """You are an expert in information extraction.
Extract only the key content relevant to the research question from the provided web page content."""

_EXTRACT_PROMPT = """Research question: {question}

Web page content:
{content}

Summarize the key information relevant to the research question in 200 characters or less.
If there is no relevant content, respond only with "No relevant content found"."""


async def _extract_content(
    question: str,
    content: str,
    llm: LLMProvider,
) -> str:
    return await llm.complete(
        messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(
            question=question, content=content[:2000]
        )}],
        system=_EXTRACT_SYSTEM,
        max_tokens=300,
        temperature=0.1,
    )


# ── MASS-RAG: 3-agent parallel + Synthesis ───────────────────────────────────

_MASS_SUMMARIZER_SYSTEM = """You are an expert research summarizer.
Your task is to provide a focused, accurate summary of documents relevant to a research question."""

_MASS_SUMMARIZER_PROMPT = """Research question: {question}

Source documents:
{doc_pool}

Summarize the key information from these documents directly relevant to the research question.
Be concise (under 200 words). Focus on facts and specifics, not generalities.
Respond with plain text only."""

_MASS_EXTRACTOR_SYSTEM = """You are a key-span extraction specialist.
Your task is to identify and extract the most informative text spans from source documents."""

_MASS_EXTRACTOR_PROMPT = """Research question: {question}

Source documents:
{doc_pool}

Extract the key information spans directly relevant to the research question.
For each span, identify the source citation ID(s) and type.
Respond ONLY in this JSON format:
{{"key_spans": [{{"text": "exact or near-verbatim text from the document", "source_citation_ids": ["cit_abc123"], "type": "fact|definition|evidence|example"}}]}}
Extract at most 6 key spans. Prefer spans from high-confidence sources."""

_MASS_EXTRACTOR_SCHEMA = '{"key_spans": [{"text": "...", "source_citation_ids": ["..."], "type": "fact"}]}'

_MASS_REASONER_SYSTEM = """You are a research reasoning specialist.
Your task is to draw inferences and identify conclusions across multiple source documents."""

_MASS_REASONER_PROMPT = """Research question: {question}

Source documents:
{doc_pool}

Identify key inferences and conclusions that can be drawn from these documents.
Respond ONLY in this JSON format:
{{"inferences": [{{"claim": "the inference or conclusion", "supporting_evidence": "brief note on what supports this"}}]}}
Provide at most 4 inferences. Be specific and grounded in the source documents."""

_MASS_REASONER_SCHEMA = '{"inferences": [{"claim": "...", "supporting_evidence": "..."}]}'

_MASS_SYNTHESIS_SYSTEM = """You are a research synthesis expert.
Your task is to combine findings from multiple specialized analysis agents into a coherent structured output."""

_MASS_SYNTHESIS_PROMPT = """Research question: {question}

Summary analysis:
{summary}

Key spans extracted:
{key_spans_text}

Inferences drawn:
{inferences_text}

Synthesize these findings into a comprehensive structured analysis.
Respond ONLY in this JSON format:
{{"summary": "...", "key_spans": [{{"text": "...", "source_citation_ids": ["..."], "type": "fact|definition|evidence|example"}}], "inferences": [{{"claim": "...", "supporting_span_indices": [0, 1]}}]}}"""

_MASS_SYNTHESIS_SCHEMA = '{"summary": "...", "key_spans": [{"text": "...", "source_citation_ids": ["..."], "type": "fact"}], "inferences": [{"claim": "...", "supporting_span_indices": [0]}]}'

# ── MASS-RAG Phase D: Local Refiner (M8) ─────────────────────────────────────
# After cloud verification (M7a synthesis), the local LLM re-checks the synthesis
# against the original document pool.  It corrects hallucinations or over-generalizations
# introduced by the cloud pass and ensures citation IDs are accurate.
# This completes the full Speculative RAG cycle: local draft → cloud verify → local refine.

_MASS_REFINER_SYSTEM = """You are a research quality auditor.
Your task is to verify a synthesis against its source documents and correct any errors."""

_MASS_REFINER_PROMPT = """Research question: {question}

Cloud synthesis to review:
Summary: {synthesis_summary}
Key spans:
{synthesis_key_spans}
Inferences:
{synthesis_inferences}

Source documents (ground truth):
{doc_pool}

Review the synthesis for:
1. Factual accuracy — does each claim match the source documents?
2. Citation accuracy — are source_citation_ids correctly attributed?
3. Completeness — are important facts from the documents missing from the synthesis?

Correct any errors, update citation IDs where needed, and return the refined synthesis.
If the synthesis is already accurate, return it unchanged.

Respond ONLY in this JSON format:
{{"summary": "...", "key_spans": [{{"text": "...", "source_citation_ids": ["..."], "type": "fact|definition|evidence|example"}}], "inferences": [{{"claim": "...", "supporting_span_indices": [0]}}]}}"""

_MASS_REFINER_SCHEMA = '{"summary": "...", "key_spans": [{"text": "...", "source_citation_ids": ["..."], "type": "fact"}], "inferences": [{"claim": "...", "supporting_span_indices": [0]}]}'


def _format_doc_pool(docs: list[dict]) -> str:
    """
    Format doc pool for MASS-RAG agents with confidence metadata (M4c').
    Confidence label allows agents to weight AMBIGUOUS docs appropriately.
    """
    lines = []
    for doc in docs:
        confidence = doc.get("confidence", 0.5)
        conf_label = "high" if confidence >= 0.7 else "medium" if confidence >= 0.4 else "low"
        lines.append(
            f"[{doc['id']}] {doc['title']} (confidence: {conf_label}={confidence:.2f})\n"
            f"{doc['excerpt'][:300]}"
        )
    return "\n\n".join(lines)


async def _mass_rag_analyze(
    question: str,
    doc_pool: list[dict],
    llm: LLMProvider,
    dsap_enabled: bool,
    error_sink: list | None = None,
    synthesis_llm: LLMProvider | None = None,
    refiner_llm: LLMProvider | None = None,
) -> dict:
    """
    MASS-RAG: 3-agent parallel analysis + Synthesis + optional Refiner (M6a + M7a + M8).

    M6a: Summarizer, Extractor, Reasoner run via asyncio.gather.
    M7a: Synthesis is a separate 4th call that combines all agent outputs.
    M8 (Phase D): Local refiner re-checks cloud synthesis against source docs.
    M5b: DSAP Level 1+2 on structured calls; degraded fallback if synthesis fails.

    synthesis_llm: when provided (Phase B-2), M7a synthesis uses this provider.
      In HybridProvider mode this is the cloud LLM (verifier role).
      Falls back to `llm` when None (backward-compatible single-provider mode).

    refiner_llm: when provided (Phase D), a local refiner (M8) re-checks the cloud
      synthesis against the original doc_pool and corrects any hallucinations or
      citation errors.  Completes the full Spec RAG cycle:
        local draft (M6a) → cloud verify (M7a) → local refine (M8).
      Falls back to skipping M8 when None (Phase B-2 behavior unchanged).

    error_sink: if provided, DSAP stagnation events from all 3 structured agents
      are appended. Caller merges this into retrieval_quality entry for observability.
    """
    if not doc_pool:
        return {"summary": "", "key_spans": [], "inferences": []}

    content_block = _format_doc_pool(doc_pool)

    # M6a: 3 agents in parallel
    # Summarizer is plain text — not a JSON call, DSAP not applicable.
    summarizer_coro = llm.complete(
        messages=[{"role": "user", "content": _MASS_SUMMARIZER_PROMPT.format(
            question=question, doc_pool=content_block
        )}],
        system=_MASS_SUMMARIZER_SYSTEM,
        max_tokens=200,
        temperature=0.1,
    )
    extractor_coro = llm_json(
        llm=llm,
        messages=[{"role": "user", "content": _MASS_EXTRACTOR_PROMPT.format(
            question=question, doc_pool=content_block
        )}],
        system=_MASS_EXTRACTOR_SYSTEM,
        schema_hint=_MASS_EXTRACTOR_SCHEMA,
        max_tokens=400,
        temperature=0.0,
        dsap_enabled=dsap_enabled,
        fallback={"key_spans": []},
        error_sink=error_sink,
        caller_tag="mass_rag/extractor",
    )
    reasoner_coro = llm_json(
        llm=llm,
        messages=[{"role": "user", "content": _MASS_REASONER_PROMPT.format(
            question=question, doc_pool=content_block
        )}],
        system=_MASS_REASONER_SYSTEM,
        schema_hint=_MASS_REASONER_SCHEMA,
        max_tokens=300,
        temperature=0.1,
        dsap_enabled=dsap_enabled,
        fallback={"inferences": []},
        error_sink=error_sink,
        caller_tag="mass_rag/reasoner",
    )

    summary_text, extractor_out, reasoner_out = await asyncio.gather(
        summarizer_coro, extractor_coro, reasoner_coro
    )

    key_spans = extractor_out.get("key_spans", [])
    inferences = reasoner_out.get("inferences", [])

    key_spans_text = "\n".join(
        f"- [{i}] ({s.get('type', 'fact')}) {s.get('text', '')[:150]}"
        for i, s in enumerate(key_spans)
    ) or "(none)"
    inferences_text = "\n".join(
        f"- {inf.get('claim', '')}: {inf.get('supporting_evidence', '')[:100]}"
        for inf in inferences
    ) or "(none)"

    # M7a: separate Synthesis call (highest priority — failure triggers writer fallback).
    # Spec RAG: synthesis_llm (cloud) is the verifier; llm (local) drafted the 3-agent pool.
    synth_llm = synthesis_llm if synthesis_llm is not None else llm
    synthesis_out = await llm_json(
        llm=synth_llm,
        messages=[{"role": "user", "content": _MASS_SYNTHESIS_PROMPT.format(
            question=question,
            summary=summary_text or "(none)",
            key_spans_text=key_spans_text,
            inferences_text=inferences_text,
        )}],
        system=_MASS_SYNTHESIS_SYSTEM,
        schema_hint=_MASS_SYNTHESIS_SCHEMA,
        max_tokens=500,
        temperature=0.0,
        dsap_enabled=dsap_enabled,
        fallback={
            "summary": summary_text or "",
            "key_spans": key_spans,
            "inferences": [{"claim": inf.get("claim", ""), "supporting_span_indices": []}
                           for inf in inferences],
        },
        error_sink=error_sink,
        caller_tag="mass_rag/synthesis",
    )

    # M8 (Phase D): local refiner re-checks cloud synthesis against source docs.
    # Runs only when refiner_llm is provided (HybridProvider mode).
    # Skipped gracefully in single-provider mode (synthesis_out returned as-is).
    if refiner_llm is not None:
        synth_key_spans_text = "\n".join(
            f"- [{i}] ({s.get('type', 'fact')}) {s.get('text', '')[:150]}"
            for i, s in enumerate(synthesis_out.get("key_spans", []))
        ) or "(none)"
        synth_inferences_text = "\n".join(
            f"- {inf.get('claim', '')}"
            for inf in synthesis_out.get("inferences", [])
        ) or "(none)"

        refined_out = await llm_json(
            llm=refiner_llm,
            messages=[{"role": "user", "content": _MASS_REFINER_PROMPT.format(
                question=question,
                synthesis_summary=synthesis_out.get("summary", ""),
                synthesis_key_spans=synth_key_spans_text,
                synthesis_inferences=synth_inferences_text,
                doc_pool=content_block,
            )}],
            system=_MASS_REFINER_SYSTEM,
            schema_hint=_MASS_REFINER_SCHEMA,
            max_tokens=500,
            temperature=0.0,
            dsap_enabled=dsap_enabled,
            fallback=synthesis_out,  # on failure: keep cloud synthesis unchanged
            error_sink=error_sink,
            caller_tag="mass_rag/refiner",
        )
        return refined_out

    return synthesis_out


# ── Main search_worker ────────────────────────────────────────────────────────

async def search_worker(
    state: dict,   # {"sub_query": dict, "original_query": str, "depth": str, "feature_flags": dict}
    llm: LLMProvider,
    search_tool: SearchTool,
) -> dict:
    """
    Single sub-query parallel worker.
    Called via Send API — state is a plain dict, not the full ResearchState.

    Returns {"citations": list[dict], "retrieval_quality": list[dict], "mass_rag_outputs": list[dict]}
    """
    sub_query  = state["sub_query"]
    question   = sub_query["question"]
    depth      = state.get("depth", "normal")
    params     = _DEPTH_PARAMS.get(depth, _DEPTH_PARAMS["normal"])
    flags      = state.get("feature_flags", {})
    crag_on      = flags.get("crag", True)
    dsap_on      = flags.get("dsap", True)
    mass_rag_on  = flags.get("mass_rag", False)
    construct_on = flags.get("construct", False)

    # Phase B-2: HybridProvider split.
    # local_llm — CRAG evaluator (Retrieval Evaluator + DRC + extract_content) and
    #             MASS-RAG 3-agent drafters (Summarizer / Extractor / Reasoner).
    # cloud_llm — CRAG AMBIGUOUS re-adjudication and MASS-RAG Synthesis verifier.
    # Graceful degradation: plain providers use same llm for both roles.
    local_llm: LLMProvider = llm.local if hasattr(llm, "local") else llm
    cloud_llm: LLMProvider = llm.cloud if hasattr(llm, "cloud") else llm

    injection_filter = InjectionFilter()
    new_citations: list[dict] = []

    results = await search_tool.search_async(query=question, max_results=params["max_results"])
    if not results:
        return {
            "citations": [],
            "retrieval_quality": [{
                "sub_query_id": sub_query["id"],
                "verdict": "INCORRECT",
                "max_doc_score": 0.0,
                "strip_retention_ratio": 0.0,
            }],
            "mass_rag_outputs": [],
        }

    # ── CRAG: evaluate + decide query-level verdict ───────────────────────────
    if crag_on:
        # Step 1: local LLM scores all results (cheap, no cloud quota consumed)
        score_map = await _evaluate_results(question, results, local_llm, dsap_enabled=dsap_on)
        scores = [score_map.get(i, 0.5) for i in range(len(results))]
        query_verdict = _compute_query_verdict(scores)
        max_doc_score = max(scores) if scores else 0.0

        # Step 2 (Phase B-2): cloud re-adjudication for AMBIGUOUS band only.
        # Sends titles + 150-char summaries — no full-page content, no raw query.
        if query_verdict == "AMBIGUOUS":
            query_verdict, scores = await _readjudicate_ambiguous(
                question, results, scores, cloud_llm, dsap_on
            )
            score_map = {i: scores[i] for i in range(len(scores))}
            max_doc_score = max(scores) if scores else 0.0
    else:
        # crag=False: treat everything as CORRECT with score=0.7
        score_map = {i: 0.7 for i in range(len(results))}
        scores = [0.7] * len(results)
        query_verdict = "CORRECT"
        max_doc_score = 0.7

    strip_retention_ratio = 1.0  # updated below for CORRECT docs
    fetched_count = 0
    max_fetch = params["max_fetch"]

    # ── Process documents based on query verdict ──────────────────────────────

    if query_verdict == "INCORRECT":
        # INCORRECT: take top-2 summaries at low confidence, emit signal for gap_detector
        for result in results[:2]:
            filter_result = injection_filter.check(result.summary, result.url)
            cid = f"cit_{uuid.uuid4().hex[:8]}"
            new_citations.append(Citation(
                id=cid,
                url=result.url,
                title=result.title,
                excerpt=filter_result.sanitized_content,
                source_type=SourceType.WEB,
                trust_level=TrustLevel(filter_result.trust_level),
                crawled_at=datetime.now(timezone.utc).isoformat(),
                confidence=max_doc_score * 0.3,
                injection_checked=True,
            ).to_dict())

    else:
        # CORRECT or AMBIGUOUS: process documents
        retention_ratios: list[float] = []

        for i, result in enumerate(results):
            doc_score = score_map.get(i, 0.5)

            # Low-score doc: summary only (regardless of query verdict)
            if doc_score < CRAG_INCORRECT_THRESHOLD:
                filter_result = injection_filter.check(result.summary, result.url)
                cid = f"cit_{uuid.uuid4().hex[:8]}"
                new_citations.append(Citation(
                    id=cid,
                    url=result.url,
                    title=result.title,
                    excerpt=filter_result.sanitized_content,
                    source_type=SourceType.WEB,
                    trust_level=TrustLevel(filter_result.trust_level),
                    crawled_at=datetime.now(timezone.utc).isoformat(),
                    confidence=result.relevance_score * doc_score * 0.6,
                    injection_checked=True,
                ).to_dict())
                continue

            # Low Tavily score or fetch budget exhausted → summary only
            if result.relevance_score < _FETCH_THRESHOLD or fetched_count >= max_fetch:
                filter_result = injection_filter.check(result.summary, result.url)
                cid = f"cit_{uuid.uuid4().hex[:8]}"
                new_citations.append(Citation(
                    id=cid,
                    url=result.url,
                    title=result.title,
                    excerpt=filter_result.sanitized_content,
                    source_type=SourceType.WEB,
                    trust_level=TrustLevel(filter_result.trust_level),
                    crawled_at=datetime.now(timezone.utc).isoformat(),
                    confidence=result.relevance_score * doc_score * 0.6,
                    injection_checked=True,
                ).to_dict())
                continue

            # Full fetch
            fetch_result = await search_tool.fetch_page_async(result.url)
            content = fetch_result.content if fetch_result.success else result.summary
            filter_result = injection_filter.check(content, result.url)
            fetched_count += 1

            # ── CORRECT: Decompose-then-Recompose (local LLM) ────────────────
            if query_verdict == "CORRECT" and crag_on:
                excerpt, retention = await _decompose_recompose(
                    question=question,
                    content=filter_result.sanitized_content,
                    llm=local_llm,
                    dsap_enabled=dsap_on,
                )
                retention_ratios.append(retention)
                confidence = result.relevance_score * doc_score * (0.8 + 0.2 * retention)

            # ── AMBIGUOUS: standard extraction (local LLM) ────────────────────
            else:
                excerpt = await _extract_content(
                    question=question,
                    content=filter_result.sanitized_content,
                    llm=local_llm,
                )
                if "No relevant content found" in excerpt:
                    excerpt = filter_result.sanitized_content[:300]
                    confidence = result.relevance_score * doc_score * 0.4
                else:
                    excerpt = excerpt.strip()
                    confidence = result.relevance_score * doc_score

            cid = f"cit_{uuid.uuid4().hex[:8]}"
            new_citations.append(Citation(
                id=cid,
                url=result.url,
                title=result.title,
                excerpt=excerpt,
                source_type=SourceType.WEB,
                trust_level=TrustLevel(filter_result.trust_level),
                crawled_at=datetime.now(timezone.utc).isoformat(),
                confidence=confidence,
                injection_checked=True,
            ).to_dict())

        if retention_ratios:
            strip_retention_ratio = sum(retention_ratios) / len(retention_ratios)

    # ── MASS-RAG: 3-agent parallel analysis on doc pool ──────────────────────
    # Runs only for CORRECT/AMBIGUOUS verdicts when enabled.
    # fast depth skipped entirely (M4: cost vs. quality trade-off).
    mass_rag_entry: list[dict] = []
    mass_rag_errors: list[dict] = []  # DSAP Level 2 stagnation events from this sub-query
    if mass_rag_on and depth != "fast" and query_verdict in ("CORRECT", "AMBIGUOUS"):
        doc_pool = [
            {
                "id": c["id"],
                "title": c["title"],
                "excerpt": c["excerpt"],
                "confidence": c["confidence"],
            }
            for c in new_citations
        ]
        if doc_pool:
            synthesis = await _mass_rag_analyze(
                question=question,
                doc_pool=doc_pool,
                llm=local_llm,              # M6a: 3-agent drafters run locally
                dsap_enabled=dsap_on,
                error_sink=mass_rag_errors,
                synthesis_llm=cloud_llm,    # M7a: cloud verifier (Spec RAG Phase B-2)
                refiner_llm=local_llm,      # M8: local refiner (Spec RAG Phase D)
            )
            entry: dict = {
                "sub_query_id": sub_query["id"],
                "question": question,
                "summary": synthesis.get("summary", ""),
                "key_spans": synthesis.get("key_spans", []),
                "inferences": synthesis.get("inferences", []),
            }
            # C3a: CONSTRUCT field-level trustworthiness scoring
            if construct_on:
                trust_scores = await score_mass_rag_output(entry, local_llm, dsap_enabled=dsap_on)
                entry["trust_scores"] = trust_scores
            mass_rag_entry = [entry]

    # ── Retrieval quality signal ──────────────────────────────────────────────
    retrieval_quality_entry = {
        "sub_query_id":          sub_query["id"],
        "verdict":               query_verdict,
        "max_doc_score":         round(max_doc_score, 3),
        "strip_retention_ratio": round(strip_retention_ratio, 3),
        "dsap_errors":           mass_rag_errors,  # empty list if no stagnation events
    }

    return {
        "citations":         new_citations,
        "retrieval_quality": [retrieval_quality_entry],
        "mass_rag_outputs":  mass_rag_entry,
    }
