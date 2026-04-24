"""
Deep Research Agent — Component Benchmark (Layer 1)

Tests each active technique in isolation with fixed inputs.
No web search API calls required — all inputs are embedded fixtures.

Active techniques: query_decomp, crag, dsap, alignrag, stride, mass_rag (6 total)
Removed techniques (not testable): cure, auto_search, sdp, speculative_rag, construct

Usage:
  cd deep_research_agent
  source venv/bin/activate
  python -m eval.component_benchmark --provider bedrock
  python -m eval.component_benchmark --provider ollama --model qwen3:8b

Output:
  eval/results/component_bedrock.json
  eval/results/component_ollama.json
"""

from __future__ import annotations
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures — fixed inputs for reproducible measurement
# ══════════════════════════════════════════════════════════════════════════════

# ── CRAG fixtures ─────────────────────────────────────────────────────────────
# Gold labels: which results are relevant/partial/irrelevant for the query
CRAG_QUERY = "transformer architecture self-attention mechanism advantages over RNN"
CRAG_RESULTS = [
    # relevant (directly answer the query)
    {"title": "Attention Is All You Need",
     "summary": "Transformers use self-attention to process all positions simultaneously, eliminating sequential dependency in RNNs, enabling full parallelization and capturing long-range dependencies.",
     "relevance_score": 0.92, "gold": "relevant"},
    {"title": "BERT Pre-training of Deep Bidirectional Transformers",
     "summary": "BERT uses transformer encoder with bidirectional self-attention, outperforming RNN-based models on 11 NLP benchmarks including GLUE, achieving state-of-the-art with parallelizable training.",
     "relevance_score": 0.88, "gold": "relevant"},
    {"title": "Transformer vs RNN: Training Speed Comparison",
     "summary": "Transformers train 3-8x faster than LSTMs on modern GPUs due to parallelism. Self-attention complexity is O(n^2) vs O(n) for RNNs, but GPU parallelism advantages dominate for typical sequence lengths.",
     "relevance_score": 0.85, "gold": "relevant"},
    # partial (related but not directly answering)
    {"title": "GPT-4 Technical Report",
     "summary": "GPT-4 is a large multimodal model achieving human-level performance on various benchmarks. Built on transformer architecture with reinforcement learning from human feedback.",
     "relevance_score": 0.72, "gold": "partial"},
    {"title": "Vision Transformer ViT: Image Recognition",
     "summary": "Applying transformer directly to image patches achieves strong performance on ImageNet. Demonstrates self-attention generalizes beyond NLP to vision tasks with sufficient data.",
     "relevance_score": 0.68, "gold": "partial"},
    {"title": "Long-Range Arena Benchmark for Efficient Transformers",
     "summary": "Benchmark evaluating transformers on tasks requiring long-range dependencies. Various efficient attention variants trade accuracy for speed.",
     "relevance_score": 0.65, "gold": "partial"},
    # irrelevant
    {"title": "Stock Market Prediction Using Machine Learning 2024",
     "summary": "Comparison of ML algorithms for stock price prediction including LSTM and gradient boosting. LSTM achieves 67% directional accuracy on S&P 500.",
     "relevance_score": 0.41, "gold": "irrelevant"},
    {"title": "Python Web Scraping Best Practices",
     "summary": "Guide to ethical web scraping with BeautifulSoup and Scrapy. Covers rate limiting, robots.txt compliance, and proxy rotation.",
     "relevance_score": 0.22, "gold": "irrelevant"},
    {"title": "Database Indexing and Query Optimization",
     "summary": "B-tree vs hash indexes for PostgreSQL. Query plan analysis, index coverage, and partial index strategies for high-performance SQL.",
     "relevance_score": 0.18, "gold": "irrelevant"},
    {"title": "JavaScript Framework Comparison 2024",
     "summary": "React vs Vue vs Angular performance benchmarks. Component rendering speed, bundle size, and developer experience analysis.",
     "relevance_score": 0.15, "gold": "irrelevant"},
]

# ── AlignRAG fixtures ──────────────────────────────────────────────────────────
# Report contains 3 injected misaligned claims (amplification, distortion, unsupported inference)
ALIGNRAG_CITATIONS = [
    {"id": "c1", "title": "IBM Quantum Processor 2024 Benchmark",
     "excerpt": "IBM's Eagle processor achieved 99.5% two-qubit gate fidelity across 127 qubits, a significant engineering milestone for superconducting qubits."},
    {"id": "c2", "title": "Quantum vs Classical Computing Performance Study",
     "excerpt": "Quantum advantage has been demonstrated for specific combinatorial optimization problems, showing 50-100x speedup over classical heuristics for those narrow problem classes."},
    {"id": "c3", "title": "Quantum Decoherence Progress Report",
     "excerpt": "Qubit coherence times have improved from microseconds to milliseconds over the past decade. Decoherence remains the primary obstacle to fault-tolerant quantum computing."},
    {"id": "c4", "title": "Quantum Error Correction Status",
     "excerpt": "Surface codes require approximately 1,000 physical qubits per logical qubit for fault-tolerant operation. Current systems have 100-1000 physical qubits total."},
]
# 3 injected errors in this draft:
#   Error 1 (amplification): "1,000x speedup" — source says "50-100x"
#   Error 2 (distortion): "completely eliminated decoherence" — source says "remains the primary obstacle"
#   Error 3 (unsupported inference): "ready for commercial deployment" — source says 1000 physical qubits needed, current systems only have 100-1000 total
ALIGNRAG_DRAFT_WITH_ERRORS = """# Quantum Computing: Current State

## Executive Summary

IBM's quantum processor achieved 99.5% two-qubit gate fidelity [Source 1], marking a
major milestone in the field. Recent benchmarks show quantum computers deliver a 1,000x
speedup over classical computers for optimization problems [Source 2]. Scientists have
completely eliminated decoherence through advanced qubit engineering techniques [Source 3],
making quantum computers ready for commercial deployment across industries [Source 4].
"""
# Gold: 3 errors (claims about 1000x speedup, "completely eliminated", "ready for commercial deployment")
ALIGNRAG_GOLD_ERROR_COUNT = 3


# ── MASS-RAG fixtures ─────────────────────────────────────────────────────────
MASS_RAG_QUERY = "quantum decoherence mechanisms and error correction approaches"
MASS_RAG_DIMENSION = "Cause/Mechanism"
MASS_RAG_CONTENT = """
Quantum decoherence occurs when a quantum system interacts with its environment,
causing the superposition of quantum states to collapse into classical states.
The interaction rate is characterized by the decoherence time T2, which represents
how long a qubit can maintain coherence. In superconducting qubits, T2 ranges from
microseconds to milliseconds at millikelvin temperatures.

The primary mechanisms of decoherence include:
1. Phonon interactions: lattice vibrations in the substrate couple to the qubit
2. Two-level systems (TLS): material defects at interfaces create fluctuating dipoles
3. Magnetic flux noise: 1/f noise from magnetic impurities affects flux qubits
4. Charge noise: charge fluctuations near Josephson junctions

Surface code error correction addresses decoherence by encoding one logical qubit
in ~1,000 physical qubits. The code can correct errors as long as the physical
error rate stays below the threshold (~1%). Current systems achieve 0.1-0.5% error
rates, approaching but not yet reliably below threshold.

Key metrics tracked by researchers:
- Gate fidelity (target: >99.9%)
- T1 (energy relaxation time)
- T2 (dephasing time)
- Readout fidelity (current best: ~99%)
"""

# ── STRIDE fixtures ────────────────────────────────────────────────────────────
STRIDE_QUERY = "impact of generative AI on software development productivity"
# Initial plan with only 2 of 5 dimensions covered
STRIDE_INITIAL_SUB_QUERIES = [
    {"id": "sq1", "question": "How has generative AI changed software development workflows in 2024?", "dimension": "Current State/Evidence"},
    {"id": "sq2", "question": "What are the main limitations of AI coding assistants like Copilot?", "dimension": "Limitations/Challenges"},
]
# Missing: Definition/Background, Comparison/Alternatives, Cause/Mechanism

# ── Query Decomposition fixtures ───────────────────────────────────────────────
DECOMP_QUERIES = [
    "Current state of large language models in 2025",
    "Climate change mitigation strategies comparison",
    "Quantum computing commercialization challenges",
]
ALL_FIVE_DIMENSIONS = ["Definition/Background", "Current State/Evidence",
                       "Comparison/Alternatives", "Cause/Mechanism", "Limitations/Challenges"]

# ── DSAP fixtures ──────────────────────────────────────────────────────────────
DSAP_PROMPTS = [
    {
        "name": "nested_plan_json",
        "prompt": """Create a research plan JSON for "quantum computing state 2025".
Output ONLY valid JSON matching this exact schema:
{
  "intent": "analytical",
  "sub_queries": [
    {"id": "sq1", "question": "...", "dimension": "..."},
    {"id": "sq2", "question": "...", "dimension": "..."}
  ],
  "depth": "normal"
}""",
        "required_keys": ["intent", "sub_queries", "depth"],
    },
    {
        "name": "evaluation_array",
        "prompt": """Evaluate these search results for "LLM efficiency" and output ONLY valid JSON:
{"evaluations": [{"index": 0, "relevance": "relevant|partial|irrelevant"}, {"index": 1, "relevance": "relevant|partial|irrelevant"}, {"index": 2, "relevance": "relevant|partial|irrelevant"}]}
Results:
[0] FlashAttention doubles GPU throughput for transformer training
[1] Python list comprehension best practices
[2] Speculative decoding reduces LLM latency by 3x""",
        "required_keys": ["evaluations"],
    },
    {
        "name": "gap_analysis",
        "prompt": """Analyze coverage gaps and output ONLY valid JSON:
{"gaps": [{"sub_query_id": "string", "issue": "string", "gap_query": "string", "uncertainty": 0.0}], "coverage_score": 0.0, "comment": "string"}
Research question: Recent AI safety developments
Sub-queries covered: [sq1: technical AI safety approaches, sq2: AI alignment research]
Available sources: 8 total, mostly focused on sq1""",
        "required_keys": ["gaps", "coverage_score", "comment"],
    },
    {
        "name": "critic_feedback",
        "prompt": """Review this report excerpt and output ONLY valid JSON:
{"passed": false, "has_logic_errors": false, "uncited_claims": [], "unanswered_sub_queries": [], "misaligned_claims": [], "suggestions": []}
Report: "AI systems achieve perfect accuracy in all real-world applications. No AI system has ever made an error in production."
Sub-queries: [Is AI reliable in production?]""",
        "required_keys": ["passed", "has_logic_errors", "uncited_claims", "suggestions"],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Test result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TechResult:
    technique: str
    score_off: float    # score when technique is disabled
    score_on: float     # score when technique is enabled
    delta: float        # score_on - score_off
    latency_off_ms: int
    latency_on_ms: int
    details: dict = field(default_factory=dict)
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.score_on >= 0.6


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def _clean(raw: str) -> str:
    """Remove thinking tags and markdown code fences."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    return cleaned.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _parse_json(raw: str) -> dict | list | None:
    try:
        return json.loads(_clean(raw))
    except Exception:
        return None


def _dim_coverage(sub_queries: list[dict]) -> float:
    """Fraction of the 5 dimensions present in sub_queries."""
    dims = [sq.get("dimension", "") for sq in sub_queries]
    covered = sum(
        1 for d in ALL_FIVE_DIMENSIONS
        if any(d.lower() in found.lower() for found in dims)
    )
    return covered / 5.0


async def _llm_judge(llm, question: str, option_a: str, option_b: str, judge_system: str) -> float:
    """
    LLM judge: asks which of A or B better achieves the stated goal.
    Returns 1.0 (A wins), 0.5 (tie), or 0.0 (B wins — i.e. OFF condition wins).
    Higher score means ON condition is better.
    """
    prompt = f"""{question}

Option A (Technique ON):
{option_a[:600]}

Option B (Technique OFF / baseline):
{option_b[:600]}

Which option better achieves the stated goal?
Respond with ONLY a JSON object: {{"winner": "A" | "B" | "tie", "reason": "one sentence"}}"""

    raw = await llm.complete(
        messages=[{"role": "user", "content": prompt}],
        system=judge_system,
        max_tokens=128,
        temperature=0.0,
    )
    parsed = _parse_json(raw)
    if not parsed or "winner" not in parsed:
        return 0.5
    w = str(parsed["winner"]).strip().upper()
    return 1.0 if w == "A" else (0.5 if w == "TIE" else 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Individual technique tests
# ══════════════════════════════════════════════════════════════════════════════

async def test_query_decomp(llm) -> TechResult:
    """
    Metric: average dimension coverage across 3 fixed queries.
    OFF: simple decomposition (no dimension tags).
    ON: 5-dimensional decomposition.
    """
    from src.nodes.plan_generator import (
        _SYSTEM, _PROMPT, _DECOMPOSITION_DIMS, _PROMPT_SIMPLE
    )
    from src.utils.llm_json import llm_json

    scores_off, scores_on = [], []
    latencies_off, latencies_on = [], []

    for query in DECOMP_QUERIES:
        # OFF
        t0 = time.time()
        data_off = await llm_json(
            llm=llm,
            messages=[{"role": "user", "content": _PROMPT_SIMPLE.format(query=query)}],
            system=_SYSTEM, schema_hint="{}", max_tokens=1024, temperature=0.3,
            fallback={"sub_queries": []},
        )
        latencies_off.append(int((time.time() - t0) * 1000))
        scores_off.append(_dim_coverage(data_off.get("sub_queries", [])))

        # ON
        t0 = time.time()
        data_on = await llm_json(
            llm=llm,
            messages=[{"role": "user", "content": _PROMPT.format(
                query=query, decomposition_dims=_DECOMPOSITION_DIMS
            )}],
            system=_SYSTEM, schema_hint="{}", max_tokens=1024, temperature=0.3,
            fallback={"sub_queries": []},
        )
        latencies_on.append(int((time.time() - t0) * 1000))
        scores_on.append(_dim_coverage(data_on.get("sub_queries", [])))

    s_off = sum(scores_off) / len(scores_off)
    s_on = sum(scores_on) / len(scores_on)
    return TechResult(
        technique="query_decomp",
        score_off=round(s_off, 3), score_on=round(s_on, 3),
        delta=round(s_on - s_off, 3),
        latency_off_ms=sum(latencies_off) // len(latencies_off),
        latency_on_ms=sum(latencies_on) // len(latencies_on),
        details={"per_query_off": scores_off, "per_query_on": scores_on},
    )


async def test_stride(llm) -> TechResult:
    """
    Metric: dimension coverage of query decomposition with vs without STRIDE.
    OFF: generate_plan with stride=False (plain query_decomp).
    ON:  generate_plan with stride=True (meta-planner Sq → Cq flow).
    """
    from src.nodes.plan_generator import generate_plan

    _base_state = {
        "original_query": STRIDE_QUERY,
        "plan": None, "plan_approved": False, "local_search_enabled": False,
        "retrieval_quality": [], "evidence_store": [], "mass_rag_outputs": [],
        "citations": [], "draft_report": "", "critic_feedback": None,
        "final_report": "", "revision_count": 0, "research_round": 0, "gap_queries": [],
    }

    # OFF: query_decomp only, no STRIDE
    state_off = {**_base_state, "feature_flags": {"stride": False, "query_decomp": True, "dsap": True}}
    t0 = time.time()
    result_off = await generate_plan(state_off, llm)
    lat_off = int((time.time() - t0) * 1000)
    sqs_off = result_off.get("plan", {}).get("sub_queries", [])
    score_off = _dim_coverage(sqs_off)

    # ON: STRIDE enabled
    state_on = {**_base_state, "feature_flags": {"stride": True, "query_decomp": True, "dsap": True}}
    t0 = time.time()
    result_on = await generate_plan(state_on, llm)
    lat_on = int((time.time() - t0) * 1000)
    sqs_on = result_on.get("plan", {}).get("sub_queries", [])
    score_on = _dim_coverage(sqs_on)

    return TechResult(
        technique="stride",
        score_off=round(score_off, 3), score_on=round(score_on, 3),
        delta=round(score_on - score_off, 3),
        latency_off_ms=lat_off, latency_on_ms=lat_on,
        details={
            "sq_count_off": len(sqs_off),
            "sq_count_on": len(sqs_on),
            "coverage_off": round(score_off, 2),
            "coverage_on": round(score_on, 2),
            "dimensions_on": [sq.get("dimension", "") for sq in sqs_on],
        },
    )


async def test_crag(llm) -> TechResult:
    """
    Metric: precision of relevance classification against gold labels.
    OFF: all results treated as relevant (no LLM evaluation).
    ON: CRAG LLM evaluation applied.
    """
    from src.nodes.search_worker import _evaluate_results

    # Build mock SearchResult objects
    from src.tools.search import SearchResult
    mock_results = [
        SearchResult(
            url=f"https://example.com/{i}",
            title=r["title"],
            summary=r["summary"],
            relevance_score=r["relevance_score"],
        )
        for i, r in enumerate(CRAG_RESULTS)
    ]
    gold = [r["gold"] for r in CRAG_RESULTS]

    # OFF: no evaluation — all treated as "relevant"
    # Precision: relevant_correct / total_predicted_relevant
    # All predicted relevant → precision = #actually_relevant / total
    truly_relevant = sum(1 for g in gold if g == "relevant")
    score_off = truly_relevant / len(gold)  # = 3/10 = 0.30

    # ON: run CRAG evaluation
    t0 = time.time()
    relevance_map = await _evaluate_results(CRAG_QUERY, mock_results, llm)
    lat_on = int((time.time() - t0) * 1000)

    # Compute classification metrics
    pred = [relevance_map.get(i, "partial") for i in range(len(mock_results))]
    # Treat "relevant"+"partial" as positive, "irrelevant" as negative
    true_pos = sum(1 for i, g in enumerate(gold) if g in ("relevant", "partial") and pred[i] in ("relevant", "partial"))
    true_neg = sum(1 for i, g in enumerate(gold) if g == "irrelevant" and pred[i] == "irrelevant")
    false_pos = sum(1 for i, g in enumerate(gold) if g == "irrelevant" and pred[i] != "irrelevant")
    false_neg = sum(1 for i, g in enumerate(gold) if g in ("relevant", "partial") and pred[i] == "irrelevant")

    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    # Noise reduction: irrelevant correctly filtered
    noise_reduction = true_neg / max(sum(1 for g in gold if g == "irrelevant"), 1)

    score_on = round((f1 + noise_reduction) / 2, 3)

    return TechResult(
        technique="crag",
        score_off=round(score_off, 3), score_on=score_on,
        delta=round(score_on - score_off, 3),
        latency_off_ms=0, latency_on_ms=lat_on,
        details={
            "predictions": dict(zip([r["title"][:30] for r in CRAG_RESULTS], pred)),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "noise_reduction": round(noise_reduction, 3),
            "baseline_precision": round(score_off, 3),
        },
    )


async def test_dsap(llm) -> TechResult:
    """
    Metric: JSON parse success rate.
    OFF: single attempt, no error-feedback retry.
    ON: DSAP error-feedback retry (up to 2 retries).
    """
    from src.utils.llm_json import llm_json

    successes_off, successes_on = 0, 0
    latencies_off, latencies_on = [], []
    details = []

    for p in DSAP_PROMPTS:
        # OFF: single attempt
        t0 = time.time()
        result_off = await llm_json(
            llm=llm,
            messages=[{"role": "user", "content": p["prompt"]}],
            system="Output ONLY valid JSON. No explanations.",
            schema_hint=str(p["required_keys"]),
            max_tokens=512, temperature=0.0,
            max_retries=2, dsap_enabled=False,
            fallback=None,
        )
        lat_off = int((time.time() - t0) * 1000)
        latencies_off.append(lat_off)
        ok_off = result_off is not None and all(k in result_off for k in p["required_keys"])
        if ok_off:
            successes_off += 1

        # ON: with DSAP retries
        t0 = time.time()
        result_on = await llm_json(
            llm=llm,
            messages=[{"role": "user", "content": p["prompt"]}],
            system="Output ONLY valid JSON. No explanations.",
            schema_hint=str(p["required_keys"]),
            max_tokens=512, temperature=0.0,
            max_retries=2, dsap_enabled=True,
            fallback=None,
        )
        lat_on = int((time.time() - t0) * 1000)
        latencies_on.append(lat_on)
        ok_on = result_on is not None and all(k in result_on for k in p["required_keys"])
        if ok_on:
            successes_on += 1

        details.append({
            "test": p["name"],
            "off_success": ok_off,
            "on_success": ok_on,
            "lat_off_ms": lat_off,
            "lat_on_ms": lat_on,
        })

    score_off = successes_off / len(DSAP_PROMPTS)
    score_on = successes_on / len(DSAP_PROMPTS)

    return TechResult(
        technique="dsap",
        score_off=round(score_off, 3), score_on=round(score_on, 3),
        delta=round(score_on - score_off, 3),
        latency_off_ms=sum(latencies_off) // len(latencies_off),
        latency_on_ms=sum(latencies_on) // len(latencies_on),
        details={"per_test": details, "successes_off": successes_off, "successes_on": successes_on},
    )


async def test_alignrag(llm) -> TechResult:
    """
    Metric: fraction of injected misaligned claims detected.
    OFF: no citation comparison in critique — misaligned_claims should be empty.
    ON: AlignRAG citation comparison — should detect injected errors.
    """
    from src.utils.llm_json import llm_json
    from src.nodes.critic import _SYSTEM, _PROMPT, _PROMPT_NO_ALIGNRAG, _SCHEMA_HINT, _build_citations_excerpt

    sub_queries_text = "- [sq1] Current state of quantum computing performance\n- [sq2] Technical challenges in quantum error correction"
    citations_excerpt = _build_citations_excerpt(ALIGNRAG_CITATIONS, max_count=4)

    # OFF: no AlignRAG
    t0 = time.time()
    data_off = await llm_json(
        llm=llm,
        messages=[{"role": "user", "content": _PROMPT_NO_ALIGNRAG.format(
            query="Current state of quantum computing",
            sub_queries=sub_queries_text,
            draft=ALIGNRAG_DRAFT_WITH_ERRORS,
        )}],
        system=_SYSTEM, schema_hint=_SCHEMA_HINT,
        max_tokens=800, temperature=0.1, dsap_enabled=True,
        fallback={"misaligned_claims": [], "uncited_claims": [], "unanswered_sub_queries": [], "suggestions": []},
    )
    lat_off = int((time.time() - t0) * 1000)
    misaligned_off = len(data_off.get("misaligned_claims", []))

    # ON: with AlignRAG (3-phase structured output needs more tokens than old string list)
    t0 = time.time()
    data_on = await llm_json(
        llm=llm,
        messages=[{"role": "user", "content": _PROMPT.format(
            query="Current state of quantum computing",
            sub_queries=sub_queries_text,
            citation_count=len(ALIGNRAG_CITATIONS),
            citations_excerpt=citations_excerpt,
            draft=ALIGNRAG_DRAFT_WITH_ERRORS,
        )}],
        system=_SYSTEM, schema_hint=_SCHEMA_HINT,
        max_tokens=1200, temperature=0.1, dsap_enabled=True,
        fallback={"misaligned_claims": [], "uncited_claims": [], "unanswered_sub_queries": [], "suggestions": []},
    )
    lat_on = int((time.time() - t0) * 1000)
    misaligned_on = len(data_on.get("misaligned_claims", []))

    # OFF score: fraction of errors detected without AlignRAG (usually 0)
    score_off = min(misaligned_off / ALIGNRAG_GOLD_ERROR_COUNT, 1.0)
    # ON score: fraction of gold errors detected
    score_on = min(misaligned_on / ALIGNRAG_GOLD_ERROR_COUNT, 1.0)

    return TechResult(
        technique="alignrag",
        score_off=round(score_off, 3), score_on=round(score_on, 3),
        delta=round(score_on - score_off, 3),
        latency_off_ms=lat_off, latency_on_ms=lat_on,
        details={
            "gold_error_count": ALIGNRAG_GOLD_ERROR_COUNT,
            "detected_off": misaligned_off,
            "detected_on": misaligned_on,
            "misaligned_claims_on": data_on.get("misaligned_claims", []),
        },
    )



async def test_mass_rag(llm) -> TechResult:
    """
    Metric: LLM judge evaluates whether MASS-RAG multi-agent synthesis is more domain-specific.
    OFF: generic single-call extraction (_extract_content).
    ON:  MASS-RAG 3-agent parallel analysis + synthesis (_mass_rag_analyze).
    """
    from src.nodes.search_worker import _extract_content, _mass_rag_analyze

    # OFF: generic single extraction
    t0 = time.time()
    excerpt_off = await _extract_content(
        question=MASS_RAG_QUERY,
        content=MASS_RAG_CONTENT,
        llm=llm,
    )
    lat_off = int((time.time() - t0) * 1000)

    # ON: full MASS-RAG 3-agent pipeline
    doc_pool = [{"id": "d1", "title": "Quantum Decoherence Study",
                 "excerpt": MASS_RAG_CONTENT, "confidence": 0.9}]
    t0 = time.time()
    mass_result = await _mass_rag_analyze(
        question=MASS_RAG_QUERY,
        doc_pool=doc_pool,
        llm=llm,
        dsap_enabled=True,
    )
    lat_on = int((time.time() - t0) * 1000)
    excerpt_on = mass_result.get("synthesis") or mass_result.get("summary", "") or str(mass_result)[:500]

    # LLM judge
    t0 = time.time()
    judge_score = await _llm_judge(
        llm=llm,
        question="Which excerpt better captures the causal mechanisms and technical specifics of quantum decoherence for a researcher studying [Cause/Mechanism]?",
        option_a=excerpt_on,   # specialist (ON)
        option_b=excerpt_off,  # generic (OFF)
        judge_system="You are an expert in research quality evaluation. Judge based on technical depth, mechanism specificity, and research value.",
    )
    lat_judge = int((time.time() - t0) * 1000)

    # Also measure technical term density (terms like T1, T2, phonon, TLS, fidelity, etc.)
    tech_terms = ["T1", "T2", "coherence", "decoherence", "phonon", "qubit", "fidelity",
                  "TLS", "surface code", "error rate", "gate fidelity", "threshold"]
    density_off = sum(1 for t in tech_terms if t.lower() in excerpt_off.lower()) / len(tech_terms)
    density_on = sum(1 for t in tech_terms if t.lower() in excerpt_on.lower()) / len(tech_terms)

    score_off = round(density_off * 0.5 + 0.25, 3)  # baseline: some terms + 0.25 neutral
    score_on = round(judge_score * 0.6 + density_on * 0.4, 3)

    return TechResult(
        technique="mass_rag",
        score_off=score_off, score_on=score_on,
        delta=round(score_on - score_off, 3),
        latency_off_ms=lat_off, latency_on_ms=lat_on + lat_judge,
        details={
            "excerpt_off": excerpt_off[:200],
            "excerpt_on": excerpt_on[:200],
            "tech_density_off": round(density_off, 3),
            "tech_density_on": round(density_on, 3),
            "judge_score": judge_score,
        },
    )



# ══════════════════════════════════════════════════════════════════════════════
# Runner + reporting
# ══════════════════════════════════════════════════════════════════════════════

TESTS = [
    ("query_decomp",   test_query_decomp),
    ("crag",           test_crag),
    ("dsap",           test_dsap),
    ("alignrag",       test_alignrag),
    ("stride",         test_stride),
    ("mass_rag",       test_mass_rag),
]


def _bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def print_report(results: list[TechResult], provider: str) -> None:
    print(f"\n{'═'*72}")
    print(f"  Component Benchmark — {provider}")
    print(f"{'═'*72}")
    print(f"  {'Technique':<18} {'OFF':>6} {'ON':>6} {'Δ':>7}  {'Visual (ON)':22}  Status")
    print(f"  {'─'*18} {'─'*6} {'─'*6} {'─'*7}  {'─'*22}  {'─'*6}")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        delta_str = f"+{r.delta:.2f}" if r.delta > 0 else f"{r.delta:.2f}"
        note = f"  ERR: {r.error[:35]}" if r.error else ""
        print(f"  {r.technique:<18} {r.score_off:>6.2f} {r.score_on:>6.2f} {delta_str:>7}  "
              f"{_bar(r.score_on):<22}  {status}{note}")

    valid = [r for r in results if not r.error]
    if valid:
        avg_off = sum(r.score_off for r in valid) / len(valid)
        avg_on = sum(r.score_on for r in valid) / len(valid)
        avg_delta = sum(r.delta for r in valid) / len(valid)
        pass_count = sum(1 for r in results if r.passed)
        print(f"  {'─'*18} {'─'*6} {'─'*6} {'─'*7}  {'─'*22}  {'─'*6}")
        print(f"  {'AVERAGE':<18} {avg_off:>6.2f} {avg_on:>6.2f} {f'+{avg_delta:.2f}':>7}  "
              f"{'':22}  {pass_count}/{len(results)} PASS")
    print(f"{'═'*72}\n")


async def run_all_tests(llm, provider: str) -> list[TechResult]:
    results = []
    print(f"\nRunning component benchmark for: {provider}")
    print(f"{'─'*50}")

    for name, test_fn in TESTS:
        print(f"  Testing {name:<20}", end="", flush=True)
        try:
            result = await test_fn(llm)
            results.append(result)
            sign = "+" if result.delta >= 0 else ""
            print(f"  OFF={result.score_off:.2f}  ON={result.score_on:.2f}  Δ={sign}{result.delta:.2f}  "
                  f"({'PASS' if result.passed else 'FAIL'})")
        except Exception as e:
            err_result = TechResult(
                technique=name, score_off=0.0, score_on=0.0, delta=0.0,
                latency_off_ms=0, latency_on_ms=0, error=str(e)[:100],
            )
            results.append(err_result)
            print(f"  ERROR: {str(e)[:60]}")

    return results


def save_results(results: list[TechResult], provider: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = {
        "provider": provider,
        "results": [
            {
                "technique": r.technique,
                "score_off": r.score_off,
                "score_on": r.score_on,
                "delta": r.delta,
                "latency_off_ms": r.latency_off_ms,
                "latency_on_ms": r.latency_on_ms,
                "passed": r.passed,
                "details": r.details,
                "error": r.error,
            }
            for r in results
        ],
        "summary": {
            "avg_score_off": round(sum(r.score_off for r in results) / max(len(results), 1), 3),
            "avg_score_on":  round(sum(r.score_on  for r in results) / max(len(results), 1), 3),
            "avg_delta":     round(sum(r.delta      for r in results) / max(len(results), 1), 3),
            "pass_rate":     round(sum(1 for r in results if r.passed) / max(len(results), 1), 3),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {out_path}")


async def main() -> None:
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Component Benchmark — Layer 1")
    parser.add_argument("--provider", choices=["ollama", "claude", "bedrock"], default="bedrock")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.provider == "ollama":
        from src.providers.ollama import OllamaProvider
        model = args.model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        llm = OllamaProvider(model=model, host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        label = f"ollama/{model}"
        out_path = f"eval/results/component_ollama_{model.replace(':', '_')}.json"
    elif args.provider == "claude":
        from src.providers.claude import ClaudeProvider
        model = args.model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        llm = ClaudeProvider(model=model)
        label = f"claude/{model}"
        out_path = "eval/results/component_claude.json"
    else:  # bedrock
        from src.providers.bedrock import BedrockProvider
        model = args.model or os.getenv("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6")
        llm = BedrockProvider(
            model=model,
            region=os.getenv("AWS_REGION", "us-west-2"),
        )
        label = f"bedrock/{model.split('.')[-1]}"
        out_path = "eval/results/component_bedrock.json"

    results = await run_all_tests(llm, label)
    print_report(results, label)
    save_results(results, label, out_path)


if __name__ == "__main__":
    asyncio.run(main())
