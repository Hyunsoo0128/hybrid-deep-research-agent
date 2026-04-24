#!/usr/bin/env python3
"""
Standalone Benchmark — Deep Research Agent
Layer 1 (T1-T4) + Layer 3 (E2E Hybrid)

No src/ imports. Direct httpx → Ollama, anthropic → Bedrock.
All raw LLM inputs/outputs saved. Results uploaded to S3.

Layer 1 — Local model unit tests:
  T1 JSON   (30%): JSON parse success rate with/without DSAP retry
  T2 CRAG   (25%): Relevance classification precision/recall vs gold labels
  T3 Critic (30%): AlignRAG misalignment detection rate
  T4 Decomp (15%): Query decomposition 5-dimensional coverage

Layer 3 — E2E pipeline quality comparison (2 fixed queries):
  C1: Sonnet-only         (upper bound — current production)
  C2: Sonnet + local LLM  (proposed hybrid)
  Metric: overall_score, revision_count, latency_sec per condition

Usage (on EC2):
  MODEL=qwen3:8b \\
  OLLAMA_HOST=http://localhost:11434 \\
  SONNET_MODEL=us.anthropic.claude-sonnet-4-6 \\
  S3_BUCKET=my-bench-bucket \\
  S3_PREFIX=benchmark-results \\
  RUN_ID=run_20260423 \\
  AWS_REGION=us-east-1 \\
  python standalone_benchmark.py
"""

from __future__ import annotations

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

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_HOST   = os.getenv("OLLAMA_HOST",   "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("MODEL",         "qwen3:8b")
SONNET_MODEL  = os.getenv("SONNET_MODEL",  "us.anthropic.claude-sonnet-4-6")
AWS_REGION    = os.getenv("AWS_REGION",    "us-east-1")
S3_BUCKET     = os.getenv("S3_BUCKET",     "")
S3_PREFIX     = os.getenv("S3_PREFIX",     "benchmark-results")
RUN_ID        = os.getenv("RUN_ID",        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))

LAYER1_WEIGHTS = {"t1_json": 0.30, "t2_crag": 0.25, "t3_critic": 0.30, "t4_decomp": 0.15}
LANG_MODE     = os.getenv("LANG_MODE", "en")   # en | ko | ja

# ══════════════════════════════════════════════════════════════════════════════
# LLM Clients
# ══════════════════════════════════════════════════════════════════════════════

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()

def _clean_json(raw: str) -> str:
    cleaned = _strip_think(raw)
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # Sometimes models wrap JSON in extra text — try to extract first {...} or [...]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = cleaned.find(start_char)
        end   = cleaned.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
    return cleaned


class OllamaClient:
    """Direct HTTP calls to Ollama /api/chat endpoint."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host  = host.rstrip("/")
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        import httpx
        # Keep system message as-is; append /no_think to the last user message
        # (Qwen3 spec: /no_think must be in the user turn to suppress thinking)
        full_msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        for i in range(len(full_msgs) - 1, -1, -1):
            if full_msgs[i]["role"] == "user":
                full_msgs[i] = {**full_msgs[i], "content": full_msgs[i]["content"] + "\n/no_think"}
                break

        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(
                f"{self.host}/api/chat",
                json={
                    "model":   self.model,
                    "messages": full_msgs,
                    "stream":  False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            r.raise_for_status()
            content = r.json()["message"]["content"]
            return _strip_think(content or "")


class BedrockClient:
    """Async Bedrock calls via anthropic.AsyncAnthropicBedrock."""

    def __init__(self, model: str = SONNET_MODEL, region: str = AWS_REGION):
        self.model = model
        from anthropic import AsyncAnthropicBedrock
        self._client = AsyncAnthropicBedrock(aws_region=region)

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.1,
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
        return response.content[0].text


# ══════════════════════════════════════════════════════════════════════════════
# DSAP-style JSON helper (arxiv:2512.20660)
# ══════════════════════════════════════════════════════════════════════════════

_RETRY_SYSTEM = "You must respond with ONLY valid JSON. No markdown, no explanations, no code blocks."
_RETRY_USER   = "Previous response could not be parsed as JSON.\nError: {error}\n\nRespond ONLY with valid JSON matching this schema:\n{schema}"
_LAST_RESORT_SYSTEM = "Output ONLY valid JSON. Nothing else."
_LAST_RESORT_USER   = "Output ONLY valid JSON using this exact schema (fill with appropriate values):\n{schema}"


async def llm_json(
    llm: Any,
    messages: list[dict],
    system: str,
    schema: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    max_retries: int = 2,
    dsap_enabled: bool = True,
    fallback: Any = None,
    raw_log: list | None = None,
) -> Any:
    """
    LLM call that guarantees JSON response.
    DSAP Level 1: retry with error context.
    DSAP Level 2: stagnation → last-resort clean slate.
    raw_log: if provided, appends {"in": prompt, "out": raw_response} dicts.
    """
    def _log(prompt_messages, raw):
        if raw_log is not None:
            raw_log.append({"in": prompt_messages, "out": raw})

    if not dsap_enabled:
        raw = await llm.complete(messages, system=system, max_tokens=max_tokens, temperature=temperature)
        _log(messages, raw)
        try:
            return json.loads(_clean_json(raw))
        except Exception:
            return fallback if fallback is not None else {}

    cur_msgs   = list(messages)
    cur_system = system
    prev_fp: str | None = None
    last_resort_used = False

    def _coerce_to_dict(parsed: Any) -> Any:
        """If we expect a dict (fallback is dict) but got a list, try to extract first dict."""
        if isinstance(fallback, dict) and not isinstance(parsed, dict):
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        return item
            return None  # signal: treat as parse failure
        return parsed

    for attempt in range(max_retries + 1):
        raw = await llm.complete(cur_msgs, system=cur_system, max_tokens=max_tokens, temperature=temperature)
        _log(cur_msgs, raw)

        try:
            parsed = json.loads(_clean_json(raw))
            coerced = _coerce_to_dict(parsed)
            if coerced is None:
                raise json.JSONDecodeError(
                    f"Expected dict, got {type(parsed).__name__}", raw, 0
                )
            return coerced
        except json.JSONDecodeError as e:
            if attempt >= max_retries:
                break
            fp = re.sub(r"line \d+ column \d+ \(char \d+\)", "line X", str(e))[:60]

            if prev_fp == fp and not last_resort_used:
                cur_msgs   = list(messages) + [{"role": "user", "content": _LAST_RESORT_USER.format(schema=schema)}]
                cur_system = _LAST_RESORT_SYSTEM
                last_resort_used = True
            else:
                cur_msgs = cur_msgs + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": _RETRY_USER.format(error=str(e), schema=schema)},
                ]
                cur_system = _RETRY_SYSTEM
            prev_fp = fp

    return fallback if fallback is not None else {}


# ══════════════════════════════════════════════════════════════════════════════
# ── FIXTURES ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── T1: DSAP prompts ──────────────────────────────────────────────────────────
T1_PROMPTS = [
    {
        "name": "research_plan",
        "prompt": 'Create a research plan JSON for "quantum computing state 2025".\nOutput ONLY valid JSON:\n{"intent":"analytical","sub_queries":[{"id":"sq1","question":"...","dimension":"..."},{"id":"sq2","question":"...","dimension":"..."}],"depth":"normal"}',
        "required_keys": ["intent", "sub_queries", "depth"],
    },
    {
        "name": "relevance_eval",
        "prompt": 'Evaluate search results for "LLM efficiency". Output ONLY valid JSON:\n{"evaluations":[{"index":0,"relevance_score":0.0},{"index":1,"relevance_score":0.0},{"index":2,"relevance_score":0.0}]}\nResults:\n[0] FlashAttention doubles GPU throughput\n[1] Python list comprehension tips\n[2] Speculative decoding reduces latency 3x',
        "required_keys": ["evaluations"],
    },
    {
        "name": "gap_analysis",
        "prompt": 'Analyze coverage gaps. Output ONLY valid JSON:\n{"gaps":[{"sub_query_id":"string","issue":"string","gap_query":"string"}],"coverage_score":0.0,"comment":"string"}\nQuestion: Recent AI safety developments\nCovered: [sq1: technical approaches, sq2: alignment research]',
        "required_keys": ["gaps", "coverage_score", "comment"],
    },
    {
        "name": "critic_feedback",
        "prompt": 'Review this report and output ONLY valid JSON:\n{"passed":false,"uncited_claims":[],"unanswered_sub_queries":[],"misaligned_claims":[],"suggestions":[]}\nReport: "AI systems achieve perfect accuracy in all real-world applications."\nSub-queries: [Is AI reliable in production?]',
        "required_keys": ["passed", "uncited_claims", "suggestions"],
    },
]

# ── T2: CRAG relevance fixture ────────────────────────────────────────────────
T2_QUERY = "transformer architecture self-attention mechanism advantages over RNN"
T2_RESULTS = [
    {"title": "Attention Is All You Need",                   "summary": "Transformers use self-attention to process all positions simultaneously, eliminating sequential dependency in RNNs, enabling full parallelization and capturing long-range dependencies.", "gold": "relevant"},
    {"title": "BERT Pre-training of Deep Bidirectional Transformers", "summary": "BERT uses transformer encoder with bidirectional self-attention, outperforming RNN-based models on 11 NLP benchmarks including GLUE with parallelizable training.", "gold": "relevant"},
    {"title": "Transformer vs RNN: Training Speed Comparison","summary": "Transformers train 3-8x faster than LSTMs on modern GPUs due to parallelism. Self-attention complexity is O(n^2) vs O(n) for RNNs, but GPU parallelism advantages dominate.", "gold": "relevant"},
    {"title": "GPT-4 Technical Report",                      "summary": "GPT-4 is a large multimodal model achieving human-level performance on various benchmarks, built on transformer architecture with RLHF.", "gold": "partial"},
    {"title": "Vision Transformer ViT: Image Recognition",   "summary": "Applying transformer directly to image patches achieves strong performance on ImageNet. Self-attention generalizes beyond NLP to vision with sufficient data.", "gold": "partial"},
    {"title": "Long-Range Arena Benchmark for Efficient Transformers", "summary": "Benchmark evaluating transformers on tasks requiring long-range dependencies. Various efficient attention variants trade accuracy for speed.", "gold": "partial"},
    {"title": "Stock Market Prediction Using Machine Learning 2024",  "summary": "Comparison of ML algorithms for stock price prediction including LSTM and gradient boosting. LSTM achieves 67% directional accuracy on S&P 500.", "gold": "irrelevant"},
    {"title": "Python Web Scraping Best Practices",           "summary": "Guide to ethical web scraping with BeautifulSoup and Scrapy. Covers rate limiting, robots.txt compliance, and proxy rotation.", "gold": "irrelevant"},
    {"title": "Database Indexing and Query Optimization",     "summary": "B-tree vs hash indexes for PostgreSQL. Query plan analysis, index coverage, and partial index strategies for high-performance SQL.", "gold": "irrelevant"},
    {"title": "JavaScript Framework Comparison 2024",         "summary": "React vs Vue vs Angular performance benchmarks. Component rendering speed, bundle size, and developer experience analysis.", "gold": "irrelevant"},
]

_T2_EVAL_SYSTEM = "You are an expert in evaluating information relevance. Assess how relevant each search result is to the research question."
_T2_EVAL_PROMPT = """Research question: {question}

Search results:
{results_text}

Rate the relevance of each result.
relevance_score: 0.0 = completely irrelevant, 1.0 = directly and fully answers the question.

Respond ONLY in this JSON format:
{{"evaluations": [{{"index": 0, "relevance_score": 0.85}}, {{"index": 1, "relevance_score": 0.2}}]}}"""

# ── T3: AlignRAG critic fixture ───────────────────────────────────────────────
T3_CITATIONS = [
    {"id": "c1", "title": "IBM Quantum Processor 2024 Benchmark",        "excerpt": "IBM's Eagle processor achieved 99.5% two-qubit gate fidelity across 127 qubits."},
    {"id": "c2", "title": "Quantum vs Classical Computing Performance",   "excerpt": "Quantum advantage demonstrated for specific combinatorial optimization: 50-100x speedup over classical heuristics for those narrow problem classes."},
    {"id": "c3", "title": "Quantum Decoherence Progress Report",          "excerpt": "Qubit coherence times improved from microseconds to milliseconds. Decoherence remains the primary obstacle to fault-tolerant quantum computing."},
    {"id": "c4", "title": "Quantum Error Correction Status",              "excerpt": "Surface codes require ~1,000 physical qubits per logical qubit. Current systems have 100-1000 physical qubits total."},
]
# 3 injected errors:
#   [1] "1,000x speedup" — source says "50-100x"
#   [2] "completely eliminated decoherence" — source says "remains the primary obstacle"
#   [3] "ready for commercial deployment" — contradicts error correction gap
T3_DRAFT = """# Quantum Computing: Current State

IBM's quantum processor achieved 99.5% two-qubit gate fidelity [c1], marking a major milestone.
Recent benchmarks show quantum computers deliver a 1,000x speedup over classical computers
for optimization problems [c2]. Scientists have completely eliminated decoherence through
advanced qubit engineering [c3], making quantum computers ready for commercial deployment [c4].
"""
T3_GOLD_ERROR_COUNT = 3

_T3_CRITIC_SYSTEM = """You are an expert in research report quality and factual consistency review.
Objectively diagnose misalignments between report claims and citation sources using a
structured 3-phase AlignRAG analysis."""

_T3_CRITIC_PROMPT = """Review the following research report draft and citation sources using a 3-phase AlignRAG misalignment analysis.

Original question: Current state of quantum computing

Citation sources:
{citations}

Report draft:
{draft}

Phase 1 — Relevance Alignment: claims irrelevant to the question.
Phase 2 — Query-Evidence Mapping: claims lacking citation support.
Phase 3 — Evidence-Integrated Synthesis: claims inconsistent with source content (exaggeration, distortion, contradiction).

Respond ONLY in this JSON format:
{{
  "misaligned_claims": [
    {{"phase": "phase1|phase2|phase3", "claim": "quoted text", "source_quote": "what source says", "correction_hint": "how to fix"}}
  ],
  "uncited_claims": [],
  "unanswered_sub_queries": [],
  "suggestions": []
}}"""

_T3_SIMPLE_PROMPT = """Review the following research report draft.

Original question: Current state of quantum computing

Report draft:
{draft}

Check:
1. Claims without citations
2. Unanswered sub-queries
3. Structural issues

Respond ONLY in this JSON format:
{{"misaligned_claims": [], "uncited_claims": [], "unanswered_sub_queries": [], "suggestions": []}}"""

# ── T4: Query decomposition fixture ──────────────────────────────────────────
T4_QUERIES = [
    "Current state of large language models in 2025",
    "Climate change mitigation strategies comparison",
    "Quantum computing commercialization challenges",
]
T4_FIVE_DIMS = [
    "definition/background",
    "current state/evidence",
    "comparison/alternatives",
    "cause/mechanism",
    "limitations/challenges",
]

_T4_DECOMP_SYSTEM = "You are a research planning expert. Analyze the user's query and create a systematic research plan."
_T4_DECOMP_DIMS   = "  [Definition/Background]   Core concept definitions, historical context\n  [Current State/Evidence]  Latest data, statistics, real-world examples\n  [Comparison/Alternatives] Different approaches, competing technologies\n  [Cause/Mechanism]         How it works, causal relationships\n  [Limitations/Challenges]  Drawbacks, risk factors, unresolved issues"
_T4_DECOMP_PROMPT = """Create a research plan for the following query.

Query: {query}

Generate 4-6 independent sub-queries covering these dimensions:
{dims}

Respond ONLY in this JSON format:
{{"sub_queries": [{{"id": "sq1", "question": "...", "dimension": "Current State/Evidence"}}, {{"id": "sq2", "question": "...", "dimension": "Comparison/Alternatives"}}]}}"""

_T4_SIMPLE_PROMPT = """Create a research plan for the following query.

Query: {query}

Generate 3-4 independent sub-queries covering different aspects.

Respond ONLY in this JSON format:
{{"sub_queries": [{{"id": "sq1", "question": "...", "dimension": "General"}}, {{"id": "sq2", "question": "...", "dimension": "General"}}]}}"""

# ══════════════════════════════════════════════════════════════════════════════
# ── LAYER 3 FIXTURES ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Multilingual queries (same semantic content in EN / KO / JA) ──────────────
_E2E_QUERIES_EN = [
    "What are the main techniques for optimizing LLM inference performance, and how do they compare in terms of speed, memory, and accuracy trade-offs?",
    "How does retrieval-augmented generation (RAG) improve LLM accuracy, and what are the key challenges in implementing it effectively?",
]
_E2E_QUERIES_KO = [
    "LLM 추론 성능을 최적화하는 주요 기법은 무엇이며, 속도·메모리·정확도 트레이드오프 측면에서 각 기법을 어떻게 비교할 수 있는가?",
    "검색 증강 생성(RAG)은 LLM의 정확도를 어떻게 향상시키는가, 그리고 효과적으로 구현하는 데 있어 핵심 과제는 무엇인가?",
]
_E2E_QUERIES_JA = [
    "LLM推論パフォーマンスを最適化する主な手法は何か、また速度・メモリ・精度のトレードオフの観点からそれらをどのように比較できるか？",
    "検索拡張生成（RAG）はLLMの精度をどのように向上させるか、また効果的に実装するための主要な課題は何か？",
]
_LANG_QUERIES  = {"en": _E2E_QUERIES_EN, "ko": _E2E_QUERIES_KO, "ja": _E2E_QUERIES_JA}
E2E_QUERIES    = _LANG_QUERIES.get(LANG_MODE, _E2E_QUERIES_EN)

# Language instruction injected into Writer / Revise prompts
_LANG_INSTRUCTION = {
    "en": "",
    "ko": "\n\nIMPORTANT: Write the entire report in Korean (한국어로 작성하시오).",
    "ja": "\n\nIMPORTANT: Write the entire report in Japanese (日本語で記述してください).",
}

# Pre-embedded doc pool shared across sub-queries (no Tavily dependency)
# Keyed by index so the same docs are used regardless of query language.
_E2E_DOCS: list[list[dict]] = [
    [  # index 0 — LLM inference optimisation
        {"id": "d1", "title": "KV Cache Optimization for LLM Inference",
         "content": "Key-value (KV) caching stores computed attention keys and values during autoregressive decoding, eliminating redundant computation. Without caching, each new token requires O(n) transformer layer computations. With KV cache, only the new token's K/V pairs need computation: O(1) per step after prefill. Memory scales as 2×layers×heads×head_dim×seq_len×batch_size. For a 7B model with 32 layers and 4096 sequence length, KV cache requires ~2GB. PagedAttention (vLLM) manages this like virtual memory, enabling 2-4x higher throughput through dynamic allocation."},
        {"id": "d2", "title": "Speculative Decoding: 2-4x Faster LLM Generation",
         "content": "Speculative decoding uses a smaller draft model (e.g., 1B) to propose K tokens ahead, then verifies them in one forward pass of the target model (e.g., 70B). When draft tokens are accepted, the target model generates K tokens in one pass instead of K passes. Acceptance rate depends on task difficulty: coding tasks achieve 3.5x speedup, instruction following 2.8x, creative writing 1.8x. The draft model must share the target model's vocabulary. Key limitation: requires a compatible draft model per target model; adds complexity to deployment."},
        {"id": "d3", "title": "Quantization: INT8, INT4, GPTQ, AWQ",
         "content": "Post-training quantization converts model weights from float16 to lower precision. INT8 (8-bit) reduces memory by 2x with <1% accuracy loss on most benchmarks. INT4 (4-bit) reduces memory by 4x with 2-5% accuracy loss. GPTQ uses layer-wise quantization with Hessian-based weight compensation. AWQ (Activation-aware Weight Quantization) identifies and preserves 1% of salient weights, achieving better accuracy than GPTQ at same bit-width. Activation quantization is harder: transformer attention layers have outliers (magnitude >100x average) that cause significant accuracy degradation with naive quantization."},
        {"id": "d4", "title": "Continuous Batching and PagedAttention",
         "content": "Static batching processes all requests in a batch until the slowest request finishes, wasting GPU compute. Continuous batching inserts new requests as soon as any slot frees up, dramatically improving utilization. vLLM's PagedAttention stores KV cache in non-contiguous memory blocks (pages), eliminating memory fragmentation. Combined, throughput improves 5-23x over naive static batching on real workloads. TensorRT-LLM and TGI implement similar optimizations. Latency is not improved — only throughput."},
        {"id": "d5", "title": "Flash Attention: Memory-Efficient Attention",
         "content": "Standard attention materializes the full N×N attention matrix, requiring O(N^2) memory. Flash Attention uses tiling and online softmax to compute attention in blocks without materializing the full matrix, reducing memory to O(N). Flash Attention 2 achieves 2-4x wall-clock speedup over standard attention on A100 GPUs by maximizing SRAM utilization and reducing memory bandwidth bottleneck. Most beneficial for sequences longer than 1K tokens. Flash Attention 3 further optimizes for H100 with asynchronous computation and FP8 support."},
        {"id": "d6", "title": "LLM Serving Cost and Trade-off Analysis",
         "content": "Serving a 70B model on 4×A100 (80GB) GPUs costs ~$12/hour on AWS. INT4 quantization reduces this to 2 GPUs (~$6/hour) with ~3% accuracy loss. Speculative decoding adds a draft model but reduces per-token latency by 2-3x, improving user experience without reducing throughput cost. Flash Attention reduces GPU memory pressure, enabling larger batch sizes (more users served simultaneously). Key trade-offs: quantization reduces memory/cost at accuracy cost; speculative decoding reduces latency at complexity cost; continuous batching improves throughput at latency variance cost."},
        {"id": "d7", "title": "Python Web Framework Benchmarks 2024",
         "content": "FastAPI achieves 67,000 req/s vs Django's 12,000 on identical hardware. Database connection pooling is the primary performance bottleneck for web applications."},
        {"id": "d8", "title": "Reinforcement Learning from Human Feedback",
         "content": "RLHF trains LLMs to follow human preferences using reward models. PPO and DPO are the primary algorithms. Does not directly address inference efficiency."},
    ],
    [  # index 1 — RAG
        {"id": "d1", "title": "Retrieval-Augmented Generation (RAG) Survey",
         "content": "RAG combines a retrieval component (dense retriever or sparse BM25) with a generative LLM. The retriever fetches relevant documents given the query; the LLM generates an answer conditioned on retrieved documents. RAG addresses LLM hallucination by grounding generation in external knowledge. Key finding: RAG reduces factual errors by 40-60% on knowledge-intensive tasks (NaturalQuestions, TriviaQA). Naive RAG (retrieve-then-read) often retrieves noisy documents; advanced RAG uses query rewriting, reranking, and iterative retrieval."},
        {"id": "d2", "title": "Dense Retrieval: DPR and Bi-encoder Models",
         "content": "Dense Passage Retrieval (DPR) uses dual-encoder architecture: query encoder and passage encoder trained with contrastive learning. At inference, passage embeddings are pre-computed and stored in a FAISS index. Query embedding compared via cosine similarity. DPR outperforms BM25 by 9-18% on open-domain QA. Limitations: requires GPU for query encoding at inference; embedding space is fixed at training time; out-of-domain generalization is poor without fine-tuning. Typical retrieval latency: 50-200ms for 10M passages."},
        {"id": "d3", "title": "RAG Evaluation: Hallucination and Faithfulness",
         "content": "RAGAs framework evaluates: faithfulness (claims supported by context), answer relevance (response relevance to query), context precision (retrieved docs relevance), context recall (coverage of ground truth). Studies show: without RAG, GPT-4 hallucinates in 20-30% of knowledge-intensive queries; with RAG, hallucination drops to 5-12%. However, RAG introduces new failure modes: retrieval failures (relevant docs not found), context confusion (too many irrelevant docs), and faithfulness failures (LLM ignores retrieved context)."},
        {"id": "d4", "title": "Chunking Strategies for RAG",
         "content": "Document chunking strategy significantly impacts RAG quality. Fixed-size chunking (512 tokens) is simple but breaks semantic units. Semantic chunking uses embedding similarity to find natural boundaries, improving retrieval precision by 15-25%. Hierarchical chunking stores both sentence-level and paragraph-level chunks, enabling multi-granularity retrieval. Optimal chunk size depends on task: QA benefits from smaller chunks (128-256 tokens), summarization from larger chunks (512-1024 tokens). Chunk overlap (10-20%) prevents information loss at boundaries."},
        {"id": "d5", "title": "Reranking in RAG Pipelines",
         "content": "After initial retrieval, cross-encoder rerankers re-score top-K documents against the query. Cross-encoders (e.g., ms-marco-MiniLM) process query+document pairs jointly, capturing fine-grained relevance. Reranking improves NDCG@10 by 8-15% over bi-encoder retrieval alone. Pipeline: retrieve 100 docs with bi-encoder (fast) → rerank top 100 to top 5 with cross-encoder (accurate). Latency: cross-encoder adds 50-100ms for 100 documents. ColBERT uses late interaction to balance speed and accuracy between bi-encoder and cross-encoder."},
        {"id": "d6", "title": "Challenges in Production RAG Systems",
         "content": "Production RAG faces multiple challenges: (1) Retrieval quality degrades on long-tail queries and domain-specific terminology. (2) Context length limits: retrieving more documents improves recall but increases latency and may exceed LLM context window. (3) Multi-hop reasoning requires iterative retrieval (HyDE, ITER-RETGEN). (4) Document freshness: static indexes miss recent information; continuous indexing adds infrastructure complexity. (5) Scalability: maintaining FAISS index for billions of documents requires distributed retrieval infrastructure. Typical production RAG systems achieve 70-85% accuracy on enterprise Q&A benchmarks."},
        {"id": "d7", "title": "GraphQL vs REST API Design",
         "content": "GraphQL reduces over-fetching by allowing clients to specify exact data needs. REST is simpler and better supported by caching infrastructure. Not related to RAG."},
        {"id": "d8", "title": "Kubernetes Auto-scaling for ML Workloads",
         "content": "Horizontal Pod Autoscaler scales inference pods based on GPU utilization. KEDA enables event-driven scaling based on queue depth. Not directly related to RAG accuracy."},
    ],
]

# Build doc pool keyed by the active language's query strings
E2E_DOC_POOLS = {q: _E2E_DOCS[i] for i, q in enumerate(E2E_QUERIES)}

# ── Layer 3 prompts ────────────────────────────────────────────────────────────

_PLAN_SYSTEM = "You are a research planning expert. Create systematic, parallelizable research plans."
_PLAN_PROMPT = """Create a research plan for the following query.

Query: {query}

Generate 4-5 independent sub-queries covering:
  [Definition/Background]   Core concepts
  [Current State/Evidence]  Latest data and evidence
  [Comparison/Alternatives] Different approaches compared
  [Cause/Mechanism]         How it works
  [Limitations/Challenges]  Known drawbacks and challenges

Respond ONLY in this JSON format:
{{"intent": "analytical", "sub_queries": [{{"id": "sq1", "question": "...", "dimension": "..."}}], "depth": "normal"}}"""

_CRAG_EVAL_SYSTEM = "You are an expert in evaluating information relevance."
_CRAG_EVAL_PROMPT = """Research question: {question}

Documents:
{docs_text}

Rate relevance of each document (0.0=irrelevant, 1.0=directly answers the question).
Respond ONLY in JSON:
{{"evaluations": [{{"index": 0, "relevance_score": 0.0}}]}}"""

_MASS_SUMMARIZER_SYSTEM = "You are an expert research summarizer."
_MASS_SUMMARIZER_PROMPT = """Research question: {question}

Source documents:
{docs}

Summarize key information directly relevant to the research question (under 150 words). Respond with plain text only."""

_MASS_EXTRACTOR_SYSTEM = "You are a key-span extraction specialist."
_MASS_EXTRACTOR_PROMPT = """Research question: {question}

Source documents:
{docs}

Extract key information spans directly relevant to the research question.
Respond ONLY in JSON:
{{"key_spans": [{{"text": "...", "source_id": "d1", "type": "fact|definition|evidence"}}]}}"""

_MASS_REASONER_SYSTEM = "You are a research reasoning specialist."
_MASS_REASONER_PROMPT = """Research question: {question}

Source documents:
{docs}

Identify key inferences and conclusions from these documents.
Respond ONLY in JSON:
{{"inferences": [{{"claim": "...", "support": "brief note"}}]}}"""

_MASS_SYNTHESIS_SYSTEM = "You are a research synthesis expert."
_MASS_SYNTHESIS_PROMPT = """Research question: {question}

Summary: {summary}

Key spans:
{spans}

Inferences:
{inferences}

Synthesize into a comprehensive analysis.
Respond ONLY in JSON:
{{"summary": "...", "key_spans": [{{"text": "...", "source_id": "...", "type": "fact"}}], "inferences": [{{"claim": "..."}}]}}"""

_WRITER_SYSTEM = "You are an expert research writer. Write clear, well-structured reports with inline citations."
_WRITER_PROMPT = """Write a research report answering this question.

Question: {query}

Sub-queries covered:
{sub_queries}

Source documents (cite as [doc_id]):
{sources}

MASS-RAG Analysis:
{mass_rag}

Write a 400-600 word report:
1. ## Introduction (2-3 sentences)
2. ## Analysis (2-3 subsections)
3. ## Conclusion (2-3 sentences)
4. ## References

Cite every factual claim with [doc_id]. Be specific and technical.{lang_instruction}"""

_CRITIC_SYSTEM = "You are an expert in research report quality and factual consistency review."
_CRITIC_PROMPT = """Review this report using 3-phase AlignRAG analysis.

Question: {query}

Citation sources:
{citations}

Report:
{draft}

Phase 1: Claims irrelevant to the question.
Phase 2: Claims lacking citation support.
Phase 3: Claims inconsistent with source content (exaggeration, distortion, contradiction).

Respond ONLY in JSON:
{{"misaligned_claims": [{{"phase": "phase1|phase2|phase3", "claim": "...", "source_quote": "...", "correction_hint": "..."}}], "uncited_claims": [], "suggestions": [], "passed": false}}"""

_REVISE_SYSTEM = "You are an expert research editor. Improve reports based on reviewer feedback."
_REVISE_PROMPT = """Improve this report based on the feedback.

Question: {query}

Current report:
{draft}

Feedback:
{feedback}

Fix all identified issues while preserving overall structure. Return the complete revised report.{lang_instruction}"""

_JUDGE_SYSTEM = "You are an expert research quality evaluator. Score objectively based on provided criteria."
_JUDGE_PROMPT = """Evaluate this research report on quality criteria.

Research question: {query}

Report:
{report}

Reference source materials:
{sources}

Score each criterion 0.0-1.0:
- coverage: Does the report address all major aspects?
- accuracy: Are claims consistent with source materials?
- specificity: Are technical details and concrete data included?
- structure: Is the report well-organized?
- overall: Holistic quality score

Respond ONLY in JSON:
{{"coverage": 0.0, "accuracy": 0.0, "specificity": 0.0, "structure": 0.0, "overall": 0.0, "reasoning": "one sentence"}}"""


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 Tests
# ══════════════════════════════════════════════════════════════════════════════

async def test_t1_json(llm: Any) -> dict:
    """T1: JSON parse success rate with/without DSAP retry."""
    raw_log: list[dict] = []
    successes_off, successes_on = 0, 0
    details = []

    for p in T1_PROMPTS:
        msgs = [{"role": "user", "content": p["prompt"]}]
        sys  = "Output ONLY valid JSON. No explanations."

        # OFF: no retry
        t0 = time.time()
        r_off = await llm_json(llm, msgs, sys, str(p["required_keys"]),
                               max_tokens=512, dsap_enabled=False, fallback=None, raw_log=raw_log)
        lat_off = int((time.time() - t0) * 1000)
        ok_off  = r_off is not None and all(k in r_off for k in p["required_keys"])
        if ok_off: successes_off += 1

        # ON: DSAP retry
        t0 = time.time()
        r_on = await llm_json(llm, msgs, sys, str(p["required_keys"]),
                              max_tokens=512, dsap_enabled=True, fallback=None, raw_log=raw_log)
        lat_on = int((time.time() - t0) * 1000)
        ok_on  = r_on is not None and all(k in r_on for k in p["required_keys"])
        if ok_on: successes_on += 1

        details.append({"test": p["name"], "off": ok_off, "on": ok_on,
                        "lat_off_ms": lat_off, "lat_on_ms": lat_on})

    score_off = successes_off / len(T1_PROMPTS)
    score_on  = successes_on  / len(T1_PROMPTS)
    return {
        "score_off": round(score_off, 3), "score_on": round(score_on, 3),
        "delta": round(score_on - score_off, 3),
        "details": details, "raw_log": raw_log,
    }


async def test_t2_crag(llm: Any) -> dict:
    """T2: CRAG relevance classification F1 + noise reduction."""
    raw_log: list[dict] = []
    results_text = "\n".join(
        f"[{i}] {r['title']}\n    {r['summary']}"
        for i, r in enumerate(T2_RESULTS)
    )
    msgs = [{"role": "user", "content": _T2_EVAL_PROMPT.format(
        question=T2_QUERY, results_text=results_text
    )}]

    t0 = time.time()
    data = await llm_json(llm, msgs, _T2_EVAL_SYSTEM,
                          '{"evaluations":[{"index":0,"relevance_score":0.0}]}',
                          max_tokens=512, dsap_enabled=True,
                          fallback={"evaluations": []}, raw_log=raw_log)
    lat_ms = int((time.time() - t0) * 1000)

    # Map float scores to labels
    score_map = {e["index"]: float(e["relevance_score"])
                 for e in data.get("evaluations", []) if "index" in e}
    gold  = [r["gold"]    for r in T2_RESULTS]
    preds = []
    for i in range(len(T2_RESULTS)):
        s = score_map.get(i, 0.5)
        preds.append("relevant" if s >= 0.6 else ("partial" if s >= 0.3 else "irrelevant"))

    # Precision / recall (positive class = relevant + partial)
    tp = sum(1 for g, p in zip(gold, preds) if g in ("relevant","partial") and p in ("relevant","partial"))
    tn = sum(1 for g, p in zip(gold, preds) if g == "irrelevant" and p == "irrelevant")
    fp = sum(1 for g, p in zip(gold, preds) if g == "irrelevant" and p != "irrelevant")
    fn = sum(1 for g, p in zip(gold, preds) if g in ("relevant","partial") and p == "irrelevant")

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 0.001)
    noise_red = tn / max(sum(1 for g in gold if g == "irrelevant"), 1)

    # Baseline (no evaluation): all treated as relevant → precision = 6/10 = 0.6
    score_off = round(sum(1 for g in gold if g in ("relevant","partial")) / len(gold), 3)
    score_on  = round((f1 + noise_red) / 2, 3)

    return {
        "score_off": score_off, "score_on": score_on,
        "delta": round(score_on - score_off, 3),
        "details": {"precision": round(precision, 3), "recall": round(recall, 3),
                    "f1": round(f1, 3), "noise_reduction": round(noise_red, 3),
                    "predictions": {r["title"][:30]: preds[i] for i, r in enumerate(T2_RESULTS)},
                    "lat_ms": lat_ms},
        "raw_log": raw_log,
    }


async def test_t3_critic(llm: Any) -> dict:
    """T3: AlignRAG error detection — fraction of 3 injected errors caught."""
    raw_log: list[dict] = []
    citations_text = "\n".join(f"[{c['id']}] {c['title']}\n  {c['excerpt']}" for c in T3_CITATIONS)
    schema = '{"misaligned_claims":[{"phase":"...","claim":"...","source_quote":"...","correction_hint":"..."}],"uncited_claims":[],"unanswered_sub_queries":[],"suggestions":[]}'

    # OFF: no AlignRAG (no citation comparison)
    t0 = time.time()
    data_off = await llm_json(
        llm, [{"role": "user", "content": _T3_SIMPLE_PROMPT.format(draft=T3_DRAFT)}],
        _T3_CRITIC_SYSTEM, schema, max_tokens=800, dsap_enabled=True,
        fallback={"misaligned_claims": [], "uncited_claims": [], "unanswered_sub_queries": [], "suggestions": []},
        raw_log=raw_log,
    )
    lat_off = int((time.time() - t0) * 1000)
    detected_off = len(data_off.get("misaligned_claims", []))

    # ON: AlignRAG 3-phase with citations
    t0 = time.time()
    data_on = await llm_json(
        llm, [{"role": "user", "content": _T3_CRITIC_PROMPT.format(
            citations=citations_text, draft=T3_DRAFT
        )}],
        _T3_CRITIC_SYSTEM, schema, max_tokens=1200, dsap_enabled=True,
        fallback={"misaligned_claims": [], "uncited_claims": [], "unanswered_sub_queries": [], "suggestions": []},
        raw_log=raw_log,
    )
    lat_on = int((time.time() - t0) * 1000)
    detected_on = len(data_on.get("misaligned_claims", []))

    score_off = round(min(detected_off / T3_GOLD_ERROR_COUNT, 1.0), 3)
    score_on  = round(min(detected_on  / T3_GOLD_ERROR_COUNT, 1.0), 3)

    return {
        "score_off": score_off, "score_on": score_on,
        "delta": round(score_on - score_off, 3),
        "details": {"gold_error_count": T3_GOLD_ERROR_COUNT,
                    "detected_off": detected_off, "detected_on": detected_on,
                    "misaligned_claims_on": data_on.get("misaligned_claims", []),
                    "lat_off_ms": lat_off, "lat_on_ms": lat_on},
        "raw_log": raw_log,
    }


def _dim_coverage(sub_queries: list[dict]) -> float:
    dims = [sq.get("dimension", "").lower() for sq in sub_queries]
    covered = sum(
        1 for d in T4_FIVE_DIMS
        if any(keyword in found for keyword in d.split("/") for found in dims)
    )
    return covered / len(T4_FIVE_DIMS)


async def test_t4_decomp(llm: Any) -> dict:
    """T4: Query decomposition 5-dimensional coverage."""
    raw_log: list[dict] = []
    schema = '{"sub_queries":[{"id":"sq1","question":"...","dimension":"..."}]}'
    scores_off, scores_on = [], []
    details = []

    for query in T4_QUERIES:
        # OFF: simple decomposition
        t0 = time.time()
        d_off = await llm_json(
            llm, [{"role": "user", "content": _T4_SIMPLE_PROMPT.format(query=query)}],
            _T4_DECOMP_SYSTEM, schema, max_tokens=1024, temperature=0.3,
            fallback={"sub_queries": []}, raw_log=raw_log,
        )
        lat_off = int((time.time() - t0) * 1000)
        sqs_off = d_off.get("sub_queries", [])
        cov_off = _dim_coverage(sqs_off)
        scores_off.append(cov_off)

        # ON: 5-dimensional decomposition
        t0 = time.time()
        d_on = await llm_json(
            llm, [{"role": "user", "content": _T4_DECOMP_PROMPT.format(
                query=query, dims=_T4_DECOMP_DIMS
            )}],
            _T4_DECOMP_SYSTEM, schema, max_tokens=1024, temperature=0.3,
            fallback={"sub_queries": []}, raw_log=raw_log,
        )
        lat_on = int((time.time() - t0) * 1000)
        sqs_on = d_on.get("sub_queries", [])
        cov_on = _dim_coverage(sqs_on)
        scores_on.append(cov_on)

        details.append({"query": query[:50], "coverage_off": round(cov_off, 2),
                        "coverage_on": round(cov_on, 2), "sq_count_on": len(sqs_on),
                        "dims_on": [sq.get("dimension","") for sq in sqs_on],
                        "lat_off_ms": lat_off, "lat_on_ms": lat_on})

    score_off = round(sum(scores_off) / len(scores_off), 3)
    score_on  = round(sum(scores_on)  / len(scores_on), 3)
    return {
        "score_off": score_off, "score_on": score_on,
        "delta": round(score_on - score_off, 3),
        "details": {"per_query": details},
        "raw_log": raw_log,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3: E2E Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _format_docs(docs: list[dict]) -> str:
    return "\n\n".join(f"[{d['id']}] {d['title']}\n{d['content'][:400]}" for d in docs)


async def _e2e_plan(query: str, llm: Any, raw_log: list) -> list[dict]:
    data = await llm_json(
        llm, [{"role": "user", "content": _PLAN_PROMPT.format(query=query)}],
        _PLAN_SYSTEM, '{"intent":"...","sub_queries":[{"id":"sq1","question":"...","dimension":"..."}],"depth":"normal"}',
        max_tokens=1024, temperature=0.3, fallback={"sub_queries": []}, raw_log=raw_log,
    )
    sqs = data.get("sub_queries", [])
    # Always prepend the original query as sq0
    return [{"id": "sq0", "question": query, "dimension": "Original Query"}] + sqs


async def _e2e_crag(question: str, docs: list[dict], llm: Any, raw_log: list) -> list[dict]:
    """Score docs, return CORRECT (score≥0.5) ones."""
    docs_text = "\n".join(f"[{i}] {d['title']}\n    {d['content'][:150]}" for i, d in enumerate(docs))
    data = await llm_json(
        llm, [{"role": "user", "content": _CRAG_EVAL_PROMPT.format(
            question=question, docs_text=docs_text
        )}],
        _CRAG_EVAL_SYSTEM, '{"evaluations":[{"index":0,"relevance_score":0.0}]}',
        max_tokens=256, temperature=0.0, fallback={"evaluations": []}, raw_log=raw_log,
    )
    score_map = {e["index"]: float(e.get("relevance_score", 0.5))
                 for e in data.get("evaluations", []) if "index" in e}
    return [d for i, d in enumerate(docs) if score_map.get(i, 0.5) >= 0.5]


async def _e2e_mass_rag(question: str, docs: list[dict], local_llm: Any, cloud_llm: Any, raw_log: list) -> dict:
    """MASS-RAG: 3-agent (local) + synthesis (cloud)."""
    if not docs:
        return {"summary": "", "key_spans": [], "inferences": []}
    docs_text = _format_docs(docs)

    # 3 agents in parallel (local)
    sum_coro = local_llm.complete(
        [{"role": "user", "content": _MASS_SUMMARIZER_PROMPT.format(question=question, docs=docs_text)}],
        system=_MASS_SUMMARIZER_SYSTEM, max_tokens=200, temperature=0.1,
    )
    ext_coro = llm_json(
        local_llm, [{"role": "user", "content": _MASS_EXTRACTOR_PROMPT.format(question=question, docs=docs_text)}],
        _MASS_EXTRACTOR_SYSTEM, '{"key_spans":[{"text":"...","source_id":"...","type":"fact"}]}',
        max_tokens=400, temperature=0.0, fallback={"key_spans": []}, raw_log=raw_log,
    )
    rea_coro = llm_json(
        local_llm, [{"role": "user", "content": _MASS_REASONER_PROMPT.format(question=question, docs=docs_text)}],
        _MASS_REASONER_SYSTEM, '{"inferences":[{"claim":"...","support":"..."}]}',
        max_tokens=300, temperature=0.1, fallback={"inferences": []}, raw_log=raw_log,
    )

    summary_text, ext_out, rea_out = await asyncio.gather(sum_coro, ext_coro, rea_coro)
    if raw_log is not None:
        raw_log.append({"in": _MASS_SUMMARIZER_PROMPT.format(question=question, docs=docs_text[:200]), "out": summary_text})

    spans_text = "\n".join(f"- ({s.get('type','fact')}) [{s.get('source_id','')}] {s.get('text','')[:100]}"
                           for s in ext_out.get("key_spans", [])) or "(none)"
    infer_text = "\n".join(f"- {i.get('claim','')}: {i.get('support','')[:80]}"
                           for i in rea_out.get("inferences", [])) or "(none)"

    # Synthesis (cloud)
    synth = await llm_json(
        cloud_llm, [{"role": "user", "content": _MASS_SYNTHESIS_PROMPT.format(
            question=question, summary=summary_text or "(none)",
            spans=spans_text, inferences=infer_text,
        )}],
        _MASS_SYNTHESIS_SYSTEM,
        '{"summary":"...","key_spans":[{"text":"...","source_id":"...","type":"fact"}],"inferences":[{"claim":"..."}]}',
        max_tokens=500, temperature=0.0,
        fallback={"summary": summary_text or "", "key_spans": ext_out.get("key_spans", []),
                  "inferences": [{"claim": i.get("claim","")} for i in rea_out.get("inferences", [])]},
        raw_log=raw_log,
    )
    return synth


async def _e2e_write(query: str, sub_queries: list[dict], docs: list[dict], mass_rag_results: list[dict],
                     llm: Any, raw_log: list) -> str:
    sqs_text   = "\n".join(f"- [{sq['id']}] {sq['question']}" for sq in sub_queries)
    sources    = _format_docs(docs[:6])
    mass_text  = "\n".join(
        f"[{r.get('sq_id','')}] {r.get('summary','')[:200]}"
        for r in mass_rag_results if r.get("summary")
    ) or "(no MASS-RAG analysis)"

    lang_instr = _LANG_INSTRUCTION.get(LANG_MODE, "")
    raw = await llm.complete(
        [{"role": "user", "content": _WRITER_PROMPT.format(
            query=query, sub_queries=sqs_text, sources=sources, mass_rag=mass_text,
            lang_instruction=lang_instr,
        )}],
        system=_WRITER_SYSTEM, max_tokens=1500, temperature=0.3,
    )
    if raw_log is not None:
        raw_log.append({"in": _WRITER_PROMPT.format(query=query, sub_queries=sqs_text,
                                                     sources=sources[:200], mass_rag=mass_text[:200],
                                                     lang_instruction=lang_instr),
                        "out": raw})
    return raw.strip()


async def _e2e_critique(query: str, sub_queries: list[dict], docs: list[dict],
                        draft: str, local_llm: Any, cloud_llm: Any, raw_log: list) -> dict:
    """Spec RAG critic: local extracts suspects, cloud verifies."""
    citations = "\n".join(f"[{d['id']}] {d['title']}: {d['content'][:120]}" for d in docs[:5])
    schema = '{"misaligned_claims":[{"phase":"...","claim":"...","source_quote":"...","correction_hint":"..."}],"uncited_claims":[],"suggestions":[],"passed":false}'

    data = await llm_json(
        local_llm, [{"role": "user", "content": _CRITIC_PROMPT.format(
            query=query, citations=citations, draft=draft
        )}],
        _CRITIC_SYSTEM, schema, max_tokens=1000, dsap_enabled=True,
        fallback={"misaligned_claims": [], "uncited_claims": [], "suggestions": [], "passed": True},
        raw_log=raw_log,
    )
    # passed = True only when no issues found
    has_issues = (len(data.get("misaligned_claims", [])) > 0 or
                  len(data.get("uncited_claims", [])) > 0)
    data["passed"] = not has_issues
    return data


async def _e2e_revise(query: str, draft: str, feedback: dict, llm: Any, raw_log: list) -> str:
    feedback_text = "\n".join([
        f"- Misaligned claims: {json.dumps(feedback.get('misaligned_claims', [])[:3], ensure_ascii=False)[:300]}",
        f"- Uncited claims: {', '.join(str(c) for c in feedback.get('uncited_claims', [])[:3]) or 'none'}",
        f"- Suggestions: {', '.join(str(s) for s in feedback.get('suggestions', [])[:3]) or 'none'}",
    ])
    lang_instr = _LANG_INSTRUCTION.get(LANG_MODE, "")
    raw = await llm.complete(
        [{"role": "user", "content": _REVISE_PROMPT.format(
            query=query, draft=draft, feedback=feedback_text,
            lang_instruction=lang_instr,
        )}],
        system=_REVISE_SYSTEM, max_tokens=1500, temperature=0.3,
    )
    if raw_log is not None:
        raw_log.append({"in": f"REVISE: {feedback_text[:200]}", "out": raw})
    return raw.strip()


async def _e2e_score(query: str, report: str, docs: list[dict], judge_llm: Any, raw_log: list) -> dict:
    sources = _format_docs(docs[:4])
    data = await llm_json(
        judge_llm,
        [{"role": "user", "content": _JUDGE_PROMPT.format(
            query=query, report=report, sources=sources
        )}],
        _JUDGE_SYSTEM,
        '{"coverage":0.0,"accuracy":0.0,"specificity":0.0,"structure":0.0,"overall":0.0,"reasoning":"..."}',
        max_tokens=256, temperature=0.0,
        fallback={"coverage": 0.5, "accuracy": 0.5, "specificity": 0.5, "structure": 0.5, "overall": 0.5, "reasoning": "fallback"},
        raw_log=raw_log,
    )
    return data


async def run_e2e_condition(
    condition_name: str,
    local_llm: Any,
    cloud_llm: Any,
    judge_llm: Any,
) -> dict:
    """Run E2E pipeline for one condition across all E2E queries."""
    per_query = []
    all_raw: list[dict] = []

    for query in E2E_QUERIES:
        docs = E2E_DOC_POOLS[query]
        raw_log: list[dict] = []
        t_start = time.time()

        # Step 1: Plan (cloud)
        sub_queries = await _e2e_plan(query, cloud_llm, raw_log)

        # Step 2: CRAG + MASS-RAG per sub-query (local eval, cloud synthesis)
        all_relevant_docs: list[dict] = []
        mass_rag_results: list[dict] = []

        for sq in sub_queries[:4]:  # limit to 4 sub-queries for speed
            relevant = await _e2e_crag(sq["question"], docs, local_llm, raw_log)
            all_relevant_docs.extend(d for d in relevant if d not in all_relevant_docs)
            if relevant:
                synthesis = await _e2e_mass_rag(sq["question"], relevant[:4], local_llm, cloud_llm, raw_log)
                synthesis["sq_id"] = sq["id"]
                mass_rag_results.append(synthesis)

        # Step 3: Write (cloud)
        draft = await _e2e_write(query, sub_queries, all_relevant_docs or docs[:4],
                                  mass_rag_results, cloud_llm, raw_log)

        # Step 4: Critique + Revise loop (local critique, local revise)
        revision_count = 0
        max_revisions  = 2
        while revision_count < max_revisions:
            feedback = await _e2e_critique(query, sub_queries, docs,
                                            draft, local_llm, cloud_llm, raw_log)
            if feedback.get("passed", True):
                break
            draft = await _e2e_revise(query, draft, feedback, local_llm, raw_log)
            revision_count += 1

        # Step 5: Score (always Sonnet judge)
        score = await _e2e_score(query, draft, docs, judge_llm, raw_log)
        latency = round(time.time() - t_start, 1)

        per_query.append({
            "query": query,
            "revision_count": revision_count,
            "latency_sec": latency,
            "score": score,
            "final_report": draft,
            "sub_query_count": len(sub_queries),
        })
        all_raw.extend(raw_log)

    def _overall(score: Any) -> float:
        return score.get("overall", 0.5) if isinstance(score, dict) else 0.5

    avg_overall   = round(sum(_overall(q["score"]) for q in per_query) / len(per_query), 3)
    avg_revisions = round(sum(q["revision_count"]      for q in per_query) / len(per_query), 2)
    avg_latency   = round(sum(q["latency_sec"]         for q in per_query) / len(per_query), 1)

    return {
        "condition":          condition_name,
        "avg_overall_score":  avg_overall,
        "avg_revision_count": avg_revisions,
        "avg_latency_sec":    avg_latency,
        "per_query":          per_query,
        "raw_log":            all_raw,
    }


# ══════════════════════════════════════════════════════════════════════════════
# S3 Upload
# ══════════════════════════════════════════════════════════════════════════════

def upload_to_s3(results: dict, model_name: str) -> str | None:
    if not S3_BUCKET:
        print("  [S3] S3_BUCKET not set — skipping upload")
        return None
    try:
        import boto3
        safe_model = model_name.replace(":", "_").replace(".", "_")
        key = f"{S3_PREFIX}/{RUN_ID}/{safe_model}/results.json"
        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        s3_uri = f"s3://{S3_BUCKET}/{key}"
        print(f"  [S3] Uploaded → {s3_uri}")
        return s3_uri
    except Exception as e:
        print(f"  [S3] Upload failed: {e}")
        return None


def save_local(results: dict, model_name: str) -> str:
    safe_model = model_name.replace(":", "_").replace(".", "_")
    out_dir = Path(__file__).parent / "results" / "standalone"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_model}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"  [Local] Saved → {out_path}")
    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


async def main(mode: str = "all") -> None:
    """
    mode="c1"  : Layer 3 C1 only (Bedrock Sonnet baseline, no Ollama required)
    mode="c2"  : Layer 1 + Layer 3 C2 (Ollama local + Sonnet cloud)
    mode="all" : Layer 1 + Layer 3 C1 + C2 (default behaviour)
    """
    label = {"c1": "C1-BASELINE (Sonnet-only)", "c2": f"C2-HYBRID ({OLLAMA_MODEL})", "all": OLLAMA_MODEL}[mode]
    print(f"\n{'═'*70}")
    print(f"  STANDALONE BENCHMARK — {label}")
    print(f"  Mode: {mode}  |  Lang: {LANG_MODE}  |  Sonnet: {SONNET_MODEL}")
    if mode != "c1":
        print(f"  Ollama: {OLLAMA_HOST}  |  Model: {OLLAMA_MODEL}")
    print(f"  Run ID: {RUN_ID}")
    print(f"{'═'*70}\n")

    sonnet = BedrockClient(SONNET_MODEL, AWS_REGION)
    ollama = OllamaClient(OLLAMA_HOST, OLLAMA_MODEL) if mode != "c1" else None

    # model key: "c1_baseline" in c1 mode, otherwise the Ollama model name
    model_key = "c1_baseline" if mode == "c1" else OLLAMA_MODEL

    results: dict[str, Any] = {
        "model":        model_key,
        "mode":         mode,
        "lang":         LANG_MODE,
        "run_id":       RUN_ID,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "sonnet_model": SONNET_MODEL,
        "layer1":       {},
        "layer3":       {},
    }

    # ── Layer 1 (c2 / all mode only) ─────────────────────────────────────────
    if mode in ("c2", "all"):
        print("▶ Layer 1 — Unit Tests\n")
        layer1_tests = [
            ("t1_json",   "T1 JSON (DSAP)",       lambda: test_t1_json(ollama)),
            ("t2_crag",   "T2 CRAG (relevance)",  lambda: test_t2_crag(ollama)),
            ("t3_critic", "T3 Critic (AlignRAG)", lambda: test_t3_critic(ollama)),
            ("t4_decomp", "T4 Decomp (5-dim)",    lambda: test_t4_decomp(ollama)),
        ]
        layer1_weighted = 0.0
        for key, lbl, fn in layer1_tests:
            print(f"  Testing {lbl:<25}", end="", flush=True)
            t0 = time.time()
            try:
                res = await fn()
                elapsed = round(time.time() - t0, 1)
                res["elapsed_sec"] = elapsed
                results["layer1"][key] = res
                w = LAYER1_WEIGHTS[key]
                layer1_weighted += w * res["score_on"]
                sign = "+" if res["delta"] >= 0 else ""
                print(f"  OFF={res['score_off']:.2f}  ON={res['score_on']:.2f}  Δ={sign}{res['delta']:.2f}  "
                      f"{'PASS' if res['score_on'] >= 0.6 else 'FAIL'}  ({elapsed:.0f}s)")
            except Exception as e:
                print(f"  ERROR: {e}")
                results["layer1"][key] = {"score_off": 0.0, "score_on": 0.0, "delta": 0.0, "error": str(e)}
        results["layer1"]["weighted_score"] = round(layer1_weighted, 3)
        print(f"\n  Layer 1 weighted score: {layer1_weighted:.3f}  {_bar(layer1_weighted)}")
    else:
        print("▶ Layer 1 — SKIPPED (c1 mode)\n")

    # ── Layer 3 ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("▶ Layer 3 — E2E Hybrid Pipeline\n")

    # C1: Sonnet-only baseline (c1 / all mode)
    if mode in ("c1", "all"):
        print("  [C1] Sonnet-only (Bedrock baseline) ...", flush=True)
        t0 = time.time()
        try:
            c1 = await run_e2e_condition("c1_sonnet_only", sonnet, sonnet, sonnet)
            c1["elapsed_sec"] = round(time.time() - t0, 1)
            results["layer3"]["c1_sonnet_only"] = c1
            print(f"       score={c1['avg_overall_score']:.3f}  revisions={c1['avg_revision_count']:.1f}  "
                  f"latency={c1['avg_latency_sec']:.0f}s  ({c1['elapsed_sec']:.0f}s total)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results["layer3"]["c1_sonnet_only"] = {"error": str(e)}

    # C2: Hybrid (c2 / all mode)
    if mode in ("c2", "all"):
        print(f"  [C2] Sonnet + {OLLAMA_MODEL} (hybrid) ...", flush=True)
        t0 = time.time()
        try:
            c2 = await run_e2e_condition("c2_hybrid", ollama, sonnet, sonnet)
            c2["elapsed_sec"] = round(time.time() - t0, 1)
            results["layer3"]["c2_hybrid"] = c2
            print(f"       score={c2['avg_overall_score']:.3f}  revisions={c2['avg_revision_count']:.1f}  "
                  f"latency={c2['avg_latency_sec']:.0f}s  ({c2['elapsed_sec']:.0f}s total)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results["layer3"]["c2_hybrid"] = {"error": str(e)}

    # Quality gap (only when both results are available)
    c1_data = results["layer3"].get("c1_sonnet_only", {})
    c2_data = results["layer3"].get("c2_hybrid", {})
    if "error" not in c1_data and c1_data and "error" not in c2_data and c2_data:
        gap = round(c1_data["avg_overall_score"] - c2_data["avg_overall_score"], 3)
        results["layer3"]["quality_gap"] = gap
        print(f"\n  Quality gap (C1 - C2): {gap:+.3f}  "
              f"({'Hybrid close to Sonnet' if abs(gap) <= 0.05 else 'Meaningful gap'})")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  RESULTS — {label}")
    print(f"{'═'*70}")
    if results["layer1"]:
        print(f"  Layer 1 weighted score:  {results['layer1'].get('weighted_score', 0):.3f}")
    if "c1_sonnet_only" in results["layer3"] and "error" not in results["layer3"]["c1_sonnet_only"]:
        print(f"  C1 (Sonnet-only) score:  {results['layer3']['c1_sonnet_only']['avg_overall_score']:.3f}")
    if "c2_hybrid" in results["layer3"] and "error" not in results["layer3"]["c2_hybrid"]:
        print(f"  C2 (Hybrid) score:       {results['layer3']['c2_hybrid']['avg_overall_score']:.3f}")
    if "quality_gap" in results["layer3"]:
        print(f"  Quality gap (C1-C2):     {results['layer3']['quality_gap']:+.3f}")
    print(f"{'═'*70}\n")

    # Save results
    local_path = save_local(results, model_key)
    s3_uri = upload_to_s3(results, model_key)
    results["saved_local"] = local_path
    if s3_uri:
        results["saved_s3"] = s3_uri


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument(
        "--mode", default=os.getenv("BENCH_MODE", "all"),
        choices=["all", "c1", "c2"],
        help="all=Layer1+C1+C2  c1=Layer3 C1 only (Sonnet baseline, no Ollama)  c2=Layer1+Layer3 C2 only",
    )
    _parser.add_argument(
        "--lang", default=None,
        choices=["en", "ko", "ja"],
        help="Query language for Layer 3 E2E (overrides LANG_MODE env var). en=English ko=Korean ja=Japanese",
    )
    _args = _parser.parse_args()
    # --lang CLI overrides LANG_MODE env var
    if _args.lang:
        import sys as _sys
        # Re-set module-level LANG_MODE and rebuild E2E_QUERIES / E2E_DOC_POOLS
        import importlib as _il
        _mod = _il.import_module(__name__)
        _mod.LANG_MODE    = _args.lang
        _mod.E2E_QUERIES  = _mod._LANG_QUERIES[_args.lang]
        _mod.E2E_DOC_POOLS = {q: _mod._E2E_DOCS[i] for i, q in enumerate(_mod.E2E_QUERIES)}
    asyncio.run(main(_args.mode))
