"""
Plan Generator Node — Query Decomposition (arxiv:2507.00355) + STRIDE (arxiv:2604.17405)

Techniques:
  query_decomp (arxiv:2507.00355): 5-dimensional decomposition → MRR@10 +36.7%
  stride       (arxiv:2604.17405): Meta-Planner generates entity-agnostic strategy Sq first,
                                   then derives concrete plan Cq guided by Sq.
                                   When stride=False, existing decomposition operates independently.
  dsap         (arxiv:2512.20660): JSON guard functions

STRIDE implementation notes:
  - CLM/fine-tuning not required: both Meta-Planner and Cq derivation are prompted LLM calls.
  - query_decomp Reranker is orthogonal (runs in reranker_node, unaffected by STRIDE).
  - 5-dimension classification is subsumed by STRIDE Sq reasoning_steps.
    When stride=True, dimension tags come from Sq step types, not the fixed 5-dim taxonomy.
  - Previous `_stride_refine()` (coverage gap checker) replaced by proper two-step Sq→Cq flow.

Phase C — 2-stage HybridProvider routing:
  Stage 1 (local LLM): extracts a privacy-safe research profile from the original query.
    - query_decomp:  _extract_research_profile() → {topic_summary, dimensions, depth_hint}
    - stride:        _generate_abstract_strategy() + _extract_research_profile() in parallel
    The original query never leaves the local machine.
  Stage 2 (cloud LLM): receives ONLY {topic_summary, dimensions} (and Sq for STRIDE).
    - Generates the full plan with concrete, searchable sub_queries.
    - Backward-compatible: when llm is a plain provider, both stages use the same model.
"""

from __future__ import annotations
import asyncio
from ..state import ResearchPlan, SubQuery, ResearchState
from ..providers.base import LLMProvider
from ..utils.llm_json import llm_json

_SYSTEM = """You are a research planning expert.
Analyze the user's query and create a systematic, parallelizable research plan."""

_DECOMPOSITION_DIMS = """\
Each sub-query should cover a different dimension to minimize overlap:
  [Definition/Background]   Core concept definitions, historical context
  [Current State/Evidence]  Latest data, statistics, real-world examples
  [Comparison/Alternatives] Different approaches, competing technologies, analogous case comparisons
  [Cause/Mechanism]         How it works, causal relationships, reasoning analysis
  [Limitations/Challenges]  Drawbacks, risk factors, unresolved issues"""

_PROMPT = """Create a research plan for the following query.

Query: {query}

Requirements:
1. Classify the query intent: factual | analytical | comparative | predictive
2. Write one sentence describing how the system interpreted the query (for user verification)
3. Generate 4-6 independent, parallelizable sub-queries considering the following decomposition dimensions:
{decomposition_dims}
4. Attach the corresponding dimension tag to each sub-query
5. Recommend a search depth: fast (3 sources/sub-query, ~60s) | normal (5 sources/sub-query, ~2min) | deep (8 sources/sub-query, ~5min)

Respond ONLY in the following JSON format:
{{
  "intent": "analytical",
  "interpretation": "Interpreted as an analysis of technical trends related to ~",
  "sub_queries": [
    {{"id": "sq1", "question": "Specific sub-query 1", "dimension": "Current State/Evidence"}},
    {{"id": "sq2", "question": "Specific sub-query 2", "dimension": "Comparison/Alternatives"}},
    {{"id": "sq3", "question": "Specific sub-query 3", "dimension": "Limitations/Challenges"}},
    {{"id": "sq4", "question": "Specific sub-query 4", "dimension": "Cause/Mechanism"}}
  ],
  "depth": "normal",
  "estimated_time": "approx. 90 seconds"
}}"""

# Simplified prompt used when query_decomp flag is off
_PROMPT_SIMPLE = """Create a research plan for the following query.

Query: {query}

Requirements:
1. Classify the query intent: factual | analytical | comparative | predictive
2. Write one sentence describing how the system interpreted the query (for user verification)
3. Generate 3-4 independent sub-queries that cover different aspects of the topic
4. Recommend a search depth: fast | normal | deep

Respond ONLY in the following JSON format:
{{
  "intent": "analytical",
  "interpretation": "Interpreted as a comprehensive analysis of ~",
  "sub_queries": [
    {{"id": "sq1", "question": "Sub-query 1", "dimension": "General"}},
    {{"id": "sq2", "question": "Sub-query 2", "dimension": "General"}},
    {{"id": "sq3", "question": "Sub-query 3", "dimension": "General"}}
  ],
  "depth": "normal",
  "estimated_time": "approx. 90 seconds"
}}"""

_SCHEMA_HINT = """{
  "intent": "string",
  "interpretation": "string",
  "sub_queries": [{"id": "string", "question": "string", "dimension": "string"}],
  "depth": "string",
  "estimated_time": "string"
}"""


# ── Phase C: Stage 1 — Research Profile Extractor (local LLM) ───────────────
#
# Extracts a privacy-safe {topic_summary, dimensions, depth_hint} from the
# original query.  The cloud LLM (Stage 2) receives ONLY this profile —
# the original query never leaves the local machine.
#
# topic_summary should be rich enough for useful web searches but need not
# reproduce the original phrasing verbatim.

_STAGE1_SYSTEM = """You are a research topic analyst.
Extract a structured research profile from the given query.
Do not include any personally identifiable or sensitive information in the output."""

_STAGE1_PROMPT = """Analyze the following research query and extract a structured research profile.

Query: {query}

1. topic_summary: One sentence describing what this research is about.
   - Preserve domain, type of analysis, and key subject matter.
   - Write in general, briefing-style terms (e.g. "Performance comparison of LLM inference backends" rather than repeating the query verbatim).

2. dimensions: Select which of these research dimensions are most relevant (1-4 items):
   Definition/Background | Current State/Evidence | Comparison/Alternatives | Cause/Mechanism | Limitations/Challenges

3. depth_hint: Estimate search depth:
   fast (simple factual lookup) | normal (analytical, multi-source) | deep (complex, multi-faceted)

Respond ONLY in this JSON format:
{{"topic_summary": "...", "dimensions": ["...", "..."], "depth_hint": "normal"}}"""

_STAGE1_SCHEMA = '{"topic_summary": "string", "dimensions": ["string"], "depth_hint": "string"}'


# ── Phase C: Stage 2 — Hybrid plan prompts (cloud LLM receives profile, not query) ──

_PROMPT_HYBRID = """Create a research plan for the following research topic.

Research topic: {topic_summary}
Relevant dimensions identified: {dimensions_hint}

Requirements:
1. Classify the query intent: factual | analytical | comparative | predictive
2. Write one sentence describing how the system interpreted this topic (for user verification)
3. Generate 4-6 independent, parallelizable sub-queries that together cover these dimensions:
{decomposition_dims}
4. Attach the corresponding dimension tag to each sub-query
5. Recommend a search depth: fast (3 sources/sub-query, ~60s) | normal (5 sources/sub-query, ~2min) | deep (8 sources/sub-query, ~5min)

Respond ONLY in the following JSON format:
{{
  "intent": "analytical",
  "interpretation": "Interpreted as an analysis of ...",
  "sub_queries": [
    {{"id": "sq1", "question": "Specific sub-query 1", "dimension": "Current State/Evidence"}},
    {{"id": "sq2", "question": "Specific sub-query 2", "dimension": "Comparison/Alternatives"}},
    {{"id": "sq3", "question": "Specific sub-query 3", "dimension": "Limitations/Challenges"}},
    {{"id": "sq4", "question": "Specific sub-query 4", "dimension": "Cause/Mechanism"}}
  ],
  "depth": "normal",
  "estimated_time": "approx. 90 seconds"
}}"""

_STRIDE_CQ_PROMPT_HYBRID = """Create a research plan guided by an abstract reasoning strategy.

Research topic: {topic_summary}

Abstract reasoning strategy (Sq):
{strategy}

Requirements:
1. Classify the query intent: factual | analytical | comparative | predictive
2. Write one sentence describing how the system interpreted this topic (for user verification)
3. Generate one concrete, searchable sub-query for EACH reasoning step in the abstract strategy above
   - Use the research topic to make each sub-query specific and independently searchable
   - Keep the sub-query focused; avoid combining multiple steps into one
4. Recommend a search depth: fast (3 sources/sub-query, ~60s) | normal (5 sources/sub-query, ~2min) | deep (8 sources/sub-query, ~5min)

Respond ONLY in the following JSON format:
{{
  "intent": "analytical",
  "interpretation": "Interpreted as an analysis of ...",
  "sub_queries": [
    {{"id": "sq1", "question": "concrete sub-query 1", "dimension": "definition"}},
    {{"id": "sq2", "question": "concrete sub-query 2", "dimension": "current_state"}}
  ],
  "depth": "normal",
  "estimated_time": "approx. 90 seconds"
}}"""


# ── STRIDE: Meta-Planner (Sq) ─────────────────────────────────────────────────

_META_PLANNER_SYSTEM = """You are a research strategist.
Generate an entity-agnostic abstract reasoning strategy (Sq) for the given query.
Replace all specific named entities with type placeholders like [ENTITY], [ALTERNATIVES], [DOMAIN]."""

_META_PLANNER_PROMPT = """Analyze the following research query and generate an abstract reasoning strategy.

Query: {query}

Strip all specific named entities (people, products, companies, technologies, etc.) and
replace them with type placeholders (e.g. [ENTITY], [ALTERNATIVES], [DOMAIN]).
Then define the systematic reasoning steps a researcher would need to answer this type of question.

Respond ONLY in the following JSON format:
{{
  "strategy_type": "analytical",
  "reasoning_steps": [
    {{"type": "definition", "description": "Define [ENTITY] and its core characteristics"}},
    {{"type": "current_state", "description": "Current performance or status of [ENTITY]"}},
    {{"type": "mechanism", "description": "How [ENTITY] achieves its results internally"}},
    {{"type": "comparison", "description": "Compare [ENTITY] with [ALTERNATIVES]"}},
    {{"type": "limitations", "description": "Known limitations and failure modes of [ENTITY]"}}
  ],
  "entity_slots": ["[ENTITY]", "[ALTERNATIVES]"]
}}"""

_META_PLANNER_SCHEMA = """{
  "strategy_type": "string",
  "reasoning_steps": [{"type": "string", "description": "string"}],
  "entity_slots": ["string"]
}"""

_STRIDE_CQ_PROMPT = """Create a research plan for the following query, guided by an abstract reasoning strategy.

Query: {query}

Abstract reasoning strategy (Sq):
{strategy}

Requirements:
1. Classify the query intent: factual | analytical | comparative | predictive
2. Write one sentence describing how the system interpreted the query (for user verification)
3. Generate one concrete sub-query for EACH reasoning step in the abstract strategy above
   - Fill in [ENTITY] and other placeholders with the actual entities from the query
   - Keep each sub-query focused and independently searchable
4. Recommend a search depth: fast (3 sources/sub-query, ~60s) | normal (5 sources/sub-query, ~2min) | deep (8 sources/sub-query, ~5min)

Respond ONLY in the following JSON format:
{{
  "intent": "analytical",
  "interpretation": "Interpreted as an analysis of ...",
  "sub_queries": [
    {{"id": "sq1", "question": "concrete sub-query 1", "dimension": "definition"}},
    {{"id": "sq2", "question": "concrete sub-query 2", "dimension": "current_state"}}
  ],
  "depth": "normal",
  "estimated_time": "approx. 90 seconds"
}}"""


def _format_strategy(strategy: dict) -> str:
    """Format Sq for inclusion in Cq prompt."""
    lines = [f"Type: {strategy.get('strategy_type', 'analytical')}"]
    for step in strategy.get("reasoning_steps", []):
        lines.append(f"  [{step.get('type', '')}] {step.get('description', '')}")
    return "\n".join(lines)


async def _generate_abstract_strategy(
    query: str,
    llm: LLMProvider,
    dsap_on: bool,
) -> dict:
    """
    STRIDE Meta-Planner: generate entity-agnostic strategy Sq from query.
    Returns Sq dict with strategy_type, reasoning_steps, entity_slots.
    Falls back to a minimal default Sq if LLM fails.
    """
    data = await llm_json(
        llm=llm,
        messages=[{"role": "user", "content": _META_PLANNER_PROMPT.format(query=query)}],
        system=_META_PLANNER_SYSTEM,
        schema_hint=_META_PLANNER_SCHEMA,
        max_tokens=512,
        temperature=0.2,
        dsap_enabled=dsap_on,
        fallback={
            "strategy_type": "analytical",
            "reasoning_steps": [
                {"type": "definition", "description": "Define [ENTITY] and its core characteristics"},
                {"type": "current_state", "description": "Current status and evidence for [ENTITY]"},
                {"type": "comparison", "description": "Compare [ENTITY] with [ALTERNATIVES]"},
                {"type": "limitations", "description": "Known limitations of [ENTITY]"},
            ],
            "entity_slots": ["[ENTITY]"],
        },
    )
    return data


async def _extract_research_profile(
    query: str,
    llm: LLMProvider,
    dsap_on: bool,
) -> dict:
    """
    Phase C Stage 1 (local LLM): extract privacy-safe research profile.
    Returns {topic_summary, dimensions, depth_hint}.
    Original query stays local; only this profile is forwarded to Stage 2.
    """
    data = await llm_json(
        llm=llm,
        messages=[{"role": "user", "content": _STAGE1_PROMPT.format(query=query)}],
        system=_STAGE1_SYSTEM,
        schema_hint=_STAGE1_SCHEMA,
        max_tokens=256,
        temperature=0.1,
        dsap_enabled=dsap_on,
        fallback={
            "topic_summary": query[:200],  # minimal fallback: truncated query
            "dimensions": ["Current State/Evidence", "Comparison/Alternatives"],
            "depth_hint": "normal",
        },
    )
    return data


async def generate_plan(state: ResearchState, llm: LLMProvider) -> dict:
    """
    Plan Generator Node.
    LangGraph node function signature: (state) -> dict (state update)

    Phase C — 2-stage HybridProvider routing:
      Stage 1 (local_llm): extracts research profile from original query (stays local).
      Stage 2 (cloud_llm): generates full plan from profile only (no original query).
      When llm is a plain provider: single-stage, backward-compatible.
    """
    query = state["original_query"]
    flags = state.get("feature_flags", {})
    query_decomp_on = flags.get("query_decomp", True)
    stride_on = flags.get("stride", False)
    dsap_on = flags.get("dsap", True)

    # Phase C: split local/cloud when HybridProvider is active.
    # Graceful degradation: plain provider → both stages use same model.
    is_hybrid = hasattr(llm, "local")
    local_llm: LLMProvider = llm.local if is_hybrid else llm
    cloud_llm: LLMProvider = llm.cloud if is_hybrid else llm

    if stride_on:
        if is_hybrid:
            # Stage 1 (local, parallel): Meta-Planner Sq + research profile
            abstract_strategy, profile = await asyncio.gather(
                _generate_abstract_strategy(query, local_llm, dsap_on),
                _extract_research_profile(query, local_llm, dsap_on),
            )
            # Stage 2 (cloud): Cq from topic_summary + Sq — original query stays local
            user_content = _STRIDE_CQ_PROMPT_HYBRID.format(
                topic_summary=profile["topic_summary"],
                strategy=_format_strategy(abstract_strategy),
            )
            plan_llm = cloud_llm
        else:
            # Single-stage: existing STRIDE behavior (query sent to llm directly)
            abstract_strategy = await _generate_abstract_strategy(query, llm, dsap_on)
            user_content = _STRIDE_CQ_PROMPT.format(
                query=query,
                strategy=_format_strategy(abstract_strategy),
            )
            plan_llm = llm

    elif query_decomp_on:
        if is_hybrid:
            # Stage 1 (local): extract privacy-safe research profile
            profile = await _extract_research_profile(query, local_llm, dsap_on)
            # Stage 2 (cloud): generate plan from profile — original query stays local
            user_content = _PROMPT_HYBRID.format(
                topic_summary=profile["topic_summary"],
                dimensions_hint=", ".join(profile.get("dimensions", [])),
                decomposition_dims=_DECOMPOSITION_DIMS,
            )
            plan_llm = cloud_llm
        else:
            user_content = _PROMPT.format(query=query, decomposition_dims=_DECOMPOSITION_DIMS)
            plan_llm = llm

    else:
        # Simple mode: always local (no cloud call needed for basic decomposition)
        user_content = _PROMPT_SIMPLE.format(query=query)
        plan_llm = local_llm

    fallback_query_ref = profile["topic_summary"] if (is_hybrid and (stride_on or query_decomp_on)) else query
    data = await llm_json(
        llm=plan_llm,
        messages=[{"role": "user", "content": user_content}],
        system=_SYSTEM,
        schema_hint=_SCHEMA_HINT,
        max_tokens=1024,
        temperature=0.3,
        dsap_enabled=dsap_on,
        fallback={
            "intent": "analytical",
            "interpretation": f"Interpreted as a comprehensive analysis of '{fallback_query_ref}'",
            "sub_queries": [
                {"id": "sq1", "question": fallback_query_ref, "dimension": "Current State/Evidence"},
                {"id": "sq2", "question": f"{fallback_query_ref} latest trends and statistics", "dimension": "Current State/Evidence"},
                {"id": "sq3", "question": f"{fallback_query_ref} comparisons and alternatives", "dimension": "Comparison/Alternatives"},
                {"id": "sq4", "question": f"{fallback_query_ref} limitations and challenges", "dimension": "Limitations/Challenges"},
            ],
            "depth": "normal",
            "estimated_time": "approx. 90 seconds",
        },
    )

    raw_sub_queries = data.get("sub_queries", [])

    # query_decomp (arxiv:2507.00355): Q = {q} ∪ Decompose(q)
    # Always prepend a direct lookup for the original query so it is searched in parallel
    # alongside all decomposed sub-queries. The reranker scores everything against the
    # original query and keeps the top-k, so the direct hit is not wasted.
    # When stride=True: sq0 still prepended — Reranker contribution is orthogonal to STRIDE.
    if query_decomp_on or stride_on:
        original_sq = {"id": "sq0", "question": query, "dimension": "Original Query"}
        raw_sub_queries = [original_sq] + [sq for sq in raw_sub_queries if sq.get("id") != "sq0"]

    plan = ResearchPlan(
        intent=data.get("intent", "analytical"),
        interpretation=data.get("interpretation", ""),
        sub_queries=[
            SubQuery(id=sq["id"], question=sq["question"])
            for sq in raw_sub_queries
        ],
        local_files=[],
        depth=data.get("depth", "normal"),
        estimated_time=data.get("estimated_time", "approx. 90 seconds"),
    )

    return {"plan": plan.to_dict(), "plan_approved": False}
