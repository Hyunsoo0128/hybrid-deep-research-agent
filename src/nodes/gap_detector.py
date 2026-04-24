"""
Knowledge Gap Detector Node

Detects under-covered areas in collected citations and generates targeted gap queries.
Integrates signals from CRAG verdicts, VCM checklist, and STRIDE supervisor decisions.
dsap (arxiv:2512.20660): JSON guard functions applied to LLM calls.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone

from ..state import ResearchState, Citation, SourceType, TrustLevel
from ..providers.base import LLMProvider
from ..tools.search import SearchTool
from ..security.injection_filter import InjectionFilter
from ..utils.llm_json import llm_json

_MIN_CITATIONS_PER_QUERY = 2

_GAP_DEPTH_PARAMS = {
    "fast":   {"max_gap_queries": 2, "citation_limit": 15, "max_results": 3, "max_fetch": 1},
    "normal": {"max_gap_queries": 3, "citation_limit": 25, "max_results": 5, "max_fetch": 2},
    "deep":   {"max_gap_queries": 5, "citation_limit": 50, "max_results": 8, "max_fetch": 3},
}

_GAP_SYSTEM = """You are an expert in research coverage analysis.
Objectively evaluate whether the collected information sufficiently covers the research question."""

_GAP_PROMPT = """Analyze the following research plan and collected sources.

Original question: {query}

Sub-query list:
{sub_queries}

Collected sources ({citation_count} total):
{citation_summary}

{crag_hint}{vcm_hint}{stride_hint}Evaluation criteria:
1. Check whether each sub-query is supported by at least {min_citations} sources
2. Check whether any important perspectives or data are missing
3. [VCM] Prioritize gap queries that address pending or partial subgoals listed above
4. [STRIDE] Include all STRIDE rewrite sub-queries above as gap queries

Respond ONLY in the following JSON format:
{{
  "gaps": [
    {{
      "sub_query_id": "sq1",
      "issue": "No competitor comparison data",
      "gap_query": "Deep research service Perplexity vs ChatGPT comparison 2024",
      "uncertainty": 0.8
    }}
  ],
  "coverage_score": 0.7,
  "comment": "One-line overall coverage assessment"
}}

uncertainty is a value from 0.0 to 1.0 indicating how uncertain/under-covered this area is.
If there are no gaps, return an empty array for gaps and a coverage_score of 0.8 or higher."""

_GAP_SCHEMA = """{
  "gaps": [{"sub_query_id": "string", "issue": "string", "gap_query": "string", "uncertainty": 0.8}],
  "coverage_score": 0.7,
  "comment": "string"
}"""


async def detect_gaps(state: ResearchState, llm: LLMProvider) -> dict:
    """
    Knowledge Gap Detector Node.
    CURE: uncertainty-aware gap prioritization.
    AutoSearch: LLM query reformulation (applied in gap_search node).
    """
    # query_decomp Reranker: use filtered set when available (more precise coverage signal)
    citations = state.get("reranked_citations") or state.get("citations", [])
    plan = state.get("plan") or {}
    sub_queries = plan.get("sub_queries", [])
    depth = plan.get("depth", "normal")
    params = _GAP_DEPTH_PARAMS.get(depth, _GAP_DEPTH_PARAMS["normal"])
    flags = state.get("feature_flags", {})
    dsap_on = flags.get("dsap", True)

    if not citations or not sub_queries:
        return {"identified_gaps": [], "gap_queries": [], "retrieval_quality": []}

    # Build CRAG retrieval quality hints (D1 + AMBIGUOUS extension)
    # INCORRECT (uncertainty=1.0): forced high-priority gaps
    # AMBIGUOUS (uncertainty=0.5): lower-priority supplementary hints
    retrieval_quality = state.get("retrieval_quality", [])
    sq_id_to_question = {sq["id"]: sq["question"] for sq in sub_queries}

    incorrect_ids = [
        rq["sub_query_id"] for rq in retrieval_quality
        if rq.get("verdict") == "INCORRECT"
    ]
    ambiguous_ids = [
        rq["sub_query_id"] for rq in retrieval_quality
        if rq.get("verdict") == "AMBIGUOUS"
    ]

    hint_parts: list[str] = []
    if incorrect_ids:
        incorrect_questions = [sq_id_to_question.get(sid, sid) for sid in incorrect_ids]
        hint_parts.append(
            "CRAG signal — INCORRECT (low-quality results, uncertainty=1.0, MUST include as gaps):\n"
            + "\n".join(f"  - {q}" for q in incorrect_questions)
        )
    if ambiguous_ids:
        ambiguous_questions = [sq_id_to_question.get(sid, sid) for sid in ambiguous_ids]
        hint_parts.append(
            "CRAG signal — AMBIGUOUS (partial results, uncertainty=0.5, include as gaps if coverage is low):\n"
            + "\n".join(f"  - {q}" for q in ambiguous_questions)
        )

    crag_hint = ("\n\n".join(hint_parts) + "\n\n") if hint_parts else ""

    # Build VCM subgoal completion hints (V1b independent axis)
    # PENDING (no evidence): high priority — checklist status set by evidence_auditor
    # PARTIAL (insufficient evidence): medium priority
    # COMPLETE: excluded from hints
    # Known deviation: checklist is updated by evidence_auditor (after cross_validator),
    # so this hint is only effective from the 2nd gap iteration onward (deep mode).
    checklist = state.get("checklist", [])
    vcm_hint = ""
    if checklist:
        pending_subgoals = [
            item for item in checklist if item.get("status") == "pending"
        ]
        partial_subgoals = [
            item for item in checklist if item.get("status") == "partial"
        ]
        vcm_parts: list[str] = []
        if pending_subgoals:
            vcm_parts.append(
                "VCM signal — PENDING subgoals (no evidence yet, high priority):\n"
                + "\n".join(f"  - [{item.get('sub_query_id', '')}] {item.get('subgoal', '')}" for item in pending_subgoals)
            )
        if partial_subgoals:
            vcm_parts.append(
                "VCM signal — PARTIAL subgoals (insufficient evidence, medium priority):\n"
                + "\n".join(f"  - [{item.get('sub_query_id', '')}] {item.get('subgoal', '')}" for item in partial_subgoals)
            )
        if vcm_parts:
            vcm_hint = "\n\n".join(vcm_parts) + "\n\n"

    # Build STRIDE Supervisor rewrite hints (S2 decision layer)
    # "rewrite" decisions: sub-query needs reformulation → high-priority gap query
    supervisor_decisions = state.get("supervisor_decisions", [])
    stride_hint = ""
    if supervisor_decisions:
        rewrite_items = [
            d for d in supervisor_decisions
            if d.get("action") == "rewrite" and d.get("reformulated_question")
        ]
        if rewrite_items:
            stride_hint = (
                "STRIDE Supervisor — REWRITE sub-queries (reformulations, high priority):\n"
                + "\n".join(
                    f"  - [{d['sub_query_id']}] {d['reformulated_question']}"
                    for d in rewrite_items
                )
                + "\n\n"
            )

    citation_summary_lines = []
    for i, c in enumerate(citations[:params["citation_limit"]], 1):
        citation_summary_lines.append(
            f"[{i}] {c.get('title', 'No title')} (confidence: {c.get('confidence', 0):.1f})\n"
            f"    {c.get('excerpt', '')[:100]}"
        )

    sub_queries_text = "\n".join(
        f"- [{sq['id']}] {sq['question']}"
        for sq in sub_queries
    )

    data = await llm_json(
        llm=llm,
        messages=[{
            "role": "user",
            "content": _GAP_PROMPT.format(
                query=state["original_query"],
                sub_queries=sub_queries_text,
                citation_count=len(citations),
                citation_summary="\n".join(citation_summary_lines),
                min_citations=_MIN_CITATIONS_PER_QUERY,
                crag_hint=crag_hint,
                vcm_hint=vcm_hint,
                stride_hint=stride_hint,
            ),
        }],
        system=_GAP_SYSTEM,
        schema_hint=_GAP_SCHEMA,
        max_tokens=512,
        temperature=0.2,
        dsap_enabled=dsap_on,
        fallback={"gaps": [], "coverage_score": 0.8, "comment": "Gap detection failed"},
    )

    gaps = data.get("gaps", [])
    gaps = gaps[:params["max_gap_queries"]]

    identified_gaps = [g["issue"] for g in gaps if "issue" in g]
    gap_queries = [g["gap_query"] for g in gaps if "gap_query" in g]

    return {
        "identified_gaps": identified_gaps,
        "gap_queries": gap_queries,
    }


async def gap_search(
    state: ResearchState,
    llm: LLMProvider,
    search_tool: SearchTool,
) -> dict:
    """Gap Search Node — executes targeted queries to fill detected knowledge gaps."""
    gap_queries = state.get("gap_queries", [])
    if not gap_queries:
        return {
            "citations": [],
            "research_round": state.get("research_round", 0) + 1,
            "gap_queries": [],
            "identified_gaps": [],
        }

    plan = state.get("plan") or {}
    depth = plan.get("depth", "normal")
    params = _GAP_DEPTH_PARAMS.get(depth, _GAP_DEPTH_PARAMS["normal"])
    injection_filter = InjectionFilter()
    new_citations: list[dict] = []

    for query in gap_queries:
        results = await search_tool.search_async(query=query, max_results=params["max_results"])
        fetched_count = 0

        for result in results:
            if result.relevance_score < 0.6:
                continue

            if fetched_count >= params["max_fetch"]:
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
                    confidence=result.relevance_score * 0.6,
                    injection_checked=True,
                ).to_dict())
                continue

            fetch_result = await search_tool.fetch_page_async(result.url)
            content = fetch_result.content if fetch_result.success else result.summary
            filter_result = injection_filter.check(content, result.url)

            excerpt = await llm.complete(
                messages=[{
                    "role": "user",
                    "content": (
                        f"Research question: {query}\n\n"
                        f"Web page content:\n{filter_result.sanitized_content[:1500]}\n\n"
                        "Summarize the key information relevant to the research question in 150 characters or less."
                    ),
                }],
                system="You are an expert in information extraction.",
                max_tokens=200,
                temperature=0.1,
            )
            fetched_count += 1

            cid = f"cit_{uuid.uuid4().hex[:8]}"
            new_citations.append(Citation(
                id=cid,
                url=result.url,
                title=result.title,
                excerpt=excerpt.strip() if "No relevant content found" not in excerpt else filter_result.sanitized_content,
                source_type=SourceType.WEB,
                trust_level=TrustLevel(filter_result.trust_level),
                crawled_at=datetime.now(timezone.utc).isoformat(),
                confidence=result.relevance_score,
                injection_checked=True,
            ).to_dict())

    return {
        "citations": new_citations,
        "research_round": state.get("research_round", 0) + 1,
        "gap_queries": [],
        "identified_gaps": [],
    }
