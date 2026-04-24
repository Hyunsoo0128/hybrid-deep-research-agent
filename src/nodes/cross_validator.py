"""
Cross-Reference Validator Node — SDP (arxiv:2604.17677)

Techniques:
  sdp  (arxiv:2604.17677): Source Dependency Pruning — removes redundant citations that are
                            fully covered by other higher-confidence sources, reducing noise
                            in the report writing stage.
  dsap (arxiv:2512.20660): JSON guard functions
"""

from __future__ import annotations

from ..state import ResearchState
from ..providers.base import LLMProvider
from ..utils.llm_json import llm_json

_SYSTEM = """You are an expert in cross-referencing information.
Objectively analyze the consistency and reliability of information collected from multiple sources."""

_PROMPT = """Cross-validate the following list of sources.

Original question: {query}

Source list ({count} total):
{citations}

Analysis criteria:
1. Identify groups of sources making identical or similar claims (corroboration)
2. Identify pairs of sources containing contradictory information
3. Identify important claims that exist in only a single source

Respond ONLY in the following JSON format:
{{
  "corroboration_groups": [
    {{
      "topic": "One-line topic description",
      "citation_ids": ["cit_id1", "cit_id2"]
    }}
  ],
  "contradictions": [
    {{
      "topic": "Contradiction topic",
      "citation_ids": ["cit_id1", "cit_id3"],
      "summary": "A claims X but B claims Y"
    }}
  ],
  "single_source_ids": ["cit_id4"],
  "quality_score": 0.75,
  "summary": "One-line overall quality assessment"
}}"""

# ── SDP: Source Dependency Pruning ────────────────────────────────────────────

_SDP_SYSTEM = """You are an expert in research source analysis.
Identify redundant sources that are fully covered by other higher-quality sources."""

_SDP_PROMPT = """Analyze the following corroboration groups and determine which sources are redundant.

Corroboration groups:
{corroboration_groups}

All citations:
{citations_brief}

For each corroboration group with 3 or more sources, identify the lowest-confidence sources
that can be pruned (their content is fully covered by higher-confidence sources in the same group).
Keep at least 2 sources per group for cross-validation integrity.

Respond ONLY in the following JSON format:
{{
  "prunable_ids": ["cit_id3", "cit_id7"],
  "reason": "These sources are fully covered by higher-confidence sources in the same corroboration group"
}}

If no sources should be pruned, return an empty array."""

_SDP_SCHEMA = '{"prunable_ids": ["string"], "reason": "string"}'


def _build_citation_list(citations: list[dict]) -> str:
    lines = []
    for c in citations:
        lines.append(
            f"[{c['id']}] {c.get('title', 'No title')}\n"
            f"  confidence: {c.get('confidence', 0):.2f} | trust: {c.get('trust_level', 'low')}\n"
            f"  content: {c.get('excerpt', '')[:150]}"
        )
    return "\n\n".join(lines)


async def _sdp_prune(
    citations: list[dict],
    corroboration_groups: list[dict],
    llm: LLMProvider,
    dsap_on: bool,
) -> list[dict]:
    """
    SDP (arxiv:2604.17677): Remove redundant citations that are fully covered
    by higher-confidence sources in the same corroboration group.
    Minimum 2 sources retained per group for cross-validation integrity.
    """
    # Only prune if there are groups with 3+ sources
    large_groups = [g for g in corroboration_groups if len(g.get("citation_ids", [])) >= 3]
    if not large_groups:
        return citations

    groups_text = "\n".join(
        f"Group '{g['topic']}': {', '.join(g['citation_ids'])}"
        for g in large_groups
    )
    citations_brief = "\n".join(
        f"[{c['id']}] conf={c.get('confidence', 0):.2f} — {c.get('title', '')[:60]}"
        for c in citations
    )

    data = await llm_json(
        llm=llm,
        messages=[{
            "role": "user",
            "content": _SDP_PROMPT.format(
                corroboration_groups=groups_text,
                citations_brief=citations_brief,
            ),
        }],
        system=_SDP_SYSTEM,
        schema_hint=_SDP_SCHEMA,
        max_tokens=256,
        temperature=0.0,
        dsap_enabled=dsap_on,
        fallback={"prunable_ids": [], "reason": ""},
    )

    prunable = set(data.get("prunable_ids", []))
    if not prunable:
        return citations

    # Safety check: never prune below 2 sources per group
    protected: set[str] = set()
    for group in large_groups:
        group_ids = group.get("citation_ids", [])
        to_prune_in_group = [cid for cid in group_ids if cid in prunable]
        keep_count = len(group_ids) - len(to_prune_in_group)
        if keep_count < 2:
            # Protect the lowest-confidence prunable sources to keep at least 2
            group_citations = sorted(
                [c for c in citations if c["id"] in set(group_ids) and c["id"] in prunable],
                key=lambda c: c.get("confidence", 0),
                reverse=True,  # highest confidence first → protect them last
            )
            need_to_protect = 2 - keep_count
            for c in group_citations[:need_to_protect]:
                protected.add(c["id"])

    final_prunable = prunable - protected
    return [c for c in citations if c["id"] not in final_prunable]


async def cross_validate(state: ResearchState, llm: LLMProvider) -> dict:
    """Cross-Reference Validator Node with optional SDP pruning."""
    # query_decomp Reranker: combine reranked initial set + gap_search additions
    reranked = state.get("reranked_citations") or []
    all_citations = state.get("citations", [])
    if reranked:
        reranked_ids = {c["id"] for c in reranked}
        gap_additions = [c for c in all_citations if c.get("id") not in reranked_ids]
        citations = reranked + gap_additions
    else:
        citations = all_citations
    flags = state.get("feature_flags", {})
    dsap_on = flags.get("dsap", True)
    sdp_on = flags.get("sdp", False)

    if len(citations) < 2:
        return {
            "cross_validation_report": {
                "corroboration_groups": [],
                "contradictions": [],
                "single_source_ids": [c["id"] for c in citations],
                "quality_score": 0.5 if citations else 0.0,
                "summary": "Cross-validation not possible due to insufficient sources",
                "well_corroborated_count": 0,
                "contradictions_found": 0,
            }
        }

    citation_list = _build_citation_list(citations[:25])

    _CV_SCHEMA = """{
  "corroboration_groups": [{"topic": "string", "citation_ids": ["string"]}],
  "contradictions": [{"topic": "string", "citation_ids": ["string"], "summary": "string"}],
  "single_source_ids": ["string"],
  "quality_score": 0.75,
  "summary": "string"
}"""

    data = await llm_json(
        llm=llm,
        messages=[{
            "role": "user",
            "content": _PROMPT.format(
                query=state["original_query"],
                count=len(citations),
                citations=citation_list,
            ),
        }],
        system=_SYSTEM,
        schema_hint=_CV_SCHEMA,
        max_tokens=1024,
        temperature=0.1,
        dsap_enabled=dsap_on,
        fallback={
            "corroboration_groups": [],
            "contradictions": [],
            "single_source_ids": [],
            "quality_score": 0.6,
            "summary": "Cross-validation parse error — using defaults",
        },
    )

    corroboration_groups = data.get("corroboration_groups", [])

    # SDP: prune redundant citations after cross-validation
    # Note: citations reducer uses operator.add, so we store pruned list in report
    # for writer to use — we do NOT modify the state citations directly.
    pruned_citations = citations
    pruned_count = 0
    if sdp_on and corroboration_groups:
        pruned_citations = await _sdp_prune(
            citations=citations,
            corroboration_groups=corroboration_groups,
            llm=llm,
            dsap_on=dsap_on,
        )
        pruned_count = len(citations) - len(pruned_citations)

    # Calculate statistics
    corroborated_ids = set()
    for group in corroboration_groups:
        for cid in group.get("citation_ids", []):
            corroborated_ids.add(cid)

    report = {
        "corroboration_groups": corroboration_groups,
        "contradictions": data.get("contradictions", []),
        "single_source_ids": data.get("single_source_ids", []),
        "quality_score": data.get("quality_score", 0.6),
        "summary": data.get("summary", ""),
        "well_corroborated_count": len(corroborated_ids),
        "contradictions_found": len(data.get("contradictions", [])),
        # SDP metadata for writer
        "sdp_pruned_count": pruned_count,
        "effective_citation_ids": [c["id"] for c in pruned_citations],
    }

    return {"cross_validation_report": report}
