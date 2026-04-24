"""
EAM (Evidence Audit Module) — RhinoInsight (arxiv:2511.18743)

Runs in two passes:

Stage 1 (audit_evidence node — after cross_validator):
  1. Collect effective citations (reranked set + gap_search additions)
  2. URL-dedup (catches any remaining duplicates after gap_search)
  3. Sort by confidence descending
  4. Assign verification_level: "corroborated" | "single_source" | "unverified"
  5. Stage 2a: bind MASS-RAG key_spans to evidence items (claim_bindings)
  6. Update checklist status using claim_bindings (falls back to keyword overlap)

Stage 2b (annotate_misalignments node — after critique):
  Annotates evidence_store with misalignment_flags from alignrag critic_feedback.
  Each evidence item flagged by alignrag gets misalignment_flags:
    [{phase, claim, correction_hint}] per misaligned citation reference.
  Enables reviser to target specific citations for correction.
"""

from __future__ import annotations
from ..state import ResearchState

_MIN_EVIDENCE_FOR_COMPLETE = 2   # citations per sub_query to mark "complete"
_MIN_EVIDENCE_FOR_PARTIAL  = 1   # citations per sub_query to mark "partial"


def _get_effective_citations(state: ResearchState) -> list[dict]:
    """
    Combine reranked initial set + gap_search additions.
    Mirrors cross_validator's logic so evidence_store matches what cross_validator saw.
    """
    reranked = state.get("reranked_citations") or []
    all_citations = state.get("citations", [])
    if reranked:
        reranked_ids = {c["id"] for c in reranked}
        gap_additions = [c for c in all_citations if c.get("id") not in reranked_ids]
        return reranked + gap_additions
    return all_citations


def _dedup_by_url(citations: list[dict]) -> list[dict]:
    """URL-based dedup; keep highest-confidence per URL."""
    seen: dict[str, dict] = {}
    for c in citations:
        url = c.get("url", "")
        if not url:
            continue
        if url not in seen or c.get("confidence", 0) > seen[url].get("confidence", 0):
            seen[url] = c
    no_url = [c for c in citations if not c.get("url", "")]
    return list(seen.values()) + no_url


def _assign_verification_level(cit: dict, validation_report: dict | None) -> str:
    """
    Derive verification_level from cross_validation_report without an extra LLM call.
    corroborated: appears in a corroboration group
    single_source: in single_source_ids list
    unverified: not found in report
    """
    if not validation_report:
        return "unverified"
    cid = cit.get("id", "")
    corroborated_ids: set[str] = set()
    for group in validation_report.get("corroboration_groups", []):
        for gid in group.get("citation_ids", []):
            corroborated_ids.add(gid)
    if cid in corroborated_ids:
        return "corroborated"
    if cid in set(validation_report.get("single_source_ids", [])):
        return "single_source"
    return "unverified"


def _bind_claims_from_mass_rag(
    evidence_store: list[dict],
    mass_rag_outputs: list[dict],
) -> list[dict]:
    """
    Stage 2a: Bind MASS-RAG key_spans to evidence_store items.

    For each evidence item, collects key_spans that cite it (via source_citation_ids)
    and stores them as claim_bindings: [{sub_query_id, text, type}].
    """
    cit_spans: dict[str, list[dict]] = {}
    for output in mass_rag_outputs:
        sq_id = output.get("sub_query_id", "")
        for span in output.get("key_spans", []):
            for cid in span.get("source_citation_ids", []):
                cit_spans.setdefault(cid, []).append({
                    "sub_query_id": sq_id,
                    "text": span.get("text", "")[:200],
                    "type": span.get("type", ""),
                })

    enriched = []
    for ev in evidence_store:
        cid = ev.get("id", "")
        enriched.append({**ev, "claim_bindings": cit_spans.get(cid, [])})
    return enriched


def _keyword_coverage(
    evidence_store: list[dict],
    sub_queries: list[dict],
) -> dict[str, int]:
    """Keyword-overlap heuristic for checklist coverage (Stage 1 fallback)."""
    sq_keywords: dict[str, set[str]] = {}
    for sq in sub_queries:
        words = set(sq["question"].lower().split())
        sq_keywords[sq["id"]] = {w for w in words if len(w) > 3}

    coverage: dict[str, int] = {sq["id"]: 0 for sq in sub_queries}
    for ev in evidence_store:
        ev_text = f"{ev.get('title', '')} {ev.get('excerpt', '')}".lower()
        for sq_id, keywords in sq_keywords.items():
            if not keywords:
                continue
            overlap = sum(1 for kw in keywords if kw in ev_text)
            if overlap >= max(1, len(keywords) // 3):
                coverage[sq_id] = coverage.get(sq_id, 0) + 1
    return coverage


def _claim_binding_coverage(
    evidence_store: list[dict],
    sub_queries: list[dict],
) -> dict[str, int]:
    """Count claim_bindings per sub_query_id for checklist coverage (Stage 2a)."""
    coverage: dict[str, int] = {sq["id"]: 0 for sq in sub_queries}
    for ev in evidence_store:
        for binding in ev.get("claim_bindings", []):
            sq_id = binding.get("sub_query_id", "")
            if sq_id in coverage:
                coverage[sq_id] += 1
    return coverage


def _update_checklist(
    checklist: list[dict],
    evidence_store: list[dict],
    sub_queries: list[dict],
) -> list[dict]:
    """
    Update checklist item status.
    Uses claim_bindings coverage when available (Stage 2a), falls back to keyword overlap.
    """
    has_bindings = any(ev.get("claim_bindings") for ev in evidence_store)
    if has_bindings:
        coverage = _claim_binding_coverage(evidence_store, sub_queries)
    else:
        coverage = _keyword_coverage(evidence_store, sub_queries)

    updated = []
    for item in checklist:
        sq_id = item.get("sub_query_id", "")
        count = coverage.get(sq_id, 0)
        if count >= _MIN_EVIDENCE_FOR_COMPLETE:
            status = "complete"
        elif count >= _MIN_EVIDENCE_FOR_PARTIAL:
            status = "partial"
        else:
            status = "pending"
        updated.append({**item, "status": status})
    return updated


async def audit_evidence(state: ResearchState) -> dict:
    """
    EAM Stage 1 + 2a Node.

    Stage 1:
      1. Collect effective citations (reranked + gap additions)
      2. URL-dedup
      3. Sort by confidence descending
      4. Assign verification_level from cross_validation_report

    Stage 2a (when mass_rag enabled):
      5. Bind MASS-RAG key_spans to evidence items (claim_bindings)
      6. Update checklist using claim_bindings coverage (else keyword overlap)

    Skipped when rhinoinsight flag is off; passes through unchanged state.
    """
    feature_flags = state.get("feature_flags", {})
    if not feature_flags.get("rhinoinsight", False):
        return {}

    effective = _get_effective_citations(state)
    deduped   = _dedup_by_url(effective)
    sorted_ev = sorted(deduped, key=lambda c: c.get("confidence", 0), reverse=True)

    validation_report = state.get("cross_validation_report")

    evidence_store = [
        {
            "id":                 c.get("id", ""),
            "url":                c.get("url", ""),
            "title":              c.get("title", ""),
            "excerpt":            c.get("excerpt", ""),
            "confidence":         c.get("confidence", 0.0),
            "trust_level":        c.get("trust_level", "low"),
            "crawled_at":         c.get("crawled_at", ""),
            "verification_level": _assign_verification_level(c, validation_report),
            "claim_bindings":     [],
            "misalignment_flags": [],
        }
        for c in sorted_ev
    ]

    # Stage 2a: claim binding from MASS-RAG key_spans
    mass_rag_outputs = state.get("mass_rag_outputs", [])
    if mass_rag_outputs:
        evidence_store = _bind_claims_from_mass_rag(evidence_store, mass_rag_outputs)

    # Update checklist
    checklist = state.get("checklist", [])
    plan = state.get("plan") or {}
    sub_queries = plan.get("sub_queries", [])
    updated_checklist = _update_checklist(checklist, evidence_store, sub_queries)

    return {
        "evidence_store": evidence_store,
        "checklist":      updated_checklist,
    }


def annotate_misalignments(state: ResearchState) -> dict:
    """
    EAM Stage 2b Node — runs after critique, before should_revise.

    Annotates evidence_store items with misalignment_flags derived from
    alignrag critic_feedback.misaligned_claims.source_citation_ids.

    Each flagged evidence item gets:
      misalignment_flags: [{phase, claim (truncated), correction_hint}]

    This gives the reviser precise per-citation correction targets,
    and enables writer (on next pass) to surface flagged sources.

    Skipped when rhinoinsight is off or critic_feedback has no misaligned_claims.
    """
    feature_flags = state.get("feature_flags", {})
    if not feature_flags.get("rhinoinsight", False):
        return {}

    evidence_store = state.get("evidence_store", [])
    if not evidence_store:
        return {}

    critic_feedback = state.get("critic_feedback") or {}
    misaligned = critic_feedback.get("misaligned_claims", [])
    if not misaligned:
        return {}

    # Build citation_id → misalignment records index
    cit_misalignments: dict[str, list[dict]] = {}
    for item in misaligned:
        if not isinstance(item, dict):
            continue
        for cid in item.get("source_citation_ids", []):
            cit_misalignments.setdefault(cid, []).append({
                "phase":           item.get("phase", ""),
                "claim":           item.get("claim", "")[:200],
                "correction_hint": item.get("correction_hint", ""),
            })

    if not cit_misalignments:
        return {}

    annotated = [
        {**ev, "misalignment_flags": cit_misalignments.get(ev.get("id", ""), [])}
        for ev in evidence_store
    ]
    return {"evidence_store": annotated}
