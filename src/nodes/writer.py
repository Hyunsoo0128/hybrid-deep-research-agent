"""
Draft Writer Node — Section-by-section generation

To overcome the max_tokens limit of a single LLM call,
the report is generated section by section and then combined.

Generation structure:
  1. Executive Summary      (independent call, ~2k tokens)
  2. Key Findings           (independent call, ~3k tokens)
  3. Detailed Analysis      (parallel calls per sub-query, ~4k tokens each)
  4. Conclusion             (independent call, ~2k tokens)
  5. References             (formatted without LLM)

Max output for depth=deep with 6 sub-queries:
  2k + 3k + (6 x 4k) + 2k = 31k tokens ~ 15,000+ words
"""

from __future__ import annotations
from ..state import ResearchState
from ..providers.base import LLMProvider

_SYSTEM = """You are a professional research report writer.
You MUST attach a citation number in the format [Source N] after each claim.
Mark single-source claims with "single source" and cross-verified claims with "cross-verified"."""

# ── Section-specific prompts ───────────────────────────────────────────────

_SUMMARY_PROMPT = """Write the **Executive Summary** section of a research report based on the following information.

Original question: {query}
Query interpretation: {interpretation}

Cross-validation results:
{validation_summary}

Collected sources ({citation_count} total):
{evidence}

Writing rules:
- Summarize key content in 4-6 sentences
- Focus on the most important findings
- Include [Source N] citations
- Mention contradictory information if present

## Executive Summary"""

_FINDINGS_PROMPT = """Write the **Key Findings** section of a research report based on the following information.

Original question: {query}

Cross-validation results:
{validation_summary}

Collected sources:
{evidence}

Writing rules:
- Numbered list format (6-12 items)
- [Source N] citation required for each item
- Sort by importance
- Mark contradictory information with a warning and present both views

## Key Findings"""

_ANALYSIS_PROMPT = """Write a deep analysis section for the following sub-query.

Original question: {query}
Analysis target: {sub_query}

Collected sources:
{evidence}

Writing rules:
- Deep multi-perspective analysis of this topic (no length limit, be sufficiently detailed)
- Divide into multiple paragraphs
- [Source N] citation required
- Include specific numbers, statistics, and examples if available
- Present both sides when there are conflicting views

## {section_title}"""

# MASS-RAG primary-source variant: used when synthesis is available (mass_rag=True).
# Synthesis = authoritative answer for this sub-query; evidence = citation support only.
_ANALYSIS_WITH_MASSRAG_PROMPT = """Write the analysis section for the following sub-query.

Original question: {query}
Analysis target: {sub_query}

PRIMARY SOURCE — MASS-RAG Synthesis (authoritative answer for this sub-query):
{synthesis_block}

SUPPORTING EVIDENCE (for [Source N] citation numbering only — do not introduce new claims beyond the synthesis above):
{evidence}

Writing rules:
1. The synthesis above IS the answer to this sub-query. Your primary job is to present it clearly and in depth.
2. Use key_spans from the synthesis as the core claims. Each key span must appear with a [Source N] citation drawn from the supporting evidence.
3. Inferences from the synthesis must be preserved as explicit analytical claims in your text.
4. Do NOT introduce claims not grounded in the synthesis. If supporting evidence raises a point the synthesis omits, note it as "(additional source: ...)" without making it a main claim.
5. If supporting evidence contradicts the synthesis, note it as "(conflicting source: ...)" and present both views.
6. Divide into multiple paragraphs. No length limit — be sufficiently detailed.

## {section_title}"""

_BRIEF_PROMPT = """Write a concise research report based on the following information.

Original question: {query}
Query interpretation: {interpretation}

Cross-validation results:
{validation_summary}

Collected sources ({citation_count} total):
{evidence}

Writing rules:
- Write the entire report as a single flow (Executive Summary → Key Findings → Conclusion)
- Include [Source N] citations for each claim
- Be concise but include all key information
- Mark contradictory information with a warning and present both views

# {query}"""

_CONCLUSION_PROMPT = """Write the **Conclusion** section of a research report based on the following information.

Original question: {query}
Query interpretation: {interpretation}

Research summary:
{evidence_summary}

Writing rules:
- Synthesize the overall research findings
- Include practical implications
- [Source N] citations
- Mention limitations and areas requiring further research

## Conclusion"""


# ── Phase F: Privacy boundary helpers ────────────────────────────────────

def _local_citation_ids(citations: list[dict]) -> set[str]:
    """Return the set of citation IDs that originated from local files."""
    return {
        c["id"]
        for c in citations
        if c.get("source_type") in ("local", "LOCAL")
    }


def _get_mass_rag_synthesis_text(mass_rag_outputs: list[dict], citation_id: str) -> str | None:
    """
    Find MASS-RAG synthesis text for a citation ID.
    MASS-RAG key_spans link source_citation_ids to sub-query synthesis entries.
    Returns the summary text of the first matching synthesis entry, or None.
    """
    for entry in mass_rag_outputs:
        for span in entry.get("key_spans", []):
            if citation_id in span.get("source_citation_ids", []):
                summary = entry.get("summary", "").strip()
                return summary if summary else None
    return None


def _sanitize_evidence_for_privacy(
    effective_citations: list[dict],
    citations: list[dict],
    mass_rag_outputs: list[dict],
) -> list[dict]:
    """
    Phase F: Replace raw excerpts of local-file citations with MASS-RAG synthesis text.
    If no synthesis is available for a local entry, redact the excerpt.

    Privacy guarantee: raw local file content never reaches the cloud writer LLM.
    """
    local_ids = _local_citation_ids(citations)
    if not local_ids:
        return effective_citations

    sanitized = []
    for entry in effective_citations:
        if entry.get("id") not in local_ids:
            sanitized.append(entry)
            continue

        synthesis = _get_mass_rag_synthesis_text(mass_rag_outputs, entry["id"])
        if synthesis:
            # Replace raw excerpt with the local-LLM-abstracted synthesis
            sanitized.append({**entry, "excerpt": f"[MASS-RAG synthesis] {synthesis}"})
        else:
            # No synthesis available — redact to prevent raw content reaching cloud
            sanitized.append({**entry, "excerpt": "[local source — content redacted in privacy mode]"})

    return sanitized


# ── End Phase F helpers ───────────────────────────────────────────────────


# ── Helper functions ──────────────────────────────────────────────────────

def _build_evidence(citations: list[dict], max_count: int | None = None) -> str:
    items = citations[:max_count] if max_count else citations
    lines = []
    for i, c in enumerate(items, 1):
        trust_badge = {"high": "✓", "medium": "·", "low": "?"}.get(c.get("trust_level", ""), "?")
        lines.append(
            f"[Source {i}] {trust_badge} {c['title']}\n"
            f"URL: {c['url']}\n"
            f"Content: {c['excerpt']}\n"
        )
    return "\n".join(lines)


def _build_evidence_summary(citations: list[dict]) -> str:
    """Brief summary for conclusion (titles only)"""
    lines = []
    for i, c in enumerate(citations, 1):
        lines.append(f"[Source {i}] {c['title']}")
    return "\n".join(lines)


def _build_references(citations: list[dict]) -> str:
    lines = ["## References"]
    for i, c in enumerate(citations, 1):
        lines.append(f"{i}. [{c['title']}]({c['url']})")
    return "\n".join(lines)


def _get_synthesis_entry(mass_rag_outputs: list[dict], sub_query_id: str) -> dict | None:
    """Return the MASS-RAG synthesis entry for a given sub_query_id, or None if absent/empty."""
    entry = next((m for m in mass_rag_outputs if m.get("sub_query_id") == sub_query_id), None)
    if not entry or not entry.get("summary"):
        return None
    return entry


def _build_synthesis_block(entry: dict) -> str:
    """
    Format MASS-RAG synthesis entry as the PRIMARY SOURCE block for _ANALYSIS_WITH_MASSRAG_PROMPT.
    Structures summary, key_spans, and inferences clearly for the writer LLM.
    """
    lines = []

    summary = entry.get("summary", "").strip()
    if summary:
        lines.append(f"Summary: {summary}")

    key_spans = entry.get("key_spans", [])
    if key_spans:
        lines.append("\nKey spans:")
        for i, span in enumerate(key_spans[:6]):
            span_type = span.get("type", "fact")
            text = span.get("text", "")
            src_ids = ", ".join(span.get("source_citation_ids", []))
            lines.append(f"  [{i}] ({span_type}) \"{text}\" — sources: {src_ids}")

    inferences = entry.get("inferences", [])
    if inferences:
        lines.append("\nInferences:")
        for i, inf in enumerate(inferences[:4]):
            claim = inf.get("claim", "")
            span_indices = inf.get("supporting_span_indices", [])
            support = f" (key span indices: {span_indices})" if span_indices else ""
            lines.append(f"  [{i}] {claim}{support}")

    # C4b: CONSTRUCT trust score hint — qualify low-trust fields
    trust_scores = entry.get("trust_scores")
    if trust_scores:
        untrustworthy = trust_scores.get("untrustworthy_fields", [])
        per_field = trust_scores.get("per_field", {})
        doc_score = trust_scores.get("document_score", 1.0)
        trust_lines = [f"\nTrust scores (CONSTRUCT): document={doc_score:.2f}"]
        for field_name, score in per_field.items():
            flag = " [LOW TRUST — qualify with hedging language]" if field_name in untrustworthy else ""
            trust_lines.append(f"  {field_name}: {score:.2f}{flag}")
        if untrustworthy:
            trust_lines.append(
                f"  NOTE: {', '.join(untrustworthy)} scored below trust threshold. "
                "Present claims from these fields as provisional or uncertain."
            )
        lines.extend(trust_lines)

    return "\n".join(lines)


def _build_knowledge_graph_context(kg: dict | None) -> str:
    """
    CONSTRUCT (arxiv:2603.18014): Format knowledge graph as writing context.
    Writer uses entity relationships to ensure structural coherence.
    """
    if not kg or not kg.get("entities"):
        return ""

    lines = ["Knowledge Graph Context (use to ensure relationship accuracy):"]

    central = kg.get("central_claim", "")
    if central:
        lines.append(f"Central finding: {central}")

    # Top entities by importance
    entities = sorted(kg.get("entities", []), key=lambda e: e.get("importance", 0), reverse=True)
    if entities:
        lines.append("Key entities: " + ", ".join(e["label"] for e in entities[:6]))

    # Key relationships
    for rel in kg.get("relationships", [])[:8]:
        src_label = next((e["label"] for e in entities if e["id"] == rel.get("source")), rel.get("source", ""))
        tgt_label = next((e["label"] for e in entities if e["id"] == rel.get("target")), rel.get("target", ""))
        if src_label and tgt_label:
            lines.append(f"  {src_label} —[{rel.get('type', 'related')}]→ {tgt_label}: {rel.get('claim', '')}")

    return "\n".join(lines)


def _build_validation_summary(report: dict | None) -> str:
    if not report:
        return "Cross-validation not performed"
    lines = [
        f"Quality score: {report.get('quality_score', 0):.0%}",
        f"Cross-verified sources: {report.get('well_corroborated_count', 0)}",
        f"Contradictions: {report.get('contradictions_found', 0)}",
    ]
    if report.get("contradictions"):
        for c in report["contradictions"]:
            lines.append(f"  Contradiction: {c.get('summary', '')}")
    if report.get("summary"):
        lines.append(f"Assessment: {report['summary']}")
    return "\n".join(lines)


def _section_title_from_sub_query(sq: dict) -> str:
    """Convert sub-query dimension to section title"""
    dim_map = {
        "Definition/Background":   "Concept Definition and Background",
        "Current State/Evidence":  "Current State and Empirical Data",
        "Comparison/Alternatives": "Comparative Analysis and Alternatives",
        "Cause/Mechanism":         "Causal and Mechanism Analysis",
        "Limitations/Challenges":  "Limitations and Future Challenges",
    }
    dimension = sq.get("dimension", "")
    for key, title in dim_map.items():
        if key in dimension:
            return title
    # fallback: first 30 chars of sub-query question
    return sq.get("question", "Analysis")[:30]


# ── Section generation functions ──────────────────────────────────────────

async def _write_brief(
    query: str, interpretation: str, validation_summary: str,
    evidence: str, citation_count: int, llm: LLMProvider,
) -> str:
    """brief mode: generate the entire report in a single LLM call."""
    return await llm.complete(
        messages=[{"role": "user", "content": _BRIEF_PROMPT.format(
            query=query,
            interpretation=interpretation,
            validation_summary=validation_summary,
            evidence=evidence,
            citation_count=citation_count,
        )}],
        system=_SYSTEM,
        max_tokens=2000,
        temperature=0.4,
    )


async def _write_summary(
    query: str, interpretation: str, validation_summary: str,
    evidence: str, citation_count: int, llm: LLMProvider,
) -> str:
    return await llm.complete(
        messages=[{"role": "user", "content": _SUMMARY_PROMPT.format(
            query=query,
            interpretation=interpretation,
            validation_summary=validation_summary,
            evidence=evidence,
            citation_count=citation_count,
        )}],
        system=_SYSTEM,
        max_tokens=2000,
        temperature=0.4,
    )


async def _write_findings(
    query: str, validation_summary: str, evidence: str, llm: LLMProvider,
) -> str:
    return await llm.complete(
        messages=[{"role": "user", "content": _FINDINGS_PROMPT.format(
            query=query,
            validation_summary=validation_summary,
            evidence=evidence,
        )}],
        system=_SYSTEM,
        max_tokens=3000,
        temperature=0.4,
    )


async def _write_analysis(
    query: str, sq: dict, evidence: str, llm: LLMProvider,
    synthesis_entry: dict | None = None,
) -> str:
    """
    Write analysis for a single sub-query.

    When synthesis_entry is provided (mass_rag=True, CORRECT/AMBIGUOUS verdict):
      Uses _ANALYSIS_WITH_MASSRAG_PROMPT — synthesis is the PRIMARY SOURCE,
      evidence is citation support only. This matches the MASS-RAG paper's intent
      that Synthesis = authoritative answer for the sub-query.

    When synthesis_entry is None (mass_rag=False, INCORRECT verdict, fast depth):
      Falls back to _ANALYSIS_PROMPT — evidence-driven generation.
    """
    section_title = _section_title_from_sub_query(sq)

    if synthesis_entry:
        synthesis_block = _build_synthesis_block(synthesis_entry)
        prompt = _ANALYSIS_WITH_MASSRAG_PROMPT.format(
            query=query,
            sub_query=sq.get("question", ""),
            synthesis_block=synthesis_block,
            evidence=evidence,
            section_title=section_title,
        )
    else:
        prompt = _ANALYSIS_PROMPT.format(
            query=query,
            sub_query=sq.get("question", ""),
            evidence=evidence,
            section_title=section_title,
        )

    return await llm.complete(
        messages=[{"role": "user", "content": prompt}],
        system=_SYSTEM,
        max_tokens=4000,
        temperature=0.4,
    )


async def _write_conclusion(
    query: str, interpretation: str, evidence_summary: str, llm: LLMProvider,
) -> str:
    return await llm.complete(
        messages=[{"role": "user", "content": _CONCLUSION_PROMPT.format(
            query=query,
            interpretation=interpretation,
            evidence_summary=evidence_summary,
        )}],
        system=_SYSTEM,
        max_tokens=2000,
        temperature=0.4,
    )


# ── Main Node ─────────────────────────────────────────────────────────────

async def write_draft(state: ResearchState, llm: LLMProvider) -> dict:
    """
    Draft Writer Node — generation strategy per report_length mode.

    brief    : single LLM call (~2k tokens)
    standard : summary + findings + conclusion (3 calls, no sub-query analysis)
    detailed : summary + findings + per-sub-query analysis + conclusion (3+N calls)
    """
    citations = state["citations"]
    plan = state["plan"] or {}
    validation_report = state.get("cross_validation_report")
    knowledge_graph = state.get("knowledge_graph")
    report_length = state.get("report_length", "detailed")
    flags = state.get("feature_flags", {})
    mass_rag_outputs = state.get("mass_rag_outputs") or []

    # Phase F: privacy_mode config validation.
    # Raise early if raw local file content would reach the cloud LLM without MASS-RAG abstraction.
    privacy_mode = flags.get("privacy_mode", False)
    local_search_active = state.get("local_search_enabled", False)
    mass_rag_on = flags.get("mass_rag", False)
    if privacy_mode and local_search_active and not mass_rag_on:
        raise ValueError(
            "Privacy mode violation: privacy_mode=True with local files requires mass_rag=True. "
            "Enable the 'mass_rag' feature flag to ensure raw local file content is abstracted "
            "by the local LLM before reaching the cloud writer."
        )

    # EAM Stage 1 (RhinoInsight): use normalized evidence_store when available.
    # Fallback chain: evidence_store → SDP effective_citation_ids → raw citations
    evidence_store = state.get("evidence_store") or []
    if evidence_store:
        effective_citations = evidence_store
    else:
        effective_ids = (validation_report or {}).get("effective_citation_ids")
        if effective_ids:
            effective_citations = [c for c in citations if c["id"] in set(effective_ids)]
        else:
            effective_citations = citations

    # Phase F: sanitize local file excerpts before they reach the cloud writer LLM.
    if privacy_mode and local_search_active:
        effective_citations = _sanitize_evidence_for_privacy(
            effective_citations, citations, mass_rag_outputs
        )

    if not effective_citations:
        return {
            "draft_report": "Insufficient information collected to generate a report.",
            "error_log": ["no citations"],
        }

    query = state["original_query"]
    interpretation = plan.get("interpretation", "")
    sub_queries = plan.get("sub_queries", [])
    validation_summary = _build_validation_summary(validation_report)
    kg_context = _build_knowledge_graph_context(knowledge_graph)

    # Prepend knowledge graph context to evidence when available
    evidence_full = _build_evidence(effective_citations)
    if kg_context:
        evidence_full = kg_context + "\n\n---\n\n" + evidence_full
    evidence_summary = _build_evidence_summary(effective_citations)
    references_text = _build_references(effective_citations)

    # ── brief: single call ────────────────────────────────────────────
    if report_length == "brief":
        draft = await _write_brief(
            query, interpretation, validation_summary, evidence_full, len(effective_citations), llm
        )
        draft = draft + "\n\n" + references_text
        return {"draft_report": draft}

    # ── standard: summary + findings + conclusion (no analysis section) ─
    if report_length == "standard":
        summary_text = await _write_summary(
            query, interpretation, validation_summary, evidence_full, len(effective_citations), llm
        )
        findings_text = await _write_findings(query, validation_summary, evidence_full, llm)
        conclusion_text = await _write_conclusion(query, interpretation, evidence_summary, llm)
        draft = "\n\n".join([
            f"# {query}",
            summary_text,
            findings_text,
            conclusion_text,
            references_text,
        ])
        return {"draft_report": draft}

    # ── detailed: all sections (default) ──────────────────────────────
    summary_text = await _write_summary(
        query, interpretation, validation_summary, evidence_full, len(effective_citations), llm
    )
    findings_text = await _write_findings(query, validation_summary, evidence_full, llm)

    analysis_texts = []
    for sq in sub_queries:
        synthesis = _get_synthesis_entry(mass_rag_outputs, sq.get("id", ""))
        analysis_texts.append(await _write_analysis(query, sq, evidence_full, llm, synthesis))

    conclusion_text = await _write_conclusion(query, interpretation, evidence_summary, llm)

    draft = "\n\n".join([
        f"# {query}",
        summary_text,
        findings_text,
        *analysis_texts,
        conclusion_text,
        references_text,
    ])
    return {"draft_report": draft}
