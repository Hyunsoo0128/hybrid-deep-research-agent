"""
Critic Agent Node — AlignRAG 3-phase misalignment diagnosis (arxiv:2504.14858)

Phase 1 — Relevance Alignment: claims irrelevant to the query
Phase 2 — Query-Evidence Mapping: claims lacking citation support
Phase 3 — Evidence-Integrated Synthesis: claims inconsistent with source content

Implementation notes:
  - CLM fine-tuning not possible via API (Known Deviation)
  - DR1c: single revise call with phase-grouped correction instructions
  - DR2: passed computed in code from misalignment counts
  - DSAP guard for stable JSON parsing
  - Maximum revision loops then forced completion

Phase E — Speculative RAG restructure (Tier A):
  Stage 1 (local):  _generate_suspect_claims() — extract candidate claims from report
  Stage 2 (cloud):  _coherence_verify()         — claim text only, no source docs
  Stage 3 (local):  _refine_corrections()        — add evidence quotes + citation IDs
  Graceful degradation: single provider → existing single-prompt path unchanged
"""

from __future__ import annotations
from ..state import ResearchState, CriticFeedback
from ..providers.base import LLMProvider
from ..utils.llm_json import llm_json

_DEPTH_REVISIONS = {"fast": 1, "normal": 1, "deep": 3}

_SYSTEM = """You are an expert in research report quality and factual consistency review.
Objectively diagnose misalignments between report claims and citation sources using a
structured 3-phase AlignRAG analysis."""

_PROMPT = """Review the following research report draft and citation sources using a
3-phase AlignRAG misalignment analysis.

Original question: {query}

Sub-query list:
{sub_queries}

Citation sources (top {citation_count}):
{citations_excerpt}

Report draft:
{draft}

Perform the following 3-phase diagnosis:

Phase 1 — Relevance Alignment
  Flag claims that are irrelevant to the original question or sub-queries.
  These are claims addressing topics not asked about.

Phase 2 — Query-Evidence Mapping
  For each sub-query, check if the evidence in the citation sources actually supports
  the answer given. Flag claims where the cited source does not support the claim,
  or where no citation is present.

Phase 3 — Evidence-Integrated Synthesis
  Flag claims that are inconsistent with what the citation sources actually say:
  exaggeration, distortion, unsupported inference, or contradiction by source content.
  For each flagged claim, provide the relevant source quote.

Also check:
- Are there claims made without any citation? (quote the specific sentence)
- Are there sub-queries from the list above that remain completely unanswered?
- Are there structural or stylistic improvements needed?

{construct_hint}Important: Only flag significant issues. Minor imperfections do not require flagging.

Respond ONLY in the following JSON format:
{{
  "misaligned_claims": [
    {{
      "phase": "phase1|phase2|phase3",
      "claim": "quoted text from report",
      "source_citation_ids": ["cit_abc"],
      "source_quote": "what the source actually says (empty string if not applicable)",
      "correction_hint": "how to fix this claim"
    }}
  ],
  "uncited_claims": [],
  "unanswered_sub_queries": [],
  "suggestions": []
}}"""

# Simplified prompt when alignrag flag is off (no 3-phase citation alignment check)
_PROMPT_NO_ALIGNRAG = """Review the following research report draft.

Original question: {query}

Sub-query list:
{sub_queries}

Report draft:
{draft}

Review criteria:
1. Are there claims made without citations? (quote the specific sentence)
2. Are there any sub-queries from the list above that remain unanswered?
3. Are there structural issues that need improvement?

Important: Only flag significant issues. Minor imperfections do not require flagging.

Respond ONLY in the following JSON format:
{{
  "misaligned_claims": [],
  "uncited_claims": [],
  "unanswered_sub_queries": [],
  "suggestions": []
}}"""

_SCHEMA_HINT = """{
  "misaligned_claims": [
    {
      "phase": "phase1|phase2|phase3",
      "claim": "quoted text from report",
      "source_citation_ids": ["cit_abc"],
      "source_quote": "what the source actually says",
      "correction_hint": "how to fix"
    }
  ],
  "uncited_claims": ["quoted text that needs a citation"],
  "unanswered_sub_queries": ["sub_query_id or question"],
  "suggestions": ["structural/stylistic suggestions"]
}"""

_REVISE_SYSTEM = """You are an expert in research report improvement.
Improve the report based on the feedback while maintaining the overall structure."""

_REVISE_PROMPT = """Improve the following report section according to the feedback.

Original question: {query}

Current section:
{draft}

Review feedback:
{feedback}

Improvement guidelines:
- Add [citation needed] or actual source to claims missing citations
- For Phase 1 misalignments: remove or reframe claims irrelevant to the question
- For Phase 2 misalignments: add citation support or soften unsupported claims
- For Phase 3 misalignments: correct claims to match what sources actually say
- For unanswered sub-queries: explicitly state "No confirmed information available"
- Maintain overall structure"""


# ── Phase E: Spec RAG prompts ─────────────────────────────────────────────────

_SUSPECT_CLAIMS_SYSTEM = (
    "You are a critical reviewer that identifies potentially problematic claims "
    "in research reports. Be thorough — flag anything that could be off-topic, "
    "unsupported, or factually inconsistent."
)

_SUSPECT_CLAIMS_PROMPT = """Read the following research report and identify claims that might be:
- Off-topic or irrelevant to the question (potential Phase 1)
- Lacking citation support or citing non-existent sources (potential Phase 2)
- Potentially inconsistent with source material, exaggerated, or distorted (potential Phase 3)

Original question: {query}

Report:
{report}

List every claim you find suspicious. For each claim provide the exact quoted text,
why it seems suspicious, and your confidence that it is actually a problem (0.0–1.0).

Respond ONLY in JSON (array, may be empty):
[
  {{"claim_text": "exact quote from report", "reason": "why suspicious", "confidence": 0.8}}
]"""

_COHERENCE_VERIFY_SYSTEM = (
    "You are an expert fact-checker. Evaluate flagged claims from a research report "
    "and determine which are genuine misalignments."
)

_COHERENCE_VERIFY_PROMPT = """The following claims were flagged as potentially problematic
in a research report about: "{query}"

For each claim determine:
1. Is it a real problem or a false alarm?
2. If real, which AlignRAG phase applies:
   - phase1: claim is irrelevant or off-topic to the question
   - phase2: claim cites a non-existent source, or makes an assertion with no citation
   - phase3: claim contradicts, exaggerates, or distorts what sources would normally say

Suspect claims:
{suspect_claims}

Return ONLY confirmed real problems. Respond ONLY in JSON:
{{
  "confirmed_misaligned": [
    {{"claim_text": "exact quote", "phase": "phase1|phase2|phase3", "correction_hint": "brief fix"}}
  ]
}}"""

_REFINE_CORRECTIONS_SYSTEM = (
    "You are a research editor providing specific, evidence-backed correction guidance."
)

_REFINE_CORRECTIONS_PROMPT = """The following claims in a research report have been confirmed as misaligned.
Use the available source evidence to provide precise correction hints with specific quotes and citation IDs.

Original question: {query}

Confirmed misaligned claims:
{confirmed_claims}

Available source evidence:
{evidence}

For each confirmed claim, map it to the best matching source evidence.
Respond ONLY in JSON:
{{
  "misaligned_claims": [
    {{
      "phase": "phase1|phase2|phase3",
      "claim": "exact quoted text",
      "source_citation_ids": ["cit_id_or_empty"],
      "source_quote": "relevant excerpt from source (empty string if phase1)",
      "correction_hint": "specific correction based on evidence"
    }}
  ]
}}"""

_SUSPECT_CLAIMS_SCHEMA = '[{"claim_text": "...", "reason": "...", "confidence": 0.8}]'

_COHERENCE_VERIFY_SCHEMA = '{"confirmed_misaligned": [{"claim_text": "...", "phase": "phase1|phase2|phase3", "correction_hint": "..."}]}'

_REFINE_SCHEMA = (
    '{"misaligned_claims": [{"phase": "phase1|phase2|phase3", "claim": "...", '
    '"source_citation_ids": ["cit_abc"], "source_quote": "...", "correction_hint": "..."}]}'
)


def _format_suspect_claims(claims: list[dict]) -> str:
    lines = []
    for i, c in enumerate(claims, 1):
        lines.append(
            f"{i}. \"{c.get('claim_text', '')}\" "
            f"(confidence={c.get('confidence', 0):.2f}) — {c.get('reason', '')}"
        )
    return "\n".join(lines) if lines else "(none)"


def _format_confirmed_claims(claims: list[dict]) -> str:
    lines = []
    for i, c in enumerate(claims, 1):
        lines.append(
            f"{i}. [{c.get('phase', '')}] \"{c.get('claim_text', '')}\" "
            f"— {c.get('correction_hint', '')}"
        )
    return "\n".join(lines) if lines else "(none)"


def _format_evidence(evidence_store: list[dict], citations: list[dict]) -> str:
    """Format evidence for Stage 3: prefer evidence_store, fall back to citations."""
    sources = evidence_store if evidence_store else citations
    lines = []
    for s in sources[:12]:
        eid = s.get("id", "")
        title = s.get("title", "")
        excerpt = s.get("excerpt", "")[:200]
        lines.append(f"[{eid or title[:20]}] {title}\n  {excerpt}")
    return "\n".join(lines) if lines else "(no sources available)"


async def _generate_suspect_claims(
    query: str,
    report: str,
    llm: LLMProvider,
    dsap_enabled: bool = True,
) -> list[dict]:
    """Stage 1 (local): Extract suspect claims from report. Returns [] on failure."""
    prompt = _SUSPECT_CLAIMS_PROMPT.format(query=query, report=report)
    result = await llm_json(
        llm=llm,
        messages=[{"role": "user", "content": prompt}],
        system=_SUSPECT_CLAIMS_SYSTEM,
        schema_hint=_SUSPECT_CLAIMS_SCHEMA,
        max_tokens=600,
        temperature=0.1,
        dsap_enabled=dsap_enabled,
        fallback=[],
    )
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict) and r.get("claim_text")]
    return []


async def _coherence_verify(
    query: str,
    suspect_claims: list[dict],
    llm: LLMProvider,
    dsap_enabled: bool = True,
) -> list[dict]:
    """Stage 2 (cloud): Verify which suspect claims are genuine misalignments.
    Receives claim text only — no source documents (privacy boundary)."""
    if not suspect_claims:
        return []
    prompt = _COHERENCE_VERIFY_PROMPT.format(
        query=query,
        suspect_claims=_format_suspect_claims(suspect_claims),
    )
    result = await llm_json(
        llm=llm,
        messages=[{"role": "user", "content": prompt}],
        system=_COHERENCE_VERIFY_SYSTEM,
        schema_hint=_COHERENCE_VERIFY_SCHEMA,
        max_tokens=600,
        temperature=0.1,
        dsap_enabled=dsap_enabled,
        fallback={"confirmed_misaligned": []},
    )
    if isinstance(result, dict):
        return result.get("confirmed_misaligned", [])
    return []


async def _refine_corrections(
    query: str,
    confirmed_claims: list[dict],
    evidence_store: list[dict],
    citations: list[dict],
    llm: LLMProvider,
    dsap_enabled: bool = True,
) -> list[dict]:
    """Stage 3 (local): Add evidence quotes and citation IDs to confirmed misalignments."""
    if not confirmed_claims:
        return []
    evidence_text = _format_evidence(evidence_store, citations)
    prompt = _REFINE_CORRECTIONS_PROMPT.format(
        query=query,
        confirmed_claims=_format_confirmed_claims(confirmed_claims),
        evidence=evidence_text,
    )
    result = await llm_json(
        llm=llm,
        messages=[{"role": "user", "content": prompt}],
        system=_REFINE_CORRECTIONS_SYSTEM,
        schema_hint=_REFINE_SCHEMA,
        max_tokens=800,
        temperature=0.1,
        dsap_enabled=dsap_enabled,
        fallback={"misaligned_claims": []},
    )
    if isinstance(result, dict):
        raw = result.get("misaligned_claims", [])
        return [m for m in raw if isinstance(m, dict) and m.get("claim")]
    return []


# ── End Phase E Spec RAG helpers ──────────────────────────────────────────────


def _build_construct_hint(mass_rag_outputs: list[dict]) -> str:
    """
    C4d: Build CONSTRUCT trust hint for the critic prompt.
    Returns a paragraph flagging low-trust sub-queries, or empty string if none.
    """
    flagged = []
    for entry in mass_rag_outputs:
        trust = entry.get("trust_scores")
        if not trust:
            continue
        untrustworthy = trust.get("untrustworthy_fields", [])
        if untrustworthy:
            doc_score = trust.get("document_score", 1.0)
            flagged.append(
                f'  - Sub-query "{entry.get("question", "")[:80]}": '
                f'low-trust fields: {", ".join(untrustworthy)} '
                f'(document_score={doc_score:.2f})'
            )
    if not flagged:
        return ""
    lines = [
        "CONSTRUCT Trust Alerts — the following sub-query analyses have low-trust fields.",
        "Pay extra attention to Phase 3 misalignments for these claims:",
    ] + flagged + [""]
    return "\n".join(lines) + "\n"


def _build_citations_excerpt(citations: list[dict], max_count: int = 10) -> str:
    lines = []
    for i, c in enumerate(citations[:max_count], 1):
        lines.append(
            f"[{i}] {c.get('title', 'No title')}\n"
            f"    {c.get('excerpt', '')[:120]}"
        )
    return "\n".join(lines)


def _format_misaligned_feedback(misaligned: list) -> str:
    """DR1c: Format misaligned_claims by phase for revise() feedback text."""
    if not misaligned:
        return " none"

    by_phase: dict[str, list] = {}
    for item in misaligned:
        if isinstance(item, dict):
            phase = item.get("phase", "general")
            by_phase.setdefault(phase, []).append(item)
        else:
            by_phase.setdefault("general", []).append({"claim": str(item)})

    lines = []
    for phase in sorted(by_phase.keys()):
        lines.append(f"\n  {phase.upper()}:")
        for it in by_phase[phase]:
            lines.append(f"    - Claim: {it.get('claim', '')}")
            if it.get("source_quote"):
                lines.append(f"      Source says: {it.get('source_quote', '')}")
            if it.get("correction_hint"):
                lines.append(f"      Fix: {it.get('correction_hint', '')}")
    return "\n".join(lines)


async def critique(state: ResearchState, llm: LLMProvider) -> dict:
    """Critic Agent Node — AlignRAG 3-phase diagnosis.

    Phase E: When llm is a HybridProvider, runs Spec RAG 3-stage pipeline:
      Stage 1 (local)  → Stage 2 (cloud) → Stage 3 (local)
    Otherwise falls back to the existing single-prompt path.
    """
    draft = state["draft_report"]
    revision_count = state.get("revision_count", 0)

    depth = (state.get("plan") or {}).get("depth", "normal")
    max_revisions = _DEPTH_REVISIONS.get(depth, 1)

    # Maximum revisions reached → pass immediately
    if revision_count >= max_revisions:
        return {
            "critic_feedback": CriticFeedback(
                passed=True,
                uncited_claims=[],
                unanswered_sub_queries=[],
                suggestions=["Maximum revisions reached. Completing with current version."],
                misaligned_claims=[],
            ).to_dict()
        }

    flags = state.get("feature_flags", {})
    alignrag_on = flags.get("alignrag", True)
    spec_rag_critic_on = flags.get("spec_rag_critic", False)
    dsap_on = flags.get("dsap", True)

    query = state["original_query"]
    citations = state.get("citations", [])

    # ── Phase E: Spec RAG path (HybridProvider + spec_rag_critic flag) ───────
    # spec_rag_critic is independent of alignrag: Spec RAG eliminates self-preference
    # bias in the drafter/verifier separation regardless of alignrag phase checks.
    from ..providers.hybrid import HybridProvider  # local import avoids circular dep
    if spec_rag_critic_on and isinstance(llm, HybridProvider):
        local_llm = llm.local
        cloud_llm = llm.cloud

        # Stage 1 (local): generate suspect claims from the report
        suspect_claims = await _generate_suspect_claims(
            query=query,
            report=draft,
            llm=local_llm,
            dsap_enabled=dsap_on,
        )

        # Stage 2 (cloud): verify which claims are genuine misalignments
        # Privacy: only claim texts are sent — no source documents
        confirmed = await _coherence_verify(
            query=query,
            suspect_claims=suspect_claims,
            llm=cloud_llm,
            dsap_enabled=dsap_on,
        )

        # Stage 3 (local): enrich confirmed claims with evidence quotes + citation IDs
        misaligned_dicts = await _refine_corrections(
            query=query,
            confirmed_claims=confirmed,
            evidence_store=state.get("evidence_store") or [],
            citations=citations,
            llm=local_llm,
            dsap_enabled=dsap_on,
        )

        # Stage 3 fallback: if refiner returned nothing but cloud confirmed issues,
        # use cloud output directly (without source quotes)
        if not misaligned_dicts and confirmed:
            misaligned_dicts = [
                {
                    "phase": c.get("phase", "general"),
                    "claim": c.get("claim_text", ""),
                    "source_citation_ids": [],
                    "source_quote": "",
                    "correction_hint": c.get("correction_hint", ""),
                }
                for c in confirmed
                if c.get("claim_text")
            ]

        passed = len(misaligned_dicts) == 0
        feedback = CriticFeedback(
            passed=passed,
            uncited_claims=[],
            unanswered_sub_queries=[],
            suggestions=[],
            misaligned_claims=misaligned_dicts,
        )
        return {"critic_feedback": feedback.to_dict()}

    # ── Single-provider fallback path (unchanged) ─────────────────────────────
    plan = state.get("plan") or {}
    sub_queries_text = "\n".join(
        f"- [{sq['id']}] {sq['question']}"
        for sq in plan.get("sub_queries", [])
    )

    citations_excerpt = _build_citations_excerpt(citations, max_count=10)

    # C4d: CONSTRUCT hint — flag low-trust MASS-RAG fields for closer scrutiny
    construct_hint = _build_construct_hint(state.get("mass_rag_outputs") or [])

    if alignrag_on:
        prompt_content = _PROMPT.format(
            query=query,
            sub_queries=sub_queries_text,
            citation_count=min(len(citations), 10),
            citations_excerpt=citations_excerpt,
            draft=draft,
            construct_hint=construct_hint,
        )
    else:
        prompt_content = _PROMPT_NO_ALIGNRAG.format(
            query=query,
            sub_queries=sub_queries_text,
            draft=draft,
        )

    _fallback: dict = {
        "misaligned_claims": [],
        "uncited_claims": [],
        "unanswered_sub_queries": [],
        "suggestions": [],
    }

    data = await llm_json(
        llm=llm,
        messages=[{"role": "user", "content": prompt_content}],
        system=_SYSTEM,
        schema_hint=_SCHEMA_HINT,
        max_tokens=800,
        temperature=0.1,
        dsap_enabled=dsap_on,
        fallback=_fallback,
    )

    misaligned = data.get("misaligned_claims", [])
    # Normalize: old string items (e.g. from fallback or no-alignrag path) → keep as-is
    misaligned_dicts = [
        m if isinstance(m, dict) else {"phase": "general", "claim": str(m), "source_citation_ids": [], "source_quote": "", "correction_hint": ""}
        for m in misaligned
    ]

    uncited = data.get("uncited_claims", [])
    unanswered = data.get("unanswered_sub_queries", [])

    # DR2: passed computed in code from misalignment counts
    passed = len(misaligned_dicts) == 0 and len(uncited) == 0 and len(unanswered) == 0

    feedback = CriticFeedback(
        passed=passed,
        uncited_claims=uncited,
        unanswered_sub_queries=unanswered,
        suggestions=data.get("suggestions", []),
        misaligned_claims=misaligned_dicts,
    )

    return {"critic_feedback": feedback.to_dict()}


async def revise(state: ResearchState, llm: LLMProvider) -> dict:
    """
    Feedback-based rewrite Node — DR1c phase-grouped correction instructions.

    Splits the report into ## sections and revises each section sequentially
    (prevents Ollama single-GPU queue timeout on long reports).
    """
    feedback = state.get("critic_feedback") or {}

    # DR1c: phase-grouped misalignment instructions
    misaligned_text = _format_misaligned_feedback(feedback.get("misaligned_claims", []))
    feedback_text = "\n".join([
        f"- Misaligned claims:{misaligned_text}",
        f"- Uncited claims: {', '.join(feedback.get('uncited_claims', [])[:5]) or 'none'}",
        f"- Unanswered sub-queries: {', '.join(feedback.get('unanswered_sub_queries', [])) or 'none'}",
        f"- Suggestions: {', '.join(feedback.get('suggestions', [])[:5]) or 'none'}",
    ])

    draft = state["draft_report"]
    query = state["original_query"]

    raw_sections = _split_sections(draft)

    revised_sections = []
    for sec in raw_sections:
        revised_sections.append(await _revise_section(sec, query, feedback_text, llm))

    return {
        "draft_report": "\n\n".join(revised_sections),
        "revision_count": state.get("revision_count", 0) + 1,
    }


def _split_sections(draft: str) -> list[str]:
    """Split by ## headings. Returns as a single piece if no headings found."""
    import re
    parts = re.split(r"(?=\n## )", draft)
    return [p.strip() for p in parts if p.strip()]


async def _revise_section(
    section: str, query: str, feedback: str, llm: LLMProvider
) -> str:
    """Apply feedback to a single section."""
    if section.startswith("## References"):
        return section

    prompt = _REVISE_PROMPT.format(
        query=query,
        draft=section,
        feedback=feedback,
    )
    revised = await llm.complete(
        messages=[{"role": "user", "content": prompt}],
        system=_REVISE_SYSTEM,
        max_tokens=4000,
        temperature=0.3,
    )
    return revised.strip()


def should_revise(state: ResearchState) -> str:
    """Routing based on Critic result."""
    feedback = state.get("critic_feedback") or {}
    if feedback.get("passed", True):
        return "finalize"
    return "revise"
