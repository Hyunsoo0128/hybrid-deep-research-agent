"""
CONSTRUCT Quality Scorer — arxiv:2603.18014

Scores the trustworthiness of each field in MASS-RAG structured outputs using
a Judge LLM. Works on black-box APIs without logprobs or fine-tuning.

Implementation: 2-call simplified version (C2b) of the paper's 5-template approach.
  Template 1 (Document-level): overall trustworthiness of the MASS-RAG output
  Template 2 (Field-level):    per-field scores for summary / key_spans / inferences

Applied to: MASS-RAG Synthesizer output (which contains all 3 agent results).
  C1b: score all three agent output types via their fields in the synthesis dict.

Output per MASS-RAG entry (C3a — added to mass_rag_outputs):
  "trust_scores": {
    "document_score": 0.72,
    "per_field": {"summary": 0.95, "key_spans": 0.68, "inferences": 0.54},
    "untrustworthy_fields": ["inferences"]   # fields below TRUST_THRESHOLD
  }

Downstream use:
  C4b: writer reduces emphasis on low-trust fields
  C4d: critic prioritises low-trust claims for misalignment checking

Known deviations:
  - Paper uses 5 verifier templates; this uses 2 (Document + Field-level).
    Paper ablation: removing individual templates costs 3-6% accuracy.
    2 templates ≈ 70% of full effect at 40% of the cost.
  - Paper benchmarks against Financial/PII/Insurance datasets.
    Not reproduced here (C6a) — evaluation via existing e2e benchmark.
  - Targeted field regeneration (C5b) deferred to future work.
"""

from __future__ import annotations
import asyncio
from ..providers.base import LLMProvider
from ..utils.llm_json import llm_json

# Fields below this score are flagged as untrustworthy
TRUST_THRESHOLD = 0.5

# ── Template 1: Document-level trustworthiness ───────────────────────────────

_DOC_LEVEL_SYSTEM = """You are a trustworthiness evaluator for research outputs.
Assess the overall reliability of the provided structured research analysis."""

_DOC_LEVEL_PROMPT = """Evaluate the overall trustworthiness of this structured research output.

Research sub-query: {question}

Research output:
Summary: {summary}

Key evidence spans ({span_count} items):
{key_spans_preview}

Inferences ({inference_count} items):
{inferences_preview}

Evaluation criteria:
1. Are the key_spans factually grounded (not fabricated)?
2. Are the inferences logically derived from the key_spans (not overreach)?
3. Is the summary internally consistent with the key_spans?

Respond ONLY in the following JSON format:
{{"document_score": 0.72, "explanation": "one-sentence rationale"}}

document_score: 0.0 (completely untrustworthy) to 1.0 (fully trustworthy)"""

_DOC_LEVEL_SCHEMA = '{"document_score": 0.5, "explanation": "string"}'

# ── Template 2: Field-level trustworthiness ──────────────────────────────────

_FIELD_LEVEL_SYSTEM = """You are a field-level trustworthiness evaluator.
Score each field of this structured research output independently."""

_FIELD_LEVEL_PROMPT = """Score the trustworthiness of each field independently.

Research sub-query: {question}

Fields to evaluate:
1. summary: "{summary_preview}"
2. key_spans: {span_count} spans extracted from sources
3. inferences: {inference_count} logical inferences

Scoring criteria:
- summary (0.0-1.0): Does it accurately reflect the key_spans without adding unsupported claims?
- key_spans (0.0-1.0): Are the extracted spans relevant and accurately attributed to sources?
- inferences (0.0-1.0): Are the logical conclusions directly supported by the key_spans?

Respond ONLY in the following JSON format:
{{"summary": 0.9, "key_spans": 0.75, "inferences": 0.6}}"""

_FIELD_LEVEL_SCHEMA = '{"summary": 0.9, "key_spans": 0.75, "inferences": 0.6}'


def _preview(items: list, max_items: int = 3, text_key: str = "text") -> str:
    if not items:
        return "(none)"
    previews = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            previews.append(f"  - {item.get(text_key, str(item))[:100]}")
        else:
            previews.append(f"  - {str(item)[:100]}")
    if len(items) > max_items:
        previews.append(f"  ... ({len(items) - max_items} more)")
    return "\n".join(previews)


async def score_mass_rag_output(
    mass_rag_entry: dict,
    llm: LLMProvider,
    dsap_enabled: bool = True,
) -> dict:
    """
    CONSTRUCT: score a single MASS-RAG output entry.

    Runs 2 verifier calls in parallel (asyncio.gather):
      1. Document-level overall trustworthiness
      2. Field-level per-field scores (summary / key_spans / inferences)

    Returns trust_scores dict to be merged into the mass_rag_output entry.
    """
    question = mass_rag_entry.get("question", "")
    summary = mass_rag_entry.get("summary", "")
    key_spans = mass_rag_entry.get("key_spans", [])
    inferences = mass_rag_entry.get("inferences", [])

    # Short-circuit: if MASS-RAG produced no content, score as low trust
    if not summary and not key_spans and not inferences:
        return {
            "document_score": 0.0,
            "per_field": {"summary": 0.0, "key_spans": 0.0, "inferences": 0.0},
            "untrustworthy_fields": ["summary", "key_spans", "inferences"],
        }

    summary_preview = (summary[:150] + "...") if len(summary) > 150 else summary

    doc_coro = llm_json(
        llm=llm,
        messages=[{"role": "user", "content": _DOC_LEVEL_PROMPT.format(
            question=question,
            summary=summary_preview,
            span_count=len(key_spans),
            key_spans_preview=_preview(key_spans, max_items=3),
            inference_count=len(inferences),
            inferences_preview=_preview(inferences, max_items=3, text_key="claim"),
        )}],
        system=_DOC_LEVEL_SYSTEM,
        schema_hint=_DOC_LEVEL_SCHEMA,
        max_tokens=128,
        temperature=0.0,
        dsap_enabled=dsap_enabled,
        fallback={"document_score": 0.5, "explanation": "scoring failed"},
    )

    field_coro = llm_json(
        llm=llm,
        messages=[{"role": "user", "content": _FIELD_LEVEL_PROMPT.format(
            question=question,
            summary_preview=summary_preview[:80],
            span_count=len(key_spans),
            inference_count=len(inferences),
        )}],
        system=_FIELD_LEVEL_SYSTEM,
        schema_hint=_FIELD_LEVEL_SCHEMA,
        max_tokens=64,
        temperature=0.0,
        dsap_enabled=dsap_enabled,
        fallback={"summary": 0.5, "key_spans": 0.5, "inferences": 0.5},
    )

    doc_result, field_result = await asyncio.gather(doc_coro, field_coro)

    doc_score = float(doc_result.get("document_score", 0.5))
    per_field = {
        "summary":    float(field_result.get("summary", 0.5)),
        "key_spans":  float(field_result.get("key_spans", 0.5)),
        "inferences": float(field_result.get("inferences", 0.5)),
    }
    untrustworthy = [f for f, s in per_field.items() if s < TRUST_THRESHOLD]

    return {
        "document_score":      round(doc_score, 3),
        "per_field":           {k: round(v, 3) for k, v in per_field.items()},
        "untrustworthy_fields": untrustworthy,
    }
