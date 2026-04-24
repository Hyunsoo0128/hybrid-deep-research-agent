"""
Reranker Node — query_decomp (arxiv:2507.00355)

After parallel sub-query retrieval completes, reranks all collected citations
against the original query to restore precision lost in decomposed retrieval.

Paper key contribution: decomposition improves recall, reranking restores precision.
The combination achieves MRR@10 +36.7% over naive RAG.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 via fastembed (ONNX, no PyTorch).
Fallback: confidence-based sort if cross-encoder is unavailable.
"""

from __future__ import annotations
import asyncio
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

_TOP_K = {
    "fast":   10,
    "normal": 20,
    "deep":   40,
}


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Lazy singleton — model downloads on first call (~80 MB ONNX)."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder  # type: ignore
    return TextCrossEncoder(model_name=_RERANKER_MODEL)


def _dedup_by_url(citations: list[dict]) -> list[dict]:
    """URL-based deduplication; keep highest-confidence entry per URL."""
    seen: dict[str, dict] = {}
    for c in citations:
        url = c.get("url", "")
        if not url:
            continue
        if url not in seen or c.get("confidence", 0) > seen[url].get("confidence", 0):
            seen[url] = c
    # preserve any citation that had no URL
    no_url = [c for c in citations if not c.get("url", "")]
    return list(seen.values()) + no_url


def _rerank_sync(query: str, citations: list[dict]) -> list[tuple[dict, float]]:
    """Synchronous cross-encoder scoring; run in thread pool from async context."""
    model = _get_cross_encoder()
    documents = [
        f"{c.get('title', '')} {c.get('excerpt', '')}".strip()
        for c in citations
    ]
    scores = list(model.rerank(query, documents))
    return list(zip(citations, scores))


async def rerank_citations(state: dict) -> dict:
    """
    Reranker Node.

    1. URL-deduplicates accumulated citations from parallel search workers
    2. Cross-encodes each citation against the original query
    3. Keeps top-k by rerank score
    4. Writes result to state["reranked_citations"] (plain list, overwrites)

    Downstream nodes (gap_detector, cross_validator) prefer reranked_citations
    when non-empty; writer inherits the selection via cross_validation_report.

    Skipped when query_decomp flag is False.
    """
    flags = state.get("feature_flags", {})
    if not flags.get("query_decomp", True):
        return {"reranked_citations": []}

    citations = state.get("citations", [])
    if not citations:
        return {"reranked_citations": []}

    original_query = state.get("original_query", "")
    plan = state.get("plan") or {}
    depth = plan.get("depth", "normal")
    top_k = _TOP_K.get(depth, _TOP_K["normal"])

    # Step 1: URL deduplication
    deduped = _dedup_by_url(citations)

    # Step 2: Cross-encoder reranking (sync model → thread pool)
    try:
        loop = asyncio.get_event_loop()
        pairs = await loop.run_in_executor(None, _rerank_sync, original_query, deduped)
        pairs.sort(key=lambda x: x[1], reverse=True)
        reranked = [c for c, _ in pairs[:top_k]]
        logger.debug(
            "reranker: %d → dedup %d → top-%d (cross-encoder)",
            len(citations), len(deduped), len(reranked),
        )
    except Exception as exc:
        logger.warning("reranker: cross-encoder failed (%s), falling back to confidence sort", exc)
        deduped_sorted = sorted(deduped, key=lambda c: c.get("confidence", 0), reverse=True)
        reranked = deduped_sorted[:top_k]

    return {"reranked_citations": reranked}
