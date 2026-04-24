"""
Local Search Worker Node — Local file semantic search (parallel execution via Send API)

Same interface as web search_worker:
  Input:  {"sub_query": dict, "original_query": str}
  Output: {"citations": list[dict]}  <- accumulated into main state via operator.add

Characteristics of local file Citations:
  - source_type: LOCAL
  - trust_level: HIGH (user's own files)
  - url: "file:///absolute/path/to/file"
  - confidence: Qdrant similarity score
"""

from __future__ import annotations

from ..state import Citation, SourceType, TrustLevel
from ..tools.local_file_search import LocalFileSearch


async def local_search_worker(
    state: dict,   # {"sub_query": dict, "original_query": str}
    local_file_search: LocalFileSearch,
) -> dict:
    """
    Local file search worker.
    fastembed-based semantic search → returns list of Citations.
    """
    sub_query = state["sub_query"]
    question = sub_query["question"]

    results = local_file_search.search(query=question, top_k=5, score_threshold=0.35)
    if not results:
        return {"citations": []}

    new_citations: list[dict] = []
    for result in results:
        # Skip chunks that are too short or empty
        if len(result.excerpt.strip()) < 50:
            continue

        cid = f"local_{result.chunk_index:04d}_{result.filename[:12].replace(' ', '_')}"

        new_citations.append(Citation(
            id=cid,
            url=f"file://{result.filepath}",
            title=f"{result.filename} (chunk {result.chunk_index + 1})",
            excerpt=result.excerpt[:500],   # max 500 chars
            source_type=SourceType.LOCAL,
            trust_level=TrustLevel.HIGH,    # user's own file → high trust
            crawled_at="",                  # crawl time not needed for local files
            confidence=result.score,
            injection_checked=True,         # injection check not needed for local files
        ).to_dict())

    return {"citations": new_citations}
