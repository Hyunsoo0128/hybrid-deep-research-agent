"""
Web Search Node

Executes sequential search for each sub-query (Phase 1).
Parallelized in Phase 2 using LangGraph Send API.

ACI principles:
  - search() → returns summaries only
  - fetch_page() called only for high-relevance URLs
  - Empty results handled explicitly
  - Rate limit: tenacity retry (built into SearchTool)
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone

from ..state import ResearchState, Citation, SourceType, TrustLevel
from ..tools.search import SearchTool
from ..security.injection_filter import InjectionFilter
from ..providers.base import LLMProvider

_FETCH_THRESHOLD = 0.7   # Only fetch full content for URLs above this score
_MAX_FETCH_PER_QUERY = 2  # Maximum fetches per query (cost control)

_EXTRACT_SYSTEM = """You are an expert in information extraction.
Extract only the key content relevant to the research question from the provided web page content."""

_EXTRACT_PROMPT = """Research question: {question}

Web page content:
{content}

Summarize the key information relevant to the research question in 200 characters or less.
If there is no relevant content, respond only with "No relevant content found"."""


async def web_search(state: ResearchState, llm: LLMProvider, search_tool: SearchTool) -> dict:
    """
    Web Search Node.
    Sequentially searches sub-queries from the approved plan and returns a list of Citations.
    """
    plan = state["plan"]
    if not plan:
        return {"error_log": ["plan is missing"]}

    injection_filter = InjectionFilter()
    new_citations: list[dict] = []
    updated_sub_queries = []

    for sq_data in plan["sub_queries"]:
        sq_id = sq_data["id"]
        question = sq_data["question"]

        # 1. Search (summaries only)
        results = await search_tool.search_async(query=question, max_results=5)

        if not results:
            updated_sub_queries.append({**sq_data, "status": "failed", "answer": "No search results"})
            continue

        # 2. Fetch full content only for high-relevance URLs
        fetched_count = 0
        sq_citation_ids = []

        for result in results:
            # Low relevance → create Citation from summary only (regardless of fetch limit)
            if result.relevance_score < _FETCH_THRESHOLD:
                filter_result = injection_filter.check(result.summary, result.url)
                cid = f"cit_{uuid.uuid4().hex[:8]}"
                citation = Citation(
                    id=cid,
                    url=result.url,
                    title=result.title,
                    excerpt=filter_result.sanitized_content,
                    source_type=SourceType.WEB,
                    trust_level=TrustLevel(filter_result.trust_level),
                    crawled_at=datetime.now(timezone.utc).isoformat(),
                    confidence=result.relevance_score * 0.6,  # lower confidence since summary-only
                    injection_checked=True,
                )
                new_citations.append(citation.to_dict())
                sq_citation_ids.append(cid)
                continue

            # High relevance: fallback to summary if fetch limit reached
            if fetched_count >= _MAX_FETCH_PER_QUERY:
                filter_result = injection_filter.check(result.summary, result.url)
                cid = f"cit_{uuid.uuid4().hex[:8]}"
                citation = Citation(
                    id=cid,
                    url=result.url,
                    title=result.title,
                    excerpt=filter_result.sanitized_content,
                    source_type=SourceType.WEB,
                    trust_level=TrustLevel(filter_result.trust_level),
                    crawled_at=datetime.now(timezone.utc).isoformat(),
                    confidence=result.relevance_score * 0.6,
                    injection_checked=True,
                )
                new_citations.append(citation.to_dict())
                sq_citation_ids.append(cid)
                continue

            # Fetch full content
            fetch_result = await search_tool.fetch_page_async(result.url)
            content = fetch_result.content if fetch_result.success else result.summary

            # Injection filter
            filter_result = injection_filter.check(content, result.url)

            # Extract relevant content with LLM (ACI: avoid using raw full content)
            excerpt = await llm.complete(
                messages=[{
                    "role": "user",
                    "content": _EXTRACT_PROMPT.format(
                        question=question,
                        content=filter_result.sanitized_content[:2000],
                    )
                }],
                system=_EXTRACT_SYSTEM,
                max_tokens=300,
                temperature=0.1,
            )

            fetched_count += 1

            if "No relevant content found" in excerpt:
                # Fetched but no relevant content → add fallback citation using original summary
                filter_result = injection_filter.check(result.summary, result.url)
                cid = f"cit_{uuid.uuid4().hex[:8]}"
                citation = Citation(
                    id=cid,
                    url=result.url,
                    title=result.title,
                    excerpt=filter_result.sanitized_content,
                    source_type=SourceType.WEB,
                    trust_level=TrustLevel(filter_result.trust_level),
                    crawled_at=datetime.now(timezone.utc).isoformat(),
                    confidence=result.relevance_score * 0.4,  # low relevance
                    injection_checked=True,
                )
                new_citations.append(citation.to_dict())
                sq_citation_ids.append(cid)
                continue

            cid = f"cit_{uuid.uuid4().hex[:8]}"
            citation = Citation(
                id=cid,
                url=result.url,
                title=result.title,
                excerpt=excerpt.strip(),
                source_type=SourceType.WEB,
                trust_level=TrustLevel(filter_result.trust_level),
                crawled_at=datetime.now(timezone.utc).isoformat(),
                confidence=result.relevance_score,
                injection_checked=True,
            )
            new_citations.append(citation.to_dict())
            sq_citation_ids.append(cid)

        # Update sub-query status
        updated_sub_queries.append({
            **sq_data,
            "status": "done" if sq_citation_ids else "failed",
            "citation_ids": sq_citation_ids,
        })

    # Update sub_queries in plan
    updated_plan = {**plan, "sub_queries": updated_sub_queries}

    return {
        "plan": updated_plan,
        "citations": new_citations,
        "research_round": state.get("research_round", 0) + 1,
    }
