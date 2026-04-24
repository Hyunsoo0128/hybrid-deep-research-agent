"""
Web search tool — applying ACI (Agent-Computer Interface) principles

Core principles:
  search()      → returns summary only (200 chars). Full body text NOT included.
  fetch_page()  → full body text for a specific URL. Call only when needed.
  Empty results are returned explicitly (prevents model confusion)
"""

from __future__ import annotations
import asyncio
import os
from dataclasses import dataclass
from typing import Literal

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@dataclass
class SearchResult:
    url: str
    title: str
    summary: str            # up to 200 character summary
    relevance_score: float  # 0.0 ~ 1.0
    raw_content: str = ""   # populated when fetch_page() is called


@dataclass
class FetchResult:
    url: str
    title: str
    content: str            # full body text (up to 3,000 chars)
    success: bool
    reason: str = ""        # failure reason (paywall, timeout, etc.)


class SearchTool:
    """
    Tavily API wrapper.
    Agents use only this class. Only the internal implementation changes when swapping APIs.
    """

    def __init__(self, api_key: str | None = None):
        from tavily import TavilyClient
        self._client = TavilyClient(
            api_key=api_key or os.environ["TAVILY_API_KEY"]
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: Literal["basic", "advanced"] = "basic",
    ) -> list[SearchResult]:
        """
        Search by query. Returns summaries only (200 chars).
        Call fetch_page(url) separately if full body text is needed.

        Returns:
            List of SearchResult sorted by relevance score descending.
            Returns empty list if no results (no exception raised).
        """
        try:
            resp = self._client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=False,  # summary only → saves tokens
            )
            results = []
            for r in resp.get("results", []):
                content = r.get("content", "") or ""
                results.append(SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", "No title"),
                    summary=content[:200],
                    relevance_score=float(r.get("score", 0.5)),
                ))
            return sorted(results, key=lambda x: x.relevance_score, reverse=True)
        except Exception as e:
            # Search failed → return empty list (agent handles gracefully)
            return []

    async def search_async(
        self,
        query: str,
        max_results: int = 5,
        search_depth: Literal["basic", "advanced"] = "basic",
    ) -> list[SearchResult]:
        """Async search — runs without blocking the event loop."""
        return await asyncio.to_thread(self.search, query, max_results, search_depth)

    async def fetch_page_async(self, url: str, max_chars: int = 3000) -> FetchResult:
        """Async page fetch — runs without blocking the event loop."""
        return await asyncio.to_thread(self.fetch_page, url, max_chars)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=False,
    )
    def fetch_page(self, url: str, max_chars: int = 3000) -> FetchResult:
        """
        Extract full body text from a specific URL.
        Confirm relevant URLs with search() first, then call only for needed ones.

        Returns:
            FetchResult. Returns success=False instead of raising exception on failure.
        """
        try:
            resp = self._client.search(
                query=url,
                max_results=1,
                search_depth="advanced",
                include_raw_content=True,
            )
            results = resp.get("results", [])
            if not results:
                return FetchResult(url=url, title="", content="", success=False, reason="no_results")

            r = results[0]
            raw = r.get("raw_content") or r.get("content") or ""
            return FetchResult(
                url=url,
                title=r.get("title", ""),
                content=raw[:max_chars],
                success=True,
            )
        except Exception as e:
            return FetchResult(url=url, title="", content="", success=False, reason=str(e))
