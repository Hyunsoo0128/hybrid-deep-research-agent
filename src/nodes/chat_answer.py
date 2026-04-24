"""
Chat Answer Nodes — Phase 4

memory_answer   — Answer based on existing report/citations (no additional search)
targeted_search — 1-2 additional searches → augment context → connect to memory_answer
new_research_signal — Return signal indicating new research is needed
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone

from ..chat_state import ChatState
from ..providers.base import LLMProvider
from ..tools.search import SearchTool
from ..security.injection_filter import InjectionFilter
from ..state import Citation, SourceType, TrustLevel

# ── Common constants ───────────────────────────────────────────────────────

_MAX_HISTORY_TURNS = 5      # Number of recent conversation turns to include in context
_MAX_CITATIONS_IN_CONTEXT = 15   # Number of citations to include in answer generation

# ── Memory Answer ──────────────────────────────────────────────────────────

_MEMORY_SYSTEM = """You are a deep research assistant.
Answer the user's follow-up questions accurately based on the completed research report and citation sources.
For content not in the report, explicitly state "This information was not confirmed in the report"."""

_MEMORY_PROMPT = """[Completed Research]
Original question: {query}

Full report:
{final_report}

Key citation sources ({citation_count} total):
{citations_text}

[Conversation History]
{history}

[New Question]
{message}

Answer the question based on the report and citation sources above.
When citing, indicate the source in the format [Source Title]."""


def _format_citations(citations: list[dict], max_count: int) -> str:
    lines = []
    for i, c in enumerate(citations[:max_count], 1):
        lines.append(
            f"[{i}] {c.get('title', 'No title')}\n"
            f"    {c.get('excerpt', '')[:150]}"
        )
    return "\n".join(lines)


def _format_history(history: list[dict], max_turns: int) -> str:
    recent = history[-(max_turns * 2):]
    lines = []
    for h in recent:
        role = "User" if h["role"] == "user" else "Assistant"
        lines.append(f"{role}: {h['content']}")
    return "\n".join(lines) if lines else "(first question)"


async def memory_answer(state: ChatState, llm: LLMProvider) -> dict:
    """Answer based on existing report and citations."""
    context = state["research_context"]
    citations = context.get("citations", [])
    history = state.get("conversation_history", [])

    # Merge any citations added by targeted_search
    extra = state.get("extra_citations", [])
    all_citations = citations + extra

    response = await llm.complete(
        messages=[{
            "role": "user",
            "content": _MEMORY_PROMPT.format(
                query=context.get("query", ""),
                final_report=context.get("final_report", "")[:4000],
                citation_count=len(all_citations),
                citations_text=_format_citations(all_citations, _MAX_CITATIONS_IN_CONTEXT),
                history=_format_history(history, _MAX_HISTORY_TURNS),
                message=state["message"],
            ),
        }],
        system=_MEMORY_SYSTEM,
        max_tokens=1500,
        temperature=0.3,
    )

    return {"response": response}


# ── Targeted Search ────────────────────────────────────────────────────────

_QUERY_GEN_SYSTEM = """You are an expert in generating search queries.
Extract queries suitable for web search from the follow-up question."""

_QUERY_GEN_PROMPT = """Original research: {query}
Follow-up question: {message}

Generate 1-2 web search queries to answer this question.
Respond only as a JSON array: ["query1", "query2"]"""


async def targeted_search(
    state: ChatState,
    llm: LLMProvider,
    search_tool: SearchTool,
) -> dict:
    """
    1-2 additional web searches.
    Stores results in extra_citations → used by memory_answer.
    """
    context = state["research_context"]
    injection_filter = InjectionFilter()

    # Generate search queries
    raw = await llm.complete(
        messages=[{
            "role": "user",
            "content": _QUERY_GEN_PROMPT.format(
                query=context.get("query", ""),
                message=state["message"],
            ),
        }],
        system=_QUERY_GEN_SYSTEM,
        max_tokens=200,
        temperature=0.1,
    )

    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        queries = [q for q in __import__("json").loads(cleaned) if isinstance(q, str)][:2]
    except Exception:
        queries = [state["message"]]

    new_citations: list[dict] = []

    for query_text in queries:
        results = await search_tool.search_async(query=query_text, max_results=3)
        for result in results[:2]:  # max 2 per query
            filter_result = injection_filter.check(result.summary, result.url)
            cid = f"chat_{uuid.uuid4().hex[:8]}"
            new_citations.append(Citation(
                id=cid,
                url=result.url,
                title=result.title,
                excerpt=filter_result.sanitized_content,
                source_type=SourceType.WEB,
                trust_level=TrustLevel(filter_result.trust_level),
                crawled_at=datetime.now(timezone.utc).isoformat(),
                confidence=result.relevance_score * 0.7,
                injection_checked=True,
            ).to_dict())

    return {"extra_citations": new_citations}


# ── New Research Signal ────────────────────────────────────────────────────

async def new_research_signal(state: ChatState, llm: LLMProvider) -> dict:
    """
    Generate a short guidance message indicating new research is needed.
    Prompts the client to start a new research session.
    """
    response = await llm.complete(
        messages=[{
            "role": "user",
            "content": (
                f"The user asked a new question unrelated to the existing research "
                f"('{state['research_context'].get('query', '')}'): '{state['message']}'\n\n"
                "Kindly explain that new research is needed and suggest starting a new research session in 1-2 sentences."
            ),
        }],
        system="You are a friendly research assistant.",
        max_tokens=200,
        temperature=0.3,
    )
    return {"response": response, "route": "new_research"}
