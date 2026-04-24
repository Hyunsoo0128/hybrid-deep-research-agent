"""
Router Agent Node — Phase 4

Classifies follow-up questions into 3 routes:
  memory      — Can answer directly from existing report/citations
  targeted    — Answer after 1-2 additional searches
  new_research — Entirely new topic outside the scope of the research

Focused on fast classification (max_tokens=200, temperature=0).
"""

from __future__ import annotations

from ..chat_state import ChatState
from ..providers.base import LLMProvider
from ..utils.llm_json import llm_json

_SYSTEM = """You are a question classifier for a research assistant.
Determine whether the user's follow-up question can be answered using existing research results."""

_PROMPT = """Existing research query: {query}

Report summary (first 500 chars):
{report_summary}

Conversation history ({history_count} turns):
{history}

New question: {message}

Classification criteria:
- memory: Answer can be extracted from the existing report/citations (summarizing, explaining, comparing, confirming, etc.)
- targeted: Specific additional information not in the report is needed (latest news, figures, specific cases, etc.)
- new_research: Completely different topic from the existing research

Respond ONLY in the following JSON format:
{{"route": "memory", "reason": "one-line reason"}}"""


async def router_node(state: ChatState, llm: LLMProvider) -> dict:
    """Router Agent: classify follow-up questions."""
    context = state["research_context"]
    report_summary = context.get("final_report", "")[:500]
    history = state.get("conversation_history", [])

    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content'][:100]}"
        for h in history[-6:]  # most recent 3 turns
    ) or "(first question)"

    data = await llm_json(
        llm=llm,
        messages=[{
            "role": "user",
            "content": _PROMPT.format(
                query=context.get("query", ""),
                report_summary=report_summary,
                history_count=len(history) // 2,
                history=history_text,
                message=state["message"],
            ),
        }],
        system=_SYSTEM,
        schema_hint='{"route": "memory|targeted|new_research", "reason": "string"}',
        max_tokens=200,
        temperature=0.0,
        fallback={"route": "memory", "reason": "fallback"},
    )

    route = data.get("route", "memory")
    if route not in ("memory", "targeted", "new_research"):
        route = "memory"

    return {"route": route}
