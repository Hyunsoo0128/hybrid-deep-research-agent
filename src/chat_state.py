"""
Chat Session State — Phase 4

State used in conversation mode after research is complete.
An independent TypedDict separate from the main ResearchState.

1 conversation turn = new Chat Graph execution (stateless per-turn).
Conversation history is accumulated in main.py sessions and passed as context each turn.
"""

from __future__ import annotations
from typing import TypedDict


class ChatState(TypedDict):
    # Session identifier
    session_id: str

    # Current turn input
    message: str

    # Research context (extracted from completed research)
    research_context: dict   # {query, final_report, citations_summary, plan_interpretation}

    # Conversation history (previous turns)
    conversation_history: list[dict]   # [{"role": "user"|"assistant", "content": "..."}]

    # Router classification result
    route: str   # "memory" | "targeted" | "new_research"

    # Final answer
    response: str

    # Additional sources found during targeted search
    extra_citations: list[dict]
