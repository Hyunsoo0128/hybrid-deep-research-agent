"""
Chat Graph — Phase 4

An independent LangGraph for handling conversation mode after research is complete.

Flow:
  [follow-up question] → router → [memory]       → memory_answer  → END
                                → [targeted]     → targeted_search → memory_answer → END
                                → [new_research] → new_research_signal → END

Separated from the main research graph:
  - Each conversation turn is a separate graph execution (stateless per-turn)
  - Conversation history is managed in main.py sessions
  - MemorySaver not needed (no checkpointing)
"""

from __future__ import annotations
from functools import partial

from langgraph.graph import StateGraph, END

from .chat_state import ChatState
from .providers.base import LLMProvider
from .tools.search import SearchTool
from .nodes.router import router_node
from .nodes.chat_answer import memory_answer, targeted_search, new_research_signal


def route_after_router(state: ChatState) -> str:
    """Branch based on router result."""
    return state.get("route", "memory")


def build_chat_graph(llm: LLMProvider, search_tool: SearchTool) -> StateGraph:
    """Create the conversation mode graph."""
    builder = StateGraph(ChatState)

    builder.add_node("router",              partial(router_node, llm=llm))
    builder.add_node("memory_answer",       partial(memory_answer, llm=llm))
    builder.add_node("targeted_search",     partial(targeted_search, llm=llm, search_tool=search_tool))
    builder.add_node("new_research_signal", partial(new_research_signal, llm=llm))

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "memory":       "memory_answer",
            "targeted":     "targeted_search",
            "new_research": "new_research_signal",
        },
    )

    builder.add_edge("targeted_search",     "memory_answer")
    builder.add_edge("memory_answer",       END)
    builder.add_edge("new_research_signal", END)

    # No checkpointer — each turn runs independently
    return builder.compile()
