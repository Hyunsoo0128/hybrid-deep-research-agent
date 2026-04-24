"""
LangGraph Assembly

Pipeline flow:
  [query] → generate_plan → checklist_node (VCM) → [interrupt: plan approval]
          → search_orchestrator → [Send API parallel fan-out] search_worker × N
                                                             + local_search_worker × N (if enabled)
          → reranker   (fan-out join: dedup + cross-encoder top-k)
          → supervisor (STRIDE: retrieve/rewrite/answer per sub-query)
          → gap_detector → [has gaps] gap_search → cross_validator
                         → [no gaps]              → cross_validator
          → evidence_auditor (EAM Stage 1+2a: normalize + MASS-RAG claim binding)
          → write_draft → critique → evidence_stage2 (EAM Stage 2b: misalignment flags)
          → [passed] finalize
          → [rewrite] revise → critique → evidence_stage2 (up to N times)
"""

from __future__ import annotations
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command, Send
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from .state import ResearchState
from .providers.base import LLMProvider
from .providers.hybrid import HybridProvider
from .tools.search import SearchTool
from .nodes.plan_generator import generate_plan
from .nodes.checklist_node import build_checklist
from .nodes.search_worker import search_worker
from .nodes.local_search_worker import local_search_worker
from .nodes.reranker import rerank_citations
from .nodes.supervisor import supervise
from .nodes.gap_detector import detect_gaps, gap_search
from .nodes.evidence_auditor import audit_evidence, annotate_misalignments
from .nodes.cross_validator import cross_validate
from .nodes.writer import write_draft
from .nodes.critic import critique, revise, should_revise
from .tools.local_file_search import LocalFileSearch
from .state import DEFAULT_FEATURE_FLAGS


# ── Node Wrappers ──────────────────────────────────────────────────────────

async def plan_generator_node(state: ResearchState, llm: LLMProvider) -> dict:
    return await generate_plan(state, llm)


async def checklist_node(state: ResearchState, llm: LLMProvider) -> dict:
    return await build_checklist(state, llm)


async def plan_review_node(state: ResearchState) -> Command:
    """
    INTERRUPT #1 — User plan approval/modification.
    Routes to search_orchestrator on resume.
    """
    user_response = interrupt({
        "type": "plan_review",
        "plan": state["plan"],
        "message": "Please review the research plan and approve or modify it.",
    })

    if user_response.get("approved"):
        updated_plan = user_response.get("plan") or state["plan"]
        return Command(
            update={
                "plan": updated_plan,
                "plan_approved": True,
                "report_length": user_response.get("report_length", "detailed"),
            },
            goto="search_orchestrator",
        )
    else:
        return Command(
            update={"plan_approved": False},
            goto="generate_plan",
        )


def search_orchestrator_node(state: ResearchState) -> dict:
    """
    Pass-through node for search fan-out.
    fan_out_to_workers in add_conditional_edges returns the list of Send objects.
    """
    return {}


def fan_out_to_workers(state: ResearchState) -> list[Send]:
    """
    Send each sub-query to an independent worker.
    - Always: search_worker (web search)
    - When local_search_enabled=True: also add local_search_worker (local files)

    LangGraph runs this list in parallel and merges results using operator.add.
    feature_flags is included in the Send payload so each worker can read it.
    """
    plan = state.get("plan") or {}
    sub_queries = plan.get("sub_queries", [])
    local_enabled = state.get("local_search_enabled", False)
    flags = state.get("feature_flags", DEFAULT_FEATURE_FLAGS)

    depth = plan.get("depth", "normal")
    sends: list[Send] = []
    for sq in sub_queries:
        sends.append(Send("search_worker", {
            "sub_query": sq,
            "original_query": state["original_query"],
            "depth": depth,
            "feature_flags": flags,
        }))
        if local_enabled:
            sends.append(Send("local_search_worker", {
                "sub_query": sq,
                "original_query": state["original_query"],
            }))

    return sends



async def reranker_node(state: ResearchState) -> dict:
    return await rerank_citations(state)


async def supervisor_node(state: ResearchState, llm: LLMProvider) -> dict:
    return await supervise(state, llm)


async def evidence_auditor_node(state: ResearchState) -> dict:
    return await audit_evidence(state)


def evidence_stage2_node(state: ResearchState) -> dict:
    return annotate_misalignments(state)


async def gap_detector_node(state: ResearchState, llm: LLMProvider) -> dict:
    return await detect_gaps(state, llm)


def should_gap_search(state: ResearchState) -> str:
    """Route to gap_search if gaps exist, otherwise go directly to cross_validator."""
    if state.get("gap_queries"):
        return "gap_search"
    return "cross_validator"


def should_continue_research(state: ResearchState) -> str:
    """
    Routing after gap_search completes.

    deep mode: up to 2 additional rounds of search (re-enter gap_detector if round < 2)
    fast/normal: go directly to cross_validator after 1 round of gap filling
    """
    depth = (state.get("plan") or {}).get("depth", "normal")
    round_num = state.get("research_round", 0)
    if depth == "deep" and round_num < 2:
        return "gap_detector"
    return "cross_validator"


async def gap_search_node(
    state: ResearchState,
    llm: LLMProvider,
    search_tool: SearchTool,
) -> dict:
    return await gap_search(state, llm, search_tool)


async def cross_validator_node(state: ResearchState, llm: LLMProvider) -> dict:
    return await cross_validate(state, llm)



async def write_draft_node(state: ResearchState, llm: LLMProvider) -> dict:
    return await write_draft(state, llm)


async def critique_node(state: ResearchState, llm: LLMProvider) -> dict:
    return await critique(state, llm)


async def revise_node(state: ResearchState, llm: LLMProvider) -> dict:
    return await revise(state, llm)


def finalize_node(state: ResearchState) -> dict:
    return {"final_report": state["draft_report"]}


# ── Graph Builder ───────────────────────────────────────────────────────────

def build_graph(
    llm: LLMProvider,
    search_tool: SearchTool,
    local_file_search: LocalFileSearch | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    agentcore_arns: dict[str, str] | None = None,
) -> StateGraph:
    """
    Build the research graph.

    agentcore_arns (Phase G): optional dict mapping node roles to AgentCore runtime ARNs.
    When provided, the corresponding cloud nodes invoke AgentCore instead of Bedrock directly.

    Keys:
        "writer"  → write_draft node
        "critic"  → critique node (Stage 2 coherence verify)
        "gap"     → gap_detector node
        "plan"    → plan_generator Stage 2

    Unrecognized keys are ignored. Missing keys fall back to the resolved cloud provider.
    If agentcore_arns is None or empty, behaviour is identical to Phase A–F (no change).

    Example:
        build_graph(
            llm=HybridProvider(cloud=BedrockProvider(), local=OllamaProvider()),
            search_tool=...,
            agentcore_arns={
                "writer": "arn:aws:bedrock-agentcore:us-east-1:123456789:agent-runtime/abc",
                "gap":    "arn:aws:bedrock-agentcore:us-east-1:123456789:agent-runtime/def",
            },
        )
    """
    builder = StateGraph(ResearchState)

    # ── Provider routing (Phase B-1) ───────────────────────────────────────
    # When llm is a HybridProvider, route each node to the appropriate sub-provider.
    # Non-hybrid mode: cloud == local == llm (backward compatible, zero change).
    #
    # Tier A nodes (MASS-RAG, Critic) receive the full HybridProvider so they can
    # access .cloud and .local internally for the Speculative RAG pattern (Phase D, E).
    # All other nodes receive a single pre-resolved provider.
    if isinstance(llm, HybridProvider):
        cloud = llm.cloud   # generation: plan Stage 2, gap_detector, writer
        local = llm.local   # evaluation: checklist, supervisor, cross_validator,
                            #             evidence_auditor, quality_scorer, revise
        tier_a = llm        # Spec RAG pattern nodes: search_worker, critique
    else:
        cloud = local = tier_a = llm

    # ── Phase G: AgentCore provider swap ──────────────────────────────────
    # Replace cloud provider for selected nodes with AgentCoreProvider.
    # graph structure (edges, conditions, Send API) is entirely unchanged.
    # Only the provider injected into the node partial changes.
    if agentcore_arns:
        from .providers.agentcore import AgentCoreProvider
        fallback_model = getattr(cloud, "model", "us.anthropic.claude-sonnet-4-6")

        def _ac(role: str) -> LLMProvider:
            """Return AgentCoreProvider for role if ARN present, else cloud fallback."""
            arn = agentcore_arns.get(role, "").strip()
            if arn:
                return AgentCoreProvider(agent_runtime_arn=arn, fallback_model=fallback_model)
            return cloud

        cloud_writer  = _ac("writer")
        cloud_gap     = _ac("gap")
        cloud_plan    = _ac("plan")
        # critic Stage 2 is handled inside critique() via llm.cloud — swap tier_a's cloud
        if agentcore_arns.get("critic") and isinstance(llm, HybridProvider):
            from .providers.hybrid import HybridProvider as HP
            critic_cloud = AgentCoreProvider(
                agent_runtime_arn=agentcore_arns["critic"],
                fallback_model=fallback_model,
            )
            tier_a = HP(cloud=critic_cloud, local=llm.local)
    else:
        cloud_writer = cloud_gap = cloud_plan = cloud

    # local_search_worker: returns empty results (no-op) if local_file_search is not provided
    if local_file_search is not None:
        local_worker = partial(local_search_worker, local_file_search=local_file_search)
    else:
        async def local_worker(state: dict) -> dict:
            return {"citations": []}

    # Register nodes — provider injected per routing table above
    # Phase G: cloud_writer / cloud_gap / cloud_plan swap to AgentCoreProvider when ARNs set.
    #          All other providers and graph structure are unchanged.
    builder.add_node("generate_plan",       partial(plan_generator_node, llm=llm))    # plan uses full llm (Phase C 2-stage)
    builder.add_node("checklist_node",      partial(checklist_node,      llm=local))
    builder.add_node("plan_review",         plan_review_node)
    builder.add_node("search_orchestrator", search_orchestrator_node)
    builder.add_node("search_worker",       partial(search_worker,       llm=tier_a,       search_tool=search_tool))
    builder.add_node("local_search_worker", local_worker)
    builder.add_node("reranker",            reranker_node)
    builder.add_node("supervisor",          partial(supervisor_node,     llm=local))
    builder.add_node("gap_detector",        partial(gap_detector_node,   llm=cloud_gap))   # Phase G swap
    builder.add_node("gap_search",          partial(gap_search_node,     llm=cloud_gap,    search_tool=search_tool))
    builder.add_node("cross_validator",     partial(cross_validator_node,llm=local))
    builder.add_node("evidence_auditor",    evidence_auditor_node)
    builder.add_node("write_draft",         partial(write_draft_node,    llm=cloud_writer)) # Phase G swap
    builder.add_node("critique",            partial(critique_node,       llm=tier_a))       # Phase G: tier_a.cloud swapped if critic ARN set
    builder.add_node("evidence_stage2",     evidence_stage2_node)
    builder.add_node("revise",              partial(revise_node,         llm=local))
    builder.add_node("finalize",            finalize_node)

    # Edges
    builder.set_entry_point("generate_plan")
    builder.add_edge("generate_plan",  "checklist_node")
    builder.add_edge("checklist_node", "plan_review")
    # plan_review → Command(goto="search_orchestrator") or Command(goto="generate_plan")

    # Parallel fan-out: each sub-query → independent search_worker via Send API
    builder.add_conditional_edges(
        "search_orchestrator",
        fan_out_to_workers,
        ["search_worker", "local_search_worker"],
    )

    # All parallel workers complete → reranker (fan-out join) → gap_detector
    builder.add_edge("search_worker",       "reranker")
    builder.add_edge("local_search_worker", "reranker")
    builder.add_edge("reranker",            "supervisor")
    builder.add_edge("supervisor",          "gap_detector")

    # Gap conditional branching
    builder.add_conditional_edges(
        "gap_detector",
        should_gap_search,
        {"gap_search": "gap_search", "cross_validator": "cross_validator"},
    )

    builder.add_conditional_edges(
        "gap_search",
        should_continue_research,
        {"gap_detector": "gap_detector", "cross_validator": "cross_validator"},
    )
    builder.add_edge("cross_validator",  "evidence_auditor")
    builder.add_edge("evidence_auditor", "write_draft")
    builder.add_edge("write_draft",     "critique")

    # EAM Stage 2b: annotate evidence_store with misalignment flags after each critique pass
    builder.add_edge("critique", "evidence_stage2")
    builder.add_conditional_edges(
        "evidence_stage2",
        should_revise,
        {"finalize": "finalize", "revise": "revise"},
    )

    builder.add_edge("revise",   "critique")
    builder.add_edge("finalize", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=saver)
