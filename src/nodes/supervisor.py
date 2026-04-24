"""
STRIDE Supervisor Node — arxiv:2604.17405

Runs between reranker and gap_detector.

Evaluates retrieval quality per sub-query (from CRAG verdicts) and assigns
one of three actions:
  answer:   CORRECT verdict with sufficient evidence → no further retrieval needed
  retrieve: INCORRECT or AMBIGUOUS → gap_detector should add more retrieval
  rewrite:  INCORRECT with low relevance → reformulate the sub-query itself
            (provides reformulated_question for gap_detector to use)

Output written to state["supervisor_decisions"]. gap_detector reads this and
incorporates "rewrite" reformulations as a third hint axis (STRIDE rewrites).

Known deviation: paper's Supervisor uses a full dependency graph between sub-queries
to determine execution order and data flow. API-only implementation uses per-sub-query
CRAG verdicts without dependency tracking. Dependency-aware ordering deferred to future
work requiring graph redesign.
"""

from __future__ import annotations
from ..state import ResearchState
from ..providers.base import LLMProvider
from ..utils.llm_json import llm_json

_SUPERVISOR_SYSTEM = """You are a research orchestration supervisor.
Evaluate retrieved information quality and decide the next action for each sub-query."""

_SUPERVISOR_PROMPT = """Evaluate search results for each sub-query and assign an action.

Original question: {query}

Sub-query results (CRAG verdict + max relevance score):
{sub_query_results}

For each sub-query, assign ONE action:
- "answer":   sufficient CORRECT evidence found → no further search needed
- "retrieve": INCORRECT or AMBIGUOUS results → needs more retrieval on the same question
- "rewrite":  INCORRECT with low relevance scores (below 0.4) → the sub-query itself
              needs reformulation; provide a reformulated_question that searches for
              the same information from a different angle

Respond ONLY in the following JSON format:
[
  {{"sub_query_id": "sq1", "action": "answer", "reformulated_question": null}},
  {{"sub_query_id": "sq2", "action": "retrieve", "reformulated_question": null}},
  {{"sub_query_id": "sq3", "action": "rewrite", "reformulated_question": "alternative search query for same goal"}}
]"""

_SUPERVISOR_SCHEMA = """[{"sub_query_id": "string", "action": "answer|retrieve|rewrite", "reformulated_question": "string|null"}]"""

# Low relevance threshold for "rewrite" recommendation
_REWRITE_SCORE_THRESHOLD = 0.4


async def supervise(state: ResearchState, llm: LLMProvider) -> dict:
    """
    STRIDE Supervisor Node.

    Reads CRAG retrieval_quality from state, assigns action per sub-query,
    writes supervisor_decisions to state.

    Skipped when stride=False — returns empty dict.
    """
    flags = state.get("feature_flags", {})
    if not flags.get("stride", False):
        return {}

    plan = state.get("plan") or {}
    sub_queries = plan.get("sub_queries", [])
    if not sub_queries:
        return {}

    retrieval_quality = state.get("retrieval_quality", [])
    dsap_on = flags.get("dsap", True)

    # Build sub-query quality summary for the prompt
    verdict_map = {rq["sub_query_id"]: rq for rq in retrieval_quality}
    sub_query_results_lines = []
    for sq in sub_queries:
        sq_id = sq["id"]
        rq = verdict_map.get(sq_id, {})
        verdict = rq.get("verdict", "UNKNOWN")
        score = rq.get("max_doc_score", 0.0)
        sub_query_results_lines.append(
            f"[{sq_id}] {sq['question']}\n"
            f"  verdict={verdict}, max_relevance_score={score:.2f}"
        )

    data = await llm_json(
        llm=llm,
        messages=[{
            "role": "user",
            "content": _SUPERVISOR_PROMPT.format(
                query=state["original_query"],
                sub_query_results="\n".join(sub_query_results_lines),
            ),
        }],
        system=_SUPERVISOR_SYSTEM,
        schema_hint=_SUPERVISOR_SCHEMA,
        max_tokens=512,
        temperature=0.1,
        dsap_enabled=dsap_on,
        fallback=[],
    )

    # Normalize: llm_json may wrap list output in a dict
    decisions: list[dict] = data if isinstance(data, list) else []

    # Heuristic override: if LLM returned "retrieve" for a sub-query with very low score,
    # upgrade to "rewrite" to ensure reformulation is attempted
    verdict_map_by_id = {rq["sub_query_id"]: rq for rq in retrieval_quality}
    sq_question_map = {sq["id"]: sq["question"] for sq in sub_queries}

    enriched: list[dict] = []
    for decision in decisions:
        sq_id = decision.get("sub_query_id", "")
        action = decision.get("action", "retrieve")
        rq = verdict_map_by_id.get(sq_id, {})
        score = rq.get("max_doc_score", 1.0)

        # Heuristic: INCORRECT + very low score → rewrite (if LLM said retrieve)
        if (
            action == "retrieve"
            and rq.get("verdict") == "INCORRECT"
            and score < _REWRITE_SCORE_THRESHOLD
            and not decision.get("reformulated_question")
        ):
            # No LLM call — generate a simple reformulation heuristic
            original_q = sq_question_map.get(sq_id, "")
            action = "rewrite"
            decision = {
                **decision,
                "action": "rewrite",
                "reformulated_question": f"{original_q} (alternative formulation)",
            }

        enriched.append({**decision, "action": action})

    return {"supervisor_decisions": enriched}
