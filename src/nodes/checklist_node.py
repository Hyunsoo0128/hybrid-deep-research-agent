"""
VCM (Verification Checklist Module) — RhinoInsight (arxiv:2511.18743)

Runs immediately after plan_generator.  Converts the research plan's sub-queries
into a structured checklist of verifiable sub-goals.  Each item tracks completion
status and bound evidence IDs; status is updated by evidence_auditor after search.

Paper role: "generates a structured checklist of sub-goals before research begins,
tracks completion after each search cycle, surfaces uncovered goals to gap detection."
"""

from __future__ import annotations
from ..state import ResearchState
from ..providers.base import LLMProvider
from ..utils.llm_json import llm_json

_SYSTEM = """You are a research planning expert.
Convert research sub-questions into precise, verifiable checklist items."""

_PROMPT = """Research query: {query}

Research sub-queries:
{sub_queries}

For each sub-query, generate one verifiable checklist item — a specific, falsifiable
statement of what the research should establish.

Rules:
- Each item must be independently verifiable (can be checked with a yes/no or specific fact)
- Use the same ID as the sub-query it corresponds to
- Start status as "pending"

Respond ONLY in this JSON format:
{{
  "checklist": [
    {{
      "id": "chk_sq0",
      "subgoal": "Identify at least 2 concrete quantum computing breakthroughs announced in 2024",
      "sub_query_id": "sq0",
      "status": "pending",
      "evidence_ids": []
    }}
  ]
}}"""

_SCHEMA = """{
  "checklist": [
    {"id": "string", "subgoal": "string", "sub_query_id": "string",
     "status": "pending", "evidence_ids": []}
  ]
}"""


async def build_checklist(state: ResearchState, llm: LLMProvider) -> dict:
    """
    VCM Node — builds the initial research checklist from the approved plan.
    Skipped when rhinoinsight flag is off; returns empty checklist.
    """
    flags = state.get("feature_flags", {})
    if not flags.get("rhinoinsight", False):
        return {"checklist": []}

    plan = state.get("plan") or {}
    sub_queries = plan.get("sub_queries", [])
    if not sub_queries:
        return {"checklist": []}

    dsap_on = flags.get("dsap", True)

    sub_queries_text = "\n".join(
        f"  [{sq['id']}] {sq['question']}"
        for sq in sub_queries
    )

    data = await llm_json(
        llm=llm,
        messages=[{
            "role": "user",
            "content": _PROMPT.format(
                query=state["original_query"],
                sub_queries=sub_queries_text,
            ),
        }],
        system=_SYSTEM,
        schema_hint=_SCHEMA,
        max_tokens=1024,
        temperature=0.1,
        dsap_enabled=dsap_on,
        fallback={"checklist": [
            {
                "id": f"chk_{sq['id']}",
                "subgoal": sq["question"],
                "sub_query_id": sq["id"],
                "status": "pending",
                "evidence_ids": [],
            }
            for sq in sub_queries
        ]},
    )

    checklist = data.get("checklist", [])

    # Ensure IDs are unique and have required fields
    seen_ids: set[str] = set()
    validated: list[dict] = []
    for item in checklist:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id") or f"chk_{item.get('sub_query_id', len(validated))}"
        if item_id in seen_ids:
            item_id = f"{item_id}_{len(validated)}"
        seen_ids.add(item_id)
        validated.append({
            "id": item_id,
            "subgoal": item.get("subgoal", ""),
            "sub_query_id": item.get("sub_query_id", ""),
            "status": "pending",
            "evidence_ids": [],
        })

    return {"checklist": validated}
