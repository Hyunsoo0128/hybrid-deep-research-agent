"""
ResearchState — LangGraph global session state definition
"""

from __future__ import annotations
from typing import Annotated, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import operator
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps


# ── Data Models ────────────────────────────────────────────────────────────

class TrustLevel(str, Enum):
    HIGH   = "high"    # Direct user input, official documentation
    MEDIUM = "medium"  # Known domains
    LOW    = "low"     # General web crawl results


class SourceType(str, Enum):
    WEB   = "web"
    LOCAL = "local"


@dataclass
class Citation:
    id: str
    url: str
    title: str
    excerpt: str                        # Actually cited text
    source_type: SourceType
    trust_level: TrustLevel
    crawled_at: str                     # ISO format
    confidence: float                   # 0.0 ~ 1.0
    corroborated_by: list[str] = field(default_factory=list)  # Citation IDs supporting the same fact
    injection_checked: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "excerpt": self.excerpt,
            "source_type": self.source_type.value,
            "trust_level": self.trust_level.value,
            "crawled_at": self.crawled_at,
            "confidence": self.confidence,
            "corroborated_by": self.corroborated_by,
            "injection_checked": self.injection_checked,
        }


@dataclass
class SubQuery:
    id: str
    question: str
    status: str = "pending"            # pending | in_progress | done | failed
    answer: str | None = None
    citation_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "status": self.status,
            "answer": self.answer,
            "citation_ids": self.citation_ids,
        }


@dataclass
class ResearchPlan:
    intent: str                        # factual | analytical | comparative | predictive
    interpretation: str                # The system's interpretation of the query
    sub_queries: list[SubQuery]
    local_files: list[str]
    depth: str                         # fast | normal | deep
    estimated_time: str

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "interpretation": self.interpretation,
            "sub_queries": [sq.to_dict() for sq in self.sub_queries],
            "local_files": self.local_files,
            "depth": self.depth,
            "estimated_time": self.estimated_time,
        }


@dataclass
class CriticFeedback:
    passed: bool
    uncited_claims: list[str]
    unanswered_sub_queries: list[str]
    suggestions: list[str]
    misaligned_claims: list[dict] = field(default_factory=list)

    @property
    def has_logic_errors(self) -> bool:
        return len(self.misaligned_claims) > 0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "has_logic_errors": self.has_logic_errors,
            "uncited_claims": self.uncited_claims,
            "unanswered_sub_queries": self.unanswered_sub_queries,
            "misaligned_claims": self.misaligned_claims,
            "suggestions": self.suggestions,
        }


# ── LangGraph State ─────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    # Session
    session_id: str

    # Query
    original_query: str

    # Plan
    plan: dict | None                  # ResearchPlan.to_dict()
    plan_approved: bool

    # Research (accumulated via append)
    citations: Annotated[list[dict], operator.add]
    research_round: int

    # Local file search (Phase 3)
    local_search_enabled: bool

    # Gaps (Phase 2)
    identified_gaps: list[str]
    gap_queries: list[str]
    gap_search_approved: bool

    # Cross-validation (Phase 2)
    cross_validation_report: dict | None

    # Synthesis
    report_length: str                  # brief | standard | detailed
    draft_report: str
    critic_feedback: dict | None       # CriticFeedback.to_dict()
    revision_count: int
    final_report: str

    # Conversation
    conversation_history: Annotated[list, add_messages]
    conversation_route: str            # memory | targeted | new_research

    # Technique feature flags (on/off per technique)
    feature_flags: dict

    # CRAG retrieval quality signals (accumulated via append, one entry per sub-query)
    # Each entry: {sub_query_id, verdict, max_doc_score, strip_retention_ratio}
    retrieval_quality: Annotated[list[dict], operator.add]

    # MASS-RAG (arxiv:2604.18509): 3-agent parallel synthesis output, one entry per sub-query.
    # Accumulated via operator.add (multiple parallel workers contribute).
    # Schema: [{sub_query_id, question, summary, key_spans: [{text, source_citation_ids, type}],
    #           inferences: [{claim, supporting_span_indices}]}]
    mass_rag_outputs: Annotated[list[dict], operator.add]

    # query_decomp Reranker (arxiv:2507.00355): top-k citations reranked against original query.
    # Plain list (no operator.add) — overwritten by reranker_node after parallel search completes.
    # Downstream nodes prefer this over raw citations when non-empty.
    reranked_citations: list[dict]

    # RhinoInsight VCM (arxiv:2511.18743): structured checklist of research sub-goals.
    # Generated by checklist_node after plan_generator; status updated by evidence_auditor.
    # Plain list (overwrite) — one checklist per session.
    # Schema: [{"id": str, "subgoal": str, "sub_query_id": str, "status": "pending|partial|complete",
    #            "evidence_ids": list[str]}]
    checklist: list[dict]

    # RhinoInsight EAM Stage 1 (arxiv:2511.18743): normalized evidence store.
    # Built by evidence_auditor from effective citations after cross_validator.
    # Plain list (overwrite) — writer prefers this over raw citations when non-empty.
    # Schema: [{"id": str, "url": str, "title": str, "excerpt": str, "confidence": float,
    #            "trust_level": str, "crawled_at": str, "verification_level": str}]
    evidence_store: list[dict]

    # STRIDE Supervisor (arxiv:2604.17405): per-sub-query action decisions.
    # Written by supervisor_node (between reranker and gap_detector).
    # Schema: [{sub_query_id, action: "retrieve|rewrite|answer",
    #           reformulated_question: str|None}]
    supervisor_decisions: list[dict]

    # Circuit breaker
    remaining_steps: RemainingSteps
    error_log: Annotated[list[str], operator.add]


DEFAULT_FEATURE_FLAGS: dict[str, bool] = {
    # ── Stage 1: Search strategy ──────────────────────────────────────────
    "query_decomp":    True,   # arxiv:2507.00355 — Query decomposition + reranking
    "crag":            True,   # arxiv:2401.15884 — Corrective RAG: 3-way relevance classification
    "stride":          False,  # arxiv:2604.17405 — Meta-Planner + Supervisor routing (retrieve/rewrite/answer)
    # ── Stage 2: Evidence building ────────────────────────────────────────
    "mass_rag":        False,  # arxiv:2604.18509 — 3-agent parallel: Summarizer/Extractor/Reasoner + Synthesis
    "rhinoinsight":    False,  # arxiv:2511.18743 — VCM checklist + EAM Stage 1 evidence normalization
    # ── Stage 3: Verification & alignment ────────────────────────────────
    "alignrag":        True,   # arxiv:2504.14858 — Citation-response consistency check (Phase1/2/3)
    "spec_rag_critic": False,  # Phase E — Spec RAG 3-stage critic (drafter/verifier/refiner, HybridProvider only)
    # ── Stage 4: Quality enhancement ─────────────────────────────────────
    "construct":       False,  # arxiv:2603.18014 — Field-level JSON trustworthiness scoring
    "proclaim":        False,  # arxiv:2603.28488 — Courtroom debate + Progressive RAG (not yet implemented)
    "navirag":         False,  # arxiv:2604.12766 — Hierarchical knowledge tree navigation (not yet implemented)
    # ── Infrastructure / always ON ───────────────────────────────────────
    "dsap":            True,   # arxiv:2512.20660 — JSON Guard pattern with error-feedback retry
    "sdp":             False,  # arxiv:2604.17677 — Source Dependency Pruning (pending)
    # ── Settings ──────────────────────────────────────────────────────────
    "privacy_mode":    False,  # Phase F — block raw local file excerpts from reaching cloud LLM
}


def initial_state(
    session_id: str,
    query: str,
    local_search_enabled: bool = False,
    report_length: str = "detailed",
    feature_flags: dict | None = None,
) -> dict:
    """Helper to create the initial state"""
    flags = {**DEFAULT_FEATURE_FLAGS, **(feature_flags or {})}
    return {
        "session_id": session_id,
        "original_query": query,
        "plan": None,
        "plan_approved": False,
        "citations": [],
        "research_round": 0,
        "local_search_enabled": local_search_enabled,
        "identified_gaps": [],
        "gap_queries": [],
        "gap_search_approved": False,
        "cross_validation_report": None,
        "report_length": report_length,
        "draft_report": "",
        "critic_feedback": None,
        "revision_count": 0,
        "final_report": "",
        "conversation_history": [],
        "conversation_route": "",
        "feature_flags": flags,
        "retrieval_quality": [],
        "mass_rag_outputs": [],
        "reranked_citations": [],
        "checklist": [],
        "evidence_store": [],
        "supervisor_decisions": [],
        "error_log": [],
    }
