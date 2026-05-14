# Architecture — Stage-Aware Local-Cloud Inference

This document describes the system architecture as implemented and evaluated in the paper:

> **Stage-Aware Local-Cloud Inference: Hybrid Pipelines Consistently Outperform Matched Cloud-Only Baselines**

Last updated: 2026-05-14

---

## Stage-Routing Principle

Stage-Aware Local-Cloud Inference rests on the observation that pipeline stages differ fundamentally in reasoning demand (Kahneman, 2011). The dividing criterion is **input scope**:

- **System 1 (local)**: A stage that processes a single document or draft requires only fast, bounded-context operations. Retrieval classification, passage scoring, section drafting, and self-critique each process a single bounded input — a document excerpt, a scored passage, or a draft paragraph — with no cross-document lookup.

- **System 2 (cloud)**: A stage that must simultaneously integrate evidence across multiple independent sources requires deliberate reasoning. Cross-document synthesis integrates claims from independent documents; coverage-gap detection surveys the full evidence state for under-covered topics — both require deliberate cross-document reasoning beyond what local models achieve in isolation.

Restricting cloud calls to System 2 stages enforces a **Privacy Boundary** by construction.

---

## Privacy Boundary

The Privacy Boundary is formalized as a constraint on the inputs to the cloud model call `g_c`:

```
inputs(g_c) ∩ {q, C} = ∅
```

where `q` is the original query and `C` is the full document corpus. The cloud receives only compact local-model outputs — plan skeletons, section drafts, or coverage summaries — never the private inputs directly.

**What the cloud receives:**
- Document abstractions: titles and 150-character excerpts, never full document bodies
- Local drafts: compact prose produced by the local model, containing no verbatim document excerpts

**What the cloud never receives:**
- Original user queries
- Retrieved document bodies
- Intermediate documents or file contents

This constraint is enforced structurally: `q` and `C` are never passed as arguments to any cloud API call.

---

## Four-Phase Pipeline

The four pipeline phases execute in sequence: **Planning → Retrieval → Drafting → Verification**. In each phase, System 1 operations execute locally and System 2 operations execute on the cloud.

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js Frontend                          │
│   Query → Plan Approval → Live Progress → Report + Chat      │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / SSE
┌──────────────────────────▼──────────────────────────────────┐
│                  FastAPI + SSE Server                        │
└───────────┬──────────────────────────┬──────────────────────┘
            │                          │
   ┌────────▼────────┐       ┌─────────▼────────┐
   │  Research Graph │       │   Chat Graph      │
   │  (LangGraph)    │       │   (LangGraph)     │
   └────────┬────────┘       └───────────────────┘
            │
   ┌────────▼──────────────────────────────────────┐
   │  PLANNING PHASE                               │
   │  generate_plan    [LOCAL]  Sq skeleton        │
   │  plan_elaboration [CLOUD]  Cq from skeleton   │
   │  checklist_node   [LOCAL]  VCM sub-goals      │
   │  [INTERRUPT: plan_review]                     │
   └────────┬──────────────────────────────────────┘
            │ N sub_queries (Send API fan-out)
   ┌────────▼──────────────────────────────────────┐
   │  RETRIEVAL PHASE                              │
   │  search_worker × N  [LOCAL]  CRAG 1st-pass    │
   │  crag_recheck       [CLOUD]  AMBIGUOUS only   │
   │                              (title+excerpt)  │
   │  reranker           [LOCAL]  cross-encoder    │
   └────────┬──────────────────────────────────────┘
            │
   ┌────────▼──────────────────────────────────────┐
   │  DRAFTING PHASE                               │
   │  mass_rag_drafters  [LOCAL]  parallel drafts  │
   │  synthesis          [CLOUD]  from drafts only │
   └────────┬──────────────────────────────────────┘
            │
   ┌────────▼──────────────────────────────────────┐
   │  VERIFICATION PHASE                           │
   │  gap_detector    [CLOUD]  coverage-gap survey │
   │  gap_search      [LOCAL]  targeted re-retrieval│
   │  cross_validator [LOCAL]  consistency check   │
   │  evidence_auditor[LOCAL]  EAM claim binding   │
   │  critique        [LOCAL]  AlignRAG self-critique│
   │  revise          [LOCAL]  section rewrite     │
   └───────────────────────────────────────────────┘
```

---

## Phase Details

### Planning Phase

Local models decompose the query and generate an abstract research skeleton (System 1); the cloud elaborates each section with concrete execution steps (System 2), receiving only the plan skeleton.

- `generate_plan` [LOCAL]: STRIDE Meta-Planner generates entity-agnostic abstract strategy Sq with [ENTITY] slots; query_decomp applies 5-dimension framework
- `plan_elaboration` [CLOUD]: Derives concrete plan Cq from Sq — cloud receives only the skeleton, not the original query
- `checklist_node` [LOCAL]: RhinoInsight VCM generates verifiable sub-goal checklist

### Retrieval Phase

Local models classify and score document relevance entirely locally (System 1); uncertain cases are escalated to the cloud for re-adjudication using only document titles and short excerpts (System 2).

- `search_worker × N` [LOCAL]: CRAG 3-way classification (CORRECT/AMBIGUOUS/INCORRECT) + Decompose-then-Recompose
- `crag_recheck` [CLOUD]: Re-adjudicates AMBIGUOUS cases using title + 150-char excerpt only — resolves over-conservative tendency without exposing document bodies
- `reranker` [LOCAL]: Cross-encoder reranking (ms-marco-MiniLM, ONNX) against original query

### Drafting Phase

Local models produce section drafts in parallel (System 1); the cloud synthesizes these into a coherent final report, receiving only the draft text — never the underlying documents.

- `mass_rag_drafters` [LOCAL]: MASS-RAG 3-agent parallel drafting (Summarizer/Extractor/Reasoner) — each agent reads source documents directly
- `synthesis` [CLOUD]: Synthesizes 3-agent outputs into coherent report — receives only locally-generated draft text, no raw documents

### Verification Phase

The cloud first surveys the assembled evidence to identify coverage gaps and trigger targeted re-retrieval (System 2); then local models extract claims, check cross-source consistency, and assign trust scores within the complete evidence set (System 1).

- `gap_detector` [CLOUD]: Surveys locally-compiled coverage index (structured record of retrieved topics and source counts) to identify under-covered topics
- `gap_search` [LOCAL]: Targeted re-retrieval for identified gaps
- `cross_validator` [LOCAL]: Cross-source consistency checking within fixed, bounded evidence pool
- `evidence_auditor` [LOCAL]: RhinoInsight EAM claim binding + misalignment flagging
- `critique` [LOCAL]: AlignRAG 3-phase self-critique
- `revise` [LOCAL]: Section-by-section rewrite from feedback

---

## Node Routing Table

```
Node                        Route     Technique              Cloud receives
──────────────────────────────────────────────────────────────────────────────
generate_plan               LOCAL     STRIDE Sq + query_decomp  —
plan_elaboration            CLOUD     STRIDE Cq              plan skeleton
checklist_node              LOCAL     RhinoInsight VCM       —

search_worker × N (parallel)
  CRAG scoring              LOCAL     CRAG 1st-pass          —
  CRAG AMBIGUOUS recheck    CLOUD     CRAG re-adjudication   title + 150-char excerpt
  MASS-RAG Summarizer       LOCAL     MASS-RAG draft         —
  MASS-RAG Extractor        LOCAL     MASS-RAG draft         —
  MASS-RAG Reasoner         LOCAL     MASS-RAG draft         —
  MASS-RAG Synthesis        CLOUD     MASS-RAG synthesis     3-agent draft text only
  CONSTRUCT scorer          LOCAL     CONSTRUCT              —

reranker                    LOCAL     Spec. Reranking (ONNX) —
supervisor                  LOCAL     STRIDE Supervisor      —

gap_detector                CLOUD     Coverage-gap detection coverage index (local-generated)
gap_search                  LOCAL     —                      —
cross_validator             LOCAL     —                      —
evidence_auditor            LOCAL     RhinoInsight EAM       —

critique                    LOCAL     AlignRAG               —
revise                      LOCAL     DSAP JSON guard        —
──────────────────────────────────────────────────────────────────────────────
Total cloud calls: ~6–8              Total local calls: ~18–22
```

---

## Nine RAG Technique Adaptations

Each technique is partitioned across System 1 (local) and System 2 (cloud) tiers:

| Phase | Technique | System 1 (Local) | System 2 (Cloud) | Cloud receives |
|-------|-----------|-----------------|-----------------|----------------|
| Planning | Query Decomp | sub-query generation | — | none |
| Planning | STRIDE | abstract planning (Sq) | concrete planning (Cq) | plan skeleton |
| Planning | DSAP | section outline | section elaboration | outline |
| Retrieval | CRAG | 1st-pass classify | re-evaluate uncertain | doc title + excerpt |
| Retrieval | Spec. Rerank | cross-encoder scoring | — | none |
| Drafting | MASS-RAG | parallel drafting | multi-draft synthesis | local drafts |
| Drafting | AlignRAG | self-critique rewrite | — | none |
| Verification | RhinoInsight | claim extraction + trust scoring | — | none |
| Verification | CONSTRUCT | evidence structuring + consistency | — | none |

Techniques with no System 2 stage (Query Decomp, Speculative Reranking, AlignRAG, RhinoInsight, CONSTRUCT) execute entirely locally and expose no data to the cloud.

---

## LangGraph State Machine

Two LangGraph state machines — research and chat — share a common LLM provider layer.

### Research Graph (Implemented)

```
generate_plan → checklist_node → [interrupt: plan_review]
  → search_orchestrator
      → [Send API parallel fan-out] search_worker × N
      │    inside per sub-query:
      │      crag (3-way verdict) → MASS-RAG 3-agent + Synthesis
      │      if construct=True → CONSTRUCT score → trust_scores in entry
      │    + local_search_worker × N
      → reranker  (dedup + cross-encoder top-k)
      → supervisor   (STRIDE: retrieve/rewrite/answer per sub-query)
      → gap_detector (CRAG hints + VCM hints + STRIDE hints)
           → [has gaps] gap_search ─┐
           → [no gaps]              ┘→ cross_validator
                                        → evidence_auditor
                                        → write_draft
                                        → critique
                                        → evidence_stage2
                                          → [passed]  finalize
                                          → [revise]  revise → critique → evidence_stage2 (loop)
```

### HybridProvider

```python
class HybridProvider:
    """Routes each node to System 1 (local) or System 2 (cloud) tier."""

    CLOUD_NODES = {"plan_elaboration", "crag_recheck", "synthesis", "gap_detector"}

    def __init__(self, cloud: LLMProvider, local: LLMProvider):
        self._cloud = cloud
        self._local = local

    async def complete(self, messages, system="", node_hint="", **kwargs) -> str:
        p = self._cloud if node_hint in self.CLOUD_NODES else self._local
        return await p.complete(messages, system=system, **kwargs)

    async def embed(self, text: str) -> list[float]:
        return await self._local.embed(text)
```

`LLM_PROVIDER=hybrid` in `.env` activates this routing. `embed()` always routes to local.

---

## State Schema

```python
# Planning
plan: dict                    # sub_queries, interpretation, research_skeleton
checklist: list[dict]         # [{id, subgoal, sub_query_id, status, evidence_ids}]

# Retrieval
citations: Annotated[list[dict], operator.add]  # accumulated across parallel workers
retrieval_quality: Annotated[list[dict], operator.add]  # CRAG verdicts per sub-query
reranked_citations: list[dict]  # top-k after cross-encoder (plain overwrite)

# Drafting
mass_rag_outputs: Annotated[list[dict], operator.add]
# per sub-query: {sub_query_id, summary, key_spans, inferences}
# + trust_scores when construct=True: {document_score, per_field, untrustworthy_fields}

# Verification
evidence_store: list[dict]    # normalized evidence with claim_bindings + misalignment_flags
cross_validation_report: dict
supervisor_decisions: list[dict]  # STRIDE: retrieve/rewrite/answer per sub-query
critic_feedback: dict         # {passed, misaligned_claims, uncited_claims, ...}

# Output
final_report: str
```

---

## Why Local Models Improve System 1 Stages

An unexpected finding from the paper: replacing frontier-model System 1 stages with 2–4B local models not only preserves but **improves** overall quality. Two complementary mechanisms:

1. **Task-scope alignment**: Frontier models tend to over-elaborate outputs for constrained generation tasks such as CRAG classification or trust scoring, producing verbose reasoning where a concise label is required. Smaller models, with narrower output distributions, stay on-task more reliably.

2. **Synthesis leverage**: When local models generate section drafts, the cloud synthesis stage receives imperfect but diverse raw material, creating genuine revision leverage that is absent when a frontier draft is already near-final.

These mechanisms are consistent with the confound-control results: the hybrid gain is present regardless of which cloud model handles System 2 stages, ruling out model-diversity as the sole explanation.

---

## Performance Characteristics

From the paper (N=600 per condition, 120 queries × 5 runs):

| Configuration | Quality (Med) | Cloud Tokens/q | Cost/q | Latency/q |
|---------------|---------------|----------------|--------|-----------|
| Cloud-only Sonnet 4.6 | 0.798 | 136,891 | $1.128 | 270s |
| exaone3.5:2.4b + Sonnet 4.6 | **0.869** | 45,918 | $0.375 | 233s |
| gemma3:4b + Sonnet 4.6 | 0.867 | 47,330 | $0.379 | 312s |
| exaone3.5:2.4b + Haiku 4.5 | 0.828 | 42,568 | $0.093 | 155s |
| exaone3.5:2.4b (all-local) | 0.802 | 0 | $0.000 | 174s |

Hardware: NVIDIA L4 GPU (24GB VRAM), Ubuntu 22.04. Local models served via Ollama with Q4_K_M quantization.

---

## Design Decisions

### D1. System 1/System 2 Boundary

The boundary is determined by input scope, not model capability. A stage is System 1 if it processes a single bounded input (one document, one draft, one passage). A stage is System 2 if it must integrate across multiple independent sources simultaneously.

This means CRAG classification (one document at a time) is System 1, while coverage-gap detection (surveying the full evidence state) is System 2 — even though both could theoretically be handled by either tier.

### D2. CRAG AMBIGUOUS Re-adjudication

Small local models over-predict AMBIGUOUS due to low calibration confidence. The cloud re-adjudication step resolves this: when the local verdict is AMBIGUOUS, the cloud re-classifies using only document titles and 150-character excerpts — resolving the over-conservative tendency without exposing document bodies to the cloud.

### D3. STRIDE Stage Separation

STRIDE generates an abstract research plan (Sq) followed by a concrete execution plan (Cq) in a single model. We split these stages: the local model generates Sq directly from the original query, then the cloud model generates Cq from Sq and a topic summary derived locally — preserving frontier-model planning quality while keeping the original query local.

### D4. MASS-RAG Speculative Synthesis

MASS-RAG assigns multiple agents to draft sections in parallel. In the hybrid variant, local models produce parallel section drafts; the cloud synthesizes these into a coherent final report, receiving only the drafted text — no retrieved documents.

### D5. Verification Phase Ordering

The Verification phase begins with cloud-based gap detection (System 2) that surveys the assembled evidence before claim verification begins. This ordering is critical: verifying claims against an incomplete evidence set yields unreliable results. Once the evidence set is complete, local models handle consistency checking (System 1) — pattern-matching over a known, bounded set requires no integrative reasoning.

### D6. All-Local Fallback

When `LLM_PROVIDER=ollama`, all System 2 stages fall back to the local model. This enables air-gapped deployment at a 6–7 point quality gap vs. Hybrid+Sonnet (0.802 vs. 0.869). The all-local path is validated: exaone3.5:2.4b all-local (0.802) exceeds cloud-only Haiku (0.671) and cloud-only Llama 70B (0.688).
