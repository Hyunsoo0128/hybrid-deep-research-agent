# RAG Techniques: Paper Analysis & Implementation

Last updated: 2026-05-14

This document covers the 9 RAG techniques implemented in this pipeline, adapted to the Stage-Aware Local-Cloud Inference architecture. Each technique is partitioned across System 1 (local) and System 2 (cloud) tiers.

---

## Implementation Status Overview

| # | Technique | arxiv | Phase | System 1 (Local) | System 2 (Cloud) | File |
|---|-----------|-------|-------|-----------------|-----------------|------|
| 1 | CRAG | 2401.15884 | Retrieval | 1st-pass classify | AMBIGUOUS re-adjudication | `search_worker.py` |
| 2 | Query Decomp + Reranker | 2507.00355 | Planning | sub-query gen + reranking | — | `plan_generator.py`, `reranker.py` |
| 3 | RhinoInsight (VCM + EAM) | 2511.18743 | Planning + Verification | checklist + claim binding | — | `checklist_node.py`, `evidence_auditor.py` |
| 4 | MASS-RAG | 2604.18509 | Drafting | parallel drafting (3 agents) | multi-draft synthesis | `search_worker.py` |
| 5 | AlignRAG | 2504.14858 | Verification | self-critique + rewrite | — | `critic.py` |
| 6 | DSAP | 2512.20660 | Cross-cutting | JSON guard functions | — | `utils/llm_json.py` |
| 7 | STRIDE | 2604.17405 | Planning | abstract planning (Sq) | concrete planning (Cq) | `plan_generator.py`, `supervisor.py` |
| 8 | CONSTRUCT | 2603.18014 | Verification | evidence structuring + consistency | — | `quality_scorer.py` |
| 9 | Speculative Reranking | 2407.08223 | Retrieval | cross-encoder scoring | — | `reranker.py` |

---

## 1. CRAG — Corrective Retrieval-Augmented Generation
**arxiv**: 2401.15884 | **File**: `src/nodes/search_worker.py`

### Paper Summary
Evaluates retrieved documents before generation and takes corrective action based on quality. Three-way branch: CORRECT (high quality) → Decompose-then-Recompose, INCORRECT (low quality) → web search fallback, AMBIGUOUS → both paths. Key algorithm: split documents into sentence-level strips, filter irrelevant strips, reassemble into clean context.

**Performance**: PopQA +14.5%, PubHealth +13.1% vs naive RAG.

### Stage-Aware Adaptation

| Operation | Tier | Rationale |
|-----------|------|-----------|
| 1st-pass classification | System 1 (local) | Single document at a time — bounded context |
| AMBIGUOUS re-adjudication | System 2 (cloud) | Boundary cases require better calibration |
| Decompose-then-Recompose | System 1 (local) | Strip-level pattern matching |

**Privacy**: Cloud receives only document title + 150-character excerpt for AMBIGUOUS cases. Never receives full document bodies.

### Implementation

| Element | Paper | Implementation |
|---------|-------|----------------|
| 3-way classification | CORRECT/INCORRECT/AMBIGUOUS | ✅ CORRECT ≥ 0.5 / AMBIGUOUS 0.3–0.5 / INCORRECT < 0.3 |
| Evaluator | T5-large fine-tuned | LLM batch (float score per doc) — acceptable approximation |
| Decompose-then-Recompose | Strip-level decompose → filter → reassemble | ✅ A2 batch: 1 LLM call, strips ≥ 0.5 retained |
| AMBIGUOUS path | Internal refinement + web search | ✅ Cloud re-adjudication (title+excerpt only) |
| INCORRECT path | Web search fallback | ✅ gap_detector delegation |

---

## 2. Query Decomposition + Speculative Reranking
**arxiv**: 2507.00355 | **Files**: `src/nodes/plan_generator.py`, `src/nodes/reranker.py`

### Paper Summary
Decomposes complex multi-hop questions into sub-questions for parallel retrieval (improves recall), then re-ranks all candidate documents against the **original query** (restores precision). The Reranker is the core contribution: MRR@10 +36.7%, F1 +11.6%.

### Stage-Aware Adaptation

Entirely System 1 — no cloud contact. ONNX cross-encoder inference runs locally.

### Implementation

| Element | Paper | Implementation |
|---------|-------|----------------|
| Sub-question decomposition | Dynamic number based on complexity | 5-dimension framework (enhancement) |
| Original query preserved | Q = {q} ∪ Decompose(q) | ✅ sq0 always prepended |
| Reranker | BGE-reranker-large, scored against original query | cross-encoder/ms-marco-MiniLM-L-6-v2 via fastembed ONNX |
| Top-k | Configurable | fast=10, normal=20, deep=40 |

---

## 3. RhinoInsight — VCM + EAM
**arxiv**: 2511.18743 | **Files**: `src/nodes/checklist_node.py`, `src/nodes/evidence_auditor.py`

### Paper Summary
Addresses "context rot" in linear research pipelines through two control modules. VCM generates a structured checklist of sub-goals before research begins, tracks completion, surfaces uncovered goals to gap detection. EAM normalizes retrieved results into a structured evidence store and binds specific citations to specific claims.

**Performance**: GAIA +10.6% (58.3% → 68.9%), HLE 27.1% accuracy.

### Stage-Aware Adaptation

Entirely System 1 — checklist generation and evidence normalization are bounded-context operations on single documents/claims.

### Implementation

| Element | Paper | Implementation |
|---------|-------|----------------|
| VCM checklist generation | Before research begins | ✅ checklist_node after plan_generator |
| VCM status tracking | After each search cycle | ✅ evidence_auditor: claim_bindings-based |
| VCM → gap_detector | Surface pending items | ✅ PENDING/PARTIAL → VCM hint axis |
| EAM Stage 1: normalize | source/timestamp/confidence | ✅ evidence_store: dedup + sort + verification_level |
| EAM Stage 2a: claim binding | outline alignment per node | ✅ mass_rag key_spans → claim_bindings |
| EAM Stage 2b: misalignment | claim-level citation flags | ✅ critic_feedback → misalignment_flags |
| Markovian compression | Context slicing per step | ❌ Not implemented (out of scope for API-based system) |

---

## 4. MASS-RAG — Multi-Agent Synthesis RAG
**arxiv**: 2604.18509 | **File**: `src/nodes/search_worker.py`

### Paper Summary
Replaces single-path document filtering with three parallel specialist agents: Summarizer (abstract summaries), Extractor (verbatim supporting spans), Reasoner (cross-document inference). A Synthesis Agent reconciles the three outputs.

**Performance**: ARC-Challenge +27.1%, ASQA EM +19.9%.

### Stage-Aware Adaptation

| Operation | Tier | Rationale |
|-----------|------|-----------|
| Summarizer, Extractor, Reasoner | System 1 (local) | Each processes a bounded document set |
| Synthesis | System 2 (cloud) | Integrates across multiple agent outputs |

**Privacy**: Cloud receives only locally-generated draft text (summary, key_spans, inferences). Never receives retrieved documents.

### Implementation

| Element | Paper | Implementation |
|---------|-------|----------------|
| 3-agent parallel | Summarizer/Extractor/Reasoner on same doc pool | ✅ asyncio.gather |
| Synthesis Agent | 4th agent combining all outputs | ✅ separate call after gather |
| Output schema | summary + spans + inferences | ✅ {summary, key_spans, inferences} |
| fast depth guard | Skip when depth=fast | ✅ depth != "fast" check |
| DSAP guard | JSON retry on structured calls | ✅ llm_json on all agents |

**Known deviation**: Paper targets single-query QA benchmarks; this system generates long-form multi-section reports. Synthesis output feeds the writer as PRIMARY SOURCE rather than being the final answer directly.

---

## 5. AlignRAG — Alignment-Based RAG Critic
**arxiv**: 2504.14858 | **File**: `src/nodes/critic.py`

### Paper Summary
Trains a dedicated Critic Language Model (CLM) via Contrastive Critique Synthesis (CCS) to diagnose three retrieval-reasoning misalignment types: Phase 1 (relevance assessment error), Phase 2 (query-evidence mapping failure), Phase 3 (evidence-integrated synthesis error). CLM outputs an edit signal; generator refines iteratively until CLM emits [Good] token.

**Performance**: Average accuracy 62.8%, 8B CLM outperforms 72B general LLM by +2.2%.

### Stage-Aware Adaptation

Entirely System 1 — self-critique is a bounded-context operation (checking one claim against one source at a time).

### Implementation

| Element | Paper | Implementation |
|---------|-------|----------------|
| Phase 1: Relevance | Span importance misalignment | ✅ 3-phase diagnosis in critic.py |
| Phase 2: Query-Evidence | Relationship mapping failure | ✅ same pipeline |
| Phase 3: Synthesis | Unsupported conclusion | ✅ same pipeline |
| Output format | [Good]/[Bad] token + edit signal | ✅ structured JSON per misalignment |
| Dynamic termination | [Good] → exit | ✅ code-computed `passed` |
| Revision signal | Per-phase correction | ✅ phase-grouped prompt in revise() |

**Known deviation**: CLM fine-tuning (CCS+CFT) not reproducible via API-only access. Approximated by structured 3-phase prompt with explicit misalignment schema.

---

## 6. DSAP — Dual-State Architecture for Reliable LLM Agents
**arxiv**: 2512.20660 | **File**: `src/utils/llm_json.py`

### Paper Summary
Addresses LLM non-determinism through Guard functions and three-level recovery. Level 1 — Context Refinement: retry with accumulated error feedback. Level 2 — Informed Backtracking: on stagnation, cascade-invalidate dependent steps. Level 3 — Human Escalation.

**Performance**: Reliability improvement up to +66 pp (SWE-Bench).

### Stage-Aware Adaptation

Entirely System 1 — applied to all local model calls. Critical for System 1 reliability since local models produce more malformed JSON than frontier models.

### Implementation

| Element | Paper | Implementation |
|---------|-------|----------------|
| Level 1: Context Refinement | Error + schema feedback retry | ✅ _RETRY_SYSTEM + _RETRY_USER |
| Level 2: Stagnation detection | Similarity θ≥0.7 over error messages | ✅ exact match on normalized fingerprint |
| Level 2: Strategy switch | Alternative approach on stagnation | ✅ _LAST_RESORT_SYSTEM clean-slate prompt |
| Level 2: Cascade invalidation | Invalidate dependent downstream nodes | ❌ Not implemented |
| Level 3: Escalation | Workflow-level retry | ❌ Not implemented |

---

## 7. STRIDE — Strategic Iterative Decision-Making
**arxiv**: 2604.17405 | **Files**: `src/nodes/plan_generator.py`, `src/nodes/supervisor.py`

### Paper Summary
Solves premature entity grounding in iterative RAG. Meta-Planner generates abstract strategy Sq based on entity types (not specific entities), then converts to concrete plan Cq. Supervisor manages execution state, decides retrieve/rewrite/answer per sub-query.

**Performance**: 2WikiMultihopQA +7.0% EM, token usage −54 to −71%.

### Stage-Aware Adaptation

| Operation | Tier | Rationale |
|-----------|------|-----------|
| Abstract planning (Sq) | System 1 (local) | Bounded to the query — entity-agnostic skeleton |
| Concrete planning (Cq) | System 2 (cloud) | Requires frontier reasoning to elaborate |
| Supervisor routing | System 1 (local) | retrieve/rewrite/answer is a classification task |

**Privacy**: Cloud receives only the plan skeleton (Sq) — never the original query text.

### Implementation

| Element | Paper | Implementation |
|---------|-------|----------------|
| Meta-Planner (Sq) | Entity-agnostic abstract strategy | ✅ _generate_abstract_strategy(): Sq with [ENTITY] slots |
| Cq derivation | Sq → concrete plan with real entities | ✅ _STRIDE_CQ_PROMPT: guided generation using Sq |
| Supervisor: retrieve/rewrite/answer | Per-sub-query action | ✅ supervisor.py: CRAG verdict → action assignment |
| Supervisor: rewrite injection | Reformulated question → re-search | ✅ rewrite → gap_detector STRIDE hint axis |
| Dependency graph (Ω) | Dependency-aware parallel scheduling | ❌ Not implemented |

---

## 8. CONSTRUCT — Real-Time Trustworthiness Scoring
**arxiv**: 2603.18014 | **File**: `src/nodes/quality_scorer.py`

### Paper Summary
Scores the trustworthiness of each field in LLM-generated structured output using a Judge LLM. Works on black-box APIs without logprobs or fine-tuning. Enables targeted re-generation of only low-confidence fields.

### Stage-Aware Adaptation

Entirely System 1 — trustworthiness scoring is a bounded classification task on a single output field.

### Implementation

| Element | Paper | Implementation |
|---------|-------|----------------|
| Applied to | Any structured JSON | MASS-RAG outputs (3-agent synthesis) |
| Template count | 5 verifier templates | 2 (Document-level + Field-level) |
| Score storage | Per-field in output | mass_rag_outputs[i]["trust_scores"] |
| Writer use | Targeted regen | Hedging language hint in synthesis block |
| Critic use | CLM re-diagnosis | CONSTRUCT trust alerts in _PROMPT |
| Threshold | Configurable | TRUST_THRESHOLD = 0.5 |

**Known deviation**: Paper uses 5 verifier templates; this uses 2. ~70% of full effect at 40% cost.

---

## 9. Speculative Reranking
**arxiv**: 2407.08223 | **File**: `src/nodes/reranker.py`

### Paper Summary
Original paper's goal is inference speed optimization via small drafter + large verifier. Reinterpreted here as cross-encoder reranking for retrieval precision.

### Stage-Aware Adaptation

Entirely System 1 — ONNX cross-encoder inference runs locally with no cloud contact.

### Implementation

Cross-encoder/ms-marco-MiniLM-L-6-v2 via fastembed ONNX backend (~80MB). Scores `title + excerpt` against original query. Top-k by depth: fast=10, normal=20, deep=40.

---

## Technique Interaction Summary

| Combination | Effect |
|-------------|--------|
| Query Decomp → CRAG → MASS-RAG | Coverage → precision → depth funnel |
| STRIDE + VCM | Structure (STRIDE) + tracking (VCM): complementary at planning stage |
| EAM + AlignRAG + CONSTRUCT | Normalize → consistency check → quantify: 3-layer verification chain |
| DSAP + CONSTRUCT | DSAP = structural validity of JSON; CONSTRUCT = content validity |
| CRAG AMBIGUOUS → cloud re-adjudication | Resolves local model over-conservatism without document exposure |
| MASS-RAG local drafts → cloud synthesis | Creates revision leverage; explains why hybrid outperforms cloud-only |
