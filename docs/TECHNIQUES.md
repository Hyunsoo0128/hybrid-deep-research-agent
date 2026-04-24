# RAG Techniques: Paper Analysis & Implementation Plan

Last updated: 2026-04-23 (Phase 3 complete + Hybrid Strategy Phase A–F — v1.1)

This document covers all 10 research techniques selected for this pipeline —
paper summaries, performance benchmarks, current implementation status, and
the phased implementation plan.

---

## Implementation Status Overview

| # | Technique | arxiv | Phase | Status | File |
|---|-----------|-------|-------|--------|------|
| 1 | CRAG | 2401.15884 | 1 | **Correct** | `search_worker.py` |
| 2 | Query Decomp + Reranker | 2507.00355 | 1 | **Correct** | `plan_generator.py`, `reranker.py` |
| 3 | RhinoInsight (VCM + EAM) | 2511.18743 | 1–2 | **Partial** | `checklist_node.py`, `evidence_auditor.py` |
| 4 | MASS-RAG | 2604.18509 | 2 | **Partial** | `search_worker.py` (3-agent), `writer.py` |
| 5 | AlignRAG | 2504.14858 | 2+E | **Partial + Spec RAG** | `critic.py` |
| 6 | DSAP | 2512.20660 | 1–2 | **Partial** | `utils/llm_json.py` |
| 7 | STRIDE | 2604.17405 | 3 | **Partial** | `plan_generator.py`, `supervisor.py` |
| 8 | CONSTRUCT | 2603.18014 | 3 | **Partial** | `quality_scorer.py` |
| 9 | PROClaim | 2603.28488 | — | **Future Work** | — |
| 10 | NaviRAG | 2604.12766 | — | **Future Work** | — |

**Correct**: Core algorithm matches paper, deviations are approximations only.
**Partial**: Implemented core idea; paper's full algorithm has unreproducible components (API-only, offline KB, fine-tuning) or deferred extensions.
**Future Work**: Consciously excluded — architectural mismatch with primary use case or cost/benefit judgment. See Phase 4 section.

---

## Removed Techniques

| Technique | arxiv | Reason |
|---|---|---|
| CURE | 2604.12046 | RL fine-tuning required — cannot apply to API-based systems |
| AutoSearch | 2604.17337 | RL fine-tuning required — cannot apply to API-based systems |
| Speculative RAG | 2407.08223 | Original value proposition (small drafter + large verifier) doesn't fit either usage mode: Bedrock users don't need it, Ollama users can't get the benefit without a large verifier. **Re-evaluation**: hybrid local/cloud routing makes the drafter(local)/verifier(cloud) split structurally possible. Privacy constraint strengthens this further — see [HYBRID_STRATEGY.md](HYBRID_STRATEGY.md). |

---

## Techniques Overview

### Status Legend
- **Correct** — implementation matches paper
- **Partial** — core idea present but key algorithm missing
- **Wrong** — name matches but implementation doesn't reflect paper
- **New** — not yet implemented

---

## Phase 1: Structural Foundation

---

### 1. CRAG — Corrective Retrieval Augmented Generation
**arxiv**: 2401.15884 | **File**: `src/nodes/search_worker.py`

#### Paper Summary
Evaluates retrieved documents before generation and takes corrective action
based on quality. Three-way branch: CORRECT (high quality) → refine internally,
INCORRECT (low quality) → fall back to web search, AMBIGUOUS → both paths.
The key algorithm is **Decompose-then-Recompose**: split documents into
sentence-level strips, filter irrelevant strips, reassemble into clean context.

#### Performance
- PopQA: +14.5% vs naive RAG
- PubHealth: +13.1% vs naive RAG

#### Current Status: **Correct** (Phase 1-1, 2026-04-23)
| Element | Paper | Implementation |
|---|---|---|
| 3-way classification | CORRECT / INCORRECT / AMBIGUOUS | ✅ CORRECT ≥ 0.5 / AMBIGUOUS 0.3–0.5 / INCORRECT < 0.3 |
| Evaluator | T5-large fine-tuned | LLM batch (float score per doc) — acceptable approximation |
| Decompose-then-Recompose | Strip-level decompose → filter → reassemble | ✅ A2 batch: 1 LLM call, strips ≥ 0.5 retained |
| AMBIGUOUS path | Internal refinement + web search | ✅ Extraction (refinement) + gap hint at uncertainty=0.5 |
| INCORRECT path | Web search fallback | ✅ D1: gap_detector delegation at uncertainty=1.0 |

#### Implementation Design (finalized 2026-04-23)

**A. Decompose-then-Recompose — A2 (batch with extraction)**
- CORRECT documents: one LLM call combining strip scoring + extraction
  - Input: full fetched content
  - Output: `{"strips": [...], "filtered_excerpt": "..."}` (strips ≥ 0.5 retained)
- AMBIGUOUS/INCORRECT documents: standard summary extraction, no DRC

**B. Strip splitting — B3 (regex, 2-sentence sliding window)**
```python
strips = re.split(r'(?<=[.!?])\s+', content)
# Group into 2-sentence units; drop strips shorter than 20 chars
```

**C. Query-level 3-way verdict — max(doc_scores) with thresholds**
- CORRECT   : max_score ≥ 0.5
- INCORRECT  : max_score < 0.3
- AMBIGUOUS  : 0.3 ≤ max_score < 0.5
- Evaluator prompt extended: returns `relevance_score` (float) per doc

**D. INCORRECT path — D1: gap_detector delegation**
- search_worker does NOT retry on INCORRECT
- Instead, returns `retrieval_quality` signal in state:
  ```python
  {"sub_query_id": str, "verdict": str, "max_doc_score": float, "strip_retention_ratio": float}
  ```
- gap_detector reads `state["retrieval_quality"]`; INCORRECT sub_queries are
  forced into the gap candidate list
- Rationale: same Tavily engine + keyword rewrite unlikely to improve results;
  gap_detector generates a genuinely different angle query

---

### 2. query_decomp — Question Decomposition for RAG
**arxiv**: 2507.00355 | **File**: `src/nodes/plan_generator.py`

#### Paper Summary
Decomposes complex multi-hop questions into sub-questions, retrieves in
parallel across all sub-questions + original query, then **reranks all
candidate documents using the original query** (not sub-questions).
The Reranker step is the paper's core contribution: decomposition improves
recall, reranking restores precision, the combination maximises both.

#### Performance
- MRR@10: +36.7% vs naive RAG
- F1: +11.6% vs naive RAG

#### Current Status: **Correct** (Phase 1-2, 2026-04-23)
| Element | Paper | Implementation |
|---|---|---|
| Sub-question decomposition | Dynamic number based on complexity | 5-dimension framework — enhancement, does not conflict |
| Original query preserved | Q = {q} ∪ Decompose(q) | sq0 always prepended by plan_generator ✅ |
| **Reranker** | **BGE-reranker-large, scored against original query** | **cross-encoder/ms-marco-MiniLM-L-6-v2 via fastembed** ✅ |

#### Implementation Design (finalized 2026-04-23)

**R1. Model: cross-encoder/ms-marco-MiniLM-L-6-v2 via fastembed**
- ONNX backend (no PyTorch); model ~80 MB, downloaded on first call
- Lazy singleton (`@lru_cache(maxsize=1)`); re-used across requests
- Fallback: confidence-based sort if cross-encoder import/run fails

**R2. Rerank basis: original_query scored against `title + excerpt`**
- Paper uses BGE-reranker-large; ms-marco-MiniLM is a lighter but well-tested approximation

**R3. Top-k by depth**
- fast: 10  /  normal: 20  /  deep: 40

**R4. URL-based dedup before reranking**
- Keep highest-confidence entry per URL
- Citations with no URL are preserved as-is

**R5. Integration: new `reranker_node` in graph**
- Position: fan-out join → `reranker` → `gap_detector`
- `reranker_node` writes to `state["reranked_citations"]` (plain list, overwrites)
- `gap_detector`: reads `reranked_citations` for coverage analysis
- `cross_validator`: reads `reranked_citations + gap_search_additions`
- `writer`: inherits effective set via `cross_validation_report.effective_citation_ids`

**Q = {q} ∪ Decompose(q)**
- `plan_generator` prepends `sq0 = {original_query, dimension="Original Query"}` when `query_decomp=True`
- sq0 participates in parallel fan-out like any other sub-query

---

### 3. RhinoInsight — VCM + EAM
**arxiv**: 2511.18743 | **Files**: `src/nodes/checklist_node.py` (new),
`src/nodes/evidence_auditor.py` (new)

#### Paper Summary
Addresses "context rot" in linear research pipelines through two control
modules. **VCM (Verification Checklist Module)**: generates a structured
checklist of sub-goals before research begins, tracks completion after each
search cycle, surfaces uncovered goals to gap detection. **EAM (Evidence
Audit Module)**: Stage 1 normalises retrieved results into a structured
evidence store (source / timestamp / confidence), aligns evidence to outline
nodes; Stage 2 ranks evidence per node and binds specific citations to
specific claims before writing. Uses Markovian workspace compression:
each step only carries (query + current sub-goal + compressed relevant
memory slice) to prevent context accumulation.

#### Performance
- GAIA: +10.6% (58.3% → 68.9%)
- HLE: 27.1% accuracy (exceeds GPT-5 at 26.3%)
- DeepConsult win rate: 68.51% (vs Gemini 61.27%)
- Ablation: no modules → 3.65, VCM only → 5.31, EAM only → 5.45, both → **6.82**

#### Current Status: **Partial** (Phase 2-4/2-5, 2026-04-23)
| Element | Paper | Implementation |
|---|---|---|
| VCM checklist generation | Before research begins | ✅ checklist_node after plan_generator |
| VCM status tracking | After each search cycle | ✅ evidence_auditor: claim_bindings-based (Stage 2a) or keyword-overlap fallback |
| VCM uncovered → gap_detector | Surface pending items | ✅ Phase 2-5: PENDING/PARTIAL → VCM hint axis in gap_detector |
| EAM Stage 1: normalize | source/timestamp/confidence | ✅ evidence_store: dedup + sort + verification_level |
| EAM Stage 2a: claim binding | outline alignment per node | ✅ mass_rag key_spans → claim_bindings per evidence item |
| EAM Stage 2b: misalignment | claim-level citation flags | ✅ critic_feedback.misaligned_claims → misalignment_flags per evidence item |
| Markovian compression | Context slicing per step | ❌ Not implemented (out of scope for API-based system) |

#### Implementation Design (finalized 2026-04-23)

**R1: VCM — separate `checklist_node` after `plan_generator`** (R1a)
- Converts each sub-query into a verifiable subgoal statement
- Position in graph: `generate_plan → checklist_node → plan_review`
- Flag: `rhinoinsight=False` → node is a no-op (returns empty checklist)

**R2: Checklist update — once in `evidence_auditor` after `cross_validator`** (R2a)
- Status: complete (≥2 evidence) / partial (1) / pending (0)
- Keyword-overlap heuristic (Phase 2 will replace with outline alignment)

**R3: Flat checklist structure** (R3a)
```python
[{"id": str, "subgoal": str, "sub_query_id": str,
  "status": "pending|partial|complete", "evidence_ids": list[str]}]
```

**R4: Separate `evidence_auditor` node after `cross_validator`** (R4b)
- Position: `cross_validator → evidence_auditor → write_draft`
- No extra LLM call: verification_level derived from `cross_validation_report`

**R5: Evidence store — Citation-based normalization + dedup**
- Dedup by URL (catches gap_search additions missed by reranker)
- Sort by confidence descending
- verification_level: corroborated / single_source / unverified

**R6: Plain overwrite for both `checklist` and `evidence_store`** (no `operator.add`)

**Writer update:**
- `write_draft` prefers `evidence_store` over SDP `effective_citation_ids` over raw citations

**Known deviations:**
- **VCM → gap_detector timing**: checklist is updated by evidence_auditor (after cross_validator).
  First gap_detector call runs before checklist status is available.
  VCM hint is only effective from the 2nd gap iteration onward (deep mode).
  fast/normal mode: 1 gap iteration only — VCM hint has no effect on first run.
- **Markovian compression**: requires per-step context slicing infrastructure not available
  in LangGraph's state model without significant redesign. Deferred indefinitely.
- **Checklist status init**: keyword-overlap heuristic on first pass; claim_bindings-based
  (Stage 2a) only when mass_rag enabled.

---

## Phase 2: Perception & Collection

---

### 4. MASS-RAG — Multi-Agent Synthesis RAG
**arxiv**: 2604.18509 | **File**: `src/nodes/search_worker.py`

#### Paper Summary
Replaces single-path document filtering with three parallel specialist agents
that process the same retrieved documents from different angles.
**Summarizer**: produces abstract query-relevant summaries.
**Extractor**: identifies verbatim supporting text spans (no paraphrasing).
**Reasoner**: performs cross-document inference to surface implicit connections.
A **Synthesis Agent** then reconciles the three outputs into a final unified
response. Particularly effective on tasks requiring complex multi-step reasoning.

#### Performance
- ARC-Challenge: +27.1% vs MAIN-RAG
- ASQA EM: +19.9% (39.2% → 47.51%)
- TriviaQA: +3.5%

#### Current Status: **Partial** (Phase 2-1, 2026-04-23)
| Element | Paper | Implementation |
|---|---|---|
| 3-agent parallel | Summarizer/Extractor/Reasoner on same doc pool | ✅ asyncio.gather (M6a) |
| Synthesis Agent | 4th agent combining all outputs | ✅ separate call after gather (M7a) |
| Synthesis = final answer | Synthesis IS the answer for the query | ⚠ PRIMARY SOURCE in writer prompt (Option A), not raw Synthesis output |
| Input pool | All retrieved docs for sub-query | ✅ CORRECT+AMBIGUOUS docs as pool (M1a) |
| Input: raw documents | Paper uses retrieval output directly | ⚠ Input is post-CRAG excerpts (CRAG+MASS-RAG stacking, D2) |
| Output schema | summary + spans + inferences | ✅ M3b: {summary, key_spans, inferences} |
| fast depth guard | Skip when depth=fast | ✅ depth != "fast" check |
| INCORRECT guard | Skip for irrelevant sub-queries | ✅ verdict in (CORRECT, AMBIGUOUS) |
| Confidence metadata | AMBIGUOUS docs labeled in prompt | ✅ conf_label={high/medium/low} (M4c') |
| DSAP guard | JSON retry on structured calls | ✅ llm_json on Extractor/Reasoner/Synthesis |

#### Known Deviations

**Deviation 1: Synthesis is PRIMARY SOURCE, not raw final answer**

Paper: `retrieval → 3-agent → Synthesis → final answer` (Synthesis = answer itself)

This system: `retrieval → 3-agent → Synthesis → writer (_ANALYSIS_WITH_MASSRAG_PROMPT)` — writer uses Synthesis as the primary source but still generates the final prose. This is unavoidable because:
- Paper targets single-query QA benchmarks; this system generates long-form multi-section reports
- sub-query Synthesis outputs need coherent integration into one report — writer handles this
- EAM Stage 2 binding requires a writer-generated draft as the base

Mitigation (Option A): `_ANALYSIS_WITH_MASSRAG_PROMPT` explicitly labels Synthesis as "PRIMARY SOURCE — authoritative answer". Evidence is restricted to "citation support only — do not introduce new claims". Inferences from Synthesis are preserved as mandatory claims. This makes the writer a presenter/formatter rather than a new-claim generator.

**Deviation 2: Input is post-CRAG excerpts**

Paper: MASS-RAG receives raw retrieved documents. This system: CRAG DRC runs first, then MASS-RAG gets the filtered excerpts. Intentional stacking (D2: coverage funnel CRAG → MASS-RAG). CRAG already dropped irrelevant strips; MASS-RAG gets cleaner input.

#### Design Decisions (finalized 2026-04-23)

**M1a: Per sub-query doc pool** — All CORRECT+AMBIGUOUS docs for a sub-query form one pool.
INCORRECT sub-queries are skipped entirely (CRAG already flagged them as noise).

**M3b: Structured output**
```python
{
    "summary": str,             # Summarizer: concise domain synthesis
    "key_spans": [              # Extractor: verbatim spans with provenance
        {"text": str, "source_citation_ids": [str], "type": "fact|definition|evidence|example"}
    ],
    "inferences": [             # Reasoner: cross-doc conclusions
        {"claim": str, "supporting_span_indices": [int]}
    ]
}
```

**M4: Token limits** — Summarizer=200, Extractor=400, Reasoner=300, Synthesis=500.
fast depth disabled entirely (cost vs. quality trade-off for quick queries).

**M4c': Confidence metadata in prompt** — Rather than filtering AMBIGUOUS docs out,
include them with conf_label (high/medium/low) so agents can weight appropriately.

**M5b: DSAP retry + degraded fallback** — Extractor/Reasoner/Synthesis all use `llm_json`
with `fallback` values. Synthesis fallback uses Summarizer output + Extractor key_spans.

**M6a: asyncio.gather** — Summarizer (plain text), Extractor, Reasoner run in parallel.

**M7a: Separate Synthesis** — Synthesis is a 4th call after gather, combining all outputs.
Total: 4 LLM calls per sub-query when mass_rag=True, normal/deep depth, non-INCORRECT verdict.

**Writer integration** — `_build_mass_rag_context()` in writer.py formats the synthesis
output as additional context for per-sub-query analysis sections.
State field `mass_rag_outputs: Annotated[list[dict], operator.add]` accumulates across workers.

---

### 5. AlignRAG
**arxiv**: 2504.14858 | **File**: `src/nodes/critic.py`

#### Paper Summary
Trains a dedicated **Critic Language Model (CLM)** via Contrastive Critique
Synthesis (CCS): pairs of expert vs weak responses teach the CLM to diagnose
three retrieval-reasoning misalignment types.
**Phase 1**: relevance assessment error (wrong span importance).
**Phase 2**: query-evidence mapping failure (wrong relationship identified).
**Phase 3**: evidence-integrated synthesis error (unsupported conclusion).
CLM outputs an edit signal; generator refines iteratively until CLM emits
`[Good]` token (dynamic termination).

#### Performance
- Average accuracy: 62.8%
- 8B CLM outperforms 72B general LLM by +2.2%
- OOD tasks: +12.1%

#### Current Status: **Partial + Spec RAG** (Phase E complete, 2026-04-23)
| Element | Paper | Implementation |
|---|---|---|
| Critic model | Fine-tuned LLaMA3.1-8B CLM via CCS+CFT | Phase E: Spec RAG 3-stage (local drafter + cloud verifier + local refiner) |
| Training | Contrastive Critique Synthesis | ❌ Not applicable — approximated by Spec RAG drafter/verifier separation |
| Phase 1: Relevance | Span importance misalignment | ✅ `_generate_suspect_claims` (Stage 1) + `_coherence_verify` (Stage 2) |
| Phase 2: Query-Evidence | Relationship mapping failure | ✅ same pipeline |
| Phase 3: Synthesis | Unsupported conclusion | ✅ **cloud verifier handles** — local recall=0.00 confirmed by Phase E-0 |
| Output format | [Good]/[Bad] token + edit signal | ✅ approx: structured JSON per misalignment |
| Dynamic termination | [Good] → exit | ✅ DR2: code-computed `passed` |
| Revision signal | Per-phase correction | ✅ DR1c: phase-grouped prompt in `revise()` |

**Phase E Spec RAG pipeline** (`HybridProvider` only):
```
Stage 1 (local):  _generate_suspect_claims()  →  [{claim_text, reason, confidence}]
Stage 2 (cloud):  _coherence_verify()          →  [{claim_text, phase, correction_hint}]
                  ← claim text only, no source docs (privacy boundary) →
Stage 3 (local):  _refine_corrections()        →  [{phase, claim, source_citation_ids, source_quote, correction_hint}]
```
Graceful degradation: single provider → existing single-prompt path (unchanged).

**H2 resolution**: Baseline (single local) recall=0.00 → Phase E Spec RAG recall=**0.67** on ALIGNRAG_DRAFT_WITH_ERRORS fixture (3 gold errors). Self-preference bias eliminated by drafter/verifier separation.

#### Extensions Beyond Paper
- **`source_citation_ids`**: explicit citation attribution per misalignment — enables precise
  revise targeting and EAM Stage 2 claim binding. Not present in paper's CLM output.
- **`correction_hint`**: structured fix suggestion per misalignment — bridges critic → reviser
  without requiring a separate CDA (Corrective Document Augmentation) step.
- **Code-computed `passed`** (DR2): avoids LLM output contradictions where the model
  simultaneously flags issues but returns `passed=true`.
- **Phase E**: Spec RAG drafter/verifier split approximates CCS training signal at inference time.

#### Known Deviations
1. **CLM fine-tuning**: Paper trains a dedicated 8B CLM via CCS+CFT.
   Not reproducible via API-only access. Phase E Spec RAG partially compensates by
   having a separate cloud verifier — eliminates self-preference bias structurally.
2. **[Good]/[Bad] tokens**: Replaced by DR2 deterministic code computation (more stable).
3. **Edit signal granularity**: Paper's CLM produces token-level signals; our
   `correction_hint` is sentence-level — sufficient for reviser guidance.

---

### 6. DSAP — Dual-State Architecture for Reliable LLM Agents
**arxiv**: 2512.20660 | **File**: `src/utils/llm_json.py` + `src/graph.py`

#### Paper Summary
Addresses LLM non-determinism through Guard functions and three-level recovery.
Each workflow step has a **Guard** `G(artifact, context) → {⊤, ⊥retry, ⊥fatal}`.
Level 1 — Context Refinement: retry with accumulated error feedback.
Level 2 — Informed Backtracking: on stagnation (θ=0.7 error similarity,
r_patience consecutive failures), cascade-invalidate dependent steps and
inject failure summary into the upstream step's context.
Level 3 — Human Escalation: on budget exhaustion or fatal guard failure.

#### Performance
- Reliability improvement: up to +66 pp (SWE-Bench)

#### Current Status: **Partial** (Phase 2-3, 2026-04-23)
| Element | Paper | Implementation |
|---|---|---|
| Level 1: Context Refinement | Error + schema feedback retry | ✅ `_RETRY_SYSTEM` + `_RETRY_USER` with error + schema_hint |
| Level 2: Stagnation detection | Similarity θ≥0.7 over error messages | ⚠ Exact match on normalized fingerprint (approximation) |
| Level 2: Strategy switch | Alternative approach on stagnation | ✅ `_LAST_RESORT_SYSTEM` + clean-slate schema-only prompt |
| Level 2: Cascade invalidation | Invalidate dependent downstream nodes | ❌ Not implemented |
| Level 2: Upstream injection | Feed failure context to upstream node's next call | ❌ Logged only (`error_sink`) |
| Level 3: Escalation | Workflow-level retry | ❌ Not implemented |
| Guard function verdict | `{⊤, ⊥retry, ⊥fatal}` | ⚠ Binary: success/retry only |

#### Known Deviations

**Level 1: Full implementation.**

**Level 2: Partial implementation.**
- Stagnation detection: exact-match fingerprint (normalized error type + message) instead of
  similarity threshold (θ=0.7). Position info (line/column/char) stripped before comparison
  to prevent false non-stagnation when the same structural error recurs at different positions.
- Strategy switch: `_LAST_RESORT_SYSTEM` strict prompt + clean-slate messages instead of
  an "alternative approach library". Rationale: discarding accumulated error context is itself
  the strategy change — if error-feedback loops aren't helping, adding more of the same makes
  it worse.
- Cascade invalidation: not implemented. Requires LangGraph-level node dependency graph +
  `Command(goto=...)` mid-execution re-trigger + checkpoint management + infinite-loop guards.
  This is a framework-level redesign, not a per-node change. Out of scope for Phase 2-3.
- Upstream injection: `error_sink` provides observability (which node stagnated, at which
  attempt, with which error) but does not inject failure context into the upstream node's next
  LLM call. Downstream nodes use independent fallbacks rather than receiving failure context.
  Rationale: all nodes already have well-defined fallbacks that achieve graceful degradation
  without upstream re-invocation. Cascade invalidation's marginal benefit is low given that
  each node's fallback already handles the worst case:
  - Synthesis stagnation → writer switches to evidence-driven prompt (already implemented)
  - Extractor/Reasoner stagnation → degraded synthesis (still produces summary)
  - critic stagnation → `passed=True` (already the fallback)
  - gap_detector stagnation → `gap_queries=[]` (skip gap search)

**Level 3: Not implemented.** Workflow-level retry is out of scope.

**Guard function verdict: Binary (⊤/⊥retry) instead of ternary (⊤/⊥retry/⊥fatal).**
Fatal errors (network, auth) propagate via exception and are not caught by `llm_json()`.
All JSON parsing errors are treated as retry-eligible. `⊥fatal` path not implemented.

#### Design Decisions (Phase 2-3)

**`_error_fingerprint()`**: strips position info (line/column/char) before comparing.
Without normalization, "line 3 col 5" vs "line 4 col 2" appear as different errors
even though they are the same structural failure — causing false non-stagnation detection.

**`dsap_errors` in retrieval_quality**: stagnation events from MASS-RAG agents are stored
alongside CRAG quality signals per sub-query (`caller_tag`, `error_type`, `action`).
Enables correlation analysis: "did sub-queries with DSAP stagnation produce lower-quality synthesis?"
Other nodes will add `caller_tag` in subsequent phases when their JSON calls become observability targets.

**Future work**: Cascade Invalidation deferred. If a future phase requires it, the entry
point is LangGraph's `Command(goto=node_name, update={...})` pattern, combined with an
explicit `dsap_retry_count` field in state to prevent infinite loops.

---

## Phase 3: Planning Redesign & Quality Scoring

---

### 7. STRIDE — Strategic Iterative Decision-Making
**arxiv**: 2604.17405 | **Files**: `src/nodes/plan_generator.py`, `src/nodes/supervisor.py`

#### Paper Summary
Solves premature entity grounding in iterative RAG. Three layers:
**Meta-Planner** generates an abstract strategy Sq based on entity *types*
(not specific entities), then converts to concrete plan Cq.
**Supervisor** manages execution state Ω = (resolved, pending, failed),
decides retrieve / rewrite / answer per sub-query, and runs independent
sub-queries in parallel.
**Extractor** pulls atomic facts from documents.
**Reasoner** synthesises facts into final answer.

#### Performance
- 2WikiMultihopQA: +7.0% EM vs DualRAG
- Token usage: −54 to −71%
- Inference time: −60 to −80%

#### Current Status: **Partial** (Phase 3-1, 2026-04-23)
| Element | Paper | Implementation |
|---|---|---|
| Meta-Planner (Sq) | Entity-agnostic abstract strategy template | ✅ `_generate_abstract_strategy()`: LLM generates Sq with [ENTITY] slots |
| Cq derivation | Sq → concrete plan with real entities | ✅ `_STRIDE_CQ_PROMPT`: guided generation using Sq |
| Supervisor: retrieve/rewrite/answer | Per-sub-query action from Ω state | ✅ `supervisor.py`: CRAG verdict → action assignment |
| Supervisor: rewrite injection | Reformulated question → re-search | ✅ rewrite → gap_detector STRIDE hint axis |
| Dependency graph (Ω) | Dependency-aware parallel scheduling | ❌ Not implemented (requires graph redesign) |
| Extractor | Atomic fact extraction per document | ❌ Role covered by MASS-RAG Extractor agent |
| Reasoner | Multi-hop fact synthesis | ❌ Role covered by MASS-RAG Reasoner agent |

#### Known Deviations
1. **Dependency graph**: Paper's Supervisor tracks Ω = (resolved, pending, failed) with
   explicit dependencies between sub-queries. Not implemented — requires LangGraph-level
   redesign with conditional fan-out per dependency resolution. All sub-queries run in
   parallel via Send API regardless of dependency.
2. **Extractor/Reasoner**: Paper has dedicated agents for these roles.
   Functionally covered by MASS-RAG's Extractor and Reasoner agents (Phase 2-1).
3. **query_decomp 5-dim taxonomy**: Subsumed by STRIDE Sq reasoning_steps.
   When stride=True, dimension tags come from Sq step types, not fixed 5-dim taxonomy.
   Reranker (Phase 1-2 core contribution) unchanged and orthogonal to STRIDE.

---

### 8. CONSTRUCT — Real-Time Trustworthiness Scoring
**arxiv**: 2603.18014 | **File**: `src/nodes/quality_scorer.py`

#### Paper Summary
Scores the trustworthiness of each field in LLM-generated structured output
using a Judge LLM. Works on black-box APIs without logprobs or fine-tuning.
Provides both output-level score and per-field score, enabling targeted
re-generation of only low-confidence fields.

#### Performance
- Outperforms Prometheus and Semantic Entropy on precision/recall across
  4 structured output benchmarks

#### Current Status: **Partial**

| Decision | Paper | This impl |
|----------|-------|-----------|
| C1 Applied to | Any structured JSON | MASS-RAG outputs (3-agent synthesis) |
| C2 Template count | 5 verifier templates | 2 (Document-level + Field-level) |
| C3 Score storage | Per-field in output | `mass_rag_outputs[i]["trust_scores"]` |
| C4b Writer use | Targeted regen | Hedging language hint in synthesis block |
| C4d Critic use | CLM re-diagnosis | Construct trust alerts in `_PROMPT` |
| C5a Threshold | Configurable | `TRUST_THRESHOLD = 0.5` |
| C6a Eval | Financial/PII benchmarks | e2e benchmark (existing) |

#### Implementation: C2b — 2-call simplified

```
score_mass_rag_output(entry, llm)
  → asyncio.gather(
      Template 1 (Document-level, 128 tokens): overall trustworthiness + explanation
      Template 2 (Field-level, 64 tokens):     summary / key_spans / inferences scores
    )
  → {document_score, per_field: {summary, key_spans, inferences}, untrustworthy_fields}
```

Applied per sub-query when `construct=True` in `feature_flags`.
Result stored in `mass_rag_outputs[i]["trust_scores"]`.

#### Downstream Effects

- **C4b Writer** (`writer.py`): `_build_synthesis_block()` appends trust score line per field.
  Fields in `untrustworthy_fields` get `[LOW TRUST — qualify with hedging language]` tag.
- **C4d Critic** (`critic.py`): `_build_construct_hint()` adds "CONSTRUCT Trust Alerts"
  block to `_PROMPT` for Phase 3 misalignment scrutiny.

#### Known Deviations

- Paper uses 5 verifier templates; this uses 2. Ablation shows ~3–6% accuracy drop
  per removed template. 2 templates ≈ 70% of full effect at 40% cost.
- Targeted field regeneration (paper C5b) deferred — implement if low-trust fields
  degrade benchmark score.
- `construct=False` by default; enable per-run via `feature_flags={"construct": True}`.

---

## Phase 4: Future Work

These techniques are consciously excluded from v1.0. Both have significant architectural
mismatches with the primary use case (web-search-centric general research reports).

---

### 9. PROClaim — Courtroom Multi-Agent Debate
**arxiv**: 2603.28488 | **Status**: Future Work

#### Paper Summary
Courtroom-style fact verification. Plaintiff and Defense agents argue for
and against each claim; a Judge panel delivers a verdict.
**Progressive RAG**: each debate round adds new evidence with novelty
score ≥ 0.20 (deduplication threshold). **Role-switching test**: swap
Plaintiff ↔ Defense and re-debate to detect position anchoring vs
evidence-based reasoning. Heterogeneous judge panel (3 different LLMs)
cancels individual bias.

#### Performance
- Standard MAD: 71.67% → PROClaim: 81.67% (+10.0 pp)
- P-RAG removal: −7.5 pp (largest single contributor)
- ~211K tokens per debate

#### Exclusion Rationale
- **Use case mismatch**: PROClaim specializes in adversarial claim verification for
  controversial/legal/political content. Primary use case here is general research reports
  where most claims are factual, not contested.
- **Verification coverage**: alignrag (3-phase misalignment) + CONSTRUCT (field-level
  trustworthiness) already cover the verification surface for this use case.
- **Cost**: ~211K tokens per debate. Running on every claim in a research report would
  multiply token cost by 5–10×, with marginal gain on non-controversial content.
- **Lightweight version loses core value**: A 2-agent + single judge simplification removes
  heterogeneous aggregation (bias cancellation) and weakens role-switching consistency.
  The paper's main contribution is the ensemble design — approximating it loses the gains.

#### If Implemented
Selective activation makes sense: trigger only when alignrag Phase 3 flags a claim with
high confidence AND the topic is known-controversial (politics, legal, medical).
Extension point: `feature_flags["proclaim"]` already reserved.

---

### 10. NaviRAG — Active Knowledge Navigation
**arxiv**: 2604.12766 | **File**: `src/nodes/local_search_worker.py` (new)

#### Paper Summary
Structures a document collection into a hierarchical knowledge tree offline.
At query time: Phase 1 (vector localisation) finds the relevant subtree,
Phase 2 (top-down navigation) descends the tree with absorb/expand
decisions at each node. Assembles multi-granularity context (vector +
summary + raw). LightRAG uses 5× more tokens for comparable tasks.

#### Performance
- NarrativeQA: +4.80% F1 vs Vanilla RAG
- Token usage: 5× less than LightRAG
- Query latency: 4.4× faster than GraphRAG

#### Exclusion Rationale
- **Architecture mismatch**: NaviRAG requires offline knowledge base construction (~50 min+).
  The primary path is live web search (Tavily) — offline pre-processing doesn't fit.
- **Local-file scope**: The paper's value proposition applies specifically to large local document
  collections. `local_search_worker.py` already exists as a basic extension point.
- **Low primary-path impact**: Web search + CRAG + MASS-RAG already handle coverage and depth
  for the main use case. NaviRAG would only improve the optional local-file path.

#### Extension Point
If local-file research becomes a primary use case, `local_search_worker.py` is the natural
integration target. NaviRAG's offline KB build would replace the current flat-file approach.
`feature_flags["navirag"]` can be reserved for this future activation.

**Re-evaluation under hybrid strategy**: NaviRAG + Privacy-Preserving Speculative RAG form
a coherent private enterprise research path — NaviRAG handles fully-local KB navigation,
Privacy Spec RAG handles local draft → cloud judgment → local refinement. Together they
enable a complete research pipeline where cloud never accesses raw documents or original
queries. See [HYBRID_STRATEGY.md — NaviRAG Re-inclusion Path](HYBRID_STRATEGY.md#navirag-re-inclusion-path).

---

## Revised Implementation Order

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design rationale,
technique interaction analysis, and 6 explicit design decisions (D1–D6).

```
Pre-work (Phase 0) ✓ DONE
  Removed: CURE, AutoSearch, speculative_rag, graph_builder.py
  Fixed: paper citations in state.py and docs

Phase 1 — Structural Foundation ✅
  1. crag          Decompose-then-Recompose + 3-way verdict + gap_detector delegation (D1)
  2. query_decomp  sq0 prepend + cross-encoder reranker (Q = {q} ∪ Decompose(q))
  3. RhinoInsight  VCM checklist_node + EAM Stage 1 (evidence normalization)
                   Note: EAM Stage 2 implemented in Phase 2 after MASS-RAG

Phase 2 — Perception & Verification ✅
  4. MASS-RAG      3-agent parallel (Summarizer/Extractor/Reasoner) + Synthesis per sub-query
                   EAM Stage 2a (claim_bindings) + Stage 2b (misalignment_flags)
  5. alignrag      3-phase misalignment diagnosis + DR1c/DR2 + structured misaligned_claims
  6. dsap          Level 1+2 recovery in llm_json (context refinement + stagnation detection)
  7. VCM→gap       PENDING/PARTIAL subgoals → independent hint axis in gap_detector (deep 2nd iter+)

Phase 3 — Planning Redesign & Quality Scoring ✅
  8. STRIDE        Meta-Planner Sq→Cq 2-step + Supervisor retrieve/rewrite/answer
                   STRIDE hints → 3rd axis in gap_detector; query_decomp remains as Cq grounding
  9. CONSTRUCT     C2b 2-call scorer (Document + Field-level) on MASS-RAG outputs
                   C4b writer hint + C4d critic alert; C5b targeted regen deferred

Phase 4 — Future Work (consciously excluded from v1.0)
  10. PROClaim     Excluded: adversarial claim debate; ~211K tokens/debate; mismatches general
                   research use case; alignrag+CONSTRUCT cover verification surface
  11. NaviRAG      Excluded: offline KB build required; primary path is web search;
                   local_search_worker.py is extension point if local-file becomes primary
```

---

## Why This Order

RhinoInsight is in Phase 1 because it addresses the pipeline's foundational problem:
error accumulation and context rot in the linear plan→search→write flow. Layering
MASS-RAG or STRIDE on top of a context-contaminated pipeline wastes their potential.

EAM Stage 2 is deliberately split into Phase 2 (not Phase 1) because D2 requires
MASS-RAG output as EAM Stage 2's input. Implementing EAM Stage 2 before MASS-RAG
would require a rewrite in Phase 2.

---

## Planned Enhancement: Hybrid Local/Cloud LLM Routing

**Status**: Not implemented — design complete, validated by E2E benchmark findings.

### Motivation

E2E benchmark (2026-04-23) measured qwen3:8b at ~481s per query in the full pipeline,
even with `/no_think` token added. This makes local-only deployment impractical for
interactive research sessions (target: <3 min/query).

However, three use cases still make local LLM integration valuable:

1. **Privacy**: The original research query and plan never leave the user's machine.
   Only evaluation intermediate outputs (scored excerpts, structured JSON) are sent to cloud.
2. **Rate limit distribution**: Cloud API rate limits are hit primarily by evaluation loops
   (CRAG batch scoring, CONSTRUCT 2-call scorer). Local LLM handles these high-frequency,
   low-latency-sensitive calls; cloud handles the few high-quality generation calls.
3. **Cost reduction**: Routing only generation nodes to cloud reduces estimated cost by ~75%
   ($0.75 → ~$0.19/query on phase1_2_3).

### Design: HybridProvider + Tier-Based Routing

```python
# src/providers/hybrid.py
class HybridProvider:
    """Exposes .cloud and .local for explicit routing."""
    def __init__(self, cloud: LLMProvider, local: LLMProvider):
        self.cloud = cloud
        self.local = local

    async def complete(self, messages, system="", **kwargs) -> str:
        return await self.cloud.complete(messages, system=system, **kwargs)

    async def embed(self, text: str) -> list[float]:
        return await self.local.embed(text)
```

```python
# src/graph.py — provider injection via partial()
if isinstance(llm, HybridProvider):
    cloud = llm.cloud    # generation nodes
    local = llm.local    # evaluation nodes
    tier_a = llm         # Speculative RAG nodes (access .cloud/.local internally)
else:
    cloud = local = tier_a = llm  # backward compatible

builder.add_node("writer",     partial(write_draft_node, llm=cloud))
builder.add_node("supervisor", partial(supervisor_node,  llm=local))
builder.add_node("critique",   partial(critique_node,    llm=tier_a))
```

**Key design choice**: `partial()` injection at graph build time, not runtime dispatch.
- Zero changes to existing node code — each node receives a pre-resolved provider.
- Tier A nodes (search_worker, critique) receive the full HybridProvider and access
  `.cloud`/`.local` internally for the Speculative RAG draft/verify/refine pattern.
- Graceful degradation: `hasattr(llm, 'local')` checks fall through for non-hybrid providers.

### Node Routing Table

| Node | Route | Reason |
|------|-------|--------|
| `plan_generator` | Cloud | STRIDE Sq→Cq requires strong reasoning |
| `writer` | Cloud | Report quality is user-facing; primary quality driver |
| `critic` | Cloud | AlignRAG diagnosis requires cross-claim coherence |
| `gap_detector` | Cloud | Strategic gap analysis + STRIDE hint generation |
| `search_worker` | Local | CRAG evaluation: batch scoring + DRC extraction |
| `quality_scorer` | Local | CONSTRUCT: 2-call trustworthiness scoring |
| `supervisor` | Local | retrieve/rewrite/answer verdict is classification |
| `cross_validator` | Local | Source comparison: structured judgment |
| `checklist_node` | Local | VCM subgoal generation from plan |
| `evidence_auditor` | Local | Claim binding + misalignment flagging |
| `reranker` | Local | ONNX cross-encoder (no LLM call) |

### Projected Performance (phase1_2_3)

| | Cloud-only (current) | Hybrid (estimated) |
|--|--|--|
| Cost/query | $0.75 | ~$0.19 (−75%) |
| Latency | 216s | ~120–150s (local eval: 4-5s/call) |
| Privacy | Query exposed | Query private |
| API rate risk | High (eval loops) | Low (eval → local) |

### Known Risks

1. **qwen3 thinking mode overhead**: `/no_think` token reduces but doesn't eliminate
   extended reasoning. Local evaluation calls using qwen3 must be monitored for latency
   regression. Mitigation: smaller qwen3:1.5b or exaone3.5:2.4b for evaluation-only calls.
2. **JSON reliability gap**: Local models produce more malformed JSON than cloud.
   DSAP Level 1+2 in `llm_json.py` is the mitigation layer — must be applied to all
   local-routed calls without exception.
3. **Score calibration drift**: CRAG verdict thresholds (0.3/0.5) were calibrated on Claude.
   Local model relevance scores may use different scale. Requires recalibration run before
   production routing.

### Extension Point

`feature_flags["hybrid_routing"]` reserved. Activation: set `LLM_PROVIDER=hybrid`,
`HYBRID_CLOUD_PROVIDER=bedrock|claude`, `HYBRID_LOCAL_MODEL=qwen3:8b|exaone3.5:7.8b`.
