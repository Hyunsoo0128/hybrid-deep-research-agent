# Architecture v2 — Technique Integration Design

This document records explicit design decisions.
Last updated: 2026-04-23 (Phase 3 complete — Phase 4 designated Future Work).

Phase 1/2/3 implementation deviations are noted inline with **[Phase N actual]** markers.

---

## Pipeline Stage Map

Each technique is assigned a primary stage and control target:

| Technique | Stage | Controls |
|---|---|---|
| query_decomp | Input | Query coverage (decompose + rerank) |
| STRIDE Meta-Planner | Plan | Abstract strategy Sq → concrete plan Cq |
| RhinoInsight VCM | Plan | Sub-goal tracking and progress |
| crag | Retrieval | Document quality filtering |
| EAM Stage 1 | Retrieval | Evidence normalization |
| MASS-RAG | Processing | Multi-angle evidence extraction |
| EAM Stage 2 | Pre-Write | Evidence ranking + citation binding |
| STRIDE Supervisor | Orchestration | Dependency-aware execution (retrieve/rewrite/answer) |
| alignrag | Verification | Reasoning-evidence consistency |
| CONSTRUCT | Verification | Structured output field confidence |
| PROClaim | Verification (selective) | Controversial claim debate |
| dsap | Cross-cutting | JSON output reliability |
| NaviRAG | Retrieval (local only) | Hierarchical local document navigation |

---

## Revised Pipeline Flow

### Target (full system, all phases)
```
Query
  │
  ├─ [STRIDE Meta-Planner]  Abstract strategy Sq → Cq              (Phase 3)
  │    └─ [query_decomp]    Grounds Cq: Q = {q} ∪ Decompose(q)     (Phase 1 ✅)
  │
  ├─ [RhinoInsight VCM]     Generates verifiable sub-goal checklist (Phase 1 ✅)
  │
  └─ [STRIDE Supervisor]    Dependency-aware execution              (Phase 3)
       │
       └─ Per sub-query (parallel fan-out via Send API):
            ├─ Tavily search
            └─ [crag]  3-way filter → CORRECT / AMBIGUOUS / INCORRECT  (Phase 1 ✅)
                  └─ CORRECT:   Decompose-then-Recompose → clean strips
                  └─ AMBIGUOUS: extraction + gap hint (uncertainty=0.5)
                  └─ INCORRECT: minimal summary + gap hint (uncertainty=1.0)
  │
  ├─ [Reranker]             Dedup + cross-encoder top-k vs original query  (Phase 1 ✅)
  │
  ├─ [gap_detector]         Coverage analysis; INCORRECT/AMBIGUOUS hints → gap queries
  ├─ [gap_search]           (conditional) Targeted supplementary retrieval
  │
  ├─ [cross_validator]      Corroboration / contradiction / SDP pruning
  │
  ├─ [EAM Stage 1]          Normalize: dedup + sort + verification_level  (Phase 1 ✅)
  │                         Checklist status update (complete/partial/pending)
  │                         [Phase 2: + MASS-RAG 3-agent input; + EAM Stage 2 binding]
  │
  ├─ [writer]               Draft; prefers evidence_store → SDP ids → raw  (Phase 1 ✅)
  │
  ├─ [alignrag critic]      3-phase misalignment check                     (Phase 2)
  │    └─ [dsap]            JSON guard (cross-cutting, Phase 1 partial ✅)
  │
  ├─ [CONSTRUCT]            Field-level trustworthiness scoring             (Phase 3)
  ├─ [PROClaim]             (selective) Courtroom debate                   (Phase 4)
  └─ [revise → finalize]
```

### Implemented (Phase 3 actual graph)
```
generate_plan → checklist_node → [interrupt: plan_review]
  → search_orchestrator
      → [Send API parallel fan-out] search_worker × N
      │    inside per sub-query:
      │      crag (3-way verdict) → MASS-RAG 3-agent + Synthesis
      │      if construct=True → CONSTRUCT score_mass_rag_output() → trust_scores in entry
      │    + local_search_worker × N
      → reranker  (dedup + cross-encoder top-k)
      → supervisor   (STRIDE: retrieve/rewrite/answer per sub-query)
      → gap_detector (CRAG hints + VCM hints* + STRIDE hints*)
           *deep mode 2nd iter only (VCM); STRIDE: rewrite decisions from supervisor
           → [has gaps] gap_search ─┐
           → [no gaps]              ┘→ cross_validator
                                        → evidence_auditor  (EAM Stage 1 + Stage 2a)
                                        → write_draft  (synthesis_block includes CONSTRUCT trust hints)
                                        → critique  (alignrag 3-phase + CONSTRUCT trust alerts)
                                        → evidence_stage2   (EAM Stage 2b: misalignment flags)
                                          → [passed]  finalize
                                          → [revise]  revise → critique → evidence_stage2 (loop)
```

---

## Design Decisions

### D1. query_decomp vs STRIDE — Role Separation

**Decision**: STRIDE handles abstract planning; query_decomp handles concrete grounding.

- STRIDE Meta-Planner generates entity-type-agnostic strategy Sq
  (e.g., "need: definition + current state + comparison")
- STRIDE converts Sq → concrete plan Cq with actual entities
- query_decomp's 5-dimension framework maps onto Cq grounding:
  each sub-query gets a dimension tag (Definition/Background, Current State, etc.)
- The Reranker step (rerank all retrieved docs against original query) runs
  after all sub-query searches complete, regardless of whether STRIDE is on

**When STRIDE is off**: query_decomp operates independently as before (5-dimension decomp → parallel search → rerank).

**When both are on**: STRIDE produces Cq, query_decomp applies dimension tagging to Cq sub-queries. They do not conflict.

**[Phase 1 actual]**: STRIDE not yet implemented. query_decomp operates independently:
- plan_generator prepends `sq0 = {original_query, dimension="Original Query"}` — Q = {q} ∪ Decompose(q)
- `reranker_node` runs after parallel fan-out join: cross-encoder top-k vs original query
- `dimension` field dropped in SubQuery dataclass (pre-existing limitation); MASS-RAG fallback unaffected since mass_rag=False

**[Phase 3 actual]**: Both STRIDE and query_decomp implemented:
- STRIDE off (`stride=False`): query_decomp operates independently as Phase 1 actual above.
- STRIDE on (`stride=True`): 2-step flow — `_generate_abstract_strategy()` → entity-agnostic Sq
  with [ENTITY] slots → `_STRIDE_CQ_PROMPT` derives concrete Cq guided by Sq strategy.
  sq0 (original query) still prepended as first sub-query (Reranker grounding, orthogonal to STRIDE).
- query_decomp dimension tags (Definition/Background etc.) remain in use as sub-query labels;
  they apply to STRIDE-generated Cq sub-queries as well as the direct decomp path.
- **Known Deviation**: Dependency graph Ω not implemented. Per-node dependency tracking from paper
  replaced by per sub-query Supervisor verdict (retrieve/rewrite/answer). Extractor/Reasoner
  roles from STRIDE paper are covered by MASS-RAG agents.

---

### D2. EAM vs MASS-RAG — Ordering

**Decision**: EAM Stage 1 runs BEFORE MASS-RAG. MASS-RAG output feeds EAM Stage 2.

```
raw search results
  → crag filter
  → [reranker]  top-k selection
  → [cross_validator]
  → EAM Stage 1  (normalize: dedup + sort + verification_level)     ← Phase 1 actual
  → [Phase 2: MASS-RAG] (Summarizer/Extractor/Reasoner)
  → [Phase 2: EAM Stage 2] (rank per outline section, bind citations)
```

**[Phase 1 actual deviation]**: EAM Stage 1 runs after cross_validator (R4b decision),
not inside the per-sub-query loop as originally sketched. Rationale: cross_validator
produces corroboration groups → evidence_auditor derives verification_level from them
without an extra LLM call. The ordering EAM-after-CV is semantically correct and cheaper.

**[Phase 2 actual]**: EAM Stage 2 split into 2a and 2b:
- **Stage 2a** (in `audit_evidence`, same node as Stage 1): MASS-RAG `key_spans.source_citation_ids`
  → `claim_bindings` per evidence item. Checklist coverage uses claim_bindings when available.
- **Stage 2b** (`evidence_stage2` node, inserted after `critique`): `critic_feedback.misaligned_claims`
  → `misalignment_flags` per evidence item. Overwrites each iteration (no accumulation).
  Graph: `critique → evidence_stage2 → should_revise`. Revise loop: `revise → critique → evidence_stage2`.

**Deviation from original D2 sketch**: MASS-RAG does not feed into a separate EAM Stage 2 node
between search and write. Instead MASS-RAG runs inside `search_worker` (per sub-query),
and its synthesis feeds `writer` directly as PRIMARY SOURCE. EAM Stage 2a binds key_spans
to evidence_store concurrently in `audit_evidence`. The "MASS-RAG sees EAM-normalized inputs"
invariant is maintained (MASS-RAG reads `reranked_citations` which Stage 1 already normalized).

---

### D3. Confidence Score Authority

Three separate systems produce confidence scores. Their authority is scoped:

| Score | Source | Scope | Meaning |
|---|---|---|---|
| `retrieval_confidence` | crag evaluator | Document level | Is this document relevant to the sub-query? |
| `evidence_confidence` | EAM Stage 1 | Evidence level | Normalized trust (source authority × recency × crag score) |
| `field_confidence` | CONSTRUCT | Output field level | Is this JSON field / report claim trustworthy? |

**Cascade rule**: EAM takes CRAG's `retrieval_confidence` as its initial value, then applies source authority and recency multipliers. This prevents double-scoring.

**Final report confidence**: `min(evidence_confidence, field_confidence)` — the lower of the two is reported as the authoritative trust score for each claim.

**[Phase 1 actual]**: EAM Stage 1 implemented without multipliers — `verification_level`
(corroborated/single_source/unverified) is derived from `cross_validation_report` at zero
extra LLM cost. `evidence_confidence` = raw `citation.confidence` (no authority/recency
multiplier yet).

**[Phase 2 actual]**: EAM Stage 2a adds `claim_bindings` per evidence item (MASS-RAG spans).
EAM Stage 2b adds `misalignment_flags` per evidence item (alignrag critic output).
Full confidence cascade (CRAG × authority × recency multiplier) deferred to Phase 3 alongside
CONSTRUCT field scoring.

---

### D4. STRIDE Supervisor vs VCM — Synchronization

**Decision**: Supervisor plan mutations trigger VCM checklist recompute.

When the STRIDE Supervisor decides to `rewrite` a sub-query or adds a new retrieval step:
1. The updated sub-query is written back to the plan in state
2. VCM re-evaluates the checklist against the updated plan
3. Newly added sub-goals are appended as unchecked items
4. Previously completed items are preserved

**Implementation note**: VCM does not own the plan; it observes it.
The Supervisor is the single authority on plan mutations.
VCM's role is tracking, not controlling.

**[Phase 1 actual]**: STRIDE Supervisor not yet implemented (Phase 3).
- VCM checklist created once (checklist_node after plan_generator) and updated once
  (evidence_auditor after cross_validator) — no mid-session recompute hook yet.

**[Phase 2 actual]**: VCM pending/partial subgoals now wired to gap_detector (Phase 2-5).
Two independent hint axes in gap_detector prompt: CRAG signals (retrieval quality) +
VCM signals (subgoal completion). LLM synthesizes gap queries from both axes.
**Known timing limitation**: checklist is updated by evidence_auditor (after cross_validator).
First gap_detector call runs before checklist status is available → VCM hint only effective
from 2nd gap iteration onward (deep mode). fast/normal: VCM hint has no effect on 1st run.

---

### D5. Verification Chain

**Default (always on)**:
- `alignrag`: 3-phase misalignment check on every draft
- `CONSTRUCT`: field confidence scoring on all JSON-producing nodes

**Selective (on when triggered)**:
- `PROClaim`: activated only when a claim is explicitly flagged as controversial
  - Trigger: alignrag Phase 3 returns a claim with `controversial: true`
  - Or: user explicitly enables `proclaim` flag
  - Cost justification: ~211K tokens per debate — not viable for every claim

**Rationale**: alignrag + CONSTRUCT cover the full verification surface.
PROClaim adds adversarial depth but overlaps with alignrag on non-controversial content.
Running both always-on doubles verification cost without proportional gain.

**[Phase 2 actual]**: alignrag fully implemented (Phase 2-2):
- `critic.py` `_PROMPT`: 3-phase diagnosis (Phase 1 relevance / Phase 2 mapping / Phase 3 synthesis)
- Schema: unified `misaligned_claims[{phase, claim, source_citation_ids, source_quote, correction_hint}]`
- DR2: `passed` computed in code from misalignment counts (no LLM self-report)
- DR1c: phase-grouped `_REVISE_PROMPT` (correction_hint per misalignment)
- Dynamic termination: `should_revise()` routes to finalize when passed=True
- Known deviation: CLM fine-tuning not possible via API (ceiling ~vanilla+2-5%)

**[Phase 3 actual]**: CONSTRUCT implemented (C1b+C2b+C3a+C4b+C4d):
- Applied to MASS-RAG outputs only (C1b), not all JSON-producing nodes (D6 Phase 3a target)
- 2-call simplified (C2b): Document-level + Field-level verifiers via asyncio.gather
- trust_scores stored in `mass_rag_outputs[i]["trust_scores"]` (C3a)
- C4b: writer's `_build_synthesis_block()` annotates low-trust fields with hedging instruction
- C4d: `_build_construct_hint()` injects CONSTRUCT Trust Alerts into critic `_PROMPT`
- `construct=False` default; enabled via `feature_flags={"construct": True}`
- Known deviation: 5-template paper → 2-template impl (~70% effect at 40% cost).
  Targeted regeneration (C5b) and paper benchmark reproduction (C6a) deferred.

dsap JSON guard: Level 1+2 implemented (`llm_json` utility, cross-cutting).

---

### D6. CONSTRUCT Scope

**Phase 3a (implemented)**:
- Applied to MASS-RAG synthesis outputs (C1b), not all JSON-producing nodes
- Document-level + Field-level scores per sub-query entry
- `untrustworthy_fields` (score < 0.5) → writer hedging hint (C4b) + critic scrutiny (C4d)
- Targeted regeneration of low-trust fields (paper C5b) deferred — implement if benchmark shows regression

**Phase 3b (deferred)**:
- Report section-level confidence (treat each section as a "field")
- Would require adapting the method from structured JSON to free-form text
- Not from the paper directly — would be a novel extension; excluded from v1.0 scope

---

## Synergy Map

### Clear synergies

| Combination | Effect |
|---|---|
| query_decomp → crag → MASS-RAG | Coverage → precision → depth funnel (each stage improves input quality for next) |
| STRIDE + VCM | Structure (STRIDE) + tracking (VCM): naturally complementary at planning stage |
| EAM + alignrag + CONSTRUCT | Normalize → consistency check → quantify: 3-layer verification chain |
| dsap + CONSTRUCT | dsap = structural validity of JSON; CONSTRUCT = content validity. Same output, different layers. |

### Interactions requiring care

| Issue | Resolution |
|---|---|
| crag confidence + EAM confidence double-scoring | EAM takes crag score as initial value (cascade, not independent) |
| MASS-RAG synthesis before EAM outline-alignment | Fixed by D2: EAM Stage 1 precedes MASS-RAG |
| alignrag + PROClaim functional overlap | Fixed by D5: PROClaim is selective, not default |
| STRIDE Supervisor dynamic rewrite vs VCM tracking | Fixed by D4: Supervisor mutates plan, VCM observes |

### Structural risks

| Risk | Mitigation |
|---|---|
| MASS-RAG × query_decomp cost amplification (5 sub-queries × 3 agents = 15 LLM calls) | Measure per-query cost in Phase 2 benchmark; add budget cap option |
| CONSTRUCT scope creep into free-form report | Phase 3a applies to JSON nodes only; report-level scoring deferred to Phase 3b as experimental |
| PROClaim + NaviRAG integration mismatch | PROClaim: selective activation. NaviRAG: parallel path (local files only), not integrated into main web search flow |

---

## Updated Implementation Order

Based on the above decisions, Phase ordering remains unchanged but with these additions:

**Phase 1** ✅:
- crag: Decompose-then-Recompose + 3-way verdict + AMBIGUOUS/INCORRECT gap hints
- query_decomp: sq0 prepend + cross-encoder reranker (Q = {q} ∪ Decompose(q))
- RhinoInsight: VCM (checklist_node) + EAM Stage 1 (evidence normalization)
- dsap: Level 1 context refinement

**Phase 2** ✅:
- MASS-RAG: 3-agent parallel (Summarizer/Extractor/Reasoner) + Synthesis per sub-query
  - Writer: Synthesis as PRIMARY SOURCE via `_ANALYSIS_WITH_MASSRAG_PROMPT`
  - EAM Stage 2a: MASS-RAG key_spans → claim_bindings in evidence_store
- dsap: Level 2 stagnation detection + last-resort strategy (error_sink, caller_tag)
- alignrag: 3-phase diagnosis + DR1c/DR2 + structured misaligned_claims schema
  - EAM Stage 2b: misalignment_flags in evidence_store (post-critique node)
- VCM → gap_detector: PENDING/PARTIAL hints as independent axis (deep mode 2nd iter+)

**Phase 3** ✅:
- STRIDE: Meta-Planner (Sq→Cq 2-step) + Supervisor (retrieve/rewrite/answer per sub-query)
  - query_decomp remains as grounding mechanism for Cq (per D1)
  - STRIDE hints → 3rd axis in gap_detector (alongside CRAG and VCM)
  - Known Deviation: dependency graph Ω not implemented; per-subquery Supervisor verdict used instead
- CONSTRUCT: C2b 2-call scorer (Document + Field-level) on MASS-RAG outputs (per D6 Phase 3a)
  - C4b writer hint + C4d critic alert; targeted regeneration (C5b) deferred
  - Known Deviation: 5-template → 2-template (~70% effect at 40% cost)

**Phase 4** — Future Work (not implemented):
- PROClaim (arxiv:2603.28488): Courtroom multi-agent debate for controversial claims.
  Excluded: primary use case is general research reports, not adversarial claim verification.
  alignrag + CONSTRUCT cover the verification surface for this use case.
  Cost: ~211K tokens per debate — not viable for general use.
- NaviRAG (arxiv:2604.12766): Hierarchical knowledge tree navigation.
  Excluded: requires offline KB build (~50 min); web-search-centric pipeline is primary path.
  local_search_worker exists as extension point if local-file research becomes primary use case.

---

## State Schema — Implemented and Planned

```python
# ── Phase 1 — implemented ────────────────────────────────────────────────
retrieval_quality: Annotated[list[dict], operator.add]
# per sub-query: {sub_query_id, verdict, max_doc_score, strip_retention_ratio, dsap_errors}

reranked_citations: list[dict]   # plain overwrite; top-k after cross-encoder
# produced by reranker_node; downstream nodes prefer this over raw citations

checklist: list[dict]            # plain overwrite; one per session
# [{id, subgoal, sub_query_id, status: pending|partial|complete, evidence_ids}]

evidence_store: list[dict]       # plain overwrite; writer prefers this
# Phase 1: {id, url, title, excerpt, confidence, trust_level, crawled_at, verification_level}
# Phase 2: + claim_bindings: [{sub_query_id, text, type}]   (Stage 2a, from MASS-RAG key_spans)
#          + misalignment_flags: [{phase, claim, correction_hint}]  (Stage 2b, from alignrag)

# ── Phase 2 — implemented ─────────────────────────────────────────────────
mass_rag_outputs: Annotated[list[dict], operator.add]
# per sub-query (CORRECT/AMBIGUOUS, non-fast):
#   {sub_query_id, question, summary, key_spans, inferences}
#   Phase 3: + trust_scores: {document_score, per_field: {summary, key_spans, inferences},
#                              untrustworthy_fields}  (CONSTRUCT, when construct=True)
# key_spans: [{text, source_citation_ids, type}]
# inferences: [{claim, supporting_span_indices}]

# critic_feedback: dict  (already in Phase 1 schema — schema extended in Phase 2)
# Phase 2 schema: {passed (code-computed), has_logic_errors (derived),
#   uncited_claims, unanswered_sub_queries, suggestions,
#   misaligned_claims: [{phase, claim, source_citation_ids, source_quote, correction_hint}]}

# ── Phase 3 — implemented ─────────────────────────────────────────────────
supervisor_decisions: list[dict]
# [{sub_query_id, question, action: retrieve|rewrite|answer, reformulated_question?,
#   verdict, max_doc_score}]
# produced by supervisor_node; gap_detector reads rewrite decisions as STRIDE hint

# Note: CONSTRUCT trust_scores stored inline in mass_rag_outputs (see above)
# No separate top-level state field — trust data travels with the entry it scores.

# ── Future Work (Phase 4) — not implemented ───────────────────────────────
# controversial_claims: list[str]  # would be needed for PROClaim selective activation
```

### Note on reducer semantics
- `citations`, `retrieval_quality`, `error_log`: `Annotated[list, operator.add]` — append only
- All other new fields: **plain overwrite** (no operator.add) — one authoritative value per session
- Do NOT use operator.add for any new Phase 2+ fields unless explicitly accumulating from parallel workers

---

## Planned Enhancement: Hybrid Local/Cloud LLM Provider

**Status**: Design complete. Not implemented. Informed by E2E benchmark (2026-04-23).

### Problem

qwen3:8b measured at ~481s/query in the full pipeline (local-only). Primary bottleneck:
evaluation loops (CRAG batch scoring: 6–8 calls, CONSTRUCT 2-call scorer) use the same
high-quality cloud model as report generation, but don't need it.

### Architecture: RoutedLLM Wrapper

```
LLM_PROVIDER=hybrid
       │
       ▼
RoutedLLM(cloud=BedrockProvider, local=OllamaProvider)
       │
       ├── node_hint in CLOUD_NODES  →  BedrockProvider.complete()
       └── node_hint in LOCAL_NODES  →  OllamaProvider.complete()
                                         OllamaProvider.embed()  (always local)
```

The wrapper implements the same `LLMProvider` Protocol (`complete`, `stream`, `embed`).
It is a drop-in replacement — `src/graph.py` passes a single provider to all nodes.
Nodes receive `node_hint` as a string argument to `complete()`; RoutedLLM uses it for
dispatch. No node code changes required.

### Routing: Generation vs Evaluation split

```
Cloud (generation — few, high-quality calls):
  plan_generator  → STRIDE Sq→Cq reasoning
  writer          → report generation (primary user-facing output)
  critic          → AlignRAG 3-phase diagnosis
  gap_detector    → strategic gap query generation

Local (evaluation — many, classification calls):
  search_worker   → CRAG batch scoring + DRC extraction
  quality_scorer  → CONSTRUCT trustworthiness 2-call scorer
  supervisor      → retrieve/rewrite/answer classification
  cross_validator → source comparison judgment
  checklist_node  → VCM subgoal generation
  evidence_auditor→ claim binding + misalignment flagging
```

### Provider Layer Impact

No changes to existing provider files. `RoutedLLM` is a new `src/providers/hybrid.py`:

```python
class HybridProvider:
    CLOUD_NODES = {"plan_generator", "writer", "critic", "gap_detector"}

    def __init__(self, cloud: LLMProvider, local: LLMProvider):
        self._cloud = cloud
        self._local = local

    async def complete(self, messages, system="", node_hint="", **kwargs) -> str:
        p = self._cloud if node_hint in self.CLOUD_NODES else self._local
        return await p.complete(messages, system=system, **kwargs)

    async def embed(self, text: str) -> list[float]:
        return await self._local.embed(text)
```

`src/providers/__init__.py` factory: `LLM_PROVIDER=hybrid` → `HybridProvider(cloud, local)`.

### Projected Impact

| Metric | Cloud-only (phase1_2_3) | Hybrid (estimated) |
|--------|------------------------|-------------------|
| Cost/query | $0.75 | ~$0.19 (−75%) |
| Latency | 216s | ~120–150s |
| Privacy | query exposed to cloud | original query stays local |
| API rate exposure | high (eval loops) | low (eval → local) |

### Known Risks

1. **qwen3 thinking overhead on eval nodes**: Even with `/no_think`, complex prompts
   may trigger partial thinking. Use qwen3:1.5b or exaone3.5:2.4b for eval nodes.
2. **CRAG threshold calibration**: 0.3/0.5 thresholds calibrated on Claude. Local model
   may score with different distribution. Requires a recalibration run on 5 queries
   before production routing.
3. **JSON reliability**: Local models produce more malformed JSON. DSAP Level 1+2
   (`llm_json.py`) must be applied to all local-routed nodes — already the case in
   current architecture since all structured outputs go through `call_llm_json()`.
