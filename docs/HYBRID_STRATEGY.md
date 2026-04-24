# Hybrid LLM Strategy — Privacy-Preserving Architecture

> Status: **Phase A–F complete (2026-04-23)**. Phase G (AgentCore) pending.
> Informed by E2E benchmark findings (eval/results/bedrock_full_v1.json).
> See also: [ARCHITECTURE.md](ARCHITECTURE.md#planned-enhancement-hybrid-localcloud-llm-provider), [OBSERVATIONS.md](OBSERVATIONS.md)
>
> **Implementation summary**:
> - Phase A: `HybridProvider` wrapper (`src/providers/hybrid.py`)
> - Phase B-1: `build_graph()` tier routing (local/cloud/tier_a)
> - Phase B-2: CRAG AMBIGUOUS re-adjudication with cloud (`search_worker.py`)
> - Phase C: `plan_generator` 2-stage — local profile → cloud plan
> - Phase D: MASS-RAG Spec RAG cycle — local drafters → cloud verifier → local refiner (M6a/M7a/M8)
> - Phase E-0: qwen3:8b suspect_claims pre-test → phase3 recall=0.00 → cloud verifier required
> - Phase E: Critic Spec RAG — local Stage1 → cloud Stage2 → local Stage3; H2 resolved (recall 0→0.67)
> - Phase F: writer privacy boundary — local file excerpt sanitization + `privacy_mode` config guard
> - Phase G: `AgentCoreProvider` drop-in for cloud nodes + `build_graph(agentcore_arns=...)` swap

---

## Full Pipeline: 5-Layer Architecture

```
User query (local only)
      │
      │  Original query → extract dimension labels only (local)
      ▼
╔══════════════════════════════════════════════════════════════╗
║  LAYER 1 — PLANNING                                          ║
║                                                              ║
║  plan_generator Stage 1  ▣ LOCAL   query_decomp dimension    ║
║         │  decomposition                                     ║
║         │  pass dimension labels only (original query not    ║
║         │  exposed)                                          ║
║  plan_generator Stage 2  ☁ CLOUD   STRIDE Sq→Cq refinement  ║
║         │                                                    ║
║  checklist_node          ▣ LOCAL   RhinoInsight VCM          ║
║         │                                                    ║
║  [user approval interrupt]                                   ║
╚══════════════════════════════════╤═══════════════════════════╝
                                   │ N sub_queries
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
╔══════════════════════════════════════════════════════════════╗
║  LAYER 2 — RETRIEVAL + EVALUATION  (almost entirely local)   ║
║                                                              ║
║  search_worker × N                                           ║
║    CRAG scoring         ▣ LOCAL   document relevance         ║
║                                   judgment                   ║
║    CRAG AMBIGUOUS       ☁ CLOUD   boundary cases             ║
║    re-adjudication                (0.3–0.5) only             ║
║    MASS-RAG Summarizer  ▣ LOCAL   Tier A: draft              ║
║    MASS-RAG Extractor   ▣ LOCAL   Tier A: draft              ║
║    MASS-RAG Reasoner    ▣ LOCAL   Tier A: draft              ║
║                                                              ║
║  reranker               ▣ LOCAL   cross-encoder (ONNX)       ║
║  supervisor             ▣ LOCAL   STRIDE Supervisor          ║
║  quality_scorer         ▣ LOCAL   CONSTRUCT confidence score ║
╚══════════════════════════════════╤═══════════════════════════╝
                                   │ 3-agent draft text only
╔══════════════════════════════════╧═══════════════════════════╗
║  LAYER 3 — VERIFICATION  (cloud — receives draft text only)  ║
║                                                              ║
║  MASS-RAG Verifier      ☁ CLOUD   Tier A: contradiction      ║
║                                   detection, best element    ║
║                                   selection                  ║
╚══════════════════════════════════╤═══════════════════════════╝
                                   │ judgment + instructions
╔══════════════════════════════════╧═══════════════════════════╗
║  LAYER 4 — SYNTHESIS  (local — re-references original docs)  ║
║                                                              ║
║  MASS-RAG Refiner       ▣ LOCAL   Tier A: refine             ║
║  gap_detector           ☁ CLOUD   gap inference from         ║
║                                   synthesis output           ║
║  cross_validator        ▣ LOCAL   cross-source validation    ║
║  evidence_auditor       ▣ LOCAL   RhinoInsight EAM Stage1+2  ║
╚══════════════════════════════════╤═══════════════════════════╝
                                   │ web excerpts (public) +
                                   │ MASS-RAG synthesis
                                   │ (abstracted)
                                   │ ※ raw local file text NOT
                                   │   transmitted
╔══════════════════════════════════╧═══════════════════════════╗
║  LAYER 5 — GENERATION + QUALITY CHECK                        ║
║                                                              ║
║  writer                 ☁ CLOUD   report generation          ║
║         │  report draft                                      ║
║  critic suspect         ▣ LOCAL   Tier A: suspect_claims     ║
║         │  claim text only (citation content not exposed)    ║
║  critic verify          ☁ CLOUD   Tier A: logical            ║
║         │                         consistency check          ║
║         │  judgment only                                     ║
║  critic refine          ▣ LOCAL   Tier A: correction hints   ║
║  revise                 ▣ LOCAL   DSAP JSON guard applied    ║
╚══════════════════════════════════════════════════════════════╝
```

**Legend**: ▣ LOCAL (qwen3:8b)  ☁ CLOUD (Claude)  Tier A = draft→verify→refine pattern

---

## Complete Node Routing Table

```
Node                        Route           Technique                 Cloud input
─────────────────────────────────────────────────────────────────────────────────────
plan_generator Stage 1      ▣ LOCAL         query_decomp              —
plan_generator Stage 2      ☁ CLOUD         STRIDE Sq→Cq              dimension labels
checklist_node              ▣ LOCAL         RhinoInsight VCM          —

search_worker (×N parallel)
  CRAG scoring              ▣ LOCAL         CRAG DRC                  —
  CRAG AMBIGUOUS            ☁ CLOUD         CRAG                      evidence summary
  re-adjudication
  MASS-RAG Summarizer       ▣ LOCAL         MASS-RAG Tier A draft     —
  MASS-RAG Extractor        ▣ LOCAL         MASS-RAG Tier A draft     —
  MASS-RAG Reasoner         ▣ LOCAL         MASS-RAG Tier A draft     —
  MASS-RAG Verifier         ☁ CLOUD         MASS-RAG + Spec RAG       3-agent output
  MASS-RAG Refiner          ▣ LOCAL         MASS-RAG Tier A refine    —
  CONSTRUCT scorer          ▣ LOCAL         CONSTRUCT                 —

reranker                    ▣ LOCAL ONNX    Query Decomp Reranker     —
supervisor                  ▣ LOCAL         STRIDE Supervisor         —

gap_detector                ☁ CLOUD         CRAG+VCM+STRIDE hint      synthesis output
cross_validator             ▣ LOCAL         —                         —
evidence_auditor            ▣ LOCAL         RhinoInsight EAM          —

writer                      ☁ CLOUD         —                         synthesis + web excerpts
critic Stage 1 (suspect)    ▣ LOCAL         AlignRAG Tier A draft     —
critic Stage 2 (verify)     ☁ CLOUD         AlignRAG + Spec RAG       suspect_claims
critic Stage 3 (refine)     ▣ LOCAL         AlignRAG Tier A refine    —
revise                      ▣ LOCAL         DSAP Level 1+2            —
─────────────────────────────────────────────────────────────────────────────────────
Total cloud calls: ~6–8                     Total local calls: ~18–22
```

---

## Framing: Constraints Improving Design

Three constraints discovered during development each forced a better solution:

| Constraint | Initial perception | Resulting improvement |
|------------|-------------------|----------------------|
| citation_density v1 penalized better retrieval | Metric bug | v2 formula `min(inline/20, 1.0)` — more robust evaluation |
| Local LLM too slow for standalone use (~481s/query) | Limitation | Hybrid routing: local for evaluation, cloud for generation → −75% cost |
| Privacy requirement blocks raw documents from cloud | Limitation | Privacy-Preserving Speculative RAG: constraint forces drafter/verifier split — structurally impossible to implement without it |

The third constraint is the most interesting: privacy is not just a business requirement
layered onto an existing design — it is the architectural force that makes Speculative RAG
implementable. Without the constraint, there is no structural reason to split drafter and
verifier. With it, the split is mandatory.

---

## Relationship to Basic Hybrid Routing

`ARCHITECTURE.md` already defines a Generation (cloud) vs. Evaluation (local) node-level
split. This document extends that design in two layers:

```
Basic hybrid:     node A (local) → node B (cloud)                   node-level routing
Privacy Spec RAG: draft (local) → verify (cloud) → refine (local)   within a single node
AgentCore (Ph.G): cloud node → AgentCore managed agent               deployment target only
```

The three layers are compatible and independently adoptable. Privacy Spec RAG controls
*what data* crosses the privacy boundary. AgentCore controls *where* the cloud LLM runs.
LangGraph remains the orchestrator throughout all phases.

### Full implementation arc

```
Phase A    HybridProvider wrapper
Phase B-1  Non-CRAG nodes → local
Phase B-2  CRAG threshold recalibration + CRAG → local
Phase C    plan_generator 2-stage (privacy fix)
Phase D    MASS-RAG Speculative RAG (Tier A)
Phase E-0  Critic pre-test: qwen3:8b suspect_claims standalone
Phase E    Critic Speculative RAG (Tier A) + H2 re-test
Phase F    writer input boundary + config validation
Phase G    Cloud nodes → Bedrock AgentCore agents
           (LangGraph orchestration unchanged; llm.complete() → agent.invoke())
Phase H    NaviRAG (conditional on enterprise path)
Phase I    Strands SDK evaluation (separate project gate)
           Requires: Send API, interrupt, conditional edge alternatives designed first
```

---

## Phase 0 Removals Reconsidered

Two techniques were removed in Phase 0:

**Speculative RAG** (arxiv:2407.08223) — removed because:
> "Small drafter + large verifier doesn't fit either usage mode: Bedrock users don't need
> it, Ollama users can't get the benefit without a large verifier."

**NaviRAG** (arxiv:2604.12766) — excluded because:
> "Requires offline KB build. Web-search-centric pipeline is primary path."

Under the hybrid + privacy strategy, both removals are revisitable:

- **Speculative RAG**: hybrid provides the drafter (local) + verifier (cloud) split for the
  first time. Privacy constraint makes the split structurally mandatory — when raw documents
  cannot leave the machine, local draft is the only viable path before cloud verification.
- **NaviRAG**: becomes the natural retrieval layer for the exact use case Privacy Spec RAG
  targets — private local document collections (enterprise KB, medical records, legal archives).

Together they form a coherent **Private Enterprise Research Agent** path that didn't exist
before. Phase 0 removed both techniques for web-search scenarios. The privacy scenario
reinstates both simultaneously.

---

## Node Classification

### Tier A — Privacy Speculative RAG (draft → verify → refine within node)

| Node | Fit | Design |
|------|-----|--------|
| MASS-RAG | **Excellent** | 3-agent (local) produces diverse drafts → cloud detects contradictions + selects best elements → local synthesis with raw doc access |
| Critic (AlignRAG) | **Good** | Local generates suspect_claims list → cloud checks cross-claim logical consistency → local refiner uses original evidence to produce correction hints |

### Tier B — Simple Routing (partial injection in graph.py)

| Node | Route | Reason |
|------|-------|--------|
| `plan_generator` | **2-stage** — see below | Cannot receive original query at cloud; requires structural split |
| `writer` | Cloud | User-facing output; 2–3 draft generation = 18 local calls for detailed reports — cost outweighs benefit |
| `gap_detector` | Cloud | Strategic gap query generation; receives MASS-RAG synthesis (abstracted) — not raw docs |
| `search_worker` CRAG | Local + Cloud (AMBIGUOUS only) | DRC complete locally; score 0.3–0.5 range only sent to cloud for re-adjudication |
| `quality_scorer` | Local | CONSTRUCT 2-call scoring — classification task |
| `supervisor` | Local | retrieve/rewrite/answer 3-way classification |
| `checklist_node` / `evidence_auditor` | Local | Structured mapping tasks |

### Tier C — Re-included via Privacy Spec RAG

| Technique | Phase 0 removal reason | Re-inclusion basis |
|-----------|----------------------|-------------------|
| Speculative RAG | Single provider made drafter/verifier split meaningless | Local = drafter, cloud = verifier — split now natural and privacy-motivated |
| NaviRAG | Offline KB build incompatible with web-search primary path | Privacy scenario makes local documents the primary path; NaviRAG + Privacy Spec RAG = complete private research agent |

---

## plan_generator — 2-Stage Design (Privacy Fix)

### The contradiction

`plan_generator` is the node that receives the original query and produces sub_queries.
Saying "receives sub_queries only" is a contradiction — it cannot receive what it creates.
If sub_queries were produced first, a separate node did that work, not plan_generator.

### Solution: split into local coarse decomposition + cloud STRIDE refinement

```
Original query
      │
      ▼  LOCAL — Stage 1: coarse decomposition
      │
      query_decomp dims:
        ["definition/background", "mechanism/cause",
         "current state/evidence", "comparison/alternative",
         "limitations/challenges"]
      │
      │  coarse dimension labels only (not original query text)
      │
      ▼  CLOUD — Stage 2: STRIDE Sq→Cq refinement
      │
      STRIDE:
        Sq: entity-agnostic abstract strategy per dimension
        Cq: concrete search queries grounded in coarse dims
      │
      final sub_queries → checklist_node → search_worker
```

**What cloud receives**:
```json
{
  "topic_summary": "Analysis of the causes and consequences of global warming",
  "dimensions": ["causal_mechanism", "current_state", "mitigation"]
}
```
`topic_summary` is a 1-sentence paraphrase generated by the local model — not the original
query verbatim. `dimensions` are abstract category labels. Original query text never transmitted.

**Privacy trade-off**: `topic_summary` carries semantic content (it is a paraphrase). This
is not complete privacy — it is "reduced surface exposure, not zero exposure." The cloud
can infer the research topic from `topic_summary`. For deployments where the topic itself
is confidential, Option B (full local plan_generator) must be used despite quality risk.

**Why this works**: query_decomp (arxiv:2507.00355) already defines the 5-dimension
framework locally. STRIDE's value is the Sq→Cq abstract→concrete reasoning step, which
operates on the dimension+topic structure rather than the raw query text.

### Trade-off: Option A vs. Option B

| | Option A (2-stage, recommended) | Option B (full local plan_generator) |
|--|--|--|
| Original query exposure | None | None |
| STRIDE quality | Cloud (full) | Local qwen3:8b (unvalidated) |
| Cloud calls | +1 (Stage 2) | −1 (saved) |
| Risk | Low — STRIDE only refines, not creates | Medium — STRIDE component benchmark failed on qwen3:8b |
| Fallback | If cloud Stage 2 fails: use local coarse dims directly | N/A |

Option B note: E2E benchmark showed STRIDE LOO delta within noise even with Claude — but
that is partly attributable to the dependency graph (Ω) not being implemented. Routing
STRIDE fully local risks compounding an already-partial implementation.

---

## writer — Input Privacy Boundary

The writer node's privacy boundary depends on what evidence it receives.

### Evidence categories

| Evidence type | Source | Privacy status | Transmission rule |
|---------------|--------|---------------|-------------------|
| Web search excerpts | Tavily (public) | Public information | OK to transmit directly |
| MASS-RAG synthesis | Local LLM generated | Already abstracted | OK to transmit |
| Checklist / evidence_store metadata | Local LLM generated | Already abstracted | OK to transmit |
| Local file excerpts | Raw private documents | **Sensitive** | Must NOT transmit directly |

### Rule for local file excerpts

When local file search is used (Qdrant + fastembed), raw excerpts from private documents
enter `evidence_store`. These must NOT reach the cloud writer directly.

Required path: local file excerpt → MASS-RAG Summarizer/Extractor/Reasoner (local) →
synthesis output (abstracted) → writer (cloud).

This means MASS-RAG is **mandatory** in any privacy-sensitive deployment using local files,
not optional. Without MASS-RAG synthesis as the abstraction layer, local file content
would reach the cloud writer as raw excerpt text.

```
Web search result:  excerpt → writer (direct, public)
Local file:         excerpt → MASS-RAG synthesis (local) → writer (abstracted only)
```

This constraint also means the `mass_rag=False` condition cannot be used in privacy mode
when local file search is active.

---

## MASS-RAG Speculative RAG — Detail

```
Raw documents (local only)
      │
      ▼  LOCAL ────────────────────────────────
      │
      ├─ Summarizer → draft_summary
      ├─ Extractor  → {key_spans, source_citation_ids}
      └─ Reasoner   → {inferences, supporting_span_indices}
      │
      │  3 agent outputs only (raw documents stripped)
      │
      ▼  CLOUD ────────────────────────────────
      │
      Verifier:
        "Reasoner inference 2 conflicts with Extractor span 4"
        "Summarizer omits key_span 3 from summary"
      │
      │  judgment + correction instructions only
      │
      ▼  LOCAL ────────────────────────────────
      │
      Refiner: re-reads raw documents, applies verifier feedback
               → final synthesis output
```

Cloud sees: locally-generated summaries, key spans, inferences (abstracted text)
Cloud does NOT see: raw document content, user query, file paths

---

## Critic (AlignRAG) Speculative RAG — Detail

```
Local draft:
  scan report → suspect_claims list
  [{claim_text, reason, confidence}]
      │
      │  claim text only (no original citation content)
      │
Cloud verify:
  cross-claim logical consistency check
  → "claim 3 and claim 7 contradict"
  → "claim 5 lacks supporting evidence"
      │
Local refine:
  accesses original evidence_store
  → correction_hints per misaligned claim
```

### Why this may fix H2 (zero-shot AlignRAG failure)

The E2E benchmark confirmed H2: zero-shot AlignRAG detected 0 misalignments across all 5
queries including explicit contradiction cases (q4, q5).

Root cause analysis: H2 fails because a single LLM generates the report and then verifies
it in the same session. **Self-preference bias** — the model cannot effectively critique
its own output because it inherits the same reasoning path that produced the original
claims. It "sees" the logic as consistent because it was the one that made it consistent.

Privacy Spec RAG breaks this bias structurally:
- Local model generates the report (drafter)
- Cloud model verifies it (verifier) — a *different* model reviewing someone else's work
- The verifier has no stake in defending the original reasoning

This is not a prompt engineering fix. It is a structural separation of writer and reviewer.
Whether it fully resolves H2 without CLM fine-tuning requires empirical validation (Phase C).

---

## Privacy Boundary

### What cloud receives vs. does not receive

| Category | Cloud receives | Cloud does NOT receive |
|----------|---------------|------------------------|
| Documents | Local LLM summaries / judgments | Raw document text |
| Query | Sub_query form (plan_generator) | Original user query text |
| Citations | `[Source N]` labels | Citation excerpt content |
| Scores | Numeric scores + label rationale | Source text that produced scores |

### Caveat: "direct access blocked" ≠ "leakage-proof"

**Indirect leakage via draft content**: The local drafter, to write a useful draft, will
include paraphrased facts from source documents. Cloud sees these in the draft. For public
web content (Tavily results), this is acceptable — the information is already public.
For genuinely sensitive documents (proprietary data, personal records), drafts must apply
masking rules:
- Replace specific figures with `[VALUE]`
- Replace citations with `[Source N]`
- Cloud receives structural reasoning, not factual content

Masking rules in drafter prompts reduce leakage but do not eliminate it — a model may
paraphrase a specific fact without using its exact form.

**Sub-query semantic leakage**: The `plan_generator` (cloud) receives sub_queries, not the
original query. But sub_queries carry the same semantic intent. "Company X competitive
vulnerabilities" → sub_queries still reveal the research topic. For enterprise environments
where query content is itself confidential, sub_queries must be treated as sensitive data.

**Accurate framing**: this architecture blocks raw document direct access. It does not
provide mathematical privacy guarantees. For maximum privacy, use NaviRAG (fully local
retrieval) + Privacy Spec RAG writer/critic with masking rules.

---

## Performance Projection

| | Cloud-only | Hybrid (basic) | Hybrid + Privacy Spec RAG |
|--|--|--|--|
| Cloud calls | 20+ | ~4–5 | ~6–8 |
| Local calls | ~4 | ~12 | ~18–22 |
| Cost/query | $0.75 | ~$0.19 | ~$0.22 |
| Latency (realistic) | 216s | ~350s | ~600–720s |
| Latency (pessimistic) | 216s | ~380s | ~800s |
| Privacy (web content) | low | partial | high (direct access blocked) |
| Privacy (local files) | low | partial | high — requires MASS-RAG as mandatory layer |
| Sub-query privacy | none | none | partial (semantic leakage remains) |

**Cloud call budget breakdown** (Privacy Spec RAG):
```
plan_generator Stage 2              × 1
MASS-RAG verifier (per sub_query)   × 3–4   ← within budget
gap_detector                        × 1
writer                              × 1
critic coherence verify             × 1
─────────────────────────────────────────
Base total                          7–8 calls
CRAG AMBIGUOUS re-adjudication      + 0–2 calls (conditional)
─────────────────────────────────────────
Maximum                             10 calls
```

**Latency basis**:
- Cloud calls: ~7–8 × 10s = 70–80s
- Local calls: ~18–22 × 30s = 540–660s
- Total: 610–740s realistic; 800s+ when qwen3:8b produces longer outputs on complex prompts

**Hybrid ≠ faster**: latency is 3× cloud-only in the realistic estimate. The value
proposition is cost (−70%), privacy (direct access blocked), and rate limit distribution —
not speed. Deployment context must accept 10–13 min/query for the full privacy-preserving
stack. For latency-sensitive use: basic hybrid (−75% cost, ~350s) is the better tradeoff.

---

## NaviRAG Re-inclusion Path

```
Private Enterprise Research Agent (proposed)
│
├── Retrieval: NaviRAG
│   Offline KB build (fully local, ~50 min setup)
│   Hierarchical tree navigation (fully local)
│   Zero web search → zero external data transmission
│
└── Generation: Privacy Spec RAG
    draft (local) → verify (cloud) → refine (local)
    Cloud receives: locally-generated abstractions only
    Cloud does not receive: KB content, user query, document text
```

This represents a coherent product arc:
- **Current**: public web research (Tavily + cloud generation)
- **Proposed**: private enterprise research (local KB + privacy-preserving generation)

The two paths share all nodes except retrieval. NaviRAG re-inclusion is conditional on
local document research becoming a supported primary use case — it is not a drop-in
addition to the web-search path.

---

## Implementation Roadmap

### Phase A — HybridProvider + factory
**Files**: `src/providers/hybrid.py` (new), `src/providers/__init__.py`

- `HybridProvider(cloud, local)`: `.cloud`, `.local` properties; default `complete()` delegates to cloud
- `LLM_PROVIDER=hybrid` factory; env vars: `HYBRID_CLOUD_PROVIDER`, `HYBRID_LOCAL_PROVIDER`, `HYBRID_LOCAL_MODEL`
- `embed()` always routes to local
- **Validation**: run e2e benchmark with HybridProvider where all nodes still use cloud (default delegation). Quality must be identical to cloud-only baseline. This confirms the wrapper adds zero regression before any routing changes.

---

### Phase B-1 — Non-CRAG nodes routed local
**Files**: `src/graph.py` (build_graph only)

Route non-CRAG evaluation nodes to local: `checklist_node`, `supervisor`, `cross_validator`,
`evidence_auditor`, `quality_scorer`, `revise`.
Keep `search_worker` (CRAG + MASS-RAG), `plan_generator`, `gap_detector`, `writer`, `critique` on cloud.

**Validation**: phase1_2_3 overall ±0.03 vs. cloud-only (5-query benchmark).
If regression > 0.03, isolate which local node caused it before continuing.

---

### Phase B-2 — CRAG threshold recalibration + CRAG local routing
**Files**: `src/nodes/search_worker.py` (threshold constants), `src/graph.py`

**Why separate**: CRAG thresholds (0.3/0.5) were calibrated on Claude (Bedrock CRAG Δ+0.66 vs Ollama Δ+0.07). Routing `search_worker` to local before recalibration makes Phase B-1 failure diagnosis ambiguous — is the regression from routing or from threshold mismatch?

Steps:
1. Run CRAG scoring with qwen3:8b on 5-query fixture, log raw relevance score distribution
2. Compare distribution to Claude baseline — adjust thresholds if mean shifts significantly
3. Route `search_worker` to local with recalibrated thresholds
4. Add CRAG AMBIGUOUS cloud re-adjudication (score 0.3–0.5 range only)

**Validation**: CRAG verdict accuracy (CORRECT/INCORRECT/AMBIGUOUS ratios) comparable to Phase 0 baseline. Overall ±0.03 maintained.

---

### Phase C — plan_generator 2-stage
**Files**: `src/nodes/plan_generator.py`

- Extract `_extract_dimensions(query, local_llm) → {topic_summary, dimensions}`
  - `topic_summary`: 1-sentence paraphrase (local LLM) — semantic leakage accepted, verbatim query not transmitted
  - `dimensions`: subset of 5 standard labels
- `generate_plan`: Stage 1 uses `llm.local`, Stage 2 uses `llm.cloud` with `{topic_summary, dimensions}` as input
- Fallback: if `hasattr(llm, 'local')` is False, Stage 1 skipped, original query passed directly (backward compatible)

**Validation**: A/B comparison on 5 queries — plan quality with topic+dims vs. original query. Acceptable if sub_query relevance is comparable (manual review).

---

### Phase D — MASS-RAG Speculative RAG (Tier A) ✅
**Files**: `src/nodes/search_worker.py` (`_mass_rag_analyze`)

- 3 agents run on `local_llm`; add `[VALUE]`/`[Source N]` masking rules to agent prompts
- After agents: cloud verifier call (`cloud_llm`) — 3-agent outputs only, no raw docs (M7a)
- After verifier: local refiner (`local_llm`) — accesses raw `doc_pool` + verifier feedback (M8)
- Graceful degradation: if `not hasattr(llm, 'local')`, skip verifier/refiner, use existing synthesis

**Validation**: LJudge 5-query before/after verifier step. Confirms cloud verifier adds signal.

---

### Phase E-0 — Critic pre-test: qwen3:8b suspect_claims standalone ✅
**Files**: [BENCHMARK.md Layer 3](BENCHMARK.md#layer-3-hybrid-strategy-validation)

Results (5 fixture queries, 11 expected flags):

| Phase | Recall | FP count |
|-------|--------|----------|
| phase1 (off-topic) | 0.33 | — |
| phase2 (fabricated citation) | 0.33 | — |
| phase3 (numeric contradiction) | **0.00** | — |
| **Overall macro recall** | **0.20** | 7 |

Gate decision: phase3 recall < 0.5 → **cloud verifier REQUIRED** in Phase E.
Mode: filtering mode (local generates ≥ 1 claim; confirmed TP=2, FP=7).
Results saved: See [BENCHMARK.md](BENCHMARK.md#layer-3-hybrid-strategy-validation)

---

### Phase E — Critic Speculative RAG (Tier A) ✅
**Files**: `src/nodes/critic.py` (`critique`)

- Stage 1 (local): `_generate_suspect_claims(report, local_llm)` → `[{claim_text, reason, confidence}]`
- Stage 2 (cloud): `_coherence_verify(suspect_claims, cloud_llm)` — **claim text only, no citations** (privacy boundary)
- Stage 3 (local): `_refine_corrections(confirmed, evidence_store, local_llm)` → evidence quotes + citation IDs
- Stage 3 fallback: if refiner returns empty and cloud confirmed issues, use cloud output (no source_quote)
- Graceful degradation: if single provider, Stage 1 skipped, existing single-prompt path used

**H2 Re-test results** (See [BENCHMARK.md](BENCHMARK.md#layer-3-hybrid-strategy-validation)):

| Condition | Detected | Recall | H2 |
|-----------|----------|--------|-----|
| Baseline (single local LLM) | 0 / 3 | 0.00 | FAIL |
| Phase E Spec RAG (local→cloud→local) | **2 / 3** | **0.67** | **PASS** |

Recall delta: +0.67. **H2 resolved.** Root cause confirmed: self-preference bias eliminated by Spec RAG drafter/verifier separation.

---

### Phase F — writer input boundary + config validation ✅
**Files**: `src/nodes/writer.py`, `src/state.py`

- `state.py`: added `"privacy_mode": False` to `DEFAULT_FEATURE_FLAGS`
- `writer.py`: added `_local_citation_ids()`, `_get_mass_rag_synthesis_text()`, `_sanitize_evidence_for_privacy()`
- `write_draft()`: raises `ValueError` for `privacy_mode=True` + `local_search_enabled=True` + `mass_rag=False`
- `write_draft()`: when `privacy_mode=True`, replaces local citation excerpts with MASS-RAG synthesis or `[redacted]` before building evidence string for cloud LLM

**Validation**: unit tests pass — raw local content does not appear in evidence when `privacy_mode=True`.

---

### Phase G — Bedrock AgentCore Integration ✅
**Files**: `src/providers/agentcore.py` (new), `src/agents/bedrock_proxy_handler.py` (new),
           `src/graph.py`

**Prerequisite**: Phase A–F complete and validated.

#### Design principle: LangGraph + AgentCore co-existence

LangGraph remains the orchestrator. AgentCore hosts stateless cloud nodes as managed
agents. The migration scope is contained to node function internals — graph structure
is unchanged.

```
[Local PC — LangGraph orchestration (unchanged)]
      │
      ├─ Local nodes: Ollama direct call (Phase A–F, unchanged)
      │
      └─ Cloud nodes: AgentCore agent via invoke() API
            ┌──────────────────────────────────────────┐
            │  writer_agent.invoke({                   │
            │    synthesis: ...,                       │  ← explicit argument construction
            │    sources: ["[Source 1]", "[Source 2]"] │  ← in LangGraph node code
            │  })                                      │  ← auditable, not agent-autonomous
            └──────────────────────────────────────────┘
```

#### Why "auditable" matters for privacy

The critical property of this structure: what reaches the cloud is determined by **code**,
not by agent reasoning.

```
Auditable (LangGraph node):
  def write_draft_node(state, agent):
      payload = {
          "synthesis": state["mass_rag_outputs"],   # locally generated
          "sources":   ["[Source N]" for ...],       # labels only, not content
      }
      return agent.invoke(payload)                  # code review confirms what's sent

Not auditable (Strands autonomous agent):
  agent.run("write a report")
  # agent decides what to include in tool calls — privacy not code-verifiable
```

The LangGraph node function is the privacy enforcement point. `agent.invoke()` replaces
`llm.complete()` as the call target, but the argument construction remains explicit and
reviewable.

#### Code change scope

```python
# Phase A–F (current):
builder.add_node("write_draft", partial(write_draft_node, llm=cloud_llm))

# Phase G — AgentCoreProvider is a drop-in LLMProvider:
build_graph(
    llm=HybridProvider(cloud=BedrockProvider(), local=OllamaProvider()),
    search_tool=...,
    agentcore_arns={
        "writer": "arn:aws:bedrock-agentcore:us-east-1:ACCOUNT:agent-runtime/writer-v1",
        "gap":    "arn:aws:bedrock-agentcore:us-east-1:ACCOUNT:agent-runtime/gap-v1",
        "critic": "arn:aws:bedrock-agentcore:us-east-1:ACCOUNT:agent-runtime/critic-v1",
        "plan":   "arn:aws:bedrock-agentcore:us-east-1:ACCOUNT:agent-runtime/plan-v1",
    },
)
# No node code changes required — AgentCoreProvider implements LLMProvider interface.
# llm.complete(messages) → invoke_agent_runtime(payload=json(messages)) transparently.
```

Deployed agent code (`src/agents/bedrock_proxy_handler.py`):
```python
# Thin Bedrock proxy running inside AgentCore managed runtime.
# Receives JSON payload → calls Bedrock → returns {"text": "..."}.
def handler(event):
    response = anthropic_bedrock_client.messages.create(**event)
    return {"text": response.content[0].text}
```

LangGraph graph structure (edges, conditions, Send API, interrupt) is entirely unchanged.

#### AgentCore agents to deploy

| Agent | Input | Output | Privacy enforcement |
|-------|-------|--------|-------------------|
| `writer-agent` | MASS-RAG synthesis + `[Source N]` labels | Report draft | Node strips raw excerpts before `invoke()` |
| `critic-agent` | suspect_claims list (Phase E output) | coherence judgment | Node passes claim text only |
| `gap-agent` | coverage summary + checklist status | gap queries | Node passes locally-generated summary |
| `plan-stage2-agent` | `{topic_summary, dimensions}` | STRIDE Cq | Node passes paraphrased topic only |

#### AgentCore benefits unlocked

- **Scaling**: concurrent users handled by AgentCore managed runtime (current: single FastAPI process)
- **Monitoring**: per-agent invocation logs, latency metrics, error rates in CloudWatch
- **Versioning**: `writer-agent-v1` → `writer-agent-v2` with canary rollout, no client code change
- **Session memory**: AgentCore manages cross-session context if needed (currently SQLite)

#### Validation

1. Deploy writer-agent to AgentCore, run 5-query benchmark. Quality delta vs. direct Bedrock call?
2. Confirm no raw document content appears in AgentCore invocation logs (privacy audit).
3. Latency delta: `llm.complete()` vs. `agent.invoke()` round-trip overhead.

---

### Phase H — NaviRAG (conditional, enterprise path only)
**Files**: `src/nodes/local_search_worker.py`

Conditional on local document research becoming a primary use case. Requires separate test KB.
Defer until Phase A–G validated in production.

---

### Phase I — Strands SDK evaluation (separate project gate)

**Prerequisite**: Phase G complete. Strands SDK adoption requires resolving three
LangGraph features with no direct Strands equivalent before migration begins.

| LangGraph feature | Strands equivalent | Required design work |
|------------------|-------------------|---------------------|
| `Send` API fan-out (N parallel search_workers) | `asyncio.gather` + explicit tool routing | Design + implement async fan-out pattern |
| `NodeInterrupt` / human-in-the-loop resume | AgentCore human approval hook | Design interrupt protocol |
| Conditional edges + cycles (critique→revise loop) | Agent loop with explicit termination condition | Implement loop guard |

**Gate condition**: If all three have validated Strands implementations and the migration
delivers measurable benefit (e.g., AgentCore orchestration + unified observability), proceed.
If the work amounts to "re-implementing LangGraph in Strands", defer indefinitely.

The primary motivation for Strands adoption would be: unified framework (local + cloud
agents both using Strands SDK), AgentCore orchestration instead of local LangGraph process,
and managed state machine. These benefits do not justify migration until Phase G is proven.

---

## Open Questions

1. **H2 resolution probability**: Does local suspect identification + cloud coherence check
   produce non-zero `misaligned_claims` on q4/q5? Self-preference bias elimination is the
   hypothesis — but cloud verifier receives claim text without source evidence, so it can
   only check internal consistency, not factual grounding. May still miss fact-level
   misalignments.

2. **Draft quality floor for verifier utility**: If qwen3:8b produces poor-quality
   key_spans/inferences, does the cloud verifier's feedback become useful? Or does garbage
   in → garbage feedback? Need to compare qwen3:8b MASS-RAG output quality against Claude
   baseline before trusting the verifier step.

3. **CRAG Δ gap under hybrid routing**: Component benchmark: Bedrock CRAG Δ+0.66,
   Ollama Δ+0.07. If local CRAG scoring is this much weaker, does the AMBIGUOUS-only
   cloud re-adjudication recover the gap, or is CORRECT/INCORRECT scoring itself the
   problem? May require re-routing CRAG fully to cloud (defeats cost benefit for that node).

4. **Masking rule enforcement**: Drafter prompt instruction to mask `[VALUE]` and
   `[Source N]` — does the model reliably follow this on all outputs, or does it leak
   through paraphrase? Requires adversarial testing on 10–20 sensitive document samples.

5. **plan_generator 2-stage quality**: Stage 1 (local coarse decomposition) produces
   dimension labels, not full sub_queries. Does STRIDE Stage 2 (cloud) produce better
   Cq from dimension labels vs. from the original query directly? If not, the privacy
   benefit comes at a quality cost. Requires A/B test: original query → plan_generator
   vs. dimension labels → plan_generator.

6. **MASS-RAG mandatory for local file privacy**: When local file search is active,
   MASS-RAG synthesis is the only abstraction layer preventing raw excerpt transmission
   to the cloud writer. This creates a hard dependency: `mass_rag=True` required in
   privacy mode with local files. Need to implement an explicit validation check that
   rejects `privacy_mode=True` + `mass_rag=False` + `local_files=True` configuration.
