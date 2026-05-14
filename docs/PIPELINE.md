# Pipeline Details — Stage-Aware Local-Cloud Inference

Describes the role, inputs/outputs, and System 1/System 2 routing of each node in the four-phase pipeline.

---

## Pipeline Overview

```
Planning Phase    → Retrieval Phase    → Drafting Phase    → Verification Phase
generate_plan        search_worker×N      mass_rag_drafters    gap_detector
plan_elaboration     crag_recheck         synthesis            gap_search
checklist_node       reranker                                  cross_validator
[plan_review]        supervisor                                evidence_auditor
                                                               critique/revise
```

**System 1 (local)**: generate_plan, checklist_node, search_worker (CRAG scoring), reranker, supervisor, mass_rag_drafters, gap_search, cross_validator, evidence_auditor, critique, revise

**System 2 (cloud)**: plan_elaboration, crag_recheck (AMBIGUOUS only), synthesis, gap_detector

---

## Node List

| Node | File | Phase | Tier | Role |
|------|------|-------|------|------|
| `generate_plan` | `nodes/plan_generator.py` | Planning | LOCAL | STRIDE Sq + query_decomp |
| `plan_elaboration` | `nodes/plan_generator.py` | Planning | CLOUD | STRIDE Cq from skeleton |
| `checklist_node` | `nodes/checklist_node.py` | Planning | LOCAL | RhinoInsight VCM |
| `plan_review` | `graph.py` | Planning | — | User approval interrupt |
| `search_orchestrator` | `graph.py` | Retrieval | — | Send API fan-out trigger |
| `search_worker` | `nodes/search_worker.py` | Retrieval | LOCAL | CRAG + MASS-RAG drafting |
| `crag_recheck` | `nodes/search_worker.py` | Retrieval | CLOUD | AMBIGUOUS re-adjudication |
| `local_search_worker` | `nodes/local_search_worker.py` | Retrieval | LOCAL | Local file search |
| `reranker` | `nodes/reranker.py` | Retrieval | LOCAL | Cross-encoder reranking |
| `supervisor` | `nodes/supervisor.py` | Retrieval | LOCAL | STRIDE: retrieve/rewrite/answer |
| `synthesis` | `nodes/writer.py` | Drafting | CLOUD | Multi-draft synthesis |
| `gap_detector` | `nodes/gap_detector.py` | Verification | CLOUD | Coverage-gap analysis |
| `gap_search` | `nodes/gap_detector.py` | Verification | LOCAL | Additional search for gaps |
| `cross_validator` | `nodes/cross_validator.py` | Verification | LOCAL | Source cross-validation |
| `evidence_auditor` | `nodes/evidence_auditor.py` | Verification | LOCAL | RhinoInsight EAM |
| `write_draft` | `nodes/writer.py` | Verification | LOCAL | Report draft assembly |
| `critique` | `nodes/critic.py` | Verification | LOCAL | AlignRAG 3-phase diagnosis |
| `evidence_stage2` | `nodes/evidence_auditor.py` | Verification | LOCAL | EAM Stage 2b misalignment flags |
| `revise` | `nodes/critic.py` | Verification | LOCAL | Section-by-section rewrite |
| `quality_scorer` | `nodes/quality_scorer.py` | Cross-cutting | LOCAL | CONSTRUCT trust scoring |
| `finalize` | `graph.py` | — | — | Finalize the report |

---

## 1. Planning Phase

### generate_plan — Query Decomposition + Abstract Planning [LOCAL]

**Purpose**: Decompose the query and generate an abstract research skeleton (Sq) that the cloud can elaborate without seeing the original query.

**STRIDE Meta-Planner** (arxiv:2604.17405):
- Stage 1 (local): Generates entity-agnostic abstract strategy Sq with [ENTITY] slots
  - e.g., "need: definition + current state + comparison" without specific entities
- Stage 2 (cloud, `plan_elaboration`): Derives concrete plan Cq from Sq — cloud receives only the skeleton

**Query Decomposition** (arxiv:2507.00355):
- 5-dimension framework applied to Cq sub-queries:
  - [Definition/Background] — Core concept definitions, historical context
  - [Status/Evidence] — Latest data, statistics, real-world cases
  - [Comparison/Alternatives] — Other approaches, competing technologies
  - [Cause/Mechanism] — How it works, causal relationships
  - [Limitations/Challenges] — Drawbacks, risk factors, unresolved issues
- Original query always prepended as sq0 (Q = {q} ∪ Decompose(q))

**Output**: `plan.sub_queries` (4–6 items with dimension tags)

### plan_elaboration — Concrete Plan Generation [CLOUD]

**Purpose**: Elaborate the abstract skeleton into concrete execution steps.

**Privacy**: Receives only the plan skeleton (Sq) — never the original query text. The cloud infers the research topic from the skeleton structure, not from raw query content.

### checklist_node — Sub-Goal Tracking [LOCAL]

**Purpose**: Generate a verifiable checklist of sub-goals before research begins.

**RhinoInsight VCM** (arxiv:2511.18743): Converts each sub-query into a verifiable subgoal statement. Tracks completion after each search cycle; surfaces uncovered goals to gap detection.

**Output**: `checklist` — `[{id, subgoal, sub_query_id, status: pending|partial|complete, evidence_ids}]`

### plan_review — User Interrupt

Uses LangGraph `interrupt()` API to pause execution and wait for user approval. Users can review, edit, or reject the plan before search begins.

---

## 2. Retrieval Phase

### search_worker × N — CRAG + MASS-RAG Drafting [LOCAL]

**Purpose**: Execute N sub-queries in parallel, filter documents, and produce section drafts.

**CRAG** (arxiv:2401.15884) — 3-way classification:

```
Tavily results → LLM batch evaluation → CORRECT / AMBIGUOUS / INCORRECT
                                            │            │           │
                                       full fetch    summary only  skip
                                       score≥0.5     score 0.3–0.5   score<0.3
```

- CORRECT: Decompose-then-Recompose — strip scoring + extraction in single LLM call
- AMBIGUOUS: Escalated to `crag_recheck` (cloud) using title + 150-char excerpt only
- INCORRECT: Delegated to `gap_detector` as gap signal (no pipeline-level retry)

**MASS-RAG Drafting** (arxiv:2604.18509) — 3-agent parallel:
- Summarizer [LOCAL]: Full document summarization
- Extractor [LOCAL]: Key fact and citation span extraction
- Reasoner [LOCAL]: Cross-document inference and causal relationships
- All three agents read source documents directly; only draft text forwarded to synthesis

**Parameters by depth**:

| depth | max_results (Tavily) | max_fetch (full body) |
|-------|----------------------|-----------------------|
| fast | 4 | 1 |
| normal | 7 | 3 |
| deep | 12 | 6 |

### crag_recheck — AMBIGUOUS Re-adjudication [CLOUD]

**Purpose**: Resolve AMBIGUOUS CRAG verdicts without exposing document bodies.

**Privacy**: Receives only document title + 150-character excerpt. Never receives full document content. Resolves the over-conservative tendency of small local models on boundary cases (score 0.3–0.5).

### reranker — Cross-Encoder Reranking [LOCAL]

**Purpose**: Re-rank all candidate documents against the original query (not sub-queries).

**Speculative Reranking** (arxiv:2407.08223): Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (fastembed ONNX backend, ~80MB). Decomposition improves recall; reranking restores precision.

- Rerank basis: original_query scored against `title + excerpt`
- Top-k by depth: fast=10, normal=20, deep=40
- URL-based dedup before reranking

### supervisor — STRIDE Routing [LOCAL]

**Purpose**: Per sub-query routing decision (retrieve/rewrite/answer).

**STRIDE Supervisor** (arxiv:2604.17405): Assigns action per sub-query based on CRAG verdict and evidence quality. Rewrite decisions feed into gap_detector as STRIDE hint axis.

---

## 3. Drafting Phase

### synthesis — Multi-Draft Synthesis [CLOUD]

**Purpose**: Synthesize locally-generated section drafts into a coherent final report.

**Privacy**: Receives only the 3-agent draft text (summary, key_spans, inferences) — never the underlying retrieved documents. The cloud synthesizes from locally-generated abstractions.

**MASS-RAG Synthesis** (arxiv:2604.18509): Detects contradictions across 3 drafts, selects best elements, integrates into coherent report.

**Generation strategy by report_length**:

| Mode | LLM calls | Expected length |
|------|-----------|-----------------|
| `brief` | 1 | ~2,000 tokens |
| `standard` | 3 | ~7,000 tokens total |
| `detailed` | 3+N | ~15,000+ tokens total |

---

## 4. Verification Phase

### gap_detector — Coverage-Gap Detection [CLOUD]

**Purpose**: Survey the assembled evidence to identify under-covered topics and trigger targeted re-retrieval.

**Privacy**: Receives a locally-compiled coverage index — a structured record of retrieved topics and source counts — not the raw documents. This cross-document judgment must precede claim verification, as verifying claims against an incomplete evidence set yields unreliable results.

**Hint axes**: CRAG signals (retrieval quality) + VCM signals (subgoal completion) + STRIDE signals (rewrite decisions).

**Parameters by depth**:

| depth | max_gap_queries | gap search results |
|-------|----------------|-------------------|
| fast | 2 | 3 |
| normal | 3 | 5 |
| deep | 5 | 8 |

**Deep mode multi-round loop**: In deep mode, iterates up to 3 rounds (1 initial + 2 gap fill passes).

### cross_validator — Cross-Source Consistency [LOCAL]

**Purpose**: Evaluate factual consistency across multiple sources.

Analyzes up to 25 citations to identify:
- Cross-confirmed groups: sources supporting the same fact → `✓ cross-confirmed`
- Conflicting information: mutually contradictory sources → `⚠️ conflict`
- Single-source claims: important information with only one source → `⚠️ single-source`

### evidence_auditor — EAM Claim Binding [LOCAL]

**Purpose**: Normalize evidence store and bind claims to citations.

**RhinoInsight EAM** (arxiv:2511.18743):
- Stage 1: Normalize — dedup + sort + verification_level (corroborated/single_source/unverified)
- Stage 2a: MASS-RAG key_spans → claim_bindings per evidence item
- Stage 2b: critic_feedback.misaligned_claims → misalignment_flags per evidence item

### critique — AlignRAG 3-Phase Diagnosis [LOCAL]

**Purpose**: Detect factual misalignments between report claims and cited sources.

**AlignRAG** (arxiv:2504.14858) — 3-phase diagnosis:
1. Phase 1: Relevance assessment — wrong span importance
2. Phase 2: Query-evidence mapping — wrong relationship identified
3. Phase 3: Evidence-integrated synthesis — unsupported conclusion

Output: `misaligned_claims: [{phase, claim, source_citation_ids, source_quote, correction_hint}]`

`passed` computed in code: `len(misaligned)==0 and len(uncited)==0 and len(unanswered)==0`

**Maximum revisions by depth**:

| depth | max_revisions |
|-------|---------------|
| fast | 1 |
| normal | 1 |
| deep | 3 |

### revise — Section-by-Section Rewrite [LOCAL]

**Purpose**: Apply critique feedback to revise the report section by section.

**DSAP** (arxiv:2512.20660): JSON guard functions applied to all structured outputs. Level 1 (context refinement) + Level 2 (stagnation detection + strategy switch).

Splits report by `## ` headings and modifies each section independently. Sources section excluded from modification.

---

## 5. Chat Graph Nodes

### router [LOCAL]
Classifies follow-up questions into 3 paths (temperature=0, deterministic):
- `memory`: Can be answered from the report
- `targeted`: Needs 1–2 additional searches
- `new_research`: Outside scope of existing research

### memory_answer [LOCAL]
Answers using full report + citation sources + conversation history.

### targeted_search [LOCAL]
Generates 1–2 search queries → executes → stores in `extra_citations` → adds context to memory_answer.

---

## Pipeline Performance Characteristics

From the paper (NVIDIA L4, Ubuntu 22.04):

| Configuration | Latency/query | Cloud calls | Local calls |
|---------------|---------------|-------------|-------------|
| Cloud-only Sonnet | 270s | ~20+ | ~4 |
| exaone3.5:2.4b + Sonnet | 233s | ~6–8 | ~18–22 |
| gemma3:4b + Sonnet | 312s | ~6–8 | ~18–22 |
| exaone3.5:2.4b (all-local) | 174s | 0 | ~22–26 |

**Cloud call budget breakdown** (hybrid):
```
plan_elaboration (STRIDE Cq)        × 1
crag_recheck (AMBIGUOUS only)       × 0–2  (conditional)
synthesis (MASS-RAG)                × 3–4  (per sub-query)
gap_detector                        × 1
─────────────────────────────────────────
Total                               5–8 calls
```
