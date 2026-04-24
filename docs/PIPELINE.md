# Pipeline Details

Describes the role, inputs/outputs, and design principles of each node.

---

## Node List

| Node | File | Role |
|------|------|------|
| `generate_plan` | `nodes/plan_generator.py` | Query decomposition + STRIDE meta-planning |
| `checklist_node` | `nodes/checklist_node.py` | RhinoInsight VCM: sub-queries → verifiable checklist |
| `plan_review` | `graph.py` | User approval interrupt |
| `search_orchestrator` | `graph.py` | Send API fan-out trigger |
| `search_worker` | `nodes/search_worker.py` | CRAG filtering + MASS-RAG 3-agent synthesis |
| `local_search_worker` | `nodes/local_search_worker.py` | Local file search |
| `reranker` | `nodes/reranker.py` | Cross-encoder reranking (ms-marco-MiniLM) |
| `supervisor` | `nodes/supervisor.py` | STRIDE: per sub-query routing (retrieve/rewrite/answer) |
| `gap_detector` | `nodes/gap_detector.py` | Coverage gap analysis with CRAG/VCM/STRIDE signals |
| `gap_search` | `nodes/gap_detector.py` | Additional search for gap queries |
| `cross_validator` | `nodes/cross_validator.py` | Source cross-validation + SDP pruning |
| `evidence_auditor` | `nodes/evidence_auditor.py` | RhinoInsight EAM: evidence normalization + claim binding |
| `write_draft` | `nodes/writer.py` | Report draft writing + privacy sanitization |
| `critique` | `nodes/critic.py` | AlignRAG 3-phase diagnosis + Speculative RAG |
| `evidence_stage2` | `nodes/evidence_auditor.py` | EAM Stage 2b: misalignment flag annotation |
| `revise` | `nodes/critic.py` | Section-by-section rewrite from feedback |
| `quality_scorer` | `nodes/quality_scorer.py` | CONSTRUCT: field-level trust scoring (called within search_worker) |
| `finalize` | `graph.py` | Finalize the report |

---

## 1. generate_plan — Query Decomposition

**Purpose**: Decompose a single query into sub-queries that can be searched in parallel from multiple angles.

**Improvement** (arxiv:2507.00355 applied):
Instead of simply "split the query into N parts", **explicitly specify decomposition dimensions** to minimize semantic overlap.

```
Decomposition dimensions:
  [Definition/Background]    Core concept definitions, historical context
  [Status/Evidence]          Latest data, statistics, real-world cases
  [Comparison/Alternatives]  Other approaches, competing technologies, analogues
  [Cause/Mechanism]          How it works, causal relationships
  [Limitations/Challenges]   Drawbacks, risk factors, unresolved issues
```

**Example**: Query "Current state of quantum computing"
- sq1: "Core principles of quantum computing and qubit technology definitions" [Definition/Background]
- sq2: "2024-2025 practical quantum computing status and IBM/Google roadmaps" [Status/Evidence]
- sq3: "Quantum computing vs classical computing performance comparison and use cases" [Comparison/Alternatives]
- sq4: "Quantum decoherence problem and error correction techniques" [Cause/Mechanism]
- sq5: "Technical and cost challenges blocking quantum computing commercialization" [Limitations/Challenges]

**Output**: `plan.sub_queries` (4–6 items, each including a dimension tag)

---

## 2. plan_review — User Interrupt

**Purpose**: A checkpoint where users can review, approve, modify, or reject the plan.

Uses the LangGraph `interrupt()` API to pause execution and wait for user input.

```python
user_response = interrupt({
    "type": "plan_review",
    "plan": state["plan"],
    "message": "Please review the research plan and approve or modify it.",
})
```

On resume, routing goes to `Command(goto="search_orchestrator")` to start the search.
On rejection, routing returns to `Command(goto="generate_plan")` to regenerate.

---

## 3. search_orchestrator + search_worker — Parallel Search

**Purpose**: Execute N sub-queries in true parallel.

When `fan_out_to_workers()` returns `list[Send]`, LangGraph runs them in parallel and automatically merges `citations`.

**CRAG Retrieval Evaluator** (arxiv:2401.15884):
Each worker does not use Tavily search results as-is; it first performs **batch relevance evaluation** with an LLM.

```
Tavily results → LLM batch evaluation → relevant / partial / irrelevant
                                            │            │           │
                                       full fetch    summary only  skip
                                       score×1.0     score×0.6×0.6   -
                                       (if empty → score×0.4 fallback)
```

**Parameters by depth**:

| depth | max_results (Tavily) | max_fetch (full body) |
|-------|----------------------|-----------------------|
| fast | 4 | 1 |
| normal | 7 | 3 |
| deep | 12 | 6 |

Dual filtering effect:
- Tavily relevance score ≥ 0.7: term-matching based 1st filter (fast)
- LLM CRAG evaluation: semantic relevance to research intent 2nd filter (accurate)

**Zero-citation prevention**: If all results are `irrelevant`, top 2 are promoted to partial (confidence = score × 0.6 × 0.6).

---

## 4. gap_detector + gap_search — Gap Detection & Multi-round Iteration

**Purpose**: Evaluate whether the collected citation sources sufficiently cover the sub-queries in the research plan.

The LLM analyzes:
1. Whether each sub-query is supported by at least 2 sources
2. Whether important perspectives or data are missing

When gaps are found, additional search queries are added to `gap_queries` and routed to the `gap_search` node. Otherwise, routing goes directly to `cross_validator`.

```
coverage_score calculation:
  (number of answered sub-queries) / (total sub-queries) × source confidence weight
```

**Parameters by depth**:

| depth | max_gap_queries | analysis source limit | gap search result count |
|-------|----------------|----------------|----------------|
| fast | 2 | 15 | 3 |
| normal | 3 | 25 | 5 |
| deep | 5 | 50 | 8 |

**Deep mode multi-round loop** (same structure as commercial deep research tools):

```
gap_search complete
    ↓
should_continue_research()
    ├── depth == "deep" AND research_round < 2  →  gap_detector (re-enter)
    └── otherwise                               →  cross_validator
```

In deep mode, the process iterates up to 3 rounds (1 initial search + 2 gap fill passes) to maximize coverage.

---

## 5. cross_validator — Cross-Validation

**Purpose**: Evaluate factual consistency across multiple sources + provide quality signals for report writing.

The LLM analyzes up to 25 citations to identify:
- **Cross-confirmed groups**: Sources supporting the same fact → marked `✓ cross-confirmed`
- **Conflicting information**: Pairs of mutually contradictory sources → marked `⚠️ conflict`
- **Single-source claims**: Important information with only one source → marked `⚠️ single-source`

The returned `cross_validation_report` is passed to the writer node and used to display confidence levels for each claim during report writing.

**Note**: Because the `citations` field is managed by the `operator.add` reducer, the cross_validator does not directly modify citations.

---

## 6. write_draft — Report Writing (Section-by-Section Generation)

**Purpose**: Synthesize collected evidence into a structured research report. To overcome the max_tokens limit of a single LLM call, sections are generated separately.

Inputs:
- `citations`: Full citation list
- `cross_validation_report`: Cross-validation quality signals
- `plan.interpretation`: Query interpretation
- `report_length`: brief | standard | detailed

**Generation strategy by report_length**:

| Mode | LLM calls | Sections generated | Expected length |
|------|-------------|-----------|-----------|
| `brief` | 1 | Single prompt (summary + findings + conclusion combined) | 2,000 max_tokens |
| `standard` | 3 | Key summary + main findings + conclusion | 7,000 max_tokens total |
| `detailed` | 3+N | Key summary + main findings + per-sub-query analysis (N) + conclusion | 15,000+ max_tokens total |

For detailed mode with 8 sub-queries:
```
2k(summary) + 3k(findings) + 8×4k(analysis) + 2k(conclusion) = 39k tokens
```
This is approximately 5× the single-call max_tokens (8k) and is impossible without section-by-section generation.

**Report structure (detailed)**:
```
# [Original Query]

## Key Summary
## Main Findings  [numbered list, each item citing [Source N]]
## Concept Definitions and Background    ← sub-query [Definition/Background] analysis
## Current Status and Empirical Data     ← sub-query [Status/Evidence] analysis
## Comparative Analysis and Alternatives ← sub-query [Comparison/Alternatives] analysis
## Cause and Mechanism Analysis          ← sub-query [Cause/Mechanism] analysis
## Limitations and Future Challenges     ← sub-query [Limitations/Challenges] analysis
## Conclusion
## Sources
```

Automatic labeling based on cross-validation results:
- `✓ cross-confirmed`: Claims agreed upon by multiple sources
- `⚠️ single-source`: Claims with only one source
- Conflicting information: Both sides presented together

> **Ollama timeout prevention**: Sections are executed sequentially to prevent request queue explosion on single-GPU environments.

---

## 7. critique + revise — Quality Review Loop

**Purpose**: Guarantee report quality using the Evaluator-Optimizer pattern.

**AlignRAG improvement** (arxiv:2504.14858):
In addition to existing logic/citation review, **factual alignment** checking is added.

```
Review items:
  1. Logical contradictions or self-conflicts
  2. Assertive claims without citations
  3. Unanswered sub-queries
  4. [AlignRAG] Claims that actually mismatch their cited sources
  5. Structural issues
```

By including citation source excerpts in the critique context, it directly verifies whether "the report's claims are expressed differently from the actual source content".

**Maximum rewrites by depth**:

| depth | max_revisions |
|-------|---------------|
| fast | 1 |
| normal | 1 |
| deep | 3 |

When the maximum count is reached, `passed=True` is forced to finalize (infinite loop prevention).

**revise section-by-section editing**: The revise node also splits the report by `## ` headings and modifies each section independently. The sources list section is excluded from modification.

Result: `CriticFeedback.passed == True` → `finalize` / `False` → `revise`

---

## 8. Chat Graph Nodes

### router
Classifies follow-up questions into 3 paths (max_tokens=200, temperature=0 — optimized for fast judgment):
- `memory`: Can be answered directly from the report
- `targeted`: Needs 1–2 additional searches
- `new_research`: Completely outside the scope of existing research

### memory_answer
Answers using the full report + citation sources + conversation history as context. Explicitly notifies when content is not in the report.

### targeted_search
Generates 1–2 search queries with an LLM → executes them → stores in `extra_citations` → adds context to `memory_answer`.

### new_research_signal
Generates a message indicating new research is needed. Guides the client to start a new research session.

---

## Pipeline Performance Characteristics

| Stage | Time | LLM calls |
|------|-----------|-------------|
| Plan generation | ~3s | 1 |
| Parallel search (5 sub-queries) | ~15–30s | 5–15 (including CRAG) |
| Gap detection | ~3s | 1 |
| Gap additional search | ~5–10s | 1–3 |
| Cross-validation | ~5s | 1 |
| Report writing (brief) | ~5s | 1 |
| Report writing (standard) | ~15s | 3 |
| Report writing (detailed, N=5) | ~40s | 8 |
| Quality review | ~5s | 1 |
| **Total (normal + standard)** | **~50–80s** | **~14–27** |
| **Total (deep + detailed)** | **~120–180s** | **~30–50** |

> Local LLM (Qwen3 8B, M1 Pro 16GB) is 3–5× slower than cloud.
> The parallel search stage accounts for approximately 40–50% of total time.
> In detailed mode, report writing is the second largest contributor.
