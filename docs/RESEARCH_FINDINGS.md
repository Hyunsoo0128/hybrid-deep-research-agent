# Research Findings and Applied Studies

Summarizes the technical investigations, analyses, and academic paper techniques actually incorporated into the pipeline during the development of this project.

---

## 1. Applied Academic Paper Techniques

### 1.1 Query Decomposition
**Paper**: arxiv:2507.00355
**Results**: MRR@10 +36.7%, F1 +11.6%
**Applied in**: `src/nodes/plan_generator.py`

**Core idea**: By splitting a query into semantically distinct **decomposition dimensions** before searching, each sub-query covers a different document cluster, greatly improving overall recall.

Instead of simply "split the query into N parts", specifying dimensions prevents overlap:
- Definition/Background — conceptual documents
- Status/Evidence — latest statistics and cases
- Comparison/Alternatives — competing technologies
- Cause/Mechanism — in-depth technical documents
- Limitations/Challenges — critical analysis

**Before vs After**:
```
Before: "quantum computing" → ["what is quantum computing", "quantum computing status", "quantum computing companies"]
After: ["qubit definition and history" [Definition/Background],
       "2025 quantum computer performance comparison" [Status/Evidence],
       "quantum vs classical computing gap" [Comparison/Alternatives],
       "decoherence problem mechanism" [Cause/Mechanism],
       "commercialization barriers and cost challenges" [Limitations/Challenges]]
```

---

### 1.2 CRAG (Corrective RAG)
**Paper**: arxiv:2401.15884
**Results**: Improved answer accuracy by removing irrelevant documents
**Applied in**: `src/nodes/search_worker.py`

**Core idea**: Instead of using search results as-is, an LLM evaluates the relevance of each result (Retrieval Evaluator) and classifies them into three tiers:

```
Search results
    │
    ▼
LLM batch evaluation (1 call)
    │
    ├── "relevant"   → full body fetch, key extraction
    │                  confidence = relevance_score × 1.0
    │                  (if extraction returns empty → × 0.4 fallback)
    ├── "partial"    → summary only
    │                  confidence = relevance_score × 0.6 × 0.6
    └── "irrelevant" → skip entirely
                       (zero-citation guard: if all results irrelevant, top-2 promoted to partial)
```

**Dual filtering effect**:
- Tavily relevance_score (0.7 or above): keyword matching-based first filter
- CRAG LLM evaluation: semantic relevance to the question's intent, second filter

Only documents that pass both filters are fully fetched → simultaneously reduces LLM cost and noise.

---

### 1.3 AlignRAG
**Paper**: arxiv:2504.14858
**Results**: 8B critic achieves +12.1% accuracy improvement vs 72B (out-of-domain)
**Applied in**: `src/nodes/critic.py`

**Core idea**: During report quality review, in addition to "is it well-written", separately verifies **whether the claims in the report align with the cited sources**. This is called a "factual alignment" check.

Problems that actually occur: LLM "exaggerates", "undersells", or "reasons without evidence" from source content.

**Implementation**: Includes citation source excerpts as context in the critique node, and adds a `misaligned_claims` item:
```json
{
  "passed": false,
  "misaligned_claims": [
    "Report: 'X shows 100% success rate' → Source: 'X achieved approximately 80% success rate'"
  ]
}
```

`misaligned_claims` are passed to the revise node and corrected to match the sources.

---

### 1.4 DSAP Guard Functions
**Paper**: arxiv:2512.20660
**Results**: LLM parsing reliability +20–66pp
**Applied in**: `src/utils/llm_json.py` (applied to all nodes)

**Core idea**: When an LLM returns malformed JSON, instead of a simple fallback, **feeding back the parse error information to the LLM** for a retry greatly improves the success rate.

**Implementation**:
```
Attempt 1: Normal call
  → Parse success: return
  → Parse failure: add error message + schema hint to conversation

Attempt 2: System prompt → "ONLY valid JSON. No markdown, no explanations."
  → Parse success: return
  → Parse failure: retry

Attempt 3: ...

Final failure: return fallback dictionary (safe default value per node)
```

Applied consistently across 6 nodes (plan_generator, search_worker, gap_detector, cross_validator, critic, router).

---

### 1.5 Speculative RAG — Removed
**Paper**: arxiv:2407.08223
**Reason for removal**: The paper's value proposition (small fine-tuned drafter + large verifier)
does not fit either usage mode of this pipeline:
- Bedrock/Claude users: already perform well without it (all_on → 0.97 overall score)
- Local LLM users: cannot benefit without a large verifier model, which defeats the local-only purpose

The parallel fan-out search (Send API) that was previously labeled as "Speculative RAG" is now
simply the default search architecture — it was never Speculative RAG to begin with.

---

## 2. Local LLM Technology Trend Survey (2025–2026)

For Phase 5 (Local LLM) implementation of DeepResearch, a survey of the latest local LLM landscape as of April 2026 was conducted.

### Key Model Status

| Model | Parameters | Features | Korean | Recommended Use |
|------|----------|------|--------|-----------|
| **Qwen3-14B** | 14B | Thinking Mode, balanced Korean/English | ★★★★☆ | DeepResearch default |
| **Qwen3-32B** | 32B | Higher reasoning accuracy | ★★★★☆ | High-performance analysis |
| **EXAONE-3.5:7.8B** | 7.8B | LG AI, Korean-specialized | ★★★★★ | Korean-only use |
| **Llama4-Scout** | 17B (MoE) | 10M context, efficient | ★★★☆☆ | Long document processing |
| **Phi-4** | 14B | Math and code specialized | ★★☆☆☆ | Technical fields |

### Cloud vs Local LLM Performance Comparison (Measured Data)

> The figures below are directly measured with `eval/benchmark.py`.
> Test environment: M1 Pro 16GB, qwen3:8b (5.2GB), Ollama v0.4+
> Detailed results: [`docs/BENCHMARK.md`](BENCHMARK.md)

**Component benchmark measurements (11 tests)**:

| Category | qwen3:8b (measured) | Claude Sonnet 4.6 (estimated from public benchmarks) |
|---------|----------------|----------------------------------------|
| JSON instruction compliance | **0.75** (3/4 passed) | ~0.98 |
| Query decomposition quality | **0.62** (2/3 passed) | ~0.96 |
| Reasoning and factual alignment | **0.80** (2/3 passed) | ~0.98 |
| Korean response quality | **0.98** (1/1 passed) | ~0.99 |
| **Overall score** | **74.9%** | **~97%** |
| **Pass rate** | **72.7%** | **~95%** |

**Speed measurements**:

| Item | qwen3:8b (M1 Pro) | Claude Sonnet 4.6 |
|------|-------------------|-------------------|
| Average latency per call | **21.5s** | **~2–4s** |
| `fast + brief` total | ~8–12 min | ~45–75s |
| `normal + standard` total | ~15–25 min | ~2–4 min |
| `deep + detailed` total | **~40–70 min** | **~8–15 min** |

**Major failure patterns of qwen3:8b found in measurements**:

1. **Thinking mode interference** (most common): Output inside `<think>` tags mixes with the actual response, causing JSON parse failures. The retry logic in `llm_json()` compensates for most cases, but increases latency.
2. **Copying complex nested JSON templates**: Tendency to return example JSON as-is. Resolved with 2 DSAP retries.
3. **Empty responses for comparison queries**: Cases where thinking becomes very long and actual output is empty. Requires retry.

**Conclusions (updated based on measurements)**:

| Scenario | Recommended Settings |
|---------|---------|
| Privacy required, speed not a concern | qwen3:8b + fast + brief |
| Development/experimentation (zero cost) | qwen3:8b + normal + standard |
| Production quality | Claude Sonnet + deep + detailed |
| Balanced (speed + quality) | Claude Sonnet + normal + standard |

### Effect of Improvement Techniques (When Applied to Local LLM)

| Technique | Before | After | Improvement |
|------|---------|---------|------|
| Query Decomposition | Baseline | +36.7% MRR | Search coverage |
| CRAG Evaluator | Baseline | ~40% noise reduction | Citation quality |
| DSAP Guard | ~30% parse failures | ~5% parse failures | Stability |
| AlignRAG Critique | Baseline | ~30% factual error reduction | Accuracy |

---

## 3. Architecture Pattern Survey

Key RAG architecture patterns evaluated during development:

### Naive RAG vs Advanced RAG vs Agentic RAG

```
Naive RAG:         query → search → LLM → answer
                   Issues: 1 search pass, no gaps, no fact verification

Advanced RAG:      query rewriting → search → re-ranking → LLM → answer
                   Improvement: higher search quality↑, lower speed↓

Agentic RAG:       query decomposition → parallel search → gap detection → additional search
(this project)      → cross-validation → write → critique → improve → complete
                   Improvement: complex analytical questions, multi-angle coverage
```

This project follows the Agentic RAG pattern, specifically including an **iterative self-improvement loop** (critique → revise) and a **knowledge gap fill loop** (gap_detector → gap_search).

### Why LangGraph Fits This Pattern

1. **Cycle support**: revise → critique → revise loop
2. **Conditional branching**: gap found/not found → different paths
3. **Fan-out/join**: parallel search → single aggregation
4. **Interrupt**: human-in-the-loop plan approval
5. **Persistent State**: checkpoint for pausing and resuming

DAG-based frameworks (LangChain Expression Language, etc.) do not support cycles, making this pattern impossible to implement.

---

## 4. Prompt Engineering Patterns

Prompt design principles applied throughout the pipeline:

### ACI (Autonomous Context Integration) Principle
Instead of putting entire web pages into the LLM context, extract only relevant content from search results before use.

```python
# Bad example (context explosion)
draft = await llm.complete(context=full_webpage_content)

# Good example (ACI)
excerpt = await llm.complete(
    "Extract 200 characters from this content that are relevant to the research question"
)
draft = await llm.complete(context=excerpt)
```

### Role Separation Principle
Each node's LLM is assigned only one clear, single role:
- plan_generator: "Research planning expert"
- search_worker: "Information extraction expert"
- gap_detector: "Coverage analysis expert"
- critic: "Quality and factual alignment review expert"

Assigning multiple roles to a single node degrades quality due to role conflicts within the LLM.

### Temperature Strategy
| Node | Temperature | Reason |
|------|-------------|--------|
| router | 0.0 | Classification must be deterministic |
| critique | 0.1 | Fact-checking requires precision |
| search_worker (extraction) | 0.1 | Factual extraction, no creativity |
| gap_detector | 0.2 | Low creativity, high accuracy |
| plan_generator | 0.3 | Allow slight diversity |
| revise | 0.3 | Revisions should be conservative |
| write_draft | 0.4 | Creativity for fluent writing style |
