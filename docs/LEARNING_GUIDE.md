# Multi-Agent Deep Research — Learning Guide

This guide is designed for **reading this project's code like a textbook**. Each design decision is backed by a research paper, and each paper has a corresponding implementation in this codebase.

The goal is to answer: "What do you need to know to build Stage-Aware Local-Cloud Inference properly?"

> This codebase implements the architecture from the paper:
> **Stage-Aware Local-Cloud Inference: Hybrid Pipelines Consistently Outperform Matched Cloud-Only Baselines**
>
> The core insight: not all pipeline stages require frontier-model reasoning. System 1 stages (bounded-context operations) run locally on 2–4B models; System 2 stages (integrative reasoning) run on frontier cloud LLMs. See [HYBRID_STRATEGY.md](HYBRID_STRATEGY.md) for the full design.

---

## Table of Contents

1. [Why Multi-Agent? — The Limits of Naive RAG](#1-why-multi-agent)
2. [Architecture Patterns: Agentic RAG vs Speculative RAG](#2-agentic-rag-patterns)
3. [Paper 1: Dimension-Based Query Decomposition](#3-query-decomposition)
4. [Paper 2: CRAG — Self-Correcting Retrieval](#4-crag--corrective-rag)
5. [Paper 3: AlignRAG — Citation-Claim Alignment Verification](#5-alignrag--factual-alignment)
6. [Paper 4: DSAP Guard Functions — Reliable LLM JSON](#6-dsap-guard-functions)
7. [Implementing Cycles, Parallelism, and Interrupts with LangGraph](#7-langgraph-design-patterns)
8. [State Design — ResearchState TypedDict](#8-state-design)
9. [LLM Provider Abstraction — Protocol Pattern](#9-llm-provider-abstraction)
10. [Real-World Issues with Local LLMs](#10-local-llm-operational-issues)
11. [Learning Path — What to Read and in What Order](#11-learning-path)

---

## 1. Why Multi-Agent?

### Structural Limitations of Naive RAG

```
Naive RAG:   query → vector search → top-K documents → LLM → answer
```

This structure has three fundamental problems:

**Problem 1: Coverage Limits of a Single Retrieval**
When a query like "What is the current state and limitations of quantum computing?" is embedded as a vector, only the documents nearest to that vector in embedding space are retrieved. But "what is quantum computing" (definition) and "2025 quantum computer error rates" (current data) are far apart in embedding space. A single query cannot cover both clusters simultaneously.

**Problem 2: Retrieved Documents May Be Irrelevant**
Keywords can match while intent differs. Searching "quantum computing" also returns documents on quantum cryptography, quantum sensors, and quantum communication. This noise in the LLM context contaminates the answer.

**Problem 3: No Verification After Retrieval**
There is no way to catch an LLM "exaggerating" or "freely inferring" from sources. Even if a source says "approximately 80% success rate," the LLM may write "100% success rate" in its report.

### How Agentic RAG Solves This

```
Agentic RAG:
  query decomposition (5 dimensions) → parallel retrieval
                                      → CRAG evaluation (relevance filtering)
                                      → gap detection (coverage analysis)
                                      → additional retrieval (filling gaps)
                                      → cross-validation (conflict detection)
                                      → draft report writing
                                      → AlignRAG critique (claim-source alignment)
                                      → revision
```

Each step compensates for the failure of the previous one. Let's walk through this structure paper by paper.

---

## 2. Agentic RAG Patterns

### RAG Architecture Evolution

```
Generation 1 — Naive RAG (2020~2022)
  Simple vector search + LLM answer
  Problems: single retrieval, no fact verification

Generation 2 — Advanced RAG (2022~2023)
  Query rewriting + reranking + LLM answer
  Improvement: retrieval quality↑, speed↓

Generation 3 — Agentic RAG (2023~present)
  Autonomous multi-step: decompose → retrieve → gap → verify → write → critique
  The pattern this project implements
```

**Key Difference**: In Agentic RAG, the LLM does not merely summarize retrieval results — it **autonomously decides what information is missing** and triggers additional retrieval. This requires a cycle (loop) in the graph.

### References

- [A Survey of Retrieval-Augmented Generation for Large Language Models](https://arxiv.org/abs/2312.10997) — Overview survey of the full RAG landscape
- [Agentic RAG: A Survey on Retrieval-Augmented Generation with LLM-as-an-Agent](https://arxiv.org/abs/2501.09136) — Classification of Agentic RAG patterns
- [Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting](https://arxiv.org/abs/2407.08223) — Reducing latency with parallel drafting

---

## 3. Query Decomposition

### Paper

**"Multi-Dimensional Query Decomposition for Complex Question Answering"**
[arxiv:2507.00355](https://arxiv.org/abs/2507.00355)
- **Results**: MRR@10 +36.7%, F1 +11.6%

### Core Problem

When a complex question is searched as a single query, only documents near the query vector in embedding space are retrieved. These documents are semantically similar to each other, meaning **information from other angles is not covered**.

"Simply split the query into N sub-queries" is not a solution — the N sub-queries may all point to the same document cluster.

### The Paper's Idea

By **explicitly specifying decomposition dimensions**, each sub-query can be forced to point to a semantically distinct document cluster:

| Dimension | Target Document Cluster | Example Sub-Query |
|-----------|------------------------|-------------------|
| Definition/Background | Concept explanations, historical documents | "qubit definition and history" |
| Current State/Evidence | Latest statistics, case studies | "2025 quantum computer performance metrics" |
| Comparison/Alternative | Competing technology comparisons | "quantum vs classical computing differences" |
| Mechanism/Cause | Technical deep-dive documents | "decoherence mechanism" |
| Limitations/Challenges | Critical analysis | "commercialization barriers and cost issues" |

5 sub-queries covering 5 distinct document clusters → significant improvement in overall recall.

### Implementation

```python
# src/nodes/plan_generator.py

_DECOMPOSITION_DIMS = """
- definition/background: concepts, history, overview documents
- current state/evidence: latest statistics, case studies, data
- comparison/alternative: competing technologies, trade-offs
- mechanism/cause: technical deep-dives, how-it-works
- limitations/challenges: critical analysis, barriers, open problems
"""

_PROMPT = """
You are a research planning expert. Decompose the following query into
sub-queries, one per dimension listed below. Each sub-query must target
a different semantic region in the document space.

Dimensions:
{decomposition_dims}

Query: {query}

Return JSON:
{{"queries": [{{"text": "...", "dimension": "..."}}]}}
"""
```

The `dimension` field is the key. It gives the LLM the context that "this sub-query must point to documents of this dimension" when generating each sub-query.

### Fan-Out in LangGraph

```python
# src/graph.py
def spawn_search_workers(state: ResearchState):
    return [Send("search_worker", {"query": q, "dimension": d})
            for q, d in zip(state["queries"], state["dimensions"])]

graph.add_conditional_edges("generate_plan", spawn_search_workers)
```

The `Send` API spawns N `search_worker` nodes in parallel. When all complete, results are automatically merged into the `citations` list.

---

## 4. CRAG — Corrective RAG

### Paper

**"Corrective Retrieval Augmented Generation"**
[arxiv:2401.15884](https://arxiv.org/abs/2401.15884)
Yan Shi et al., 2024

### Core Problem

Using documents returned by a search API as-is causes two problems:
1. **Keyword matching errors**: documents where the words match but the meaning differs
2. **Context explosion**: passing all documents to the LLM causes token cost explosion and irrelevant content contaminates the answer

### The Paper's Idea

*Retrieval Evaluator* — an LLM evaluates the relevance of search results and classifies them into three categories:

```
"relevant"    → aligned with research intent, use full content
"partial"     → only partially related, use summary only
"irrelevant"  → unrelated, skip (fallback: apply lower weight)
```

The original paper performs a **corrective search** (re-retrieval via web search) when results are "irrelevant." In this project, the gap detection node (`gap_detector`) handles this role, so CRAG focuses on "evaluation + filtering."

### Implementation

```python
# src/nodes/search_worker.py

async def _evaluate_relevance(query: str, results: list, llm) -> list:
    """LLM batch-evaluates all search results in a single call"""
    evaluation_prompt = f"""
    Research query: {query}

    For each search result below, classify its relevance:
    - "relevant": directly answers the query
    - "partial": tangentially related
    - "irrelevant": unrelated

    Results:
    {format_results(results)}

    Return JSON: {{"evaluations": [{{"index": 0, "relevance": "relevant"}}]}}
    """
    evaluations = await llm_json(llm, evaluation_prompt)

    scored = []
    for i, result in enumerate(results):
        rel = evaluations[i]["relevance"]
        if rel == "relevant":
            content = await fetch_full_content(result["url"])
            scored.append({**result, "content": content, "confidence": 1.0})
        elif rel == "partial":
            scored.append({**result, "content": result["summary"], "confidence": 0.6})
        # "irrelevant" → skip
    return scored
```

**Dual filtering effect**:
- Tavily `relevance_score ≥ 0.7`: keyword-matching-based first filter (fast, free)
- CRAG LLM evaluation: semantic relevance to research intent, second filter (slow, precise)

Only documents passing both filters get full fetch → simultaneously reduces LLM cost and context noise.

---

## 5. AlignRAG — Factual Alignment

### Paper

**"AlignRAG: Aligning the Retrieval-Augmented Generation with Critiques on Factual Alignment"**
[arxiv:2504.14858](https://arxiv.org/abs/2504.14858)
- **Results**: 8B critic achieves +12.1% accuracy over 72B generation model (out-of-domain)

### Core Problem

LLMs do not faithfully reproduce source content. Hallucinations occur in three patterns:

1. **Amplification**: source says "approximately 80% success" → report says "100% success rate achieved"
2. **Unsupported inference**: causal relationships not present in any source are added
3. **Unit/numerical errors**: "100 million dollars" → "1 billion dollars", "2024" → "2025"

Typical quality review ("is it well-written?", "is it clear?") does not detect these errors, because the sentences are fluent and appear reasonable.

### The Paper's Idea

**Factual Alignment Check**: provide both the report draft and citation source excerpts *together* to a critic LLM, and **explicitly** verify whether each claim is supported by the source.

Key finding: a small critic model (8B) outperforms a large generation model (72B) at detecting alignment errors. The critique task requires precise comparison, not creativity.

### Implementation

```python
# src/nodes/critic.py

async def critique(state: ResearchState) -> dict:
    draft = state["draft_report"]
    citations = state["citations"]

    # Include source excerpts as context — this is the core of AlignRAG
    source_excerpts = "\n\n".join([
        f"[Source {i+1}] {c['url']}\n{c['excerpt']}"
        for i, c in enumerate(citations)
    ])

    prompt = f"""
    Review this research report for factual alignment with its sources.

    REPORT:
    {draft}

    SOURCE EXCERPTS:
    {source_excerpts}

    Check for:
    1. Claims that overstate what sources say
    2. Inferences not supported by any source
    3. Numerical errors (units, magnitudes, dates)

    Return JSON:
    {{
      "passed": true/false,
      "overall_quality": 1-10,
      "misaligned_claims": [
        "Report: '...' → Source says: '...'"
      ],
      "suggestions": ["..."]
    }}
    """

    result = await llm_json(llm, prompt)
    return {"critique_result": result}
```

`misaligned_claims` is passed to the `revise()` node where each claim is corrected to match the source:

```python
# src/nodes/critic.py

async def revise(state: ResearchState) -> dict:
    critique = state["critique_result"]
    misaligned = "\n".join(critique.get("misaligned_claims", []))

    revision_prompt = f"""
    Revise the report to correct these factual misalignments:
    {misaligned}

    Keep all other content unchanged.
    """
    # Sequential section-by-section revision (prevents Ollama queue overflow)
    revised_sections = []
    for section in raw_sections:
        revised_sections.append(await _revise_section(section, revision_prompt, llm))
    ...
```

---

## 6. DSAP Guard Functions

### Paper

**"Dynamic Schema-Aware Prompting for Reliable LLM JSON Output"**
[arxiv:2512.20660](https://arxiv.org/abs/2512.20660)
- **Results**: JSON parsing reliability +20~66 percentage points

### Core Problem

Every node in the pipeline parses LLM output as JSON. A parsing failure = node failure = pipeline halt. In particular:
- qwen3 Thinking Mode: `<think>...</think>` blocks interrupt the JSON output
- Nested schemas: LLMs tend to copy example JSON verbatim
- Special characters: escaping errors

Simple `try/except` + fallback **silently swallows** errors. Even if a node returns empty results, the pipeline continues running, and the final report is completed with empty citations.

### The Paper's Idea

**Feed back** parsing error information to the LLM so it self-corrects. Adding the error message and schema to the conversation tells the LLM what to fix, dramatically increasing the success rate on retry.

### Implementation

```python
# src/utils/llm_json.py

async def llm_json(
    llm,
    prompt: str,
    system: str = "",
    fallback: dict = None,
    max_retries: int = 3
) -> dict:
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        raw = await llm.complete(messages, system=system)

        # Remove qwen3 Thinking Mode tags
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)

        # Extract JSON block (handles ```json ... ``` markdown)
        raw = extract_json_block(raw)

        try:
            return json.loads(raw)

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                # Error feedback + schema hint added to conversation — the core of DSAP
                error_feedback = f"""
                Your previous response failed JSON parsing: {str(e)}
                Raw response was: {raw[:200]}...

                Please return ONLY valid JSON matching this schema:
                {json.dumps(fallback or {}, indent=2)}

                No markdown, no explanations, no <think> tags.
                """
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": error_feedback})

                # 2nd attempt: stricter system prompt
                if attempt == 1:
                    system = "ONLY valid JSON. No markdown. No explanations."

    # Final failure: safe default per node
    return fallback or {}
```

This function is applied identically across all 6 nodes (`plan_generator`, `search_worker`, `gap_detector`, `cross_validator`, `critic`, `router`).

---

## 7. LangGraph Design Patterns

### Why LangGraph?

The deep research pipeline requires three patterns that DAG (directed acyclic graph)-based frameworks cannot handle:

**Pattern 1: Cycle**
```
critique → passed?
              No  → revise → critique  (loop)
              Yes → done
```

**Pattern 2: Conditional Fanout**
```
generate_plan → [Send("search_worker", q1),
                 Send("search_worker", q2),   (dynamic parallel spawn)
                 Send("search_worker", q3)]
```

**Pattern 3: Interrupt/Resume**
```
generate_plan → INTERRUPT
               [user reviews and edits the plan]
               → Command(goto="search_orchestrator", update={...})
```

DAG-based frameworks (LangChain LCEL, etc.) do not support all three of these patterns.

### Graph Assembly

```python
# src/graph.py

from langgraph.graph import StateGraph, END
from langgraph.types import Send, Command, interrupt

graph = StateGraph(ResearchState)

# Register nodes
graph.add_node("generate_plan", generate_plan)
graph.add_node("plan_review", plan_review_node)  # interrupt
graph.add_node("search_worker", search_worker)   # parallel spawn
graph.add_node("gap_detector", gap_detector)
graph.add_node("write_draft", write_draft)
graph.add_node("critique", critique)
graph.add_node("revise", revise)

# Define edges
graph.add_edge(START, "generate_plan")
graph.add_edge("generate_plan", "plan_review")

# Conditional fanout: plan_review → N search_workers
graph.add_conditional_edges("plan_review", spawn_search_workers)

# Conditional loop: critique → revise or done
graph.add_conditional_edges("critique", should_revise,
    {"revise": "revise", "done": END})
graph.add_edge("revise", "critique")  # completes the cycle

# Checkpointer (recovery after server restart)
checkpointer = AsyncSqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(checkpointer=checkpointer,
                    interrupt_before=["plan_review"])
```

### Interrupt/Resume Pattern

```python
# plan_review_node — how to wait for user input
async def plan_review_node(state: ResearchState):
    # Graph is suspended here
    user_response = interrupt({"plan": state["plan"]})

    # Resumes when user response arrives
    return Command(
        update={
            "plan": user_response.get("plan", state["plan"]),
            "plan_approved": True,
            "report_length": user_response.get("report_length", "detailed"),
        },
        goto="search_orchestrator",
    )
```

In FastAPI:
```python
# src/main.py

@app.post("/research/approve")
async def approve_plan(req: ApproveRequest):
    # Resume the suspended graph with the user's response
    await graph_app.ainvoke(
        Command(resume={"approved": req.approved,
                        "plan": req.plan,
                        "report_length": req.report_length}),
        config={"configurable": {"thread_id": req.session_id}}
    )
```

---

## 8. State Design

### ResearchState TypedDict

```python
# src/state.py

class ResearchState(TypedDict):
    # Input
    session_id: str
    query: str
    depth: str               # "fast" | "normal" | "deep"
    report_length: str       # "brief" | "standard" | "detailed"
    local_search_enabled: bool

    # Plan
    plan: dict               # {"title": ..., "queries": [...]}
    plan_approved: bool

    # Search results
    citations: list[dict]    # [{url, title, excerpt, confidence, ...}]
    gap_search_round: int    # multi-round gap detection counter

    # Cross-validation
    validation_summary: str

    # Report
    draft_report: str
    final_report: str

    # Critique
    critique_result: dict    # {passed, misaligned_claims, suggestions}
    revision_count: int

    # Chat
    chat_history: list[dict]
```

**Design principles**:
- Shared state read and written by all nodes — clear inter-node interface
- `gap_search_round` controls multi-round loop (deep mode: max 2)
- `revision_count` controls critique/revise loop (upper limit per depth)
- LangGraph checkpointer automatically saves this state to SQLite

---

## 9. LLM Provider Abstraction

### Protocol Pattern

Three LLM backends (Bedrock, Claude, Ollama) are used through a uniform interface:

```python
# src/providers/base.py

class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str: ...

    async def stream(
        self,
        messages: list[dict],
        system: str = "",
    ) -> AsyncIterator[str]: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

Python `Protocol` uses structural typing rather than inheritance. A new provider only needs to implement these 3 methods — no modifications to existing code are required.

**System Prompt Handling Differences**:

| Provider | How the `system` parameter is passed |
|----------|--------------------------------------|
| Anthropic Claude | `system=` parameter (API-level support) |
| AWS Bedrock | `system=` parameter (same) |
| Ollama | Inserted as `{"role": "system", "content": ...}` in the messages array |

```python
# src/providers/ollama.py

async def complete(self, messages, system="", **kwargs):
    full_messages = messages
    if system:
        full_messages = [{"role": "system", "content": system}] + messages
    response = await self.client.chat(
        model=self.model,
        messages=full_messages,
        ...
    )
```

How to add a new provider: [docs/ADDING_LLM_PROVIDER.md](ADDING_LLM_PROVIDER.md)

---

## 10. Local LLM Operational Issues

Real-world problems and solutions that arise when using local LLMs in a production pipeline.

### Issue 1: Thinking Mode Tag Contamination

qwen3 outputs its internal reasoning in a `<think>` block before responding. This block interferes with JSON parsing.

```
<think>
I need to decompose the query. First, considering the definition/background dimension...
</think>
{"queries": [...]}   ← this JSON needs to be parsed
```

**Solution**: Remove `<think>` blocks in `llm_json()` before parsing:
```python
raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
```

### Issue 2: Single GPU Queue Overflow

Sending 5 concurrent LLM calls with `asyncio.gather()` saturates Ollama's internal queue, causing timeouts.

**Solution**: Switch from parallel to sequential execution for section generation and revision:
```python
# Before (timeouts)
results = await asyncio.gather(*[_write_section(s) for s in sections])

# After (stable)
results = []
for section in sections:
    results.append(await _write_section(section))
```

Cloud LLMs scale externally, so this change has no performance impact for them.

### Issue 3: Complex Schema Template Copying

qwen3 has a tendency to return the example JSON from the prompt verbatim instead of actual data.

**Solution**: During DSAP retry, explicitly state "return actual data, not the example" and minimize example JSON in prompts.

### Measured Performance Summary

| Setup | Overall Score | Latency per Call | Suitable Scenario |
|-------|--------------|------------------|-------------------|
| qwen3:8b + Ollama | 74.9% | 21.5 sec | Privacy, cost-free development |
| Claude Sonnet 4.6 | ~97% | ~2-4 sec | Production quality |

Detailed results: [docs/BENCHMARK.md](BENCHMARK.md)

---

## 11. Learning Path

### For Newcomers (Starting from RAG Basics)

1. `src/state.py` — understand the data structures of the full pipeline
2. `src/graph.py` — understand how nodes connect
3. `src/nodes/plan_generator.py` — the simplest node, Query Decomposition implementation
4. `src/utils/llm_json.py` — DSAP Guard Functions, used by every node
5. [arxiv:2507.00355](https://arxiv.org/abs/2507.00355) — Query Decomposition paper
6. `src/nodes/search_worker.py` — CRAG evaluator implementation

### For Learning Multi-Agent Design

1. [docs/ARCHITECTURE.md](ARCHITECTURE.md) — full LangGraph graph structure
2. [docs/TECH_DECISIONS.md](TECH_DECISIONS.md) — rationale for each technical decision
3. [LangGraph Official Docs — Cycles and Branching](https://langchain-ai.github.io/langgraph/concepts/)
4. `src/graph.py` — `spawn_search_workers`, `should_continue_research` conditional edges

### Learning via Paper → Code Mapping

| Paper to Read | Code to Read Next |
|---------------|-------------------|
| arxiv:2507.00355 (Query Decomp) | `src/nodes/plan_generator.py` |
| arxiv:2401.15884 (CRAG) | `src/nodes/search_worker.py` — `_evaluate_relevance()` |
| arxiv:2504.14858 (AlignRAG) | `src/nodes/critic.py` — `critique()`, `revise()` |
| arxiv:2512.20660 (DSAP) | `src/utils/llm_json.py` |
| arxiv:2407.08223 (Speculative RAG) | `src/graph.py` — `Send` API parallel fanout |

### Local LLM Experimentation

1. [docs/BENCHMARK.md](BENCHMARK.md) — review measured results
2. `eval/standalone_benchmark.py` — run it directly to measure on your own environment
3. `src/providers/ollama.py` — how the Ollama provider is implemented
4. [docs/ADDING_LLM_PROVIDER.md](ADDING_LLM_PROVIDER.md) — connecting a new model/provider

---

## Additional References

### Key Papers

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — the original RAG paper (Lewis et al., 2020)
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) — self-critique RAG
- [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) — hypothetical document embeddings

### Framework Documentation

- [LangGraph Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/) — cycles, fanout, checkpointer
- [LangGraph How-To: Human-in-the-loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/) — interrupt/resume implementation
- [Tavily Search API Docs](https://docs.tavily.com) — search_depth, relevance_score
- [Qdrant Documentation](https://qdrant.tech/documentation/) — vector search, local mode
