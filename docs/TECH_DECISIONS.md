# Technology Decision Rationale

Records the alternatives and reasons behind each technology decision.

---

## 1. LangGraph — Agent Orchestration

**Alternatives**: LangChain AgentExecutor, direct asyncio loop, AutoGen, CrewAI

**Reasons for choosing**:

| Criterion | LangGraph | Alternatives |
|------|-----------|------|
| Parallel fan-out | Send API — native support | Manual asyncio.gather() required |
| Mid-execution checkpoints | interrupt/Command — built-in | Manual state saving |
| State persistence | SQLite/Redis/Postgres options | Custom implementation |
| Streaming events | astream_events v2 — per-node events | Custom callback system |
| Cyclic graphs | revise → critique loop native | Not possible with DAG-only frameworks |

**Core reason**: The `interrupt/Command` API fits precisely for human-in-the-loop flows such as "plan approval". The framework handles the flow of the user modifying the plan and then resuming at the framework level.

---

## 2. FastAPI + SSE — API Server

**Alternatives**: Flask, Django, WebSocket-based streaming

**Reasons for choosing FastAPI**:
- Native `async/await` support (integrates naturally with LangGraph astream_events)
- Automatic request/response validation via Pydantic models
- Automatic OpenAPI documentation generation

**SSE vs WebSocket**:
- SSE: Unidirectional server → client, HTTP/1.1 compatible, auto-reconnect
- WebSocket: Bidirectional but higher overhead
- Research streaming is unidirectional, so SSE is more appropriate

**POST SSE chat**: Standard EventSource only supports GET, so chat implements POST SSE directly using `fetch` + `ReadableStream`.

---

## 3. Tavily — Web Search

**Alternatives**: SerpAPI, Google Custom Search, Bing API, DuckDuckGo, Brave Search

**Reasons for choosing**:
- **Research-specialized**: Superior coverage of academic, news, and technical documentation compared to general search APIs
- `search_depth="advanced"` option improves summary quality
- Returns `relevance_score` → enables dual filtering when combined with CRAG evaluator
- Official Python SDK + async (`asyncio.to_thread` wrapping)

**Async handling**: Since the Tavily SDK only provides a synchronous API, it is wrapped asynchronously with `asyncio.to_thread()`. Blocking I/O runs in a thread pool to prevent blocking the event loop.

---

## 4. Qdrant + fastembed — Local File Search

**Alternatives**: ChromaDB, Weaviate, Pinecone, OpenAI Embeddings + Faiss

**Reasons for choosing Qdrant**:
- Single binary, no external dependencies (local path mode)
- High-performance ANN search with Rust implementation
- Mature `qdrant-client` Python SDK

**Reasons for choosing fastembed**:
- Runs local embedding models without external APIs
- `BAAI/bge-small-en-v1.5` default model (130MB, fast)
- Can be combined with Ollama embed() to build a fully offline embedding pipeline

**Trade-off**: In qdrant-client 1.17.1, `add()`/`query()` methods are deprecated → named vector name conflict issue when using new `Document` API. Currently continuing with deprecated API using `warnings.catch_warnings()`. Migration needed in the future.

---

## 5. AWS Bedrock — Default LLM Backend

**Alternatives**: Anthropic API direct, Azure OpenAI, Vertex AI

**Reasons for choosing**:
- **Enterprise-friendly**: IAM Role-based authentication, VPC internal traffic
- **Data sovereignty**: Data does not leave AWS regions
- Latest Claude models available immediately via Inference Profile
- Same SDK interface maintained with `anthropic[bedrock]` package

**Claude vs GPT-4 choice**: Claude has an advantage in Korean language quality, long-context processing performance, and instruction-following ability — which is critical for research agents.

---

## 6. Ollama — Local LLM

**Alternatives**: llama.cpp direct, vLLM, Hugging Face Transformers

**Reasons for choosing**:
- **Installation simplicity**: Single binary, model management with `ollama pull <model>`
- OpenAI-compatible REST API → easy interface unification
- Official Python SDK (`ollama`) supports async completion, streaming, and embeddings
- Automatic detection of Apple Silicon (MPS) / NVIDIA CUDA

**Model selection**: `qwen3:14b` as default (as of 2026):
- Balanced Korean + English
- 14B parameters → sufficient speed on RTX 3080 and above
- Supports Thinking Mode (Extended Thinking)

---

## 7. AsyncSqliteSaver — Checkpointer

**Alternatives**: MemorySaver (in-memory), RedisSaver, PostgresSaver

**Reasons for choosing**:
- **No external dependencies**: No Redis/Postgres server required
- SQLite is a single file → simple deployment and migration
- Fully async with `aiosqlite`
- Official LangGraph package (`langgraph-checkpoint-sqlite`)

**Compared to MemorySaver**: Checkpoints are preserved on server restart. Even if the server restarts while in `interrupt` state, the user can reconnect to the report stream and resume execution.

---

## 8. DSAP Guard Functions — JSON Stability

**Alternatives**: Pydantic structured output, Instructor library, grammar-constrained sampling

**Reasons for choosing**:
- `Instructor` depends on Anthropic SDK → compatibility issues with Bedrock and Ollama
- Grammar-constrained sampling is only supported in Ollama (`format: "json"`)
- The DSAP approach (error context + retry) works identically across all LLM providers

**Implementation**: `utils/llm_json.py`
- On parse failure, adds error message + schema hint to conversation and retries
- On retry, replaces system prompt with strict `"ONLY valid JSON"` mode
- Returns fallback dictionary after a maximum of 2 retries

---

## 9. Next.js — Frontend

**Alternatives**: React + Vite, Vue.js, SvelteKit, plain HTML

**Reasons for choosing**:
- App Router: clear separation of server components and client components
- Native TypeScript support
- Report markdown rendering with `react-markdown` + `remark-gfm`
- EventSource / ReadableStream: browser-native APIs — no additional libraries needed

**State management**: No Redux or Zustand. `useState` + unidirectional state machine (`AppState` type) is sufficient. The research pipeline has a linear flow, so complex state management is unnecessary.

---

## 10. Sequential LLM Execution — Preventing Ollama Queue Overflow

**Problem**: When multiple LLM calls are run in parallel with `asyncio.gather()`, queue timeouts occur in Ollama.

**Cause**: Ollama processes requests serially on a single GPU. When 4–5 calls arrive concurrently, the internal queue saturates and some requests time out. Cloud LLMs (Bedrock, Claude) are not affected because they handle parallel processing through external scaling.

**Solution**: Changed `asyncio.gather()` → sequential `await` for section generation (`write_draft`) and section revision (`revise`).

```python
# Before (timeout in Ollama)
results = await asyncio.gather(*[_write_section(s) for s in sections])

# After (stable with all providers)
results = []
for section in sections:
    results.append(await _write_section(section))
```

**Trade-offs**:
- Cloud LLM: 5 sections × 2s = 10s (parallel) → 10s (sequential) — **no difference** (dominated by API latency)
- Ollama: 5 × 15s = 75s (sequential) — stable without timeouts
- The actual total time for deep+detailed is dominated by the number of searches per `depth`, so the impact of sequential section writing is minimal.

---

## 11. Independent Report Length Control — Section-by-Section Sequential Generation

**Problem**: Unable to generate long reports due to the `max_tokens` limit of a single LLM call (typically 4k–8k tokens).

**Solution**: Split the report into independent sections and generate each section with a separate LLM call.

| Mode | Calls | Expected length | Implementation |
|------|---------|-----------|------|
| `brief` | 1 | 2,000 max_tokens | Single full summary |
| `standard` | 3 | 7,000 max_tokens total | Summary + findings + conclusion |
| `detailed` | 3+N | 15,000+ max_tokens total | Per-sub-query analysis sections added |

**Independent design from `depth`**: Research depth (number of searches, gap detection passes) and report length are separate axes. Combinations such as `deep+brief` (broad research, short summary) and `fast+detailed` (quick research, detailed report) are all supported.

**State propagation path**: `plan_review_node` → `Command(update={"report_length": ...})` → LangGraph state → `write_draft()` branching. The user's selection is passed to the pipeline via LangGraph's interrupt/resume mechanism.

---

## Decision Summary Matrix

| Component | Choice | Core Reason |
|-----------|------|-----------|
| Agent framework | LangGraph | interrupt/Send API, checkpointer |
| API server | FastAPI | async native, SSE |
| Web search | Tavily | Research-specialized, relevance_score |
| Vector DB | Qdrant | No external dependencies, high performance |
| Embeddings | fastembed | Fully local, no API needed |
| LLM (cloud) | Bedrock | Enterprise, data sovereignty |
| LLM (local) | Ollama | Simple installation, model management |
| Checkpointer | AsyncSqliteSaver | Single file, restart recovery |
| JSON stability | DSAP Guard Functions | Provider-agnostic |
| Frontend | Next.js 14 | TypeScript, App Router |
| LLM parallelism | Sequential execution | Prevent Ollama queue overflow |
| Report length | Section-by-section sequential generation | Overcome max_tokens limit |
