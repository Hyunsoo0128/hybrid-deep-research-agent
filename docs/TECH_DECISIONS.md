# Technology Decision Rationale

Records the alternatives and reasons behind each technology decision, updated to reflect the Stage-Aware Local-Cloud Inference architecture.

---

## 1. LangGraph — Agent Orchestration

**Alternatives**: LangChain AgentExecutor, direct asyncio loop, AutoGen, CrewAI

**Reasons for choosing**:

| Criterion | LangGraph | Alternatives |
|-----------|-----------|--------------|
| Parallel fan-out | Send API — native support | Manual asyncio.gather() required |
| Mid-execution checkpoints | interrupt/Command — built-in | Manual state saving |
| State persistence | SQLite/Redis/Postgres options | Custom implementation |
| Streaming events | astream_events v2 — per-node events | Custom callback system |
| Cyclic graphs | revise → critique loop native | Not possible with DAG-only frameworks |

**Core reason**: The `interrupt/Command` API fits precisely for human-in-the-loop flows such as plan approval. The Send API enables the parallel fan-out pattern (N search workers) that is central to the pipeline's performance.

---

## 2. FastAPI + SSE — API Server

**Alternatives**: Flask, Django, WebSocket-based streaming

**Reasons for choosing FastAPI**:
- Native `async/await` support (integrates naturally with LangGraph astream_events)
- Automatic request/response validation via Pydantic models
- Automatic OpenAPI documentation generation

**SSE vs WebSocket**: SSE is unidirectional server → client, HTTP/1.1 compatible, auto-reconnect. Research streaming is unidirectional, so SSE is more appropriate. POST SSE chat implements POST SSE directly using `fetch` + `ReadableStream`.

---

## 3. Tavily — Web Search

**Alternatives**: SerpAPI, Google Custom Search, Bing API, DuckDuckGo, Brave Search

**Reasons for choosing**:
- Research-specialized: superior coverage of academic, news, and technical documentation
- `search_depth="advanced"` option improves summary quality
- Returns `relevance_score` → enables dual filtering when combined with CRAG evaluator
- Official Python SDK + async (`asyncio.to_thread` wrapping)

**Benchmark note**: All 35-condition experiments use Tavily web search via a frozen snapshot (7 sub-queries × 5 results = 35 documents per query). Performance on other retrieval backends may differ.

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
- Fully offline embedding pipeline — consistent with Privacy Boundary requirements

---

## 5. AWS Bedrock — Default Cloud LLM Backend

**Alternatives**: Anthropic API direct, Azure OpenAI, Vertex AI

**Reasons for choosing**:
- Enterprise-friendly: IAM Role-based authentication, VPC internal traffic
- Data sovereignty: data does not leave AWS regions
- Latest Claude models available immediately via Inference Profile
- Same SDK interface maintained with `anthropic[bedrock]` package

**Paper evaluation**: Sonnet 4.6, Haiku 4.5, and Llama 3.3 70B all evaluated via Bedrock. Sonnet 4.6 is the recommended cloud backend for maximum quality.

---

## 6. Ollama — Local LLM

**Alternatives**: llama.cpp direct, vLLM, Hugging Face Transformers

**Reasons for choosing**:
- Installation simplicity: single binary, model management with `ollama pull <model>`
- OpenAI-compatible REST API → easy interface unification
- Official Python SDK supports async completion, streaming, and embeddings
- Automatic detection of Apple Silicon (MPS) / NVIDIA CUDA

**Model selection** (from paper results):
- `exaone3.5:2.4b`: best quality/speed for hybrid (0.869 with Sonnet, 233s/query)
- `gemma3:4b`: alternative with comparable quality (0.867 with Sonnet, 312s/query)
- Q4_K_M quantization used for all models in the paper's evaluation

**Hardware**: NVIDIA L4 (24GB VRAM) used in paper. Broadly comparable to RTX 4090 (24GB VRAM) — best-performing configurations can run on commodity consumer hardware.

---

## 7. HybridProvider — System 1/System 2 Routing

**Design**: `HybridProvider(cloud, local)` wrapper implementing the same `LLMProvider` protocol as all other providers. Routes each node to the appropriate tier based on `node_hint`.

**Key design choice**: `partial()` injection at graph build time, not runtime dispatch.
- Zero changes to existing node code — each node receives a pre-resolved provider
- System 2 nodes (plan_elaboration, crag_recheck, synthesis, gap_detector) receive cloud provider
- System 1 nodes receive local provider
- Graceful degradation: `hasattr(llm, 'local')` checks fall through for non-hybrid providers

**Why this matters**: The routing is auditable — what reaches the cloud is determined by code, not by agent reasoning. This is the privacy enforcement point.

---

## 8. AsyncSqliteSaver — Checkpointer

**Alternatives**: MemorySaver (in-memory), RedisSaver, PostgresSaver

**Reasons for choosing**:
- No external dependencies: no Redis/Postgres server required
- SQLite is a single file → simple deployment and migration
- Fully async with `aiosqlite`
- Checkpoints preserved on server restart — users can reconnect to report stream and resume

---

## 9. DSAP Guard Functions — JSON Stability

**Alternatives**: Pydantic structured output, Instructor library, grammar-constrained sampling

**Reasons for choosing**:
- `Instructor` depends on Anthropic SDK → compatibility issues with Bedrock and Ollama
- Grammar-constrained sampling only supported in Ollama (`format: "json"`)
- DSAP approach (error context + retry) works identically across all LLM providers

**Critical for hybrid**: Local models (System 1) produce more malformed JSON than frontier models. DSAP Level 1+2 in `llm_json.py` is the reliability layer that makes System 1 local model calls production-viable.

---

## 10. Next.js — Frontend

**Alternatives**: React + Vite, Vue.js, SvelteKit, plain HTML

**Reasons for choosing**:
- App Router: clear separation of server components and client components
- Native TypeScript support
- Report markdown rendering with `react-markdown` + `remark-gfm`
- EventSource / ReadableStream: browser-native APIs — no additional libraries needed

---

## 11. Sequential LLM Execution — Preventing Ollama Queue Overflow

**Problem**: When multiple LLM calls are run in parallel with `asyncio.gather()`, queue timeouts occur in Ollama.

**Cause**: Ollama processes requests serially on a single GPU. When 4–5 calls arrive concurrently, the internal queue saturates and some requests time out.

**Solution**: Changed `asyncio.gather()` → sequential `await` for section generation and revision.

**Trade-offs**:
- Cloud LLM: no difference (dominated by API latency)
- Ollama: stable without timeouts
- Hybrid: local System 1 calls are sequential; cloud System 2 calls are independent

---

## 12. Frozen Retrieval Snapshot — Benchmark Reproducibility

**Decision**: All 35-condition experiments use a pre-collected frozen snapshot of Tavily search results (7 sub-queries × 5 results = 35 documents per query).

**Rationale**: Ensures fair comparison across conditions — all configurations see identical retrieved documents. Eliminates retrieval variance as a confound when comparing System 1/System 2 routing effects.

**Trade-off**: Performance on live search may differ from frozen snapshot results. The paper notes this as a limitation.

---

## Decision Summary Matrix

| Component | Choice | Core Reason |
|-----------|--------|-------------|
| Agent framework | LangGraph | interrupt/Send API, checkpointer |
| API server | FastAPI | async native, SSE |
| Web search | Tavily | Research-specialized, relevance_score |
| Vector DB | Qdrant | No external dependencies, high performance |
| Embeddings | fastembed | Fully local, Privacy Boundary compatible |
| LLM (cloud) | Bedrock | Enterprise, data sovereignty |
| LLM (local) | Ollama | Simple installation, Q4_K_M quantization |
| Hybrid routing | HybridProvider | Auditable privacy enforcement point |
| Checkpointer | AsyncSqliteSaver | Single file, restart recovery |
| JSON stability | DSAP Guard Functions | Provider-agnostic, critical for local models |
| Frontend | Next.js 14 | TypeScript, App Router |
| LLM parallelism | Sequential execution | Prevent Ollama queue overflow |
| Benchmark retrieval | Frozen snapshot | Reproducibility, fair comparison |
