# Local LLM Guide: Model Selection and Hardware Requirements

This guide helps you choose the right local LLM model and hardware for running the Deep Research Agent in Hybrid or Local-only mode.

## Pipeline Parallelism

The pipeline has several stages that issue concurrent local LLM calls. Understanding these is critical for hardware sizing.

### Concurrent Call Points

| Stage | Concurrent Calls | What Happens |
|-------|-----------------|--------------|
| **Search fan-out** | N (sub-queries) | LangGraph Send API dispatches N parallel `search_worker` nodes. Each worker calls the local LLM for CRAG scoring. Typical N = 4–6. |
| **MASS-RAG 3-agent** | 3 per sub-query | Summarizer, Extractor, and Reasoner run via `asyncio.gather`. All three call the local LLM simultaneously. |
| **CONSTRUCT scoring** | 2 | Document-level and field-level trust scoring run via `asyncio.gather`. |
| **Plan generator (Hybrid)** | 2 | Stage 1 extracts research profile (local) while preparing the strategy prompt. |

### Peak Concurrency

In the worst case (deep mode, 6 sub-queries, MASS-RAG enabled):

```
Search phase:  6 CRAG scoring calls (parallel via Send API)
MASS-RAG:      3 agent calls per sub-query (parallel via asyncio.gather)
               → but sub-queries are sequential, so peak = 3 concurrent calls
CONSTRUCT:     2 calls (parallel)
```

**Effective peak: 3–6 concurrent local LLM calls**, depending on the pipeline stage.

Ollama handles concurrent requests by queuing them when GPU memory is insufficient. With enough VRAM, it serves multiple requests truly in parallel. With insufficient VRAM, requests are serialized — the pipeline still works, just slower.

## Model Comparison

All models tested with Ollama on the Deep Research Agent pipeline. Scores are from the component benchmark (CRAG accuracy, MASS-RAG quality, JSON parse success rate).

### Recommended Models

| Model | Parameters | VRAM (Q4) | Speed (tok/s) | CRAG Accuracy | JSON Reliability | Best For |
|-------|-----------|-----------|---------------|---------------|-----------------|----------|
| **qwen3:4b** | 4B | ~3 GB | ~40 | ★★★ | ★★★★ | 16GB RAM, fast iteration |
| **qwen3:8b** | 8B | ~5 GB | ~25 | ★★★★ | ★★★★ | 24GB RAM, balanced |
| **qwen3:14b** | 14B | ~9 GB | ~15 | ★★★★★ | ★★★★★ | 32GB+ RAM, best quality |
| **qwen3:32b** | 32B | ~20 GB | ~8 | ★★★★★ | ★★★★★ | 64GB+ RAM, maximum quality |

### Alternative Models

| Model | Parameters | VRAM (Q4) | Notes |
|-------|-----------|-----------|-------|
| gemma3:4b | 4B | ~3 GB | Good general quality, shorter context window (8K) |
| gemma3:12b | 12B | ~8 GB | Strong reasoning, Google-trained |
| phi-4-mini | 3.8B | ~3 GB | Strong at classification/scoring tasks |
| llama3.2:3b | 3B | ~2 GB | Fastest option, lower quality |
| exaone3.5:7.8b | 7.8B | ~5 GB | Korean-specialized (LG AI Research) |

### Why Qwen3 is the Default

1. **Thinking mode control**: `/no_think` token disables extended reasoning, preventing 10–50× slowdowns in agentic workloads
2. **JSON reliability**: Qwen3 produces well-formed JSON more consistently than alternatives, reducing DSAP retry overhead
3. **Multilingual**: Strong Korean + English support for international research queries
4. **Apache 2.0 license**: No commercial restrictions

## Hardware Sizing Guide

### By Use Case

| Use Case | Min RAM | Recommended Model | Parallel Capability | Expected Latency |
|----------|---------|-------------------|--------------------|-----------------| 
| **Quick evaluation** | 16 GB | qwen3:4b | 2 concurrent | ~8 min/query |
| **Development & testing** | 32 GB | qwen3:14b | 3 concurrent | ~3 min/query |
| **Production hybrid** | 64 GB | qwen3:14b | 6 concurrent (full) | ~1.5 min/query |
| **Maximum local quality** | 64 GB+ | qwen3:32b | 3 concurrent | ~3 min/query |

### Detailed Hardware Breakdown

#### Apple Silicon Mac (Unified Memory)

| Config | Model | Parallel Slots | Notes |
|--------|-------|---------------|-------|
| M-series 16GB | qwen3:4b | 2 | OS uses ~5GB; model ~3GB; 2 KV caches fit |
| M-series 24GB | qwen3:8b | 2 | Comfortable for development |
| M-series 36GB | qwen3:14b | 3 | MASS-RAG 3-agent runs truly parallel |
| M-series 48GB | qwen3:14b | 6 | Full search fan-out parallel |
| M-series 64GB | qwen3:32b | 3 | Best quality with parallel MASS-RAG |

#### NVIDIA GPU Desktop

| GPU | VRAM | Model | Parallel Slots | Speedup vs Mac |
|-----|------|-------|---------------|----------------|
| RTX 4060 | 8 GB | qwen3:4b | 2 | ~3× |
| RTX 4070 | 12 GB | qwen3:8b | 2 | ~4× |
| RTX 4090 | 24 GB | qwen3:14b | 3–4 | ~8× |
| RTX 5090 | 32 GB | qwen3:14b | 6 (full) | ~10× |
| A100 | 40/80 GB | qwen3:32b | 6 (full) | ~15× |

NVIDIA GPUs are significantly faster than Apple Silicon for LLM inference due to higher memory bandwidth and CUDA optimization.

#### System RAM vs GPU VRAM

- **Apple Silicon**: Unified memory — GPU and CPU share the same pool. The numbers above account for OS overhead (~5GB).
- **NVIDIA**: Model runs in GPU VRAM. System RAM (32GB+ recommended) is used for the Python process, search results, and document processing. System RAM does not affect LLM inference speed.

### Ollama Parallel Configuration

Ollama's parallel request handling can be tuned:

```bash
# Set maximum parallel model instances (default: 1)
OLLAMA_NUM_PARALLEL=3 ollama serve

# Or set via environment variable
export OLLAMA_NUM_PARALLEL=3
ollama serve
```

**`OLLAMA_NUM_PARALLEL` determines how many concurrent requests Ollama serves.** Each parallel slot requires additional VRAM for KV cache (~0.5–1.5 GB per slot depending on model size and context length).

Formula for VRAM estimation:
```
Total VRAM ≈ Model size (Q4) + (KV cache per slot × OLLAMA_NUM_PARALLEL) + overhead
```

Example for qwen3:14b with 3 parallel slots:
```
9 GB (model) + (1.2 GB × 3 slots) + 1 GB (overhead) ≈ 13.6 GB
→ Fits in 16GB VRAM, comfortable in 24GB
```

## Benchmark: Model Size vs Pipeline Quality

From our component benchmark (CRAG + MASS-RAG + AlignRAG combined):

| Model | Overall Score | Cost/Query (Hybrid) | Quality/$ |
|-------|--------------|--------------------|-----------| 
| qwen3:4b (local only) | 0.65 | $0.00 | — |
| qwen3:8b (local only) | 0.72 | $0.00 | — |
| qwen3:8b (hybrid) | 0.89 | $0.25 | 3.56 |
| qwen3:14b (hybrid) | 0.91 | $0.25 | 3.64 |
| Bedrock Sonnet (cloud only) | 0.93 | $0.75 | 1.24 |

**Key insight**: The quality gap between qwen3:8b and qwen3:14b in hybrid mode is small (0.89 vs 0.91) because Bedrock handles the quality-critical generation tasks. The local model primarily performs classification and drafting where the 8B model is sufficient.

## Recommendations

1. **Start with qwen3:4b on 16GB** — verify the pipeline works, then upgrade the model
2. **For hybrid mode, model quality matters less** — Bedrock compensates for local model limitations
3. **For local-only mode, use the largest model your hardware supports** — no cloud fallback
4. **Set `OLLAMA_NUM_PARALLEL` to match your VRAM budget** — more parallelism = faster pipeline
5. **Monitor with `ollama ps`** — shows loaded models and VRAM usage in real time
