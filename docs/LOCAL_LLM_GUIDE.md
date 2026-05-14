# Local LLM Guide: Model Selection and Hardware Requirements

This guide helps you choose the right local LLM model and hardware for running the Deep Research Agent in Hybrid or All-local mode, based on the paper's 35-condition evaluation results.

---

## Recommended Models (from Paper Results)

The paper evaluates 8 local models across all conditions. Results are for Hybrid+Sonnet 4.6 configuration (N=600 per condition):

| Model | Size | Quality (Med) | Latency/q | VRAM (Q4) | Recommendation |
|-------|------|---------------|-----------|-----------|----------------|
| **exaone3.5:2.4b** | ~2.4B | **0.869** | 233s | ~3GB | **Best overall** |
| **gemma3:4b** | ~4B | 0.867 | 312s | ~4GB | **Best alternative** |
| qwen3:4b | ~4B | 0.801 | 471s | ~4GB | Good, slower |
| llama3.2:3b | ~3B | 0.781 | 187s | ~2GB | Fastest, lower quality |
| qwen3:8b | ~8B | 0.855 | 590s | ~6GB | Good, but slower than 2.4B |
| exaone3.5:7.8b | ~7.8B | 0.845 | 382s | ~6GB | Good |
| llama3.1:8b | ~8B | 0.840 | 317s | ~6GB | Good |
| gemma3:12b | ~12B | 0.838 | 525s | ~9GB | Larger, not better |

**Key finding**: Performance does not increase monotonically with model size. exaone3.5:2.4b (0.869) and gemma3:4b (0.867) exceed all 8B and 12B variants, with faster inference.

---

## Quick Start

```bash
# Install recommended model
ollama pull exaone3.5:2.4b      # best quality/speed
ollama pull nomic-embed-text    # for local file search

# Configure
echo "LLM_PROVIDER=hybrid
HYBRID_CLOUD_PROVIDER=bedrock
HYBRID_LOCAL_PROVIDER=ollama
HYBRID_LOCAL_MODEL=exaone3.5:2.4b" >> .env
```

---

## All-Local Mode (Zero Cloud Cost)

For air-gapped deployments or maximum privacy:

| Model | Quality | Cost/q | vs Cloud-only Sonnet |
|-------|---------|--------|----------------------|
| exaone3.5:2.4b | 0.802 | $0.000 | +0.004 |
| gemma3:4b | 0.803 | $0.000 | +0.005 |
| qwen3:8b | 0.814 | $0.000 | +0.017 |

All-local exaone3.5:2.4b (0.802) exceeds cloud-only Haiku (0.671) and cloud-only Llama 70B (0.688).

```bash
ollama pull exaone3.5:2.4b
```

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=exaone3.5:2.4b
```

---

## Hardware Requirements

### Minimum (2–4B models)

| Hardware | Model | Mode | Expected Quality |
|----------|-------|------|-----------------|
| 8GB VRAM GPU | exaone3.5:2.4b | Hybrid+Sonnet | 0.869 |
| 8GB VRAM GPU | gemma3:4b | Hybrid+Sonnet | 0.867 |
| 16GB RAM (CPU) | exaone3.5:2.4b | Hybrid+Sonnet | 0.869 (slower) |

### Recommended (paper's hardware)

- **NVIDIA L4 (24GB VRAM)** — used in all paper experiments
- Broadly comparable to **RTX 4090 (24GB VRAM)**
- Ubuntu 22.04, Ollama with Q4_K_M quantization

### Apple Silicon

| Config | Model | Mode |
|--------|-------|------|
| M-series 16GB | exaone3.5:2.4b | Hybrid or All-local |
| M-series 24GB | gemma3:4b | Hybrid or All-local |
| M-series 36GB+ | qwen3:8b | Hybrid or All-local |

---

## Ollama Setup

### Install

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

### Pull models

```bash
# Best hybrid local model (paper result)
ollama pull exaone3.5:2.4b

# Alternative
ollama pull gemma3:4b

# For local file search (embeddings)
ollama pull nomic-embed-text
```

### Parallel configuration

The pipeline issues concurrent local LLM calls (CRAG scoring, MASS-RAG 3-agent). Set `OLLAMA_NUM_PARALLEL` to match your VRAM:

```bash
# Set maximum parallel model instances
OLLAMA_NUM_PARALLEL=4 ollama serve
```

| Model | VRAM/slot | NUM_PARALLEL (24GB) |
|-------|-----------|---------------------|
| exaone3.5:2.4b | ~1GB | 8+ |
| gemma3:4b | ~2GB | 6 |
| qwen3:4b | ~2GB | 6 |
| qwen3:8b | ~3GB | 4 |
| gemma3:12b | ~5GB | 3 |

---

## Model Family Notes

### EXAONE 3.5 (LG AI Research)
- exaone3.5:2.4b: **Best overall** in paper (0.869 hybrid, 0.802 all-local)
- Strong instruction following, reliable JSON output
- Fast inference (233s/query with Sonnet)

### Gemma 3 (Google DeepMind)
- gemma3:4b: **Best alternative** (0.867 hybrid, 0.803 all-local)
- Strong multilingual capability
- Slightly slower than exaone3.5:2.4b (312s/query)

### Qwen 3 (Alibaba)
- qwen3:4b: Good quality (0.801 hybrid) but slower (471s/query)
- qwen3:8b: Better quality (0.855 hybrid) but much slower (590s/query)
- Note: qwen3:4b all-local shows bimodal distribution — not recommended for all-local

### Llama 3 (Meta)
- llama3.2:3b: Fastest (187s/query) but lower quality (0.781 hybrid)
- llama3.1:8b: Good quality (0.840 hybrid), reasonable speed (317s/query)

---

## Troubleshooting

**Slow inference**: Check `ollama ps` to see if the model is loaded in GPU VRAM. If running on CPU, inference will be 5–10× slower.

**JSON parse failures**: DSAP guard functions in `llm_json.py` handle retries automatically. If failures are frequent, try a different model (exaone3.5:2.4b and gemma3:4b have the most reliable JSON output).

**Out of memory**: Reduce `OLLAMA_NUM_PARALLEL` or switch to a smaller model. The pipeline works with NUM_PARALLEL=1 (sequential), just slower.

**Ollama timeouts**: Increase timeout in `src/providers/ollama.py`:
```python
self._client = AsyncClient(host=_host, timeout=600)  # 10 minutes
```
