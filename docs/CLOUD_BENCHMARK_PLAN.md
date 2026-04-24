# Cloud Benchmark Plan — Local LLM Candidate Evaluation

**Purpose**: Bulk evaluation of small LLM candidate models on EC2 GPU instances to fill the "local" role in the Hybrid pipeline.
**Background**: The current baseline model `qwen3:8b` FAILs on 4 techniques: stride/crag/sdp/mass_rag.
Determine whether better candidates exist, and where the size-performance trade-off optimum lies.

---

## 1. Evaluation Premises

- **No changes to production code** — only the benchmark environment moves to EC2
- **Role of the local LLM** (per the Hybrid pipeline):
  - Spec RAG Stage 1 (drafting suspect claims) — core of AlignRAG recall
  - Spec RAG Stage 3 (applying corrections and refining)
  - CRAG relevance classification (3-way: relevant / partial / irrelevant)
  - MASS-RAG expert persona synthesis
  - STRIDE Supervisor routing decisions (requires JSON accuracy)
  - DSAP JSON retry convergence speed
- **Cloud LLM fixed**: Bedrock Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`)

---

## 2. EC2 Environment

### Instance Selection

The pipeline `search_worker` fan-out runs 5–7 sub-queries concurrently.
`OLLAMA_NUM_PARALLEL` must match the number of sub-queries for true parallel processing;
required VRAM = model size × number of sub-queries by this criterion.

| Instance | GPU | VRAM | 8B NUM_PARALLEL | 14B NUM_PARALLEL | $/hr |
|---------|-----|------|----------------|-----------------|------|
| g6.xlarge | L4 | 24GB | 4 (insufficient) | 2 (insufficient) | $0.80 |
| **`g6e.xlarge`** | **L40S** | **48GB** | **8 (full coverage)** | **5 (full coverage)** | **~$2.0** |
| g5.12xlarge | 4×A10G | 96GB | multi-model concurrent | multi-model concurrent | $5.67 |

**Selection**: `g6e.xlarge` (us-west-2)
- L40S 48GB — truly parallel processing for the full sub-query fan-out
- 8B models: NUM_PARALLEL=8, 14B models: NUM_PARALLEL=5
- Speed-first choice suited for many benchmark iterations

### Model Storage Strategy

Pre-pull all candidate models to the EBS disk; load only the target model into VRAM at benchmark time.
- Total disk space for all models: ~50GB → EBS 100GB is sufficient
- Only one model is loaded into VRAM at a time (Ollama auto-unloads)

```bash
# Pre-download all candidate models on EC2 (background)
for model in llama3.2:3b qwen3:4b phi4-mini \
             qwen3:8b gemma3:9b exaone3.5:7.8b llama3.1:8b mistral:7b \
             qwen3:14b gemma3:12b; do
  ollama pull $model
done
```

### Handling Pipeline Parallel Sections — OLLAMA_NUM_PARALLEL

Sections that run in parallel in the pipeline:
- `search_worker` fan-out: N sub-queries → each worker simultaneously requests Ollama CRAG classification
- `local_search_worker` fan-out: same pattern

With the current architecture, sequential Ollama processing negates the parallel fan-out benefit.
Setting `OLLAMA_NUM_PARALLEL` to open N slots in VRAM allows concurrent requests to be truly processed in parallel.

**NUM_PARALLEL settings by model size** (based on g6e.xlarge VRAM 48GB):

| Model | VRAM/slot | NUM_PARALLEL | VRAM used | Fan-out coverage |
|------|---------|-------------|---------|-----------|
| llama3.2:3b | ~2GB | 16 | ~32GB | Full ✅ |
| qwen3:4b | ~3GB | 12 | ~36GB | Full ✅ |
| phi4-mini | ~3GB | 12 | ~36GB | Full ✅ |
| qwen3:8b | ~5GB | 8 | ~40GB | Full ✅ |
| gemma3:9b | ~6GB | 7 | ~42GB | Full ✅ |
| exaone3.5:7.8b | ~5GB | 8 | ~40GB | Full ✅ |
| llama3.1:8b | ~5GB | 8 | ~40GB | Full ✅ |
| mistral:7b | ~5GB | 8 | ~40GB | Full ✅ |
| qwen3:14b | ~9GB | 5 | ~45GB | Full ✅ |
| gemma3:12b | ~8GB | 6 | ~48GB | Full ✅ |

### Setup Procedure

```bash
# 1. Launch EC2 (Deep Learning AMI Ubuntu 22.04 — CUDA pre-installed)
aws ec2 run-instances \
  --image-id ami-xxxxxxxxx \
  --instance-type g6.xlarge \
  --key-name <key-name> \
  --security-group-ids <sg-id> \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ollama-benchmark}]'

# 2. Security group — open port 11434 to your IP only
aws ec2 authorize-security-group-ingress \
  --group-id <sg-id> \
  --protocol tcp --port 11434 \
  --cidr <MY_IP>/32

# 3. SSH into EC2 and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 4. Pre-pull all models (background, ~30 min)
for model in llama3.2:3b qwen3:4b phi4-mini \
             qwen3:8b gemma3:9b exaone3.5:7.8b llama3.1:8b mistral:7b \
             qwen3:14b gemma3:12b; do
  ollama pull $model &
done
wait
```

### Benchmark Run Script (from local laptop)

Restart Ollama with the appropriate `NUM_PARALLEL` when switching models:

```bash
#!/bin/bash
# run_benchmark.sh
EC2_IP=<EC2_PUBLIC_IP>

declare -A NUM_PARALLEL=(
  ["llama3.2:3b"]=8  ["qwen3:4b"]=7    ["phi4-mini"]=7
  ["qwen3:8b"]=4     ["gemma3:9b"]=3   ["exaone3.5:7.8b"]=4
  ["llama3.1:8b"]=4  ["mistral:7b"]=4
  ["qwen3:14b"]=2    ["gemma3:12b"]=2
)

for MODEL in "${!NUM_PARALLEL[@]}"; do
  NP=${NUM_PARALLEL[$MODEL]}
  echo "=== $MODEL (NUM_PARALLEL=$NP) ==="

  # Restart Ollama on EC2 with correct parallelism setting
  ssh ec2-user@$EC2_IP \
    "pkill ollama; sleep 2; OLLAMA_HOST=0.0.0.0:11434 OLLAMA_NUM_PARALLEL=$NP ollama serve &"
  sleep 5

  export OLLAMA_HOST=http://$EC2_IP:11434
  export OLLAMA_MODEL=$MODEL

  python -m eval.critic_pretest        # Phase 1
  python -m eval.component_benchmark --provider ollama --model $MODEL  # Phase 2
  python -m eval.e2e_benchmark --provider hybrid --model $MODEL --slim  # Phase 3
done
```

---

## 3. Candidate Model List

Target spec for research desktop: GPU VRAM 16–24GB, RAM 32GB or more.

### Tier A — 3–4B (extra-small / fast response priority)

| Model | Size | VRAM | Speed estimate | Notes |
|------|------|------|---------|------|
| `llama3.2:3b` | 2.0GB | ~4GB | ~120 tok/s | Meta official, minimal footprint |
| `qwen3:4b` | 2.6GB | ~5GB | ~100 tok/s | Alibaba, strong instruction following |
| `phi4-mini` | 3.8B | ~6GB | ~90 tok/s | Microsoft, high reasoning capability for its size |

### Tier B — 7–9B (main target / standard for research desktops)

| Model | Size | VRAM | Speed estimate | Notes |
|------|------|------|---------|------|
| `qwen3:8b` ⭐ | 5.2GB | ~8GB | ~40 tok/s | **Current baseline** |
| `gemma3:9b` | 6.0GB | ~9GB | ~35 tok/s | Google, strong reasoning / long context |
| `exaone3.5:7.8b` | 4.7GB | ~8GB | ~45 tok/s | LG AI, balanced Korean + English |
| `llama3.1:8b` | 4.9GB | ~8GB | ~45 tok/s | Meta, general-purpose baseline |
| `mistral:7b` | 4.1GB | ~7GB | ~50 tok/s | Mistral, stable JSON output |

### Tier C — 12–14B (high-end desktop / maximum quality target)

| Model | Size | VRAM | Speed estimate | Notes |
|------|------|------|---------|------|
| `qwen3:14b` | 9.0GB | ~14GB | ~25 tok/s | Alibaba, expected significant accuracy improvement over 8B |
| `gemma3:12b` | 8.1GB | ~13GB | ~28 tok/s | Google, strong fact alignment |

> g6.xlarge VRAM 24GB — all Tier C models fit under Q4 quantization.

---

## 4. Evaluation Phases

### Phase 1 — Quick Screening (~10 min per model)

**Tool**: `eval/critic_pretest.py`
**Content**: AlignRAG 3-phase recall (phase1 off-topic / phase2 fabricated citation / phase3 numeric contradiction)
**Total time**: 8 models × 10 min = ~80 min

**Pass criteria (advance to Phase 2)**:

| Condition | Threshold |
|------|------|
| Overall recall | ≥ 0.40 (baseline qwen3:8b = 0.20) |
| phase3 recall | ≥ 0.20 (qwen3:8b = 0.00, complete failure) |

> Phase 1 pass bar set low — Spec RAG Stage 2 (cloud verifier) compensates, so direction matters more than absolute accuracy.

### Phase 2 — Component Benchmark (~20 min per model)

**Tool**: `eval/component_benchmark.py`
**Target**: Phase 1 passing models (expected 4–6)
**Total time**: 5 models × 20 min = ~100 min

**Metrics**:

| Technique | Metric | qwen3:8b baseline |
|------|---------|-------------|
| `alignrag` | recall vs gold | 0.67 |
| `crag` | F1 (3-way classification) | FAIL (0.07 Δ) |
| `stride` | JSON validity + routing accuracy | FAIL (0.00 Δ) |
| `mass_rag` | persona adherence score | FAIL (+0.05 Δ) |
| `sdp` | pruning accuracy | FAIL (0.00 Δ) |
| `dsap` | JSON parse success rate | 1.00 |
| `query_decomp` | sub-query diversity score | 0.80 |

**Pass criteria (advance to Phase 3)**:
PASS technique count vs. qwen3:8b ≥ 8/11 (currently 7/11)

### Phase 3 — E2E Hybrid Benchmark (~30 min per model)

**Tool**: `eval/e2e_benchmark.py --provider hybrid --slim`
**Target**: Top 2–3 models from Phase 2
**Condition**: `phase1_2_3` (full AlignRAG + Spec RAG critic)

**Metrics**:

| Metric | Description |
|--------|------|
| keyword_coverage | Domain keyword inclusion rate |
| citation_density | Citation density |
| misaligned_claims_detected | Spec RAG Stage 1 drafting quality |
| spec_rag_recall | Stage 1+3 final misalignment detection rate |
| revision_count | Number of critique → revise loop iterations |
| latency_per_query | Time per query (seconds) |
| hybrid_cost_usd | Bedrock cost (local = $0) |

---

## 5. Pass Criteria Summary (Production Deployment Threshold)

| Item | Minimum | Target |
|------|---------|------|
| Phase 1 overall recall | ≥ 0.40 | ≥ 0.50 |
| Phase 2 PASS technique count | ≥ 8/11 | ≥ 9/11 |
| Phase 3 spec_rag_recall | ≥ 0.67 (current qwen3:8b level) | ≥ 0.80 |
| Phase 3 latency | ≤ 600s/query | ≤ 400s/query |
| Phase 3 revision_count | ≤ 2 | ≤ 1 |

---

## 6. Cost Estimate

### EC2 Costs

| Phase | Duration | g6.xlarge Spot |
|------|---------|--------------|
| Phase 1 (8 models) | ~2 hours | ~$0.48 |
| Phase 2 (5 models) | ~2 hours | ~$0.48 |
| Phase 3 (3 models) | ~2 hours | ~$0.48 |
| Setup / model downloads | ~1 hour | ~$0.24 |
| **Total** | **~7 hours** | **~$1.68** |

> If Spot is interrupted, restart with the same settings and skip completed models.

### Bedrock Costs (Haiku 4.5)

Phase 3 E2E: 3 models × 3 queries × ~$0.02/query ≈ **~$0.18**

### Model Download Size

Total for all candidate models: ~50GB (EBS 100GB is sufficient)

**Total estimated cost: under ~$6**

---

## 7. Execution Order

```
1. Launch EC2 g6.xlarge + install Ollama
2. Pre-pull all candidate models (background)
3. Phase 1: run critic_pretest sequentially for 8 models → select passing models
4. Phase 2: component_benchmark for passing models → select top models
5. Phase 3: e2e_benchmark hybrid for top models → finalize recommended model
6. Terminate EC2 instance (cost saving)
7. Add results to BENCHMARK.md as Layer 4
```

---

## 8. Using the Results

- Select a **single recommended model** → update `.env` default `OLLAMA_MODEL`
- Provide **size-specific recommendations** → can be linked with TechniquesPanel depth presets
  - fast preset → recommend 3B model
  - normal preset → recommend 7–8B model
  - deep preset → recommend 14B model
- Analyze FAIL technique patterns → identify opportunities to improve prompts/schemas for those techniques
