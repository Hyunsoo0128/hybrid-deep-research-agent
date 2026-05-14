# Cloud Benchmark Plan — Reproducing the 35-Condition Evaluation

**Purpose**: Infrastructure setup for reproducing the full 35-condition evaluation from the paper:

> **Stage-Aware Local-Cloud Inference: Hybrid Pipelines Consistently Outperform Matched Cloud-Only Baselines**

The paper evaluates 35 configurations (120 queries × 5 runs) across cloud-only, hybrid, and all-local tiers using a Triple Judge Jury.

---

## 1. Experiment Overview

### 35 Conditions

| Tier | Conditions | Description |
|------|-----------|-------------|
| Cloud-only | 3 | Sonnet 4.6, Haiku 4.5, Llama 3.3 70B |
| Hybrid | 24 | 8 local models × 3 cloud backends |
| All-local | 8 | 8 local models, no cloud calls |

### 8 Local Models

| Model | Size | Tier |
|-------|------|------|
| exaone3.5:2.4b | ~2.4B | ~2–4B |
| gemma3:4b | ~4B | ~2–4B |
| qwen3:4b | ~4B | ~2–4B |
| llama3.2:3b | ~3B | ~2–4B |
| qwen3:8b | ~8B | ~7–8B |
| exaone3.5:7.8b | ~7.8B | ~7–8B |
| llama3.1:8b | ~8B | ~7–8B |
| gemma3:12b | ~12B | ~12B |

### 3 Cloud Backends

- Sonnet 4.6 (`us.anthropic.claude-sonnet-4-6`)
- Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`)
- Llama 3.3 70B (via Bedrock)

---

## 2. EC2 Environment

### Instance Selection

| Instance | GPU | VRAM | Use case | $/hr |
|---------|-----|------|----------|------|
| **g6.xlarge** | L4 | 24GB | Hybrid/all-local (2–4B, 7–8B models) | ~$0.80 |
| **g6e.xlarge** | L40S | 48GB | Hybrid/all-local (12B models, full parallelism) | ~$2.0 |
| t3.xlarge | CPU | — | Cloud-only conditions | ~$0.17 |

**Paper hardware**: NVIDIA L4 (24GB VRAM), Ubuntu 22.04. Broadly comparable to RTX 4090 (24GB VRAM).

### Setup Procedure

```bash
# 1. Launch EC2 (Deep Learning AMI Ubuntu 22.04 — CUDA pre-installed)
aws ec2 run-instances \
  --image-id ami-xxxxxxxxx \
  --instance-type g6.xlarge \
  --key-name <key-name> \
  --security-group-ids <sg-id> \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hybrid-benchmark}]'

# 2. SSH into EC2 and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 3. Pre-pull all local models
for model in exaone3.5:2.4b gemma3:4b qwen3:4b llama3.2:3b \
             qwen3:8b exaone3.5:7.8b llama3.1:8b gemma3:12b; do
  ollama pull $model
done

# 4. Pull embedding model
ollama pull nomic-embed-text
```

### OLLAMA_NUM_PARALLEL Settings

| Model | VRAM/slot | NUM_PARALLEL (L4 24GB) |
|-------|-----------|------------------------|
| exaone3.5:2.4b | ~2GB | 8 |
| gemma3:4b | ~3GB | 6 |
| qwen3:4b | ~3GB | 6 |
| llama3.2:3b | ~2GB | 8 |
| qwen3:8b | ~5GB | 4 |
| exaone3.5:7.8b | ~5GB | 4 |
| llama3.1:8b | ~5GB | 4 |
| gemma3:12b | ~8GB | 2 |

---

## 3. Running the Benchmark

### Single condition

```bash
# Best hybrid configuration (paper's top result)
python eval/standalone_benchmark.py \
  --mode hybrid \
  --local-model exaone3.5:2.4b \
  --cloud sonnet \
  --queries eval/fixtures/benchmark_queries.json \
  --runs 5

# Cloud-only baseline
python eval/standalone_benchmark.py \
  --mode cloud-only \
  --cloud sonnet \
  --queries eval/fixtures/benchmark_queries.json \
  --runs 5

# All-local
python eval/standalone_benchmark.py \
  --mode all-local \
  --local-model exaone3.5:2.4b \
  --queries eval/fixtures/benchmark_queries.json \
  --runs 5
```

### Full 35-condition benchmark

```bash
# Deploy parallel EC2 instances for multi-model runs
python eval/deploy_ec2.py --conditions all

# Or run sequentially on a single instance
python eval/run_phase1.py   # ~2–4B models
python eval/run_phase2.py   # ~7–8B models
python eval/run_phase3.py   # ~12B models
```

### Feature-flag ablation (Bedrock only)

```bash
python -m eval.e2e_benchmark --provider bedrock --conditions baseline phase1 phase1_2_3
python eval/analyze_results.py eval/results/bedrock_full_v1.json
```

---

## 4. Evaluation: Triple Judge Jury

The paper uses a Triple Judge Jury for evaluation. To reproduce:

```bash
# Run evaluation with all three judges
python eval/evaluate_reports.py \
  --results eval/results/standalone/ \
  --judges deepseek-r1 claude-opus-4.6 mistral-large-3 \
  --output eval/results/judge_scores.json
```

**Judge models**:
- Judge A: DeepSeek R1 (671B) — `deepseek-r1`
- Judge B: Claude Opus 4.6 — `us.anthropic.claude-opus-4-6`
- Judge C: Mistral Large 3 (675B) — `mistral-large-3`

**G-Eval rubric** (5 dimensions, 1–10 scale, divided by 10):
1. Coverage — does the report address all key aspects?
2. Accuracy — are claims well-supported and factually sound?
3. Citation Quality — are sources used effectively?
4. Depth — does the report demonstrate analytical depth?
5. Coherence — is the report well-organized and clearly written?

Final score = median of three judges' scores.

---

## 5. Benchmark Queries

120 multilingual queries across 5 domains and 5 query types:

| Domain | EN | KO+JA | Total |
|--------|-----|-------|-------|
| AI / ML | 23 | 3 | 26 |
| Science & Tech | 20 | 3 | 23 |
| Business | 23 | 4 | 27 |
| Medical | 17 | 4 | 21 |
| Law & Policy | 17 | 6 | 23 |
| **Total** | **100** | **20** | **120** |

Retrieved documents are pre-collected via Tavily web search into a frozen snapshot (7 sub-queries × 5 results = 35 documents per query). This ensures fair comparison across conditions.

---

## 6. Cost Estimate

### EC2 Costs (full 35-condition run)

| Phase | Instances | Duration | Cost |
|-------|-----------|----------|------|
| Cloud-only (3 conditions) | 1× t3.xlarge | ~6 hours | ~$1.00 |
| Hybrid ~2–4B (16 conditions) | 4× g6.xlarge | ~8 hours | ~$25.60 |
| Hybrid ~7–8B (12 conditions) | 3× g6.xlarge | ~10 hours | ~$24.00 |
| Hybrid ~12B (4 conditions) | 1× g6.xlarge | ~8 hours | ~$6.40 |
| All-local (8 conditions) | 2× g6.xlarge | ~8 hours | ~$12.80 |
| **Total** | | | **~$70** |

### Bedrock Costs (cloud API calls)

| Condition | Queries | Runs | Cost/query | Total |
|-----------|---------|------|------------|-------|
| Cloud-only Sonnet | 120 | 5 | $1.128 | ~$677 |
| Cloud-only Haiku | 120 | 5 | $0.376 | ~$226 |
| Cloud-only Llama 70B | 120 | 5 | $0.600 | ~$360 |
| Hybrid+Sonnet (8 configs) | 120 | 5 | ~$0.375 | ~$1,800 |
| Hybrid+Haiku (8 configs) | 120 | 5 | ~$0.093 | ~$446 |
| Hybrid+Llama 70B (8 configs) | 120 | 5 | ~$0.110 | ~$528 |
| **Total** | | | | **~$4,037** |

**Note**: The paper's full evaluation is expensive. For a quick reproduction, run a subset:
- 5 representative conditions × 20 queries × 3 runs ≈ ~$200

---

## 7. Results Storage

Results are stored in `eval/results/`:

```
eval/results/
├── standalone/
│   ├── cloud_only_sonnet/         # Cloud-only Sonnet results
│   ├── cloud_only_haiku/          # Cloud-only Haiku results
│   ├── cloud_only_llama70b/       # Cloud-only Llama 70B results
│   ├── exaone2.4b_sonnet/         # Hybrid: exaone3.5:2.4b + Sonnet
│   ├── gemma4b_sonnet/            # Hybrid: gemma3:4b + Sonnet
│   ├── ...
│   └── _summary.json              # Aggregated results across all conditions
└── judge_scores.json              # Triple Judge Jury scores
```

Each condition directory contains:
- `results.json`: per-query scores and reports
- `summary.json`: aggregated metrics (mean, std, median)
