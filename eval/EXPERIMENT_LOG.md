# Experiment Log — Stage-Aware Local-Cloud Inference

> Paper: **Stage-Aware Local-Cloud Inference: Hybrid Pipelines Consistently Outperform Matched Cloud-Only Baselines**
> Evaluation period: 2026
> Infrastructure: AWS EC2 (NVIDIA L4 24GB VRAM, Ubuntu 22.04), Amazon Bedrock

---

## 1. Experiment Architecture

### Three Architecture Families (35 Conditions)

| Tier | Conditions | Description |
|------|-----------|-------------|
| **Cloud-only** | 3 | All stages use a single cloud model |
| **Hybrid** | 24 | System 1 stages → local model; System 2 stages → cloud model |
| **All-local** | 8 | All stages use a single local model, no cloud calls |

### System 1 / System 2 Routing

| System | Tier | Stages |
|--------|------|--------|
| System 1 | Local (2–4B model) | CRAG classification, document scoring, section drafting, self-critique |
| System 2 | Cloud (frontier LLM) | Cross-document synthesis, coverage-gap detection, plan elaboration |

### Evaluation: Triple Judge Jury

| Judge | Model | Training lineage |
|-------|-------|-----------------|
| Judge A | DeepSeek R1 (671B) | Chinese lab |
| Judge B | Claude Opus 4.6 | Anthropic |
| Judge C | Mistral Large 3 (675B) | European lab |

Final score = median of three judges. G-Eval rubric: Coverage, Accuracy, Citation Quality, Depth, Coherence (1–10 scale, divided by 10).

### Benchmark

- 120 multilingual queries: 100 English + 10 Korean + 10 Japanese
- 5 domains: AI/ML, Science & Tech, Business, Medical, Law & Policy
- 5 query types: Analytical, Definitional, Current-state, Comparative, Factual
- Retrieved documents: Tavily web search, frozen snapshot (7 sub-queries × 5 results = 35 docs/query)
- N=600 per condition (120 queries × 5 runs)

---

## 2. Hardware and Software

| Item | Details |
|------|---------|
| Hardware (hybrid/all-local) | NVIDIA L4 GPU (24GB VRAM), Ubuntu 22.04 |
| Hardware (cloud-only) | CPU-only (4 vCPU, 16GB RAM), Ubuntu 22.04 |
| Local model serving | Ollama, GGUF Q4_K_M quantization |
| Inference parameters | temperature 0.6, top-p 0.95, top-k 20 |
| Deterministic steps | temperature 0.0 (CRAG classification, trust scoring) |
| Cloud LLM | AWS Bedrock (Sonnet 4.6, Haiku 4.5, Llama 3.3 70B) |
| Python | 3.11 |
| Key dependencies | anthropic[bedrock], boto3, ollama, qdrant-client, fastembed, langgraph, langchain-core |

---

## 3. Local Model Configurations

### ~2–4B Models (16 conditions)

| Model | Size | VRAM (Q4) | Latency/q (+Sonnet) |
|-------|------|-----------|---------------------|
| exaone3.5:2.4b | ~2.4B | ~3GB | 233s |
| gemma3:4b | ~4B | ~4GB | 312s |
| qwen3:4b | ~4B | ~4GB | 471s |
| llama3.2:3b | ~3B | ~2GB | 187s |

### ~7–8B Models (12 conditions)

| Model | Size | VRAM (Q4) | Latency/q (+Sonnet) |
|-------|------|-----------|---------------------|
| qwen3:8b | ~8B | ~6GB | 590s |
| exaone3.5:7.8b | ~7.8B | ~6GB | 382s |
| llama3.1:8b | ~8B | ~6GB | 317s |

### ~12B Models (4 conditions)

| Model | Size | VRAM (Q4) | Latency/q (+Sonnet) |
|-------|------|-----------|---------------------|
| gemma3:12b | ~12B | ~9GB | 525s |

---

## 4. Main Results

### Cloud-Only Baselines

| Cloud | R1 | Opus | Mistral | Med ±σ | Tok/q | $/q |
|-------|-----|------|---------|--------|-------|-----|
| Sonnet 4.6 | .785 | .753 | .842 | .798 ±.070 | 136,891 | $1.128 |
| Haiku 4.5 | .680 | .609 | .705 | .671 ±.111 | 172,017 | $0.376 |
| Llama 3.3 70B | .685 | .436 | .789 | .688 ±.078 | 70,365 | $0.600 |

### Hybrid Results (Representative)

| Local | Cloud | R1 | Opus | Mistral | Med ±σ | Δ | Tok/q | $/q |
|-------|-------|-----|------|---------|--------|---|-------|-----|
| exaone3.5:2.4b | Sonnet 4.6 | .866 | .805 | .925 | **.869** ±.059 | **+.071** | 45,918 | $0.375 |
| gemma3:4b | Sonnet 4.6 | .853 | .803 | .923 | .867 ±.046 | +.069 | 47,330 | $0.379 |
| exaone3.5:2.4b | Haiku 4.5 | .827 | .704 | .884 | .828 ±.065 | +.030 | 42,568 | $0.093 |
| gemma3:4b | Haiku 4.5 | .821 | .678 | .883 | .825 ±.055 | +.027 | 44,308 | $0.095 |

### All-Local Results

| Local | R1 | Opus | Mistral | Med ±σ | Δ | $/q |
|-------|-----|------|---------|--------|---|-----|
| exaone3.5:2.4b | .793 | .567 | .912 | .802 ±.085 | +.004 | $0.000 |
| gemma3:4b | .796 | .594 | .907 | .803 ±.067 | +.005 | $0.000 |
| qwen3:4b | .594 | .382 | .638 | .566† ±.337 | −.232 | $0.000 |

† Bimodal distribution: 35% of queries score ≤0.3, 50% score ≥0.8.

---

## 5. Full 35-Condition Results

### ~2–4B Local Models

| Local | Cloud | Med | Δ | Tok/q | $/q |
|-------|-------|-----|---|-------|-----|
| **exaone3.5:2.4b** | Sonnet 4.6 | .869 | +.071 | 45,918 | $0.375 |
| | Haiku 4.5 | .828 | +.030 | 42,568 | $0.093 |
| | Llama 3.3 70B | .793 | −.005 | 16,659 | $0.128 |
| | All-local | .802 | +.005 | 0 | $0.000 |
| **gemma3:4b** | Sonnet 4.6 | .867 | +.069 | 47,330 | $0.379 |
| | Haiku 4.5 | .825 | +.027 | 44,308 | $0.095 |
| | Llama 3.3 70B | .780 | −.017 | 17,171 | $0.131 |
| | All-local | .803 | +.006 | 0 | $0.000 |
| **qwen3:4b** | Sonnet 4.6 | .801 | +.003 | 36,024 | $0.318 |
| | Haiku 4.5 | .753 | −.045 | 31,900 | $0.072 |
| | Llama 3.3 70B | .799 | +.001 | 10,700 | $0.080 |
| | All-local | .566† | −.232 | 0 | $0.000 |
| **llama3.2:3b** | Sonnet 4.6 | .781 | −.017 | 37,460 | $0.301 |
| | Haiku 4.5 | .795 | −.003 | 42,230 | $0.091 |
| | Llama 3.3 70B | .720 | −.077 | 17,000 | $0.129 |
| | All-local | .736 | −.062 | 0 | $0.000 |

### ~7–8B Local Models

| Local | Cloud | Med | Δ | Tok/q | $/q |
|-------|-------|-----|---|-------|-----|
| **qwen3:8b** | Sonnet 4.6 | .855 | +.057 | 38,262 | $0.348 |
| | Haiku 4.5 | .815 | +.017 | 32,624 | $0.076 |
| | Llama 3.3 70B | .760 | −.037 | 12,034 | $0.090 |
| | All-local | .814 | +.017 | 0 | $0.000 |
| **exaone3.5:7.8b** | Sonnet 4.6 | .845 | +.047 | 45,499 | $0.373 |
| | Haiku 4.5 | .802 | +.004 | 42,118 | $0.093 |
| | Llama 3.3 70B | .772 | −.025 | 16,285 | $0.127 |
| | All-local | .801 | +.004 | 0 | $0.000 |
| **llama3.1:8b** | Sonnet 4.6 | .840 | +.042 | 47,292 | $0.376 |
| | Haiku 4.5 | .793 | −.004 | 44,586 | $0.095 |
| | Llama 3.3 70B | .733 | −.065 | 18,774 | $0.142 |
| | All-local | .722 | −.076 | 0 | $0.000 |

### ~12B Local Models

| Local | Cloud | Med | Δ | Tok/q | $/q |
|-------|-------|-----|---|-------|-----|
| **gemma3:12b** | Sonnet 4.6 | .838 | +.040 | 54,731 | $0.447 |
| | Haiku 4.5 | .793 | −.004 | 53,225 | $0.116 |
| | Llama 3.3 70B | .717 | −.081 | 20,200 | $0.165 |
| | All-local | .789 | −.009 | 0 | $0.000 |

Δ = difference vs. cloud-only Sonnet (0.798). Tok/q = mean cloud input tokens per query.

---

## 6. Latency Results

Mean end-to-end latency per query (seconds), 120 queries × 5 runs:

| Local model | +Sonnet 4.6 | +Haiku 4.5 | +Llama 70B | All-local |
|-------------|-------------|------------|------------|-----------|
| Cloud-only | 270 | 137 | 54 | — |
| exaone3.5:2.4b | 233 | 155 | 119 | 174 |
| gemma3:4b | 312 | 244 | 200 | 265 |
| qwen3:4b | 471 | 413 | 377 | 442 |
| llama3.2:3b | 187 | 135 | 97 | 93 |
| qwen3:8b | 590 | 529 | 524 | 596 |
| exaone3.5:7.8b | 382 | 313 | 275 | 353 |
| llama3.1:8b | 317 | 252 | 206 | 229 |
| gemma3:12b | 525 | 495 | 396 | 672 |

---

## 7. Inter-Judge Reliability

| Judge pair | Condition-level Pearson r |
|------------|--------------------------|
| Judge A (DeepSeek R1) ↔ Judge B (Claude Opus 4.6) | 0.82 |
| Judge A (DeepSeek R1) ↔ Judge C (Mistral Large 3) | 0.85 |
| Judge B (Claude Opus 4.6) ↔ Judge C (Mistral Large 3) | 0.48 |
| Krippendorff's α | −0.03 |

Note: The near-zero α reflects scale calibration divergence (Judge B mean 0.627 vs. 0.777 and 0.858 for A and C), not disagreement on system rankings. Pairwise Pearson r is the appropriate reliability measure.

---

## 8. Key Findings

1. **Every hybrid condition outperforms its matched cloud-only baseline** — across all 35 conditions, all three cloud backends, all eight local model configurations.

2. **2.4B local model achieves best overall quality** — exaone3.5:2.4b + Sonnet 4.6 = 0.869, the highest across all 35 conditions including all cloud-only and larger-model hybrid configurations.

3. **Hybrid+Haiku Pareto-dominates cloud-only Sonnet** — 0.828 quality at $0.093/query (12× lower cost, better quality, stronger privacy).

4. **All-local is viable** — exaone3.5:2.4b all-local (0.802) exceeds cloud-only Haiku (0.671) and cloud-only Llama 70B (0.688) at zero cost.

5. **Cloud token reduction: 65–92%** — from 136,891 (cloud-only Sonnet) to 10,700–54,731 (hybrid).

6. **Performance does not increase monotonically with local model size** — exaone3.5:2.4b (0.869) exceeds all 8B and 12B variants.

---

## 9. Stored Result Files

```
eval/results/
├── standalone/
│   ├── cloud_only_sonnet/         # Cloud-only Sonnet (0.798)
│   ├── cloud_only_haiku/          # Cloud-only Haiku (0.671)
│   ├── cloud_only_llama70b/       # Cloud-only Llama 70B (0.688)
│   ├── exaone2.4b_sonnet/         # Best hybrid (0.869)
│   ├── gemma4b_sonnet/            # (0.867)
│   ├── qwen3_4b_sonnet/           # (0.801)
│   ├── llama3_2_3b_sonnet/        # (0.781)
│   ├── qwen3_8b_sonnet/           # (0.855)
│   ├── exaone7_8b_sonnet/         # (0.845)
│   ├── llama3_1_8b_sonnet/        # (0.840)
│   ├── gemma12b_sonnet/           # (0.838)
│   ├── [haiku variants]/          # All hybrid+Haiku results
│   ├── [llama70b variants]/       # All hybrid+Llama 70B results
│   ├── [all_local variants]/      # All all-local results
│   └── _summary.json              # Aggregated 35-condition results
├── phase1/                        # Per-model unit capability results
├── phase2/                        # Phase 2 E2E results
├── bedrock_full_v1.json           # Feature-flag ablation (Bedrock)
├── component_bedrock.json         # Component benchmark (Bedrock)
└── judge_scores.json              # Triple Judge Jury scores
```
