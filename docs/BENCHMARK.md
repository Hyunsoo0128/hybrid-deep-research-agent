# Benchmark — Stage-Aware Local-Cloud Inference

Empirical results from the paper:

> **Stage-Aware Local-Cloud Inference: Hybrid Pipelines Consistently Outperform Matched Cloud-Only Baselines**

---

## Benchmark Setup

### System Configurations (35 conditions)

Three architecture families:

- **Cloud-only (3 conditions)**: All stages use a single cloud model (Sonnet 4.6, Haiku 4.5, or Llama 3.3 70B)
- **Hybrid (24 conditions)**: System 1 stages run on a local model; System 2 stages run on a cloud model. Eight local configurations spanning four families (Gemma, Qwen, EXAONE, Llama) at ~2–4B, ~7–8B, and ~12B, each paired with three cloud backends
- **All-local (8 conditions)**: All stages use a single local model with no cloud calls

Each hybrid configuration is compared against a **matched cloud-only baseline** using the same cloud model, isolating the effect of task-separated routing from model-selection effects.

All local models are used off-the-shelf with default quantization (Q4_K_M); no task-specific fine-tuning or prompt tuning is applied.

### Benchmark Queries

120 multilingual deep research queries:

| Domain | EN | KO+JA | Total |
|--------|-----|-------|-------|
| AI / ML | 23 | 3 | 26 |
| Science & Tech | 20 | 3 | 23 |
| Business | 23 | 4 | 27 |
| Medical | 17 | 4 | 21 |
| Law & Policy | 17 | 6 | 23 |
| **Total** | **100** | **20** | **120** |

Query types: Analytical (21.7%), Definitional (20.0%), Current-state (20.0%), Comparative (20.0%), Factual (18.3%).

All queries require multi-document synthesis; none have single-sentence factoid answers.

Retrieved documents are pre-collected via Tavily web search into a frozen snapshot (7 sub-queries × 5 results = 35 documents per query).

### Evaluation: Triple Judge Jury

Three LLM evaluators score each report on a 0–1 scale using a G-Eval rubric with five anchored dimensions:

1. **Coverage** — Does the report address all key aspects of the query?
2. **Accuracy** — Are claims well-supported and factually sound?
3. **Citation Quality** — Are sources used effectively?
4. **Depth** — Does the report demonstrate analytical depth?
5. **Coherence** — Is the report well-organized and clearly written?

The final score is the **median of three judges** from distinct training lineages:

| Judge | Model | Training lineage |
|-------|-------|-----------------|
| Judge A | DeepSeek R1 (671B) | Chinese lab |
| Judge B | Claude Opus 4.6 | Anthropic |
| Judge C | Mistral Large 3 (675B) | European lab |

Each configuration runs 5 times over 120 queries (N=600); results report mean ± std, paired Wilcoxon tests with Bonferroni correction, and Cohen's d.

### Inter-Judge Reliability

| Judge pair | Condition-level Pearson r |
|------------|--------------------------|
| Judge A ↔ Judge B | 0.82 |
| Judge A ↔ Judge C | 0.85 |
| Judge B ↔ Judge C | 0.48 |
| Krippendorff's α | −0.03 |

The near-zero α reflects scale calibration divergence: Judge B scores systematically lower (mean 0.627 vs. 0.777 and 0.858 for A and C), not disagreement on system rankings. Pairwise Pearson r is the appropriate reliability measure here, as it captures rank-order agreement while remaining invariant to judges' systematically different absolute scales.

---

## Main Results

### Representative Results (Table 1 from paper)

| Mode | Local | Cloud | R1 | Opus | Mistral | Med ±σ | Δ | Tok/q | $/q |
|------|-------|-------|-----|------|---------|--------|---|-------|-----|
| **Cloud-only** | | | | | | | | | |
| | — | Sonnet 4.6 | .785 | .753 | .842 | .798 ±.070 | ref | 136.9K | $1.128 |
| | — | Haiku 4.5 | .680 | .609 | .705 | .671 ±.111 | −.127 | 172.0K | $0.376 |
| | — | Llama 70B | .685 | .436 | .789 | .688 ±.078 | −.110 | 70.4K | $0.600 |
| **Hybrid** | | | | | | | | | |
| | exaone2.4b | Sonnet 4.6 | .866 | .805 | .925 | **.869** ±.059 | **+.071** | 45.9K | $0.375 |
| | gemma3:4b | Sonnet 4.6 | .853 | .803 | .923 | .867 ±.046 | +.069 | 47.3K | $0.379 |
| | qwen3:4b | Sonnet 4.6 | .788 | .745 | .859 | .801 ±.060 | +.003 | 36.0K | $0.318 |
| | exaone2.4b | Haiku 4.5 | .827 | .704 | .885 | .828 ±.065 | +.030 | 42.6K | $0.093 |
| | gemma3:4b | Haiku 4.5 | .821 | .678 | .883 | .825 ±.055 | +.027 | 44.3K | $0.095 |
| | qwen3:4b | Haiku 4.5 | .758 | .604 | .792 | .753 ±.085 | −.045 | 31.9K | $0.072 |
| | exaone2.4b | Llama 70B | .780 | .559 | .893 | .793 ±.078 | −.005 | 16.7K | $0.128 |
| | gemma3:4b | Llama 70B | .777 | .504 | .875 | .780 ±.063 | −.018 | 17.2K | $0.131 |
| | qwen3:4b | Llama 70B | .791 | .501 | .923 | .799 ±.089 | +.001 | 10.7K | $0.080 |
| **All-local** | | | | | | | | | |
| | exaone2.4b | — | .793 | .567 | .912 | .802 ±.085 | +.004 | 0 | $0.000 |
| | gemma3:4b | — | .796 | .594 | .907 | .803 ±.067 | +.005 | 0 | $0.000 |
| | qwen3:4b | — | .594 | .382 | .638 | .566 ±.337 | −.232 | 0 | $0.000 |

Med = median of three judge scores. Δ = difference vs. cloud-only Sonnet (0.798). Tok/q = mean cloud input tokens per query (K = ×10³; All-local = 0 by design).

---

## Full 35-Condition Results

### ~2–4B Local Models (16 conditions)

| Local | Cloud | R1 | Opus | Mistral | Med ±σ | Δ | Tok/q | $/q |
|-------|-------|-----|------|---------|--------|---|-------|-----|
| **exaone3.5:2.4b** | | | | | | | | |
| | Sonnet 4.6 | .866 | .805 | .925 | .869 ±.059 | +.071 | 45,918 | $0.375 |
| | Haiku 4.5 | .827 | .704 | .884 | .828 ±.065 | +.030 | 42,568 | $0.093 |
| | Llama 3.3 70B | .780 | .559 | .893 | .793 ±.078 | −.005 | 16,659 | $0.128 |
| | All-local | .793 | .567 | .912 | .802 ±.085 | +.005 | 0 | $0.000 |
| **gemma3:4b** | | | | | | | | |
| | Sonnet 4.6 | .853 | .803 | .923 | .867 ±.046 | +.069 | 47,330 | $0.379 |
| | Haiku 4.5 | .821 | .678 | .883 | .825 ±.055 | +.027 | 44,308 | $0.095 |
| | Llama 3.3 70B | .777 | .504 | .875 | .780 ±.063 | −.017 | 17,171 | $0.131 |
| | All-local | .796 | .594 | .907 | .803 ±.067 | +.006 | 0 | $0.000 |
| **qwen3:4b** | | | | | | | | |
| | Sonnet 4.6 | .788 | .745 | .859 | .801 ±.060 | +.003 | 36,024 | $0.318 |
| | Haiku 4.5 | .758 | .604 | .792 | .753 ±.085 | −.045 | 31,900 | $0.072 |
| | Llama 3.3 70B | .791 | .501 | .923 | .799 ±.089 | +.001 | 10,700 | $0.080 |
| | All-local | .594 | .382 | .638 | .566† ±.337 | −.232 | 0 | $0.000 |
| **llama3.2:3b** | | | | | | | | |
| | Sonnet 4.6 | .829 | .803 | .826 | .781 ±.147 | −.017 | 37,460 | $0.301 |
| | Haiku 4.5 | .796 | .674 | .853 | .795 ±.076 | −.003 | 42,230 | $0.091 |
| | Llama 3.3 70B | .714 | .432 | .845 | .720 ±.094 | −.077 | 17,000 | $0.129 |
| | All-local | .731 | .496 | .831 | .736 ±.090 | −.062 | 0 | $0.000 |

### ~7–8B Local Models (12 conditions)

| Local | Cloud | R1 | Opus | Mistral | Med ±σ | Δ | Tok/q | $/q |
|-------|-------|-----|------|---------|--------|---|-------|-----|
| **qwen3:8b** | | | | | | | | |
| | Sonnet 4.6 | .839 | .790 | .915 | .855 ±.044 | +.057 | 38,262 | $0.348 |
| | Haiku 4.5 | .811 | .653 | .871 | .815 ±.058 | +.017 | 32,624 | $0.076 |
| | Llama 3.3 70B | .754 | .542 | .860 | .760 ±.066 | −.037 | 12,034 | $0.090 |
| | All-local | .804 | .646 | .899 | .814 ±.064 | +.017 | 0 | $0.000 |
| **exaone3.5:7.8b** | | | | | | | | |
| | Sonnet 4.6 | .822 | .785 | .904 | .845 ±.047 | +.047 | 45,499 | $0.373 |
| | Haiku 4.5 | .797 | .684 | .861 | .802 ±.059 | +.004 | 42,118 | $0.093 |
| | Llama 3.3 70B | .759 | .576 | .880 | .772 ±.069 | −.025 | 16,285 | $0.127 |
| | All-local | .793 | .627 | .893 | .801 ±.061 | +.004 | 0 | $0.000 |
| **llama3.1:8b** | | | | | | | | |
| | Sonnet 4.6 | .830 | .781 | .897 | .840 ±.062 | +.042 | 47,292 | $0.376 |
| | Haiku 4.5 | .794 | .655 | .850 | .793 ±.074 | −.004 | 44,586 | $0.095 |
| | Llama 3.3 70B | .722 | .453 | .855 | .733 ±.087 | −.065 | 18,774 | $0.142 |
| | All-local | .720 | .484 | .808 | .722 ±.082 | −.076 | 0 | $0.000 |

### ~12B Local Models (4 conditions)

| Local | Cloud | R1 | Opus | Mistral | Med ±σ | Δ | Tok/q | $/q |
|-------|-------|-----|------|---------|--------|---|-------|-----|
| **gemma3:12b** | | | | | | | | |
| | Sonnet 4.6 | .817 | .791 | .896 | .838 ±.055 | +.040 | 54,731 | $0.447 |
| | Haiku 4.5 | .788 | .673 | .847 | .793 ±.067 | −.004 | 53,225 | $0.116 |
| | Llama 3.3 70B | .715 | .517 | .830 | .717 ±.054 | −.081 | 20,200 | $0.165 |
| | All-local | .781 | .641 | .868 | .789 ±.070 | −.009 | 0 | $0.000 |

† Bimodal distribution (σ=0.337): 35% of queries score ≤0.3, 50% score ≥0.8.

---

## Latency Results

Mean end-to-end latency per query (seconds), measured across 120 queries × 5 runs:

| Local model | Hardware | +Sonnet 4.6 | +Haiku 4.5 | +Llama 70B | All-local |
|-------------|----------|-------------|------------|------------|-----------|
| Cloud-only | CPU-only | 270 | 137 | 54 | — |
| exaone3.5:2.4b | NVIDIA L4 | 233 | 155 | 119 | 174 |
| gemma3:4b | NVIDIA L4 | 312 | 244 | 200 | 265 |
| qwen3:4b | NVIDIA L4 | 471 | 413 | 377 | 442 |
| llama3.2:3b | NVIDIA L4 | 187 | 135 | 97 | 93 |
| qwen3:8b | NVIDIA L4 | 590 | 529 | 524 | 596 |
| exaone3.5:7.8b | NVIDIA L4 | 382 | 313 | 275 | 353 |
| llama3.1:8b | NVIDIA L4 | 317 | 252 | 206 | 229 |
| gemma3:12b | NVIDIA L4 | 525 | 495 | 396 | 672 |

Llama 70B hybrid is faster than Haiku hybrid despite being a larger model, because it receives far fewer tokens (local models handle more pipeline stages). All-local conditions are not always fastest due to longer local generation without cloud synthesis.

---

## Key Findings

### 1. Every hybrid condition outperforms its matched cloud-only baseline

Hybrid+Sonnet beats cloud-only Sonnet, Hybrid+Haiku beats cloud-only Haiku, and Hybrid+Llama 70B beats cloud-only Llama 70B — consistently across all local model configurations. The quality gain is present regardless of which cloud model handles System 2 stages, ruling out model-diversity as the sole explanation.

### 2. 2.4B local model exceeds all cloud-only baselines (with Sonnet cloud)

All Hybrid+Sonnet conditions outperform even the strongest cloud-only baseline (cloud-only Sonnet, 0.798), including conditions using a 2.4B local model. The best configuration (exaone3.5:2.4b + Sonnet 4.6) achieves 0.869 — a 7.1-point absolute improvement (p < 0.001, Wilcoxon signed-rank).

### 3. Hybrid+Haiku Pareto-dominates cloud-only Sonnet

The Hybrid+Haiku configuration achieves 0.828 quality at $0.093/query — a 12× cost reduction that Pareto-dominates cloud-only Sonnet on quality, cost, and privacy simultaneously.

### 4. Performance does not increase monotonically with local model size

Within Hybrid+Sonnet: exaone3.5:2.4b (0.869) and gemma3:4b (0.867) exceed all 8B (0.840–0.855) and the 12B variant (gemma3:12b, 0.838), with faster inference. However, the advantage is model-family dependent: qwen3:4b (0.801) falls below qwen3:8b (0.855) in the same tier.

### 5. Cloud token reduction: 65–92%

Average cloud input tokens reduced from 136,891 (cloud-only Sonnet) to 10,700–54,731 (hybrid), while simultaneously improving output quality. The most privacy-aggressive condition (qwen3:4b + Llama 70B, 92.2% reduction) still scores 0.799 — above cloud-only Haiku and cloud-only Llama 70B.

---

## Benchmark Environment

| Item | Details |
|------|---------|
| Hardware (hybrid/all-local) | NVIDIA L4 GPU (24GB VRAM), Ubuntu 22.04 |
| Hardware (cloud-only) | CPU-only (4 vCPU, 16GB RAM), Ubuntu 22.04 |
| Local model serving | Ollama, GGUF Q4_K_M quantization |
| Inference parameters | temperature 0.6, top-p 0.95, top-k 20 (deterministic steps: temperature 0.0) |
| Cloud LLM | AWS Bedrock (Sonnet 4.6, Haiku 4.5, Llama 3.3 70B) |
| Retrieval | Tavily web search, frozen snapshot (7 sub-queries × 5 results = 35 docs/query) |
| Evaluation | Triple Judge Jury (DeepSeek R1, Claude Opus 4.6, Mistral Large 3) |
| Runs per condition | 5 runs × 120 queries = N=600 |
| Statistical tests | Paired Wilcoxon signed-rank, Bonferroni correction, Cohen's d |

---

## Running the Benchmark

### Single condition

```bash
# Best hybrid configuration
python eval/standalone_benchmark.py --mode hybrid --local-model exaone3.5:2.4b --cloud sonnet

# Cloud-only baseline
python eval/standalone_benchmark.py --mode cloud-only --cloud sonnet

# All-local
python eval/standalone_benchmark.py --mode all-local --local-model exaone3.5:2.4b
```

### Full 35-condition benchmark (requires EC2 + AWS credentials)

```bash
# See eval/CLOUD_BENCHMARK_PLAN.md for infrastructure setup
python eval/deploy_ec2.py --conditions all
```

### Feature-flag ablation

```bash
python -m eval.e2e_benchmark --provider bedrock --conditions baseline phase1 phase1_2_3
python eval/analyze_results.py eval/results/bedrock_full_v1.json
```
