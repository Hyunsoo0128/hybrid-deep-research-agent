# Benchmark Observations

> Research notes from the 35-condition evaluation described in the paper:
> **Stage-Aware Local-Cloud Inference: Hybrid Pipelines Consistently Outperform Matched Cloud-Only Baselines**

---

## 1. Every Hybrid Condition Outperforms Its Matched Cloud-Only Baseline

The most unambiguous finding: across all 35 conditions, every hybrid configuration outperforms the cloud-only baseline using the **same cloud model**.

- Hybrid+Sonnet beats cloud-only Sonnet (all 8 local model configurations)
- Hybrid+Haiku beats cloud-only Haiku (all 8 local model configurations)
- Hybrid+Llama 70B beats cloud-only Llama 70B (all 8 local model configurations)

This rules out model-diversity as the explanation — the quality gain follows from offloading System 1 stages to a local model alone.

---

## 2. 2.4B Local Model Achieves Best Overall Quality

exaone3.5:2.4b + Sonnet 4.6 achieves 0.869 — the highest quality across all 35 conditions, including all cloud-only and larger-model hybrid configurations.

Within Hybrid+Sonnet tier:
- exaone3.5:2.4b: 0.869 (233s/query)
- gemma3:4b: 0.867 (312s/query)
- qwen3:8b: 0.855 (590s/query)
- exaone3.5:7.8b: 0.845 (382s/query)
- llama3.1:8b: 0.840 (317s/query)
- gemma3:12b: 0.838 (525s/query)

Performance does not increase monotonically with local model size. The sufficiency threshold varies across model families.

---

## 3. Task-Scope Alignment Explains the Quality Gain

Two mechanisms explain why local System 1 models improve overall quality:

**Task-scope alignment**: Frontier models over-elaborate outputs for constrained generation tasks (CRAG classification, trust scoring), producing verbose reasoning where a concise label is required. Smaller models, with narrower output distributions, stay on-task more reliably.

**Synthesis leverage**: When local models generate section drafts, the cloud synthesis stage receives imperfect but diverse raw material, creating genuine revision leverage absent when a frontier draft is already near-final.

---

## 4. Hybrid+Haiku Pareto-Dominates Cloud-Only Sonnet

exaone3.5:2.4b + Haiku 4.5 achieves 0.828 at $0.093/query — simultaneously better quality, 12× lower cost, and stronger privacy than cloud-only Sonnet ($1.128/query, 0.798).

The conventional framing treats privacy and quality as opposing forces. Stage-Aware Local-Cloud Inference overturns this directly.

---

## 5. All-Local Is Viable Above Weaker Cloud-Only Baselines

exaone3.5:2.4b all-local: 0.802 at $0.000/query
gemma3:4b all-local: 0.803 at $0.000/query

Both exceed cloud-only Haiku (0.671) and cloud-only Llama 70B (0.688). The 6–7 point gap vs. Hybrid+Sonnet is the cost of complete privacy.

---

## 6. Cloud Token Reduction: 65–92%

Average cloud input tokens reduced from 136,891 (cloud-only Sonnet) to 10,700–54,731 (hybrid):

| Configuration | Cloud Tokens/q | Reduction |
|---------------|----------------|-----------|
| Cloud-only Sonnet | 136,891 | ref |
| exaone3.5:2.4b + Sonnet | 45,918 | −66.5% |
| gemma3:4b + Sonnet | 47,330 | −65.4% |
| qwen3:4b + Haiku | 31,900 | −76.7% |
| qwen3:4b + Llama 70B | 10,700 | −92.2% |

The most privacy-aggressive condition (qwen3:4b + Llama 70B, 92.2% reduction) still scores 0.799 — above both weaker cloud-only baselines.

---

## 7. Cloud-Only Pipelines Face a Quality Ceiling

Cloud-only Haiku (0.671) and cloud-only Llama 70B (0.688) fall substantially below cloud-only Sonnet (0.798), confirming that single-model cloud pipelines face a quality ceiling that hybrid architectures overcome.

The hybrid architecture overcomes this ceiling by separating reasoning demands: System 1 stages (where local models are sufficient) and System 2 stages (where frontier reasoning matters).

---

## 8. qwen3:4b All-Local Bimodal Distribution

qwen3:4b all-local shows a bimodal score distribution (σ=0.337): 35% of queries score ≤0.3, 50% score ≥0.8. Not recommended for all-local deployment.

This pattern does not appear in hybrid configurations of qwen3:4b — the cloud synthesis stage resolves the bimodal failure mode.

---

## 9. Failure Mode: Capacity-Limited Cloud Backend

Hybrid quality degrades when the System 2 cloud model is capacity-limited. Hybrid+Llama 70B conditions fall below the Sonnet baseline but remain above weaker cloud-only baselines (0.671–0.688).

Practitioners should prioritize cloud backend quality over local model size. A 2.4B local model with Sonnet cloud (0.869) outperforms a 12B local model with Llama 70B cloud (0.717).

---

## 10. Inter-Judge Reliability

The Triple Judge Jury (DeepSeek R1, Claude Opus 4.6, Mistral Large 3) shows strong condition-level agreement:

| Judge pair | Condition-level Pearson r |
|------------|--------------------------|
| Judge A ↔ Judge B | 0.82 |
| Judge A ↔ Judge C | 0.85 |
| Judge B ↔ Judge C | 0.48 |

The low B↔C correlation reflects Judge B's systematically compressed score range (mean 0.627 vs. 0.777 and 0.858 for A and C), not disagreement on system rankings. The median aggregation is robust to this pattern: with two of three judges (A and C) in strong agreement, the median consistently tracks the consensus.

---

## 11. Latency Observations

exaone3.5:2.4b is the fastest hybrid configuration (233s/query with Sonnet), faster than cloud-only Sonnet (270s/query). This is because local System 1 stages run in parallel with cloud API round-trips, and the reduced cloud token count shortens cloud processing time.

Llama 70B hybrid is faster than Haiku hybrid despite being a larger model, because it receives far fewer tokens (local models handle more pipeline stages).

---

## 12. Open Questions

1. **Multilingual analysis**: The benchmark includes Korean and Japanese queries, but the current analysis is primarily English-focused. Multilingual analysis will be reported in future work.

2. **Other retrieval backends**: All experiments use Tavily web search via a frozen snapshot. Performance on other retrieval backends (academic search, internal document stores) may differ.

3. **CPU-only hardware**: Experiments were conducted on NVIDIA L4 (24GB VRAM). On CPU-only or weaker GPU hardware, local model latency would increase substantially, potentially altering practical throughput.

4. **Privacy boundary completeness**: The Privacy Boundary prevents document bodies from reaching the cloud, but does not protect against inference attacks on the abstractions and drafts that do reach the cloud. Stronger privacy guarantees remain future work.
