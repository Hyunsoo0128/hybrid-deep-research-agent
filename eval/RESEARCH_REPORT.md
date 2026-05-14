# Stage-Aware Local-Cloud Inference: Experimental Report

> **Paper**: Stage-Aware Local-Cloud Inference: Hybrid Pipelines Consistently Outperform Matched Cloud-Only Baselines
> Evaluation: 35 conditions, 120 queries × 5 runs, Triple Judge Jury

---

## Abstract

Deploying frontier LLMs for deep research surfaces a trilemma in regulated industries: cloud APIs violate data-residency requirements, local models alone cannot match synthesis quality, and cloud-only pipelines become cost-prohibitive at scale. We argue that this trilemma is an architectural artifact: it arises from treating all pipeline stages as equally demanding.

In practice, document scoring, retrieval classification, section drafting, and self-critique require only lightweight inference, well within the capability of 2–4B local models; cross-document synthesis and coverage-gap detection, by contrast, require frontier-scale reasoning. We propose **Stage-Aware Local-Cloud Inference**, a hybrid architecture that routes each stage to the appropriate compute tier, instantiated across nine established RAG techniques.

Evaluating 35 configurations (120 queries × 5 runs per condition) with a three-model judge jury, **every hybrid condition outperforms its matched cloud-only baseline**. Our best hybrid (exaone3.5:2.4b + Sonnet 4.6) achieves **0.869 vs. 0.798** for the cloud-only Sonnet baseline. Across hybrid conditions, cloud token exposure is reduced by **65–92%** and cost by up to **12×**.

---

## 1. Research Question

> **Can a hybrid of small local LLMs and large cloud LLMs resolve the privacy-cost-quality trilemma in deep research pipelines?**

Three hypotheses tested:

1. **H1 (Stage routing)**: Routing System 1 stages to local models and System 2 stages to cloud models improves quality over cloud-only baselines.
2. **H2 (Privacy boundary)**: The Privacy Boundary (original queries and documents never reaching the cloud) can be enforced structurally without quality degradation.
3. **H3 (Cost efficiency)**: Hybrid routing reduces cloud token exposure and cost while maintaining or improving quality.

All three hypotheses are confirmed.

---

## 2. System Design

### Stage-Routing Principle

The dividing criterion is **input scope**:

- **System 1 (local)**: A stage that processes a single document or draft — bounded-context operations. Retrieval classification, passage scoring, section drafting, self-critique.
- **System 2 (cloud)**: A stage that must integrate evidence across multiple independent sources — deliberate cross-document reasoning. Cross-document synthesis, coverage-gap detection, plan elaboration.

### Privacy Boundary

Formal constraint: `inputs(cloud) ∩ {original_query, full_corpus} = ∅`

The cloud receives only:
- Document abstractions: titles and 150-character excerpts
- Local drafts: compact prose from the local model, no verbatim document excerpts

### Four-Phase Pipeline

```
Planning Phase    → Retrieval Phase    → Drafting Phase    → Verification Phase
[LOCAL] Sq        [LOCAL] CRAG 1st     [LOCAL] 3-agent     [CLOUD] gap detection
[CLOUD] Cq        [CLOUD] AMBIGUOUS    [CLOUD] synthesis   [LOCAL] consistency
[LOCAL] VCM       [LOCAL] reranking                        [LOCAL] self-critique
```

### Nine RAG Technique Adaptations

| Phase | Technique | System 1 (Local) | System 2 (Cloud) |
|-------|-----------|-----------------|-----------------|
| Planning | Query Decomp | sub-query generation | — |
| Planning | STRIDE | abstract planning (Sq) | concrete planning (Cq) |
| Planning | DSAP | section outline | section elaboration |
| Retrieval | CRAG | 1st-pass classify | re-evaluate AMBIGUOUS |
| Retrieval | Spec. Rerank | cross-encoder scoring | — |
| Drafting | MASS-RAG | parallel drafting | multi-draft synthesis |
| Drafting | AlignRAG | self-critique rewrite | — |
| Verification | RhinoInsight | claim extraction + trust | — |
| Verification | CONSTRUCT | evidence structuring | — |

---

## 3. Experimental Setup

### System Configurations (35 conditions)

- **Cloud-only (3)**: Sonnet 4.6, Haiku 4.5, Llama 3.3 70B
- **Hybrid (24)**: 8 local models × 3 cloud backends
- **All-local (8)**: 8 local models, no cloud calls

All local models used off-the-shelf with Q4_K_M quantization; no fine-tuning.

### Benchmark

120 multilingual queries: 100 English + 10 Korean + 10 Japanese, spanning 5 domains (AI/ML, Science & Tech, Business, Medical, Law & Policy) and 5 query types. Retrieved documents: Tavily web search, frozen snapshot (35 docs/query).

### Evaluation: Triple Judge Jury

| Judge | Model | Lineage |
|-------|-------|---------|
| Judge A | DeepSeek R1 (671B) | Chinese lab |
| Judge B | Claude Opus 4.6 | Anthropic |
| Judge C | Mistral Large 3 (675B) | European lab |

G-Eval rubric: Coverage, Accuracy, Citation Quality, Depth, Coherence. Final score = median of three judges. N=600 per condition (120 queries × 5 runs). Paired Wilcoxon tests with Bonferroni correction.

---

## 4. Main Results

### Representative Results

| Mode | Local | Cloud | Med ±σ | Δ | Tok/q | $/q |
|------|-------|-------|--------|---|-------|-----|
| **Cloud-only** | — | Sonnet 4.6 | .798 ±.070 | ref | 136.9K | $1.128 |
| Cloud-only | — | Haiku 4.5 | .671 ±.111 | −.127 | 172.0K | $0.376 |
| Cloud-only | — | Llama 70B | .688 ±.078 | −.110 | 70.4K | $0.600 |
| **Hybrid** | exaone3.5:2.4b | Sonnet 4.6 | **.869** ±.059 | **+.071** | 45.9K | $0.375 |
| Hybrid | gemma3:4b | Sonnet 4.6 | .867 ±.046 | +.069 | 47.3K | $0.379 |
| Hybrid | exaone3.5:2.4b | Haiku 4.5 | .828 ±.065 | +.030 | 42.6K | $0.093 |
| Hybrid | gemma3:4b | Haiku 4.5 | .825 ±.055 | +.027 | 44.3K | $0.095 |
| **All-local** | exaone3.5:2.4b | — | .802 ±.085 | +.004 | 0 | $0.000 |
| All-local | gemma3:4b | — | .803 ±.067 | +.005 | 0 | $0.000 |

### Key Finding 1: Every Hybrid Condition Outperforms Its Matched Baseline

Hybrid+Sonnet beats cloud-only Sonnet, Hybrid+Haiku beats cloud-only Haiku, Hybrid+Llama 70B beats cloud-only Llama 70B — consistently across all local model configurations. The quality gain is present regardless of which cloud model handles System 2 stages, ruling out model-diversity as the sole explanation.

### Key Finding 2: 2.4B Local Model Achieves Best Overall Quality

exaone3.5:2.4b + Sonnet 4.6 achieves 0.869 — the highest quality across all 35 conditions. This exceeds cloud-only Sonnet (0.798) by 7.1 points (p < 0.001, Wilcoxon signed-rank).

### Key Finding 3: Hybrid+Haiku Pareto-Dominates Cloud-Only Sonnet

exaone3.5:2.4b + Haiku 4.5 achieves 0.828 at $0.093/query — simultaneously better quality, 12× lower cost, and stronger privacy than cloud-only Sonnet ($1.128/query, 0.798).

### Key Finding 4: All-Local Is Viable

exaone3.5:2.4b all-local (0.802) exceeds cloud-only Haiku (0.671) and cloud-only Llama 70B (0.688) at zero cost. The 6–7 point gap vs. Hybrid+Sonnet is the cost of complete privacy.

---

## 5. Effect of Local Model Scale

Performance does not increase monotonically with local model size. Within Hybrid+Sonnet:

| Local Model | Size | Quality | Latency/q |
|-------------|------|---------|-----------|
| exaone3.5:2.4b | ~2.4B | **0.869** | 233s |
| gemma3:4b | ~4B | 0.867 | 312s |
| qwen3:8b | ~8B | 0.855 | 590s |
| exaone3.5:7.8b | ~7.8B | 0.845 | 382s |
| llama3.1:8b | ~8B | 0.840 | 317s |
| gemma3:12b | ~12B | 0.838 | 525s |

The advantage is model-family dependent: qwen3:4b (0.801) falls below qwen3:8b (0.855) in the same tier, while exaone3.5:2.4b exceeds all larger models.

---

## 6. Privacy Boundary Analysis

Cloud-only pipelines send 70,365–172,017 tokens/query. Hybrid routing reduces this to 10,700–54,731 tokens (65–92% reduction vs. cloud-only Sonnet), with the original query, full documents, and intermediate drafts retained locally.

| Configuration | Cloud Tokens/q | Reduction | Quality |
|---------------|----------------|-----------|---------|
| Cloud-only Sonnet | 136,891 | ref | 0.798 |
| exaone3.5:2.4b + Sonnet | 45,918 | −66.5% | 0.869 |
| qwen3:4b + Llama 70B | 10,700 | −92.2% | 0.799 |

The most privacy-aggressive condition (92.2% reduction) still scores 0.799 — above both weaker cloud-only baselines.

---

## 7. Cloud Model Ablation

Holding the local model fixed (exaone3.5:2.4b):

| Cloud Backend | Quality | Cost/q | vs Cloud-only Sonnet |
|---------------|---------|--------|----------------------|
| Sonnet 4.6 | 0.869 | $0.375 | +0.071 |
| Haiku 4.5 | 0.828 | $0.093 | +0.030 |
| Llama 3.3 70B | 0.793 | $0.128 | −0.005 |

The synthesis model sets a soft quality floor. Practitioners should prioritize cloud backend quality over local model size.

---

## 8. Why Local Models Improve System 1 Stages

Two complementary mechanisms explain the unexpected quality gain:

**Task-scope alignment**: Frontier models over-elaborate outputs for constrained generation tasks (CRAG classification, trust scoring), producing verbose reasoning where a concise label is required. Smaller models stay on-task more reliably.

**Synthesis leverage**: When local models generate section drafts, the cloud synthesis stage receives imperfect but diverse raw material, creating genuine revision leverage absent when a frontier draft is already near-final.

---

## 9. Practical Configuration Tiers

| Tier | Config | Quality | Cost/q | Use case |
|------|--------|---------|--------|----------|
| **Hybrid + Sonnet** | exaone3.5:2.4b + Sonnet 4.6 | 0.869 | $0.375 | Highest quality |
| **Hybrid + Haiku** | exaone3.5:2.4b + Haiku 4.5 | 0.828 | $0.093 | Pareto-optimal (12× cost reduction) |
| **All-local** | exaone3.5:2.4b | 0.802 | $0.000 | Air-gapped / zero cloud cost |

---

## 10. Limitations

**LLM-based evaluation**: Primary metric relies on LLM judges. Mitigated by triple-judge jury from distinct training lineages, structured G-Eval rubric, and condition-level inter-judge correlations (r=0.82–0.85).

**Single retrieval backend**: All experiments use Tavily web search via a frozen snapshot. Performance on other retrieval backends may differ.

**English-centric analysis**: Benchmark includes Korean and Japanese queries, but current analysis is primarily English-focused. Multilingual analysis will be reported in future work.

**Deployment assumptions**: Experiments conducted on NVIDIA L4 (24GB VRAM). On CPU-only or weaker GPU hardware, local model latency would increase substantially.

**Privacy boundary completeness**: The Privacy Boundary prevents document bodies from reaching the cloud, but does not protect against inference attacks on abstractions and drafts. Stronger privacy guarantees remain future work.

---

## 11. Conclusion

Stage-Aware Local-Cloud Inference resolves the privacy-cost-quality trilemma through stage-level reasoning routing. The System 1/System 2 routing principle and Privacy Boundary provide a unifying framework for adapting nine established RAG techniques to this hybrid regime, with privacy enforced by construction.

Our large-scale evaluation demonstrates that off-the-shelf 4B local models, when given the right roles, exceed cloud-only systems in report quality while reducing cloud token exposure by 65–92% and cost by up to 12×.

The privacy-quality trade-off is not an inherent property of the task; it is an artifact of treating every pipeline stage as equally demanding.

---

## Appendix A: Full 35-Condition Results

See `eval/EXPERIMENT_LOG.md` for the complete results table.

## Appendix B: Benchmark Query Examples

| Domain | Analytical | Definitional | Current-state | Comparative | Factual |
|--------|-----------|-------------|---------------|-------------|---------|
| AI/ML | What led to the emergence of large language models? | How does RAG work and what problems does it solve? | What is the current state of AI safety research? | How do frontier LLMs compare on math and coding benchmarks? | What evidence exists for genuinely emergent AI capabilities? |
| Medical | What are the long-term health consequences of COVID-19? | How does CAR-T cell therapy work? | What is the current state of Alzheimer's treatment research? | How does gene therapy compare to CRISPR? | What does evidence show about intermittent fasting? |
| Law & Policy | What are the global implications of the EU AI Act? | How does GDPR work and what rights does it grant? | What is the current state of cryptocurrency regulation? | How does the US approach to data privacy compare to GDPR? | What does research show about gun control laws? |

## Appendix C: Judge Prompt Template

```
You are evaluating a deep research report on the following query: "{query}"

Report:
{report}

---
Evaluate the report on these 5 dimensions.
The report may be written in English, Korean, or Japanese -- evaluate in the report's language.

1. COVERAGE (does the report address all key aspects of the query?)
   9-10: All major angles covered comprehensively
   7-8: Most aspects covered, minor gaps
   5-6: Core question answered but notable gaps
   3-4: Shallow, significant aspects missing
   1-2: Fails to address the query

2. ACCURACY (are claims well-supported and factually sound?)
   9-10: All claims supported, no errors
   7-8: Mostly accurate, minor issues
   5-6: Some unsupported/questionable claims
   3-4: Several doubtful claims
   1-2: Major factual errors

3. CITATION QUALITY (are sources used effectively?)
   9-10: Rich inline citations, evidence clearly linked to claims
   7-8: Good citation practice, mostly well-linked
   5-6: Some citations but inconsistent
   3-4: Few citations, mostly uncited
   1-2: No citations or fabricated refs

4. DEPTH (does the report demonstrate analytical depth?)
   9-10: Insightful synthesis of multiple perspectives
   7-8: Good analysis, some synthesis
   5-6: Descriptive rather than analytical
   3-4: Surface-level summary
   1-2: No meaningful analysis

5. COHERENCE (is the report well-organized and clearly written?)
   9-10: Excellent structure, logical flow
   7-8: Good organization, minor issues
   5-6: Adequate but some disconnected sections
   3-4: Poor organization
   1-2: Incoherent

Briefly reason about each dimension (1 sentence each), then output the score.
Final line must be exactly: SCORE: X.X
```
