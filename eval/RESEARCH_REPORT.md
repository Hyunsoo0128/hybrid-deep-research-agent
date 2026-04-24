# Experimental Report: Deep Research System with State-of-the-Art Multi-Agent RAG Techniques

> Date: 2026-04-24
> Experiment Period: 2026-04-22 ~ 2026-04-24
> Infrastructure: AWS EC2 (g6e.xlarge / t3.xlarge), Amazon Bedrock (Claude Sonnet 4.6), Ollama

---

## 1. Objectives

### 1-1. Background

Autonomous deep research systems powered by large language models (LLMs) aim to automatically generate comprehensive reports on complex questions by searching and synthesizing information from multiple sources. However, existing approaches face two fundamental limitations.

First, **cloud-based large LLMs used alone** can deliver high quality, but all data processed—user queries, retrieved documents, intermediate reasoning results—is transmitted externally through cloud APIs. When handling sensitive data such as internal corporate documents, medical records, or legal materials, this is a fundamental constraint.

Second, **local small LLMs used alone** keep data from leaving the premises, but their reasoning capabilities are insufficient for generating high-quality reports.

### 1-2. Core Research Question

> **When applying techniques proposed in recent multi-agent RAG papers, can a hybrid of small local LLMs and large cloud LLMs achieve commercial-grade deep research quality?**

Specifically, three hypotheses were tested:

1. **Technique effectiveness**: Do methods such as CRAG, AlignRAG, MASS-RAG, DSAP, and Speculative RAG proposed in the literature actually contribute to measurable capability improvements?
2. **Hybrid quality**: Can a local small LLM + cloud large LLM hybrid narrow the quality gap compared to cloud-only operation?
3. **Multilingual generalization**: Are the same results reproducible across English (EN), Korean (KO), and Japanese (JA) environments?

---

## 2. Applied Techniques (Based on 9 Papers)

This system adopts core ideas from 9 papers published between 2024 and 2026. The techniques are classified into 4 groups by **purpose**, and introduced in chronological order within each group.

### Paper Timeline Overview

| Submission Date | arxiv ID | Technique | Purpose Group |
|---|---|---|---|
| 2024-01-29 | 2401.15884 | CRAG | Retrieval Quality Enhancement |
| 2024-07-11 | 2407.08223 | Speculative RAG | Multi-Agent Analysis |
| 2025-04-21 | 2504.14858 | AlignRAG | Hallucination Detection & Quality Control |
| 2025-07-01 | 2507.00355 | Query Decomp + Reranker | Retrieval Quality Enhancement |
| 2025-11-24 | 2511.18743 | RhinoInsight | Hallucination Detection & Quality Control |
| 2025-12-18 | 2512.20660 | DSAP | Planning, Iteration & Reliability |
| 2026-02-24 | 2603.18014 | CONSTRUCT | Hallucination Detection & Quality Control |
| 2026-04-19 | 2604.17405 | STRIDE | Planning, Iteration & Reliability |
| 2026-04-20 | 2604.18509 | MASS-RAG | Multi-Agent Analysis |

---

## 2-1. Retrieval Quality Enhancement

Retrieval is the starting point of a RAG pipeline. The two techniques in this group improve retrieval quality at different layers: CRAG **post-evaluates the reliability** of already-retrieved documents, while Query Decomp + Reranker **pre-decomposes queries** before retrieval and **re-ranks results** afterward.

---

### 2-1-1. CRAG — Corrective Retrieval-Augmented Generation
**Paper**: arxiv:2401.15884 | **Submitted**: January 29, 2024

#### Core Idea
Unlike conventional RAG that assumes retrieved documents are always useful, CRAG first **evaluates** retrieval results and selects one of three paths based on quality.

- **CORRECT** (relevance ≥ 0.5): Decompose into sentences and recompose using only relevant parts (Decompose-then-Recompose)
- **AMBIGUOUS** (0.3 ~ 0.5): Use both internal knowledge and web search in parallel
- **INCORRECT** (< 0.3): Replace with web re-retrieval

The paper reported PopQA +14.5%, PubHealth +13.1% performance gains.

#### Implementation
- 3-way classifier implemented in `src/nodes/search_worker.py`
- Document relevance scoring: LLM-based float scores instead of T5-large fine-tuned model (approximate substitute)
- Decompose-then-Recompose: strip score evaluation + extraction performed in a single LLM call (batch optimization)
- On INCORRECT verdict, re-retrieval is delegated to gap_detector (no pipeline-level retry needed)
- **Hybrid application**: CORRECT/INCORRECT handled locally; only AMBIGUOUS boundary cases (0.3~0.5) escalated to cloud for re-judgment → sensitive data exposed to cloud only minimally at boundary cases

---

### 2-1-2. Query Decomp + Reranker
**Paper**: arxiv:2507.00355 | **Submitted**: July 1, 2025

#### Core Idea
Decomposing complex multi-hop questions into sub-questions for parallel retrieval improves recall, but documents retrieved per sub-question may have lower precision against the original question. The key contribution is a **Reranker** that re-ranks all candidate documents against the **original question** rather than the sub-questions. MRR@10 +36.7%, F1 +11.6% were reported.

#### Implementation
- `src/nodes/plan_generator.py`: Always includes the original query (sq0) in the sub-query list (Q = {q} ∪ Decompose(q))
- `src/nodes/reranker.py`: Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (fastembed ONNX backend, ~80MB)
- Re-ranks full `title + excerpt` against the original query, retaining top-k (by depth: fast=10, normal=20, deep=40)
- URL-based deduplication performed before reranking
- **Fully local execution**: ONNX inference with no cloud contact, no sensitive data transmitted externally

---

## 2-2. Multi-Agent Analysis

Moving beyond a structure where a single LLM exclusively processes retrieval results, multiple agents divide responsibilities to achieve broader coverage and higher quality. Both techniques in this group share a common pattern of **small local agent (drafter) + large cloud agent (verifier)**, which naturally aligns with privacy boundaries and forms the core of the hybrid design.

---

### 2-2-1. Speculative RAG — Draft-Verify-Refine
**Paper**: arxiv:2407.08223 | **Submitted**: July 11, 2024

#### Core Idea

The original paper's goal is **inference speed optimization**. Small drafter models generate multiple RAG drafts in parallel, then a large verifier model selects the best draft in a single forward pass. The approach reduces the number of large model calls while maintaining final quality through the large model.

Core structure:
- **Drafter (small model)**: Generates multiple RAG drafts in parallel → processing speed advantage
- **Verifier (large model)**: Evaluates and selects from drafts in a single pass → quality assurance
- **Inference guidance**: Selected draft + verifier judgment → small model performs final refinement

#### Reinterpretation and Application in This System

Speculative RAG was initially excluded from consideration:
> *"The small drafter + large verifier structure does not fit either mode. Bedrock-only users do not need a drafter, and Ollama-only users cannot benefit without a large verifier."*

However, once the hybrid local/cloud architecture was established, the **local = drafter, cloud = verifier** separation emerged naturally. Furthermore, privacy requirements made this separation a **structural necessity** — if source documents cannot be sent to the cloud, local draft generation becomes the only viable path. Beyond the paper's speed optimization goal, **data boundary enforcement** is the primary adoption rationale in this system.

The Speculative RAG pattern was applied independently in two nodes:

**① Within MASS-RAG (Phase D)**

```
Stage 1  Local Drafter ×3 Parallel
         Summarizer → Full document summarization
         Extractor  → Key fact and citation span extraction
         Reasoner   → Inference and causal relationship derivation
         (Each agent reads source documents directly)
         ↓ Only 3 draft texts forwarded (source documents not forwarded)
Stage 2  Cloud Verifier
         → Contradiction detection across 3 drafts, selection and integration of best elements
         ↓ judgment + instructions
Stage 3  Local Refiner
         → Final refinement incorporating cloud judgment
```

**② Within AlignRAG Critic (Phase E)**

```
Stage 1  Local (suspect_claims draft)
         → Draft list of suspect claims from report
         ↓ Only suspect claim text forwarded (citation source text not forwarded)
Stage 2  Cloud (logical consistency verification)
         → Detect logical errors and inconsistencies across claims
         ↓ judgment
Stage 3  Local (correction hint refinement)
         → Re-reference source evidence to concretize correction guidance
```

#### Pre-Validation of Implementation Suitability

Before applying Speculative RAG to the Critic, local-only (qwen3:8b) performance was pre-measured using `eval/critic_pretest.py`. Result: AlignRAG Phase 3 (comprehensive logical verification) recall=**0.00** — it was confirmed that logical consistency verification is impossible with a small model alone. This experimentally supported the necessity of Stage 2 (cloud verification), and recall improved from 0.00 → **0.67** after application.

---

### 2-2-2. MASS-RAG — Multi-Agent Synthesis RAG
**Paper**: arxiv:2604.18509 | **Submitted**: April 20, 2026

#### Core Idea
Instead of a single LLM analyzing retrieval results, 3 agents with distinct roles independently process in **parallel** and then synthesize.

- **Summarizer**: Summarizes the entire document
- **Extractor**: Extracts key facts and citation spans
- **Reasoner**: Derives inferences, estimates, and causal relationships

The diversity of each agent provides broader coverage than a single agent.

#### Implementation
- 3-agent parallel processing in `src/nodes/search_worker.py` (`asyncio.gather`)
- Output schema: `{summary, key_spans: [{text, source_citation_ids, type}], inferences: [{claim, supporting_span_indices}]}`
- **Hybrid Speculative RAG pattern (see 2-2-1)**:
  - Stage 1 (local): Summarizer, Extractor, Reasoner agents each produce drafts independently
  - Stage 2 (cloud): Verifier detects contradictions across 3 drafts, selects best elements → cloud receives only 3-agent output text (source documents not transmitted)
  - Stage 3 (local): Refiner performs final refinement based on cloud judgment

---

## 2-3. Hallucination Detection and Quality Control

Even when a RAG pipeline retrieves with high precision, factually incorrect claims can be inserted during generation. The three techniques in this group handle different layers of post-generation verification: AlignRAG inspects source alignment of individual claims, RhinoInsight tracks research objective coverage, and CONSTRUCT quantifies field-level trustworthiness of structured outputs in real time.

---

### 2-3-1. AlignRAG — Alignment-Based RAG Critic
**Paper**: arxiv:2504.14858 | **Submitted**: April 21, 2025

#### Core Idea
Detects misaligned claims in generated reports through a 3-phase diagnostic process.

- **Phase 1**: Query-evidence relevance review
- **Phase 2**: Document-claim alignment inspection
- **Phase 3**: Comprehensive logical consistency verification

Detected misaligned claims are forwarded to the revision stage along with correction hints.

#### Implementation
- `src/nodes/critic.py` fully rewritten
- `misaligned_claims`: Stored in detail as `[{phase, claim, source_citation_ids, source_quote, correction_hint}]`
- `passed` value computed in code, not from LLM output: `len(misaligned)==0 and len(uncited)==0 and len(unanswered)==0`
- **Hybrid Speculative RAG pattern (see 2-2-1)**:
  - Stage 1 (local): Draft suspect_claims (recall limited if LLM capability is low)
  - Stage 2 (cloud): Logical consistency verification → cloud receives only claim text (citation source text not transmitted)
  - Stage 3 (local): Correction hint refinement
- Pre-test (eval/critic_pretest.py) confirmed qwen3:8b Phase 3 recall=0.00 → cloud verification deemed mandatory

---

### 2-3-2. RhinoInsight — VCM + EAM
**Paper**: arxiv:2511.18743 | **Submitted**: November 24, 2025

#### Core Idea
Addresses "context rot" in linear pipelines — the phenomenon where initial research objectives become diluted across multiple retrieval cycles — using two modules.

- **VCM (Verification Checklist Module)**: Generates a sub-objective checklist before research begins, and tracks achievement after each retrieval cycle. Unmet items are forwarded to gap detection.
- **EAM (Evidence Audit Module)**: Normalizes retrieval results into a structured evidence repository, and attaches misalignment flags to claims identified by the Critic stage.

#### Implementation
- `src/nodes/checklist_node.py`: Generates verifiable sub-objective checklists from planned sub-queries
- `src/nodes/evidence_auditor.py`:
  - Stage 1: Normalizes retrieval results into evidence_store + deduplication + verification level classification
  - Stage 2: Runs after Critic, attaches misalignment flags to evidence items based on `source_citation_ids` in `misaligned_claims`
- Unmet checklist items → forwarded as VCM hints to gap_detector
- **Fully local execution**

---

### 2-3-3. CONSTRUCT — Real-Time Trustworthiness Scoring
**Paper**: arxiv:2603.18014 | **Submitted**: February 24, 2026

#### Core Idea
Uses a Judge LLM to score the trustworthiness of each field in LLM-generated structured output in real time. Operates on black-box APIs without logprobs or fine-tuning, enabling selective regeneration of only low-trust fields.

#### Implementation
- `src/nodes/quality_scorer.py`: Document-level + Field-level 2 templates executed in parallel via `asyncio.gather`
- Applied to MASS-RAG 3-agent synthesis output (`construct=True` feature flag)
- Low-trust fields → annotated as "LOW TRUST" in writer's synthesis block → Critic notified
- Threshold: `TRUST_THRESHOLD = 0.5`
- **Fully local execution**

---

## 2-4. Planning, Iteration, and Reliability

As the retrieval-generation loop repeats, the exploration scope can converge inappropriately, and structured output parsing failures accumulate. The two techniques in this group improve **planning abstraction (STRIDE)** and **output recovery stability (DSAP)** respectively, increasing the overall reliability of the pipeline.

---

### 2-4-1. DSAP — Dual-State Architecture for Reliable LLM Agents
**Paper**: arxiv:2512.20660 | **Submitted**: December 18, 2025

#### Core Idea
When an LLM fails to produce structured output (e.g., JSON), DSAP recovers by switching between two states rather than simple retry.

- **Level 1 (error context)**: Retry with previous error message included ("a previous error occurred — please correct it")
- **Level 2 (clean slate)**: If the same error repeats consecutively, discard the contaminated context entirely and retry with only the schema

#### Implementation
- `src/utils/llm_json.py`:
  - Error fingerprint normalization: detect identical error patterns
  - Consecutive identical fingerprints → switch to Last Resort strategy (`_LAST_RESORT_SYSTEM` + schema-only prompt)
  - `error_sink: list`, `caller_tag: str` parameters connected to MASS-RAG 3-agents
- **Fully local execution**: JSON parsing recovery is error-handling logic with no cloud contact

---

### 2-4-2. STRIDE — Strategic Iterative Decision-Making for RAG
**Paper**: arxiv:2604.17405 | **Submitted**: April 19, 2026

#### Core Idea
Addresses "premature entity grounding" in iterative RAG — the phenomenon where focusing on specific entities too early narrows the exploration scope — using a Meta-Planner. The Meta-Planner first generates **abstract strategies (Sq)** without specific entities, then converts them into concrete plans (Cq). Results on 2WikiMultihopQA: +7.0% EM, token usage -54~-71%.

#### Implementation
- `src/nodes/plan_generator.py`: When `stride=True`, `_generate_abstract_strategy()` → generates Sq with `[ENTITY]` slots → concretized to Cq
- `src/nodes/supervisor.py`: Assigns retrieve/rewrite/answer actions to each sub-query based on CRAG verdict results. On rewrite, the reformulated question is forwarded as STRIDE hints to gap_detector.
- Unimplemented: dependency graph (Ω) — the paper's dependency-aware parallel scheduling requires LangGraph redesign and was not implemented. Extractor/Reasoner roles are covered by MASS-RAG.

---

## 3. Hybrid Architecture and Privacy Design

### 3-1. Division of Roles: Cloud LLM vs. Local LLM

The entire system operates with two types of LLM instances.

| Type | Model | Role |
|---|---|---|
| **Cloud** | Claude Sonnet 4.6 (Amazon Bedrock) | Final report synthesis, complex logical judgment, gap reasoning, 3-agent result verification |
| **Local** | qwen3:8b / gemma3:4b, etc. (Ollama) | Query decomposition, relevance evaluation, MASS-RAG sub-agents, trustworthiness scoring, JSON parsing recovery, report revision |

Out of all LLM calls, local calls account for approximately 18~22, and cloud calls approximately 6~8.

#### 5-Layer Pipeline

```
Stage 1  Planning
         - Query dimension decomposition    → Local (original query not exposed)
         - STRIDE abstract→concrete strategy elaboration → Cloud (dimension labels only)
         - Verification objective checklist generation → Local

Stage 2  Document Retrieval and Evaluation  (mostly local)
         - CRAG document relevance evaluation → Local
         - CRAG boundary case re-judgment  → Cloud (relevance summary only)
         - MASS-RAG 3-agent parallel analysis → Local
         - Original query-based reranking (Reranker) → Local
         - CONSTRUCT trustworthiness scoring → Local

Stage 3  Multi-Agent Verification
         - MASS-RAG Verifier   → Cloud (receives only 3-agent output text)

Stage 4  Information Synthesis
         - MASS-RAG Refiner    → Local
         - Uncovered area detection → Cloud (based on synthesis results)
         - Evidence Audit (EAM)  → Local

Stage 5  Report Generation and Quality Inspection
         - Report writing (Writer)  → Cloud
         - AlignRAG suspect claims draft → Local
         - AlignRAG logical consistency verification → Cloud (claim text only)
         - AlignRAG correction guidance refinement → Local
         - Report revision (Revise)  → Local
```

### 3-2. Privacy Protection Design Principles

#### Architectural Privacy Potential

In this architecture, **what the cloud receives are derivatives of the original data**.

| Pipeline Stage | Cloud Input | Original Data |
|---|---|---|
| Plan Stage 2 (STRIDE) | Dimension labels ("trade-off analysis", "definition/background", etc.) | Original query text not transmitted |
| CRAG AMBIGUOUS re-judgment | Relevance summary text | Original document body not transmitted |
| MASS-RAG Verifier | Local 3-agent output | Original retrieved documents not transmitted |
| AlignRAG Verify | Suspect claim text | Citation source text not transmitted |
| Writer | Web excerpts (public) + MASS-RAG synthesis results | Local internal file source text not transmitted |

The original query, original retrieved documents, and local internal file text are designed to never reach the cloud. The cloud always receives "abstracted derivatives."

#### Enterprise Use Case Scenario

Consider a pharmaceutical company automating deep research that compares internal clinical trial documents against competitor publications.

- **Risk of internal documents**: If clinical data, research hypotheses, and patient information are transmitted to an external cloud API, this constitutes a regulatory violation or trade secret leak.
- **In this architecture**: Local MASS-RAG agents read internal documents and perform summarization, extraction, and reasoning. Only these 3 summary texts are forwarded to the Cloud Verifier. The source text of internal documents never reaches the cloud.
- **Cloud's role**: Detecting contradictions across 3 summaries and selecting the best elements — a task performable without the source text.

This pattern was inspired by the "small drafter + large verifier" structure of Speculative RAG (arxiv:2407.08223). While the original Speculative RAG was designed for speed optimization, the **drafter (local) → verifier (cloud)** separation naturally forms a privacy boundary in a hybrid local/cloud architecture, and was adopted as a design principle on those grounds.

> **Note**: Source documents used in this experiment consisted of public materials including arxiv papers and publicly available technical blogs. The privacy protection characteristics described above are design-level principles; the actual protective effect on sensitive data requires separate validation.

---

## 4. Experimental Design

### 4-1. Two Experimental Conditions

| Condition | Name | Configuration | Infrastructure |
|---|---|---|---|
| **C1** | Bedrock Sonnet-only | Claude Sonnet 4.6 performs all roles alone. Quality upper bound. | t3.xlarge |
| **C2** | Hybrid | Local roles (query decomposition, relevance evaluation, Critic, sub-agents) → Ollama. Final synthesis → Sonnet | g6e.xlarge (NVIDIA L40S GPU) |

### 4-2. Evaluation Methodology

**Unit Function Tests — Per-Technique Individual Verification (T1~T4)**

Independently measures how well local models perform the core role of each technique. DSAP before/after comparison also measures error recovery effectiveness.

| Test | What is Measured | Related Technique |
|---|---|---|
| T1 (JSON Parsing) | Structured JSON output success rate | DSAP |
| T2 (CRAG) | Relevance classification precision/recall/F1 | CRAG |
| T3 (AlignRAG Critic) | Detection rate of 3 injected errors | AlignRAG |
| T4 (MASS-RAG) | Sub-agent structured output success rate | MASS-RAG, DSAP |

**Comprehensive Report Quality Test — End-to-End Evaluation (C1/C2)**

Completed reports for 2 real research queries are scored by Claude Sonnet.

- Query 1: *"What are the main techniques for optimizing LLM inference performance, and how do they compare in terms of speed, memory, and accuracy trade-offs?"*
- Query 2: *"How does retrieval-augmented generation (RAG) improve the accuracy of LLMs, and what are the key challenges in implementing it effectively?"*

Scoring criteria: coverage, accuracy, specificity, structure → overall composite score (0.0~1.0)

### 4-3. Evaluation Languages

The same pipeline was run three times across 3 languages. Source documents remain in English (cross-lingual RAG); only the query and report language changes.

| Language | Query Example (Q1) |
|---|---|
| EN | "What are the main techniques for optimizing LLM inference performance, and how do they compare in terms of speed, memory, and accuracy trade-offs?" |
| KO | "LLM 추론 성능을 최적화하는 주요 기법은 무엇이며, 속도·메모리·정확도 트레이드오프 측면에서 각 기법을 어떻게 비교할 수 있는가?" |
| JA | "LLM推論パフォーマンスを最適化する主な手法は何か、また速度・メモリ・精度のトレードオフの観点からそれらをどのように比較できるか？" |

### 4-4. Models Under Evaluation (C2 Local Models)

| Model | Parameters | Notes |
|---|---|---|
| qwen3:4b | ~4B | Qwen3 series minimum |
| gemma3:4b | ~4B | Google Gemma3 |
| phi4-mini:latest | ~3.8B | Microsoft Phi4 |
| qwen3:8b | ~8B | Qwen3 8B |
| exaone3.5:7.8b | ~7.8B | LG AI Research (Korean specialist) |
| llama3.1:8b | ~8B | Meta LLaMA 3.1 |
| qwen3:14b | ~14B | Qwen3 14B |
| gemma3:12b | ~12B | Google Gemma3 12B |

---

## 5. Experimental Results

### 5-1. Unit Function Test Results

Combined results from 2 runs under English (EN) baseline:

| Model | T1 JSON (DSAP on) | T2 CRAG (DSAP on) | T3 AlignRAG (DSAP on) | T4 MASS-RAG (DSAP on) | Unit Test Overall |
|---|---|---|---|---|---|
| exaone3.5:7.8b | 1.000 | 0.950 | 1.000 | 1.000 | **0.989** |
| gemma3:12b | 1.000 | 0.955 | 1.000 | 0.933 | **0.979** |
| llama3.1:8b | 1.000 | 0.950 | 1.000 | 0.933 | **0.979** |
| gemma3:4b | 1.000 | 1.000 | 1.000 | 0.867 | **0.955** |
| qwen3:14b | 1.000 | — | 1.000 | 1.000 | **0.844\*** |
| qwen3:8b | 1.000 | — | 1.000 | 1.000 | **0.844\*** |
| phi4-mini:latest | 0.750 | 0.333 | 1.000 | 0.733 | **0.518** |
| qwen3:4b | 0.000 | — | 0.000 | 0.000 | **0.094\*** |

\* qwen3 series: T2 unmeasurable (empty output) — affected by `/no_think` bug

**DSAP effect observed**: The change from DSAP disabled to enabled is pronounced in T1. For example, gemma3:4b improved T1 from 0.750 (disabled) to 1.000 (enabled), and T4 from approximately 0.0 (disabled) to 0.867 (enabled), confirming the DSAP retry mechanism's quantitative impact.

**AlignRAG Critic T3 detailed results**: Detection rate for 3 injected errors (①1000× speed exaggeration, ②accuracy figure error, ③claim of non-existent feature):

| Model | Detected / 3 |
|---|---|
| exaone3.5:7.8b | 3/3 ✅ |
| gemma3:12b | 3/3 ✅ |
| llama3.1:8b | 3/3 ✅ |
| gemma3:4b | 1/3 → 3/3 (after DSAP retry) |
| qwen3:4b | 0/3 ❌ |

**CRAG T2 Precision/Recall (gemma3 series)**:

| Model | Precision | Recall | F1 | Noise Reduction |
|---|---|---|---|---|
| gemma3:4b | 1.000 | 1.000 | 1.000 | 1.000 |
| gemma3:12b | 1.000 | 0.830 | 0.910 | 1.000 |

---

### 5-2. Comprehensive Report Quality Comparison — C1 (Cloud-only) vs C2 (Hybrid) (English)

> C1 baseline (Bedrock Sonnet-only): **0.940** (Q1=0.93, Q2=0.95)

| Model | C2 overall | Gap vs C1 | Revision Count | Latency | Notes |
|---|---|---|---|---|---|
| **C1 Sonnet** | **0.940** | baseline | 0.5 | 257s | Upper bound |
| gemma3:4b | **0.895** | −0.045 | 2.0 | 216s | ✅ |
| llama3.1:8b | **0.880** | −0.060 | 2.0 | 183s | ✅ |
| gemma3:12b | **0.855** | −0.085 | 2.0 | 299s | ✅ (hallucination caution) |
| exaone3.5:7.8b | **0.780** | −0.160 | 2.0 | 209s | ⚠️ large gap |
| phi4-mini:latest | **0.775** | −0.165 | 2.0 | 170s | ⚠️ large gap |
| qwen3:8b | **0.735** | −0.205 | 1.5 | 303s | ⚠️ report revision stage abbreviated |
| qwen3:14b | **0.500** | −0.440 | 2.0 | 400s | ❌ report revision stage severely abbreviated |
| qwen3:4b | **0.955\*** | — | 0.0 | 238s | ⚠️ **invalid** |

\* qwen3:4b C2=0.955 reflects no local contribution — Critic failed entirely, preserving Sonnet's draft as-is. Effectively identical to C1.

---

### 5-3. Actual Report Output Excerpts

#### [EN C1] Sonnet-only — Q1 LLM Inference Optimization (overall=0.93)

```markdown
# Optimizing LLM Inference Performance: Techniques, Trade-offs, and Benchmarks

## Introduction
As large language models scale to tens or hundreds of billions of parameters,
deploying them efficiently at inference time has become a critical engineering
challenge. A suite of complementary optimization techniques—spanning memory
management, numerical precision reduction, and algorithmic improvements—has
emerged to address bottlenecks in speed, memory footprint, and cost.

## Memory and Precision Optimization
Quantization is the most widely adopted technique for reducing memory consumption.
Post-training quantization converts FP16 weights to lower-precision formats:
INT8 delivers a 2× memory reduction with less than 1% accuracy loss on most
benchmarks, while INT4 achieves a 4× reduction at the cost of 2–5% accuracy
degradation [d3]. Advanced methods mitigate this degradation: GPTQ applies
layer-wise quantization with Hessian-based weight compensation, and AWQ
(Activation-aware Weight Quantization) preserves approximately 1% of salient
weights to maintain representational fidelity [d3].

...

Speculative decoding demonstrates task-dependent speedups: coding tasks
achieve 3.5×, instruction-following 2.8×, and creative writing 1.8× [d2],
with no accuracy loss.
```

#### [EN C2 gemma3:4b] Hybrid — Q1 LLM Inference Optimization (overall=0.94)

```markdown
# Optimizing LLM Inference Performance: Techniques, Trade-offs,
  and Practical Considerations

## Introduction
Large language models are rapidly scaling in size, presenting significant
challenges for efficient inference. A range of complementary methods—spanning
memory management, numerical precision reduction, algorithmic improvements,
and batching strategies—have emerged to address the computational bottlenecks
inherent in autoregressive decoding.

## Memory and Compute Efficiency: KV Caching and Flash Attention
KV caching is a cornerstone of efficient autoregressive decoding. Without it,
each new token generation step would require computations across all transformer
layers – a costly O(n) operation. KV caching dramatically reduces this to O(1)
after the initial "prefill" phase [d1]. However, this caching mechanism comes
with a memory cost: the KV cache scales linearly with the number of layers,
heads, head dimension, sequence length, and batch size
(2 × layers × heads × head_dim × seq_len × batch_size) [d1].

Flash Attention eliminates O(N²) memory for attention by using tiling,
achieving 2–4× real-time speedup on A100 GPUs [d5].
```

> **Scoring comparison**: Both C1 (Sonnet-only) and C2 (gemma3:4b hybrid) achieved overall scores in the 0.93~0.94 range. The two reports differ somewhat in the techniques covered and citation style, but are nearly equivalent in quality.

#### [KO C1] Sonnet-only Korean Report Excerpt (overall=0.91)

```markdown
# LLM 추론 성능 최적화 기법: 속도·메모리·정확도 트레이드오프 분석

## 1. 서론
대규모 언어 모델(LLM)의 추론 비용과 지연시간은 실제 프로덕션 환경 배포의
핵심 병목으로 작용한다. 이를 해결하기 위해 양자화(Quantization), KV 캐시,
투기적 디코딩(Speculative Decoding), 플래시 어텐션(Flash Attention),
연속 배치(Continuous Batching) 등 다양한 최적화 기법이 개발되었다.

## 2.1 메모리 효율 중심 기법: 양자화와 KV 캐시
INT8은 메모리를 2배 절감하면서 정확도 손실이 1% 미만에 그치며, INT4는
메모리를 4배 절감하지만 정확도 손실이 2~5%로 증가한다 [d3]. GPTQ는
헤시안(Hessian) 기반 가중치 보정을 통해 양자화 오류를 최소화하고,
AWQ(Activation-aware Weight Quantization)는 중요 가중치 상위 1%를
보존함으로써 더 나은 정확도를 달성한다 [d3].

투기적 디코딩은 소형 드래프트 모델(예: 1B)이 K개의 토큰을 선제안하고,
대형 타깃 모델(예: 70B)이 단일 순전파로 일괄 검증한다 [d2]. 태스크별
속도 향상은 코딩 3.5배, 지시 따르기 2.8배, 창의적 글쓰기 1.8배 [d2].
```

#### [JA C2 gemma3:4b] Japanese Hybrid Report Excerpt (overall=0.88)

```markdown
## 大規模言語モデル（LLM）推論パフォーマンス最適化手法の比較分析

### はじめに
大規模言語モデル（LLM）の推論コストは、実運用における主要なボトルネックと
なっており、スループット・レイテンシ・メモリ使用量の最適化が急務となっている。

### 1. メモリ削減を目的とした量子化手法
INT8（8ビット）量子化はメモリを約2倍削減し、精度劣化は平均して1%未満に
抑えられる [d3]。INT4（4ビット）量子化はメモリを4倍削減できるが、精度劣化は
2〜5%程度生じる [d3]。GPTQ（Hessian行列を用いたレイヤーごとの重み補正）や
AWQ（重要な重みの上位1%を保持するActivation-aware Weight Quantization）と
いった高度な手法は、精度劣化を最小化しながら高い圧縮率を実現する [d3]。

### 2. KVキャッシュと連続バッチ処理
KVキャッシュは自己回帰デコーディング中にアテンションのキーと値を保存・再利用
することで、各ステップの計算量をO(n)からO(1)に削減する [d1]。
PagedAttentionはKVキャッシュを非連続メモリに格納して断片化を解消し、
静的バッチ対比5〜23倍のスループット向上を実現する [d4]。
```

#### [Notable Case] gemma3:12b — Hallucination URL Insertion (overall=0.86)

Despite having the highest unit test overall score (0.979), gemma3:12b exhibited a pattern of inserting non-existent URLs and papers into the final report.

```markdown
## References
- [1] AWQ: Activation-aware Weight Quantization for LLM Inference Optimization.
  Microsoft Research.
  https://www.microsoft.com/en-us/research/blog/awq-activation-aware-...
  ← Scorer: "potentially fabricated URL"

- [2] A Comprehensive Survey on LLM Inference Optimization.
  https://arxiv.org/abs/2310.04346
  ← Scorer: "unverifiable arxiv link"

- [3] FlashAttention-2: Faster Attention with Better Parallelism.
  https://flashattention.com/flashattention-2/
  ← Scorer: "non-existent domain"

- Zhao et al. (2023). [Insert Title of Zhao et al. Paper Here].
  [Insert Link or DOI Here]
  ← Model output the placeholder as-is
```

> Scorer comment: *"The report introduces several unverifiable or potentially fabricated claims... specific Microsoft blog URL, an arxiv link that appears constructed rather than cited from source material."*

This case illustrates that high unit function test scores (CRAG, AlignRAG, etc.) do not guarantee factual reliability in the final report. The AlignRAG Critic attempted to detect this, but fabricated URLs constitute hallucinatory generation rather than logical misalignment and fall outside the scope of the technique.

---

### 5-4. Multilingual Experiment Results — EN / KO / JA Comparison

#### C1 Baseline (Sonnet-only) by Language

| Language | avg overall | Q1 | Q2 | Latency |
|---|---|---|---|---|
| EN | **0.940** | 0.93 | 0.95 | 257s |
| JA | **0.935** | 0.93 | 0.94 | 234s |
| KO | **0.910** | 0.91 | 0.91 | 221s |

Sonnet maintained 0.91~0.94 quality across all three languages.

#### C2 Hybrid — Overall Scores by Language

| Model | EN | KO | JA | KO−EN | JA−EN |
|---|---|---|---|---|---|
| **gemma3:4b** | **0.895** | **0.890** | **0.885** | −0.005 | −0.010 |
| **llama3.1:8b** | **0.880** | **0.785** | **0.850** | −0.095 | −0.030 |
| **gemma3:12b** | **0.855** | **0.805** | **0.820** | −0.050 | −0.035 |
| **phi4-mini** | **0.775** | **0.790** | **0.705** | +0.015 | −0.070 |
| **exaone3.5:7.8b** | **0.780** | **0.630** | **0.675** | −0.150 | −0.105 |
| **qwen3:14b** | **0.500** | **0.715** | **0.720** | +0.215 | +0.220 |
| **qwen3:8b** | **0.735** | **0.275** | **0.000** | −0.460 | −0.735 |
| qwen3:4b | (invalid) | (failed) | (invalid) | — | — |

#### [Notable Case] qwen3:8b — Complete Failure in Japanese (JA=0.000)

qwen3:8b scored 0.735 in EN but 0.000 in JA. Actual report content:

```
final_report: ""  (empty string)

Scorer: "No report content was provided to evaluate against the research
question and reference materials."
"The report field is completely empty, containing no content whatsoever."
```

Cause: When the report revision stage prompt contains Japanese instructions (`IMPORTANT: Write the entire report in Japanese`), qwen3:8b re-enters its internal thinking mode. The thinking suppression token (`/no_think`) conflicts with non-English instructions, suppression fails, and an empty string is returned after thinking tag removal processing.

#### [Notable Case] qwen3:14b — Reversal in Korean/Japanese (KO=0.715, JA=0.720 vs EN=0.500)

qwen3:14b ranked last in EN at 0.500, but scored 0.715 and 0.720 in KO and JA respectively. Report excerpt:

```markdown
# LLM 추론 성능 최적화 기법: 속도·메모리·정확도 트레이드오프 분석

## 1. 서론
대규모 언어 모델(LLM)의 추론 비용과 지연 시간은 실제 서비스 배포에서
핵심적인 병목 요인으로 작용한다.

## 2.1 메모리 효율화 기법: 양자화와 KV 캐시
INT8 양자화는 메모리를 2배 절감하면서 정확도 손실이 1% 미만에 그친다 [d3].
AWQ는 중요 가중치 상위 1%를 보존하여 더 우수한 정확도를 달성한다 [d3].
KV 캐시는 O(n)의 연산을 감소시킨다 [d1]. 다만 메모리 사용량은
`2 × 레이어 수 × 헤드 수 × 헤...  ← 보고서가 여기서 잘림
```

Scorer: *"The report covers quantization, KV cache, FlashAttention, and PagedAttention with accurate technical details and good structure, but appears incomplete (section 2.2 cuts off mid-sentence)."*

The report was truncated at 1,237 characters (approximately 1/4 the length of the EN report), yet received 0.715 — higher than EN=0.500. This appears to be because Korean conveys more information per token than English. In EN, only half the full report was generated, omitting key sections such as speculative decoding and detailed quantization analysis. In KO, the same token budget allowed core content to be expressed more compactly.

---

## 6. Key Findings

### Finding 1 — Hybrid C2 Achieves Quality Comparable to C1 (for Specific Models)

gemma3:4b (C2=0.895) and llama3.1:8b (C2=0.880) maintained quality at −0.045 and −0.060 respectively relative to Sonnet-only (C1=0.940). These two models, handling query decomposition, relevance evaluation, and critique locally while delegating only final synthesis to Sonnet, demonstrated that substantial quality is achievable in this hybrid configuration.

### Finding 2 — Unit Function Tests Do Not Directly Correspond to Comprehensive Report Quality

exaone3.5:7.8b showed the highest individual unit test composite score (0.989) but achieved a significantly lower comprehensive report quality score (0.780). In contrast, gemma3:4b scored 0.955 on unit tests but achieved the best comprehensive report quality at 0.895. High individual function test scores do not necessarily translate to strong performance across the full pipeline.

### Finding 3 — Quantitative Effect of DSAP

T1 JSON parsing change from DSAP disabled to enabled: gemma3:4b 0.750→1.000, gemma3:12b 0.250→1.000. T4 MASS-RAG also improved from approximately 0 (disabled) to approximately 0.9 (enabled). The contribution of error-context retry and clean slate switching to actual parsing success rates is numerically confirmed.

### Finding 4 — Multilingual Stability of gemma3:4b

gemma3:4b scored EN=0.895, KO=0.890, JA=0.885, with a cross-language variance within ±0.01. This is the most stable multilingual pattern among all 8 models.

### Finding 5 — Multilingual Performance Degradation of exaone3.5:7.8b

exaone3.5:7.8b, LG AI Research's Korean specialist model, recorded −0.150 in KO compared to EN. In the cross-lingual structure of synthesizing English source documents into Korean, the model with strong Korean generation capability exhibited unexpectedly unstable behavior. However, the cause cannot be definitively identified from this experiment alone.

### Finding 6 — Revise Stage Constraints in qwen3 Series

Both qwen3:8b (KO=0.275, JA=0.000) and qwen3:14b (EN=0.500) shared a pattern of extremely short or empty reports at the revision stage. In this pipeline, the revision stage is handled by the local model, and qwen3 series models exhibited a tendency to terminate early during long report generation despite a `max_tokens=1500` setting. In non-English environments, an additional problem was observed where non-English instructions re-triggered thinking mode.

### Finding 7 — Hallucination Pattern in gemma3:12b

gemma3:12b, one of the models with the highest individual unit test composite score (0.979), exhibited a pattern of generating non-existent URLs and paper citations in the final report. Unit function scores and factual reliability are distinct dimensions.

---

## 7. Conclusion

This experiment constructed a deep research pipeline integrating state-of-the-art multi-agent RAG techniques (CRAG, Query Decomp, RhinoInsight, MASS-RAG, AlignRAG, DSAP, STRIDE, CONSTRUCT) in a hybrid local/cloud architecture, and evaluated 8 small local LLMs across English, Korean, and Japanese environments.

In the **C1 (cloud-only) vs C2 (hybrid) comparison**, gemma3:4b and llama3.1:8b narrowed the quality gap to −0.045 and −0.060 respectively. This demonstrates that certain small local LLMs can collaborate effectively with cloud models within a paper-based multi-agent architecture. In contrast, qwen3 series models failed to deliver sufficient performance in this pipeline due to constraints related to thinking suppression tokens.

In **multilingual experiments**, Sonnet (C1) maintained stable quality across all three languages, while local models showed variation by language. gemma3:4b was the most stable, with variance within ±0.01 across all three languages.

Note that source documents in this experiment consisted of public materials, and all comprehensive report quality evaluations were scored by Claude Sonnet as the judge. These two conditions should be taken into account when interpreting the results.
