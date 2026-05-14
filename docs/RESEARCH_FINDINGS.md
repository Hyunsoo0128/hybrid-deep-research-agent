# Research Findings — Stage-Aware Local-Cloud Inference

Summarizes the key findings from the paper and the applied academic techniques.

> **Paper**: Stage-Aware Local-Cloud Inference: Hybrid Pipelines Consistently Outperform Matched Cloud-Only Baselines

---

## 1. Core Finding: The Trilemma is an Architectural Artifact

The privacy-cost-quality trilemma in regulated-industry LLM deployments is not an inherent constraint — it is a consequence of treating all pipeline stages as equally demanding.

**The trilemma**:
- Privacy: Cloud APIs route queries and documents through third-party servers
- Cost: Deep research pipelines push per-query costs above $1 at frontier pricing
- Quality: Local-only pipelines degrade analytical depth

**The resolution**: Stage-Aware Local-Cloud Inference routes each stage to the appropriate compute tier based on reasoning demand. System 1 stages (bounded-context operations on single documents) run locally on 2–4B models. System 2 stages (integrative reasoning across multiple sources) run on frontier cloud LLMs.

**Empirical validation**: Every hybrid condition outperforms its matched cloud-only baseline across 35 conditions, 120 queries × 5 runs, evaluated by a Triple Judge Jury.

---

## 2. Applied Academic Paper Techniques

### 2.1 CRAG — Corrective Retrieval-Augmented Generation
**Paper**: arxiv:2401.15884 | **Phase**: Retrieval

**Core idea**: Evaluate retrieved documents before generation and take corrective action based on quality. Three-way branch: CORRECT (≥0.5) → Decompose-then-Recompose, AMBIGUOUS (0.3–0.5) → cloud re-adjudication, INCORRECT (<0.3) → gap_detector delegation.

**Stage-Aware adaptation**: CRAG classification runs locally (System 1 — single document at a time). Only AMBIGUOUS boundary cases are escalated to the cloud using title + 150-character excerpt only — never full document bodies.

**Implementation**: `src/nodes/search_worker.py`

**Finding**: Small local models over-predict AMBIGUOUS due to low calibration confidence. Cloud re-adjudication of AMBIGUOUS cases (without document bodies) resolves this over-conservative tendency.

---

### 2.2 Query Decomposition + Speculative Reranking
**Paper**: arxiv:2507.00355 | **Phase**: Planning + Retrieval

**Core idea**: Decompose complex queries into sub-questions for parallel retrieval (improves recall), then re-rank all candidate documents against the **original query** (restores precision). The Reranker is the paper's core contribution.

**Stage-Aware adaptation**: Entirely System 1 — no cloud contact. ONNX cross-encoder inference runs locally.

**Implementation**: `src/nodes/plan_generator.py`, `src/nodes/reranker.py`

**Finding**: Original query always preserved in sub-query list (Q = {q} ∪ Decompose(q)). Cross-encoder/ms-marco-MiniLM-L-6-v2 via fastembed (~80MB, ONNX backend).

---

### 2.3 STRIDE — Strategic Iterative Decision-Making
**Paper**: arxiv:2604.17405 | **Phase**: Planning

**Core idea**: Meta-Planner generates an abstract strategy Sq based on entity types (not specific entities), then converts to concrete plan Cq. Supervisor manages execution state, decides retrieve/rewrite/answer per sub-query.

**Stage-Aware adaptation**: Sq generation runs locally (System 1 — bounded to the query). Cq elaboration runs on the cloud (System 2 — requires frontier reasoning), receiving only the skeleton. Original query never transmitted to cloud.

**Implementation**: `src/nodes/plan_generator.py`, `src/nodes/supervisor.py`

**Finding**: The Sq→Cq split is the key privacy mechanism for the planning phase. The cloud elaborates from an entity-agnostic skeleton, not from the raw query.

---

### 2.4 MASS-RAG — Multi-Agent Synthesis RAG
**Paper**: arxiv:2604.18509 | **Phase**: Drafting

**Core idea**: Three parallel specialist agents (Summarizer, Extractor, Reasoner) process retrieved documents from different angles. A Synthesis Agent reconciles the three outputs.

**Stage-Aware adaptation**: All three drafting agents run locally (System 1 — each processes a bounded document set). Synthesis runs on the cloud (System 2 — integrates across multiple agent outputs), receiving only the locally-generated draft text — never the underlying documents.

**Implementation**: `src/nodes/search_worker.py`

**Finding**: Local models generate imperfect but diverse raw material, creating genuine revision leverage for the cloud synthesis stage. This is absent when a frontier draft is already near-final — explaining why hybrid outperforms cloud-only.

---

### 2.5 AlignRAG — Alignment-Based RAG Critic
**Paper**: arxiv:2504.14858 | **Phase**: Verification

**Core idea**: Detect misaligned claims in generated reports through 3-phase diagnosis: relevance assessment, query-evidence mapping, evidence-integrated synthesis.

**Stage-Aware adaptation**: Entirely System 1 — self-critique runs locally. The local model checks its own draft against the evidence store without cloud involvement.

**Implementation**: `src/nodes/critic.py`

**Finding**: AlignRAG self-critique is a bounded-context operation (checking one claim against one source at a time) — well within System 1 capability. No cloud contact required.

---

### 2.6 DSAP — Dual-State Architecture for Reliable LLM Agents
**Paper**: arxiv:2512.20660 | **Phase**: Cross-cutting

**Core idea**: Guard functions with three-level recovery. Level 1: retry with accumulated error feedback. Level 2: stagnation detection + strategy switch. Level 3: human escalation.

**Stage-Aware adaptation**: Applied to all local model calls. Critical for System 1 reliability — local models produce more malformed JSON than frontier models.

**Implementation**: `src/utils/llm_json.py`

**Finding**: DSAP Level 1+2 is the reliability layer that makes System 1 local model calls production-viable. Without it, JSON parse failures would cascade through the pipeline.

---

### 2.7 RhinoInsight — VCM + EAM
**Paper**: arxiv:2511.18743 | **Phase**: Planning + Verification

**Core idea**: VCM (Verification Checklist Module) generates a structured checklist of sub-goals before research begins, tracks completion, surfaces uncovered goals to gap detection. EAM (Evidence Audit Module) normalizes retrieved results into a structured evidence store and binds specific citations to specific claims.

**Stage-Aware adaptation**: Entirely System 1 — checklist generation and evidence normalization are bounded-context operations.

**Implementation**: `src/nodes/checklist_node.py`, `src/nodes/evidence_auditor.py`

**Finding**: VCM pending/partial subgoals feed into gap_detector as an independent hint axis, complementing CRAG signals and STRIDE rewrite signals.

---

### 2.8 CONSTRUCT — Real-Time Trustworthiness Scoring
**Paper**: arxiv:2603.18014 | **Phase**: Verification

**Core idea**: Score the trustworthiness of each field in LLM-generated structured output using a Judge LLM. Works on black-box APIs without logprobs or fine-tuning.

**Stage-Aware adaptation**: Entirely System 1 — trustworthiness scoring is a bounded classification task on a single output field.

**Implementation**: `src/nodes/quality_scorer.py`

**Finding**: Applied to MASS-RAG outputs. Fields in `untrustworthy_fields` (score < 0.5) get hedging language hints in the writer prompt and increased scrutiny in the critic prompt.

---

## 3. Key Experimental Findings

### 3.1 Every hybrid condition outperforms its matched cloud-only baseline

Across all 35 conditions, every hybrid configuration outperforms the cloud-only baseline using the same cloud model. This holds for all three cloud backends (Sonnet 4.6, Haiku 4.5, Llama 3.3 70B) and all eight local model configurations.

The quality gain is not attributable to model diversity — it follows from offloading System 1 stages to a local model alone.

### 3.2 2.4B local model achieves best overall quality

exaone3.5:2.4b + Sonnet 4.6 achieves 0.869 — the highest quality across all 35 conditions, including all cloud-only and larger-model hybrid configurations. This exceeds cloud-only Sonnet (0.798) by 7.1 points.

### 3.3 Hybrid+Haiku Pareto-dominates cloud-only Sonnet

exaone3.5:2.4b + Haiku 4.5 achieves 0.828 at $0.093/query — simultaneously better quality, 12× lower cost, and stronger privacy than cloud-only Sonnet ($1.128/query, 0.798).

### 3.4 All-local is viable for air-gapped deployments

exaone3.5:2.4b all-local achieves 0.802 — above cloud-only Haiku (0.671) and cloud-only Llama 70B (0.688), at zero cloud cost. The 6–7 point gap vs. Hybrid+Sonnet is the cost of complete privacy.

### 3.5 Cloud token reduction: 65–92%

Average cloud input tokens reduced from 136,891 (cloud-only Sonnet) to 10,700–54,731 (hybrid). The most privacy-aggressive condition (qwen3:4b + Llama 70B) achieves 92.2% reduction while scoring 0.799 — above both weaker cloud-only baselines.

---

## 4. Why Local Models Improve System 1 Stages

An unexpected finding: replacing frontier-model System 1 stages with 2–4B local models not only preserves but improves overall quality. Two mechanisms:

**Task-scope alignment**: Frontier models over-elaborate outputs for constrained generation tasks (CRAG classification, trust scoring), producing verbose reasoning where a concise label is required. Smaller models, with narrower output distributions, stay on-task more reliably.

**Synthesis leverage**: When local models generate section drafts, the cloud synthesis stage receives imperfect but diverse raw material, creating genuine revision leverage absent when a frontier draft is already near-final.

---

## 5. Failure Modes

**Capacity-limited cloud backend**: Hybrid quality degrades when the System 2 cloud model is capacity-limited. Hybrid+Llama 70B conditions fall below the Sonnet baseline but remain above weaker cloud-only baselines. Practitioners should prioritize cloud backend quality over local model size.

**Model-family dependency**: The sufficiency threshold for local model size varies across architectures. qwen3:4b (0.801) falls below qwen3:8b (0.855) in the same tier, while exaone3.5:2.4b (0.869) exceeds all larger models. Validate within your target model family.

**qwen3:4b all-local bimodal distribution**: qwen3:4b all-local shows a bimodal score distribution (σ=0.337): 35% of queries score ≤0.3, 50% score ≥0.8. Not recommended for all-local deployment.

---

## 6. Benchmark Methodology

**120 multilingual queries**: 100 English + 10 Korean + 10 Japanese, spanning 5 domains (AI/ML, Science & Tech, Business, Medical, Law & Policy) and 5 query types (Analytical, Definitional, Current-state, Comparative, Factual).

**Triple Judge Jury**: DeepSeek R1 (671B), Claude Opus 4.6, Mistral Large 3 (675B). Median of three judges from distinct training lineages. G-Eval rubric with 5 anchored dimensions (Coverage, Accuracy, Citation Quality, Depth, Coherence).

**Statistical rigor**: N=600 per condition (120 queries × 5 runs). Paired Wilcoxon signed-rank tests with Bonferroni correction. Cohen's d effect sizes.

**Inter-judge reliability**: Condition-level Pearson r = 0.82–0.85 (A↔B, A↔C). Judges agree on system rankings despite different absolute scales.
