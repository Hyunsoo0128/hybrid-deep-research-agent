# Stage-Aware Local-Cloud Inference: Deep Research Agent

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="docs/images/05-query-input.png" width="720" alt="Deep Research — Query Input" />
</p>
<p align="center">
  <img src="docs/images/03-research-report.png" width="720" alt="Deep Research — Research Report" />
</p>

> For a full UI walkthrough with all screenshots, see [docs/UI_GUIDE.md](docs/UI_GUIDE.md).

---

## 📄 Research Paper

**[eval/RESEARCH_REPORT.md](eval/RESEARCH_REPORT.md)** — Start here.

This repository is the implementation behind the paper:

> **Stage-Aware Local-Cloud Inference: Hybrid Pipelines Consistently Outperform Matched Cloud-Only Baselines**

The paper evaluates 35 configurations (120 queries × 5 runs) across cloud-only, hybrid, and all-local tiers using a Triple Judge Jury (DeepSeek R1, Claude Opus 4.6, Mistral Large 3). Every hybrid condition outperforms its matched cloud-only baseline.

---

## Key Findings

From the paper (full data: `eval/results/`, complete results: `eval/RESEARCH_REPORT.md`):

| Mode | Local | Cloud | Quality (Med) | Δ vs Sonnet | Cloud Tokens/q | Cost/q |
|------|-------|-------|---------------|-------------|----------------|--------|
| Cloud-only | — | Sonnet 4.6 | 0.798 | ref | 136.9K | $1.128 |
| Cloud-only | — | Haiku 4.5 | 0.671 | −0.127 | 172.0K | $0.376 |
| Cloud-only | — | Llama 70B | 0.688 | −0.110 | 70.4K | $0.600 |
| **Hybrid** | **exaone3.5:2.4b** | **Sonnet 4.6** | **0.869** | **+0.071** | **45.9K** | **$0.375** |
| Hybrid | gemma3:4b | Sonnet 4.6 | 0.867 | +0.069 | 47.3K | $0.379 |
| Hybrid | exaone3.5:2.4b | Haiku 4.5 | 0.828 | +0.030 | 42.6K | $0.093 |
| Hybrid | gemma3:4b | Haiku 4.5 | 0.825 | +0.027 | 44.3K | $0.095 |
| All-local | exaone3.5:2.4b | — | 0.802 | +0.004 | 0 | $0.000 |
| All-local | gemma3:4b | — | 0.803 | +0.005 | 0 | $0.000 |

**What this means**: A 2.4B local model (exaone3.5:2.4b) handling System 1 stages achieves **+7.1 points** over cloud-only Sonnet while reducing cloud token exposure by **66.5%** and cost by **3×**. The Hybrid+Haiku configuration exceeds cloud-only Sonnet quality at **12× lower cost**.

Evaluation: Triple Judge Jury — DeepSeek R1 (671B), Claude Opus 4.6, Mistral Large 3 (675B). Median of three judges. N=600 per condition (120 queries × 5 runs).

---

## What This Is

A production-grade deep research system implementing Stage-Aware Local-Cloud Inference — a hybrid architecture that routes each pipeline stage to the appropriate compute tier based on reasoning demand.

The core insight: **not all pipeline stages require frontier-model reasoning**.

- **System 1 (local, 2–4B models)**: CRAG classification, document scoring, section drafting, self-critique — bounded-context operations on single documents
- **System 2 (cloud, frontier LLM)**: Cross-document synthesis, coverage-gap detection, plan elaboration — integrative reasoning across multiple sources

**Privacy Boundary** (enforced by construction): Original queries and full document corpora never reach the cloud. The cloud receives only document titles + 150-character excerpts and locally-generated draft text.

Three simultaneous goals:

1. **A working system** — fully self-hostable, same pipeline as commercial deep research products
2. **A measured benchmark** — 35 conditions, 120 multilingual queries, 5 repeated runs, triple-judge evaluation
3. **A readable codebase** — each architectural decision is explained with the paper that motivated it

---

## Applied Research Papers

9 papers implemented in production code, each adapted to the System 1/System 2 routing principle.

| Paper | Technique | Phase | System 1 (Local) | System 2 (Cloud) | Implementation |
|-------|-----------|-------|-----------------|-----------------|----------------|
| [2401.15884](https://arxiv.org/abs/2401.15884) | **CRAG** | Retrieval | 1st-pass classify + score | Re-evaluate AMBIGUOUS (title+excerpt only) | `src/nodes/search_worker.py` |
| [2507.00355](https://arxiv.org/abs/2507.00355) | **Query Decomp + Reranker** | Planning | Sub-query generation + cross-encoder reranking | — | `src/nodes/plan_generator.py`, `reranker.py` |
| [2511.18743](https://arxiv.org/abs/2511.18743) | **RhinoInsight** (VCM + EAM) | Verification | Claim extraction + trust scoring | — | `src/nodes/checklist_node.py`, `evidence_auditor.py` |
| [2604.18509](https://arxiv.org/abs/2604.18509) | **MASS-RAG** | Drafting | Parallel section drafting (3 agents) | Multi-draft synthesis (local drafts only) | `src/nodes/search_worker.py` |
| [2504.14858](https://arxiv.org/abs/2504.14858) | **AlignRAG** | Verification | Self-critique + rewrite | — | `src/nodes/critic.py` |
| [2512.20660](https://arxiv.org/abs/2512.20660) | **DSAP** | Cross-cutting | JSON guard functions with error-context retry | — | `src/utils/llm_json.py` |
| [2604.17405](https://arxiv.org/abs/2604.17405) | **STRIDE** | Planning | Abstract plan skeleton (Sq) | Concrete execution plan (Cq) from skeleton | `src/nodes/plan_generator.py`, `supervisor.py` |
| [2603.18014](https://arxiv.org/abs/2603.18014) | **CONSTRUCT** | Verification | Evidence structuring + consistency | — | `src/nodes/quality_scorer.py` |
| [2407.08223](https://arxiv.org/abs/2407.08223) | **Speculative Reranking** | Retrieval | Cross-encoder scoring | — | `src/nodes/reranker.py` |

Technique adaptation details and deviations from reference papers: [docs/TECHNIQUES.md](docs/TECHNIQUES.md)

---

## Architecture

Stage-Aware Local-Cloud Inference routes each of the four pipeline phases to the appropriate compute tier.

```
Planning Phase
  generate_plan [LOCAL]     → abstract research skeleton (Sq)
  plan_elaboration [CLOUD]  → concrete execution steps (Cq) from skeleton only
  checklist_node [LOCAL]    → RhinoInsight VCM sub-goal tracking
  [INTERRUPT: plan_review]

Retrieval Phase
  search_worker × N [LOCAL] → CRAG 1st-pass classify + document scoring
  crag_recheck [CLOUD]      → re-adjudicate AMBIGUOUS (title + 150-char excerpt only)
  reranker [LOCAL]          → cross-encoder top-k

Drafting Phase
  mass_rag_drafters [LOCAL] → parallel section drafts (Summarizer/Extractor/Reasoner)
  synthesis [CLOUD]         → coherent report from local drafts only (no raw documents)

Verification Phase
  gap_detector [CLOUD]      → coverage-gap detection from local coverage index
  gap_search [LOCAL]        → targeted re-retrieval
  cross_validator [LOCAL]   → cross-source consistency
  evidence_auditor [LOCAL]  → RhinoInsight EAM claim binding
  critique [LOCAL]          → AlignRAG self-critique + rewrite
```

**Privacy Boundary** (Equation 1 from paper): `inputs(cloud) ∩ {original_query, full_corpus} = ∅`

The cloud receives only:
- Document abstractions: titles and 150-character excerpts, never full document bodies
- Local drafts: compact prose produced by the local model, containing no verbatim document excerpts

Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/HYBRID_STRATEGY.md](docs/HYBRID_STRATEGY.md)

---

## Reproducing the Experiments

The paper's experiments use 120 queries × 5 runs across 35 conditions.

### Quick single-condition run

```bash
# Hybrid: exaone3.5:2.4b local + Sonnet 4.6 cloud (best configuration)
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

Results land in `eval/results/`. The paper's full results are in `eval/RESEARCH_REPORT.md`.

### Feature-flag ablation (Bedrock only)

```bash
python -m eval.e2e_benchmark --provider bedrock --conditions baseline phase1 phase1_2_3
python eval/analyze_results.py eval/results/bedrock_full_v1.json
```

---

## Quick Start

### Requirements

- Python 3.11+, Node.js 18+
- [Tavily API key](https://tavily.com) (free tier)
- LLM: AWS Bedrock **or** Anthropic API key **or** Ollama (free, local)
- For hybrid mode: Ollama + GPU (NVIDIA L4 / RTX 4090 class, 24GB VRAM recommended; RTX 3080+ works for 2–4B models)

### Install

```bash
git clone https://github.com/Hyunsoo0128/deep-research-agent
cd deep-research-agent

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

### Configure

```bash
cp .env.example .env
```

**Hybrid (recommended — matches paper's best configuration)**
```env
LLM_PROVIDER=hybrid
HYBRID_CLOUD_PROVIDER=bedrock
HYBRID_LOCAL_PROVIDER=ollama
HYBRID_LOCAL_MODEL=exaone3.5:2.4b
TAVILY_API_KEY=tvly-...
```

```bash
ollama pull exaone3.5:2.4b      # best hybrid local model (paper result)
ollama pull gemma3:4b            # alternative
ollama pull nomic-embed-text
```

**Cloud-only (Bedrock)**
```env
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-west-2
TAVILY_API_KEY=tvly-...
```

**Cloud-only (Claude API)**
```env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

**All-local (zero cloud cost)**
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=exaone3.5:2.4b
TAVILY_API_KEY=tvly-...
```

### Run

```bash
# Backend (http://localhost:8000)
uvicorn src.main:app --reload

# Frontend (http://localhost:3000)
cd frontend && npm run dev
```

---

## Configuration Tiers

Three practical tiers emerge from the paper's results:

| Tier | Config | Quality | Cost/query | Use case |
|------|--------|---------|------------|----------|
| **Hybrid + Sonnet** | exaone3.5:2.4b + Sonnet 4.6 | 0.869 | $0.375 | Highest quality |
| **Hybrid + Haiku** | exaone3.5:2.4b + Haiku 4.5 | 0.828 | $0.093 | Exceeds cloud-only Sonnet at 12× lower cost |
| **All-local** | exaone3.5:2.4b | 0.802 | $0.000 | Air-gapped / zero cloud cost |

The Hybrid+Haiku configuration Pareto-dominates cloud-only Sonnet on quality, cost, and privacy simultaneously.

---

## UI Features

The frontend (`localhost:3000`) provides a full research workflow in the browser.

| Feature | Description |
|---------|-------------|
| **Query input** | Submit a research question; the system generates a structured research plan |
| **Plan review & edit** | Inspect and edit generated sub-queries before execution; set Research Depth (Fast / Normal / Deep) and Report Length (Brief / Standard / Detailed) |
| **Technique toggles** | Enable or disable each of the 9 RAG techniques individually before approving the plan |
| **Live progress** | Real-time stream of sources found, gap detection, cross-validation score, and synthesis status |
| **Report view** | Rendered Markdown report with inline citations; streamed incrementally as it is generated |
| **Follow-up chat** | Ask questions about the completed report; the system routes between memory recall and targeted re-search |
| **Session history** | Previously completed research sessions are listed in the sidebar and can be restored |
| **Local file search** | Index a local directory (Qdrant + fastembed) so the pipeline can retrieve from private documents alongside web results |
| **Privacy Mode** | Enforces the Privacy Boundary: raw local file content never reaches cloud LLMs |
| **LLM settings** | Switch provider (Bedrock / Claude API / Ollama / Hybrid) at runtime without restarting the server |

Full UI walkthrough: [docs/UI_GUIDE.md](docs/UI_GUIDE.md)

---

## Project Structure

```
deep-research-agent/
├── eval/
│   ├── RESEARCH_REPORT.md         # ★ Full experimental report — read this first
│   ├── EXPERIMENT_LOG.md          # 35-condition experiment timeline and raw data
│   ├── standalone_benchmark.py    # Main benchmark runner
│   ├── deploy_ec2.py              # EC2 parallel deployment for multi-model runs
│   ├── run_phase1.py / run_phase2.py / run_phase3.py
│   ├── e2e_benchmark.py           # End-to-end pipeline benchmark
│   └── results/
│       ├── phase1/                # Per-model unit capability results
│       ├── phase2/                # Phase 2 E2E results
│       └── standalone/            # ★ Full 35-condition results
├── src/
│   ├── graph.py                   # Research pipeline (LangGraph)
│   ├── chat_graph.py              # Chat pipeline (LangGraph)
│   ├── state.py                   # ResearchState + feature flags
│   ├── nodes/
│   │   ├── plan_generator.py      # Query Decomp + STRIDE (Planning phase)
│   │   ├── search_worker.py       # CRAG + MASS-RAG (Retrieval + Drafting phases)
│   │   ├── reranker.py            # Speculative Reranking
│   │   ├── supervisor.py          # STRIDE supervisor routing
│   │   ├── checklist_node.py      # RhinoInsight VCM
│   │   ├── evidence_auditor.py    # RhinoInsight EAM
│   │   ├── gap_detector.py        # Coverage-gap detection (System 2)
│   │   ├── cross_validator.py     # Source cross-validation
│   │   ├── writer.py              # Report generation (System 2 synthesis)
│   │   ├── critic.py              # AlignRAG self-critique
│   │   ├── quality_scorer.py      # CONSTRUCT field-level trust scoring
│   │   └── [chat nodes]
│   ├── providers/
│   │   ├── bedrock.py / claude.py / ollama.py
│   │   ├── hybrid.py              # System 1/System 2 routing (HybridProvider)
│   │   └── agentcore.py           # AWS Bedrock AgentCore (optional)
│   ├── tools/
│   │   ├── search.py              # Tavily web search
│   │   └── local_file_search.py   # Qdrant + fastembed
│   └── utils/
│       └── llm_json.py            # DSAP guard functions
├── frontend/                      # Next.js 14 UI
└── docs/
    ├── ARCHITECTURE.md            # System design, stage-routing principle
    ├── HYBRID_STRATEGY.md         # Privacy Boundary + System 1/System 2 routing
    ├── TECHNIQUES.md              # Paper analysis + implementation scope
    ├── BENCHMARK.md               # Benchmark methodology + full results
    ├── PIPELINE.md                # Node-by-node reference
    ├── RESEARCH_FINDINGS.md       # Applied paper techniques and findings
    ├── CLOUD_BENCHMARK_PLAN.md    # EC2 setup for reproducing experiments
    ├── LEARNING_GUIDE.md          # Multi-agent design concepts
    └── LOCAL_LLM_GUIDE.md         # Ollama model selection and setup
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/research/start` | Submit query → generate plan |
| `POST` | `/research/approve` | Approve/edit plan |
| `GET` | `/research/stream/{id}` | SSE research stream |
| `GET` | `/research/{id}` | Fetch completed report |
| `POST` | `/research/{id}/chat` | Follow-up question |
| `POST` | `/files/index` | Index local files |
| `GET/DELETE` | `/files/index` | Index status / clear |
| `GET/POST` | `/settings` | LLM config / switch provider |
| `GET` | `/health` | Server status |

Interactive docs: `http://localhost:8000/docs`

---

## Documentation

| Document | Contents |
|----------|----------|
| [eval/RESEARCH_REPORT.md](eval/RESEARCH_REPORT.md) | **Start here** — full experimental report (35 conditions) |
| [eval/EXPERIMENT_LOG.md](eval/EXPERIMENT_LOG.md) | Experiment timeline, all 35-condition results, raw data |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Stage-routing principle, LangGraph structure, Privacy Boundary |
| [docs/HYBRID_STRATEGY.md](docs/HYBRID_STRATEGY.md) | System 1/System 2 routing design, privacy analysis |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | Benchmark methodology, Triple Judge Jury, full results table |
| [docs/TECHNIQUES.md](docs/TECHNIQUES.md) | Per-paper implementation scope and deviations |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Node-by-node reference |
| [docs/RESEARCH_FINDINGS.md](docs/RESEARCH_FINDINGS.md) | Applied paper techniques and key findings |
| [docs/UI_GUIDE.md](docs/UI_GUIDE.md) | Frontend walkthrough — all panels, controls, and workflow |
| [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) | Multi-agent design theory, paper walkthroughs |
| [docs/TECH_DECISIONS.md](docs/TECH_DECISIONS.md) | Technology choices — Tavily, Qdrant, fastembed, and why |
| [docs/ADDING_LLM_PROVIDER.md](docs/ADDING_LLM_PROVIDER.md) | Add OpenAI, Gemini, Groq, etc. |
| [docs/LOCAL_LLM_GUIDE.md](docs/LOCAL_LLM_GUIDE.md) | Ollama model selection and setup |

---

## License

[MIT License](LICENSE) — Copyright (c) 2026 Hyunsoo Kim

If you use this project in your work or research, attribution is appreciated:

```
Stage-Aware Local-Cloud Inference: Deep Research Agent — Hyunsoo Kim
https://github.com/Hyunsoo0128/deep-research-agent
```

## Author

**Hyunsoo Kim, Ph.D.**
Senior GTM Specialist Solutions Architect — GenAI, Amazon Web Services · Ph.D., The University of Tokyo · Former AI/ML Researcher — Samsung Electronics, POSCO

[Google Scholar](https://scholar.google.com/citations?user=n09wpHYAAAAJ&hl=en) · [LinkedIn](https://www.linkedin.com/in/hyunsoo-kim-ph-d-297b46186) · hyunkai@amazon.com
