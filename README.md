# Deep Research Agent

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📄 Research Report

**[eval/RESEARCH_REPORT.md](eval/RESEARCH_REPORT.md)** — Start here.

9 RAG papers (2024–2026), 8 local LLMs, 3 languages (EN/KO/JA), measured on real research queries against Claude Sonnet 4.6 as the upper bound. The report covers every implementation decision, what worked, what didn't, and why.

This repository is the code behind that report — every technique the report describes is implemented and runnable here.

---

## Key Findings

From the experiments in the report (full data: `eval/results/`):

| Local Model | C2 Hybrid Score | vs C1 Sonnet (0.940) | Languages |
|-------------|-----------------|----------------------|-----------|
| **gemma3:4b** | **0.895** | −0.045 | EN/KO/JA stable (±0.01) |
| **llama3.1:8b** | **0.880** | −0.060 | EN best |
| **gemma3:12b** | **0.855** | −0.085 | ⚠️ fabricated URLs in reports |
| exaone3.5:7.8b | 0.780 | −0.160 | KO −0.150 despite Korean specialization |
| qwen3:8b | 0.735 | −0.205 | JA = 0.000 (thinking mode bug) |

> C1 = Claude Sonnet 4.6 only (upper bound). C2 = local model handles evaluation/critique roles, Sonnet handles final synthesis.

**What this means**: `gemma3:4b` (a ~4B parameter model) achieves 95% of Sonnet quality on research tasks when used in the right hybrid architecture — at a fraction of the cost, with no raw documents leaving your machine.

---

## What This Is

A production-grade deep research system implementing the same workflow as commercial deep research products (plan generation, parallel search, gap detection, cross-validation, critique, synthesis), built to answer one question:

> **Can a small local LLM + cloud LLM hybrid match cloud-only quality on real research tasks — while keeping sensitive data local?**

Three simultaneous goals:

1. **A working system** — fully self-hostable, same pipeline as commercial deep research products
2. **A measured benchmark** — not "it works", but quantified quality gap vs Claude Sonnet across 8 models and 3 languages
3. **A readable codebase** — each architectural decision is explained with the paper that motivated it

---

## Applied Research Papers

9 papers implemented in production code. Techniques are individually togglable from the UI.

| Paper | Technique | Role in Pipeline | Implementation |
|-------|-----------|-----------------|----------------|
| [2401.15884](https://arxiv.org/abs/2401.15884) | **CRAG** | Post-retrieval relevance filter (CORRECT/AMBIGUOUS/INCORRECT) | `src/nodes/search_worker.py` |
| [2507.00355](https://arxiv.org/abs/2507.00355) | **Query Decomposition + Reranker** | 5-dimensional sub-query generation + cross-encoder reranking | `src/nodes/plan_generator.py`, `reranker.py` |
| [2511.18743](https://arxiv.org/abs/2511.18743) | **RhinoInsight** (VCM + EAM) | Verification checklist + evidence binding | `src/nodes/checklist_node.py`, `evidence_auditor.py` |
| [2604.18509](https://arxiv.org/abs/2604.18509) | **MASS-RAG** | 3-agent parallel analysis (Summarizer/Extractor/Reasoner) | `src/nodes/search_worker.py` |
| [2504.14858](https://arxiv.org/abs/2504.14858) | **AlignRAG** | 3-phase factual misalignment detection | `src/nodes/critic.py` |
| [2512.20660](https://arxiv.org/abs/2512.20660) | **DSAP** | JSON guard functions with error-context retry | `src/utils/llm_json.py` |
| [2604.17405](https://arxiv.org/abs/2604.17405) | **STRIDE** | Meta-Planner (abstract → concrete strategy) + Supervisor routing | `src/nodes/plan_generator.py`, `supervisor.py` |
| [2603.18014](https://arxiv.org/abs/2603.18014) | **CONSTRUCT** | Field-level trustworthiness scoring | `src/nodes/quality_scorer.py` |
| [2407.08223](https://arxiv.org/abs/2407.08223) | **Speculative RAG** | Local drafter → cloud verifier → local refiner (privacy boundary) | `src/nodes/critic.py`, `search_worker.py` |

Deviations from reference papers and exact implementation scope: [docs/TECHNIQUES.md](docs/TECHNIQUES.md)

---

## Reproducing the Experiments

The report's experiments run in two layers.

### Layer 1 — Unit capability (per-technique)

Tests whether a local model can reliably perform each role: JSON parsing (T1), CRAG relevance scoring (T2), AlignRAG critic (T3), MASS-RAG sub-agent (T4).

```bash
# Single model
python eval/critic_pretest.py --model gemma3:4b

# All Phase 1 models in sequence
python eval/run_phase1.py
```

Results land in `eval/results/phase1/`. The report's Table (Section 5-1) comes from `eval/results/standalone/_run_20260423_195123_summary.json`.

### Layer 3 — End-to-end report quality (C1 vs C2)

Runs the full research pipeline on 2 real queries, scores the output report with Claude Sonnet as judge.

```bash
# Local benchmark (no EC2)
python eval/standalone_benchmark.py --mode c2 --model gemma3:4b

# Reproduce the full experiment (requires EC2 + AWS credentials)
# See eval/CLOUD_BENCHMARK_PLAN.md for infrastructure setup
python eval/deploy_ec2.py --models gemma3:4b llama3.1:8b gemma3:12b
```

Confirmed results are in `eval/results/standalone/`. The `/no_think` bug that affected qwen3 results, and the fix, are documented in `eval/EXPERIMENT_LOG.md`.

### Feature-flag ablation (Bedrock only)

```bash
python -m eval.e2e_benchmark --provider bedrock --conditions baseline phase1 phase1_2_3
python eval/analyze_results.py eval/results/bedrock_full_v1.json
```

---

## Architecture

Two LangGraph state machines — research and chat — share a common LLM provider layer.

```
┌─────────────────────────────────────────────────────────┐
│                    Next.js Frontend                      │
│   Query → Plan Approval → Live Progress → Report + Chat  │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP / SSE
┌──────────────────────────▼──────────────────────────────┐
│                  FastAPI + SSE Server                    │
└───────────┬──────────────────────────┬──────────────────┘
            │                          │
   ┌────────▼────────┐       ┌─────────▼────────┐
   │  Research Graph │       │   Chat Graph      │
   │  (LangGraph)    │       │   (LangGraph)     │
   │                 │       │                   │
   │ generate_plan   │       │ router            │
   │ ↓ INTERRUPT ────┼───────┼─ human approval   │
   │ search × N ◀────┼───────┼─ Send API fanout  │
   │ gap_detector    │       │ memory_answer     │
   │ cross_validate  │       │ targeted_search   │
   │ write_draft     │       └───────────────────┘
   │ critique/revise │
   └────────┬────────┘
            │
   ┌────────┴──────────────────┐
   ▼                           ▼
LLM Provider             Search Tools
Bedrock / Claude / Ollama  Tavily + Qdrant (local files)
```

**Hybrid routing**: In `LLM_PROVIDER=hybrid` mode, the local model handles evaluation and critique roles at $0/call: CRAG scoring, AlignRAG suspect extraction, CONSTRUCT trust scoring, and plan generation Stage 1 (extracting a privacy-safe research profile from the original query). The cloud model (Bedrock/Claude) receives only the abstracted profile — never the original query or raw documents — and handles sub-query generation, gap detection, and final synthesis. Raw local files are abstracted by the local LLM (MASS-RAG) before any content reaches the cloud.

Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/HYBRID_STRATEGY.md](docs/HYBRID_STRATEGY.md)

---

## Quick Start

### Requirements

- Python 3.11+, Node.js 18+
- [Tavily API key](https://tavily.com) (free tier)
- LLM: AWS Bedrock **or** Anthropic API key **or** Ollama (free, local)

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

**Bedrock**
```env
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-west-2
TAVILY_API_KEY=tvly-...
```

**Claude API**
```env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

**Ollama (fully local, free)**
```bash
ollama pull gemma3:4b        # recommended from benchmark results
ollama pull nomic-embed-text
```
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma3:4b
TAVILY_API_KEY=tvly-...
```

**Hybrid (local eval + cloud synthesis)**
```env
LLM_PROVIDER=hybrid
HYBRID_CLOUD_PROVIDER=bedrock
HYBRID_LOCAL_PROVIDER=ollama
HYBRID_LOCAL_MODEL=gemma3:4b
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
| **Privacy Mode** | Blocks raw local file content from reaching cloud LLMs — local files are abstracted by a local LLM (MASS-RAG) before synthesis |
| **LLM settings** | Switch provider (Bedrock / Claude API / Ollama / Hybrid) at runtime without restarting the server |

Full UI walkthrough: [docs/UI_GUIDE.md](docs/UI_GUIDE.md)

---

## Project Structure

```
deep-research-agent/
├── eval/
│   ├── RESEARCH_REPORT.md         # ★ Experimental report — read this first
│   ├── EXPERIMENT_LOG.md          # Experiment timeline, bugs fixed, raw data
│   ├── standalone_benchmark.py    # Main benchmark (Layer 1 + Layer 3)
│   ├── deploy_ec2.py              # EC2 parallel deployment for multi-model runs
│   ├── run_phase1.py / run_phase2.py / run_phase3.py
│   ├── critic_pretest.py          # AlignRAG pre-test (Spec RAG Stage 2 validation)
│   └── results/
│       ├── phase1/                # Layer 1 results — 9 models
│       ├── phase2/                # Phase 2 E2E results
│       └── standalone/            # ★ Full experiment results (run_195123 = confirmed data)
├── src/
│   ├── graph.py                   # Research pipeline (LangGraph)
│   ├── chat_graph.py              # Chat pipeline (LangGraph)
│   ├── state.py                   # ResearchState + feature flags
│   ├── nodes/
│   │   ├── plan_generator.py      # Query Decomp + STRIDE
│   │   ├── search_worker.py       # CRAG + MASS-RAG + Speculative RAG
│   │   ├── reranker.py            # Cross-encoder reranking
│   │   ├── supervisor.py          # STRIDE supervisor routing
│   │   ├── checklist_node.py      # RhinoInsight VCM
│   │   ├── evidence_auditor.py    # RhinoInsight EAM
│   │   ├── gap_detector.py        # Knowledge gap detection
│   │   ├── cross_validator.py     # Source cross-validation
│   │   ├── writer.py              # Report generation
│   │   ├── critic.py              # AlignRAG + Speculative RAG critic
│   │   ├── quality_scorer.py      # CONSTRUCT field-level trust scoring
│   │   └── [chat nodes]
│   ├── providers/
│   │   ├── bedrock.py / claude.py / ollama.py
│   │   ├── hybrid.py              # Cloud + local routing
│   │   └── agentcore.py           # AWS Bedrock AgentCore (optional)
│   ├── tools/
│   │   ├── search.py              # Tavily web search
│   │   └── local_file_search.py   # Qdrant + fastembed
│   └── utils/
│       └── llm_json.py            # DSAP guard functions
├── frontend/                      # Next.js 14 UI
└── docs/
    ├── ARCHITECTURE.md            # System design, LangGraph structure
    ├── HYBRID_STRATEGY.md         # Privacy boundary + hybrid routing
    ├── TECHNIQUES.md              # Paper analysis + implementation scope
    ├── BENCHMARK.md               # Benchmark methodology
    ├── PIPELINE.md                # Node-by-node reference
    ├── CLOUD_BENCHMARK_PLAN.md    # EC2 setup for reproducing experiments
    ├── LEARNING_GUIDE.md          # Multi-agent design concepts
    └── LOCAL_LLM_GUIDE.md         # Ollama setup guide
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
| [eval/RESEARCH_REPORT.md](eval/RESEARCH_REPORT.md) | **Start here** — full experimental report |
| [docs/UI_GUIDE.md](docs/UI_GUIDE.md) | Frontend walkthrough — all panels, controls, and workflow |
| [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) | Multi-agent design theory, paper walkthroughs |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | LangGraph structure, SSE design, state machine |
| [docs/TECHNIQUES.md](docs/TECHNIQUES.md) | Per-paper implementation scope and deviations |
| [docs/HYBRID_STRATEGY.md](docs/HYBRID_STRATEGY.md) | Privacy boundary design |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | Benchmark methodology |
| [docs/TECH_DECISIONS.md](docs/TECH_DECISIONS.md) | Technology choices — Tavily, Qdrant, fastembed, and why |
| [docs/ADDING_LLM_PROVIDER.md](docs/ADDING_LLM_PROVIDER.md) | Add OpenAI, Gemini, Groq, etc. |
| [docs/LOCAL_LLM_GUIDE.md](docs/LOCAL_LLM_GUIDE.md) | Ollama model selection and setup |

---

## License

[MIT License](LICENSE) — Copyright (c) 2026 Hyunsoo Kim

If you use this project in your work or research, attribution is appreciated:

```
Deep Research Agent — Hyunsoo Kim, Ph.D. (The University of Tokyo)
https://github.com/Hyunsoo0128/deep-research-agent
```

## Author

**Hyunsoo Kim, Ph.D.**
Senior GTM Specialist Solutions Architect — GenAI, Amazon Web Services · Ph.D., The University of Tokyo · Former AI/ML Researcher — Samsung Electronics, POSCO

[Google Scholar](https://scholar.google.com/citations?user=n09wpHYAAAAJ&hl=en) · [LinkedIn](https://www.linkedin.com/in/hyunsoo-kim-ph-d-297b46186) · hyunkai@amazon.com

