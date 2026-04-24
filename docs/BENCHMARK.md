# Performance Benchmarks

Empirical results from two benchmark layers:
- **Layer 1 (Component)**: Each of the 11 research techniques tested in isolation with fixed inputs
- **Layer 2 (E2E)**: Full pipeline tested with multiple technique configurations × 3 fixed queries using mocked search results

Both layers tested on **Bedrock Claude Sonnet 4.6** and **Ollama qwen3:8b** (Apple M1 Pro 16GB).

---

## Layer 3: Hybrid Strategy Validation (2026-04-23)

Targeted benchmarks for the Hybrid LLM Strategy (Phase A–F). See `HYBRID_STRATEGY.md` for full design.

### Phase E-0: Critic Spec RAG pre-test (`eval/critic_pretest.py`)

Tests qwen3:8b's ability to detect 3 AlignRAG misalignment types on 5 fixture queries (11 expected flags).

| Phase | Recall | Notes |
|-------|--------|-------|
| phase1 (off-topic) | 0.33 | weak |
| phase2 (fabricated citation) | 0.33 | weak |
| phase3 (numeric contradiction) | **0.00** | complete failure |
| Overall macro | **0.20** | TP=2, FN=9, FP=7 |

Gate: phase3 recall < 0.5 → cloud verifier required in Phase E (FAIL → proceed to Spec RAG design).

### Phase E: H2 Re-test — AlignRAG Spec RAG critic (`eval/critic_h2_retest.py`)

Tests whether Phase E Spec RAG resolves H2 (zero-shot AlignRAG detecting 0 misalignments).
Fixture: `ALIGNRAG_DRAFT_WITH_ERRORS` (3 injected errors: 1000x amplification, "eliminated decoherence", "ready for commercial deployment").

| Condition | Detected / Gold | Recall | H2 |
|-----------|-----------------|--------|----|
| Baseline (single local qwen3:8b) | 0 / 3 | 0.00 | **FAIL** |
| Phase E Spec RAG (local→cloud→local) | **2 / 3** | **0.67** | **PASS** |

Recall delta: **+0.67**. H2 resolved.
Cloud model: `us.anthropic.claude-haiku-4-5-20251001-v1:0` (Haiku 4.5).
Undetected error: "ready for commercial deployment" — cloud judged as defensible inference (not false alarm).

Results saved: `eval/results/critic_pretest.json`, `eval/results/critic_h2_retest.json`

---

## Benchmark Environment

| Item | Details |
|------|---------|
| **Layer 1 Tool** | `eval/component_benchmark.py` |
| **Layer 2 Tool** | `eval/e2e_benchmark.py` |
| **Measurement Date** | April 2026 |
| **Local LLM** | qwen3:8b (5.2GB, Ollama v0.4+) |
| **Cloud LLM** | Claude Sonnet 4.6 via AWS Bedrock (`us.anthropic.claude-sonnet-4-6`) |
| **Hardware** | Apple M1 Pro 16GB |

---

## Layer 1: Technique Component Benchmark

Each technique is tested independently with fixed embedded fixtures — no live web search. Each test measures an OFF score (technique disabled) vs ON score (technique enabled) and computes Δ.

### Results: Bedrock Claude Sonnet 4.6 — 11/11 PASS

| Technique | Paper | OFF | ON | Δ | Status |
|---|---|---|---|---|---|
| query_decomp | arxiv:2507.00355 | 0.00 | **1.00** | +1.00 | PASS |
| stride | arxiv:2604.17405 | 0.40 | **1.00** | +0.60 | PASS |
| crag | arxiv:2401.15884 | 0.30 | **0.95** | +0.66 | PASS |
| dsap | arxiv:2512.20660 | 0.75 | **1.00** | +0.25 | PASS |
| alignrag | arxiv:2504.14858 | 1.00 | **1.00** | 0.00 | PASS |
| cure | arxiv:2604.12046 | 0.35 | **1.00** | +0.65 | PASS |
| auto_search | arxiv:2604.17337 | 0.50 | **1.00** | +0.50 | PASS |
| sdp | arxiv:2604.17677 | 0.00 | **1.00** | +1.00 | PASS |
| construct | arxiv:2603.18014 | 0.00 | **1.00** | +1.00 | PASS |
| mass_rag | arxiv:2604.18509 | 0.62 | 0.60 | -0.03 | PASS |
| speculative_rag | arxiv:2407.08223 | 1.00 | **1.00** | 0.00 | PASS |
| **AVERAGE** | | **0.39** | **0.96** | **+0.56** | **11/11** |

### Results: Ollama qwen3:8b — 7/11 PASS

| Technique | OFF | ON | Δ | Status |
|---|---|---|---|---|
| query_decomp | 0.00 | **0.80** | +0.80 | PASS |
| stride | 0.40 | 0.40 | 0.00 | **FAIL** |
| crag | 0.30 | 0.38 | +0.07 | **FAIL** |
| dsap | 0.25 | **1.00** | +0.75 | PASS |
| alignrag | 0.33 | **0.67** | +0.33 | PASS |
| cure | 0.35 | **1.00** | +0.65 | PASS |
| auto_search | 0.50 | **0.65** | +0.15 | PASS |
| sdp | 0.00 | 0.00 | 0.00 | **FAIL** |
| construct | 0.00 | **0.90** | +0.90 | PASS |
| mass_rag | 0.25 | 0.30 | +0.05 | **FAIL** |
| speculative_rag | 1.00 | **1.00** | 0.00 | PASS |
| **AVERAGE** | **0.24** | **0.61** | **+0.37** | **7/11** |

### qwen3:8b FAIL Analysis

| Technique | Root Cause |
|---|---|
| **stride** | Cannot generate `dimensions_covered` JSON field consistently — 8B model's structured output limit |
| **crag** | Relevance scoring is not binary (0/1) — blends scores, reducing F1 against gold labels |
| **sdp** | Fails to generate `groups_to_prune` field in required JSON schema |
| **mass_rag** | Does not follow specialist persona instructions — produces generic output regardless |

---

## Layer 2: End-to-End Pipeline Benchmark

The full research pipeline is run with mocked Tavily search results (pre-collected, fixed) across 3 test queries and multiple technique configurations.

### Test Queries

| ID | Query | Type |
|---|---|---|
| Q1 | What are the main causes and effects of global warming? | Analytical |
| Q2 | How does transformer architecture work in large language models? | Technical explanation |
| Q3 | What is the current state of quantum computing and its practical applications? | Current state survey |

### Test Conditions

| Condition | Techniques Active | Tested On |
|---|---|---|
| baseline | None (pure LLM) | Bedrock + Ollama |
| applied | query_decomp, crag, alignrag, dsap, speculative_rag | Bedrock + Ollama |
| new_only | stride, construct, sdp, cure, auto_search, mass_rag | Bedrock only |
| all_on | All 11 techniques | Bedrock + Ollama |
| applied_no_dsap | applied minus dsap | Bedrock only |

### Scoring Metrics

| Metric | Method |
|---|---|
| keyword_coverage | Fraction of expected domain keywords present in report |
| citation_density | [Source N] reference count ÷ total citation count |
| report_length | Word count ÷ 500, capped at 1.0 |
| structure | Presence of headings / lists / References section |
| llm_judge | LLM pairwise comparison vs baseline: 1.0=better, 0.5=tie, 0.0=worse |

### Results: Bedrock Claude Sonnet 4.6

| Condition | KW Cov | Cite Dens | Overall | Judge Δ | Avg Words | Time(s) |
|---|---|---|---|---|---|---|
| baseline | 0.79 | 1.00 | 0.95 | — | 1,484 | 171 |
| applied | 0.85 | 0.97 | 0.95 | -0.50 | 2,717 | 288 |
| new_only | 0.80 | 0.92 | 0.93 | +0.17 | 1,716 | 201 |
| **all_on** | **0.94** | 0.94 | **0.97** | **+0.50** | **3,393** | **365** |
| applied_no_dsap | 0.85 | 1.00 | 0.96 | +0.17 | 1,508 | 133 |

### Results: Ollama qwen3:8b

| Condition | KW Cov | Cite Dens | Overall | Judge Δ | Avg Words | Time(s) |
|---|---|---|---|---|---|---|
| baseline | 0.53 | 0.71 | 0.81 | — | 700 | 431 |
| applied | 0.59 | 0.87 | 0.86 | 0.00 | 800 | 721 |
| **all_on** | **0.64** | 0.54 | 0.80 | 0.00 | **1,300** | **893** |

### Key Findings

**1. all_on is best on Bedrock**
- Keyword coverage 0.94 (+19% vs baseline)
- LLM judge rated all_on better than baseline for all 3 queries
- CONSTRUCT (knowledge graph) is the main driver of report length increase: baseline 1,484 → all_on 3,393 words (+128%)

**2. Techniques have minimal effect on qwen3:8b**
- applied: judge Δ = 0.00 (tied with baseline)
- all_on: citation_density drops 0.71 → 0.54 — JSON/instruction failures cause citation omissions
- Consistent with Layer 1 results: 4 of 6 new techniques FAIL on qwen3:8b

**3. applied shows negative judge score on Bedrock (-0.50)**
- Overall score is identical (0.95) but judge prefers baseline
- Judge prompt truncates reports to 1,500 chars — applied's length advantage is eliminated in the comparison window
- This is a known limitation of the LLM-as-judge approach with truncation

**4. Speed**
- Bedrock: baseline 171s → all_on 365s (2.1× overhead)
- Ollama: baseline 431s → all_on 893s (2.1× overhead, but 2.5× slower than Bedrock overall)

---

## Technique Effect Summary (Combined Layer 1 + Layer 2)

| Technique | L1 Bedrock Δ | L1 Ollama Δ | Works on 8B? | Notes |
|---|---|---|---|---|
| query_decomp | +1.00 | +0.80 | Yes | Strong on both models |
| dsap | +0.25 | +0.75 | Yes | More critical for small models |
| cure | +0.65 | +0.65 | Yes | Model-agnostic gap prioritization |
| construct | +1.00 | +0.90 | Yes | Main driver of report depth increase |
| auto_search | +0.50 | +0.15 | Partial | Works but weaker on 8B |
| alignrag | 0.00 | +0.33 | Yes | Ceiling effect on Bedrock |
| crag | +0.66 | +0.07 | No | Requires precise binary scoring |
| stride | +0.60 | 0.00 | No | Requires complex JSON output |
| sdp | +1.00 | 0.00 | No | JSON schema compliance required |
| mass_rag | -0.03 | +0.05 | Marginal | Persona following is weak overall |
| speculative_rag | 0.00 | 0.00 | Yes | Architecture-level, always active |

---

## Recommended Technique Configurations

| Use Case | Recommended Flags | Reason |
|---|---|---|
| **Bedrock / Claude — best quality** | all_on | All techniques effective; +19% keyword coverage |
| **Bedrock / Claude — faster** | applied (5 techniques) | ~30% faster than all_on with similar overall score |
| **Ollama qwen3:8b** | query_decomp + dsap + cure + construct | 4 techniques that reliably improve 8B models |
| **Ollama — minimal overhead** | query_decomp + dsap | Only verified improvements with low latency cost |

---

## Running the Benchmarks

### Layer 1: Component Benchmark

```bash
# Bedrock
python -m eval.component_benchmark --provider bedrock

# Ollama
python -m eval.component_benchmark --provider ollama --model qwen3:8b

# Results saved to:
# eval/results/component_bedrock.json
# eval/results/component_ollama_qwen3_8b.json
```

### Layer 2: E2E Pipeline Benchmark

```bash
# Bedrock — full 5 conditions
python -m eval.e2e_benchmark --provider bedrock

# Ollama — slim 3 conditions (baseline / applied / all_on)
python -m eval.e2e_benchmark --provider ollama --model qwen3:8b --slim

# Results saved to:
# eval/results/e2e_bedrock.json
# eval/results/e2e_ollama_qwen3_8b.json
```

> **Note**: Layer 2 requires pre-collected search fixtures at `eval/fixtures/mock_search_results.json`. These are included in the repository. To re-collect with a fresh Tavily API call, delete the file and run the fixture collection script in `eval/e2e_benchmark.py`.
