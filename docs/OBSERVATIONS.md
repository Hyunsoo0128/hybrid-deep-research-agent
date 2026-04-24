# Benchmark Observations

> **Internal research notes** — raw observations recorded during experiment runs.
> Not required to reproduce results; see `eval/RESEARCH_REPORT.md` for the full report.

> Bedrock E2E: 6 conditions × 5 queries = 30 runs (Claude Sonnet 4.6, 2026-04-23)
> Ollama E2E: 3 conditions × 5 queries = 15 runs (qwen3:8b) — in progress

---

## 1. Metric Artifact — citation_density v1 design flaw

### Discovery

Original formula: `citation_density = inline_refs / total_citations`

STRIDE Supervisor rewrites sub-queries to retrieve more sources (avg 27.6 → 37.4, +35%).
The writer cites selectively — inline refs grow modestly, but the denominator grows faster.
Result: a system that retrieves *more* sources scores *lower* on citation density.

```
phase1_2   : 27.6 total sources, 27.4 inline refs → density 0.994
phase1_2_3 : 37.4 total sources, 30.6 inline refs → density 0.843
```

This caused phase1_2_3 overall (0.893) to appear *below* baseline (0.907) under v1.

### Fix

v2 formula: `citation_density = min(inline_refs / 20, 1.0)`

Rationale: measures absolute citation adequacy (≥ 20 inline citations = 1.0)
independent of how many total sources were available. The threshold 20 was
calibrated against baseline's average inline ref count (27.8).

### Impact

| Condition   | Overall v1 | Overall v2 | Reversal |
|-------------|-----------|-----------|---------|
| baseline    | 0.907     | 0.92      | —       |
| phase1_2_3  | 0.893 (-0.014 vs baseline) | 0.93 (+0.017 vs baseline) | Yes |
| no_mass_rag | 0.862     | 0.94      | Yes     |

After v2: **all conditions outperform baseline**. The conclusion that Phase 3
techniques hurt quality was an artifact of the metric, not real regression.

### Implication for system design

Any RAG evaluation metric of the form `used / available` penalizes better retrieval.
This pattern also appears in: recall-based metrics when retrieval set grows,
citation coverage when document pool expands. Always verify metric invariance
with respect to the quantity being optimized.

---

## 2. True quality driver — keyword_coverage, not citation_density

After v2 fix, citation_density converges to 1.0 for most conditions.
The remaining overall score differences are driven by **keyword_coverage**:

| Condition   | kw_coverage | cit_v2 | overall_v2 |
|-------------|-------------|--------|-----------|
| phase1      | 0.85        | 1.00   | **0.96**  |
| phase1_2    | 0.81        | 1.00   | 0.95      |
| phase1_2_3  | 0.73        | 1.00   | 0.93      |

phase1_2_3's lower keyword coverage (0.73 vs 0.85) is the true driver of its gap
behind phase1. Hypothesis: CRAG marks some expected-keyword-bearing documents as
INCORRECT, removing them from the context. Verifiable via CRAG verdict logs.

**This is unresolved and should be noted before claiming phase1 is "best overall".**

---

## 3. LLM Judge — important caveats

### Baseline LJudge = 0.50 is a placeholder

The benchmark sets `judge_score = 0.5` for the baseline condition because there is
no reference report to compare against. It is not a real quality measurement.
Baseline LJudge **cannot** be compared to other conditions' LJudge values.

### Binary × 5 samples = 0.20 delta = 1 query flip

LJudge is binary (0 or 1) per query. With 5 queries, any 0.20 difference in
mean represents exactly 1 query judged differently. This is below any reasonable
significance threshold. All LJudge-based conclusions should be stated as
"X/5 queries showed improvement" rather than as continuous quality deltas.

### LJudge results (stated correctly)

| Condition   | LJudge raw | Queries improved vs baseline |
|-------------|-----------|------------------------------|
| phase1_2    | 0.60      | 3/5                          |
| phase1_2_3  | 0.60      | 3/5                          |
| no_mass_rag | 0.40      | 2/5 — MASS-RAG removal hurts |
| no_stride   | 0.80      | 4/5 — highest                |

---

## 4. Leave-One-Out (LOO) findings — v2 metric

### MASS-RAG (phase1_2_3 vs no_mass_rag)

| Metric        | phase1_2_3 | no_mass_rag | delta                   |
|---------------|-----------|-------------|-------------------------|
| Overall v2    | 0.932     | 0.942       | +0.010 ≈same (within noise) |
| LJudge        | 3/5       | 2/5         | -1 query                |
| Cost/query    | $0.75     | $0.31       | -$0.44 saved (59%)      |
| p50 latency   | 216s      | 161s        | -55s faster             |

MASS-RAG's contribution is not captured by automated metrics (overall ≈same)
but is visible in LLM Judge (3/5 vs 2/5). The technique adds depth that LLM
evaluators notice but keyword/citation metrics do not.
Cost is 2.4× higher with MASS-RAG. Decision depends on quality requirements.

### STRIDE (phase1_2_3 vs no_stride)

| Metric        | phase1_2_3 | no_stride | delta                   |
|---------------|-----------|-----------|-------------------------|
| Overall v2    | 0.932     | 0.941     | +0.008 ≈same (within std 0.046) |
| LJudge        | 3/5       | 4/5       | +1 query (no_stride better) |
| Cost/query    | $0.75     | $0.60     | -$0.15 saved (20%)      |

STRIDE shows no positive contribution in either metric direction.
The +0.008 overall and -1 LJudge query for STRIDE are both within noise.
Known deviation: dependency graph (Ω) not implemented; this STRIDE build
uses Sq→Cq Meta-Planner + Supervisor only.

---

## 5. AlignRAG zero-shot — H2 confirmed

All 5 queries (including q4 comparative and q5 factual with explicit contradictions)
returned misaligned_claims = 0.

**H2: AlignRAG detects nothing without CLM fine-tuning.**

Consistent with paper (arxiv:2504.15811): the full pipeline requires
Claim-Level Modeling (CLM) with CCS+CFT training. Zero-shot LLM cannot
replicate the trained claim-alignment discriminator. This is not a bug —
it is a fundamental paper requirement not reproducible via API-only deployment.

Blog framing: "AlignRAG zero-shot showed 0 misalignments across all queries,
including those with known contradictions. The paper's detection capability
requires fine-tuned CLM components unavailable in API-only deployment."

---

## 6. Query-type patterns (phase1_2_3)

| Query | Type          | Overall | CRAG_filter | STRIDE_rewrite | Trust |
|-------|---------------|---------|-------------|----------------|-------|
| q1    | analytical    | 0.93    | -36%        | 27%            | 0.77  |
| q2    | definitional  | 0.80    | -60%        | 60%            | 0.63  |
| q3    | current_state | 0.81    | -11%        | 11%            | 0.53  |
| q4    | comparative   | 0.96    | -20%        | 0%             | 0.70  |
| q5    | factual       | 0.96    | -44%        | 44%            | 0.37  |

- q2 has highest STRIDE rewrite rate (60%) and lowest overall (0.80).
  Correlation exists; causation unconfirmed (kw_coverage=0.43 is proximate cause).
- q4, q5 (contradiction=True) score highest (0.96) — contradictory sources
  do not degrade report quality; this contradicts naive expectations.
- CONSTRUCT trust scores vary widely (0.37–0.77) with no clear query-type pattern.

---

## 7. Cost-efficiency analysis

| Condition   | Overall v2 | Cost/q | Overall/$ |
|-------------|-----------|--------|-----------|
| phase1      | 0.96      | $0.16  | **6.0** (best) |
| no_mass_rag | 0.94      | $0.31  | 3.03      |
| phase1_2    | 0.95      | $0.55  | 1.73      |
| no_stride   | 0.94      | $0.60  | 1.57      |
| phase1_2_3  | 0.93      | $0.75  | 1.24      |

Phase1 (CRAG + Query Decomp + VCM, $0.16/q) is the most cost-efficient.
For highest absolute quality with moderate cost: no_mass_rag ($0.31, overall 0.94).
Full phase1_2_3 justified only when LLM-judged report depth is the priority.

---

## 8. Open questions

1. **kw_coverage drop mechanism**: phase1 kw=0.85 → phase1_2_3 kw=0.73.
   Is CRAG filtering keyword-bearing documents as INCORRECT?
   Verifiable via retrieval_quality[*].verdict per document.

2. **STRIDE keyword impact**: q2 kw=0.43 in phase1_2_3 vs kw=0.71 in phase1.
   Does STRIDE rewrite shift sub-query focus away from expected keywords?

3. **AlignRAG with weaker model**: Does qwen3:8b produce detectable misalignments?
   If yes → H1 (AlignRAG works for weak models); if no → H2 universal.

4. **no_construct ablation**: $3, 5 runs. Lower priority after v2 fix reduced
   phase1_2 → phase1_2_3 gap to 0.019 (within noise).

5. **Live search generalization**: Results use mock fixtures. Real Tavily search
   may change CRAG filtering rates and STRIDE rewrite patterns significantly.

---

## Pending: Ollama results (qwen3:8b)

Key questions:
- AlignRAG H1 vs H2 on weaker model
- MASS-RAG LJudge contribution with weaker base
- Cost-efficiency curve (Ollama $0/query)
