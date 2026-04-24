# Deep Research Agent — Experiment Log and Results

> Date: 2026-04-24
> Experiment objective: Evaluate how effectively local small LLMs operate alongside multi-agent RAG pipeline techniques (AlignRAG, CRAG, MASS-RAG, DSAP). Measure the quality gap between the upper bound (Bedrock Sonnet-only) and hybrid (Sonnet + local model) configurations.

---

## 1. Experiment Architecture

### Two Conditions

| Condition | Role Assignment | Infrastructure |
|---|---|---|
| **C1** (Bedrock Sonnet-only) | Sonnet handles all roles exclusively. Upper bound. | t3.xlarge |
| **C2** (Hybrid) | Local roles (query decomp, relevance eval, critic, sub-agent) → Ollama. Final synthesis → Sonnet | g6e.xlarge (L40S GPU) |

### Evaluation Layers

- **Layer 1**: Unit capability tests for local models (T1–T4, 4 tasks)
- **Layer 3**: Final report quality evaluation on 2 real E2E research queries (scored by Sonnet)

### Implemented Techniques

| Technique | Paper | Role |
|---|---|---|
| **DSAP** | arxiv:2512.20660 | On JSON parse failure: L1 (error context) and L2 (clean slate) retries |
| **CRAG** | arxiv:2401.15884 | Search result relevance scoring → relevant/partial/irrelevant classification |
| **AlignRAG** | arxiv:2504.14858 | 3-stage hallucination detection (Critic detects misaligned_claims) |
| **MASS-RAG** | arxiv:2604.18509 | 3 sub-agents in parallel (local) + synthesis (Sonnet) |
| **STRIDE** | — | Report revision loop |

### Evaluation Queries (Layer 3)

1. *"What are the main techniques for optimizing LLM inference performance, and how do they compare in terms of speed, memory, and accuracy trade-offs?"*
2. *"How does retrieval-augmented generation (RAG) improve LLM accuracy, and what are the key challenges in implementing it effectively?"*

---

## 2. Model Candidates

| Model | Parameters | Tier | VRAM (est.) | num_parallel |
|---|---|---|---|---|
| qwen3:4b | ~4B | A | ~4GB | 12 |
| gemma3:4b | ~4B | A | ~4GB | 12 |
| phi4-mini:latest | ~3.8B | A | ~4GB | 12 |
| qwen3:8b | ~8B | B | ~8GB | 8 |
| exaone3.5:7.8b | ~7.8B | B | ~8GB | 8 |
| gemma3:9b | ~9B | B | ~8GB | 8 |
| llama3.1:8b | ~8B | B | ~8GB | 8 |
| qwen3:14b | ~14B | C | ~12GB | 5 |
| gemma3:12b | ~12B | C | ~12GB | 6 |

---

## 3. Experiment History (Chronological)

### 3-1. Initial Exploration (2026-04-22)

> File location: `eval/results/` (prior to standalone)

| File | Contents | Key Results |
|---|---|---|
| `phase0_baseline.json` | Bedrock Sonnet single-condition run (no baseline feature flags) | overall=0.958 |
| `phase0_baseline_ollama.json` | qwen3:8b Ollama baseline | overall=0.850 |
| `component_bedrock.json` | Bedrock component-level unit tests | avg_score_on=0.960, pass_rate=1.0 |
| `component_ollama_qwen3_8b.json` | qwen3:8b component-level unit tests | avg_score_on=0.645, pass_rate=0.636 |
| `phase1_smoke.json` | Bedrock smoke test | ts=2026-04-22T18:33:30Z |
| `phase1_crag.json` | CRAG standalone test (Bedrock) | ts=2026-04-22T17:29:58Z |
| `critic_pretest.json` | AlignRAG Critic pre-test | — |
| `critic_h2_retest.json` | AlignRAG Critic retest | — |

### 3-2. Phase 1 — Unit Capability Screening (2026-04-22 ~ 23)

> File location: `eval/results/phase1/`
> Objective: Measure Ollama models' ability to detect research pipeline phases (recall/precision)
> Pass gate: overall_recall ≥ 0.4, phase3_recall ≥ 0.2

| Model | overall_recall | phase3_recall | Pass |
|---|---|---|---|
| exaone3.5:7.8b | 0.63 | 0.80 | ✅ |
| qwen3:14b | 0.63 | 0.40 | ✅ |
| llama3.2:3b | 0.57 | 0.80 | ✅ |
| gemma3:12b | 0.47 | 0.60 | ✅ |
| llama3.1:8b | 0.40 | 0.60 | ✅ |
| qwen3:8b | 0.30 | 0.00 | ❌ |
| phi4-mini:latest | 0.27 | 0.20 | ❌ |
| mistral:7b | 0.10 | 0.20 | ❌ |
| qwen3:4b | 0.00 | 0.00 | ❌ |

Phase 2 candidates: `exaone3.5:7.8b`, `qwen3:14b`, `llama3.2:3b`, `gemma3:12b`, `llama3.1:8b`

### 3-3. Phase 2 — E2E Benchmark (Legacy)

> File location: `eval/results/phase2/`
> Results: All models score=0.000 (all failed, pass gate not met)
> Cause: Presumably incomplete benchmark pipeline or environment issues at the time. No Phase 3 candidates.

### 3-4. Bedrock Full v1 + Ollama Slim v1 (2026-04-23 01:48)

> File location: `eval/results/bedrock_full_v1.json`, `eval/results/ollama_slim_v1.json`

**bedrock_full_v1** — Sonnet standalone, comparison by feature flag combination:

| Condition | overall |
|---|---|
| baseline (no flags) | 0.907 |
| phase1 (CRAG) | 0.945 |
| phase1_2 (CRAG + AlignRAG) | **0.952** |
| phase1_2_3 (+ MASS-RAG) | 0.893 |
| phase1_2_3_no_mass_rag | 0.862 |
| phase1_2_3_no_stride | 0.925 |

→ **CRAG + AlignRAG combination is the best (0.952)**. Adding MASS-RAG caused a slight drop.

**ollama_slim_v1** — qwen3:14b, conditions: baseline/phase1_2/phase1_2_3 → all 0.000
(legacy benchmark pipeline issue)

### 3-5. Standalone Benchmark Initial Runs (2026-04-23 14:xx)

> File location: `eval/results/standalone/run_20260423_142236/`, `_142347/`, `_143216/`, `_144100/`

Development and debugging phase for standalone_benchmark.py. Key bug fix history:

| Bug | Symptom | Fix |
|---|---|---|
| `llm_json` type mismatch | "list indices must be integers" | Added `_coerce_to_dict()` — extracts first dict when list is returned |
| `_e2e_revise` join error | "expected str instance" | Added `str()` conversion |
| defensive score access | NoneType error | Added `_overall()` helper function |
| `--mode` flag missing | Unable to run C1/C2 separately | Added `--mode c1/c2/all` parameter |

run_20260423_143216 ran 7 models but all Layer 3 C2 scores were 0.000 (pipeline bug).

### 3-6. Gemma3:12b Standalone Rerun

> File location: `eval/results/standalone/run_20260423_gemma12b/gemma3_12b/results.json`

Presumably a rerun of gemma3:12b alone on a separate EC2 instance. Result file exists.

---

## 4. Key Experiment Results (run_20260423_195123) — **Current Confirmed Data**

> Run time: 2026-04-23 19:51 UTC
> Infrastructure: EC2 g6e.xlarge (7× g6e.xlarge + 1× t3.xlarge)
> Script: `eval/standalone_benchmark.py` (version at the time, includes `/no_think` bug)
> S3: `s3://<YOUR_S3_BUCKET>/benchmark-results/run_20260423_195123/`

### 4-1. C1 Baseline (Bedrock Sonnet-only)

> File: `run_20260423_195123/c1_baseline/results.json`

- avg_overall: **0.940**
- avg_revision: 0.50
- avg_latency: 257.1s/query

| Query | overall | coverage | accuracy | specificity | structure | revisions |
|---|---|---|---|---|---|---|
| LLM inference optimization techniques | 0.93 | 0.92 | 0.95 | 0.90 | 0.95 | 1 |
| RAG improvement of LLM accuracy | 0.95 | 0.92 | 0.97 | 0.95 | 0.96 | 0 |

Sonnet accurately cited specific figures from sources (INT8 2× memory reduction, PagedAttention throughput improvement, etc.). The RAG query passed in a single pass without revision.

### 4-2. Layer 1 Unit Capability Scores

> File: `layer1` section of each model's `results.json`

| Model | T1 JSON | T2 CRAG | T3 Critic | T4 MASS-RAG | **L1 Weighted** | Notes |
|---|---|---|---|---|---|---|
| exaone3.5:7.8b | 1.00 | 0.95 | 1.00 | 1.00 | **0.989** | Near-perfect on all tasks |
| gemma3:12b | 1.00 | 0.955 | 1.00 | 0.933 | **0.979** | Slight deduction on T4 |
| llama3.1:8b | 1.00 | 0.95 | 1.00 | 0.933 | **0.979** | Same as gemma3:12b |
| gemma3:4b | 0.75 | 1.00 | 0.333 | 0.867 | **0.705** | Weak on T1/T3 |
| qwen3:14b | 1.00 | 0.375 | 1.00 | 1.00 | **0.844** | T2 bug (⚠️ /no_think) |
| phi4-mini:latest | 0.75 | 0.333 | 0.333 | 0.733 | **0.518** | Generally weak |
| qwen3:4b | 0.00 | 0.375 | 0.00 | 0.00 | **0.094** | ⚠️ /no_think bug across the board |
| qwen3:8b | 0.75 | 0.375 | 1.00 | 1.00 | **0.769** | ⚠️ /no_think bug (T2 empty 3/3) |

**T2 CRAG Detailed Metrics (gemma3 family):**

| Model | precision | recall | f1 | noise_reduction |
|---|---|---|---|---|
| gemma3:4b | 1.00 | 1.00 | 1.00 | 1.00 |
| gemma3:12b | 1.00 | 0.83 | 0.91 | 1.00 |

**T3 AlignRAG Critic (detection rate for 3 injected errors):**

- gemma3:12b: 3/3 detected ✅ — clearly detected exaggerations such as 1,000x speedup
- llama3.1:8b: 3/3 detected ✅
- exaone3.5:7.8b: 3/3 detected ✅
- gemma3:4b: 1/3 detected ❌ — weak error detection capability

### 4-3. Layer 3 C2 Hybrid Scores

| Model | C2 overall | C1 gap | revisions | latency | Pass |
|---|---|---|---|---|---|
| **C1 Sonnet** | **0.940** | ref | 0.5 | 257s | ref |
| gemma3:4b | 0.895 | +0.045 | 2.0 | 216s | ✅ |
| llama3.1:8b | 0.880 | +0.060 | 2.0 | 183s | ✅ |
| gemma3:12b | 0.855 | +0.085 | 2.0 | 299s | ✅ |
| exaone3.5:7.8b | 0.780 | +0.160 | 2.0 | 209s | ⚠️ gap exceeded |
| phi4-mini:latest | 0.775 | +0.165 | 2.0 | 170s | ⚠️ gap exceeded |
| qwen3:14b | 0.625 | +0.315 | 0.5 | 419s | ⚠️ /no_think bug |
| qwen3:8b | 0.935 | +0.005 | 1.0 | 399s | ✅ (L1 contaminated) |
| qwen3:4b | 0.960 | -0.020 | 0.0 | 327s | ✅ (fake) |

> **Pass gate**: C2 ≥ 0.60 AND quality gap ≤ 0.10

**⚠️ qwen3:4b, qwen3:8b, qwen3:14b C2 results are unreliable** — `/no_think` bug caused all local model outputs to be empty strings. Sonnet performed all roles as fallback. qwen3:4b C2=0.960 is effectively the same as C1.

### 4-4. Per-Query Final Report Evaluation

**gemma3:4b**

| Query | overall | coverage | accuracy | specificity | structure |
|---|---|---|---|---|---|
| LLM inference optimization | 0.94 | 0.95 | 0.95 | 0.92 | 0.93 |
| RAG accuracy improvement | 0.85 | 0.87 | **0.78** | 0.88 | 0.92 |

Cause of accuracy drop: Source stated "40–60% error reduction" but was incorrectly cited as "20–30%".

**gemma3:12b**

| Query | overall | coverage | accuracy | specificity | structure |
|---|---|---|---|---|---|
| LLM inference optimization | 0.86 | 0.92 | **0.78** | 0.88 | 0.95 |
| RAG accuracy improvement | 0.85 | 0.85 | 0.82 | 0.88 | 0.90 |

Cause of accuracy drop: Hallucinated URLs and paper citations:
- `https://www.microsoft.com/en-us/research/blog/awq-...` (fake URL)
- `https://arxiv.org/abs/2310.04346` (fabricated arxiv link)
- `https://flashattention.com/flashattention-2/` (non-existent domain)
- `Zhao et al. (2023)` (non-existent paper, self-labeled as placeholder)

---

## 5. Discovered Bugs and Fix History

### Bug 1 — `llm_json` type mismatch (fixed)

**Symptom**: C1 run throws "list indices must be integers or slices, not str"
**Cause**: No handling logic when model returns JSON list `[{...}]` instead of dict
**Fix**: Added `_coerce_to_dict()` internal function. Extracts first dict element when a list is returned. Triggers DSAP retry on failure.

```python
def _coerce_to_dict(parsed):
    if isinstance(fallback, dict) and not isinstance(parsed, dict):
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
        return None  # → DSAP retry
    return parsed
```

### Bug 2 — `_e2e_revise` join error (fixed)

**Symptom**: "sequence item 0: expected str instance, int/dict found"
**Cause**: Elements of `feedback.get('uncited_claims', [])` are not always strings
**Fix**: Added `str()` conversion

```python
f"- Uncited claims: {', '.join(str(c) for c in feedback.get('uncited_claims', [])[:3]) or 'none'}"
```

### Bug 3 — defensive score access (fixed)

**Symptom**: `AttributeError` on `.get()` when score is returned as a float instead of a dict
**Fix**: `_overall()` helper function

```python
def _overall(score):
    return score.get("overall", 0.5) if isinstance(score, dict) else 0.5
```

### Bug 4 — `/no_think` placement error (fixed, applied from run_20260423_201943 onwards)

**Symptom**: qwen3 family models enter thinking mode and generate `<think>...</think>` blocks. After `_strip_think` is applied, an empty string is returned.
**Cause**: `/no_think` token was appended to the system message, but the official Qwen3 spec requires it to be in the user message to suppress thinking.

```python
# Before fix (incorrect):
sys_content = f"{system}\n/no_think"
full_msgs = [{"role": "system", "content": sys_content}] + messages

# After fix (correct):
full_msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
for i in range(len(full_msgs) - 1, -1, -1):
    if full_msgs[i]["role"] == "user":
        full_msgs[i] = {**full_msgs[i], "content": full_msgs[i]["content"] + "\n/no_think"}
        break
```

**Affected models**: qwen3:4b (across the board, L1=0.094), qwen3:8b (T2 3/3 empty), qwen3:14b (T2 2/3 empty)

---

## 6. Rerun Results — run_20260423_201943 (after qwen3 /no_think fix)

> Run time: 2026-04-23 20:19 UTC
> Objective: Rerun 3 qwen3 models after fixing the `/no_think` bug
> Fixed script: `s3://<YOUR_S3_BUCKET>/benchmark-scripts/standalone_benchmark.py`
> File location: `eval/results/standalone/run_20260423_201943/`

| Model | L1 | T1 | T2 | T3 | T4 | C2 | Rev | Lat | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| qwen3:4b | 0.094 | 0.00 (empty=16) | 0.375 (empty=3) | 0.00 (empty=6) | 0.00 (empty=18) | 0.955 | 0.0 | 238s | ⚠️ Invalid |
| qwen3:8b | 0.844 | 1.00 (empty=1) | 0.375 (empty=3) | 1.00 (empty=1) | 1.00 | 0.735 | 1.5 | 303s | ⚠️ |
| qwen3:14b | 0.844 | 1.00 (empty=2) | 0.375 (empty=2) | 1.00 | 1.00 | 0.500 | 2.0 | 400s | ❌ |

### Analysis

**qwen3:4b**: `/no_think` fix had no effect — the 4B model ignores the token and cannot suppress thinking.
Critique always falls back to (`passed=True`), preserving the Sonnet draft → C2=0.955 is **still fake**.

**qwen3:8b**: T2 still produces empty outputs (3/3). Other tasks achieved 1.0 thanks to the fix.
C2=0.735 is a genuine hybrid score, but report truncation occurred during the revise step (~3,000 characters).

**qwen3:14b**: C2=0.500. Revise output was truncated at 1,029 characters (approximately 1/4 of Sonnet's output).
"cutting off mid-sentence during the PagedAttention discussion and never covering speculative decoding, quantization..." (Sonnet's scoring evaluation).

### Root Cause — Architectural Design Issue Identified

C2 E2E pipeline Step 4 (revise) is handled by the local model:

```
Step 1: Plan     → cloud (Sonnet)
Step 2: CRAG     → local eval, cloud synthesis
Step 3: Write    → cloud (Sonnet, max_tokens=1500)
Step 4: Revise   → local (Ollama, max_tokens=1500) ← problem
Step 5: Score    → cloud (Sonnet judge)
```

Despite `max_tokens=1500`, the qwen3 family fails to generate long reports (generation is cut short).
Non-qwen models (gemma3, llama3.1, exaone) generate sufficiently long reports during the revise step → higher C2 scores.

---

## 7. Final Leaderboard (Aggregated Across All Experiments)

> C1 Baseline (Bedrock Sonnet-only): **0.940**  rev=0.5  lat=257s

| Model | L1 | C2 | C1 gap | Rev | Lat | Verdict | Data Source |
|---|---|---|---|---|---|---|---|
| **C1 Sonnet** | — | **0.940** | ref | 0.5 | 257s | Reference | run_195123 |
| gemma3:4b | 0.705 | **0.895** | +0.045 | 2.0 | 216s | ✅ | run_195123 |
| llama3.1:8b | 0.879 | **0.880** | +0.060 | 2.0 | 183s | ✅ | run_195123 |
| gemma3:12b | 0.979 | **0.855** | +0.085 | 2.0 | 299s | ✅ (hallucination warning) | run_195123 |
| exaone3.5:7.8b | 0.989 | 0.780 | +0.160 | 2.0 | 209s | ⚠️ gap exceeded | run_195123 |
| phi4-mini:latest | 0.518 | 0.775 | +0.165 | 2.0 | 170s | ⚠️ gap exceeded | run_195123 |
| qwen3:8b | 0.844 | 0.735 | +0.205 | 1.5 | 303s | ⚠️ revise truncation | run_201943 |
| **qwen3:4b** | **0.094** | **0.955\*** | — | 0.0 | 238s | ⚠️ **Invalid** | run_201943 |
| qwen3:14b | 0.844 | 0.500 | +0.440 | 2.0 | 400s | ❌ revise truncation | run_201943 |

\* qwen3:4b C2=0.955 is a fake score with zero local model contribution

**Pass gate**: C2 ≥ 0.60 AND quality gap ≤ 0.10 AND data valid

**Recommended models**: `gemma3:4b`, `llama3.1:8b`, `gemma3:12b`

---

## 8. Complete List of Stored Result Files

```
eval/results/
├── bedrock_full_v1.json          # Sonnet feature flag comparison (6 conditions)
├── component_bedrock.json        # Bedrock component-level scores
├── component_ollama_qwen3_8b.json # qwen3:8b component scores
├── critic_h2_retest.json         # AlignRAG critic retest
├── critic_pretest.json           # AlignRAG critic pre-test
├── e2e_bedrock.json              # Bedrock E2E (15 queries)
├── e2e_ollama_qwen3_8b.json      # qwen3:8b E2E (9 queries)
├── ollama_slim_v1.json           # qwen3:14b slim (legacy, all 0)
├── phase0_baseline.json          # Sonnet baseline overall=0.958
├── phase0_baseline_ollama.json   # qwen3:8b baseline overall=0.850
├── phase1_crag.json              # CRAG standalone (Bedrock)
├── phase1_smoke.json             # Smoke test (Bedrock)
├── smoke_graph.json              # Smoke graph data
│
├── phase1/
│   ├── _phase1_summary.json      # Phase 1 summary (candidates: exaone/qwen3:14b/llama3.2:3b/gemma3:12b/llama3.1:8b)
│   ├── exaone3_5_7_8b.json
│   ├── gemma3_12b.json
│   ├── llama3_1_8b.json
│   ├── llama3_2_3b.json
│   ├── mistral_7b.json
│   ├── phi4-mini_latest.json
│   ├── qwen3_14b.json
│   ├── qwen3_4b.json
│   └── qwen3_8b.json
│
├── phase2/
│   ├── _phase2_summary.json      # All models failed (score=0.000)
│   ├── exaone3_5_7_8b.json
│   ├── gemma3_12b.json
│   ├── llama3_1_8b.json
│   ├── llama3_2_3b.json
│   └── qwen3_14b.json
│
└── standalone/
    ├── _final_leaderboard.json   # Final leaderboard (based on run_195123)
    ├── _run_20260423_195123_summary.json  # 195123 summary
    ├── _run_20260423_194902_summary.json  # Previous run summary
    ├── _run_20260423_144100_summary.json
    ├── _run_20260423_143216_summary.json
    ├── _run_20260423_142347_summary.json
    ├── _run_20260423_142236_summary.json
    │
    ├── run_20260423_142236/      # Initial debugging run
    ├── run_20260423_142347/      # Initial debugging run
    ├── run_20260423_142712/      # qwen3:4b standalone (legacy)
    │   └── qwen3_4b/results.json
    ├── run_20260423_143216/      # 7 models (all C2 0.000, bug)
    │   ├── exaone3_5_7_8b/results.json
    │   ├── gemma3_4b/results.json
    │   ├── llama3_1_8b/results.json
    │   ├── phi4-mini_latest/results.json
    │   ├── qwen3_14b/results.json
    │   └── qwen3_8b/results.json
    ├── run_20260423_144100/      # Rerun attempt
    ├── run_20260423_gemma12b/    # gemma3:12b standalone rerun
    │   └── gemma3_12b/results.json
    └── run_20260423_195123/      # ★ Primary experiment (confirmed data)
        ├── c1_baseline/results.json      # C1: Sonnet-only (0.940)
        ├── exaone3_5_7_8b/results.json   # C2: 0.780
        ├── gemma3_12b/results.json       # C2: 0.855
        ├── gemma3_4b/results.json        # C2: 0.895
        ├── llama3_1_8b/results.json      # C2: 0.880
        ├── phi4-mini_latest/results.json # C2: 0.775
        └── qwen3_14b/results.json        # C2: 0.625 (bug-contaminated)
```

---

## 9. results.json Schema

Each `results.json` has the following structure:

```json
{
  "model": "model-name",
  "mode": "c1 | c2",
  "run_id": "run_YYYYMMDD_HHMMSS",
  "timestamp": "ISO8601",
  "sonnet_model": "us.anthropic.claude-sonnet-4-6",

  "layer1": {
    "weighted_score": 0.979,
    "t1_json": {
      "score_off": 0.75,
      "score_on": 1.0,
      "delta": 0.25,
      "details": [...],
      "raw_log": [{"in": [...messages], "out": "...model output..."}]
    },
    "t2_crag": {
      "score_off": 0.6,
      "score_on": 0.955,
      "details": {"precision": 1.0, "recall": 0.83, "f1": 0.91, "noise_reduction": 1.0, "predictions": {...}},
      "raw_log": [...]
    },
    "t3_critic": { ... },
    "t4_decomp": { ... }
  },

  "layer3": {
    "c1_sonnet_only": {
      "avg_overall_score": 0.940,
      "avg_revision_count": 0.5,
      "avg_latency_sec": 257.1,
      "per_query": [
        {
          "query": "...",
          "revision_count": 1,
          "latency_sec": 280.2,
          "final_report": "# Full report markdown text...",
          "score": {
            "overall": 0.93,
            "coverage": 0.92,
            "accuracy": 0.95,
            "specificity": 0.90,
            "structure": 0.95,
            "reasoning": "Sonnet's scoring rationale text..."
          },
          "sub_query_count": 4
        }
      ]
    },
    "c2_hybrid": { ... (same structure) }
  }
}
```

---

## 10. Key Findings

1. **C1 (Sonnet standalone) upper bound is 0.940** — The RAG query passed in a single pass without revision, demonstrating high quality.

2. **gemma3:4b achieved the best C2 performance (0.895, gap=0.045)** — Despite a lower L1 score, actual report quality was excellent. Simple local roles may in fact be advantageous for collaboration with Sonnet.

3. **gemma3:12b has hallucination issues** — L1=0.979 is the highest, but the report contains fake URLs and fabricated paper citations. A reliability risk for deployment.

4. **exaone3.5:7.8b has the highest L1 (0.989) but C2=0.780** — Even strong local model capability produces a gap at the Sonnet synthesis stage. Root cause requires further analysis.

5. **qwen3 family L1 data is contaminated by the `/no_think` bug** — Awaiting rerun results after fix (run_20260423_201943).

6. **DSAP is effective** — T4 score_off≈0.0 → score_on≈0.9 (consistent across gemma3 family). T1 also improved from score_off=0.75 → score_on=1.0 (gemma3:12b).

7. **Feature flag effect (Sonnet baseline)**: CRAG+AlignRAG combination was most effective, improving 0.907→0.952. Adding MASS-RAG caused a slight drop from 0.952→0.893.

---

## 11. Multilingual Experiment — Korean / Japanese (In Progress)

> Run time: 2026-04-23 21:00 UTC~
> Objective: Measure report quality changes when running the same E2E pipeline with Korean (ko) and Japanese (ja) queries
> Infrastructure: 18 instances total (C1 t3.xlarge ×2, C2 g6e.xlarge ×16)

### Multilingual Implementation

| Layer | Changes |
|---|---|
| **Layer 1 (T1–T4)** | No change — fixed English JSON output tasks, language-agnostic |
| **Layer 3 E2E** | `LANG_MODE` env var → selects query/language instruction |
| **Source documents** | Kept in English (cross-lingual RAG test) |
| **Writer/Revise prompts** | Added `{lang_instruction}` placeholder |

**KO Queries:**
1. `"LLM 추론 성능을 최적화하는 주요 기법은 무엇이며, 속도·메모리·정확도 트레이드오프 측면에서 각 기법을 어떻게 비교할 수 있는가?"`
2. `"검색 증강 생성(RAG)은 LLM의 정확도를 어떻게 향상시키는가, 그리고 효과적으로 구현하는 데 있어 핵심 과제는 무엇인가?"`

**JA Queries:**
1. `"LLM推論パフォーマンスを最適化する主な手法は何か、また速度・メモリ・精度のトレードオフの観点からそれらをどのように比較できるか？"`
2. `"検索拡張生成（RAG）はLLMの精度をどのように向上させるか、また効果的に実装するための主要な課題は何か？"`

### KO Experiment Instances

| run_id | Contents | S3 Path |
|---|---|---|
| run_20260423_210029 | C1 KO baseline (t3.xlarge) | `benchmark-results/run_20260423_210029/` |
| run_20260423_210237 | C2 KO × 8 models (g6e.xlarge) | `benchmark-results/run_20260423_210237/` |

### JA Experiment Instances

| run_id | Contents | S3 Path |
|---|---|---|
| (C1 JA run_id TBD) | C1 JA baseline (t3.xlarge) | Collecting results |
| run_20260423_210447 | C2 JA × 8 models (g6e.xlarge) | `benchmark-results/run_20260423_210447/` |

### C1 Baseline Comparison Across 3 Languages ✅ Complete

| Language | run_id | avg_overall | latency | Notes |
|---|---|---|---|---|
| EN | run_20260423_195123 | **0.940** | 257s | Q1=0.93, Q2=0.95 |
| KO | run_20260423_210029 | **0.910** | 221s | Q1=0.91, Q2=?? |
| JA | run_20260423_212647 | **0.935** | 234s | Q1=0.93, Q2=0.94 |

**Sonnet maintains consistent quality largely regardless of language** (relative to EN 0.940: KO -0.030, JA -0.005).
JA C1 bug: The initial JA C1 instance (i-0df4e8a5474433e3e) was incorrectly run with LANG_MODE="ko" (presumably due to run_id collision when launched simultaneously). Rerun completed normally.

### KO Results — C2 (run_20260423_210237) ✅ Complete (7/8, qwen3:4b instance error)

| Model | T1 on | T3 on | C2 avg | Q1 | Q2 | vs EN | Rev |
|---|---|---|---|---|---|---|---|
| phi4-mini:latest | 0.750 | 1.000 | **0.790** | 0.78 | 0.80 | +0.015 | 2.0 |
| gemma3:4b | 1.000 | 1.000 | **0.890** | 0.89 | 0.89 | -0.005 | 2.0 |
| llama3.1:8b | 1.000 | 0.667 | **0.785** | 0.77 | 0.80 | -0.095 | 2.0 |
| exaone3.5:7.8b | 1.000 | 1.000 | **0.630** | 0.71 | 0.55 | -0.150 | 2.0 |
| qwen3:4b | — | — | (instance error) | — | — | — | — |
| qwen3:8b | 1.000 | 1.000 | **0.275** | 0.40 | 0.15 | -0.460 | 2.0 |
| qwen3:14b | 1.000 | 1.000 | **0.715** | 0.78 | 0.65 | +0.215 | 2.0 |
| gemma3:12b | 1.000 | 1.000 | **0.805** | ~0.80 | ~0.81 | -0.050 | 2.0 |

### JA Results — C2 (run_20260423_210447) ✅ Complete (8/8)

| Model | T1 on | T3 on | C2 avg | Q1 | Q2 | vs EN | Rev |
|---|---|---|---|---|---|---|---|
| phi4-mini:latest | 0.750 | 0.667 | **0.705** | 0.65 | 0.76 | -0.070 | 2.0 |
| gemma3:4b | 1.000 | 1.000 | **0.885** | 0.88 | 0.89 | -0.010 | 2.0 |
| llama3.1:8b | 1.000 | 0.667 | **0.850** | 0.88 | 0.82 | -0.030 | 2.0 |
| exaone3.5:7.8b | 1.000 | 1.000 | **0.675** | 0.65 | 0.70 | -0.105 | 2.0 |
| qwen3:4b | 0.000 | 0.000 | **0.935\*** | — | — | — | 0.0 |
| qwen3:8b | 1.000 | 1.000 | **0.000** | 0.00 | 0.00 | -0.735 | 2.0 |
| qwen3:14b | 1.000 | 1.000 | **0.720** | — | — | +0.220 | 2.0 |
| gemma3:12b | 1.000 | 1.000 | **0.820** | — | — | -0.035 | 2.0 |

\* qwen3:4b JA=0.935 is fake (no local contribution, same pattern as EN)

---

## 12. Multilingual Key Findings

### C1 Upper Bound by Language
| Language | Sonnet C1 |
|---|---|
| EN | 0.940 |
| JA | 0.935 (-0.005) |
| KO | 0.910 (-0.030) |

Sonnet itself maintains high quality regardless of language. The slight KO drop reflects the cross-lingual burden of writing a Korean report based on English RAG results without Korean source documents.

### Per-Model Multilingual Stability (C2 avg, delta from EN)

| Model | EN | KO△ | JA△ | Multilingual Stability |
|---|---|---|---|---|
| **gemma3:4b** | 0.895 | -0.005 | -0.010 | ★★★★★ Excellent |
| **gemma3:12b** | 0.855 | -0.050 | -0.035 | ★★★★ Good |
| **llama3.1:8b** | 0.880 | -0.095 | -0.030 | ★★★ Average (KO weakness) |
| **phi4-mini** | 0.775 | +0.015 | -0.070 | ★★★ Average (JA weakness) |
| **qwen3:14b** | 0.500 | +0.215 | +0.220 | ★★★ (overcomes EN truncation limit) |
| **exaone3.5** | 0.780 | -0.150 | -0.105 | ★★ Weak (drops despite being a Korean-specialized model) |
| **qwen3:8b** | 0.735 | -0.460 | -0.735 | ★ Failure (JA=0.000) |
| **qwen3:4b** | (invalid) | — | (invalid) | — |

### Key Insights

1. **gemma3:4b is the strongest across languages** — 0.88–0.90 across EN/KO/JA. A small model that excels at cross-lingual RAG.

2. **exaone3.5:7.8b paradox** — A Korean-specialized model from LG AI Research, yet it scores -0.150 vs. EN on KO. Strong Korean generation ability does not translate to stable cross-lingual synthesis from English source documents.

3. **qwen3:8b JA=0.000 complete failure** — Inserting a Japanese instruction (`IMPORTANT: Write the entire report in Japanese`) in the revise prompt triggers re-entry into thinking mode → empty output after `_strip_think`. The `/no_think` token fails to suppress thinking when paired with non-English instructions.

4. **qwen3:14b KO/JA reversal improvement** — EN=0.500 reflects revise-stage truncation. KO/JA generate more information-dense reports within the same token budget → significant improvement to 0.715/0.720.

5. **Sonnet C1 language robustness** — JA -0.005, KO -0.030 relative to EN 0.940. Significantly more stable than small local models.

---

## 13. Pending Items

- [x] Collect qwen3 rerun results from run_20260423_201943 — complete
- [x] Update final leaderboard — complete
- [x] Launch multilingual (KO/JA) experiment instances — complete
- [x] Collect KO C2 results — complete (7/8, 1 qwen3:4b instance error)
- [x] Collect full JA C2 results — complete (8/8)
- [x] JA C1 baseline — complete (0.935)
- [x] Write multilingual leaderboard — complete (Section 12)
- [ ] Rerun KO qwen3:4b (no results due to instance error; fake score expected anyway)
- [ ] Analyze root cause of exaone3.5:7.8b C2 gap (L1=0.989 yet C2=0.780)
- [ ] Resolve qwen3 multilingual failure (non-English instructions + /no_think conflict)
- [ ] gemma3:9b not run (instance terminated after 9 hours due to cloud-init error; rerun undecided)
- [ ] Save run_20260423_201943 results to `eval/results/standalone/` locally
