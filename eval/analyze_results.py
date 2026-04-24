#!/usr/bin/env python3
"""
Benchmark result analyzer — generates Tables A-D from e2e_benchmark JSON output.

Usage:
    python3 eval/analyze_results.py eval/results/bedrock_full_v1.json
    python3 eval/analyze_results.py eval/results/bedrock_full_v1.json --compare eval/results/ollama_slim_v1.json
    python3 eval/analyze_results.py eval/results/bedrock_full_v1.json --query-breakdown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ── condition display order ───────────────────────────────────────────────────
CONDITION_ORDER = [
    "baseline",
    "phase1",
    "phase1_2",
    "phase1_2_3",
    "phase1_2_3_no_mass_rag",
    "phase1_2_3_no_stride",
]

CONDITION_LABELS = {
    "baseline":               "baseline",
    "phase1":                 "phase1 (CRAG+decomp+VCM)",
    "phase1_2":               "phase1_2 (+MASS-RAG+AlignRAG)",
    "phase1_2_3":             "phase1_2_3 (+STRIDE+CONSTRUCT)",
    "phase1_2_3_no_mass_rag": "LOO: no_mass_rag",
    "phase1_2_3_no_stride":   "LOO: no_stride",
}

QUERY_ORDER = ["q1", "q2", "q3", "q4", "q5"]
CONTRADICTION_QUERIES = {"q4", "q5"}


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_conditions(data: dict) -> dict[str, dict]:
    """Return {cond_name: condition_dict} in display order."""
    cond_map = {c["name"]: c for c in data["conditions"]}
    ordered = {}
    for key in CONDITION_ORDER:
        if key in cond_map:
            ordered[key] = cond_map[key]
    # append any unexpected conditions at the end
    for key, val in cond_map.items():
        if key not in ordered:
            ordered[key] = val
    return ordered


def agg(cond: dict) -> dict:
    return cond["aggregate"]


def recompute_overall(cond: dict) -> float:
    """Recompute avg_overall_score using v2 citation_density formula."""
    per_q = cond.get("per_query", [])
    if not per_q:
        return 0.0
    totals = []
    for pq in per_q:
        sc = pq.get("scores", {})
        kw = sc.get("keyword_coverage", 0)
        cd = _new_citation_density(pq)
        rl = sc.get("report_length", 0)
        st = sc.get("structure", 0)
        totals.append((kw + cd + rl + st) / 4)
    return sum(totals) / len(totals)


def per_query_map(cond: dict) -> dict[str, dict]:
    return {pq["query_id"]: pq for pq in cond["per_query"]}


def fmt(val: Any, decimals: int = 2, pct: bool = False) -> str:
    if val is None or val == "":
        return "  —  "
    if pct:
        return f"{val * 100:5.1f}%"
    return f"{val:.{decimals}f}"


def delta_str(base: float | None, cmp: float | None, higher_better: bool = True) -> str:
    if base is None or cmp is None:
        return ""
    d = cmp - base
    sign = "+" if d >= 0 else ""
    arrow = ("↑" if d > 0 else "↓") if higher_better else ("↓" if d > 0 else "↑")
    if abs(d) < 0.001:
        return "  ±0  "
    return f"{sign}{d:.3f} {arrow}"


# ── Table A: Quality scores ───────────────────────────────────────────────────

_CITATION_DENSITY_TARGET = 20  # matches eval/e2e_benchmark.py v2 formula

def _new_citation_density(pq: dict) -> float:
    """Recompute citation_density using v2 formula: min(inline_refs / TARGET, 1.0).
    inline_refs is back-calculated from original density * citation_count."""
    sc = pq.get("scores", {})
    old_density = sc.get("citation_density", 0)
    citation_count = pq.get("citation_count", 0)
    inline_refs = round(old_density * citation_count)
    return min(inline_refs / _CITATION_DENSITY_TARGET, 1.0)


def _overall_std(cond: dict) -> float:
    """Std dev of per-query overall scores = (kw + cit_v2 + len + str) / 4."""
    import statistics
    vals = []
    for pq in cond.get("per_query", []):
        sc = pq.get("scores", {})
        kw = sc.get("keyword_coverage", 0)
        cd = _new_citation_density(pq)
        rl = sc.get("report_length", 0)
        st = sc.get("structure", 0)
        vals.append((kw + cd + rl + st) / 4)
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def table_a(conditions: dict[str, dict], baseline_name: str = "baseline") -> None:
    print("\n" + "=" * 100)
    print("  TABLE A — Quality Scores (averaged across all queries)  |  delta = vs baseline")
    print("  [citation_density v2: min(inline_refs/20, 1.0) — decoupled from total sources]")
    print("=" * 100)
    header = f"  {'Condition':<28}  {'Overall':>7}  {'±std':>5}  {'KWcov':>6}  {'CitD_v2':>7}  {'LJudge':>7}  {'Cites':>5}  {'p50s':>5}  {'$/q':>7}  {'Δoverall':>10}"
    print(header)
    print("  " + "-" * 96)

    base_overall = recompute_overall(conditions[baseline_name]) if baseline_name in conditions else None

    for name, cond in conditions.items():
        a = agg(cond)
        label = CONDITION_LABELS.get(name, name)
        overall = recompute_overall(cond)
        std = _overall_std(cond)
        # citation density v2 mean
        pqs = cond.get("per_query", [])
        cit_v2_mean = sum(_new_citation_density(pq) for pq in pqs) / max(len(pqs), 1)

        if name == baseline_name or base_overall is None:
            d_str = "  (baseline)"
        else:
            d = overall - base_overall
            sign = "+" if d >= 0 else ""
            arrow = "↑" if d > 0.001 else ("↓" if d < -0.001 else "—")
            d_str = f"  {d:+.3f} {arrow}"

        print(
            f"  {label:<28}  "
            f"{fmt(overall):>7}  "
            f"{std:5.3f}  "
            f"{fmt(a.get('avg_keyword_coverage')):>6}  "
            f"{fmt(cit_v2_mean):>7}  "
            f"{fmt(a.get('avg_llm_judge')):>7}  "
            f"{a.get('avg_citations_per_report', 0):5.1f}  "
            f"{a['latency_sec']['p50']:5.0f}  "
            f"${a['cost_per_query_usd']['mean']:6.4f}  "
            f"{d_str}"
        )
    print()


# ── Table B: Technique signal metrics ─────────────────────────────────────────

def table_b(conditions: dict[str, dict]) -> None:
    print("=" * 90)
    print("  TABLE B — Technique Signal Metrics")
    print("=" * 90)
    header = f"  {'Condition':<28}  {'CRAG%':>6}  {'VCM%':>5}  {'MRag%':>6}  {'Trust':>6}  {'Revs':>5}  {'Misal':>6}  {'SupRW':>6}  {'Corr%':>6}"
    print(header)
    print("  " + "-" * 86)

    for name, cond in conditions.items():
        a = agg(cond)
        label = CONDITION_LABELS.get(name, name)
        print(
            f"  {label:<28}  "
            f"{fmt(a.get('avg_relevant_doc_ratio'), pct=True):>6}  "
            f"{fmt(a.get('avg_checklist_coverage_ratio'), pct=True):>5}  "
            f"{fmt(a.get('avg_mass_rag_synthesis_ratio'), pct=True):>6}  "
            f"{fmt(a.get('avg_trust_score')):>6}  "
            f"{a.get('avg_revision_count', 0):5.1f}  "
            f"{a.get('avg_misaligned_claims_count', 0):6.1f}  "
            f"{fmt(a.get('avg_supervisor_rewrite_ratio'), pct=True):>6}  "
            f"{fmt(a.get('avg_corroboration_ratio'), pct=True):>6}"
        )
    print()


# ── Table C: Query-type breakdown ─────────────────────────────────────────────

def table_c(conditions: dict[str, dict]) -> None:
    # use phase1_2_3 if available, else last condition
    cond_name = "phase1_2_3" if "phase1_2_3" in conditions else list(conditions)[-1]
    cond = conditions[cond_name]
    pq_map = per_query_map(cond)

    # also grab baseline per-query for delta
    base_pq = per_query_map(conditions["baseline"]) if "baseline" in conditions else {}

    print("=" * 100)
    print(f"  TABLE C — Query-type Breakdown  [{cond_name}]")
    print("=" * 100)
    header = (
        f"  {'QID':<4}  {'Type':<14}  {'Contr':>5}  "
        f"{'Overall':>7}  {'ΔCRAG':>7}  {'Misal':>6}  {'SupRW':>6}  "
        f"{'MRag%':>6}  {'Trust':>6}  {'Corr%':>6}  Query"
    )
    print(header)
    print("  " + "-" * 96)

    for pq in sorted(cond["per_query"], key=lambda x: QUERY_ORDER.index(x["query_id"]) if x["query_id"] in QUERY_ORDER else 99):
        qid = pq["query_id"]
        base = base_pq.get(qid, {})
        contradiction = "⚠ Yes" if qid in CONTRADICTION_QUERIES else "  No"

        # overall = (kw + cit + len + str) / 4  (matches benchmark formula)
        sc = pq.get("scores", {})
        overall = (sc.get("keyword_coverage", 0) + sc.get("citation_density", 0)
                   + sc.get("report_length", 0) + sc.get("structure", 0)) / 4

        base_sc = base.get("scores", {})
        base_overall = (base_sc.get("keyword_coverage", 0) + base_sc.get("citation_density", 0)
                        + base_sc.get("report_length", 0) + base_sc.get("structure", 0)) / 4 if base else None

        crag = pq.get("relevant_doc_ratio", 0)
        base_crag = base.get("relevant_doc_ratio", 1.0)
        crag_delta = crag - base_crag

        misal = pq.get("misaligned_claims_count", 0)
        sup_rw = pq.get("supervisor_rewrite_ratio", 0)
        mrag = pq.get("mass_rag_synthesis_ratio", 0)
        trust = pq.get("avg_trust_score", 0)
        corr = pq.get("corroboration_ratio", 0)

        query_short = pq.get("query", "")[:45]

        print(
            f"  {qid:<4}  {pq.get('query_type', ''):<14}  {contradiction:>5}  "
            f"{fmt(overall):>7}  "
            f"{crag_delta:+6.2f} "
            f"{misal:6.0f}  "
            f"{fmt(sup_rw, pct=True):>6}  "
            f"{fmt(mrag, pct=True):>6}  "
            f"{fmt(trust):>6}  "
            f"{fmt(corr, pct=True):>6}  "
            f"{query_short}..."
        )
    print()


# ── Table D: LOO ablation ─────────────────────────────────────────────────────

def table_d(conditions: dict[str, dict]) -> None:
    if "phase1_2_3" not in conditions:
        print("  [Table D] phase1_2_3 not found — skipping LOO analysis\n")
        return

    full = agg(conditions["phase1_2_3"])
    loo_names = [k for k in conditions if k.startswith("phase1_2_3_no_")]

    if not loo_names:
        print("  [Table D] No LOO conditions found — skipping\n")
        return

    print("=" * 110)
    print("  TABLE D — Leave-One-Out Ablation  (delta = LOO minus phase1_2_3)")
    print("  Interpretation: ↓worse_without = technique helps  |  ↑better_without = technique redundant/harmful")
    print("  [Overall score uses citation_density v2]")
    print("=" * 110)
    col_w = 32
    print(f"  {'Metric':<22}  {'phase1_2_3':>12}  " + "  ".join(f"{n:>{col_w}}" for n in loo_names))
    print("  " + "-" * 100)

    metrics = [
        ("Overall score (v2)", "avg_overall_score",          True,  False),
        ("LLM Judge",         "avg_llm_judge",              True,  False),
        ("p50 latency (s)",   None,                         False, False),
        ("Cost/query ($)",    None,                         False, False),
        ("CRAG relevant%",    "avg_relevant_doc_ratio",     True,  True),
        ("MRag synth%",       "avg_mass_rag_synthesis_ratio", True, True),
        ("CONSTRUCT trust",   "avg_trust_score",            True,  False),
        ("Revisions",         "avg_revision_count",         False, False),
        ("Misalignments",     "avg_misaligned_claims_count", True,  False),
        ("Supervisor RW%",    "avg_supervisor_rewrite_ratio", True, True),
        ("Corroboration%",    "avg_corroboration_ratio",    True,  True),
    ]

    for label, key, higher_better, pct in metrics:
        if key is None:
            if "latency" in label:
                full_val = full["latency_sec"]["p50"]
                loo_vals = [agg(conditions[n])["latency_sec"]["p50"] for n in loo_names]
            else:
                full_val = full["cost_per_query_usd"]["mean"]
                loo_vals = [agg(conditions[n])["cost_per_query_usd"]["mean"] for n in loo_names]
        elif key == "avg_overall_score":
            # Use recomputed v2 overall, not JSON value
            full_val = recompute_overall(conditions["phase1_2_3"])
            loo_vals = [recompute_overall(conditions[n]) for n in loo_names]
        else:
            full_val = full.get(key, 0)
            loo_vals = [agg(conditions[n]).get(key, 0) for n in loo_names]

        full_str = f"{full_val * 100:.1f}%" if pct else f"{full_val:.4f}"
        loo_parts = []
        for lv in loo_vals:
            lv_str = f"{lv * 100:.1f}%" if pct else f"{lv:.4f}"
            d = lv - full_val  # positive = LOO is better than full (technique was harmful/redundant)
            sign = "+" if d >= 0 else ""
            if pct:
                d_str = f"({sign}{d * 100:.1f}%)"
            else:
                d_str = f"({sign}{d:.4f})"
            # Interpret: for higher_better metrics, d>0 means "removing helped" (redundant/harmful)
            if abs(d) < 0.005:
                verdict = "≈same"
            elif higher_better:
                verdict = "↑better_without" if d > 0 else "↓worse_without"
            else:
                verdict = "↑faster_without" if d < 0 else "↓slower_without"
            loo_parts.append(f"{lv_str} {d_str} {verdict}")

        print(f"  {label:<22}  {full_str:>12}  " + "  ".join(f"{p:>{col_w}}" for p in loo_parts))
    print()


# ── Cross-provider comparison ─────────────────────────────────────────────────

def table_e(data1: dict, data2: dict, label1: str, label2: str) -> None:
    print("=" * 90)
    print(f"  TABLE E — Cross-Provider Comparison  [{label1} vs {label2}]")
    print("=" * 90)

    conds1 = get_conditions(data1)
    conds2 = get_conditions(data2)
    shared = [k for k in conds1 if k in conds2]

    header = f"  {'Condition':<28}  {'Overall ':>9}  {'Cost/q ':>9}  {'p50s ':>6}  {'Misal ':>7}  {'CRAG% ':>7}"
    print(header)
    print("  " + "-" * 70)

    for name in shared:
        a1 = agg(conds1[name])
        a2 = agg(conds2[name])
        label = CONDITION_LABELS.get(name, name)
        print(f"  {label:<28}")
        print(
            f"    {label1:<10}  "
            f"{fmt(a1.get('avg_overall_score')):>8}  "
            f"${a1['cost_per_query_usd']['mean']:7.4f}  "
            f"{a1['latency_sec']['p50']:6.0f}  "
            f"{a1.get('avg_misaligned_claims_count', 0):7.1f}  "
            f"{fmt(a1.get('avg_relevant_doc_ratio'), pct=True):>7}"
        )
        print(
            f"    {label2:<10}  "
            f"{fmt(a2.get('avg_overall_score')):>8}  "
            f"${a2['cost_per_query_usd']['mean']:7.4f}  "
            f"{a2['latency_sec']['p50']:6.0f}  "
            f"{a2.get('avg_misaligned_claims_count', 0):7.1f}  "
            f"{fmt(a2.get('avg_relevant_doc_ratio'), pct=True):>7}"
        )
    print()


# ── AlignRAG H1/H2/H3 diagnosis ──────────────────────────────────────────────

def alignrag_diagnosis(conditions: dict[str, dict]) -> None:
    cond_name = "phase1_2_3" if "phase1_2_3" in conditions else None
    if not cond_name:
        return

    cond = conditions[cond_name]
    pq_map = per_query_map(cond)

    print("=" * 70)
    print("  AlignRAG Diagnosis — H1/H2/H3 classification")
    print("=" * 70)

    contradiction_misals = []
    no_contradiction_misals = []

    for qid, pq in pq_map.items():
        misal = pq.get("misaligned_claims_count", 0)
        is_contradiction = qid in CONTRADICTION_QUERIES
        if is_contradiction:
            contradiction_misals.append((qid, misal))
        else:
            no_contradiction_misals.append((qid, misal))

    print(f"  has_contradiction queries ({', '.join(q for q,_ in contradiction_misals)}):")
    for qid, misal in contradiction_misals:
        print(f"    {qid}: misaligned_claims = {misal:.0f}")

    print(f"  no_contradiction queries ({', '.join(q for q,_ in no_contradiction_misals)}):")
    for qid, misal in no_contradiction_misals:
        print(f"    {qid}: misaligned_claims = {misal:.0f}")

    avg_contradiction = sum(m for _, m in contradiction_misals) / max(len(contradiction_misals), 1)
    avg_no_contradiction = sum(m for _, m in no_contradiction_misals) / max(len(no_contradiction_misals), 1)
    fp_count = sum(1 for _, m in no_contradiction_misals if m > 0)

    print()
    if avg_contradiction > 0 and fp_count == 0:
        verdict = "H1 ✅  AlignRAG detects real misalignments (contradiction queries only)"
    elif avg_contradiction == 0 and fp_count == 0:
        verdict = "H2 ⚠️  AlignRAG detects nothing — zero-shot limitation confirmed"
    elif fp_count > 0 and avg_contradiction == 0:
        verdict = "H3 ❌  AlignRAG false positives — flagging clean queries"
    elif avg_contradiction > 0 and fp_count > 0:
        verdict = "Mixed  AlignRAG detects some real + some false positives"
    else:
        verdict = "Inconclusive"

    print(f"  Verdict: {verdict}")
    print(f"  avg misaligned (contradiction): {avg_contradiction:.2f}")
    print(f"  avg misaligned (no contradiction): {avg_no_contradiction:.2f}")
    print(f"  false positive queries: {fp_count}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze e2e benchmark results")
    parser.add_argument("result_file", help="Path to benchmark JSON (e.g. eval/results/bedrock_full_v1.json)")
    parser.add_argument("--compare", metavar="FILE", help="Second JSON for cross-provider comparison (Table E)")
    parser.add_argument("--query-breakdown", action="store_true", help="Show per-query breakdown (Table C)")
    parser.add_argument("--all", action="store_true", help="Show all tables including Table C and AlignRAG diagnosis")
    args = parser.parse_args()

    path = Path(args.result_file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = load(path)
    conditions = get_conditions(data)

    provider = data.get("provider", "?")
    model = data.get("model", "?")
    label = data.get("phase_label", path.stem)
    ts = data.get("timestamp", "")

    print(f"\n{'='*90}")
    print(f"  Benchmark: {label}  |  {provider}/{model}  |  {ts}")
    print(f"  Conditions: {list(conditions.keys())}")
    print(f"  Queries: {data['benchmark_config']['query_count']}  |  Total runs: {data['benchmark_config']['total_runs']}")
    print(f"{'='*90}")

    table_a(conditions)
    table_b(conditions)

    if args.query_breakdown or args.all:
        table_c(conditions)

    table_d(conditions)
    alignrag_diagnosis(conditions)

    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(f"Compare file not found: {compare_path}", file=sys.stderr)
        else:
            data2 = load(compare_path)
            conds2 = get_conditions(data2)
            label1 = f"{provider}/{model.split('.')[-1]}"
            label2 = f"{data2.get('provider', '?')}/{data2.get('model', '?').split('.')[-1]}"
            table_e(data, data2, label1, label2)


if __name__ == "__main__":
    main()
