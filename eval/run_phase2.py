"""
Phase 2 — Multi-model component_benchmark runner

Reads Phase 1 passing models from eval/results/phase1/_phase1_summary.json
and runs component_benchmark for each.

Usage:
  OLLAMA_HOST=http://localhost:11435 python eval/run_phase2.py
  OLLAMA_HOST=http://localhost:11435 python eval/run_phase2.py --resume
  OLLAMA_HOST=http://localhost:11435 python eval/run_phase2.py --models exaone3.5:7.8b,llama3.2:3b
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PHASE1_SUMMARY = ROOT / "eval" / "results" / "phase1" / "_phase1_summary.json"
RESULTS_DIR    = ROOT / "eval" / "results" / "phase2"

# Phase 2 pass gate: score_on >= 0.6 for at least 4 out of 6 techniques
PASS_GATE_MIN_PASSING = 4
TOTAL_TECHNIQUES = 6


def _safe_model_name(model: str) -> str:
    return model.replace(":", "_").replace(".", "_")


def _load_phase1_candidates() -> list[str]:
    if not PHASE1_SUMMARY.exists():
        print("Phase 1 summary not found. Run run_phase1.py first.")
        sys.exit(1)
    data = json.loads(PHASE1_SUMMARY.read_text())
    return data.get("phase2_candidates", [])


async def run_one_model(model_name: str) -> dict:
    from eval.component_benchmark import run_all_tests, print_report

    host = os.getenv("OLLAMA_HOST", "http://localhost:11435")

    print(f"\n{'#' * 70}")
    print(f"  PHASE 2 — MODEL: {model_name}")
    print(f"  HOST: {host}")
    print(f"{'#' * 70}")

    from src.providers.ollama import OllamaProvider
    llm = OllamaProvider(model=model_name, host=host)
    label = f"ollama/{model_name}"

    t0 = time.time()
    try:
        results = await run_all_tests(llm, label)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"model": model_name, "error": str(e), "passed": False}
    elapsed = round(time.time() - t0, 1)

    print_report(results, label)

    pass_count = sum(1 for r in results if r.passed)
    avg_score = sum(r.score_on for r in results) / max(len(results), 1)
    passed = pass_count >= PASS_GATE_MIN_PASSING

    verdict = f"✅ PASS → Phase 3" if passed else f"❌ FAIL (pass={pass_count}/{TOTAL_TECHNIQUES})"
    print(f"\n  Phase 2 verdict: {verdict}")

    return {
        "model": model_name,
        "label": label,
        "elapsed_sec": elapsed,
        "pass_count": pass_count,
        "total_techniques": TOTAL_TECHNIQUES,
        "avg_score_on": round(avg_score, 3),
        "passed": passed,
        "techniques": [
            {
                "name": r.technique,
                "score_off": r.score_off,
                "score_on": r.score_on,
                "delta": r.delta,
                "passed": r.passed,
                "error": r.error,
            }
            for r in results
        ],
    }


async def main(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.models:
        candidates = [m.strip() for m in args.models.split(",")]
    else:
        candidates = _load_phase1_candidates()

    print(f"\nPhase 2 candidates: {candidates}")

    all_summaries: list[dict] = []

    for model_name in candidates:
        result_path = RESULTS_DIR / f"{_safe_model_name(model_name)}.json"

        if args.resume and result_path.exists():
            print(f"\n  [SKIP] {model_name} — result already exists")
            summary = json.loads(result_path.read_text())
            all_summaries.append(summary)
            continue

        summary = await run_one_model(model_name)
        all_summaries.append(summary)
        result_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  Saved: {result_path}")

    # ── Final leaderboard ──────────────────────────────────────────────────────
    print(f"\n\n{'═' * 70}")
    print("  PHASE 2 LEADERBOARD")
    print(f"{'═' * 70}")
    print(f"  {'Model':<22} {'Avg Score':<11} {'Pass Count':<12} {'Result'}")
    print(f"  {'─'*22} {'─'*11} {'─'*12} {'─'*15}")

    passed_models = []
    for s in sorted(all_summaries, key=lambda x: x.get("avg_score_on", 0), reverse=True):
        if "error" in s:
            print(f"  {s['model']:<22} ERROR")
            continue
        verdict = "✅ PASS" if s["passed"] else "❌ FAIL"
        print(f"  {s['model']:<22} {s['avg_score_on']:.3f}      "
              f"{s['pass_count']}/{s['total_techniques']}          {verdict}")
        if s["passed"]:
            passed_models.append(s["model"])

    print(f"\n  Pass gate: ≥{PASS_GATE_MIN_PASSING}/{TOTAL_TECHNIQUES} techniques with score≥0.6")
    print(f"\n  → Phase 3 candidates ({len(passed_models)}): {', '.join(passed_models) or 'None'}")

    combined_path = RESULTS_DIR / "_phase2_summary.json"
    combined_path.write_text(json.dumps({
        "models": all_summaries,
        "pass_gate": {"min_passing": PASS_GATE_MIN_PASSING, "total": TOTAL_TECHNIQUES},
        "phase3_candidates": passed_models,
    }, indent=2))
    print(f"\n  Combined results: {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 multi-model component_benchmark runner")
    parser.add_argument("--models", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args))
