"""
Phase 1 — Multi-model critic_pretest runner

Runs critic_pretest for each candidate model against EC2 Ollama (via SSM port-forward)
and writes per-model results to eval/results/phase1/

Usage:
  # All models (default)
  OLLAMA_BASE_URL=http://localhost:11435 python eval/run_phase1.py

  # Single model
  OLLAMA_BASE_URL=http://localhost:11435 python eval/run_phase1.py --models qwen3:8b

  # Resume (skip already completed)
  OLLAMA_BASE_URL=http://localhost:11435 python eval/run_phase1.py --resume
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

# ── Candidate models with NUM_PARALLEL config for g6e.xlarge (46GB VRAM) ──────
MODELS = [
    {"name": "llama3.2:3b",     "tier": "A", "num_parallel": 16},
    {"name": "qwen3:4b",        "tier": "A", "num_parallel": 12},
    {"name": "phi4-mini:latest","tier": "A", "num_parallel": 12},
    {"name": "qwen3:8b",        "tier": "B", "num_parallel": 8,  "baseline": True},
    {"name": "exaone3.5:7.8b",  "tier": "B", "num_parallel": 8},
    {"name": "llama3.1:8b",     "tier": "B", "num_parallel": 8},
    {"name": "mistral:7b",      "tier": "B", "num_parallel": 8},
    {"name": "qwen3:14b",       "tier": "C", "num_parallel": 5},
    {"name": "gemma3:12b",      "tier": "C", "num_parallel": 6},
]

# Phase 1 pass gate (per CLOUD_BENCHMARK_PLAN.md)
PASS_GATE = {
    "overall_recall": 0.40,
    "phase3_recall":  0.20,
}

RESULTS_DIR = ROOT / "eval" / "results" / "phase1"


def _safe_model_name(model: str) -> str:
    return model.replace(":", "_").replace(".", "_")


async def run_one_model(model_cfg: dict, verbose: bool) -> dict:
    """Run critic_pretest for a single model and return summary dict."""
    from eval.critic_pretest import FIXTURES, run_fixtures, _print_aggregate, _build_state

    model_name = model_cfg["name"]
    label = f"Local ({model_name})"

    # Override env for this run
    os.environ["OLLAMA_MODEL"] = model_name
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")

    from src.providers.ollama import OllamaProvider
    llm = OllamaProvider(model=model_name, host=base_url)

    print(f"\n{'#' * 70}")
    print(f"  MODEL: {model_name}  (Tier {model_cfg['tier']})")
    print(f"  HOST:  {base_url}")
    print(f"{'#' * 70}")

    t0 = time.time()
    try:
        results = await run_fixtures(label, llm, verbose)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "model": model_name,
            "tier": model_cfg["tier"],
            "error": str(e),
            "passed": False,
        }
    elapsed = round(time.time() - t0, 1)

    summary = _print_aggregate(label, results)
    summary["model"] = model_name
    summary["tier"] = model_cfg["tier"]
    summary["elapsed_sec"] = elapsed
    summary["baseline"] = model_cfg.get("baseline", False)

    # Apply pass gate
    overall_recall = summary["macro_recall"]
    phase3_recall = summary["phase3_recall"]
    summary["passed"] = (
        overall_recall >= PASS_GATE["overall_recall"]
        and phase3_recall >= PASS_GATE["phase3_recall"]
    )
    summary["pass_reason"] = (
        f"overall_recall={overall_recall:.2f} (gate≥{PASS_GATE['overall_recall']})  "
        f"phase3_recall={phase3_recall:.2f} (gate≥{PASS_GATE['phase3_recall']})"
    )

    verdict = "✅ PASS → Phase 2" if summary["passed"] else "❌ FAIL → eliminated"
    print(f"\n  Phase 1 verdict: {verdict}")
    print(f"  {summary['pass_reason']}")

    return summary


async def main(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models_to_run = MODELS
    if args.models:
        names = [m.strip() for m in args.models.split(",")]
        models_to_run = [m for m in MODELS if m["name"] in names]
        if not models_to_run:
            print(f"No matching models found for: {names}")
            sys.exit(1)

    all_summaries: list[dict] = []

    for model_cfg in models_to_run:
        result_path = RESULTS_DIR / f"{_safe_model_name(model_cfg['name'])}.json"

        if args.resume and result_path.exists():
            print(f"\n  [SKIP] {model_cfg['name']} — result already exists")
            summary = json.loads(result_path.read_text())
            all_summaries.append(summary)
            continue

        summary = await run_one_model(model_cfg, args.verbose)
        all_summaries.append(summary)

        result_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  Saved: {result_path}")

    # ── Final leaderboard ──────────────────────────────────────────────────────
    print(f"\n\n{'═' * 70}")
    print("  PHASE 1 LEADERBOARD")
    print(f"{'═' * 70}")
    print(f"  {'Model':<22} {'Tier':<5} {'Recall':<8} {'P3 Recall':<11} {'Result'}")
    print(f"  {'─'*22} {'─'*5} {'─'*8} {'─'*11} {'─'*15}")

    passed_models = []
    for s in sorted(all_summaries, key=lambda x: x.get("macro_recall", 0), reverse=True):
        if "error" in s:
            print(f"  {s['model']:<22} {s['tier']:<5} ERROR")
            continue
        verdict = "✅ PASS" if s["passed"] else "❌ FAIL"
        baseline = " ← baseline" if s.get("baseline") else ""
        print(f"  {s['model']:<22} {s['tier']:<5} "
              f"{s['macro_recall']:.3f}   {s['phase3_recall']:.3f}       "
              f"{verdict}{baseline}")
        if s["passed"]:
            passed_models.append(s["model"])

    print(f"\n  Pass gate: overall_recall≥{PASS_GATE['overall_recall']}  "
          f"phase3_recall≥{PASS_GATE['phase3_recall']}")
    print(f"\n  → Phase 2 candidates ({len(passed_models)}): {', '.join(passed_models) or 'None'}")

    # Save combined results
    combined_path = RESULTS_DIR / "_phase1_summary.json"
    combined_path.write_text(json.dumps({
        "models": all_summaries,
        "pass_gate": PASS_GATE,
        "phase2_candidates": passed_models,
    }, indent=2))
    print(f"\n  Combined results: {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 multi-model critic_pretest runner")
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model names to run (default: all)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip models with existing results"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full detected claims per fixture"
    )
    args = parser.parse_args()
    asyncio.run(main(args))
