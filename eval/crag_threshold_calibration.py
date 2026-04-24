"""
CRAG Threshold Calibration Script (Phase B-2)

Compares relevance score distributions between a local LLM (qwen3:8b / Ollama)
and a cloud LLM (Claude / Bedrock) on the same 5-query fixture set.

Purpose:
  The CRAG verdict thresholds (CORRECT ≥ 0.5, INCORRECT < 0.3) were calibrated
  on Claude.  Local LLMs may use a different score distribution — this script
  surfaces any systematic shift so operators can tune CRAG_CORRECT_THRESHOLD
  and CRAG_INCORRECT_THRESHOLD env vars accordingly.

Usage:
  # Compare both providers (requires both APIs to be configured):
  python eval/crag_threshold_calibration.py

  # Local only:
  python eval/crag_threshold_calibration.py --provider local

  # Cloud only:
  python eval/crag_threshold_calibration.py --provider cloud

  # Use different model (cloud side):
  python eval/crag_threshold_calibration.py --cloud-model claude-3-5-haiku-20241022

Output:
  - Per-query score tables for both providers
  - Distribution statistics (mean, std, median, p25/p75)
  - Verdict comparison (CORRECT / AMBIGUOUS / INCORRECT)
  - Suggested threshold adjustments when local mean deviates > 0.05 from cloud mean

Environment:
  ANTHROPIC_API_KEY     or  AWS_* for cloud
  OLLAMA_BASE_URL           for local (default: http://localhost:11434)
  OLLAMA_MODEL              for local (default: qwen3:8b)
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from types import SimpleNamespace

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nodes.search_worker import (
    _evaluate_results,
    _compute_query_verdict,
    CRAG_CORRECT_THRESHOLD,
    CRAG_INCORRECT_THRESHOLD,
)


# ── Mock search result adapter ───────────────────────────────────────────────

def _make_result(item: dict) -> SimpleNamespace:
    """Wrap fixture dict as a SimpleNamespace with .title, .summary, .url attrs."""
    return SimpleNamespace(
        title=item["title"],
        summary=item["summary"],
        url=item["url"],
        relevance_score=item.get("relevance_score", 0.5),
    )


# ── Provider factories ───────────────────────────────────────────────────────

def _build_local_llm():
    from src.providers.ollama import OllamaProvider
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaProvider(model=model, host=host)


def _build_cloud_llm(model_override: str | None = None):
    provider = os.getenv("LLM_PROVIDER", "claude").lower()
    if provider in ("bedrock", "aws"):
        from src.providers.bedrock import BedrockProvider
        model = model_override or os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-haiku-20241022-v1:0")
        return BedrockProvider(model=model)
    else:
        from src.providers.claude import ClaudeProvider
        model = model_override or os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        return ClaudeProvider(model=model)


# ── Evaluation runner ────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query_id: str
    question: str
    scores: list[float]
    verdict: str
    provider_label: str


@dataclass
class CalibrationReport:
    results: list[QueryResult] = field(default_factory=list)

    def add(self, qr: QueryResult) -> None:
        self.results.append(qr)

    def stats(self, provider_label: str) -> dict:
        all_scores = [s for r in self.results if r.provider_label == provider_label for s in r.scores]
        if not all_scores:
            return {}
        return {
            "mean":   round(statistics.mean(all_scores), 4),
            "median": round(statistics.median(all_scores), 4),
            "stdev":  round(statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0, 4),
            "p25":    round(sorted(all_scores)[len(all_scores) // 4], 4),
            "p75":    round(sorted(all_scores)[3 * len(all_scores) // 4], 4),
            "n":      len(all_scores),
        }

    def verdict_counts(self, provider_label: str) -> dict[str, int]:
        counts: dict[str, int] = {"CORRECT": 0, "AMBIGUOUS": 0, "INCORRECT": 0}
        for r in self.results:
            if r.provider_label == provider_label:
                counts[r.verdict] = counts.get(r.verdict, 0) + 1
        return counts


async def _run_provider(
    label: str,
    llm,
    fixture_queries: list[dict],
    report: CalibrationReport,
) -> None:
    print(f"\n{'─' * 60}")
    print(f"Provider: {label}")
    print(f"{'─' * 60}")

    for entry in fixture_queries:
        qid = entry["id"]
        question = entry["question"]
        results = [_make_result(r) for r in entry["results"]]

        score_map = await _evaluate_results(question, results, llm, dsap_enabled=True)
        scores = [round(score_map.get(i, 0.5), 4) for i in range(len(results))]
        verdict = _compute_query_verdict(scores)

        report.add(QueryResult(
            query_id=qid,
            question=question,
            scores=scores,
            verdict=verdict,
            provider_label=label,
        ))

        print(f"\n  [{qid}] {question[:70]}")
        for i, (r, s) in enumerate(zip(results, scores)):
            bar = "█" * int(s * 20)
            print(f"    [{i}] {s:.3f} {bar:<20} {r.title[:50]}")
        print(f"    → verdict: {verdict}  (max={max(scores):.3f})")


def _print_summary(report: CalibrationReport, providers: list[str]) -> None:
    print(f"\n{'═' * 60}")
    print("SUMMARY")
    print(f"{'═' * 60}")
    print(f"Thresholds in use:  CORRECT ≥ {CRAG_CORRECT_THRESHOLD}  |  INCORRECT < {CRAG_INCORRECT_THRESHOLD}")

    for label in providers:
        stats = report.stats(label)
        verdicts = report.verdict_counts(label)
        print(f"\n{label}:")
        print(f"  scores  n={stats['n']}  mean={stats['mean']}  median={stats['median']}  "
              f"stdev={stats['stdev']}  p25={stats['p25']}  p75={stats['p75']}")
        print(f"  verdicts  CORRECT={verdicts['CORRECT']}  AMBIGUOUS={verdicts['AMBIGUOUS']}  "
              f"INCORRECT={verdicts['INCORRECT']}")

    # Threshold adjustment suggestions
    if len(providers) == 2:
        local_stats = report.stats(providers[1])   # second = local
        cloud_stats = report.stats(providers[0])   # first  = cloud
        drift = local_stats.get("mean", 0) - cloud_stats.get("mean", 0)
        print(f"\nScore drift (local − cloud mean): {drift:+.4f}")
        if abs(drift) > 0.05:
            adj_correct   = round(CRAG_CORRECT_THRESHOLD   + drift, 2)
            adj_incorrect = round(CRAG_INCORRECT_THRESHOLD + drift, 2)
            print(f"  ⚠  Drift > 0.05 — consider adjusting thresholds:")
            print(f"     CRAG_CORRECT_THRESHOLD={adj_correct}")
            print(f"     CRAG_INCORRECT_THRESHOLD={adj_incorrect}")
            print(f"  Set these as environment variables or in your .env file.")
        else:
            print("  ✓ Drift ≤ 0.05 — current thresholds are appropriate for local LLM.")


# ── Fixture loader ───────────────────────────────────────────────────────────

def _load_fixture_queries(fixture_path: Path) -> list[dict]:
    """
    Load queries from mock_search_results.json and flatten into a list of
    {id, question, results} dicts.  Uses first sub-query per top-level query.
    """
    data = json.loads(fixture_path.read_text())
    queries: list[dict] = []
    for qid, qdata in list(data.items())[:5]:  # cap at 5 queries
        for sqid, sqdata in list(qdata.get("results", {}).items())[:1]:
            sub_query = sqdata.get("sub_query", {})
            search_results = sqdata.get("search_results", [])
            if search_results:
                queries.append({
                    "id": sqid,
                    "question": sub_query.get("question", qdata.get("query", "")),
                    "results": search_results[:7],  # cap per sub-query
                })
    return queries


# ── Main ─────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    fixture_path = ROOT / "eval" / "fixtures" / "mock_search_results.json"
    if not fixture_path.exists():
        print(f"Fixture not found: {fixture_path}", file=sys.stderr)
        sys.exit(1)

    fixture_queries = _load_fixture_queries(fixture_path)
    print(f"Loaded {len(fixture_queries)} sub-queries from fixture.")

    report = CalibrationReport()
    providers_run: list[str] = []

    if args.provider in ("cloud", "both"):
        cloud_llm = _build_cloud_llm(args.cloud_model)
        label = f"Cloud ({type(cloud_llm).__name__})"
        await _run_provider(label, cloud_llm, fixture_queries, report)
        providers_run.append(label)

    if args.provider in ("local", "both"):
        local_llm = _build_local_llm()
        label = f"Local ({os.getenv('OLLAMA_MODEL', 'qwen3:8b')})"
        await _run_provider(label, local_llm, fixture_queries, report)
        providers_run.append(label)

    _print_summary(report, providers_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRAG threshold calibration (Phase B-2)")
    parser.add_argument(
        "--provider", choices=["cloud", "local", "both"], default="both",
        help="Which provider(s) to test (default: both)",
    )
    parser.add_argument(
        "--cloud-model", default=None,
        help="Cloud model override (e.g. claude-haiku-4-5-20251001)",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
