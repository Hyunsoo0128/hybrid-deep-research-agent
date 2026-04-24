"""
Phase E-0: Critic Spec RAG Pre-test — qwen3:8b suspect_claims detection

Tests whether a local LLM (qwen3:8b via Ollama) can reliably detect the 3
AlignRAG misalignment phases before Phase E restructures the Critic node.

Decision gate:
  If qwen3:8b Phase3 recall < 0.5 on these fixtures → Phase E must use a
  cloud verifier for Phase 3 (Evidence-Integrated Synthesis) checks.
  If Phase3 recall >= 0.5 → local detection is viable; cloud verifier
  becomes optional (cost-saving path).

Usage:
  # Test local LLM only:
  python eval/critic_pretest.py --provider local

  # Compare local vs cloud:
  python eval/critic_pretest.py --provider both

  # Verbose (print full critic output per fixture):
  python eval/critic_pretest.py --provider local --verbose

Environment:
  OLLAMA_MODEL           (default: qwen3:8b)
  OLLAMA_BASE_URL        (default: http://localhost:11434)
  LLM_PROVIDER           (default: claude, for cloud baseline)
  CLAUDE_MODEL / BEDROCK_MODEL
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nodes.critic import critique as run_critique
from src.state import ResearchState


# ── Fixtures ─────────────────────────────────────────────────────────────────
# Each fixture has:
#   query, sub_queries, citations (ground truth sources), draft (report with
#   planted misalignments), and expected_flags: list of {phase, keyword}.
#
# Misalignment types planted:
#   phase1 — claim irrelevant to the query
#   phase2 — claim with no citation support (fabricated statistic)
#   phase3 — claim directly contradicts what the citation says

FIXTURES = [
    {
        "id": "f1",
        "description": "Python vs JavaScript for backend: phase3 numeric contradiction",
        "query": "Compare Python and JavaScript for backend web development",
        "sub_queries": [
            {"id": "sq1", "question": "Performance benchmarks of Python vs JavaScript backends"},
            {"id": "sq2", "question": "Ecosystem and library support for Python vs JavaScript"},
        ],
        "citations": [
            {
                "id": "cit_001",
                "title": "Node.js vs Python Backend Performance Benchmark 2024",
                "excerpt": (
                    "In TechEmpower Round 22 benchmarks, Node.js (Express) handled "
                    "approximately 180,000 requests/second while Python (FastAPI) achieved "
                    "approximately 120,000 requests/second on the same hardware."
                ),
            },
            {
                "id": "cit_002",
                "title": "Python Package Index Statistics",
                "excerpt": (
                    "PyPI hosts over 500,000 packages as of 2024, covering domains from "
                    "data science to web frameworks."
                ),
            },
        ],
        "draft": (
            "## Python vs JavaScript for Backend Development\n\n"
            "Both Python and JavaScript (via Node.js) are popular choices for backend services.\n\n"
            # Phase 3 misalignment: flips the benchmark numbers
            "In TechEmpower benchmarks, Python (FastAPI) outperformed Node.js with "
            "180,000 requests/second versus Node.js at 120,000 requests/second [cit_001].\n\n"
            # Phase 1 misalignment: frontend comparison is irrelevant
            "React and Vue.js are the dominant frontend frameworks, making JavaScript "
            "the natural choice for full-stack developers.\n\n"
            "Python's PyPI repository contains over 500,000 packages [cit_002], "
            "providing a rich ecosystem for web development."
        ),
        # What the critic should flag
        "expected_flags": [
            {"phase": "phase3", "keyword": "180,000"},   # numbers are swapped
            {"phase": "phase1", "keyword": "React"},     # frontend irrelevant to backend query
        ],
    },
    {
        "id": "f2",
        "description": "Transformer architecture: phase2 fabricated citation + phase3 contradiction",
        "query": "How does the Transformer architecture work?",
        "sub_queries": [
            {"id": "sq1", "question": "Self-attention mechanism in Transformers"},
            {"id": "sq2", "question": "Computational complexity of Transformer models"},
        ],
        "citations": [
            {
                "id": "cit_001",
                "title": "Attention Is All You Need (Vaswani et al., 2017)",
                "excerpt": (
                    "The Transformer follows an encoder-decoder structure. "
                    "Self-attention allows each position in the encoder to attend "
                    "to all positions in the previous layer of the encoder. "
                    "The self-attention mechanism has O(n²·d) time complexity "
                    "where n is the sequence length and d is the representation dimension."
                ),
            },
        ],
        "draft": (
            "## How the Transformer Architecture Works\n\n"
            "The Transformer was introduced in 'Attention Is All You Need' (2017).\n\n"
            "Self-attention allows each token to attend to every other token in the sequence, "
            "with O(n²·d) time complexity [cit_001].\n\n"
            # Phase 3: contradicts the source (says linear, source says O(n²·d))
            "A key advantage is that the self-attention mechanism has linear O(n) "
            "time complexity, making it highly efficient for long sequences [cit_001].\n\n"
            # Phase 2: fabricated statistic, no citation
            "Transformers now power over 95% of all NLP systems deployed in production."
        ),
        "expected_flags": [
            {"phase": "phase3", "keyword": "linear"},        # contradicts O(n²·d)
            {"phase": "phase2", "keyword": "95%"},           # uncited fabrication
        ],
    },
    {
        "id": "f3",
        "description": "Climate change causes: phase1 off-topic + phase3 magnitude distortion",
        "query": "What are the main human causes of climate change?",
        "sub_queries": [
            {"id": "sq1", "question": "Contribution of fossil fuel combustion to CO2 emissions"},
            {"id": "sq2", "question": "Role of deforestation in climate change"},
        ],
        "citations": [
            {
                "id": "cit_001",
                "title": "IPCC AR6 Summary for Policymakers",
                "excerpt": (
                    "Human influence has warmed the climate at an unprecedented rate. "
                    "Global surface temperature increased by 1.1°C above 1850–1900 levels "
                    "in 2011–2020. Fossil fuel combustion is the dominant cause, responsible "
                    "for approximately 64% of global CO2 emissions."
                ),
            },
            {
                "id": "cit_002",
                "title": "Global Forest Watch 2023 Report",
                "excerpt": (
                    "Deforestation accounts for approximately 10-12% of global greenhouse "
                    "gas emissions annually."
                ),
            },
        ],
        "draft": (
            "## Human Causes of Climate Change\n\n"
            "Fossil fuel combustion is the dominant driver, responsible for approximately "
            "64% of global CO2 emissions [cit_001].\n\n"
            # Phase 3: source says 10-12%, draft says 25%
            "Deforestation is a significant contributor, accounting for approximately "
            "25% of global greenhouse gas emissions [cit_002].\n\n"
            "Global temperatures have risen by 1.1°C since pre-industrial levels [cit_001].\n\n"
            # Phase 1: adaptation strategies are off-topic (query asks about causes)
            "Climate adaptation strategies, such as sea wall construction and drought-resistant "
            "crop development, are increasingly important for vulnerable regions."
        ),
        "expected_flags": [
            {"phase": "phase3", "keyword": "25%"},     # source says 10-12%
            {"phase": "phase1", "keyword": "adaptation"},  # query asks causes, not solutions
        ],
    },
    {
        "id": "f4",
        "description": "Docker vs VMs: phase2 uncited claim + phase3 contradiction",
        "query": "What are the differences between Docker containers and virtual machines?",
        "sub_queries": [
            {"id": "sq1", "question": "Resource overhead: containers vs VMs"},
            {"id": "sq2", "question": "Security isolation: containers vs VMs"},
        ],
        "citations": [
            {
                "id": "cit_001",
                "title": "Docker Documentation: Containers vs VMs",
                "excerpt": (
                    "Containers share the host OS kernel, whereas VMs run a full OS including "
                    "a separate kernel. Containers are more lightweight, typically starting in "
                    "milliseconds, while VMs may take minutes to boot. "
                    "VMs provide stronger hardware-level isolation through hypervisor technology."
                ),
            },
        ],
        "draft": (
            "## Docker Containers vs Virtual Machines\n\n"
            "Containers share the host OS kernel, making them more lightweight than VMs "
            "which run a full operating system [cit_001].\n\n"
            # Phase 3: contradicts source — says containers have stronger isolation
            "Containers provide stronger security isolation than VMs because they run "
            "as independent processes with strict namespacing [cit_001].\n\n"
            # Phase 2: specific benchmark figure with no citation
            "In production deployments, containers reduce infrastructure costs by an "
            "average of 40% compared to equivalent VM setups."
        ),
        "expected_flags": [
            {"phase": "phase3", "keyword": "stronger"},   # source says VMs have stronger isolation
            {"phase": "phase2", "keyword": "40%"},        # uncited cost figure
        ],
    },
    {
        "id": "f5",
        "description": "Quantum computing basics: all 3 phases",
        "query": "What is quantum computing and how does it differ from classical computing?",
        "sub_queries": [
            {"id": "sq1", "question": "Quantum bits (qubits) vs classical bits"},
            {"id": "sq2", "question": "Quantum speedup: which problems benefit most"},
        ],
        "citations": [
            {
                "id": "cit_001",
                "title": "IBM Quantum Computing Overview",
                "excerpt": (
                    "Quantum computers use qubits which can exist in superposition of 0 and 1 "
                    "simultaneously, unlike classical bits which are strictly 0 or 1. "
                    "Current quantum computers (NISQ era) are prone to errors and have limited "
                    "qubit counts, typically 100-1000 qubits."
                ),
            },
            {
                "id": "cit_002",
                "title": "Nature: Quantum Advantage in Cryptography",
                "excerpt": (
                    "Shor's algorithm on a fault-tolerant quantum computer would break "
                    "RSA-2048 encryption. However, practical demonstrations remain limited "
                    "to small problem sizes."
                ),
            },
        ],
        "draft": (
            "## Quantum Computing vs Classical Computing\n\n"
            "Classical bits are strictly 0 or 1, while qubits can exist in superposition [cit_001].\n\n"
            # Phase 3: source says 100-1000 qubits, draft says millions
            "Current quantum processors contain millions of qubits, enabling "
            "unprecedented parallelism [cit_001].\n\n"
            "Shor's algorithm would break RSA-2048 on a fault-tolerant quantum computer [cit_002].\n\n"
            # Phase 2: no citation for this claim
            "Quantum computers are already being used commercially for drug discovery "
            "and financial portfolio optimization at scale.\n\n"
            # Phase 1: blockchain is off-topic for quantum vs classical computing
            "Blockchain technology also uses cryptographic hashing, making it an "
            "interesting application domain."
        ),
        "expected_flags": [
            {"phase": "phase3", "keyword": "millions"},     # source says 100-1000
            {"phase": "phase2", "keyword": "commercially"}, # uncited commercial claim
            {"phase": "phase1", "keyword": "Blockchain"},   # irrelevant to query
        ],
    },
]


# ── Evaluation helpers ────────────────────────────────────────────────────────

@dataclass
class FixtureResult:
    fixture_id: str
    expected: list[dict]
    detected: list[dict]          # raw misaligned_claims from critic
    tp: int = 0                   # true positives
    fn: int = 0                   # false negatives (missed)
    fp: int = 0                   # false positives (spurious)
    precision: float = 0.0
    recall: float = 0.0

    def compute(self) -> None:
        self.tp = sum(1 for exp in self.expected if self._matched(exp))
        self.fn = len(self.expected) - self.tp
        self.fp = max(0, len(self.detected) - self.tp)
        self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        self.recall = self.tp / len(self.expected) if self.expected else 1.0

    def _matched(self, exp: dict) -> bool:
        """
        A detection matches an expected flag when:
          - reported phase matches (or detected phase contains the expected phase), AND
          - the keyword appears somewhere in the detected claim text.
        """
        kw = exp["keyword"].lower()
        exp_phase = exp["phase"]
        for det in self.detected:
            det_phase = det.get("phase", "").lower()
            det_claim = (det.get("claim", "") + " " + det.get("source_quote", "") +
                         " " + det.get("correction_hint", "")).lower()
            if exp_phase in det_phase and kw in det_claim:
                return True
        return False


def _build_state(fixture: dict) -> ResearchState:
    """Build a minimal ResearchState from fixture data."""
    return {
        "original_query": fixture["query"],
        "plan": {
            "intent": "analytical",
            "interpretation": "",
            "sub_queries": fixture["sub_queries"],
            "local_files": [],
            "depth": "normal",
            "estimated_time": "90s",
        },
        "citations": fixture["citations"],
        "draft_report": fixture["draft"],
        "revision_count": 0,
        "mass_rag_outputs": [],
        "feature_flags": {"alignrag": True, "dsap": True},
        # Required state keys with empty defaults
        "plan_approved": True,
        "local_search_enabled": False,
        "retrieval_quality": [],
        "evidence_store": [],
        "critic_feedback": None,
        "final_report": "",
        "research_round": 0,
        "gap_queries": [],
    }


# ── Provider factories ────────────────────────────────────────────────────────

def _build_local_llm():
    from src.providers.ollama import OllamaProvider
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaProvider(model=model, host=host)


def _build_cloud_llm():
    provider = os.getenv("LLM_PROVIDER", "claude").lower()
    if provider in ("bedrock", "aws"):
        from src.providers.bedrock import BedrockProvider
        model = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-haiku-20241022-v1:0")
        return BedrockProvider(model=model)
    else:
        from src.providers.claude import ClaudeProvider
        model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        return ClaudeProvider(model=model)


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_fixtures(
    label: str,
    llm,
    verbose: bool,
) -> list[FixtureResult]:
    results: list[FixtureResult] = []

    print(f"\n{'─' * 65}")
    print(f"Provider: {label}")
    print(f"{'─' * 65}")

    for fx in FIXTURES:
        state = _build_state(fx)
        critic_out = await run_critique(state, llm)
        feedback = critic_out.get("critic_feedback", {})
        detected = feedback.get("misaligned_claims", [])

        # Also include uncited_claims mapped as phase2
        for uc in feedback.get("uncited_claims", []):
            detected.append({"phase": "phase2", "claim": str(uc),
                              "source_citation_ids": [], "source_quote": "", "correction_hint": ""})

        res = FixtureResult(
            fixture_id=fx["id"],
            expected=fx["expected_flags"],
            detected=detected,
        )
        res.compute()
        results.append(res)

        status_icons = []
        for exp in fx["expected_flags"]:
            matched = res._matched(exp)
            status_icons.append(f"{'✓' if matched else '✗'}[{exp['phase']}:{exp['keyword']}]")

        print(f"\n  [{fx['id']}] {fx['description']}")
        print(f"  Expected: {', '.join(status_icons)}")
        print(f"  TP={res.tp} FN={res.fn} FP={res.fp}  "
              f"P={res.precision:.2f} R={res.recall:.2f}")

        if verbose:
            print(f"  Detected claims ({len(detected)}):")
            for d in detected:
                print(f"    [{d.get('phase','')}] {d.get('claim','')[:80]}")

    return results


def _print_aggregate(label: str, results: list[FixtureResult]) -> dict:
    total_tp = sum(r.tp for r in results)
    total_fn = sum(r.fn for r in results)
    total_fp = sum(r.fp for r in results)
    macro_p = sum(r.precision for r in results) / len(results) if results else 0
    macro_r = sum(r.recall for r in results) / len(results) if results else 0

    # Per-phase breakdown
    phase_tp: dict[str, int] = {}
    phase_total: dict[str, int] = {}
    for r in results:
        for exp in r.expected:
            ph = exp["phase"]
            phase_total[ph] = phase_total.get(ph, 0) + 1
            if r._matched(exp):
                phase_tp[ph] = phase_tp.get(ph, 0) + 1

    print(f"\n{'═' * 65}")
    print(f"AGGREGATE: {label}")
    print(f"{'═' * 65}")
    print(f"  Overall   TP={total_tp} FN={total_fn} FP={total_fp}")
    print(f"  Macro     Precision={macro_p:.3f}  Recall={macro_r:.3f}")
    print(f"\n  Per-phase recall:")
    for ph in sorted(phase_total.keys()):
        tp = phase_tp.get(ph, 0)
        tot = phase_total[ph]
        bar = "█" * tp + "░" * (tot - tp)
        print(f"    {ph}:  {tp}/{tot}  {bar}  recall={tp/tot:.2f}")

    # Phase E decision gate
    phase3_recall = phase_tp.get("phase3", 0) / phase_total.get("phase3", 1)
    print(f"\n  Phase E gate (phase3 recall): {phase3_recall:.2f}")
    if phase3_recall >= 0.5:
        print("  → PASS: local phase3 detection viable; cloud verifier optional")
    else:
        print("  → FAIL: local phase3 detection insufficient; cloud verifier REQUIRED in Phase E")

    return {
        "label": label,
        "macro_precision": round(macro_p, 3),
        "macro_recall": round(macro_r, 3),
        "phase3_recall": round(phase3_recall, 3),
        "phase_tp": phase_tp,
        "phase_total": phase_total,
        "total_tp": total_tp,
        "total_fn": total_fn,
        "total_fp": total_fp,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    all_summaries: list[dict] = []

    if args.provider in ("cloud", "both"):
        cloud_llm = _build_cloud_llm()
        label = f"Cloud ({type(cloud_llm).__name__})"
        cloud_results = await run_fixtures(label, cloud_llm, args.verbose)
        summary = _print_aggregate(label, cloud_results)
        all_summaries.append(summary)

    if args.provider in ("local", "both"):
        local_llm = _build_local_llm()
        label = f"Local ({os.getenv('OLLAMA_MODEL', 'qwen3:8b')})"
        local_results = await run_fixtures(label, local_llm, args.verbose)
        summary = _print_aggregate(label, local_results)
        all_summaries.append(summary)

    # Save results
    out_path = ROOT / "eval" / "results" / "critic_pretest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase E-0: Critic Spec RAG pre-test")
    parser.add_argument(
        "--provider", choices=["cloud", "local", "both"], default="local",
        help="Which provider(s) to test (default: local)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full detected claims per fixture",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
