# Hybrid LLM Strategy — Stage-Aware Local-Cloud Inference

This document describes the System 1/System 2 routing design and Privacy Boundary as formalized in the paper:

> **Stage-Aware Local-Cloud Inference: Hybrid Pipelines Consistently Outperform Matched Cloud-Only Baselines**

See also: [ARCHITECTURE.md](ARCHITECTURE.md), [BENCHMARK.md](BENCHMARK.md)

---

## The Privacy-Cost-Quality Trilemma

Deploying frontier LLMs for deep research surfaces a trilemma in regulated industries:

- **Privacy**: Cloud APIs route queries and retrieved documents through third-party servers, in direct conflict with data-residency regulations (GDPR, HIPAA) that govern healthcare, legal, and financial organizations.
- **Cost**: Deep research pipelines are among the most token-intensive NLP workloads; retrieval, iterative drafting, critique, and synthesis together push per-query costs above $1 at current frontier pricing.
- **Quality**: Routing all computation to smaller local models removes the privacy and cost burdens but degrades the analytical depth that makes deep research useful.

**The trilemma is an architectural artifact**, not an inherent constraint. It arises from treating all pipeline stages as equally demanding. In practice, document scoring, retrieval classification, section drafting, and self-critique require only lightweight inference, well within the capability of 2–4B local models. Cross-document synthesis and coverage-gap detection, by contrast, require frontier-scale reasoning.

---

## System 1 / System 2 Routing

The routing principle maps directly to Kahneman's dual-process theory:

| System | Compute Tier | Reasoning Type | Pipeline Stages |
|--------|-------------|----------------|-----------------|
| **System 1** | Local (2–4B model) | Fast, bounded-context | CRAG classification, document scoring, section drafting, self-critique |
| **System 2** | Cloud (frontier LLM) | Deliberate, integrative | Cross-document synthesis, coverage-gap detection, plan elaboration |

**Routing criterion**: A stage is System 1 if it processes a single bounded input (one document, one draft, one passage) with no cross-document lookup. A stage is System 2 if it must simultaneously integrate evidence across multiple independent sources.

---

## Privacy Boundary

The Privacy Boundary is enforced structurally — not probabilistically. Existing privacy mitigations (differential privacy, entity anonymization, memorization mitigation) are additive: they reduce the probability of sensitive data exposure but cannot eliminate it, because raw queries and documents still reach cloud APIs.

The Privacy Boundary we enforce is categorically different: original queries and full document corpora never reach the cloud, making their direct exposure structurally impossible rather than probabilistically unlikely.

**Formal constraint** (Equation 1 from paper):

```
inputs(g_c) ∩ {q, C} = ∅
```

where `g_c` is any cloud model call, `q` is the original query, and `C` is the full document corpus.

### What the cloud receives

| Category | Cloud receives | Cloud does NOT receive |
|----------|---------------|------------------------|
| Documents | Titles + 150-char excerpts | Full document bodies |
| Query | Plan skeleton (Sq) | Original query text |
| Drafts | Local model prose | Verbatim document excerpts |
| Scores | Numeric scores + labels | Source text that produced scores |

### Privacy boundary completeness

The Privacy Boundary prevents document bodies from reaching the cloud, but does not protect against inference attacks on the abstractions and drafts that do reach the cloud. Stronger privacy guarantees (differential privacy, secure computation) remain future work.

For deployments where the research topic itself is confidential, the all-local configuration (`LLM_PROVIDER=ollama`) provides complete privacy at a 6–7 point quality gap vs. Hybrid+Sonnet.

---

## Configuration Tiers

Three practical tiers emerge from the paper's 35-condition evaluation:

### Tier 1: Hybrid + Sonnet 4.6 (highest quality)

```env
LLM_PROVIDER=hybrid
HYBRID_CLOUD_PROVIDER=bedrock
HYBRID_LOCAL_PROVIDER=ollama
HYBRID_LOCAL_MODEL=exaone3.5:2.4b
```

- Quality: 0.869 (best hybrid, +7.1 pts over cloud-only Sonnet)
- Cost: $0.375/query
- Cloud tokens: 45,918/query (66.5% reduction vs cloud-only Sonnet)
- Privacy: original query and documents never leave local environment

### Tier 2: Hybrid + Haiku 4.5 (Pareto-optimal)

```env
LLM_PROVIDER=hybrid
HYBRID_CLOUD_PROVIDER=bedrock
HYBRID_LOCAL_PROVIDER=ollama
HYBRID_LOCAL_MODEL=exaone3.5:2.4b
HYBRID_CLOUD_MODEL=haiku
```

- Quality: 0.828 (exceeds cloud-only Sonnet 0.798)
- Cost: $0.093/query (12× lower than cloud-only Sonnet)
- This configuration Pareto-dominates cloud-only Sonnet on quality, cost, and privacy simultaneously

### Tier 3: All-local (zero cloud cost)

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=exaone3.5:2.4b
```

- Quality: 0.802 (above cloud-only Haiku 0.671 and cloud-only Llama 70B 0.688)
- Cost: $0.000/query
- Use case: air-gapped deployments, maximum privacy

---

## Local Model Selection

Performance does not increase monotonically with local model size. Within the Hybrid+Sonnet tier:

| Local Model | Size | Quality | Cost/q | Latency/q |
|-------------|------|---------|--------|-----------|
| exaone3.5:2.4b | ~2.4B | **0.869** | $0.375 | 233s |
| gemma3:4b | ~4B | 0.867 | $0.379 | 312s |
| qwen3:4b | ~4B | 0.801 | $0.318 | 471s |
| llama3.2:3b | ~3B | 0.781 | $0.301 | 187s |
| qwen3:8b | ~8B | 0.855 | $0.348 | 590s |
| exaone3.5:7.8b | ~7.8B | 0.845 | $0.373 | 382s |
| llama3.1:8b | ~8B | 0.840 | $0.376 | 317s |
| gemma3:12b | ~12B | 0.838 | $0.447 | 525s |

Key finding: exaone3.5:2.4b (0.869) and gemma3:4b (0.867) exceed all 8B and 12B variants, with faster inference. The sufficiency threshold varies across model families — practitioners should validate within their target model family.

**Recommended**: `exaone3.5:2.4b` for best quality/speed, `gemma3:4b` as alternative.

---

## Cloud Model Ablation

Holding the local model fixed (exaone3.5:2.4b):

| Cloud Backend | Quality | Cost/q | vs Cloud-only Sonnet |
|---------------|---------|--------|----------------------|
| Sonnet 4.6 | 0.869 | $0.375 | +0.071 |
| Haiku 4.5 | 0.828 | $0.093 | +0.030 |
| Llama 3.3 70B | 0.793 | $0.128 | −0.005 |

The synthesis model sets a soft quality floor: hybrid gains persist across all cloud backends, but a top-tier cloud model is necessary to maximize them. Practitioners should prioritize cloud backend quality over local model size.

---

## Why Hybrid Outperforms Cloud-Only

The quality gain from hybrid routing is not attributable to model diversity (using two different models). The confound-control design compares each hybrid configuration against the cloud-only baseline using the **same cloud model**:

- Hybrid+Sonnet beats cloud-only Sonnet
- Hybrid+Haiku beats cloud-only Haiku
- Hybrid+Llama 70B beats cloud-only Llama 70B

Since the synthesis and gap-detection model is held fixed within each pair, the improvement follows from offloading System 1 stages to a local model alone.

Two mechanisms explain this:

1. **Task-scope alignment**: Frontier models over-elaborate outputs for constrained generation tasks (CRAG classification, trust scoring), producing verbose reasoning where a concise label is required. Smaller models stay on-task more reliably.

2. **Synthesis leverage**: When local models generate section drafts, the cloud synthesis stage receives imperfect but diverse raw material, creating genuine revision leverage absent when a frontier draft is already near-final.

---

## Privacy Boundary Analysis

Cloud-only pipelines send 70,365–172,017 tokens/query. Hybrid routing reduces this to 10,700–54,731 tokens (65–92% reduction vs. cloud-only Sonnet), with the original query, full documents, and intermediate drafts retained locally.

| Configuration | Cloud Tokens/q | Reduction | Quality |
|---------------|----------------|-----------|---------|
| Cloud-only Sonnet | 136,891 | ref | 0.798 |
| exaone3.5:2.4b + Sonnet | 45,918 | −66.5% | 0.869 |
| gemma3:4b + Sonnet | 47,330 | −65.4% | 0.867 |
| qwen3:4b + Llama 70B | 10,700 | −92.2% | 0.799 |
| gemma3:4b + Llama 70B | 17,171 | −87.5% | 0.780 |

The most privacy-aggressive condition (qwen3:4b + Llama 70B, 92.2% reduction) still scores 0.799 — above cloud-only Haiku (0.671) and cloud-only Llama 70B (0.688).

---

## Implementation: HybridProvider

```python
# src/providers/hybrid.py
class HybridProvider:
    """Routes System 1 stages to local, System 2 stages to cloud."""

    # System 2 nodes: integrative reasoning across multiple sources
    CLOUD_NODES = {"plan_elaboration", "crag_recheck", "synthesis", "gap_detector"}

    def __init__(self, cloud: LLMProvider, local: LLMProvider):
        self._cloud = cloud
        self._local = local

    async def complete(self, messages, system="", node_hint="", **kwargs) -> str:
        provider = self._cloud if node_hint in self.CLOUD_NODES else self._local
        return await provider.complete(messages, system=system, **kwargs)

    async def embed(self, text: str) -> list[float]:
        return await self._local.embed(text)  # embeddings always local
```

Activation via `.env`:

```env
LLM_PROVIDER=hybrid
HYBRID_CLOUD_PROVIDER=bedrock          # or: claude
HYBRID_LOCAL_PROVIDER=ollama
HYBRID_LOCAL_MODEL=exaone3.5:2.4b     # or: gemma3:4b
```

The wrapper implements the same `LLMProvider` protocol as all other providers — it is a drop-in replacement. No node code changes required; nodes receive `node_hint` as a string argument to `complete()`.

---

## Failure Modes

Hybrid quality degrades when the System 2 cloud model is capacity-limited. The Llama 3.3 70B results confirm this: Hybrid+Llama 70B conditions fall below the Sonnet baseline but remain above weaker cloud-only baselines (0.671–0.688).

**Recommendation**: Prioritize cloud backend quality over local model size. A 2.4B local model with Sonnet cloud (0.869) outperforms a 12B local model with Llama 70B cloud (0.717).
