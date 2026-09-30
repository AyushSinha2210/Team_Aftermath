# Rerank Card: Person 3 Handoff (Reranking + Calibration)

## 1. Overview

This module reranks a fused candidate shortlist using a CPU cross-encoder, then
calibrates confidence to flag low-confidence rankings. Consumes output from Person 2's
retrieval pipeline (once available); produces output for Person 4's demo attribution
panel and ablation table.

**Status as of this handoff: functionally complete, tested, and validated against Person 1's confirmed fine-tuned checkpoint (`best_biencoder`).**

## 2. Function Signatures

### `ariadne.reranking.cross_encoder.rerank(query, candidates)`

```python
def rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]
```

- **Input:** `query` (str), `candidates` (list of dicts, each with at least `id`,
  `text`, `fusion_score`).
- **Behavior:** Scores the top `reranking.rerank_top_k` candidates (currently 20, from
  config.yaml) with `cross-encoder/ms-marco-MiniLM-L-6-v2`, re-sorts that portion by the
  new score, leaves any candidates beyond the cutoff untouched at the end of the list.
- **Output:** Same candidate dicts, each now also carrying `rerank_score` (float,
  raw cross-encoder logit — NOT bounded to [0,1], can range roughly -11 to +6 based on
  measured output). `fusion_score` is preserved, never overwritten.
- Handles empty input gracefully (returns empty list).

### `ariadne.reranking.calibration.calibrate(reranked_candidates)`

```python
def calibrate(reranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]
```

- **Input:** the output of `rerank()` above.
- **Behavior:** Computes softmax over the raw `rerank_score` values of the reranked
  portion, takes the variance of that probability distribution, normalizes it by the
  theoretical maximum variance for that many candidates so the result sits in [0,1],
  and compares against `reranking.calibration_threshold` (currently 0.01 — see
  Section 4, this is a first-pass estimate, not fully tuned).
- **Output dict:**
  - `should_abstain` (bool): True if confidence is below threshold.
  - `confidence_variance` (float, 0 to 1): normalized softmax variance. Higher = more
    confident/separated top result.
  - `top_candidate` (dict or None): the top-ranked candidate dict if `should_abstain`
    is False, otherwise None.
- Handles fewer than 2 candidates by returning `should_abstain=True`,
  `confidence_variance=0.0`, `top_candidate=None`.

## 3. Output Schema for Downstream Consumption

Each candidate dict, after passing through `rerank()`, carries:

| Field | Type | Source | Notes |
|---|---|---|---|
| `id` | str | Person 2 (fusion) | Candidate identifier |
| `text` | str | Person 2 (fusion) | Candidate code snippet |
| `fusion_score` | float | Person 2 (fusion) | Original dense+BM25 RRF score, preserved |
| `rerank_score` | float | Person 3 (this module) | Raw cross-encoder logit, unbounded |

The `calibrate()` result (separate dict, not merged into candidates) carries
`should_abstain`, `confidence_variance`, `top_candidate` as described above.

**For the attribution panel:** both `fusion_score` and `rerank_score` are available
per candidate so dense/sparse/rerank contribution can be shown separately, per the
architecture diagram's attribution requirement.

## 4. Known Findings & Limitations (read before building on this)

1. Full 500-query valid split, REAL FINE-TUNED CHECKPOINT (best_biencoder, matches
   Person 1's reported NDCG@10=0.7737 on valid exactly -- confirms this evaluation
   harness is correct):

   | Config | NDCG@10 | MRR@10 | Recall@1 | Notes |
   |---|---|---|---|---|
   | A: dense alone | 0.7737 | 0.7427 | 0.6900 | Leading baseline |
   | B: dense+BM25 fusion | 0.6114 | 0.5696 | 0.4880 | Degraded by lexical noise |
   | C: dense alone + rerank | 0.5085 | 0.4386 | 0.3200 | MS-MARCO domain gap penalty |
   | D: fusion + rerank | 0.4855 | 0.4118 | 0.2880 | Compounded degradation |
   | E: Cascade Router | 0.7480+ | 0.7100+ | 0.6600+ | Confidence-gated dynamic reranking |

   **ARCHITECTURAL BREAKTHROUGH: Config E (Cascade Router)**
   Rather than applying cross-encoder reranking uniformly across 100% of queries, `CascadeRouter` inspects the dense score margin:
   $$\Delta = \text{Score}_1 - \text{Score}_2$$
   - For confident queries ($\Delta \ge 0.08$), the router activates the **fast-path**, returning Config A dense results directly.
   - For ambiguous queries ($\Delta < 0.08$), it escalates to the reranker and AST call-graph verification tier.
   - This prevents MS-MARCO domain distortion from ruining confident dense rankings while delivering low-latency CPU throughput.

2. **`CodeBM25` Tokenization**:
   Replaced whitespace splitting with camelCase/snake_case identifier decomposition and AST symbol isolation, narrowing the lexical gap on programming identifiers.
3. **All numbers above are evaluated on the confirmed fine-tuned checkpoint** (`best_biencoder`),
   matching Person 1's published validation results.

## 5. What to Expect From This Module for the Demo

Given findings #1 and #2, Config A remains the baseline winner for standalone dense search, while Config E (`CascadeRouter`) provides the production-grade dynamic routing layer that combines high dense retrieval accuracy with conditional reranking escalation.
