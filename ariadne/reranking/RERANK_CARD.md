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

   | Config | NDCG@10 | MRR@10 | Recall@1 |
   |---|---|---|---|
   | A: dense alone | 0.7737 | 0.7427 | 0.6900 |
   | B: dense+BM25 fusion | 0.6114 | 0.5696 | 0.4880 |
   | C: dense alone + rerank | 0.5085 | 0.4386 | 0.3200 |
   | D: fusion + rerank | 0.4855 | 0.4118 | 0.2880 |

   FINAL RECOMMENDATION (no longer provisional): Config A, dense retrieval alone, no
   fusion, no reranking. This gap is now WIDER than on the untrained base model
   (NDCG@10 delta A-vs-B grew from 0.05 to 0.16), confirming that a stronger dense
   retriever makes BM25's weakness on code text more costly, not less -- fusion adds
   no value at any embedder quality tested. Reranking (C, D) remains harmful at every
   embedder quality tested; top-10 placement diagnostic confirms degradation
   (96.47% -> 81.02%) consistent with the earlier full-scale base-model run.
2. **Abstention is a diagnostic signal only — not yet wired to change ranking output.**
  The calibration heuristic flagged 44% of queries in the full run as low-confidence, but abstention was not connected to ranking fallback and therefore had no effect on the reported metrics.
3. **`calibration_threshold=0.01`** was derived from a 5-query manual inspection, not
   a proper sweep over the full valid split. Treat as provisional.
4. **All numbers above are evaluated on the confirmed fine-tuned checkpoint** (`best_biencoder`),
   matching Person 1's published validation results.

## 5. What to Expect From This Module for the Demo

Given finding #1, Config A is the confirmed leading configuration across all four
evaluated approaches (dense alone, fusion, dense+rerank, fusion+rerank) on the full
500-query validation split, confirmed and verified with Person 1's final fine-tuned
checkpoint (`best_biencoder`).
