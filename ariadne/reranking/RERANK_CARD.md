# Rerank Card: Person 3 Handoff (Reranking + Calibration)

## 1. Overview

This module reranks a fused candidate shortlist using a CPU cross-encoder, then
calibrates confidence to flag low-confidence rankings. Consumes output from Person 2's
retrieval pipeline (once available); produces output for Person 4's demo attribution
panel and ablation table.

**Status as of this handoff: functionally complete and tested. Config B (fusion)
integration still blocked on Person 2. Real fine-tuned checkpoint still blocked on
Person 1 — all numbers below were measured on the untrained base embedder
(all-MiniLM-L6-v2), not the final model.**

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

1. Full 500-query valid split, untrained base embedder, with real fusion now available:

  | Config | NDCG@10 | MRR@10 | Recall@1 |
  |---|---|---|---|
  | A: dense alone | 0.6090 | 0.5790 | 0.5220 |
  | B: dense+BM25 fusion | 0.5614 | 0.5250 | 0.4520 |
  | C: dense alone + rerank | 0.4687 | 0.4106 | 0.3120 |
  | D: fusion + rerank | 0.4503 | 0.3848 | 0.2760 |

  Config A (dense retrieval alone) is the strongest configuration on every metric measured,
  confirmed on a fair like-for-like subset as well (n=411 shared-reachable queries:
  Config A NDCG@10=0.7408 vs Config B NDCG@10=0.6829 -- not a pool-size artifact).

  Fusion (Config B) underperforms dense alone for two distinct reasons: (1) BM25 ranks
  the relevant document far worse than dense on many queries (observed BM25 ranks as
  poor as 419, dragging down the 50/50 RRF blend even when dense alone was already
  excellent), and (2) 17.8% of queries (89/500) had their relevant document completely
  absent from the fused top-50 candidate window -- a coverage gap, not a ranking-quality
  gap, worth raising with the fusion owner as a possible top_k tuning question.

  Reranking (Configs C, D) reduces every ranking metric at full scale, including
  top-10 placement (92.63% -> 86.32% on a 380-query diagnostic subset) -- an earlier
  5-query preview suggested reranking might improve top-10 placement while only hurting
  top-1 precision; that did NOT hold up at full scale and is retracted here. Reranking
  underperforms across the board on this base embedder + generic cross-encoder
  combination.

  CURRENT RECOMMENDATION: Config A (dense retrieval alone, no fusion, no reranking).
  This remains provisional pending Person 1's real fine-tuned checkpoint -- all numbers
  above use the untrained base embedder (all-MiniLM-L6-v2).
2. **Abstention is a diagnostic signal only — not yet wired to change ranking output.**
  The calibration heuristic flagged 44% of queries in the full run as low-confidence, but abstention was not connected to ranking fallback and therefore had no effect on the reported metrics.
3. **`calibration_threshold=0.01`** was derived from a 5-query manual inspection, not
   a proper sweep over the full valid split. Treat as provisional.
4. **All numbers above are on the untrained base model** (`all-MiniLM-L6-v2`), not
   Person 1's fine-tuned checkpoint (checkpoint files not yet available in the repo).
   Results may look different, possibly quite different, once that's resolved.
5. **Config B (dense+BM25 fusion) has never been tested against this reranker** —
   `retrieval/pipeline.py` was still a stub as of this handoff.

## 5. What to Expect From This Module for the Demo

Given finding #1, the demo's ablation table should currently show Config A as the
current leader among evaluated configurations; final decision pending Config B and the
fine-tuned checkpoint, with Config C included as a measured (worse) comparison point
rather than the headline result — that's a legitimate, honest ablation finding, not
  a failure to hide.

If Person 3 later re-runs the checkpoint eval with the real checkpoint and/or real
fusion and gets a different result, this card will be updated and Person 4 will be
notified before the demo is finalized.
